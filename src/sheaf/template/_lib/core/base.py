import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import (
    Any,
    AsyncContextManager,
    Callable,
    Literal,
    TypeVar,
    cast,
    final,
)

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from ..misc import pseudlid
from ..suspend import Fallback
from ..suspend import suspend as _suspend_util
from ..toolgen import callables_to_tool_schemas
from .layer_wiring import Afference, LayerWiring, PerceptMsg, _empty_afference
from .tool_executor import CalledTool, ToolExecution


class StepEvent(BaseModel):
    id: str = Field(default_factory=pseudlid)
    type: Literal[
        "agent.execution.starting",
        "agent.execution.started",
        "agent.execution.stopped",
        "agent.tools.executing",
        "agent.tools.completed",
        "agent.finish.length",
        "agent.finish.filtered",
        "agent.finish.stopped",
        "agent.messages.updated",
    ]
    timestamp: float = Field(default_factory=time.time)


class StepError(BaseModel):
    message: str


class BaseLayer(LayerWiring, ToolExecution, ABC):
    """
    Situated perception–action cycle subsumption.

    Config attributes (read-write):
    - self.model
    - self.api_key
    - self.base_url

    State attributes (read-write):
    - self.messages

    State attributes (read-only):
    - self.called_tools
    - self.assistant_reply
    - self.error

    Sense/Act methods:
    - self.percept
    - self.handler

    Task methods:
    - self.suspend
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        init_messages: list[ChatCompletionMessageParam],
        callable_tools: list[Callable[..., Any]],
    ):
        # Session
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.tools = callables_to_tool_schemas(callable_tools)

        # Turn + Step
        self.messages: list[ChatCompletionMessageParam] = init_messages

        # Step
        self.called_tools: list[CalledTool] = []
        self._init_tools(callable_tools)
        self.assistant_reply: str | None = None
        # TODO: fix implicit usage,
        # - API errors → break (stop agent)
        # - Tool errors → continue (resilient)
        # Value is set, but not included in events,
        # handler() can't see what / why something failed.
        self.error: StepError | None = None

        # Event
        self._event_transition_boundary: AsyncContextManager[Any] | None = None

    @abstractmethod
    async def percept(self, afference: Afference) -> AsyncIterator[PerceptMsg]:
        """
        Async generator that yields input messages.

        Leaf subclasses ignore afference and yield source messages.
        Middleware subclasses consume afference and yield transformed messages.
        """
        async for msg in afference:
            yield msg

    @abstractmethod
    async def handler(self, event: StepEvent) -> None:
        """
        Handle events emitted by the agent loop.

        Args:
            event: Event object with type attribute
        """
        pass

    # Spawn concurrent execution spool
    @final
    async def suspend(
        self, fallback: Fallback, *, dwell: float = 0.0, debounce: float = 0.0
    ):
        """Start a background activity for the current event scope."""
        if self._event_transition_boundary:
            await self._event_transition_boundary.__aexit__(None, None, None)
        self._event_transition_boundary = _suspend_util(
            fallback, dwell=dwell, debounce=debounce
        )
        await self._event_transition_boundary.__aenter__()

    @final
    async def _emit(self, event: StepEvent) -> None:
        """Emit an event with automatic background cleanup."""
        if self._event_transition_boundary:
            await self._event_transition_boundary.__aexit__(None, None, None)
            self._event_transition_boundary = None

        await self.handler(event)

    # Fully autonomous, pure orchestration.
    @final
    async def _dispatch(self) -> None:
        """Core autonomous agentic loop logic."""
        max_turns = float("inf")  # 🦾 ⚜️ 🌈 🪐 🛠
        turns = 0
        # Control loop.
        while True:
            turns += 1
            if turns > max_turns:
                self.error = StepError(message=f"Aborting after {max_turns} cycles.")
                break

            self.called_tools = []
            self.assistant_reply = None
            self.error = None

            await self._emit(StepEvent(type="agent.execution.starting"))  # 📡
            try:
                # Dithering...
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tools,
                    parallel_tool_calls=True,
                    tool_choice="auto",
                )
            except Exception as e:
                self.error = StepError(message=str(e))  # ❗
                break

            completion = response.choices[0]

            # TODO: is this including server‑only fields (audio, refusal, etc.)? Chat Completions request schema does not allow these kinds of messages to be send back, should we extract and re‑insert only the assistant fields?
            self.messages.append(
                cast(
                    ChatCompletionMessageParam,
                    completion.message.model_dump(exclude_none=True),
                )
            )
            await self._emit(StepEvent(type="agent.execution.started"))  # 📡

            tool_calls = completion.message.tool_calls
            if tool_calls:
                await self._emit(StepEvent(type="agent.tools.executing"))  # 📡
                try:
                    self.called_tools = await asyncio.to_thread(
                        self.call_tools, tool_calls
                    )
                    tool_results = [
                        called_tool["results"] for called_tool in self.called_tools
                    ]
                    self.messages.extend(tool_results)
                except Exception as e:
                    self.error = StepError(message=str(e))  # ❗
                await self._emit(StepEvent(type="agent.tools.completed"))  # 📡
                # TODO: should we stop the loop here?
                # if self.error:
                #     break
                continue

            if completion.finish_reason == "length":
                self.messages.append(
                    {"role": "user", "content": "Please continue with your response."}
                )
                await self._emit(StepEvent(type="agent.finish.length"))  # 📡
                continue

            if completion.finish_reason == "content_filter":
                await self._emit(StepEvent(type="agent.finish.filtered"))  # 📡
                continue

            if completion.finish_reason == "stop":
                assistant_message = completion.message
                if refusal := assistant_message.refusal:
                    self.assistant_reply = refusal
                elif content := assistant_message.content:
                    self.assistant_reply = content
                await self._emit(StepEvent(type="agent.finish.stopped"))  # 📡
                break

            self.error = StepError(
                message=f"Unexpected finish reason: '{completion.finish_reason}'"
            )  # ❗
            break

        await self._emit(StepEvent(type="agent.execution.stopped"))  # 📡

    @final
    async def _actuate(self):
        """Update messages and run the agentic loop."""
        await self._emit(StepEvent(type="agent.messages.updated"))  # 📡
        await self._dispatch()

    @final
    async def turn(
        self,
        message: str
        | ChatCompletionMessageParam
        | list[str | ChatCompletionMessageParam]
        | None = None,
    ) -> None:
        """
        Process message(s) and run the agentic loop.
        """
        if message is None:
            async for msg in self.percept(_empty_afference()):
                await self.turn(msg)
            return

        if isinstance(message, str):
            self.messages.append({"role": "user", "content": message})
            await self._actuate()
            return

        if isinstance(message, list):
            self.messages.extend(
                [
                    {"role": "user", "content": msg} if isinstance(msg, str) else msg
                    for msg in message
                ]
            )
            await self._actuate()
            return

        # if isinstance(message, ChatCompletionMessageParam):
        self.messages.append(message)
        await self._actuate()


T = TypeVar("T", bound=BaseLayer)


def situate(cls: type[T]) -> type[T]:
    """Marks a layer to be situated (metadata annotation)."""
    setattr(cls, "__situate__", True)
    return cls
