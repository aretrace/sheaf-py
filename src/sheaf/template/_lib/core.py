import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import (
    Any,
    AsyncContextManager,
    Callable,
    Literal,
    TypedDict,
    TypeVar,
    cast,
    final,
)

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, Field

from .misc import pseudlid
from .suspend import Fallback
from .suspend import suspend as _suspend_util
from .toolgen import callables_to_tool_schemas


class CalledTool(TypedDict):
    """Metadata and results from a tool execution."""

    name: str
    arguments: str
    results: ChatCompletionToolMessageParam


class ToolExecutor:
    """Handles tool execution with parallel processing."""

    def __init__(self, tool_registry: dict[str, Callable[..., Any]]):
        self._tool_registry = tool_registry

    @classmethod
    def from_callables(cls, callables: list[Callable[..., Any]]) -> "ToolExecutor":
        """Convenience constructor from a list of functions."""
        registry = {fn.__name__: fn for fn in callables}
        return cls(registry)

    def call_tools(
        self, tool_calls: list[ChatCompletionMessageToolCallUnion]
    ) -> list[CalledTool]:
        """Execute multiple tool calls in parallel and maintain original order.

        Args:
            tool_calls: The list of tool calls from the assistant message

        Returns:
            List of CalledTool dicts with name, arguments, and results
        """
        with ThreadPoolExecutor() as pool:
            # Submit all tool calls for parallel execution
            tool_call_futures_dict: dict[
                Future[ChatCompletionToolMessageParam],
                tuple[int, str, str],
            ] = {
                pool.submit(self._execute_tool, tool): (
                    tool_index,
                    tool_name := getattr(
                        getattr(tool, "function", None), "name", "unknown"
                    ),
                    tool_arguments := getattr(
                        getattr(tool, "function", None), "arguments", "{}"
                    )
                    or "{}",
                )
                for tool_index, tool in enumerate(tool_calls)
            }

            # Prepare results list to maintain original order
            original_tool_call_order_results = cast(
                list[CalledTool], [None] * len(tool_calls)
            )

            # Collect results as they complete
            for completed_tool_call_future in as_completed(tool_call_futures_dict):
                original_tool_order_index, tool_name, tool_arguments = (
                    tool_call_futures_dict[completed_tool_call_future]
                )
                tool_message_response = completed_tool_call_future.result()
                original_tool_call_order_results[original_tool_order_index] = {
                    "name": tool_name,
                    "arguments": tool_arguments,
                    "results": tool_message_response,
                }

            return original_tool_call_order_results

    def _execute_tool(
        self,
        tool: ChatCompletionMessageToolCallUnion,
    ) -> ChatCompletionToolMessageParam:
        """Execute a single tool call and return a tool message.

        Args:
            tool: The tool call from the assistant

        Returns:
            ChatCompletionToolMessageParam with the tool result
        """
        tool_id = tool.id
        # Validate tool type
        if getattr(tool, "type", None) != "function":
            print(f"Unsupported tool call: {tool_id}")
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": f"Error: {tool_id} is an unsupported tool call type, expected `function`.",
            }

        tool = cast(ChatCompletionMessageFunctionToolCall, tool)
        tool_name = tool.function.name
        tool_arguments = tool.function.arguments or "{}"

        # Parse arguments
        try:
            tool_arguments_dict = json.loads(tool_arguments) if tool_arguments else {}
        except Exception as e:
            print(f"Failed to parse tool arguments for {tool_name}: {e}")
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": "Error: Failed to parse tool arguments.",
            }

        # Look up tool function
        tool_function = self._tool_registry.get(tool_name)
        if tool_function is None:
            print(f"Error: tool {tool_name} not available.")
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": f"Error: tool `{tool_name}` not available.",
            }

        # Execute tool
        try:
            tool_result = tool_function(**tool_arguments_dict)
            tool_result_str = (
                tool_result if isinstance(tool_result, str) else json.dumps(tool_result)
            )
        except Exception as e:
            print(f"Tool {tool_name} raised: {e}")
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": str(e),
            }

        return {
            "role": "tool",
            "tool_call_id": tool_id,
            "content": tool_result_str,
        }


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
    error: str | None = None


class StepError(BaseModel):
    message: str


class BaseLayer(ABC):
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
        self._tool_executor = ToolExecutor.from_callables(callable_tools)
        self.assistant_reply: str | None = None
        # TODO: fix implicit usage,
        # - API errors → break (stop agent)
        # - Tool errors → continue (resilient)
        # Value is set, but not included in events,
        # handler() can't see what / why somthing failed.
        self.error: StepError | None = None

        # Event
        self._event_transition_boundary: AsyncContextManager[Any] | None = None

    @abstractmethod
    async def percept(self) -> AsyncIterator[str | ChatCompletionUserMessageParam]:
        """
        Async generator that yields input messages.

        ONLY embodied leaf subclasses should implement this method.
        """
        if False:
            yield

    @abstractmethod
    async def handler(self, event: StepEvent) -> None:
        """
        Handle events emitted by the agent loop.

        Args:
            event: Event object with type attribute
        """
        pass

    # Automatic middleware chaining for `handler()`.
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if "handler" in cls.__dict__:
            original = cls.__dict__["handler"]

            async def wrapped(self, event):
                for base in cls.__mro__[1:]:
                    method = base.__dict__.get("handler")
                    if method and not getattr(method, "__isabstractmethod__", False):
                        await method(self, event)
                        break
                await original(self, event)

            cls.handler = wrapped

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
        max_turns = float("inf")  # 🦾⚜️🌈🪐🛠
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
                        self._tool_executor.call_tools, tool_calls
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
            async for msg in self.percept():
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


def embody(cls: type[T]) -> type[T]:
    """Marks a layer to be embodied (metadata annotation)."""
    setattr(cls, "__embody__", True)
    return cls
