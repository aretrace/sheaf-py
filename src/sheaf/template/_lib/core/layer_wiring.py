import inspect
from collections.abc import AsyncIterator
from typing import Any, Callable

from openai.types.chat import ChatCompletionUserMessageParam

PerceptMsg = str | ChatCompletionUserMessageParam
Afference = AsyncIterator[PerceptMsg]


async def _empty_afference() -> AsyncIterator[PerceptMsg]:
    if False:
        yield


class LayerWiring:
    """Automatic wiring for handler() and percept()."""

    @staticmethod
    def _first_base_method(owner: type, name: str):
        for base in owner.__mro__[1:]:
            method = base.__dict__.get(name)
            if method and not getattr(method, "__isabstractmethod__", False):
                return method
        return None

    @staticmethod
    def _percept_accepts_afference(method: Callable[..., Any]) -> bool:
        params = list(inspect.signature(method).parameters.values())
        for p in params:
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                return True
        return len(params) >= 2

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "handler" in cls.__dict__:
            original_handler = cls.__dict__["handler"]
            base_handler = cls._first_base_method(cls, "handler")

            async def wrapped_handler(self, event):
                if base_handler:
                    await base_handler(self, event)
                await original_handler(self, event)

            setattr(
                wrapped_handler,
                "__isabstractmethod__",
                getattr(original_handler, "__isabstractmethod__", False),
            )
            cls.handler = wrapped_handler
        if "percept" in cls.__dict__:
            original_percept = cls.__dict__["percept"]
            base_percept = cls._first_base_method(cls, "percept")

            def wrapped_percept(self, afference: Afference | None = None):
                if afference is None:
                    afference = _empty_afference()
                source = (
                    original_percept(self, afference)
                    if cls._percept_accepts_afference(original_percept)
                    else original_percept(self)
                )
                if base_percept and cls._percept_accepts_afference(base_percept):
                    return base_percept(self, source)
                return source

            setattr(
                wrapped_percept,
                "__isabstractmethod__",
                getattr(original_percept, "__isabstractmethod__", False),
            )
            cls.percept = wrapped_percept
