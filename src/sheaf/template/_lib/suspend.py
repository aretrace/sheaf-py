import asyncio
import contextlib
import contextvars
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Coroutine, Optional

# Accepted types for the fallback activity shown during a suspend scope.
Fallback = (
    asyncio.Task[Any]
    | asyncio.Future[Any]
    | Coroutine[Any, Any, Any]
    | Callable[[], Coroutine[Any, Any, Any]]  # factory -> resumable
)


@dataclass
class FallbackEntry:
    """Represents a background activity managed by `suspend`.

    If `factory` is provided, the entry is *resumable*: when an inner scope ends,
    the previous entry can be restarted by calling the factory again.
    """

    task: asyncio.Task[Any]
    factory: Optional[Callable[[], Coroutine[Any, Any, Any]]] = None

    # bookkeeping for debounce / dwell
    committed: bool = False  # has this entry replaced the previous one?
    started_at: Optional[float] = None  # when this entry became visible/active


# Per-asyncio-task stack. Use None as default (avoid mutable default list pitfalls).
_FALLBACK_STACK: contextvars.ContextVar[Optional[list[FallbackEntry]]] = (
    contextvars.ContextVar("FALLBACK_STACK", default=None)
)


def _get_stack() -> list[FallbackEntry]:
    stack = _FALLBACK_STACK.get()
    if stack is None:
        stack = []
        _FALLBACK_STACK.set(stack)
    return stack


async def _await_future(fut: asyncio.Future[Any]) -> Any:
    # wrap a Future in a Task consistently.
    return await fut


async def _start_fallback(fallback: Fallback) -> FallbackEntry:
    """Normalize input and start a background task. Always returns a Task."""
    if isinstance(fallback, asyncio.Task):
        return FallbackEntry(task=fallback, factory=None)
    if isinstance(fallback, asyncio.Future):
        task = asyncio.create_task(_await_future(fallback))
        return FallbackEntry(task=task, factory=None)
    if asyncio.iscoroutine(fallback):
        task = asyncio.create_task(fallback)
        return FallbackEntry(task=task, factory=None)
    if callable(fallback):
        task = asyncio.create_task(fallback())
        return FallbackEntry(task=task, factory=fallback)
    raise TypeError(
        "suspend(...) expects Task | Future | Coroutine | Callable[[] -> Coroutine]"
    )


async def _cancel_task(task: asyncio.Task[Any]) -> None:
    """Cancel a task and await its termination so its `finally:` blocks run."""
    if task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@contextlib.asynccontextmanager
async def suspend(
    fallback: Fallback,
    *,
    dwell: float = 0.0,  # don’t hide yet
    debounce: float = 0.0,  # don’t show yet
) -> AsyncIterator[None]:
    """Run a background activity while inside this scope.

    Cancels any currently active fallback and starts `fallback`.
    On exit, cancels fallback and, if the previous was created from a factory, resumes it **in place** (so outer scopes cancel the right task later).

    Args:
        dwell (float): Once the fallback is active, keep it running for at least n seconds before stopping it, even if the scope ends sooner.
        debounce (float): Delay switching to the fallback by n seconds. If the scope exits early, the fallback may never show.

    - If `debounce > 0`, *delay* replacing the current background until this scope
      has been alive for `debounce` seconds. If the scope exits before that, we
      never switch, avoiding flicker.
    - Otherwise (or after the debounce delay), cancel the current fallback and
      start `fallback` immediately.
    - Guarantees `dwell` seconds of execution *after the switch actually
      happens* (i.e., from the moment the new background becomes visible).
    - On exit, cancels the current fallback and, if the previous was created from
      a factory, resumes it **in place** (so outer scopes cancel the right task later).

    Tip: Pass a zero-arg **factory** (lambda) if you want the outer activity to resume.
    Passing a bare coroutine makes the scope non-resumable by design.

    Note: If you pass a pre-started `asyncio.Task` and also use `debounce`, the task itself may already be running;
    `debounce` will delay *switching to it*, not its own start.
    Prefer a coroutine or factory if you want full control.
    """
    stack = _get_stack()
    loop = asyncio.get_running_loop()

    # Fast path: no debounce — original semantics.
    if debounce <= 0.0:
        # Pause (cancel) currently running background if present.
        if stack:
            await _cancel_task(stack[-1].task)

        # Start our background and push it.
        entry = await _start_fallback(fallback)
        entry.committed = True
        entry.started_at = loop.time()
        stack.append(entry)

        try:
            # Work happens in the `with` body while the task runs in the background.
            yield
        finally:
            # Ensure the background task remains for at least `dwell` seconds
            # since it actually started.
            if entry.committed and dwell > 0:
                elapsed = loop.time() - (entry.started_at or loop.time())
                remaining = dwell - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)

            # Stop our background.
            await _cancel_task(entry.task)

            # LIFO integrity: the top of the stack must be our own entry.
            assert stack and stack[-1] is entry, (
                "suspend contexts must exit in LIFO order"
            )
            stack.pop()

            # Resume previous background **in-place** if it was created from a factory and is not already running.
            if stack:
                prev = stack[-1]
                if prev.factory is not None and prev.task.done():
                    prev.task = asyncio.create_task(prev.factory())
        return

    # Debounced path: don't cancel/replace immediately — only commit after delay.
    # We push a placeholder entry whose `task` is a tiny guard; if the scope is
    # still active after `debounce`, the guard commits the switch in-place.
    entry: FallbackEntry  # forward reference for the guard closure

    async def _guard() -> None:
        try:
            # Wait out the debounce window.
            await asyncio.sleep(debounce)

            # Only commit if this scope is still the active/top entry.
            if not (stack and stack[-1] is entry):
                return

            # Cancel the previous (now beneath us) and start ours.
            prev = stack[-2] if len(stack) >= 2 else None
            if prev is not None:
                await _cancel_task(prev.task)

            started = await _start_fallback(fallback)
            # In-place swap: this entry now represents the real background task.
            entry.task = started.task
            entry.factory = started.factory
            entry.committed = True
            entry.started_at = loop.time()
        except asyncio.CancelledError:
            # Debounce was canceled (scope ended or superseded) — do nothing.
            pass

    guard_task = asyncio.create_task(_guard())
    entry = FallbackEntry(
        task=guard_task,
        factory=None,  # not resumable until we've actually committed
        committed=False,
        started_at=None,
    )
    stack.append(entry)

    try:
        # Work happens in the `with` body while either the previous background
        # keeps running (pre-commit) or ours runs (post-commit).
        yield
    finally:
        # Honor `minimum_dwell` only if we actually switched to this fallback.
        if entry.committed and dwell > 0:
            elapsed = loop.time() - (entry.started_at or loop.time())
            remaining = dwell - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)

        # Stop whatever our entry currently represents (guard or real task).
        await _cancel_task(entry.task)

        # LIFO integrity.
        assert stack and stack[-1] is entry, "suspend contexts must exit in LIFO order"
        stack.pop()

        # Resume previous **only if** we had canceled it earlier (i.e., it's done now).
        if stack:
            prev = stack[-1]
            if prev.factory is not None and prev.task.done():
                prev.task = asyncio.create_task(prev.factory())


# _T = TypeVar("_T")
# @contextlib.asynccontextmanager
# async def suspend(
#     fallback: asyncio.Task[_T] | asyncio.Future[_T] | Coroutine[Any, Any, _T],
# ):
#     """
#     Accepts either:
#         - a Task (already scheduled), or
#         - a coroutine (which will be wrapped in asyncio.create_task)
#     Cancels and awaits it when the context exits.
#     """
#     if asyncio.iscoroutine(fallback):
#         fallback = asyncio.create_task(fallback)
#     try:
#         yield
#     finally:
#         fallback.cancel()
#         with contextlib.suppress(asyncio.CancelledError):
#             await fallback
