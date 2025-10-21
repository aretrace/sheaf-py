# shel, a Bun.js-like shell for Python (using 3.14 t-strings), keeping interpolations safe.
# For Linux/macOS; Windows intentionally not supported.

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import subprocess
import sys
import threading
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from dataclasses import dataclass
from string.templatelib import Interpolation, Template, convert
from typing import IO, Any, cast

# ============================= Result & Errors ===============================


@dataclass
class ShelResult:
    script: str
    args: list[str]
    exit_code: int
    stdout: bytes
    stderr: bytes

    def text(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        return self.stdout.decode(encoding, errors)

    def json(self, encoding: str = "utf-8"):
        return json.loads(self.text(encoding=encoding))

    def lines_text(
        self, encoding: str = "utf-8", keepends: bool = False
    ) -> Iterator[str]:
        for line in self.text(encoding=encoding).splitlines(keepends=keepends):
            yield line

    def bytes(self) -> bytes:
        return self.stdout


class ShelError(RuntimeError):
    def __init__(self, message: str, result: ShelResult):
        super().__init__(message)
        self.exitCode = result.exit_code
        self.stdout = result.stdout
        self.stderr = result.stderr
        self.result = result


# ================================ Public API =================================


class Shel:
    """
    Callable shell context, analogous to Bun's $, but valid Python name: shel(...).
    """

    def __init__(self):
        self._default_environment: Mapping[str, str] | None = os.environ.copy()
        self._default_working_directory: str | None = None
        self._throws_default: bool = True

    def env(self, mapping: Mapping[str, str] | None = None):
        self._default_environment = mapping
        return self

    def cwd(self, path: str | os.PathLike[str] | None):
        self._default_working_directory = None if path is None else os.fspath(path)
        return self

    def throws(self, flag: bool):
        self._throws_default = bool(flag)
        return self

    def nothrow(self):
        return self.throws(False)

    def __call__(self, template: Template) -> "ShelCommand":
        if not isinstance(template, Template):
            raise TypeError('Use _sh(t"...") with a t-string Template')
        return ShelCommand(self, template)


shel = Shel()


class ShelCommand:
    def __init__(self, context: Shel, template: Template):
        self._context = context
        self._template = template
        self._local_environment: Mapping[str, str] | None = None
        self._local_working_directory: str | None = None
        self._quiet = False
        self._throw = context._throws_default
        self._result: ShelResult | None = None
        self._started = False

    # -------- configuration (fluent) --------

    def env(self, mapping: Mapping[str, str] | None = None):
        self._local_environment = mapping
        return self

    def cwd(self, path: str | os.PathLike[str]):
        self._local_working_directory = os.fspath(path)
        return self

    def quiet(self, flag: bool = True):
        self._quiet = bool(flag)
        return self

    def throws(self, flag: bool):
        self._throw = bool(flag)
        return self

    def nothrow(self):
        return self.throws(False)

    # ----------------- runners -----------------

    def run(self) -> ShelResult:
        if self._started and self._result is not None:
            return self._result

        # 1) Render the shell script and positional arguments from the Template.
        script, argv, hooks = _render_posix_script(self._template)

        # 2) Decide environment and working directory (local overrides default).
        environment = _resolve_environment(
            self._context._default_environment, self._local_environment
        )
        working_directory = (
            self._local_working_directory
            if self._local_working_directory is not None
            else self._context._default_working_directory
        )

        # 3) Tee to console only if not quiet and no Python sinks are active.
        tee_to_console = (not self._quiet) and not (
            hooks.stdout_sinks or hooks.stderr_sinks or hooks.both_sinks
        )

        # 4) Execute the script (streams to sinks live).
        result = _execute_shell_script(
            script=script,
            argv=argv,
            environment=environment,
            working_directory=working_directory,
            tee_to_console=tee_to_console,
            hooks=hooks,
        )

        # 5) Finalize.
        self._result = result
        self._started = True
        if self._throw and result.exit_code != 0:
            raise ShelError("Non-zero exit code", result)
        return result

    def text(self, *args, **kwargs) -> str:
        self.quiet()
        return self.run().text(*args, **kwargs)

    def json(self, *args, **kwargs):
        self.quiet()
        return self.run().json(*args, **kwargs)

    def bytes(self) -> bytes:
        self.quiet()
        return self.run().bytes()

    # --------- streaming lines (async) – no console tee ---------

    async def lines(
        self, encoding: str = "utf-8", errors: str = "strict"
    ) -> AsyncIterator[str]:
        if self._started:
            raise RuntimeError(
                ".lines() must be called before other readers on this command"
            )

        # 1) Prepare script and process.
        script, argv, hooks = _render_posix_script(self._template)
        environment = _resolve_environment(
            self._context._default_environment, self._local_environment
        )
        working_directory = (
            self._local_working_directory
            if self._local_working_directory is not None
            else self._context._default_working_directory
        )

        (
            process,
            _captured_stdout,
            captured_stderr,
            stdout_done_event,
            stderr_done_event,
            sink_manager,
        ) = _spawn_shell_process(
            script=script,
            argv=argv,
            environment=environment,
            working_directory=working_directory,
            hooks=hooks,
            stream_stdout=True,  # we will read stdout ourselves line-by-line
            tee_to_console=False,  # streaming API never tees to console
        )

        # 2) Provide stdin if present (streaming if file-like).
        _start_stdin_pump_if_needed(process, hooks)

        # 3) Start a background reader that feeds an asyncio queue one line at a time.
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        stdout_stream = cast(IO[bytes], process.stdout)

        def _read_stdout_lines():
            try:
                for chunk in iter(stdout_stream.readline, b""):
                    # Write to sinks LIVE as lines arrive.
                    if sink_manager is not None:
                        sink_manager.write_stdout(chunk)
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            finally:
                stdout_done_event.set()
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_read_stdout_lines, daemon=True).start()

        # 4) Accumulate stdout for result while yielding lines to the caller.
        stdout_accumulator = io.BytesIO()
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            stdout_accumulator.write(chunk)
            yield chunk.decode(encoding, errors).rstrip("\n")

        # 5) Finish process and collect results.
        exit_code = process.wait()
        stdout_done_event.wait()
        stderr_done_event.wait()
        stderr_all = captured_stderr.getvalue()

        # Flush sinks (no post-hoc copies; writing was live).
        if sink_manager is not None:
            sink_manager.flush_all()

        self._result = ShelResult(
            script, argv, exit_code, stdout_accumulator.getvalue(), stderr_all
        )
        self._started = True
        if self._throw and exit_code != 0:
            raise ShelError("Non-zero exit code", self._result)

    # --------- streaming lines (sync) – no console tee ---------

    def iter_lines(
        self, encoding: str = "utf-8", errors: str = "strict"
    ) -> Iterator[str]:
        if self._started:
            raise RuntimeError(
                ".iter_lines() must be called before other readers on this command"
            )

        # 1) Prepare script and process.
        script, argv, hooks = _render_posix_script(self._template)
        environment = _resolve_environment(
            self._context._default_environment, self._local_environment
        )
        working_directory = (
            self._local_working_directory
            if self._local_working_directory is not None
            else self._context._default_working_directory
        )

        (
            process,
            _captured_stdout,
            captured_stderr,
            stdout_done_event,
            stderr_done_event,
            sink_manager,
        ) = _spawn_shell_process(
            script=script,
            argv=argv,
            environment=environment,
            working_directory=working_directory,
            hooks=hooks,
            stream_stdout=True,  # we will read stdout ourselves line-by-line
            tee_to_console=False,  # streaming API never tees to console
        )

        # 2) Provide stdin if present (streaming if file-like).
        _start_stdin_pump_if_needed(process, hooks)

        # 3) Read stdout line-by-line, writing to sinks LIVE and accumulating for result.
        stdout_accumulator = io.BytesIO()
        stdout_stream = cast(IO[bytes], process.stdout)

        for raw_line in iter(stdout_stream.readline, b""):
            if sink_manager is not None:
                sink_manager.write_stdout(raw_line)
            stdout_accumulator.write(raw_line)
            yield raw_line.decode(encoding, errors).rstrip("\n")

        # 4) Wrap up.
        stdout_done_event.set()
        exit_code = process.wait()
        stderr_done_event.wait()
        stderr_all = captured_stderr.getvalue()

        if sink_manager is not None:
            sink_manager.flush_all()

        self._result = ShelResult(
            script, argv, exit_code, stdout_accumulator.getvalue(), stderr_all
        )
        self._started = True
        if self._throw and exit_code != 0:
            raise ShelError("Non-zero exit code", self._result)


# ====================== Rendering to a POSIX shell script =====================

# Tightened: allow optional FD prefix, handle both-streams variants, consume trailing spaces.
_REDIRECTION_OP_RE = re.compile(
    r"""
    (?:
        ^|[;\s]                # start, semicolon, or whitespace
    )
    (?P<op>
        <                      # stdin redirection from object
        |&>>|&>                # both streams
        |(?:\d+)?>>            # optional fd + >>
        |(?:\d+)?>             # optional fd + >
    )
    \s*$                       # any trailing whitespace to end
    """,
    re.VERBOSE,
)


class _Hooks:
    """
    Captures in-process redirections and sources.

    stdout/stderr/both sinks are lists so multiple redirections are supported,
    e.g. `> {a} > {b}` or `&> {x} &> {y}` (both get the same bytes).
    """

    def __init__(self):
        self.stdin_source: Any | None = None
        self.stdout_sinks: list[tuple[Any, bool]] = []
        self.stderr_sinks: list[tuple[Any, bool]] = []
        self.both_sinks: list[tuple[Any, bool]] = []


def _flatten(template: Template) -> Iterable[str | Interpolation]:
    """Flatten a Template into a sequence of strings and Interpolations."""
    for item in template:
        if isinstance(item, Interpolation) and isinstance(item.value, Template):
            yield from _flatten(item.value)
        else:
            yield item


def _interpolation_to_text(interp: Interpolation) -> str:
    """Convert an Interpolation to a string, applying conversion and format."""
    value = interp.value
    if interp.conversion is not None:
        value = convert(value, interp.conversion)
    if interp.format_spec:
        value = format(value, interp.format_spec)
    return str(value)


def _is_bytes_like_source(obj: Any) -> bool:
    """True if obj can supply bytes for stdin redirection."""
    return (
        isinstance(obj, (bytes, bytearray, memoryview))
        or isinstance(obj, io.BytesIO)
        or (hasattr(obj, "read") and callable(obj.read))
        or hasattr(obj, "content")
    )


def _is_write_sink_object(obj: Any) -> bool:
    """True if obj accepts bytes via .write() or is a mutable bytes buffer."""
    return (
        isinstance(obj, (bytearray, memoryview))
        or isinstance(obj, io.BytesIO)
        or (hasattr(obj, "write") and callable(obj.write))
    )


def _convert_to_bytes(data: Any) -> bytes:
    """Best-effort conversion of data to bytes."""
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode()
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    return bytes(data)


def _get_trailing_redirection_operator(text: str) -> str | None:
    match = _REDIRECTION_OP_RE.search(text)
    return match.group("op") if match else None


def _should_route_to_object(operator: str, raw_value: Any) -> bool:
    """Whether the redirection operator should be handled in-process."""
    if operator == "<":
        return _is_bytes_like_source(raw_value)
    if operator in (">", ">>", "1>", "1>>"):
        return _is_write_sink_object(raw_value)
    if operator in ("2>", "2>>"):
        return _is_write_sink_object(raw_value)
    if operator in ("&>", "&>>"):
        return _is_write_sink_object(raw_value)
    # Also support arbitrary fd prefixes like "3>" and "3>>" (treated as stdout)
    if operator.endswith(">") or operator.endswith(">>"):
        # If it looked like "N>" or "N>>" and sink is writeable, accept as stdout-class.
        return _is_write_sink_object(raw_value)
    return False


def _apply_inprocess_redirection(operator: str, raw_value: Any, hooks: _Hooks) -> None:
    """Record in-process redirections into hooks (stdin/stdout/stderr/both)."""
    if operator == "<":
        # keep the *source* so we can stream it into stdin
        hooks.stdin_source = raw_value
        return

    # IMPORTANT: handle stderr explicitly BEFORE generic numeric FDs
    if operator in ("2>", "2>>"):
        hooks.stderr_sinks.append((raw_value, operator.endswith(">>")))
        return

    # explicit stdout variants
    if operator in (">", "1>", ">>", "1>>"):
        hooks.stdout_sinks.append((raw_value, operator.endswith(">>")))
        return

    # other numeric FDs (3>, 5>>, etc.) -> treat as stdout-class
    if re.fullmatch(r"\d+>>?", operator):
        hooks.stdout_sinks.append((raw_value, operator.endswith(">>")))
        return

    if operator in ("&>", "&>>"):
        hooks.both_sinks.append((raw_value, operator.endswith(">>")))
        return


def _render_posix_script(template: Template) -> tuple[str, list[str], _Hooks]:
    """
    Convert a Template into a POSIX shell script and argv list.
    Interpolations are passed as positional arguments to keep them safe.
    """
    parts = list(_flatten(template))
    script_chunks: list[str] = []
    args: list[str] = ["pytosh"]  # becomes $0
    current_arg_index = 1
    hooks = _Hooks()
    pending_redirection_operator: str | None = None

    for piece in parts:
        # Literal text: append and remember any trailing redirection operator.
        if isinstance(piece, str):
            script_chunks.append(piece)
            pending_redirection_operator = _get_trailing_redirection_operator(piece)
            continue

        # Interpolation: decide whether it is a sink/source for a preceding redirection.
        interpolation = cast(Interpolation, piece)
        raw_value = interpolation.value
        value_text = _interpolation_to_text(interpolation)

        if pending_redirection_operator:
            operator = pending_redirection_operator
            pending_redirection_operator = None

            if _should_route_to_object(operator, raw_value):
                # Remove operator from script and capture sink/source in hooks.
                if script_chunks:
                    # Strip the operator + trailing space from the last chunk.
                    script_chunks[-1] = _REDIRECTION_OP_RE.sub("", script_chunks[-1])
                _apply_inprocess_redirection(operator, raw_value, hooks)
                continue  # Do not expose this value as "$N"

        # Default case: pass interpolation as a positional argument.
        args.append(value_text)
        slot = f'"${current_arg_index}"'
        current_arg_index += 1
        script_chunks.append(slot)

    script = "".join(script_chunks)
    return script, args, hooks


# ============================== Sink plumbing =================================


class _SinkWriter:
    """
    A small adapter that writes bytes to supported targets and normalizes
    overwrite/append semantics. Used by pump threads to stream output LIVE.
    """

    def __init__(self, target: Any, append: bool):
        self._t = target
        self._append = append
        self._lock = threading.Lock()
        self._pos = 0  # for memoryview incremental writes
        self._started = False

    def start(self):
        if self._started:
            return
        self._started = True

        t = self._t
        # Overwrite semantics: clear on start.
        if not self._append:
            if isinstance(t, bytearray):
                t[:] = b""
            elif isinstance(t, io.BytesIO):
                t.seek(0)
                t.truncate(0)
            elif hasattr(t, "seek") and hasattr(t, "truncate") and hasattr(t, "write"):
                try:
                    t.seek(0)
                    t.truncate(0)
                except Exception:
                    pass
        # memoryview: we always start at position 0 (append is not meaningful).
        self._pos = 0

    def write(self, data: bytes):
        if not data:
            return
        with self._lock:
            t = self._t
            if isinstance(t, bytearray):
                t.extend(data)
                return
            if isinstance(t, memoryview):
                n = min(len(t) - self._pos, len(data))
                if n > 0:
                    t[self._pos : self._pos + n] = data[:n]
                    self._pos += n
                return
            if isinstance(t, io.BytesIO):
                t.write(data)
                return
            if hasattr(t, "write"):
                try:
                    t.write(data)
                except TypeError:
                    # Best-effort fallback for text-mode writers
                    t.write(data.decode("utf-8", "replace"))
                return
            # Otherwise: silently ignore (mirrors prior TypeError during post-hoc write)

    def flush(self):
        t = self._t
        if hasattr(t, "flush"):
            try:
                t.flush()
            except Exception:
                pass


class _SinkManager:
    """Collects all sink writers and provides per-stream write methods."""

    def __init__(self, hooks: _Hooks):
        self._stdout = [_SinkWriter(t, a) for (t, a) in hooks.stdout_sinks]
        self._stderr = [_SinkWriter(t, a) for (t, a) in hooks.stderr_sinks]
        self._both = [_SinkWriter(t, a) for (t, a) in hooks.both_sinks]

    def start_all(self):
        for w in (*self._stdout, *self._stderr, *self._both):
            w.start()

    def write_stdout(self, data: bytes):
        for w in self._stdout:
            w.write(data)
        for w in self._both:
            w.write(data)

    def write_stderr(self, data: bytes):
        for w in self._stderr:
            w.write(data)
        for w in self._both:
            w.write(data)

    def flush_all(self):
        for w in (*self._stdout, *self._stderr, *self._both):
            w.flush()

    def has_any(self) -> bool:
        return bool(self._stdout or self._stderr or self._both)


# ============================== Execution layer ==============================


def _start_stream_pump_thread(
    stream: IO[bytes],
    capture_buffer: io.BytesIO,
    tee_to_console: bool,
    is_stdout: bool,
    sink_manager: _SinkManager | None,
) -> threading.Event:
    """
    Start a background thread to drain a byte stream into a buffer (and optionally tee).
    Returns a threading.Event that is set when pumping is finished.
    """
    done_event = threading.Event()

    def _pump():
        try:
            for chunk in iter(stream.read, b""):
                capture_buffer.write(chunk)
                if sink_manager is not None:
                    if is_stdout:
                        sink_manager.write_stdout(chunk)
                    else:
                        sink_manager.write_stderr(chunk)
                if tee_to_console:
                    target = sys.stdout.buffer if is_stdout else sys.stderr.buffer
                    target.write(chunk)
            if tee_to_console:
                try:
                    (sys.stdout if is_stdout else sys.stderr).flush()
                except Exception:
                    pass
        finally:
            done_event.set()

    threading.Thread(target=_pump, daemon=True).start()
    return done_event


def _start_stdin_pump_if_needed(process: subprocess.Popen, hooks: _Hooks) -> None:
    """
    If a stdin source was captured (< {obj}), stream it into the child process.
    - bytes/bytearray/memoryview/content: one-shot write
    - file-like / .read(): streamed in chunks
    """
    source = hooks.stdin_source
    if source is None or not process.stdin:
        return

    stdin_stream = cast(IO[bytes], process.stdin)

    def _pump_stdin():
        try:
            # file-like?
            if hasattr(source, "read") and callable(source.read):
                while True:
                    chunk = source.read(64 * 1024)
                    if not chunk:
                        break
                    stdin_stream.write(_convert_to_bytes(chunk))
            # Response-like object with .content
            elif hasattr(source, "content"):
                stdin_stream.write(_convert_to_bytes(source.content))
            # bytes-like (bytes/bytearray/memoryview/BytesIO)
            elif isinstance(source, (bytes, bytearray, memoryview)):
                stdin_stream.write(_convert_to_bytes(source))
            elif isinstance(source, io.BytesIO):
                stdin_stream.write(source.getvalue())
            else:
                # Fallback: best-effort bytes()
                stdin_stream.write(_convert_to_bytes(source))
        finally:
            try:
                stdin_stream.close()
            except Exception:
                pass

    threading.Thread(target=_pump_stdin, daemon=True).start()


def _spawn_shell_process(
    script: str,
    argv: list[str],
    environment: Mapping[str, str] | None,
    working_directory: str | None,
    hooks: _Hooks,
    stream_stdout: bool,
    tee_to_console: bool,
):
    """
    Start /bin/sh -c <script> with positional args; capture pipes.
    Exit status is that of the last command in a pipeline (pure POSIX).
    """
    process = subprocess.Popen(
        ["/bin/sh", "-c", script, "pytosh", *argv[1:]],
        stdin=subprocess.PIPE if hooks.stdin_source is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=working_directory,
        env=dict(environment) if environment is not None else None,
        text=False,
        bufsize=0,  # unbuffered I/O for timely reads
    )

    captured_stdout = io.BytesIO()
    captured_stderr = io.BytesIO()
    stdout_done_event = threading.Event()
    stderr_done_event = threading.Event()

    stdout_stream = cast(IO[bytes], process.stdout)
    stderr_stream = cast(IO[bytes], process.stderr)

    # Prepare sinks & clear overwrite targets *before* data arrives.
    sink_manager = (
        _SinkManager(hooks)
        if (hooks.stdout_sinks or hooks.stderr_sinks or hooks.both_sinks)
        else None
    )
    if sink_manager is not None:
        sink_manager.start_all()

    # Pump stdout only in non-streaming mode (run/text/json/bytes paths).
    if not stream_stdout:
        stdout_done_event = _start_stream_pump_thread(
            stream=stdout_stream,
            capture_buffer=captured_stdout,
            tee_to_console=tee_to_console,
            is_stdout=True,
            sink_manager=sink_manager,
        )

    # Always pump stderr in a background thread so it does not block the child.
    stderr_done_event = _start_stream_pump_thread(
        stream=stderr_stream,
        capture_buffer=captured_stderr,
        tee_to_console=tee_to_console,
        is_stdout=False,
        sink_manager=sink_manager,
    )

    return (
        process,
        captured_stdout,
        captured_stderr,
        stdout_done_event,
        stderr_done_event,
        sink_manager,
    )


def _execute_shell_script(
    script: str,
    argv: list[str],
    environment: Mapping[str, str] | None,
    working_directory: str | None,
    tee_to_console: bool,
    hooks: _Hooks,
) -> ShelResult:
    """Run the shell script to completion, returning a ShelResult."""
    (
        process,
        captured_stdout,
        captured_stderr,
        stdout_done_event,
        stderr_done_event,
        sink_manager,
    ) = _spawn_shell_process(
        script=script,
        argv=argv,
        environment=environment,
        working_directory=working_directory,
        hooks=hooks,
        stream_stdout=False,
        tee_to_console=tee_to_console,
    )

    # Provide stdin if present (streaming if file-like).
    _start_stdin_pump_if_needed(process, hooks)

    # Wait for the process and both pump threads to finish before reading buffers.
    exit_code = process.wait()
    stdout_done_event.wait()
    stderr_done_event.wait()

    out_bytes = captured_stdout.getvalue()
    err_bytes = captured_stderr.getvalue()

    # Sinks have been written LIVE; just flush them.
    if sink_manager is not None:
        sink_manager.flush_all()

    return ShelResult(
        script=script,
        args=argv,
        exit_code=exit_code,
        stdout=out_bytes,
        stderr=err_bytes,
    )


def _resolve_environment(
    default_environment: Mapping[str, str] | None,
    local_environment: Mapping[str, str] | None,
) -> Mapping[str, str]:
    """
    Decide which environment to use:
    - If a local environment mapping is provided, use that as-is.
    - Otherwise use the default (or os.environ if default is None).
    """
    if local_environment is not None:
        return dict(local_environment)
    return dict(default_environment or os.environ)
