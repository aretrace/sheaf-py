from __future__ import annotations

import dataclasses
import hashlib
import json
import operator
import os
import tempfile
import threading
import time
import weakref
from collections import UserList
from collections.abc import Iterable
from contextlib import contextmanager
from os import PathLike
from typing import IO, Any, Generic, Self, SupportsIndex, TypeVar, cast


def _json_default(obj: Any) -> Any:
    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        return obj.model_dump()
    # Pydantic v1
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        return obj.dict()
    # Dataclass instances (avoid classes to keep pyright happy)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    # Path-like
    try:
        import os

        if isinstance(obj, os.PathLike):
            return os.fspath(obj)
    except Exception:
        pass
    # Sets
    if isinstance(obj, set):
        return list(obj)
    # Optional: NumPy
    try:
        import numpy as np  # pyright: ignore[reportMissingImports]

        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # pyright: ignore[reportAttributeAccessIssue]
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # pyright: ignore[reportAttributeAccessIssue]
    except Exception:
        pass
    # Fallback
    return repr(obj)


def json_lines_writer(
    file_object: IO[str],
    item: Any,
    **writer_options: Any,
) -> None:
    """
    Write a single item as JSON on one line (JSON Lines format).

    Recognized writer_options:
      - ensure_ascii (bool): passed through to json.dumps (default False)
      - indent (int | None): accepted for API symmetry; ignored for JSON Lines
      - json_default (callable): optional override for json.dumps(default=...)
    """
    ensure_ascii: bool = bool(writer_options.get("ensure_ascii", False))
    json_default = writer_options.get("json_default", _json_default)

    file_object.write(
        json.dumps(item, ensure_ascii=ensure_ascii, default=json_default) + "\n"
    )


class _InterProcessFileLock:
    """
    Cross-platform exclusive lock on a dedicated lock file.

    This uses:
      - POSIX: fcntl.flock(..., LOCK_EX)
      - Windows: msvcrt.locking(..., LK_LOCK)

    The lock is held for the duration of the context manager.
    """

    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self._file_handle: IO[str] | None = None

    def __enter__(self) -> "_InterProcessFileLock":
        # Open in text mode with a stable encoding; existence is sufficient.
        self._file_handle = cast(IO[str], open(self.lock_path, "a+", encoding="utf-8"))
        if os.name == "posix":
            import fcntl  # type: ignore

            fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_EX)
        else:
            import msvcrt  # type: ignore

            # Lock a single byte from the start of the file; this is a standard pattern.
            self._file_handle.seek(0)
            msvcrt.locking(self._file_handle.fileno(), msvcrt.LK_LOCK, 1)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._file_handle is None:
                return
            if os.name == "posix":
                import fcntl  # type: ignore

                fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_UN)
            else:
                import msvcrt  # type: ignore

                self._file_handle.seek(0)
                msvcrt.locking(self._file_handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            try:
                if self._file_handle is not None:
                    self._file_handle.close()
            except Exception:
                pass
            self._file_handle = None


T = TypeVar("T")


class Journal(UserList[T], Generic[T]):
    """
    A list-like log that mirrors every change to a human-readable JSON Lines file.

    - One item per line in JSONL; what you see in the file matches list positions.
    - Append operations stream new lines to the end for speed.
    - Any in-place mutation (replace, delete, insert, sort, etc.) rewrites the file
      so the on-disk order exactly matches the in-memory list.

    Durability-friendly:
    - Optional auto-flush after writes and optional `os.fsync()` for stronger durability.
    - Safe cleanup: the underlying file is auto-closed at GC/interpreter shutdown
      and can also be closed manually via `.close()`.

    Multi-writer support (this refactor):
    - All writes (appends and rewrites) take an exclusive inter-process lock on
      a dedicated lock file located in an OS-conventional per-user runtime directory
      (e.g., $XDG_RUNTIME_DIR or /run/user/<uid> on Linux; the system temp dir elsewhere).
      This serializes writers so lines do not interleave.
    - Before any full rewrite triggered by a mutation, we refresh `self.data` from disk
      under the lock to avoid dropping other writers' recent appends. The rewrite then
      atomically replaces the file via a unique temporary path.
    - Appends always target the current inode; we refresh the append handle under the lock.
    """

    def __init__(
        self,
        initial_items: Iterable[T] | None = None,
        *,
        file_path: str | PathLike[str] = "./journal.jsonl",
        writer_options: dict[str, Any] | None = None,
        open_mode: str = "a",
        auto_flush: bool = True,
        force_fsync: bool = False,
        write_initial_items_to_file: bool = True,
        text_encoding: str = "utf-8",
    ):
        super().__init__(list(initial_items) if initial_items is not None else [])

        self.file_path: str = os.fspath(file_path)
        self.writer_options: dict[str, Any] = (
            dict(writer_options) if writer_options else {}
        )
        self.auto_flush: bool = auto_flush
        self.force_fsync: bool = force_fsync
        self._append_open_mode: str = open_mode
        self._text_encoding: str = text_encoding
        self._closed: bool = False

        # Ensure the journal directory exists before the first open.
        journal_directory = os.path.dirname(self.file_path) or "."
        os.makedirs(journal_directory, exist_ok=True)

        # Locking
        self._lock_path: str = self._default_lock_path()
        self._thread_lock = threading.RLock()

        # Persistent handle used for fast appends; in-place mutations do a safe full rewrite.
        self._file_handle: IO[str] | None = cast(
            IO[str],
            open(self.file_path, self._append_open_mode, encoding=self._text_encoding),
        )
        self._file_identity: tuple[int, int] | None = self._current_file_identity()

        self._finalizer = weakref.finalize(self, self._safe_close, self._file_handle)

        if write_initial_items_to_file and self.data:
            # Append all initial items under a single lock for consistency.
            with self._acquire_exclusive_lock():
                self._refresh_append_handle_locked()
                for item in self.data:
                    self._append_one(item)

    # ------------- I/O helpers -------------

    @staticmethod
    def _safe_close(file_object: IO[str] | None) -> None:
        """Close the file object if present; swallow any exception."""
        try:
            if file_object is not None:
                file_object.close()
        except Exception:
            pass

    def _ensure_writable(self) -> None:
        """Raise if the list has been closed for further writes."""
        if self._closed or self._file_handle is None:
            raise RuntimeError("Cannot write: file is already closed.")

    @staticmethod
    def _per_user_runtime_dir() -> str:
        """
        Return an OS-conventional per-user runtime directory suitable for lock files.

        Priority:
        - POSIX: $XDG_RUNTIME_DIR if writable; else /run/user/<uid> if present; else tempfile.gettempdir()
        - Windows/others: tempfile.gettempdir()
        """
        if os.name == "posix":
            xdg = os.environ.get("XDG_RUNTIME_DIR")
            if xdg and os.path.isdir(xdg) and os.access(xdg, os.W_OK):
                return xdg
            try:
                uid = os.getuid()  # type: ignore[attr-defined]
                candidate = f"/run/user/{uid}"
                if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
                    return candidate
            except Exception:
                pass
            return tempfile.gettempdir()
        else:
            return tempfile.gettempdir()

    def _default_lock_path(self) -> str:
        """
        Compute a stable lock file path in the per-user runtime directory
        derived from the target JSONL path (basename + short hash).
        """
        base_dir = os.path.join(self._per_user_runtime_dir(), "journal", "locks")
        base_name = os.path.basename(self.file_path) or "journal.jsonl"
        digest = hashlib.sha256(
            os.path.abspath(self.file_path).encode("utf-8")
        ).hexdigest()[:16]
        return os.path.join(base_dir, f"{base_name}.{digest}.lock")

    @contextmanager
    def _acquire_exclusive_lock(self):
        """
        Take a per-instance thread lock and a cross-process file lock.

        All write operations (append or rewrite) go through this context.
        """
        with self._thread_lock:
            # Ensure both the journal directory and the lock directory exist.
            os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)
            os.makedirs(os.path.dirname(self._lock_path) or ".", exist_ok=True)
            with _InterProcessFileLock(self._lock_path):
                yield

    def _current_file_identity(self) -> tuple[int, int] | None:
        """
        Return a (device, inode) identity for the current append handle, if any.
        """
        try:
            if self._file_handle is None:
                return None
            stat_result = os.fstat(self._file_handle.fileno())
            return (stat_result.st_dev, stat_result.st_ino)
        except Exception:
            return None

    def _path_identity(self) -> tuple[int, int] | None:
        """Return (device, inode) for the on-disk path, or None if missing."""
        try:
            stat_result = os.stat(self.file_path)
            return (stat_result.st_dev, stat_result.st_ino)
        except FileNotFoundError:
            return None
        except Exception:
            return None

    def _refresh_append_handle_locked(self) -> None:
        """
        Ensure the persistent append handle targets the current inode.

        Callers must hold the exclusive lock.
        """
        # If the handle is missing or points at a different inode, reopen it.
        need_reopen = False
        if self._file_handle is None:
            need_reopen = True
        else:
            current_identity = self._current_file_identity()
            path_identity = self._path_identity()
            if current_identity != path_identity:
                # Another writer may have replaced the file; reopen.
                need_reopen = True

        if need_reopen:
            # Close old handle (best effort) and reopen.
            try:
                if self._file_handle is not None:
                    self._file_handle.close()
            except Exception:
                pass
            self._file_handle = cast(
                IO[str],
                open(
                    self.file_path, self._append_open_mode, encoding=self._text_encoding
                ),
            )
            self._file_identity = self._current_file_identity()

    def _load_disk_items_locked(self) -> list[Any]:
        """
        Read and parse all JSONL items from disk under the lock.

        Callers must hold the exclusive lock.
        """
        items: list[Any] = []
        try:
            with open(self.file_path, "r", encoding=self._text_encoding) as in_file:
                for line in in_file:
                    if line.strip() == "":
                        continue
                    items.append(json.loads(line))
        except FileNotFoundError:
            return []
        return items

    def _replace_data_from_disk_locked(self) -> None:
        """
        Replace in-memory data with the current on-disk content.

        This prevents in-place mutations from discarding other writers' earlier appends.
        Callers must hold the exclusive lock.
        """
        self.data[:] = cast(list[T], self._load_disk_items_locked())

    def _append_one(self, item: T) -> None:
        """
        Append a single item (streaming one JSON line), then flush/fsync as requested.

        Callers must hold the exclusive lock.
        """
        self._ensure_writable()
        file_object = self._file_handle
        assert file_object is not None  # For type checkers.

        json_lines_writer(file_object, item, **self.writer_options)

        if self.auto_flush:
            file_object.flush()
            if self.force_fsync:
                os.fsync(file_object.fileno())

    def _unique_temp_path(self) -> str:
        """
        Produce a unique temporary path for safe rewrites to avoid collisions
        across multiple writers.
        """
        directory = os.path.dirname(self.file_path) or "."
        base = os.path.basename(self.file_path)
        process_id = os.getpid()
        thread_id = threading.get_ident()
        nonce = time.monotonic_ns()
        return os.path.join(directory, f"{base}.tmp.{process_id}.{thread_id}.{nonce}")

    def _rewrite_all_locked(self) -> None:
        """
        Rewrite the entire file so that lines 1:1 match `self.data`.

        Implementation notes:
        - Runs under the exclusive lock.
        - Closes the persistent append handle to avoid platform-specific locking issues.
        - Writes to a uniquely named temporary file, fsyncs (if requested), then atomically
          replaces the target. Reopens the persistent append handle afterwards.
        """
        self._ensure_writable()

        # Close current persistent handle and mark its finalizer as executed.
        if getattr(self, "_finalizer", None) and self._finalizer.alive:
            file_object = self._file_handle
            if file_object is not None:
                try:
                    file_object.flush()  # Best-effort flush before closing.
                except Exception:
                    pass
            self._finalizer()
        self._file_handle = None

        tmp_path = self._unique_temp_path()

        with open(tmp_path, "w", encoding=self._text_encoding) as tmp_file:
            for item in self.data:
                json_lines_writer(tmp_file, item, **self.writer_options)
            tmp_file.flush()
            if self.force_fsync:
                os.fsync(tmp_file.fileno())

        # Atomic replace of the destination.
        os.replace(tmp_path, self.file_path)

        # When requested, fsync the directory entry to strengthen durability guarantees.
        if self.force_fsync:
            try:
                directory = os.path.dirname(self.file_path) or "."
                try:
                    dir_fd = os.open(directory, getattr(os, "O_DIRECTORY", os.O_RDONLY))
                except (
                    AttributeError,
                    FileNotFoundError,
                    NotADirectoryError,
                    PermissionError,
                ):
                    dir_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except Exception:
                # Directory fsync may not be available or permitted on all platforms.
                pass

        # Reopen persistent append handle and refresh identity.
        self._file_handle = cast(
            IO[str],
            open(self.file_path, self._append_open_mode, encoding=self._text_encoding),
        )
        self._file_identity = self._current_file_identity()
        self._finalizer = weakref.finalize(self, self._safe_close, self._file_handle)

    def _rewrite_all(self) -> None:
        """
        External entry point to rewrite the file; acquires the exclusive lock.
        """
        with self._acquire_exclusive_lock():
            self._rewrite_all_locked()

    # ------------- list ops (writes mirrored to disk) -------------

    # Fast-path appends (stream-only)
    def append(self, item: T) -> None:
        self._ensure_writable()
        with self._acquire_exclusive_lock():
            # Ensure we are appending to the current inode.
            self._refresh_append_handle_locked()
            super().append(item)
            self._append_one(item)

    def extend(self, other: Iterable[T]) -> None:
        self._ensure_writable()
        with self._acquire_exclusive_lock():
            self._refresh_append_handle_locked()
            for item in other:
                super().append(item)
                self._append_one(item)

    def __iadd__(self, other: Iterable[T]) -> Self:
        self.extend(other)
        return self

    # In-place mutations (full rewrite to keep file in sync)
    def __setitem__(self, i: SupportsIndex | slice, item: Any) -> None:
        self._ensure_writable()
        with self._acquire_exclusive_lock():
            # Refresh view to avoid dropping others' appends.
            self._replace_data_from_disk_locked()
            super().__setitem__(i, item)
            self._rewrite_all_locked()

    def __delitem__(self, i: SupportsIndex | slice) -> None:
        self._ensure_writable()
        with self._acquire_exclusive_lock():
            self._replace_data_from_disk_locked()
            super().__delitem__(i)
            self._rewrite_all_locked()

    def insert(self, i: SupportsIndex, item: T) -> None:
        self._ensure_writable()
        with self._acquire_exclusive_lock():
            self._replace_data_from_disk_locked()
            super().insert(operator.index(i), item)
            self._rewrite_all_locked()

    def pop(self, i: SupportsIndex = -1) -> T:
        self._ensure_writable()
        with self._acquire_exclusive_lock():
            self._replace_data_from_disk_locked()
            value = cast(T, super().pop(operator.index(i)))
            self._rewrite_all_locked()
            return value

    def clear(self) -> None:
        self._ensure_writable()
        with self._acquire_exclusive_lock():
            self._replace_data_from_disk_locked()
            super().clear()
            self._rewrite_all_locked()

    def remove(self, item: T) -> None:
        self._ensure_writable()
        with self._acquire_exclusive_lock():
            self._replace_data_from_disk_locked()
            super().remove(item)
            self._rewrite_all_locked()

    def reverse(self) -> None:
        self._ensure_writable()
        with self._acquire_exclusive_lock():
            self._replace_data_from_disk_locked()
            super().reverse()
            self._rewrite_all_locked()

    def sort(self, *args: Any, **kwargs: Any) -> None:
        self._ensure_writable()
        with self._acquire_exclusive_lock():
            self._replace_data_from_disk_locked()
            super().sort(*args, **kwargs)
            self._rewrite_all_locked()

    def __imul__(self, n: SupportsIndex) -> Self:
        self._ensure_writable()
        with self._acquire_exclusive_lock():
            self._replace_data_from_disk_locked()
            super().__imul__(operator.index(n))
            self._rewrite_all_locked()
            return self

    # ------------- utilities -------------

    def fork(
        self,
        *,
        file_path: str | PathLike[str],
        writer_options: dict[str, Any] | None = None,
        write_existing: bool = True,
        **kwargs: Any,
    ) -> Journal[T]:
        """
        Create a new Journal seeded with this instance's items.

        Parameters mirror __init__, but `file_path` is required. By default,
        all existing items are written to the new file unless `write_existing=False`.
        """
        return Journal(
            initial_items=self.data,
            file_path=file_path,
            writer_options=writer_options or self.writer_options,
            write_initial_items_to_file=write_existing,
            **kwargs,
        )

    # manual resource cleanup
    def close(self) -> None:
        """Close the underlying file handle and disable further writes."""
        if getattr(self, "_finalizer", None) and self._finalizer.alive:
            self._finalizer()
        self._file_handle = None
        self._closed = True
