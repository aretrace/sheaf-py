import difflib
import json
import os
import random
import re
import shutil
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from typing import Annotated


def list_directory(
    path: Annotated[
        str, "Directory path to list. Hidden files are included. Defaults to '.'"
    ] = ".",
) -> str:
    """List immediate entries in a directory (non-recursive; includes hidden files).

    This tool returns a **JSON-formatted string** describing each entry (name, type, size, mtime).
    It is intentionally **non-recursive**—use ``walk_tree`` for a multi-level view.

    Args:
        path (str): Directory to list. May be absolute or relative. Defaults to ``"."``.

    Returns:
        str: A pretty-printed JSON string with:
            - ``ok`` (bool): True on success.
            - ``path`` (str): Absolute normalized directory path.
            - ``entries`` (list[dict]): One dict per immediate entry:
              ``{"name": str, "type": "file"|"dir"|"symlink"|"other", "size": int, "mtime": float}``.

    Raises:
        None directly. On failure, returns JSON with:
            ``{"ok": false, "error": {"kind": <Type>, "message": <str>, "details": {...}}}``.

    Examples:
        >>> print(list_directory("."))
        {
          "ok": true,
          "path": "/home/me/project",
          "entries": [
            {"name": ".git", "type": "dir", "size": 4096, "mtime": 1724950000.0},
            {"name": "README.md", "type": "file", "size": 1203, "mtime": 1724951111.0}
          ]
        }

    Notes:
        - Hidden files (names starting with ``.``) are **included** by default.
        - The result order is the filesystem’s order (not sorted).
        - Use ``read_text`` / ``write_text`` / ``edit_text`` for file operations.
    """

    def make_error(kind: str, message: str, **details) -> str:
        return json.dumps(
            {
                "ok": False,
                "error": {"kind": kind, "message": message, "details": details},
            },
            indent=2,
        )

    try:
        absolute_path = os.path.abspath(path)
        if not os.path.exists(absolute_path):
            return make_error("FileNotFoundError", f"Path does not exist: {path}")
        if not os.path.isdir(absolute_path):
            return make_error("NotADirectoryError", f"Path is not a directory: {path}")

        entries = []
        for entry_name in os.listdir(absolute_path):
            entry_path = os.path.join(absolute_path, entry_name)
            try:
                status = os.lstat(entry_path)
                if stat.S_ISDIR(status.st_mode):
                    entry_type = "dir"
                elif stat.S_ISREG(status.st_mode):
                    entry_type = "file"
                elif stat.S_ISLNK(status.st_mode):
                    entry_type = "symlink"
                else:
                    entry_type = "other"
                entries.append(
                    {
                        "name": entry_name,
                        "type": entry_type,
                        "size": int(getattr(status, "st_size", 0)),
                        "mtime": float(status.st_mtime),
                    }
                )
            except OSError as error:
                entries.append(
                    {"name": entry_name, "type": "unreadable", "error": str(error)}
                )

        return json.dumps(
            {"ok": True, "path": absolute_path, "entries": entries}, indent=2
        )

    except PermissionError as error:
        return make_error("PermissionError", str(error))
    except OSError as error:
        return make_error("OSError", str(error))


def read_text(
    path: Annotated[
        str, "Path to a UTF-8 text file to read. Refuses likely-binary files."
    ],
) -> str:
    """Read a UTF‑8 text file safely (refuses likely-binary content).

    Heuristic: scan the first 8192 bytes for NUL (``\\x00``). If present, the file is treated as likely
    binary and the call fails with a structured error. This mirrors common tooling heuristics.

    Args:
        path (str): Filesystem path to the text file to read.

    Returns:
        str: JSON string:
            - ``ok`` (bool): True on success.
            - ``path`` (str): Absolute path.
            - ``size_bytes`` (int): Size of the returned UTF-8 data (post-encoding).
            - ``content`` (str): Full file text.

    Raises:
        None directly. On failure, returns:
            ``{"ok": false, "error": {"kind": "BinaryFileSuspected"|...}}``.

    Examples:
        >>> result = read_text("README.md")
        >>> # parse JSON if needed
        >>> # json.loads(result)["content"]

    Warnings:
        - Only use for **UTF‑8 text**. Binary or other encodings will error.

    Notes:
        - Reads the entire file into memory; avoid for very large files.
        - This design helps agents avoid mis-editing binary data.
    """

    def make_error(kind: str, message: str, **details) -> str:
        return json.dumps(
            {
                "ok": False,
                "path": path,
                "error": {"kind": kind, "message": message, "details": details},
            },
            indent=2,
        )

    try:
        absolute_path = os.path.abspath(path)
        with open(absolute_path, "rb") as raw_file:
            header_bytes = raw_file.read(8192)
            if b"\x00" in header_bytes:
                return make_error(
                    "BinaryFileSuspected",
                    "NUL byte found in first 8192 bytes; refusing to read as text.",
                )
        with open(absolute_path, "r", encoding="utf-8") as text_file:
            content_text = text_file.read()
        size_bytes = len(content_text.encode("utf-8"))
        return json.dumps(
            {
                "ok": True,
                "path": absolute_path,
                "size_bytes": size_bytes,
                "content": content_text,
            },
            indent=2,
        )

    except FileNotFoundError as error:
        return make_error("FileNotFoundError", str(error))
    except IsADirectoryError as error:
        return make_error("IsADirectoryError", str(error))
    except PermissionError as error:
        return make_error("PermissionError", str(error))
    except UnicodeDecodeError as error:
        return make_error(
            "UnicodeDecodeError", "File is not valid UTF-8.", reason=str(error)
        )
    except OSError as error:
        return make_error("OSError", str(error))


def write_text(
    path: Annotated[
        str, "Path to write. Creates or replaces the entire UTF-8 text file atomically."
    ],
    content: Annotated[str, "Full text to write (UTF-8)."],
    create_backup: Annotated[
        bool, "If True and file exists, create '<name>.bak' before replacing."
    ] = True,
) -> str:
    """Atomically write a UTF‑8 text file, preserving mode and optionally creating a backup.

    Pattern:
      1) Write to a temporary file in the same directory.
      2) Flush and fsync the temp file.
      3) If overwriting, copy mode bits from the existing file.
      4) Atomically replace with ``os.replace``.
      5) Best-effort fsync on the containing directory to persist the rename.

    Args:
        path (str): Destination file path.
        content (str): Entire text content to write.
        create_backup (bool): If True and the destination exists, create ``<path>.bak`` first.

    Returns:
        str: JSON string:
            - ``ok`` (bool): True on success.
            - ``path`` (str): Absolute path of the written file.
            - ``bytes_written`` (int): Number of bytes written (UTF-8).
            - ``backup_path`` (str, optional): Path of the created backup if any.

    Raises:
        None directly. On failure, returns an ``error`` object with kind and message.

    Examples:
        >>> print(write_text("notes.txt", "hello\\n"))
        >>> print(write_text("config.ini", "[core]\\n", create_backup=True))

    Notes:
        - This avoids torn writes and favors crash safety over raw speed.
        - If you do not need backups, pass ``create_backup=False``.
    """

    def make_error(kind: str, message: str, **details) -> str:
        return json.dumps(
            {
                "ok": False,
                "path": path,
                "error": {"kind": kind, "message": message, "details": details},
            },
            indent=2,
        )

    try:
        absolute_path = os.path.abspath(path)
        directory_path = os.path.dirname(absolute_path) or "."
        os.makedirs(directory_path, exist_ok=True)

        file_existed_before = os.path.exists(absolute_path)
        previous_mode_bits = None
        if file_existed_before:
            try:
                previous_mode_bits = os.stat(absolute_path).st_mode
            except OSError:
                previous_mode_bits = None

        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=".tmp_write_", dir=directory_path, text=True
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary_file:
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            if file_existed_before and previous_mode_bits is not None:
                try:
                    os.chmod(temporary_path, previous_mode_bits)
                except OSError:
                    pass

            backup_path = None
            if file_existed_before and create_backup:
                try:
                    backup_path = absolute_path + ".bak"
                    shutil.copy2(absolute_path, backup_path)
                except OSError:
                    backup_path = None  # non-fatal

            os.replace(temporary_path, absolute_path)
            # best-effort directory fsync to persist rename
            try:
                directory_fd = os.open(directory_path, os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except Exception:
                pass

            return json.dumps(
                {
                    "ok": True,
                    "path": absolute_path,
                    "bytes_written": len(content.encode("utf-8")),
                    **({"backup_path": backup_path} if backup_path else {}),
                },
                indent=2,
            )

        finally:
            try:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
            except Exception:
                pass

    except PermissionError as error:
        return make_error("PermissionError", str(error))
    except OSError as error:
        return make_error("OSError", str(error))


def edit_text(
    path: Annotated[str, "Target UTF-8 text file to edit."],
    old_text: Annotated[str, "Exact substring to replace (literal; no regex)."],
    new_text: Annotated[str, "Replacement text (literal)."],
    replace_all: Annotated[
        bool, "If True, replace all matches; if False, require exactly one match."
    ] = False,
) -> str:
    """Edit a text file via literal search/replace with **unique-match safety** and atomic write.

    Default behavior enforces **exactly one** match. If none or multiple matches are found, the tool returns a
    clear, machine-parseable error including **candidate line numbers** so the agent can refine its context.

    Args:
        path (str): Path to the UTF-8 text file.
        old_text (str): Exact substring to find.
        new_text (str): Replacement text.
        replace_all (bool): If True, replace all occurrences; if False, require exactly one match.

    Returns:
        str: JSON string:
            - ``ok`` (bool): True on success.
            - ``path`` (str): Edited file path.
            - ``replacements`` (int): Number of replacements applied.
            - ``changed_lines`` (list[int]): 1-based line numbers of affected regions (approximate).
            - ``diff`` (str): Unified diff preview (may be truncated if large).

    Errors:
        If no or multiple matches (and ``replace_all`` is False), returns:
            ``{"ok": false, "error": {"kind": "NoMatch"|"MultipleMatches", "details": {"occurrences": int, "candidate_lines": [int,...]}}}``.
        Other failures include ``BinaryFileSuspected``, ``UnicodeDecodeError``, etc.

    Examples:
        Replace a single precise location:
            >>> edit_text("pyproject.toml", 'name = "demo"', 'name = "app"')
        Replace all:
            >>> edit_text("README.md", "FooLib", "BarLib", replace_all=True)

    Notes:
        - Literal replacement only (no regex). For structured patches, build a diff externally then write with ``write_text``.
        - Uses the same atomic write pattern as ``write_text``.
    """

    def make_error(kind: str, message: str, **details) -> str:
        return json.dumps(
            {
                "ok": False,
                "path": path,
                "error": {"kind": kind, "message": message, "details": details},
            },
            indent=2,
        )

    absolute_path = os.path.abspath(path)
    try:
        with open(absolute_path, "rb") as raw_file:
            header_bytes = raw_file.read(8192)
            if b"\x00" in header_bytes:
                return make_error(
                    "BinaryFileSuspected",
                    "NUL byte in first 8192 bytes; refusing to edit as text.",
                )
        with open(absolute_path, "r", encoding="utf-8") as text_file:
            original_text = text_file.read()
    except FileNotFoundError as error:
        return make_error("FileNotFoundError", str(error))
    except IsADirectoryError as error:
        return make_error("IsADirectoryError", str(error))
    except PermissionError as error:
        return make_error("PermissionError", str(error))
    except UnicodeDecodeError as error:
        return make_error(
            "UnicodeDecodeError", "File is not valid UTF-8.", reason=str(error)
        )
    except OSError as error:
        return make_error("OSError", str(error))

    if old_text == "":
        return make_error("InvalidArgument", "old_text must be non-empty.")

    # Find all occurrence byte offsets, then compute line numbers.
    occurrence_offsets = []
    search_start = 0
    while True:
        index = original_text.find(old_text, search_start)
        if index == -1:
            break
        occurrence_offsets.append(index)
        search_start = index + len(old_text)

    if not occurrence_offsets:
        return make_error("NoMatch", f'No match for "{old_text}" found.', occurrences=0)

    if not replace_all and len(occurrence_offsets) != 1:
        candidate_line_numbers = [
            original_text.count("\n", 0, pos) + 1 for pos in occurrence_offsets
        ]
        return make_error(
            "MultipleMatches",
            f'Found {len(occurrence_offsets)} matches for "{old_text}". Provide more context or set replace_all=True.',
            occurrences=len(occurrence_offsets),
            candidate_lines=candidate_line_numbers,
        )

    replaced_text = (
        original_text.replace(old_text, new_text)
        if replace_all
        else original_text.replace(old_text, new_text, 1)
    )

    changed_line_numbers = [
        original_text.count("\n", 0, pos) + 1 for pos in occurrence_offsets
    ]
    replacements_count = len(occurrence_offsets) if replace_all else 1

    old_lines = original_text.splitlines(keepends=False)
    new_lines = replaced_text.splitlines(keepends=False)
    diff_iterator = difflib.unified_diff(
        old_lines, new_lines, fromfile=path, tofile=path, n=3, lineterm=""
    )
    diff_text = "\n".join(diff_iterator)
    if len(diff_text) > 100_000:
        diff_text = diff_text[:100_000] + "\n... (diff truncated)"

    # Atomic write (inline; no shared helpers to keep tools independent).
    try:
        directory_path = os.path.dirname(absolute_path) or "."
        file_existed_before = os.path.exists(absolute_path)
        previous_mode_bits = (
            os.stat(absolute_path).st_mode if file_existed_before else None
        )

        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=".tmp_edit_", dir=directory_path, text=True
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary_file:
                temporary_file.write(replaced_text)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            if file_existed_before and previous_mode_bits is not None:
                try:
                    os.chmod(temporary_path, previous_mode_bits)
                except OSError:
                    pass

            os.replace(temporary_path, absolute_path)
            try:
                directory_fd = os.open(directory_path, os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except Exception:
                pass
        finally:
            try:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
            except Exception:
                pass
    except PermissionError as error:
        return make_error("PermissionError", str(error))
    except OSError as error:
        return make_error("OSError", str(error))

    return json.dumps(
        {
            "ok": True,
            "path": absolute_path,
            "replacements": replacements_count,
            "changed_lines": changed_line_numbers,
            "diff": diff_text,
        },
        indent=2,
    )


def diff_files(
    path_a: Annotated[str, "First text file (original)."],
    path_b: Annotated[str, "Second text file (modified)."],
    context_lines: Annotated[
        int, "Number of context lines in the unified diff. Defaults to 3."
    ] = 3,
) -> str:
    """Create a **unified diff** between two UTF‑8 text files (empty string if identical).

    Args:
        path_a (str): Path to the original file.
        path_b (str): Path to the modified file.
        context_lines (int): Number of context lines to include in each hunk.

    Returns:
        str: JSON string:
            - ``ok`` (bool): True on success.
            - ``from`` (str): Absolute path of ``path_a``.
            - ``to`` (str): Absolute path of ``path_b``.
            - ``diff`` (str): Unified diff text (may be empty).

    Errors:
        Returns ``BinaryFileSuspected`` if either file looks binary, and other structured errors for I/O issues.

    Examples:
        >>> print(diff_files("old.py", "new.py"))
    """

    def make_error(kind: str, message: str, **details) -> str:
        return json.dumps(
            {
                "ok": False,
                "from": path_a,
                "to": path_b,
                "error": {"kind": kind, "message": message, "details": details},
            },
            indent=2,
        )

    def looks_binary(file_path: str) -> bool:
        try:
            with open(file_path, "rb") as raw_file:
                return b"\x00" in raw_file.read(8192)
        except OSError:
            return False

    try:
        absolute_a = os.path.abspath(path_a)
        absolute_b = os.path.abspath(path_b)
        if looks_binary(absolute_a) or looks_binary(absolute_b):
            return make_error(
                "BinaryFileSuspected",
                "One or both files appear to be binary; unified text diff not shown.",
            )
        with (
            open(absolute_a, "r", encoding="utf-8") as file_a,
            open(absolute_b, "r", encoding="utf-8") as file_b,
        ):
            lines_a = file_a.read().splitlines(keepends=False)
            lines_b = file_b.read().splitlines(keepends=False)
        diff_iterator = difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=path_a,
            tofile=path_b,
            n=context_lines,
            lineterm="",
        )
        diff_text = "\n".join(diff_iterator)
        return json.dumps(
            {"ok": True, "from": absolute_a, "to": absolute_b, "diff": diff_text},
            indent=2,
        )

    except FileNotFoundError as error:
        return make_error("FileNotFoundError", str(error))
    except PermissionError as error:
        return make_error("PermissionError", str(error))
    except UnicodeDecodeError as error:
        return make_error(
            "UnicodeDecodeError", "File(s) not valid UTF-8.", reason=str(error)
        )
    except OSError as error:
        return make_error("OSError", str(error))


def search_text(
    root: Annotated[str, "Directory to search recursively."],
    query: Annotated[
        str, "Substring or regex to search for (set use_regex to enable regex)."
    ],
    use_regex: Annotated[
        bool, "If True, treat 'query' as a regular expression."
    ] = False,
    max_matches_per_file: Annotated[int, "Maximum matches to record per file."] = 50,
) -> str:
    """Recursively search **all files** under ``root`` for ``query`` (includes hidden files).

    The function **skips likely-binary files** (NUL-heuristic) and non-UTF‑8 text. It returns
    structured matches (path, line, column, snippet). Designed to be easy for an LLM to consume.

    Args:
        root (str): Directory to search from (recursive).
        query (str): The text or regex pattern to find.
        use_regex (bool): If True, interpret ``query`` as a Python regular expression.
        max_matches_per_file (int): Limit for matches recorded per file to keep output bounded.

    Returns:
        str: JSON string with:
            - ``ok`` (bool): True on success.
            - ``root`` (str): Absolute search root.
            - ``query`` (str): Echo of the query.
            - ``regex`` (bool): Whether regex mode was used.
            - ``files_scanned`` (int): Number of files scanned (text-decoded).
            - ``matches`` (list[dict]): ``{"path": str, "line": int, "column": int, "snippet": str}``.

    Errors:
        Returns structured errors for invalid arguments or I/O problems.

    Examples:
        >>> print(search_text(".", "TODO"))
        >>> print(search_text("src", r"def\\s+main\\(", use_regex=True))

    Notes:
        - Hidden files and directories are **not** skipped.
        - Very large repositories can generate a lot of output; adjust your query or limit.
    """

    def make_error(kind: str, message: str, **details) -> str:
        return json.dumps(
            {
                "ok": False,
                "root": root,
                "query": query,
                "error": {"kind": kind, "message": message, "details": details},
            },
            indent=2,
        )

    def looks_binary(file_path: str) -> bool:
        try:
            with open(file_path, "rb") as raw_file:
                return b"\x00" in raw_file.read(8192)
        except OSError:
            return True

    if not query:
        return make_error("InvalidArgument", "query must be non-empty.")

    try:
        absolute_root = os.path.abspath(root)
        if not os.path.exists(absolute_root):
            return make_error("FileNotFoundError", f"Root does not exist: {root}")
        if not os.path.isdir(absolute_root):
            return make_error("NotADirectoryError", f"Root is not a directory: {root}")

        matches = []
        files_scanned_count = 0
        compiled_pattern = re.compile(query) if use_regex else None

        for directory_path, directory_names, file_names in os.walk(absolute_root):
            # Do not skip hidden paths; scan everything under root.
            for file_name in file_names:
                file_path = os.path.join(directory_path, file_name)
                if looks_binary(file_path):
                    continue
                try:
                    with open(file_path, "r", encoding="utf-8") as text_file:
                        files_scanned_count += 1
                        for line_number, line_text in enumerate(text_file, start=1):
                            single_line = line_text.rstrip("\n")
                            if use_regex:
                                for match in compiled_pattern.finditer(single_line):  # pyright: ignore[reportOptionalMemberAccess]
                                    matches.append(
                                        {
                                            "path": file_path,
                                            "line": line_number,
                                            "column": match.start() + 1,
                                            "snippet": single_line,
                                        }
                                    )
                                    if (
                                        sum(
                                            1 for m in matches if m["path"] == file_path
                                        )
                                        >= max_matches_per_file
                                    ):
                                        break
                            else:
                                column_index = single_line.find(query)
                                while column_index != -1:
                                    matches.append(
                                        {
                                            "path": file_path,
                                            "line": line_number,
                                            "column": column_index + 1,
                                            "snippet": single_line,
                                        }
                                    )
                                    if (
                                        sum(
                                            1 for m in matches if m["path"] == file_path
                                        )
                                        >= max_matches_per_file
                                    ):
                                        break
                                    column_index = single_line.find(
                                        query, column_index + 1
                                    )
                except UnicodeDecodeError:
                    continue
                except PermissionError:
                    continue

        return json.dumps(
            {
                "ok": True,
                "root": absolute_root,
                "query": query,
                "regex": bool(use_regex),
                "files_scanned": files_scanned_count,
                "matches": matches,
            },
            indent=2,
        )

    except PermissionError as error:
        return make_error("PermissionError", str(error))
    except OSError as error:
        return make_error("OSError", str(error))


def walk_tree(
    root: Annotated[
        str, "Directory to traverse recursively. Hidden files are included."
    ] = ".",
    max_depth: Annotated[int, "Maximum depth to traverse (0 lists just root)."] = 2,
) -> str:
    """Produce a lightweight recursive directory tree up to ``max_depth`` (includes hidden files).

    Args:
        root (str): Starting directory.
        max_depth (int): Maximum depth (0 = only root).

    Returns:
        str: JSON string:
            - ``ok`` (bool): True on success.
            - ``root`` (str): Absolute root.
            - ``max_depth`` (int)
            - ``entries`` (list[dict]): One per file/dir/symlink:
              ``{"path": str, "type": "file"|"dir"|"symlink"|"other"|"unreadable", "size": int, "mtime": float, "depth": int}``.

    Examples:
        >>> print(walk_tree(".", max_depth=1))
    """

    def make_error(kind: str, message: str, **details) -> str:
        return json.dumps(
            {
                "ok": False,
                "root": root,
                "error": {"kind": kind, "message": message, "details": details},
            },
            indent=2,
        )

    try:
        absolute_root = os.path.abspath(root)
        if not os.path.exists(absolute_root):
            return make_error("FileNotFoundError", f"Root does not exist: {root}")
        if not os.path.isdir(absolute_root):
            return make_error("NotADirectoryError", f"Root is not a directory: {root}")

        entries = []
        for directory_path, directory_names, file_names in os.walk(absolute_root):
            depth_value = os.path.relpath(directory_path, absolute_root).count(os.sep)
            if depth_value > max_depth:
                directory_names[:] = []
                continue
            for entry_name in directory_names + file_names:
                entry_path = os.path.join(directory_path, entry_name)
                try:
                    status = os.lstat(entry_path)
                    if stat.S_ISDIR(status.st_mode):
                        entry_type = "dir"
                    elif stat.S_ISREG(status.st_mode):
                        entry_type = "file"
                    elif stat.S_ISLNK(status.st_mode):
                        entry_type = "symlink"
                    else:
                        entry_type = "other"
                    entries.append(
                        {
                            "path": entry_path,
                            "type": entry_type,
                            "size": int(getattr(status, "st_size", 0)),
                            "mtime": float(status.st_mtime),
                            "depth": depth_value + 1,
                        }
                    )
                except OSError:
                    entries.append(
                        {
                            "path": entry_path,
                            "type": "unreadable",
                            "depth": depth_value + 1,
                        }
                    )

        return json.dumps(
            {
                "ok": True,
                "root": absolute_root,
                "max_depth": int(max_depth),
                "entries": entries,
            },
            indent=2,
        )

    except PermissionError as error:
        return make_error("PermissionError", str(error))
    except OSError as error:
        return make_error("OSError", str(error))


# TODO: use `shel`
def run_shell(
    command: Annotated[str, "Shell command to run (non-interactive)."],
    timeout_seconds: Annotated[
        int, "Kill the command if it runs longer than this many seconds."
    ] = 60,
    max_output_chars: Annotated[
        int, "Truncate stdout/stderr to this many characters each."
    ] = 200_000,
) -> str:
    """Run a shell command with timeout and captured output (stdout + stderr).

    Args:
        command (str): Command string (will be executed with ``shell=True``).
        timeout_seconds (int): Maximum execution time in seconds before killing the process.
        max_output_chars (int): Maximum characters to keep for each of stdout/stderr (truncates beyond this).

    Returns:
        str: JSON string:
            - ``ok`` (bool): True on success.
            - ``command`` (str): Echo of the command string.
            - ``returncode`` (int): Exit code returned by the shell.
            - ``timed_out`` (bool): True if the process exceeded the timeout.
            - ``duration_seconds`` (float): Wall-clock duration.
            - ``truncated`` (bool): Whether any stream was truncated.
            - ``stdout`` (str): Captured standard output.
            - ``stderr`` (str): Captured standard error.

    Errors:
        On timeout or system error, returns ``ok: false`` with a structured error and any partial outputs.

    Examples:
        >>> print(run_shell("echo hello"))
        >>> print(run_shell("ls -1", timeout_seconds=10))

    Warnings:
        - **Security**: Do not pass untrusted strings directly (shell injection risk).
        - Long-running commands may still produce huge output; truncation protects the agent loop.
    """

    def as_json(**fields) -> str:
        return json.dumps(fields, indent=2)

    start_time = time.time()
    try:
        completed = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout_seconds
        )
        duration = time.time() - start_time
        standard_output = completed.stdout or ""
        standard_error = completed.stderr or ""
        was_truncated = False

        if len(standard_output) > max_output_chars:
            standard_output = (
                standard_output[:max_output_chars] + "\n... (stdout truncated)"
            )
            was_truncated = True
        if len(standard_error) > max_output_chars:
            standard_error = (
                standard_error[:max_output_chars] + "\n... (stderr truncated)"
            )
            was_truncated = True

        return as_json(
            ok=True,
            command=command,
            returncode=int(completed.returncode),
            timed_out=False,
            duration_seconds=duration,
            truncated=was_truncated,
            stdout=standard_output,
            stderr=standard_error,
        )

    except subprocess.TimeoutExpired as error:
        duration = time.time() - start_time
        # TimeoutExpired has .output and .stderr attributes; be defensive if not present
        partial_out = getattr(error, "output", "") or getattr(error, "stdout", "") or ""
        partial_err = getattr(error, "stderr", "") or ""
        return as_json(
            ok=False,
            command=command,
            error={
                "kind": "TimeoutExpired",
                "message": f"Command exceeded {timeout_seconds}s.",
                "details": {"returncode": None},
            },
            timed_out=True,
            duration_seconds=duration,
            stdout=partial_out,
            stderr=partial_err,
        )
    except OSError as error:
        return as_json(
            ok=False, command=command, error={"kind": "OSError", "message": str(error)}
        )


def fetch_web(
    url: Annotated[str, "URL to fetch (must start with http:// or https://)."],
    max_bytes: Annotated[
        int, "Maximum response bytes to read (protects memory)."
    ] = 1_000_000,
) -> str:
    """Fetch a URL with a simple HTTP GET and return text (UTF‑8) or a hex preview.

    Args:
        url (str): HTTP(S) URL to retrieve.
        max_bytes (int): Upper bound on bytes read from the response (``max_bytes+1`` is read to detect truncation).

    Returns:
        str: JSON string:
            - ``ok`` (bool): True on success.
            - ``url`` (str)
            - ``status`` (int|None): HTTP status code if available.
            - ``headers`` (dict[str,str]): Response headers.
            - ``truncated`` (bool): Whether the body exceeded ``max_bytes``.
            - ``content`` (str): UTF‑8 decoded text (if decodable), else omitted.
            - ``hex_preview`` (str): First bytes as hex when not UTF‑8 (and a ``note`` explaining why).

    Errors:
        Returns ``HTTPError``/``URLError``/``OSError`` with details on failure.

    Examples:
        >>> print(fetch_web("https://example.com"))
        >>> print(fetch_web("https://api.github.com", max_bytes=200_000))
    """

    def make_error(kind: str, message: str, **details) -> str:
        return json.dumps(
            {
                "ok": False,
                "url": url,
                "error": {"kind": kind, "message": message, "details": details},
            },
            indent=2,
        )

    if not (url.lower().startswith("http://") or url.lower().startswith("https://")):
        return make_error("ValueError", "Only HTTP(S) URLs are supported.")

    request = urllib.request.Request(
        url, headers={"User-Agent": "AI-CLI-Agent/1.0 (+stdlib urllib)"}
    )
    try:
        with urllib.request.urlopen(request) as response:
            status_code = getattr(response, "status", None)
            headers_dict = {key: value for key, value in response.getheaders()}

            data = response.read(max_bytes + 1)
            was_truncated = len(data) > max_bytes
            if was_truncated:
                data = data[:max_bytes]

            try:
                text = data.decode("utf-8")
                return json.dumps(
                    {
                        "ok": True,
                        "url": url,
                        "status": status_code,
                        "headers": headers_dict,
                        "truncated": was_truncated,
                        "content": text,
                    },
                    indent=2,
                )
            except UnicodeDecodeError:
                return json.dumps(
                    {
                        "ok": True,
                        "url": url,
                        "status": status_code,
                        "headers": headers_dict,
                        "truncated": was_truncated,
                        "note": "Non-UTF8 or binary content; returning hex preview.",
                        "hex_preview": data[:256].hex(),
                    },
                    indent=2,
                )
    except urllib.error.HTTPError as error:
        return make_error("HTTPError", f"HTTP {error.code}", status=error.code)
    except urllib.error.URLError as error:
        return make_error("URLError", str(error.reason))
    except OSError as error:
        return make_error("OSError", str(error))


def add_note(
    text: Annotated[
        str, "Single note content to append. A single write is used for atomic append."
    ],
    path: Annotated[
        str, "Scratchpad file path. Defaults to 'scratchpad.md'."
    ] = "scratchpad.md",
) -> str:
    """Append a note to a scratchpad file using a **single atomic append** (portable; no flock).

    This function performs a single low-level append write to minimize interleaving when multiple
    processes write concurrently. It does **not** use ``flock`` so it remains portable across
    Unix-like systems that may not expose it consistently.

    Args:
        text (str): The note text to append (a newline is added).
        path (str): Scratchpad file path. Defaults to ``"scratchpad.md"``.

    Returns:
        str: JSON string:
            - ``ok`` (bool): True on success.
            - ``path`` (str): Absolute scratchpad path.
            - ``appended_chars`` (int): Characters appended (including newline).

    Examples:
        >>> print(add_note("Plan: refactor module X"))
        >>> print(add_note("Remember to run tests", path="AGENT_NOTES.md"))
    """

    def make_error(kind: str, message: str, **details) -> str:
        return json.dumps(
            {
                "ok": False,
                "path": path,
                "error": {"kind": kind, "message": message, "details": details},
            },
            indent=2,
        )

    try:
        absolute_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(absolute_path) or ".", exist_ok=True)
        appended_text = (text + "\n").encode("utf-8")

        # Use a single O_APPEND write for best-effort atomicity on POSIX regular files.
        # One write call => minimal chance of interleaving with writers in other processes.
        file_descriptor = os.open(
            absolute_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o666
        )
        try:
            os.write(file_descriptor, appended_text)
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)

        return json.dumps(
            {"ok": True, "path": absolute_path, "appended_chars": len(text) + 1},
            indent=2,
        )

    except PermissionError as error:
        return make_error("PermissionError", str(error))
    except OSError as error:
        return make_error("OSError", str(error))


def read_notes(
    path: Annotated[
        str, "Scratchpad file path to read. Defaults to 'scratchpad.md'."
    ] = "scratchpad.md",
) -> str:
    """Read the entire scratchpad file (no locking; simple and portable).

    Args:
        path (str): Scratchpad file path.

    Returns:
        str: JSON string:
            - ``ok`` (bool): True on success.
            - ``path`` (str): Absolute path.
            - ``content`` (str): Entire file content (empty string if file does not exist).

    Examples:
        >>> print(read_notes())
    """

    def make_error(kind: str, message: str, **details) -> str:
        return json.dumps(
            {
                "ok": False,
                "path": path,
                "error": {"kind": kind, "message": message, "details": details},
            },
            indent=2,
        )

    try:
        absolute_path = os.path.abspath(path)
        if not os.path.exists(absolute_path):
            return json.dumps(
                {"ok": True, "path": absolute_path, "content": ""}, indent=2
            )
        with open(absolute_path, "r", encoding="utf-8") as file_object:
            content_text = file_object.read()
        return json.dumps(
            {"ok": True, "path": absolute_path, "content": content_text}, indent=2
        )
    except PermissionError as error:
        return make_error("PermissionError", str(error))
    except OSError as error:
        return make_error("OSError", str(error))


def get_weather(location: str) -> str:
    """Get weather information for a location.

    Args:
        location: The city or location to get weather for
    """
    return f"{location} is {random.choice(['sunny', 'cloudy'])} and {random.randint(32, 98)}°F."
