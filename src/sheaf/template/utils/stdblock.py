"""
POSIX-only minimal multiline prompt (no curses, Python standard library only).

Keys
- Enter            -> submit block
- Alt/Option+Enter -> insert newline at BOL of next line's CONTENT (after continuation prompt)
- Ctrl+J           -> insert newline (portable fallback)
- ← / →            -> move within line; cross line boundaries at ends
- ↑ / ↓            -> move lines, preserving goal *visual* column
- Backspace        -> delete across lines
- Ctrl+D           -> exit (EOF)
- Ctrl+C           -> KeyboardInterrupt
"""

import os
import re
import select
import sys
import termios
import tty
import unicodedata

CSI = "\x1b["

# Strip both SGR and non-SGR ANSI escapes so we can compute *visible* width.
# We first remove long OSC/DCS/SOS/PM/APC sequences (7-bit and 8-bit forms),
# then remove CSI and other single-ESC sequences.
_OSC_DCS_RE = re.compile(
    r"""
    (?:                                 # --- OSC (Operating System Command) ---
        (?:\x1B\]|\x9D)                 #   ESC ]  or  8-bit OSC
        [^\x07\x1B\x9C]*                #   payload (no BEL/ESC/ST)
        (?:\x07|\x1B\\|\x9C)            #   BEL or ST (ESC \) or 8-bit ST
    )
    |
    (?:                                 # --- DCS (Device Control String) ---
        (?:\x1BP|\x90)                  #   ESC P  or  8-bit DCS
        .*?                             #   payload (non-greedy)
        (?:\x1B\\|\x9C)                 #   ST (ESC \) or 8-bit ST
    )
    |
    (?:                                 # --- SOS / PM / APC strings ---
        (?:\x1BX|\x98|\x1B\^|\x9E|\x1B_|\x9F)
        .*?
        (?:\x1B\\|\x9C)
    )
    """,
    re.VERBOSE | re.DOTALL,
)

_CSI_ETC_RE = re.compile(
    r"""
    (?:\x9B[0-?]*[ -/]*[@-~])           # 8-bit CSI
    |
    (?:\x1B\[[0-?]*[ -/]*[@-~])         # 7-bit CSI
    |
    (?:\x1BO[@-~])                      # SS3 (ESC O final)
    |
    (?:                                 # Generic single-ESC sequence with optional
        \x1B                            # intermediates (but *not* ESC [ which starts CSI)
        (?:[@-Z\\^_\x60-~]|[ -/][ -/]*[@-~])
    )
    """,
    re.VERBOSE,
)


def strip_ansi(s: str) -> str:
    """Remove ANSI escape sequences (SGR and non-SGR) from s."""
    s = _OSC_DCS_RE.sub("", s)
    s = _CSI_ETC_RE.sub("", s)
    return s


def display_width(s: str) -> int:
    """
    Approximate terminal column width of s, ignoring ANSI escapes.
    Counts combining marks as width 0; East Asian Wide/Fullwidth as 2; others as 1.
    """
    s = strip_ansi(s)
    w = 0
    for ch in s:
        if unicodedata.combining(ch):
            continue
        # Treat format chars (e.g., ZWJ) as zero width.
        if unicodedata.category(ch) == "Cf":
            continue
        ea = unicodedata.east_asian_width(ch)
        w += 2 if ea in ("W", "F") else 1
    return w


class Renderer:
    """
    Draws the prompt + lines and positions the cursor at (line_idx, col_in_line),
    accounting for terminal auto-wrap. Tracks physical row offsets, not just logical
    line indices.
    """

    def __init__(self, prompt: str, continuation: str):
        self._prompt = prompt
        self._cont = continuation
        self._prompt_cols = display_width(prompt)
        self._cont_cols = display_width(continuation)
        # physical row index of the cursor within the block from the top (0-based)
        self._last_cursor_row = 0

    def _term_cols(self) -> int:
        try:
            return os.get_terminal_size(sys.stdout.fileno()).columns
        except OSError:
            # fallbacks for weird stdio setups
            for fd in (1, 2, 0):
                try:
                    return os.get_terminal_size(fd).columns
                except OSError:
                    pass
            return int(os.environ.get("COLUMNS", 80)) or 80

    def _prefix_cols_for(self, idx: int) -> int:
        return self._prompt_cols if idx == 0 else self._cont_cols

    def _rows_for_line(self, idx: int, text: str, cols: int) -> int:
        w = self._prefix_cols_for(idx) + display_width(text)
        # at least one row
        return 1 if w <= 0 else (w + cols - 1) // cols

    def render(self, lines: list[str], cur_line: int, cur_col: int):
        cols = max(1, self._term_cols())

        # 1) Go to the top of our previous block and clear downwards.
        sys.stdout.write("\r")
        if self._last_cursor_row > 0:
            sys.stdout.write(f"{CSI}{self._last_cursor_row}A")
        sys.stdout.write(f"{CSI}J")  # clear to end of screen

        # 2) Print the block, padding lines that end exactly at the right edge
        #    to avoid the 'xn' eat-newline glitch.
        total = len(lines)
        for i, ln in enumerate(lines):
            prefix = self._prompt if i == 0 else self._cont
            sys.stdout.write(prefix)
            sys.stdout.write(ln)
            if i < total - 1:
                visible = self._prefix_cols_for(i) + display_width(ln)
                if visible % cols == 0:
                    # Leave wrap-pending state safely prior to newline.
                    sys.stdout.write(" \b")
                sys.stdout.write("\r\n")
        sys.stdout.flush()

        # 3) Compute physical geometry for the current state.
        #    total_rows: total physical rows consumed by the block
        #    rows_before_cursor: physical rows strictly before the cursor's row
        total_rows = 0
        rows_before_cursor = 0
        for i, ln in enumerate(lines):
            r = self._rows_for_line(i, ln, cols)
            if i < cur_line:
                rows_before_cursor += r
            total_rows += r

        # Position within the cursor's logical line: map to the *insertion* cell.
        prefix_cols = self._prefix_cols_for(cur_line)
        cells_used = prefix_cols + display_width(lines[cur_line][:cur_col])

        if cells_used <= 0:
            row_in_line = 0
            col_in_row = 1  # CHA 1 -> first column
        elif cells_used % cols == 0:
            # Exactly at right margin: caret sits on last column of this row.
            row_in_line = (cells_used - 1) // cols
            col_in_row = cols
        else:
            # Normal case: go to the next insertion cell.
            row_in_line = cells_used // cols
            col_in_row = (cells_used % cols) + 1

        cursor_row_from_top = rows_before_cursor + row_in_line

        # 4) Move cursor to the desired physical row/column.
        sys.stdout.write("\r")
        if total_rows > 1:
            sys.stdout.write(f"{CSI}{total_rows - 1}A")  # up to first printed row
        if cursor_row_from_top > 0:
            sys.stdout.write(f"{CSI}{cursor_row_from_top}B")  # down into the target row
        sys.stdout.write(f"{CSI}{col_in_row}G")  # CHA: absolute column within the row
        sys.stdout.flush()

        # 5) Remember where we left the cursor in physical rows for next render.
        self._last_cursor_row = cursor_row_from_top


def _read_key(timeout: float = 0.05) -> str:
    """
    Read one key from stdin in raw mode. Returns:
      '\r' (Enter), '\n' (Ctrl+J), '\x04' (Ctrl+D), '\x03' (Ctrl+C),
      'UP','DOWN','LEFT','RIGHT', 'ALT_ENTER', 'ESC', or a literal character.
    """
    fd = sys.stdin.fileno()

    # Blocking read of the first byte
    b = os.read(fd, 1)
    if not b:
        return ""

    c = b[0]

    # Simple controls first
    if c == 3:  # Ctrl+C
        return "\x03"
    if c == 4:  # Ctrl+D (EOF)
        return "\x04"
    if c == 13:  # Enter
        return "\r"
    if c == 10:  # Ctrl+J (LF)
        return "\n"
    if c == 127:  # Backspace (DEL)
        return "\x7f"
    if c == 8:  # Backspace (BS)
        return "\x08"

    # Non-ESC printable: handle ASCII and UTF-8 starts
    if c != 0x1B:
        if c < 0x80:
            return chr(c)
        # Determine UTF-8 sequence length from leading byte
        if 0xC0 <= c <= 0xDF:
            need = 1
        elif 0xE0 <= c <= 0xEF:
            need = 2
        elif 0xF0 <= c <= 0xF7:
            need = 3
        else:
            return ""  # invalid start, drop it
        buf = bytearray([c])
        while need > 0:
            r, _, _ = select.select([fd], [], [], timeout)
            if not r:
                break
            chunk = os.read(fd, need)
            if not chunk:
                break
            buf.extend(chunk)
            need -= len(chunk)
        try:
            return buf.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return buf.decode("utf-8", errors="ignore") or ""

    # ESC: gather the rest of the sequence quickly
    r, _, _ = select.select([fd], [], [], timeout)
    if not r:
        return "ESC"

    tail = bytearray()
    while True:
        chunk = os.read(fd, 1)
        if not chunk:
            break
        tail += chunk
        last = tail[-1]
        # Stop at alpha or tilde (end of a CSI/SS3 sequence)
        if (65 <= last <= 90) or (97 <= last <= 122) or last == 126:
            break
        r, _, _ = select.select([fd], [], [], 0.01)
        if not r:
            break

    # Alt/Option+Enter: ESC then CR/LF
    if tail[:1] in (b"\r", b"\n"):
        return "ALT_ENTER"

    # Arrow keys: CSI (ESC [ ...) or SS3 (ESC O ...)
    if tail.startswith(b"[") and tail:
        final = chr(tail[-1])
        return {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}.get(final, "")
    if tail.startswith(b"O") and tail:
        final = chr(tail[-1])
        return {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}.get(final, "")

    # Unknown ESC-prefixed sequence: ignore to keep behavior unchanged
    return ""


def set_cursor(shape="bar", blink=True):
    """
    shape: 'block' | 'underline' | 'bar'
    blink: True for blinking, False for steady
    """
    mapping = {
        ("block", True): 1,
        ("block", False): 2,
        ("underline", True): 3,
        ("underline", False): 4,
        ("bar", True): 5,
        ("bar", False): 6,
    }
    code = mapping.get((shape, blink), 0)  # 0 = terminal default
    sys.stdout.write(f"\x1b[{code} q")
    sys.stdout.flush()


def stdblock(
    prompt: str = " > ",
    continuation: str | None = None,
    cursor_shape: str = "bar",
    cursor_blink: bool = True,
) -> str:
    """
    Read a multiline block. Return the text, or None on EOF (Ctrl+D).
    `prompt` sets the primary prompt (default " > ").
    `continuation` sets the continuation prompt (default: whitespace of len(prompt)).
    """
    # Align the default continuation to the *visible* prompt width.
    prompt_cols = display_width(prompt)
    cont = (" " * prompt_cols) if continuation is None else continuation

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    renderer = Renderer(prompt, cont)

    # Keep a list of logical lines and a cursor (line, col) in *character* indices.
    lines: list[str] = [""]
    line, col = 0, 0
    # goal_col stores the desired *visual* column (in cells) when moving vertically.
    goal_col: int | None = None

    def insert_text(s: str):
        nonlocal col
        ln = lines[line]
        lines[line] = ln[:col] + s + ln[col:]
        col += len(s)

    def newline():
        """Split line at cursor; move to next line col=0 (start of content)."""
        nonlocal line, col
        ln = lines[line]
        left, right = ln[:col], ln[col:]
        lines[line] = left
        lines.insert(line + 1, right)
        line += 1
        col = 0

    def backspace():
        nonlocal line, col
        if col > 0:
            ln = lines[line]
            lines[line] = ln[: col - 1] + ln[col:]
            col -= 1
            return
        if line > 0:
            # Join with previous line
            prev_len = len(lines[line - 1])
            lines[line - 1] += lines[line]
            del lines[line]
            line -= 1
            col = prev_len

    def move_left():
        nonlocal line, col, goal_col
        if col > 0:
            col -= 1
        elif line > 0:
            line -= 1
            col = len(lines[line])
        goal_col = None  # reset preserved column on horizontal motion

    def move_right():
        nonlocal line, col, goal_col
        if col < len(lines[line]):
            col += 1
        elif line < len(lines) - 1:
            line += 1
            col = 0
        goal_col = None  # reset preserved column on horizontal motion

    def _index_for_viscol(text: str, goal_cells: int) -> int:
        """Return the largest index i such that display_width(text[:i]) <= goal_cells."""
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if display_width(text[:mid]) <= goal_cells:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def move_vert(delta: int):
        nonlocal line, col, goal_col
        cur = line
        tgt = max(0, min(len(lines) - 1, cur + delta))
        # Preserve *visual* column when moving vertically.
        if goal_col is None:
            goal_col = display_width(lines[line][:col])
        col = _index_for_viscol(lines[tgt], goal_col)
        line = tgt

    try:
        set_cursor(cursor_shape, cursor_blink)
        tty.setraw(fd)
        renderer.render(lines, line, col)

        while True:
            key = _read_key()

            # Exit / interrupt
            if key == "\x04":  # Ctrl+D
                if len(lines) == 1 and not lines[0]:
                    sys.stdout.flush()
                    raise EOFError
                else:
                    continue
            if key == "\x03":  # Ctrl+C
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                raise KeyboardInterrupt

            # Navigation
            if key == "LEFT":
                move_left()
                renderer.render(lines, line, col)
                continue
            if key == "RIGHT":
                move_right()
                renderer.render(lines, line, col)
                continue
            if key == "UP":
                move_vert(-1)
                renderer.render(lines, line, col)
                continue
            if key == "DOWN":
                move_vert(+1)
                renderer.render(lines, line, col)
                continue

            # Newlines
            if key == "ALT_ENTER" or key == "\n":  # Alt/Option+Enter or Ctrl+J
                newline()
                renderer.render(lines, line, col)
                continue
            if key == "\r":  # Enter -> submit
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return "\n".join(lines)

            # Backspace (DEL 0x7f or BS 0x08)
            if key in ("\x7f", "\x08"):
                backspace()
                renderer.render(lines, line, col)
                continue

            # Ignore bare ESC/unknown sequences
            if key in ("ESC", ""):
                continue

            # Printable char
            insert_text(key)
            # Horizontal motion resets vertical goal column
            goal_col = None
            renderer.render(lines, line, col)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
