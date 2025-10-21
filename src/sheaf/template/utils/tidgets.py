import asyncio
import builtins
import itertools
import json
import sys
import sysconfig
from dataclasses import dataclass
from functools import partial

print = partial(builtins.print, flush=True)


def _rgb_to_esc(hex_str: str) -> str:
    h = hex_str.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"\x1b[38;2;{r};{g};{b}m"


ESC_RESET = "\x1b[0m"
ESC_USER_COLOR = _rgb_to_esc("#6A91C2")
ESC_AGENT_COLOR = _rgb_to_esc("#BB9347")
ESC_TOOL_COLOR = _rgb_to_esc("#7A6E3F")
ESC_SHELL_COLOR = _rgb_to_esc("#8C7A45")
ESC_THINKING_COLOR = _rgb_to_esc("#81a0a2")


@dataclass(slots=True)
class RoleLabel:
    User = f"⌈{ESC_USER_COLOR}You{ESC_RESET}⌋"
    Agent = f"⌈{ESC_AGENT_COLOR}Agent{ESC_RESET}⌋"
    Tool = f"⌈{ESC_TOOL_COLOR}Tool{ESC_RESET}⌋"
    Shell = ""
    Code = ""
    Thinking = f"{ESC_THINKING_COLOR}Contemplating…{ESC_RESET}"


display_is_free_threaded_banner = (  # noqa: E731
    lambda: print("\033[3mBarry Moler mode\033[0m \033[1;91mACTIVATED\033[0m")
    if (
        bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
        and (not sys._is_gil_enabled())
    )
    else None
)


async def loading_indicator(text: str) -> None:
    print("\x1b[?25l", end="")
    try:
        for character in itertools.cycle("🩷🧡💛💚💙💜"):
            print(f"\r\033[K⌈{character} {text}⌋", end="")
            await asyncio.sleep(0.13)
    finally:
        print("\r\033[K", end="")
        print("\x1b[?25h", end="")


format_tool_args = lambda args: "，".join(  # noqa: E731
    f"{k}፡ {v}" for k, v in json.loads(args).items()
)
