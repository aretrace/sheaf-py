# Sheaf

> ˈshēvz,
1. An agent toolkit.
2. A quantity of a grain, bound together.


## “Behaviors”
Sheaf runs a _behavior_: a cobordant composed of an agent input, outputs, prompts and tools. Behaviors live under `~/.config/sheaf/behaviors/<name>`.

## Overview
- **CLI** (`sheaf`) scaffolds, lists, and runs behaviors.
- **Behavior folder** holds three user files (`program.py`, `prompts.py`, `tools.py`), agnet internals (`/_lib`) plus some example support libs (`/utils`,).

## Get started
```bash
uv run sheaf --help
```

## Create your first behavior
```bash
uv run sheaf init planner --key sk-your-openrouter-key
uv run sheaf run planner
```
- `init` creates `~/.config/sheaf/behaviors/planner` with the starter files.
- `run` launches the agent.

## Swathe

```python
# program.py
from datetime import datetime
from _lib.core import BaseLayer, StepEvent, situate


class LoggingLayer(BaseLayer):
    """Adapter that logs errors."""

    async def handler(self, event: StepEvent):
        if self.error:
            print(f"[ERR] {self.error.message}")


class TimestampLayer(LoggingLayer):
    """Adapter that timestamps messages as they flow up."""

    async def percept(self, afference: Afference):
        async for msg in afference:
            yield f"[{datetime.now():%H:%M}] {msg}"


@situate
class Planner(TimestampLayer):
    async def percept(self, afference: Afference):
        while text := input("User > "):
            if text.lower() == "/q":
                print("Bye Bye Bye")
                break
            yield text

    async def handler(self, event: StepEvent):
        if event.type == "agent.finish.stopped" and self.assistant_reply:
            print(f"Agent > {self.assistant_reply}")
```
`percept()` flows from the situated leaf up through base layers, while `handler()` flows from base layers down to the situated leaf.

```python
# prompts.py
SYSTEM_MESSAGE = """
You help users plan their day. Keep answers short. Use tools when useful.
"""
```

```python
# tools.py
from pathlib import Path
from typing import Annotated

def save_plan(
    title: Annotated[str, "Short name for this plan"],
    body: Annotated[str, "What to remember"],
) -> str:
    folder = Path("plans")
    folder.mkdir(exist_ok=True)
    file = folder / f"{title.replace(' ', '_').lower()}.txt"
    file.write_text(body, encoding="utf-8")
    return f"Saved to {file}"
```

## Config
- `OPENROUTER_API_KEY` — required API key (set through `--key`, `.env`, or your shell).
- `SHEAF_MODEL` — optional model override (`openai/gpt-4.1-mini` default).
- `SHEAF_BASE_URL` — optional endpoint override (`https://openrouter.ai/api/v1` default).
