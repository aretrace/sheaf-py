import asyncio
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


def load_python_module(path: Path, name: str) -> ModuleType:
    """Load a Python module from file."""
    if not path.exists():
        raise FileNotFoundError(f"Module not found: {path}")

    if path.suffix != ".py":
        raise ValueError(f"Not a Python file: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def get_module_functions(module: ModuleType) -> list[Callable[..., Any]]:
    """Extract all public functions from a module."""
    functions = []

    for name, obj in vars(module).items():
        if not inspect.isfunction(obj):
            continue
        if obj.__module__ != module.__name__:
            continue
        if "." in obj.__qualname__:
            continue
        if name.startswith("_"):
            continue

        functions.append(obj)

    return functions


def find_situate_class(module: ModuleType) -> type:
    """Find the class marked with @situate decorator."""
    situate_classes = []

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if not getattr(obj, "__situate__", False):
            continue
        situate_classes.append(obj)

    if not situate_classes:
        raise ValueError("No class marked with @situate found")

    if len(situate_classes) > 1:
        names = [cls.__name__ for cls in situate_classes]
        raise ValueError(f"Multiple @situate classes found: {', '.join(names)}")

    return situate_classes[0]


def load_behavior(directory: Path) -> tuple[type, list[Callable], str]:
    """Load all behavior components from a directory."""
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    # Add to path for imports
    sys_path = str(directory.resolve())
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

    # program.py
    program_path = directory / "program.py"
    if not program_path.exists():
        raise FileNotFoundError(f"Missing: {program_path}")

    program = load_python_module(program_path, f"_sheaf_program_{directory.name}")
    agent_class = find_situate_class(program)

    # tools.py
    tools_path = directory / "tools.py"
    if not tools_path.exists():
        raise FileNotFoundError(f"Missing: {tools_path}")

    tools = load_python_module(tools_path, f"_sheaf_tools_{directory.name}")
    callable_tools = get_module_functions(tools)

    if not callable_tools:
        raise ValueError(f"No public functions found in {tools_path}")

    # prompts.py
    prompts_path = directory / "prompts.py"
    if not prompts_path.exists():
        raise FileNotFoundError(f"Missing: {prompts_path}")

    prompts = load_python_module(prompts_path, f"_sheaf_prompts_{directory.name}")

    if not hasattr(prompts, "SYSTEM_MESSAGE"):
        raise ValueError(f"SYSTEM_MESSAGE not found in {prompts_path}")

    system_message = prompts.SYSTEM_MESSAGE

    if not isinstance(system_message, str):
        raise ValueError(
            f"SYSTEM_MESSAGE must be str, got {type(system_message).__name__}"
        )

    return agent_class, callable_tools, system_message


def load_api_key(env_file: Path) -> str:
    """Load API key from environment or fail with helpful message."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    if not env_file.exists():
        raise ValueError(f"OPENROUTER_API_KEY not found (create {env_file})")

    raise ValueError(f"OPENROUTER_API_KEY not found in {env_file}")


async def run_agent(behavior_dir: Path) -> int:
    """Run an agent from a behavior directory."""
    try:
        api_key = load_api_key(behavior_dir / ".env")

        agent_class, callable_tools, system_message = load_behavior(behavior_dir)

        agent = agent_class(
            model=os.environ.get("SHEAF_MODEL", "openai/gpt-4.1-mini"),
            base_url=os.environ.get("SHEAF_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=api_key,
            init_messages=[{"role": "system", "content": system_message}],
            callable_tools=callable_tools,
        )

        await agent.turn()
        return 0

    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def main() -> int:
    behavior_dir_str = os.environ.get("SHEAF_BEHAVIOR_DIR")
    if not behavior_dir_str:
        print("Error: SHEAF_BEHAVIOR_DIR environment variable not set")
        return 1

    behavior_dir = Path(behavior_dir_str)
    if not behavior_dir.exists():
        print(f"Error: Behavior directory not found: {behavior_dir}")
        return 1

    return asyncio.run(run_agent(behavior_dir))


if __name__ == "__main__":
    sys.exit(main())
