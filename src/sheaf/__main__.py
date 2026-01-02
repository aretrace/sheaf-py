import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

# Config
BEHAVIORS_ROOT = Path.home() / ".config" / "sheaf" / "behaviors"
TEMPLATE_DIR = Path(__file__).resolve().parent / "template"
REQUIRED_FILES = ["program.py", "tools.py", "prompts.py"]


def is_uv_installed() -> bool:
    """Check if uv is installed."""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def require_uv() -> None:
    """Ensure uv is installed or exit."""
    if is_uv_installed():
        return

    print("Error: 'uv' is not installed. Please install it first:")
    print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
    sys.exit(1)


def has_required_files(path: Path) -> bool:
    """Check if all required files exist."""
    return all((path / f).exists() for f in REQUIRED_FILES)


def copy_template_files(dest: Path, *, overwrite: bool = False) -> None:
    """Copy template files to destination, optionally overwriting existing files."""
    if not TEMPLATE_DIR.exists():
        raise FileNotFoundError(f"Template directory not found: {TEMPLATE_DIR}")

    for src_file in TEMPLATE_DIR.rglob("*"):
        if src_file.name in ["__pycache__", ".git", ".DS_Store"]:
            continue
        if "__pycache__" in str(src_file):
            continue

        rel_path = src_file.relative_to(TEMPLATE_DIR)
        dest_file = dest / rel_path

        if src_file.is_dir():
            dest_file.mkdir(parents=True, exist_ok=True)
            continue

        if dest_file.exists() and not overwrite:
            print(f"  [skip] {rel_path}")
            continue

        dest_file.parent.mkdir(parents=True, exist_ok=True)
        if dest_file.exists() and dest_file.is_dir():
            print(f"  [skip] {rel_path} (destination is a directory)")
            continue
        shutil.copy2(src_file, dest_file)
        print(f"  [copy] {rel_path}")


def update_env_file(env_path: Path, api_key: str) -> None:
    """Update or create .env file with OPENROUTER_API_KEY."""
    lines = []
    key_found = False

    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    lines.append(f"OPENROUTER_API_KEY={api_key}\n")
                    key_found = True
                else:
                    lines.append(line)

    if not key_found:
        lines.append(f"OPENROUTER_API_KEY={api_key}\n")

    with open(env_path, "w") as f:
        f.writelines(lines)


# Commands


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize a new behavior by copying template files."""
    require_uv()

    name = args.name or "default"
    behavior_dir = BEHAVIORS_ROOT / name
    behavior_dir.parent.mkdir(parents=True, exist_ok=True)

    if behavior_dir.exists() and not args.force:
        print(f"Error: '{name}' already exists at {behavior_dir}")
        print("Use --force to overwrite")
        return 1

    print(f"Initializing '{name}' at {behavior_dir}")

    print("\nCopying template files...")
    copy_template_files(behavior_dir, overwrite=args.force)

    if args.key:
        env_path = behavior_dir / ".env"
        update_env_file(env_path, args.key)
        print(f"\nAPI key added to {env_path}")

    print("\nSyncing dependencies with uv...")
    try:
        subprocess.run(
            ["uv", "sync"], cwd=behavior_dir, capture_output=True, text=True, check=True
        )
        print("Dependencies synced successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Warning: uv sync failed: {e.stderr}")
        print("You may need to run 'uv sync' manually in the behavior directory.")

    print("\nDone! Next steps:")
    if not args.key:
        print(f" 1. Add OPENROUTER_API_KEY to .env in {behavior_dir}")
    print(f" {'2' if not args.key else '1'}. sheaf run {name}")

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run a behavior."""
    require_uv()

    if args.dir:
        behavior_dir = Path(args.dir).resolve()
    else:
        name = args.name or "default"
        behavior_dir = BEHAVIORS_ROOT / name

    if not behavior_dir.exists():
        print(f"Error: Not found: {behavior_dir}")
        if not args.dir:
            print(f"Initialize with: sheaf init {args.name or 'default'}")
        return 1

    if not has_required_files(behavior_dir):
        missing = [f for f in REQUIRED_FILES if not (behavior_dir / f).exists()]
        print(f"Error: Missing files in {behavior_dir}: {', '.join(missing)}")
        return 1

    if not (behavior_dir / "pyproject.toml").exists():
        print(f"Error: Missing pyproject.toml in {behavior_dir}")
        return 1

    # Delegate to uv run and allow tools to work with files in the user's current directory
    runner_path = behavior_dir / "_lib" / "runner.py"
    cmd = [
        "uv",
        "run",
        "--quiet",
        "--isolated",
        "--project",
        str(behavior_dir),
        "--env-file",
        str(behavior_dir / ".env"),
        "python",
        str(runner_path),
    ]

    try:
        # run in the current directory (where sheaf was invoked), not behavior_dir
        env = os.environ.copy()
        env["SHEAF_BEHAVIOR_DIR"] = str(behavior_dir)

        # override API key if provided via --key
        if args.key:
            env["OPENROUTER_API_KEY"] = args.key

        result = subprocess.run(cmd, env=env)
        return result.returncode
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """List all behaviors (simple list)."""
    if not BEHAVIORS_ROOT.exists():
        return 0

    behaviors = [d for d in BEHAVIORS_ROOT.iterdir() if d.is_dir()]
    for behavior in sorted(behaviors):
        print(behavior.name)
    return 0


def main() -> NoReturn:
    parser = argparse.ArgumentParser(prog="sheaf")
    subparsers = parser.add_subparsers(dest="command")

    # init cmd
    init_parser = subparsers.add_parser("init", help="Initialize a new behavior")
    init_parser.add_argument("name", nargs="?", default="default")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing")
    init_parser.add_argument("--key", help="OpenRouter API key to add to .env file")

    # run cmd
    run_parser = subparsers.add_parser("run", help="Run a behavior")
    run_parser.add_argument("name", nargs="?", default="default")
    run_parser.add_argument("--dir", help="Use custom directory")
    run_parser.add_argument("--key", help="OpenRouter API key (overrides .env file)")

    # list cmd
    subparsers.add_parser("list", help="List behaviors")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "init": cmd_init,
        "run": cmd_run,
        "list": cmd_list,
    }

    handler = handlers.get(args.command)
    if not handler:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()
