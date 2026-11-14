#!/usr/bin/env python3
"""Use the frozen scheduler with C5's versioned normal/fatal GDB capture."""
from functools import partial
import json
from pathlib import Path
import subprocess
import sys

try:
    from scripts import run_dh116_work_packages as scheduler
    from scripts.dh116_camera_c5_debugger import render_gdb_commands
except ModuleNotFoundError:
    import run_dh116_work_packages as scheduler
    from dh116_camera_c5_debugger import render_gdb_commands


def write_c5_debugger(base_wrapper, attempt, result):
    argv = json.loads((attempt / "launch_argv.json").read_text())["argv"]
    if argv.count("--output") != 1 or argv[-1] == "--output":
        raise ValueError("C5 launch record must bind exactly one output directory")
    output = Path(argv[argv.index("--output") + 1])
    if not output.is_absolute() or output.resolve() != output:
        raise ValueError("C5 capture requires a canonical absolute output directory")
    original_commands = scheduler.GDB_COMMANDS
    scheduler.GDB_COMMANDS = render_gdb_commands(output)
    try:
        return base_wrapper(attempt, result)
    finally:
        scheduler.GDB_COMMANDS = original_commands


def main():
    entry = Path(__file__).resolve()
    original_entry = scheduler.__file__
    original_wrapper = scheduler.native_debugger_wrapper
    if entry.parent != Path(original_entry).resolve().parent:
        raise RuntimeError("C5 entry and frozen scheduler must share the same scripts directory")
    # The frozen scheduler derives BOTH its worker executable and repository root
    # from __file__. Re-enter this sibling wrapper in the child so C5 capture is
    # selected again there, while keeping the repository root unchanged.
    scheduler.__file__ = str(entry)
    scheduler.native_debugger_wrapper = partial(write_c5_debugger, original_wrapper)
    try:
        return scheduler.main()
    finally:
        scheduler.__file__ = original_entry
        scheduler.native_debugger_wrapper = original_wrapper


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"DH116 C5 scheduler: {error}", file=sys.stderr)
        raise SystemExit(1)
