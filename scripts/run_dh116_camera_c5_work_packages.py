#!/usr/bin/env python3
"""Use the frozen scheduler with C5's versioned normal/fatal GDB capture."""
from pathlib import Path
import subprocess
import sys

try:
    from scripts import run_dh116_work_packages as scheduler
    from scripts.dh116_camera_c5_debugger import GDB_COMMANDS
except ModuleNotFoundError:
    import run_dh116_work_packages as scheduler
    from dh116_camera_c5_debugger import GDB_COMMANDS


def main():
    entry = Path(__file__).resolve()
    original_entry, original_commands = scheduler.__file__, scheduler.GDB_COMMANDS
    if entry.parent != Path(original_entry).resolve().parent:
        raise RuntimeError("C5 entry and frozen scheduler must share the same scripts directory")
    # The frozen scheduler derives BOTH its worker executable and repository root
    # from __file__. Re-enter this sibling wrapper in the child so C5 capture is
    # selected again there, while keeping the repository root unchanged.
    scheduler.__file__, scheduler.GDB_COMMANDS = str(entry), GDB_COMMANDS
    try:
        return scheduler.main()
    finally:
        scheduler.__file__, scheduler.GDB_COMMANDS = original_entry, original_commands


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"DH116 C5 scheduler: {error}", file=sys.stderr)
        raise SystemExit(1)
