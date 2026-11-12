#!/usr/bin/env python3
"""Use the frozen scheduler with C5's versioned normal/fatal GDB capture."""
import subprocess
import sys

try:
    from scripts import run_dh116_work_packages as scheduler
    from scripts.dh116_camera_c5_debugger import GDB_COMMANDS
except ModuleNotFoundError:
    import run_dh116_work_packages as scheduler
    from dh116_camera_c5_debugger import GDB_COMMANDS


def main():
    scheduler.GDB_COMMANDS = GDB_COMMANDS
    return scheduler.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"DH116 C5 scheduler: {error}", file=sys.stderr)
        raise SystemExit(1)
