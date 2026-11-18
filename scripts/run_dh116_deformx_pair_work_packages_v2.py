#!/usr/bin/env python3
"""V2 pair scheduler transport, including worker reentry and raw validation."""
from pathlib import Path

from scripts import run_dh116_deformx_pair_work_packages as original
from scripts.verify_dh116_deformx_pair_capture_v2 import ENTRY, WORKER_ENV, validate_artifacts

WRAPPER = "scripts/run_dh116_deformx_pair_work_packages_v2.py"


def main():
    bindings = {
        "__file__": str(Path(__file__).resolve()),
        "ENTRY": ENTRY,
        "WRAPPER": WRAPPER,
        "WORKER_ENV": WORKER_ENV,
        "validate_artifacts": validate_artifacts,
    }
    saved = {name: getattr(original, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(original, name, value)
        return original.main()
    finally:
        for name, value in saved.items():
            setattr(original, name, value)


if __name__ == "__main__":
    raise SystemExit(main())
