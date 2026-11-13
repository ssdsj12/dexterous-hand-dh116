#!/usr/bin/env python3
"""Run the CUDA contact-pair recipe and independently reopen its raw packet."""
import json
import os
from pathlib import Path

from scripts import run_dh116_work_packages as scheduler
from scripts.verify_dh116_deformx_pair_capture import WORKER_ENV, validate_artifacts

ENTRY = "scripts/verify_dh116_deformx_pair_capture.py"
WRAPPER = "scripts/run_dh116_deformx_pair_work_packages.py"


class PairRepository(scheduler.Repository):
    def start_native(self, name, approval):
        recipe = self.native_recipe(self.registry()["packages"][name])["spec"]
        if recipe["entrypoint"] != ENTRY or WRAPPER not in recipe["source_files"]:
            raise ValueError("expected the frozen contact-pair recipe")
        return super().start_native(name, approval)

    def _native_child(self, path, spec, output, result, *, bound=None):
        if spec["entrypoint"] != ENTRY or WRAPPER not in spec["source_files"]:
            raise ValueError("expected the frozen contact-pair worker recipe")
        previous = os.environ.get(WORKER_ENV)
        os.environ[WORKER_ENV] = str(path / "intent.json")
        try:
            super()._native_child(path, spec, output, result, bound=bound)
        finally:
            if previous is None:
                os.environ.pop(WORKER_ENV, None)
            else:
                os.environ[WORKER_ENV] = previous
        if result.get("native_qualification") is not True:
            return
        try:
            validate_artifacts(output)
        except (OSError, ValueError, KeyError, TypeError) as error:
            result.update(native_qualification=False, blocker_kind="failed_native",
                          reason=f"raw contact-pair validation failed: {error}")


def main():
    original_entry, original_repository = scheduler.__file__, scheduler.Repository
    scheduler.__file__, scheduler.Repository = str(Path(__file__).resolve()), PairRepository
    try:
        return scheduler.main()
    finally:
        scheduler.__file__, scheduler.Repository = original_entry, original_repository


if __name__ == "__main__":
    raise SystemExit(main())
