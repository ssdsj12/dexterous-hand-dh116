#!/usr/bin/env python3
"""Exclusive CPU-only C01 preflight output; never launches Isaac or a policy."""
import argparse
import json
from pathlib import Path
import sys

from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import (
    candidate_config, static_preflight, validate_config, ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--style", choices=("pinch", "tripod", "power"), default="pinch")
    parser.add_argument("--profile", choices=("approach_wiring_smoke_v1", "flexible_train_r1_full"),
                        default="approach_wiring_smoke_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = args.output.absolute()
        if output.resolve() != output or output.is_relative_to(ROOT):
            raise ValueError("exclusive private output outside source checkout required")
        config = (json.loads(args.config.read_bytes()) if args.config else
                  candidate_config(style=args.style, profile=args.profile))
        validate_config(config)
        result = static_preflight(config)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0 if result["static_ready"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
