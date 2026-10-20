#!/usr/bin/env python3
"""Run reproducible, serial DH116 performance comparisons."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


VARIANTS = ("baseline", "visual", "snapshot", "buffered")
PHASES = ("settle", "approach", "close", "hold", "lift")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=Path("runs/perf_20261020"))
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    return args


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_command(args, output: Path) -> list[str]:
    root = _root()
    cfg = root / "configs" / "table_margin_raise1mm"
    pins = json.loads((cfg / "validation.json").read_text())["files"]
    command = [
        args.python, str(root / "scripts" / "verify_fixed_dual_panda_dh116.py"),
        "--asset-dir", str(root / "assets" / "fixed_dual_panda_dh116_v1"),
        "--active-pose", str(cfg / "prepose.json"), "--active-pose-sha256", pins["prepose.json"],
        "--approach-certificate", str(cfg / "approach.json"),
        "--approach-certificate-sha256", pins["approach.json"],
        "--point-references", str(cfg / "point_references.json"),
        "--contact-manifest", str(cfg / "contact" / "manifest.json"),
        "--contact-manifest-sha256", pins["contact/manifest.json"],
        "--backend-root", str(root / "vendor" / "DeformX"),
        "--grasp-sequence", "--coordinated-closure",
        "--closure-loads", "1.0005", ".8855", "1.012", "1.0695",
        "--contact-subsamples", "16", "--contact-tangential-damping", "1000",
        "--clearance-certificate-cache", "--contact-broad-phase", "--gravity-hold",
        "--arm-effort-limit", "60", "--hand-effort-limit", ".2",
        "--hand-stiffness", "10", "--hand-damping", ".2", "--external-force-iterations",
        "--motion-prediction-order", "1", "--physics-dt", ".0001",
        "--steps", str(args.steps), "--solver-position-iterations", "1",
        "--table-wire", "--profile", "--perf-flush-steps", "1000",
        "--ui-hz", "15", "--device", args.device, "--output", str(output),
    ]
    if args.variant in ("baseline", "snapshot", "buffered"):
        command.append("--headless")
    if args.variant == "buffered":
        command.extend(["--evidence-flush-policy", "buffered"])
    else:
        command.extend(["--evidence-flush-policy", "audit"])
    return command


def run(args) -> list[Path]:
    root = _root()
    output_root = (root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    if not output_root.is_relative_to(root / "runs"):
        raise ValueError("output-root must be under project runs")
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for repetition in range(1, args.repeat + 1):
        output = output_root / f"{args.variant}_{args.phase}_r{repetition:02d}"
        if output.exists():
            raise FileExistsError(output)
        command = build_command(args, output)
        env = os.environ.copy()
        env["DH116_PERF_VARIANT"] = args.variant
        env["DH116_PERF_PHASE"] = args.phase
        subprocess.run(command, cwd=root, env=env, check=True)
        output.mkdir(exist_ok=True)
        (output / "comparison_manifest.json").write_text(json.dumps({
            "variant": args.variant, "phase": args.phase, "repetition": repetition,
            "steps_requested": args.steps, "device": args.device,
            "command": command,
        }, indent=2) + "\n")
        outputs.append(output)
    return outputs


if __name__ == "__main__":
    run(parse_args())
