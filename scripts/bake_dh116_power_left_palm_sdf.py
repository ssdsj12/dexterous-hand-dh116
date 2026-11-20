#!/usr/bin/env python3
"""Bake the proved original left Power palm in an isolated DeformX batch."""
import argparse
import json
from pathlib import Path

from scripts import audit_dh116_power_palm_surface as surface
from scripts import bake_dh116_deformx_pad_sdfs as pad_bake

ROOT = Path(__file__).resolve().parents[1]
BACKEND_LOCK = ROOT / "docs/contracts/dh116_deformx_pad_sdf_evidence_lock.json"
BACKEND_LOCK_SHA = "acca227b59b1d2e43f02db8d9b3218e96f1ae64412ee28337b95e2c9e2c7bc1f"


def input_left_palm(capture):
    metrics = surface.evaluate(capture)["palms"]["left"]
    if metrics["matches_convex_boundary"] is not True or metrics["exact_convex_facets"] is not True:
        raise ValueError("exact original palm triangle surface required")
    path = capture / "gpu" / "native" / "left_base_link.json"
    digest = metrics["gpu_row_sha256"]
    row = pad_bake.checked_json(path, digest)
    if (row["side"], row["link"]) != ("left", "base_link"):
        raise ValueError("left Power base_link identity required")
    hull, basis_delta = surface._pad_hull(row)
    return {"left.base_link": (row, hull, basis_delta, digest)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--ropesim-package-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.ropesim_package_root.resolve(strict=True)
    lock = pad_bake.checked_json(BACKEND_LOCK, BACKEND_LOCK_SHA)
    packets = input_left_palm(args.capture.resolve(strict=True))
    for name, digest in lock["backend_files"].items():
        pad_bake.checked_bytes(root / "ropesim" / name, digest)
    if args.output is None:
        print(json.dumps(dict(verified_meshes=list(packets), gpu_started=False)))
        return 0
    args.output.mkdir()
    result = pad_bake.bake_register(packets, lock, root, args.output)
    record = dict(result, scope="isolated_original_left_power_palm_gpu_sdf_and_registration")
    record["palms"] = record.pop("pads")
    record.update(parity_sha256=surface.PARITY_SHA, backend_lock_sha256=BACKEND_LOCK_SHA,
                  native_power_task_registration=False, grasp_verified=False)
    pad_bake.write_json(args.output / "result.json", record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
