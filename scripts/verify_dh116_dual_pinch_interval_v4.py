#!/usr/bin/env python3
"""Exact dual Pinch successor with the installed CuboidCfg collider binding."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from scripts import verify_dh116_dual_pinch_interval_v3 as previous
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

ENTRY = "scripts/verify_dh116_dual_pinch_interval_v4.py"
WRAPPER = "scripts/run_dh116_dual_pinch_work_packages_v4.py"
VERSION = previous.VERSION
WORKER_ENV = "DH116_DUAL_PINCH_INTERVAL_WORKER_V4"
FAILED_RECEIPT = Path(
    "/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/"
    "flex.f7-dual-pinch-zero-interval-v3-native/"
    "90d75220592f6551a389f5c9d8bbab55624492821115e41e05660a183307c943/"
    "attempt-n2pwu0f3/receipt.json"
)
FAILED_SHA = "b70888833bab2e1a1cbddff2fe9595b8953d31565d2ff7dc5c1c1a49da71c64c"
SHAPE_SOURCE = Path(
    "/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab/sim/spawners/shapes/shapes.py"
)
SHAPE_SHA = "053227b88000f63f707f6719d81b342bfa59c04ba78126e4e8b6f3760c7df4ca"
OUTPUTS = {
    "result.json": "f7023c5b00b8cace6304b3617b48850ce0f123a1a82cbea47e14041d934f5b37",
    "physical.jsonl": "36cff198922a48525cedfcbf5bd68c3fff7d2cdbbe44eec2ca1ba9c948bb0168",
    "app_close_returned.json": "37b1ba15219bb2264687d2883ed6129cc5cd54fe64d0d7afbb1e5ca7d9308ae4",
}
FAILED_ARTIFACTS = {
    "outer_exit.json": "0a3fb110331f7078a864022df81d3db3ca9a5d378a2a1b88a82b0c2df637f025",
    "native.log": "4d40154f80222e78bf972b88c665d79ad7caea4f3e0afd87a56ca57925442fe7",
}
FILES = (
    ENTRY,
    WRAPPER,
    "configs/dh116_execution/deformx_dual_pinch_zero_interval_v3.json",
    "src/go2_pvcnn/control/dh116/tabletop_bimanual_table_contacts_v2.py",
    "tests/test_dh116_dual_pinch_worker_v4.py",
)
previous_preflight = previous.preflight
execute_previous = previous.execute
read = previous.prior.read
sha = previous.prior.prior.sha


def preflight(project):
    if sha(FAILED_RECEIPT) != FAILED_SHA:
        raise ValueError("original v3 failure receipt differs")
    receipt = read(FAILED_RECEIPT)
    output = Path(receipt["output"])
    evidence = {FAILED_RECEIPT: FAILED_SHA, SHAPE_SOURCE: SHAPE_SHA}
    evidence.update({FAILED_RECEIPT.with_name(name): digest for name, digest in FAILED_ARTIFACTS.items()})
    evidence.update({output / name: digest for name, digest in OUTPUTS.items()})
    contract = previous_preflight(project)
    sources = deepcopy(contract["sources"])
    for name in FILES:
        sources["local_sources"][name] = sha(project / name)
    for path, digest in evidence.items():
        if sha(path) != digest:
            raise ValueError(f"original v3 evidence or installed shape source differs: {path}")
        sources["external_sources"][str(path)] = digest
    result = read(output / "result.json")
    outer = read(FAILED_RECEIPT.with_name("outer_exit.json"))
    if (receipt["native_qualification"] is not False
            or outer["outer_exit_code"] != 1 or outer["debugger_inferior_exit_code"] != 1
            or outer["debugger_inferior_signal"] is not None or result["physics_steps"] != 2
            or "ValueError: actual original table collider required" not in result["failure"]):
        raise ValueError("exact original v3 table-collider failure required")
    code = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(contract, sources=sources, code_config_sha256=code, v3_failure_receipt_sha256=FAILED_SHA,
                correction_scope="actual_static_cuboid_collider_descendant_same_table")


def execute(args, project, contract):
    from go2_pvcnn.control.dh116 import tabletop_bimanual_table_contacts as original
    from go2_pvcnn.control.dh116 import tabletop_bimanual_table_contacts_v2 as resolved

    with bindings((original, "open_bimanual_table_contacts", resolved.open_bimanual_table_contacts)):
        return execute_previous(args, project, contract)


def main():
    with bindings(
        (previous, "ENTRY", ENTRY), (previous, "WRAPPER", WRAPPER),
        (previous, "WORKER_ENV", WORKER_ENV), (previous, "preflight", preflight),
        (previous, "execute", execute),
    ):
        return previous.main()


if __name__ == "__main__":
    main()
