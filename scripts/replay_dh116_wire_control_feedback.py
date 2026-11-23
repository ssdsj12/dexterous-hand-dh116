#!/usr/bin/env python3
"""Read one complete force feedback window from an existing native receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned, _digest
from go2_pvcnn.control.dh116.tabletop_training_reward import WINDOW_SUBSTEPS
from go2_pvcnn.control.dh116.tabletop_wire_grasp_feedback import pack_wire_control_feedback


def _checked_json(path, expected):
    data = path.read_bytes()
    if not isinstance(expected, str) or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"SHA-256 differs from the receipt for {path.name}")
    return json.loads(data)


def replay_control_window(receipt, receipt_sha256, *, grasp, sdk_end, grasp_material_ids):
    if (grasp not in ("pinch", "tripod", "power")
            or type(sdk_end) is not int or sdk_end < WINDOW_SUBSTEPS
            or not isinstance(receipt_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", receipt_sha256)):
        raise ValueError("grasp, SDK endpoint and exact receipt SHA-256 required")
    receipt = Path(receipt)
    sealed = _checked_json(receipt, receipt_sha256)
    if sealed.get("exit_code") != 0 or sealed.get("debugger_inferior_exit_code") != 0:
        raise ValueError("verified natural native exit required for this replay")
    digests = {}
    exchanges = []
    for tick in range(sdk_end-WINDOW_SUBSTEPS+1, sdk_end+1):
        name = f"{grasp}_control_{tick}.json"
        expected = sealed["artifact_sha256"].get(name)
        row = decode_owned(_checked_json(receipt.parent/"native"/name, expected))
        if row["physics_step"] != tick:
            raise ValueError("sealed exchange does not belong to its named SDK step")
        digests[name] = expected
        exchanges.append(row)
    last = exchanges[-1]
    if type(last["policy_step"]) is not int:
        raise ValueError("sealed exchange must carry its original control policy index")
    # Offline: infer the next index from the sealed endpoint. The live caller
    # supplies its independently read pre-action policy index instead.
    clock = dict(last["rod"], policy_step=last["policy_step"]+1)
    observation = pack_wire_control_feedback(exchanges, clock, grasp=grasp,
                                             grasp_material_ids=grasp_material_ids)
    return dict(version="dh116-wire-control-feedback-replay-v1", grasp=grasp, sdk_end=sdk_end,
                outer_sha256=receipt_sha256, exchange_sha256=digests, observation=observation,
                clock_basis="sealed_completed_control_endpoint",
                observation_sha256=_digest(observation), teacher_admission=False, new_native_steps=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--grasp", choices=("pinch", "tripod", "power"), required=True)
    parser.add_argument("--sdk-end", type=int, required=True)
    parser.add_argument("--left-material-id", type=int, required=True)
    parser.add_argument("--right-material-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay_control_window(args.receipt, args.receipt_sha256, grasp=args.grasp,
        sdk_end=args.sdk_end, grasp_material_ids={"left": args.left_material_id, "right": args.right_material_id})
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
