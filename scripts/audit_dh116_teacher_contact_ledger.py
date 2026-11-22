#!/usr/bin/env python3
"""Read-only same-clock contact/action audit of existing bounded Teacher physics JSONL.

Reuses the original complete17-slot contact reducer. It creates no App, advances
no physics, changes no controller and grants no training/grasp qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import (
    IDENTITY_FIELDS, TABLE_LOAD_LIMIT_N, measure_supported_distal_contact,
)


def _finite(value, shape, label):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError("finite " + label + " with shape " + str(shape) + " required")
    return array


def _joints(hand):
    if not isinstance(hand, dict) or any(k not in hand for k in ("q", "qd", "joint_names", "units_by_name")):
        return dict(valid=False, reason="missing_joint_source", q=None, qd=None)
    names = hand["joint_names"]
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError("distinct actual joint names required")
    q = _finite(hand["q"], (1, len(names)), "joint q")
    qd = _finite(hand["qd"], q.shape, "joint qd")
    return dict(valid=True, reason=None, joint_names=names, units_by_name=hand["units_by_name"],
                q=q[0].tolist(), qd=qd[0].tolist(), measured_source=hand.get("measured_source"))


def audit_file(path, *, expected_sha256, grasp):
    """Consume and hash exactly the audited bytes; reject identity gaps or damage."""
    if grasp not in ("pinch", "tripod"):
        raise ValueError("independent supported distal audit expects pinch or tripod")
    if not isinstance(expected_sha256, str) or not re.fullmatch("[0-9a-f]{64}", expected_sha256):
        raise ValueError("explicit raw source SHA256 required")
    ledger, previous, sequence = [], None, -1
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for raw in source:
            digest.update(raw)
            record = decode_owned(json.loads(raw))
            ordinal = record["raw_sequence"]
            if type(ordinal) is not int or ordinal <= sequence:
                raise ValueError("strictly increasing original raw sequence required")
            exchange = record["payload"]["exchange"]
            rod = exchange["rod"]
            identity = {key: rod[key] for key in IDENTITY_FIELDS}
            tick = identity["physics_step"]
            if exchange["physics_step"] != tick:
                raise ValueError("exchange and rod physics clocks differ")
            if previous is not None and (
                    any(identity[k] != previous[k] for k in IDENTITY_FIELDS if k not in ("physics_step", "rod_tick"))
                    or tick != previous["physics_step"] + 1 or identity["rod_tick"] != previous["rod_tick"] + 12):
                raise ValueError("consecutive same-episode SDK and solver ticks required")
            action = exchange["action_readback"]
            for phase, delta in (("pre", -1), ("post", 0)):
                clocks = action[phase]["identity"]
                if len(clocks) != 1 or any(
                        clocks[0][key] != (identity[key] + delta * (12 if key == "rod_tick" else 1)
                            if key in ("physics_step", "rod_tick") else identity[key])
                        for key in IDENTITY_FIELDS):
                    raise ValueError("action pre/post and rod identity or physics tick differ")
            contact = measure_supported_distal_contact(rod, grasp=grasp)
            nodes = _finite(rod["node_positions"], (1, 41, 3), "material positions")
            velocity = None if rod.get("velocities") is None else _finite(rod["velocities"], nodes.shape, "node velocities")
            command = _finite(action["request"]["action24"], (1, 24), "actual submitted action24")
            ledger.append(dict(raw_sequence=ordinal, identity=identity, contact=contact,
                median_z_m=float(np.median(nodes[0, :, 2])), min_z_m=float(nodes[0, :, 2].min()),
                node_positions_world_m=nodes[0].tolist(),
                node_velocities_world_m_s=None if velocity is None else velocity[0].tolist(),
                node_speed_max_m_s=None if velocity is None else float(np.linalg.norm(velocity[0], axis=1).max()),
                joint_state={side: _joints(action["post"].get("hands", {}).get(side)) for side in ("left", "right")},
                issued_action24=command[0].tolist(),
                action_contract=action["request"].get("action_contract"),
                action_stage="submitted_normalized_command_not_actor_latent_or_measured_tracking",
                guard_failure=exchange.get("guard_failure")))
            previous, sequence = identity, ordinal
    if digest.hexdigest() != expected_sha256 or not ledger:
        raise ValueError("raw source SHA differs or physics ledger is empty")
    contact = lambda row: row["contact"]
    result = dict(version="dh116-offline-supported-teacher-contact-ledger-v1",
        source_sha256=digest.hexdigest(), evidence_kind="offline_replay_of_existing_raw_not_new_native",
        grasp=grasp, rows=len(ledger), first_sdk_step=ledger[0]["identity"]["physics_step"],
        last_sdk_step=ledger[-1]["identity"]["physics_step"],
        bilateral_rows=sum(contact(row)["bilateral"] for row in ledger),
        support_free_rows=sum(contact(row)["support_free"] for row in ledger),
        table_contact_rows=sum(max(contact(row)["external_support_force_norm_n"]["table"]) > TABLE_LOAD_LIMIT_N for row in ledger),
        per_hand={side: dict(
            loaded_rows=sum(contact(row)["per_hand"][side]["loaded"] for row in ledger),
            max_total_normal_load_n=max(contact(row)["per_hand"][side]["total_normal_load_n"] for row in ledger))
            for side in ("left", "right")},
        support_max_force_norm_n={name: max(max(contact(row)["external_support_force_norm_n"][name]) for row in ledger)
                                 for name in ("table", "support_0", "support_1")},
        node_speed_max_m_s=None if any(row["node_speed_max_m_s"] is None for row in ledger)
            else max(row["node_speed_max_m_s"] for row in ledger),
        native_runs=0, ppo_updates=0, native_qualification=False,
        teacher_admission=False, grasp_qualified=False, ledger=ledger)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-jsonl", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--grasp", choices=("pinch", "tripod"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_file(args.physics_jsonl, expected_sha256=args.source_sha256, grasp=args.grasp)
    # Exclusive output prevents replacing an earlier analysis or raw artifact.
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps({key: value for key, value in result.items() if key != "ledger"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
