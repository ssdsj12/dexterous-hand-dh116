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
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError("numeric non-boolean " + label + " required")
    array = raw.astype(float)
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



def _joint_tracking(pre, post, request, side):
    """Compare same-articulation backend targets in their own joint coordinates.

    Passive target slots are not commands. World palm requests are not SDK-local
    wrist targets, so no subtraction between those coordinate systems is made.
    """
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, MIMIC
    from go2_pvcnn.control.dh116.tabletop_joint_target_readback import WRIST_LINEAR, WRIST_ANGULAR
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT

    absent=lambda reason: dict(valid=False,reason=reason,joints=None)
    if not isinstance(pre,dict) or not isinstance(post,dict):
        return absent("missing_joint_source")
    if any("backend_targets" not in hand for hand in (pre,post)):
        return absent("missing_backend_target_source")
    if any(hand.get("backend_source")!="physx_backend_get_dof_position_targets"
           or hand.get("measured_source")!="physx_backend_get_dof_positions/get_dof_velocities"
           for hand in (pre,post)):
        return absent("unqualified_joint_or_target_source")
    identity=("joint_names","active_indices","instance_object_id","view_object_id","prim_paths","row_indices")
    if any(key not in hand for hand in (pre,post) for key in identity):
        return absent("missing_tracking_identity")
    if any(pre[key]!=post[key] for key in identity):
        raise ValueError("tracking articulation identity/order changed across physical step")
    names=post["joint_names"]
    if (not isinstance(names,list) or not names or len(names)!=len(set(names))
            or any(type(post[key]) is not int or post[key]<=0 for key in ("instance_object_id","view_object_id"))
            or not isinstance(post["prim_paths"],list) or not post["prim_paths"]
            or any(not isinstance(path,str) or not path for path in post["prim_paths"])
            or not isinstance(post["row_indices"],list) or len(post["row_indices"])!=1
            or type(post["row_indices"][0]) is not int or post["row_indices"][0]<0):
        raise ValueError("tracking requires distinct names and original articulation/row identity")
    indices=post["active_indices"]
    if (not isinstance(indices,list) or len(indices)!=len(ACTIVE)
            or any(type(i) is not int or not 0<=i<len(names) for i in indices)
            or tuple(names[i] for i in indices)!=ACTIVE):
        raise ValueError("tracking active joint mapping differs from canonical action order")
    old,new=_joints(pre),_joints(post)
    if not old["valid"] or not new["valid"]:
        return absent("missing_joint_source")
    shape=(1,len(names))
    before=_finite(pre["backend_targets"],shape,"tracking backend targets")[0]
    after=_finite(post["backend_targets"],shape,"tracking backend targets")[0]
    requested=request.get("requested_finger_targets_rad")
    if requested is not None:
        if request.get("action_contract")!=ACTION_CONTRACT:
            raise ValueError("tracking requested target requires original 24D action contract")
        requested=_finite(requested,(1,12),"tracking requested targets")[0]
    passive={name for name,_,_,_ in MIMIC}
    result=[]
    for i,name in enumerate(names):
        active=name in ACTIVE
        wrist=name in (*WRIST_LINEAR,*WRIST_ANGULAR)
        role="active_finger" if active else "wrist" if wrist else "passive_finger" if name in passive else "unmapped"
        unit=post["units_by_name"].get(name)
        expected="m" if name in WRIST_LINEAR else "rad"
        if (active or wrist or name in passive) and (unit!=expected or pre["units_by_name"].get(name)!=expected):
            raise ValueError("tracking joint coordinate unit differs")
        controlled=active or wrist
        sdk=float(after[i]) if controlled else None
        controller=None
        reason=("wrist_request_requires_coordinate_and_projection_mapping" if wrist
                else "missing_requested_finger_targets" if active else "joint_not_commanded")
        action_index=(0 if side=="left" else 6)+ACTIVE.index(name) if active else None
        if active and requested is not None:
            controller=float(requested[action_index]);reason=None
        result.append(dict(joint=name,joint_role=role,position_unit=unit,action_index=action_index,
            position_pre=old["q"][i],position=new["q"][i],velocity=new["qd"][i],
            position_delta=new["q"][i]-old["q"][i],
            sdk_target=sdk,sdk_target_delta=float(after[i]-before[i]) if controlled else None,
            target_reason=None if controlled else "passive_joint_not_commanded" if name in passive else "unmapped_joint",
            actual_minus_sdk_target=new["q"][i]-sdk if controlled else None,
            requested_target=controller,requested_target_reason=reason,
            sdk_minus_requested_target=None if controller is None else sdk-controller))
    return dict(valid=True,reason=None,joints=result,
        instance_object_id=post["instance_object_id"],view_object_id=post["view_object_id"],
        prim_paths=list(post["prim_paths"]),row_indices=list(post["row_indices"]),
        source="same_step_PhysX_backend_position_targets_and_measured_q",
        coordinate_frame="articulation_joint_coordinates",
        error_interpretation="observed_tracking_residual_not_a_failure_or_contact_proof")


def _tracking_summary(ledger, side):
    valid=[row["joint_tracking"][side] for row in ledger if row["joint_tracking"][side]["valid"]]
    def maximum(unit):
        values=[abs(joint["actual_minus_sdk_target"]) for row in valid for joint in row["joints"]
                if joint["position_unit"]==unit and joint["actual_minus_sdk_target"] is not None]
        return max(values) if values else None
    return dict(valid_rows=len(valid),missing_rows=len(ledger)-len(valid),
        full_interval_coverage=len(valid)==len(ledger),
        missing_sdk_steps=[row["identity"]["physics_step"] for row in ledger
                           if not row["joint_tracking"][side]["valid"]],
        coverage_scope="provided_consecutive_raw_physics_rows",
        residual_scope="valid_rows_only",
        max_abs_controlled_tracking_error_rad=maximum("rad"),
        max_abs_controlled_tracking_error_m=maximum("m"),
        passive_target_slots_excluded=True)


def audit_file(path, *, expected_sha256, grasp):
    """Consume and hash exactly the audited bytes; reject identity gaps or damage."""
    if grasp not in ("pinch", "tripod"):
        raise ValueError("independent supported distal audit expects pinch or tripod")
    if not isinstance(expected_sha256, str) or not re.fullmatch("[0-9a-f]{64}", expected_sha256):
        raise ValueError("explicit raw source SHA256 required")
    ledger, previous, sequence = [], None, -1
    previous_tracking={}
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
            tracking={side:_joint_tracking(action["pre"].get("hands",{}).get(side),
                action["post"].get("hands",{}).get(side),action["request"],side)
                for side in ("left","right")}
            for side,tracked in tracking.items():
                if not tracked["valid"]:
                    continue
                owner=(tracked["instance_object_id"],tracked["view_object_id"],
                    tuple(tracked["prim_paths"]),tuple(tracked["row_indices"]),
                    tuple(j["joint"] for j in tracked["joints"]))
                if side in previous_tracking and previous_tracking[side]!=owner:
                    raise ValueError("tracking owner identity changed within the same physical episode")
                previous_tracking[side]=owner
            if all(t["valid"] for t in tracking.values()):
                if any(tracking["left"][key]==tracking["right"][key] for key in
                        ("instance_object_id","view_object_id","prim_paths")):
                    raise ValueError("tracking requires distinct left/right owners")
            ledger.append(dict(raw_sequence=ordinal, identity=identity, contact=contact,
                median_z_m=float(np.median(nodes[0, :, 2])), min_z_m=float(nodes[0, :, 2].min()),
                node_positions_world_m=nodes[0].tolist(),
                node_velocities_world_m_s=None if velocity is None else velocity[0].tolist(),
                node_speed_max_m_s=None if velocity is None else float(np.linalg.norm(velocity[0], axis=1).max()),
                joint_state={side: _joints(action["post"].get("hands", {}).get(side)) for side in ("left", "right")},
                joint_tracking=tracking,
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
        joint_tracking_schema="dh116-raw-target-tracking-audit-v1",
        tracking_summary={side:_tracking_summary(ledger,side) for side in ("left","right")},
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
