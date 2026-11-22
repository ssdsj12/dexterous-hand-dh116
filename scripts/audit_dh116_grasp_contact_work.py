"""Compare recorded hand reaction work using endpoint quadrature only.

This cannot measure total mechanical energy, prove passivity, or grant grasp
qualification. It does not advance simulation or change any issued force.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned


def _vector(value, size):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError("complete finite measured vector required")
    return value


def endpoint_work(com0, com1, quat0, quat1, force_issued, torque_issued,
                  force_current, torque_current):
    """World-frame wrench times COM displacement / world rotation, in joules."""
    p0, p1 = (_vector(x, 3) for x in (com0, com1))
    q0, q1 = (_vector(x, 4) for x in (quat0, quat1))
    if any(abs(np.linalg.norm(q)-1.) > 1.e-5 for q in (q0, q1)):
        raise ValueError("measured unit XYZW quaternions required")
    dp = p1-p0
    dr = (Rotation.from_quat(q1)*Rotation.from_quat(q0).inv()).as_rotvec()
    fi, ti, fc, tc = (_vector(x, 3) for x in
                      (force_issued, torque_issued, force_current, torque_current))
    issued, current = float(fi@dp+ti@dr), float(fc@dp+tc@dr)
    return dict(issued_j=issued, current_j=current, difference_j=issued-current,
                com_displacement_m=dp.tolist(), world_rotation_increment_rad=dr.tolist(),
                energy_qualified=False)


def audit_pair(previous, current):
    """Match actual adjacent SDK records and the reaction that really was issued."""
    a, b = previous["rod"], current["rod"]
    n = previous["physics_step"]
    identities = ("run_uuid", "episode_uid", "env_id", "reset_id", "rod_uid")
    if (type(n) is not int or current["physics_step"] != n+1
            or any(a[k] != b[k] for k in identities)
            or a["physics_step"] != n or b["physics_step"] != n+1
            or previous["sdk_hand_geometry"]["physics_step"] != n
            or current["sdk_hand_geometry"]["physics_step"] != n+1
            or current["reaction_source_physics_step"] != n
            or b["rod_tick"]-a["rod_tick"] != 12
            or a["mesh_bindings"] != b["mesh_bindings"]):
        raise ValueError("same-Reset adjacent480Hz SDK /12rod substeps and bindings required")
    result = {}
    for side in ("left", "right"):
        issued = current["issued_composer_wrench"][side]
        old, fresh = (row["hand_wrenches_com_world"][side] for row in (a, b))
        ids = issued["body_ids"]
        if (not ids or len(set(ids)) != len(ids) or ids != old["body_ids"]
                or ids != fresh["body_ids"]
                or any(route.get("is_global") is not True for route in (issued, old, fresh))):
            raise ValueError("same actual body IDs and World wrench frame required")
        for field in ("forces", "torques"):
            values = [np.asarray(route[field], dtype=float) for route in (issued, old, fresh)]
            if (any(x.shape != (1, len(ids), 3) or not np.isfinite(x).all() for x in values)
                    or not np.array_equal(values[0], values[1])):
                raise ValueError("issued reaction must match the previous actual rod step")
        matched = set()
        for key, binding in b["mesh_bindings"].items():
            if binding["side"] != side:
                continue
            if binding["body_index"] not in ids or binding["body_index"] in matched:
                raise ValueError("mesh-to-body mapping incomplete or duplicated")
            matched.add(binding["body_index"])
            j = ids.index(binding["body_index"])
            geo0, geo1 = (row["sdk_hand_geometry"] for row in (previous, current))
            result[key] = endpoint_work(geo0["centers_of_mass"][key], geo1["centers_of_mass"][key],
                geo0["poses"][key][1], geo1["poses"][key][1],
                issued["forces"][0, j], issued["torques"][0, j],
                fresh["forces"][0, j], fresh["torques"][0, j])
        if matched != set(ids):
            raise ValueError("every issued body needs its own measured COM/rotation")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--start-sdk", type=int, required=True)
    parser.add_argument("--end-sdk", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.start_sdk < 1 or not args.start_sdk <= args.end_sdk:
        parser.error("inclusive positive SDK interval required, with its preceding record")
    sources, records = {}, []
    summary = defaultdict(lambda: dict(issued_j=0., current_j=0., difference_j=0.,
                                      positive_difference_rows=0, rows=0))
    previous = None
    for sdk in range(args.start_sdk-1, args.end_sdk+1):
        path = args.native / f"tripod_control_{sdk}.json"
        raw = path.read_bytes()
        sources[path.name] = hashlib.sha256(raw).hexdigest()
        current = decode_owned(json.loads(raw))
        if previous is not None:
            row = audit_pair(previous, current)
            records.append(dict(physics_step=sdk, links=row))
            for key, value in row.items():
                for field in ("issued_j", "current_j", "difference_j"):
                    summary[key][field] += value[field]
                summary[key]["rows"] += 1
                summary[key]["positive_difference_rows"] += int(value["difference_j"] > 0.)
        previous = current
    result = dict(scope="endpoint_wrench_displacement_estimate_not_energy_or_passivity_proof",
        start_sdk=args.start_sdk, end_sdk=args.end_sdk,
        source_sha256=sources, summary=dict(summary), records=records,
        energy_qualified=False, native_qualification=False, grasp_qualified=False,
        teacher_admission=False, new_physics_steps=0, ppo_updates=0)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps(dict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
