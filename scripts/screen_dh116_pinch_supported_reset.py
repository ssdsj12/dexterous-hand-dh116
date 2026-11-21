#!/usr/bin/env python3
"""Transfer each measured Pinch hand to a separate supported flexible-rod setup."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_supported_rod_reference import load_parked_reference
from scripts.audit_dh116_pinch_initial_pose import load_geometry
from scripts.certify_dh116_tripod_distal_route import support_clearance_lower
from scripts.screen_dh116_tripod_supported_reset import BASE_SCENE, ROOT, archived_tube, separation
from scripts.verify_dh116_power_shared_rod import native, sources

PINCH_OUTER_SHA = "81cd7c78916c8811b8b7f8368fcbd4767075096764a8a591b595d6a10b0b1f3c"
SUPPORTED_ROD_SHA = "c1683e8b44aa5e123aeee49a3adfd070fdf738ea20858ff0106a5756abda0aea"
NODES = {"left": [34,35], "right": [2,3]}
OFFSETS = {"left": [0.,0.,.008], "right": [.0015,0.,.001]}
SCOPE = "pinch_end_supported_contact_diagnostic"


def material_transfer(old, new):
    old, new = (np.asarray(points, dtype=float) for points in (old,new))
    if any(v.shape != (2,3) or not np.isfinite(v).all() for v in (old,new)):
        raise ValueError("two finite original material points per pose required")
    a,b = (rows[1]-rows[0] for rows in (old,new))
    m,n = np.linalg.norm(a),np.linalg.norm(b)
    if min(m,n) <= 1e-12:
        raise ValueError("original nonzero material spans required")
    a,b = a/m,b/n
    cross = np.cross(a,b)
    size, dot = np.linalg.norm(cross), float(a@b)
    if size <= 1e-12:
        if dot < 0.:
            raise ValueError("opposite material order is ambiguous")
        rot = Rotation.identity()
    else:
        rot = Rotation.from_rotvec(cross/size*np.arctan2(size,dot))
    return rot,old.mean(0),new.mean(0)


def source_rows(outer_path):
    outer = sources.checked_json(outer_path,PINCH_OUTER_SHA)
    if (outer["exit_code"]!=0 or outer["debugger_inferior_exit_code"]!=0
            or outer["forced_exit"] or outer["exceeded_budget"]):
        raise ValueError("the original Pinch run must have closed naturally")
    ledger = outer_path.parent/"native/physical.jsonl"
    digest, reset, closure = hashlib.sha256(), None, None
    with ledger.open("rb") as f:
        for line in f:
            digest.update(line)
            data = json.loads(line)
            if data.get("operation")=="reset":
                reset = decode_owned(data["payload"])["reset"]
            if data.get("operation")=="physical" and data["payload"]["exchange"]["physics_step"]==362:
                closure = decode_owned(data["payload"])["exchange"]
    if (digest.hexdigest()!=outer["artifact_sha256"][ledger.name] or reset is None or closure is None
            or reset["after_clock"]["physics_step"]!=2 or closure["guard_failure"] is not None):
        raise ValueError("sealed original Reset and force-loaded Pinch boundary required")
    return reset,closure,digest.hexdigest()


def screen(pinch_receipt, parked_receipt):
    reset,closure,ledger_sha = source_rows(pinch_receipt)
    basis = json.loads(BASE_SCENE.read_bytes())
    samples = [load_parked_reference(parked_receipt,SUPPORTED_ROD_SHA,tick) for tick in range(3,241)]
    if list(samples[-1].support_boxes)!=basis["support_scene"]["supports"]:
        raise ValueError("original measured supports changed")
    tube = archived_tube(np.stack([basis["rod_nodes_w_m"], *[x.node_positions for x in samples]]))
    boxes = [(np.asarray(b["center_m"])-np.asarray(b["size_m"])/2,
              np.asarray(b["center_m"])+np.asarray(b["size_m"])/2) for b in samples[-1].support_boxes]
    source_nodes = closure["rod"]["node_positions"][0]
    config = deepcopy(basis)
    config.update(scope=SCOPE,offline_candidate_sha256=ledger_sha)
    records = {}
    for i, (side, nodes) in enumerate(NODES.items()):
        rot, c0, c1 = material_transfer(source_nodes[nodes],samples[-1].node_positions[nodes])
        offset = np.asarray(OFFSETS[side])
        hand = reset["hands"][side]
        body = dict(zip(hand["body_names"],np.asarray(hand["body_poses_xyzw"]).reshape(-1,7),strict=True))
        geometry = load_geometry(ROOT, side=side)
        world = {}
        for group, shapes in (("source",geometry["source_colliders"]),("cooked",geometry["cooked_pads"])):
            for name, shape in shapes.items():
                pose = body[name]
                points = Rotation.from_quat(pose[3:]).apply(shape["vertices"])+pose[:3]
                world[group+":"+name] = rot.apply(points-c0)+c1+offset
        gaps = {name:separation(points,*tube) for name,points in world.items()}
        rod = min(gaps,key=gaps.get)
        support = min(support_clearance_lower(points,lo,hi) for points in world.values() for lo,hi in boxes)
        table = min(float(points[:,2].min()-.3) for points in world.values())
        root = np.asarray(reset["seed"]["reset_root_pose_w"][i])
        xyz = rot.apply(root[:3]-c0)+c1+offset
        quat = (rot*Rotation.from_quat(root[[4,5,6,3]])).as_quat()[[3,0,1,2]]
        active = np.asarray(hand["q"]).reshape(-1)[[hand["joint_names"].index(name) for name in ACTIVE]]
        config["hands"][side] = dict(pose_wxyz=np.r_[xyz,quat].tolist(),q_rad=active.tolist())
        records[side] = dict(material_nodes=nodes,material_rotation_rad=rot.magnitude(),
            offset_world_m=offset.tolist(),rod_gap_lower_m=gaps[rod],limiting_shape=rod,
            table_clearance_m=table,support_gap_m=support,shape_count=len(world),
            passed=gaps[rod]>.001 and table>=.0021 and support>.001)
    return dict(scope=SCOPE, config=config, hands=records, source_pinch_outer_sha256=PINCH_OUTER_SHA,
                source_parked_outer_sha256=SUPPORTED_ROD_SHA, source_ledger_sha256=ledger_sha,
                shared_support_config_sha256=native.sha(BASE_SCENE), init_and_archived_samples=239,
                all_passed=all(row["passed"] for row in records.values()),native_reset_verified=False,
                grasp_qualified=False,teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("pinch-receipt","parked-receipt","output"):
        parser.add_argument("--"+flag,required=True,type=Path)
    args = parser.parse_args(argv)
    result = screen(args.pinch_receipt,args.parked_receipt)
    native.write_new(args.output,result)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
