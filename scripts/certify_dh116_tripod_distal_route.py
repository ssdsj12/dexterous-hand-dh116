#!/usr/bin/env python3
"""Prove a hand-specific Tripod route against a fixed archived rod geometry."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from scripts.audit_dh116_pinch_initial_pose import (
    box_relation, load_geometry, path_displacement_bound, pose_matrix, transform,
)
from scripts.audit_dh116_wire_contact_witness import endpoint_gap
from scripts.certify_dh116_tripod_tracking_envelope import certify_box
from scripts.prepare_dh116_pinch_closure_margin import fk_poses
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, active_limits
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_measured_closure_geometry import rod_support_lower
from go2_pvcnn.control.dh116.tabletop_supported_rod_reference import load_parked_reference


ROUTE_VERSIONS = {
    "dh116-flexible-tripod-right-distal-approach-cpu-v1": ("right", False),
    "dh116-flexible-tripod-right-parked-rod-approach-cpu-v1": ("right", True),
    "dh116-flexible-tripod-left-parked-rod-approach-cpu-v1": ("left", True),
}


def prove_segment(margins, motion_m, *, max_samples, rod_displacement_reserve_m=0., fraction=1.):
    """Prove a continuous linear/slerp segment against the specified rod envelope."""
    if not np.isfinite(rod_displacement_reserve_m) or rod_displacement_reserve_m < 0.:
        raise ValueError("a finite nonnegative rod displacement reserve is required")
    if not 0. < fraction <= 1.:
        raise ValueError("the segment fraction must be in (0, 1]")

    def reserved_margins(point):
        return {name: value - rod_displacement_reserve_m if name.startswith("rod:") else value
                for name, value in margins(float(point[0])).items()}

    return certify_box(reserved_margins, {name: [value] for name, value in motion_m.items()},
                       [0.], [fraction], max_samples=max_samples, max_seconds=180.)


def archived_rod_envelope(receipt, reference, previous_steps):
    """Bound the archived centreline endpoints relative to the sealed final sample."""
    end = reference.physics_step
    if type(previous_steps) is not int or not 1 <= previous_steps <= end - 3:
        raise ValueError("a bounded consecutive archived SDK interval is required")
    start = end - previous_steps
    samples = [load_parked_reference(receipt, reference.receipt_sha256, tick) for tick in range(start, end)]
    samples.append(reference)
    if any(row.physics_step != tick or row.support_boxes != reference.support_boxes
           or not np.array_equal(row.rest_lengths_m, reference.rest_lengths_m)
           for tick, row in zip(range(start, end+1), samples, strict=True)):
        raise ValueError("archived rod envelope requires identical support, clock and material identity")
    distances = [np.linalg.norm(np.asarray(row.node_positions, dtype=float) - reference.node_positions, axis=1)
                 for row in samples]
    # The same material interpolation parameter maps each rod segment to its reference segment.
    # Every interpolated displacement is a convex combination of the corresponding vertex displacements.
    return dict(first_sdk=start, last_sdk=end, node_count=41, reference_sha256=reference.sample_sha256,
                sample_sha256=[row.sample_sha256 for row in samples],
                max_point_displacement_m=float(np.max(distances)), online_motion_bound=False)


def read_terminal(ledger, contract):
    digest = hashlib.sha256()
    packet = None
    with ledger.open("rb") as rows:
        for line in rows:
            digest.update(line)
            event = json.loads(line)
            if event.get("operation") == "physical":
                packet = event["payload"]
    if digest.hexdigest() != contract["physical_ledger_sha256"] or packet is None:
        raise ValueError("the original Tripod native ledger is missing or changed")
    exchange = decode_owned(packet)["exchange"]
    if exchange["physics_step"] != contract["native_sdk_step"]:
        raise ValueError("route must bind the exact native boundary")
    near = exchange["tripod_near_geometry"]
    if not near["passed"] or near["hand_pairs"]:
        raise ValueError("controlled Tripod source boundary must have a clean no-contact ledger")
    side = contract["side"]
    if side not in ("left", "right"):
        raise ValueError("the native Tripod link source must name one physical hand")
    hand = exchange["action_readback"]["post"]["hands"][side]
    joints = hand["q"][0, [hand["joint_names"].index(name) for name in ACTIVE]]
    geometry = near["hands"][side]
    body = dict(zip(geometry["body_names"], geometry["link_poses_xyzw"][0], strict=True))
    return joints, body, exchange["rod"]["node_positions"][0].copy()


def support_clearance_lower(vertices, low, high):
    lower = float(np.linalg.norm(
        np.maximum(np.maximum(low-vertices.max(axis=0), vertices.min(axis=0)-high),0.)))
    if lower > 0.:
        return lower
    return box_relation(vertices, (low+high)/2, high-low)["max_separating_axis_gap_m"]


def proof(model, body, base_q, nodes, contract, *, rod_displacement_reserve_m=0., final_fraction=1.):
    pose0 = body["base_link"][[0,1,2,6,3,4,5]]
    fixed = fk_poses(model, pose0, base_q)
    model = dict(model, source_colliders=deepcopy(model["source_colliders"]),
                 cooked_pads=deepcopy(model["cooked_pads"]))
    sets = {"source": model["source_colliders"], "cooked": model["cooked_pads"]}
    shapes = {}
    for kind, colliders in sets.items():
        for name, row in colliders.items():
            native_local = np.linalg.inv(fixed[name]) @ pose_matrix(body[name][[0,1,2,6,3,4,5]])
            verts = transform(row["vertices"], native_local)
            row["vertices"] = verts
            shapes[f"{kind}:{name}"] = verts
    edges = {row[2]: row for row in model["fk"].edges}
    radii = {}
    for key, vertices in shapes.items():
        name = key.split(":")[1]
        path = model["source_colliders"][name]["source"]["body_path"]
        radius = float(np.linalg.norm(vertices, axis=1).max())
        while path in edges:
            _, path, _, t0, t1, _ = edges[path]
            radius += float(np.linalg.norm(t0[:3, 3])+np.linalg.norm(t1[:3, 3]))
        radii[key] = radius  # Ancestor of base_link only increases the bound.
    supports = [(np.asarray(box["center_m"])-np.asarray(box["size_m"])/2,
                 np.asarray(box["center_m"])+np.asarray(box["size_m"])/2)
                for box in contract["support_boxes"]]
    low, high = active_limits()
    route = contract["waypoints"]
    records = []
    for index, (src, dst) in enumerate(zip(route, route[1:])):
        p0, p1 = [np.asarray(row["pose_wxyz"]) for row in (src, dst)]
        q0, q1 = [np.asarray(row["active_q_rad"]) for row in (src, dst)]
        if np.any(np.minimum(q0, q1)<low) or np.any(np.maximum(q0, q1)>high):
            raise ValueError("Tripod interval exceeds original active joint limits")
        rot = Rotation.from_quat(np.stack((p0,p1))[:,[4,5,6,3]])
        orientation = Slerp([0.,1.],rot)
        wrist = float(np.linalg.norm(p1[:3]-p0[:3]))
        angle = float((rot[1]*rot[0].inv()).magnitude())
        motion = {}
        for kind in sets:
            bounds = path_displacement_bound(model, q0, q1-q0, cooked=kind=="cooked")
            for name, value in bounds.items():
                shape = f"{kind}:{name}"
                for constraint in ("rod","support","table"):
                    motion[f"{constraint}:{shape}"] = wrist + radii[shape]*angle + value

        def margins(t):
            pose = np.r_[p0[:3] + t*(p1[:3]-p0[:3]), orientation(t).as_quat()[[3,0,1,2]]]
            worlds = fk_poses(model, pose, q0+t*(q1-q0))
            result = {}
            for key, vertices in shapes.items():
                points = transform(vertices,worlds[key.split(":")[1]])
                lower = rod_support_lower(points, nodes)
                if lower <= .001:
                    lower = endpoint_gap(points,nodes,.005)["lower_m"]
                result["rod:"+key] = lower
                result["table:"+key] = float(points[:,2].min()-.3-.0021)
                result["support:"+key] = min(support_clearance_lower(points, lo, hi)
                                             for lo, hi in supports)
            return result
        records.append(prove_segment(margins, motion, max_samples=2048,
            rod_displacement_reserve_m=rod_displacement_reserve_m,
            fraction=final_fraction if index == len(route)-2 else 1.))
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parked-receipt", type=Path)
    parser.add_argument("--archived-rod-window-steps", type=int)
    parser.add_argument("--final-segment-fraction", type=float, default=1.)
    args = parser.parse_args(argv)
    contract = json.loads(args.contract.read_bytes())
    version = ROUTE_VERSIONS.get(contract.get("version"))
    if (version is None or contract.get("side") != version[0] or len(contract["waypoints"]) < 2
            or len(contract["support_boxes"]) != 2
            or contract.get("native_measurement") is not False):
        raise ValueError("explicit hand-specific static-rod route required")
    base_q, body, nodes = read_terminal(args.ledger, contract)
    parked = contract.get("parked_rod_reference")
    if (parked is not None) != version[1]:
        raise ValueError("route version and physical reference kind differ")
    binding = envelope = None
    if not 0. < args.final_segment_fraction <= 1.:
        raise ValueError("the last segment fraction must be in (0, 1]")
    if args.final_segment_fraction != 1. and args.archived_rod_window_steps is None:
        raise ValueError("a partial route must name its archived rod envelope")
    if parked is None:
        if args.parked_receipt is not None or args.archived_rod_window_steps is not None:
            raise ValueError("a parked rod cannot override a frozen raised-reference route")
        nodes[:,2] += contract["rod_translation_z_m"]
    else:
        if args.parked_receipt is None:
            raise ValueError("the exact parked native receipt is required for this route")
        ref = load_parked_reference(args.parked_receipt, parked["receipt_sha256"], parked["sdk_step"])
        if ref.sample_sha256 != parked["sample_sha256"] or list(ref.support_boxes) != contract["support_boxes"]:
            raise ValueError("actual parked sample or support boxes differ from the frozen route")
        nodes = ref.node_positions
        binding = dict(sample_sha256=ref.sample_sha256, receipt_sha256=ref.receipt_sha256,
            sdk_step=ref.physics_step, rod_tick=ref.rod_tick,
            max_snapshot_speed_m_s=float(np.linalg.norm(ref.node_velocities,axis=-1).max()))
        if args.archived_rod_window_steps is not None:
            envelope = archived_rod_envelope(args.parked_receipt, ref, args.archived_rod_window_steps)
    side = version[0]
    reserve = envelope["max_point_displacement_m"] if envelope is not None else 0.
    records = proof(load_geometry(Path.cwd(), side=side), body, base_q, nodes, contract,
                    rod_displacement_reserve_m=reserve, final_fraction=args.final_segment_fraction)
    passed = len(records) == len(contract["waypoints"])-1 and all(row["proved"] for row in records)
    out=dict(scope=f"{side}_hand_static_rod_kinematic_approach",proved=passed,
             native_measurement=False,grasp_qualified=False,records=records)
    if binding is not None:
        out["fixed_native_rod_reference"] = binding
    if envelope is not None:
        out.update(scope=f"{side}_hand_archived_rod_envelope_kinematic_approach", archived_rod_envelope=envelope,
                   final_segment_fraction=args.final_segment_fraction, execution_ready=False)
    with args.output.open("x") as stream: json.dump(out, stream, indent=2,allow_nan=False)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
