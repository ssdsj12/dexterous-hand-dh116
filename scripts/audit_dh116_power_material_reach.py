#!/usr/bin/env python3
"""Audit bounded object-local Power wrist offsets on the same supported rod."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_teacher_seed_initialization import seed_arrays
from go2_pvcnn.control.dh116.tabletop_teacher_supported_runtime import supported_seed_plan
from scripts.audit_dh116_pinch_initial_pose import DISTAL_PAD_LINKS, checked_json, load_geometry
from scripts.audit_dh116_wire_contact_witness import audit_hand, shapes_world
from scripts.prepare_dh116_power_palm_seed import support_clearance, thumb_clearance
from scripts.prepare_dh116_wire_pose_candidate import HANDS, _material_tangents, _pad_gap_interval

ANGLES_DEG = (0., -30., -15., 15., 30.)
LATERAL_M = (0., -.004, .004, -.008, .008)
BACKOFF_M = (0., .004)
SHORTLIST = 10


def material_wrist_seed(root, station, tangent, inward, *, angle_deg, lateral_m, backoff_m):
    axis = np.cross(tangent, inward)
    length = np.linalg.norm(axis)
    if not np.isfinite(length) or length <= 1e-8:
        raise ValueError("nondegenerate material cross-section required")
    spin = Rotation.from_rotvec(np.deg2rad(angle_deg) * tangent)
    offset = root[:3] - station + lateral_m * axis / length - backoff_m * inward
    orientation = spin * Rotation.from_quat(np.roll(root[3:], -1))
    return np.r_[station + spin.apply(offset), np.roll(orientation.as_quat(), 1)]


def audit_reach(artifact, *, project):
    nodes, *_ = seed_arrays(artifact["trial_seed"])
    plan, _ = supported_seed_plan(project)
    supported = plan["seeds_by_style"]["power"]
    if not np.array_equal(nodes, np.asarray(supported["nodes"], dtype=nodes.dtype)):
        raise ValueError("same end-supported Power rod required")
    supports = plan["scene"]["supports"]
    tangents = _material_tangents(nodes)
    required = ("source:base_link", *("cooked:"+name for name in DISTAL_PAD_LINKS))
    hands = {}
    for side in HANDS:
        g = load_geometry(project, side=side)
        row = artifact["search"][side]
        prior = next(x for x in row["closure_endpoints"] if x["wrist_fraction"] == 0.)
        root, q = np.asarray(prior["root"]), np.asarray(prior["active_q"])
        station = row["node_id"]
        inward = np.asarray(row["palm_toward_fingers"])
        layout = g["reference"]["reference_layout"]
        top = layout["table_center"][2] + layout["table_size"][2] / 2
        radius = g["old_candidate"]["rod"]["radius_m"]
        candidates = []
        for angle, lateral, backoff in itertools.product(ANGLES_DEG, LATERAL_M, BACKOFF_M):
            wrist = material_wrist_seed(root, nodes[station], tangents[station], inward,
                                       angle_deg=angle, lateral_m=lateral, backoff_m=backoff)
            shapes = shapes_world(g, wrist, q)
            rise = max(0., top+CLEARANCE_M+1e-8-min(v[:, 2].min() for v in shapes.values()))
            shapes = {key: hull + [0., 0., rise] for key, hull in shapes.items()}
            wrist[2] += rise
            gaps = {key: _pad_gap_interval(hull, nodes, radius) for key, hull in shapes.items()}
            if min(value["lower_m"] for value in gaps.values()) < 0.:
                continue
            bound = max(gaps[key]["upper_m"] for key in required)
            candidates.append(dict(root=wrist.tolist(), table_correction_m=rise, angle_deg=angle,
                                   lateral_m=lateral, backoff_m=backoff, max_conservative_upper_m=bound,
                                   shapes=shapes))
        checked = []
        for candidate in sorted(candidates, key=lambda row: row["max_conservative_upper_m"])[:SHORTLIST]:
            shapes = candidate.pop("shapes")
            hits, support_gap = support_clearance(shapes, supports)
            if hits or support_gap < CLEARANCE_M or thumb_clearance(shapes) <= 0.:
                continue
            witness = audit_hand(shapes, nodes, radius, layout, DISTAL_PAD_LINKS, include_palm=True)
            if not witness["table_clearance_passed"] or not witness["complete_convex_clearance_passed"]:
                continue
            candidate["terminal"] = witness
            candidate["max_exact_upper_m"] = max(witness["gaps"][key]["upper_m"] for key in required)
            checked.append(candidate)
        baseline = max(prior["exact"]["gaps"][key]["upper_m"] for key in required)
        hands[side] = dict(node_id=station, candidates=len(ANGLES_DEG)*len(LATERAL_M)*len(BACKOFF_M),
                           rod_clear_candidates=len(candidates), unchanged_q_rad=q.tolist(),
                           baseline_max_exact_upper_m=baseline,
                           checked=sorted(checked, key=lambda row: row["max_exact_upper_m"]))
    return dict(hands=hands, support_boxes=len(supports), actual_isaac_steps=0,
                continuous_path_verified=False, locked_palm_cooked_available=False,
                native_contact_verified=False, teacher_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure-evidence", type=Path, required=True)
    parser.add_argument("--closure-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = checked_json(args.closure_evidence, args.closure_sha256)
    result = audit_reach(artifact, project=Path(__file__).resolve().parents[1])
    result["input_sha256"] = args.closure_sha256
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
