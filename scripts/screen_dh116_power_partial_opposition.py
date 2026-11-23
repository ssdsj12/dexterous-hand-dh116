
"""Audit a two-pad contact stage against the archived Power palm boundary."""

import argparse
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.linear_coupling import expand
from scripts.audit_dh116_power_fixed_wrist_digits import (
    DISTAL_PAD_LINKS, align_measured_wrist, load_geometry, load_measured_palm_boundary,
    measured_palm_boundary, shapes_world,
)
from scripts.audit_dh116_pinch_initial_pose import checked_json
from scripts.audit_dh116_wire_contact_witness import endpoint_gap
from scripts.diagnose_dh116_teacher_initialization import digest
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_contact
from scripts.prepare_dh116_wire_pose_candidate import _pad_gap_interval
from scripts.screen_dh116_power_measured_reseat import (
    PAD_PENETRATION_M, REACH_M, _locked_grids, _wrist_clearance,
    support_adjacent_nodes,
)
from scripts.verify_dh116_power_staged_palm_witness import MAX_PALM_PENETRATION_M

TRANSPORT_READINESS = "docs/contracts/dh116_teacher_power_transport_stations_v1.json"
THUMB = DISTAL_PAD_LINKS[0]
NON_TARGET_CLEARANCE_M = .00025
OPPOSITION_COS = -.25


def local_opposition_pairs(candidate, support_nodes):
    """Find geometric thumb pairs on the same or adjacent interior material nodes."""
    scores = candidate["witnesses"]
    near = candidate["independent_near_pads"]
    if THUMB not in near:
        return []
    thumb = scores[THUMB]
    pairs = []
    for partner in near:
        if partner == THUMB or partner not in DISTAL_PAD_LINKS:
            continue
        target = scores[partner]
        indices = (thumb["node_id"], target["node_id"])
        if any(i in support_nodes for i in indices) or abs(indices[0]-indices[1]) > 1:
            continue
        cosine = float(np.asarray(thumb["normal_world"]) @ np.asarray(target["normal_world"]))
        if cosine <= OPPOSITION_COS and max(abs(thumb["gap_m"]), abs(target["gap_m"])) <= REACH_M:
            pairs.append(dict(partner=partner, thumb_node=indices[0], partner_node=indices[1],
                              opposition_cos=cosine, thumb_joints_rad=thumb["joints_rad"],
                              partner_flexion_rad=target["joints_rad"][1]))
    return pairs


def staged_joints(active, pair):
    """Hold the other digits at their measured values throughout the two stages."""
    abduct = np.array(active, dtype=float, copy=True)
    abduct[0] = pair["thumb_joints_rad"][0]
    close = abduct.copy()
    close[1] = pair["thumb_joints_rad"][1]
    partner = DISTAL_PAD_LINKS.index(pair["partner"]) + 1
    close[partner] = pair["partner_flexion_rad"]
    expand(abduct)
    expand(close)
    return abduct, close


def rod_clearance_required(label, targets):
    name = label.split(":", 1)[1]
    return -PAD_PENETRATION_M if name in targets else NON_TARGET_CLEARANCE_M


def clearance_status(*, lower_m, upper_m, required_m):
    if lower_m > required_m:
        return "clear"
    if upper_m < required_m:
        return "certified_below_limit"
    return "unresolved"


def endpoint(geometry, pose, q, nodes, scene, palm_grid, *, targets=(), require_palm_reach=True):
    """Check a sampled endpoint with the native palm SDF and whole-hand rod geometry."""
    geometry_result = _wrist_clearance(geometry, pose, q, scene)
    palm = fk_poses(geometry, pose, q)["base_link"]
    radius = scene["wire_radius_m"]
    palm_gap = sample_contact(palm_grid, palm, nodes, radius=radius, dtype=np.float64)["minimum_gap_m"]
    palm_passed = (palm_gap is not None and palm_gap >= -MAX_PALM_PENETRATION_M
                   and (not require_palm_reach or palm_gap <= REACH_M))
    preliminary = []
    for label, verts in shapes_world(geometry, pose, q).items():
        # Measured native Power palms are admitted using the original registered palm SDF.
        if label == "source:base_link":
            continue
        required = rod_clearance_required(label, targets)
        bound = _pad_gap_interval(verts, nodes, radius)
        preliminary.append((bound["lower_m"] - required, label, verts, required, bound))
    failure = None
    for lower_margin, label, verts, required, coarse in sorted(preliminary, key=lambda x: x[0]):
        if lower_margin > 0.:
            break
        precise = coarse if coarse["upper_m"] < required else endpoint_gap(verts, nodes, radius)
        status = clearance_status(lower_m=precise["lower_m"], upper_m=precise["upper_m"], required_m=required)
        if status != "clear":
            failure = dict(shape=label, lower_m=precise["lower_m"], upper_m=precise["upper_m"],
                           required_m=required, status=status, solid_overlap_proved=precise["upper_m"] < 0.)
            break
    return dict(open_hull_guards=geometry_result, palm_gap_m=palm_gap, palm_sdf_passed=palm_passed,
                limiting_rod=failure, passed=bool(geometry_result["passed"] and palm_passed and failure is None))


def audit_side(row, scene, nodes, side, candidates, files, project):
    geometry = load_geometry(project, side=side)
    wrist, active, _, _ = align_measured_wrist(geometry, side, row, apply_to_geometry=True)
    grids, hashes = _locked_grids(row["rod"], side, files)
    support_nodes = set(np.flatnonzero(support_adjacent_nodes(row["rod"], nodes, scene)))
    palm_grid = grids["base_link"]
    source = endpoint(geometry, wrist, active, nodes, scene, palm_grid)
    cache = {}

    def check(state, joints, allowed=()):
        key = tuple(state), tuple(joints), allowed
        if key not in cache:
            cache[key] = endpoint(geometry, state, joints, nodes, scene, palm_grid, targets=allowed)
        return cache[key]

    rows = []
    for number, candidate in enumerate(candidates):
        for pair in local_opposition_pairs(candidate, support_nodes):
            pose = np.asarray(candidate["wrist_wxyz_m"], dtype=float)
            abduct, close = staged_joints(active, pair)
            targets = (THUMB, pair["partner"])
            stages = dict(wrist_only=check(pose, active),
                          thumb_abduction=check(pose, abduct),
                          two_pad_closure=check(pose, close, targets))
            rows.append(dict(candidate_index=number, palm_target_node=candidate["target_node_id"],
                             wrist_wxyz_m=pose.tolist(), pair=pair, start_q_rad=active.tolist(),
                             abduct_q_rad=abduct.tolist(), close_q_rad=close.tolist(), endpoints=stages,
                             sampled_endpoints_passed=bool(source["passed"]
                                                          and all(s["passed"] for s in stages.values()))))
    return dict(source_boundary=source, original_palm_sha256=hashes["base_link"], proposals=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palm-receipt", type=Path, required=True)
    parser.add_argument("--transport-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    row, scene, sources = load_measured_palm_boundary(args.palm_receipt, project=project)
    clock, nodes, _ = measured_palm_boundary(row)
    readiness = json.loads((project/TRANSPORT_READINESS).read_bytes())
    archive_sha = readiness["screen"]["artifact_sha256"]
    screened = checked_json(args.transport_audit, archive_sha)
    if screened["clock"] != clock or screened["archive"] != sources:
        raise ValueError("station proposals must bind the exact measured palm exchange")
    outer = json.loads(args.palm_receipt.read_bytes())
    plan = json.loads(args.palm_receipt.with_name("plan.json").read_bytes())
    if digest(plan) != outer["frozen_sha256"]:
        raise ValueError("native frozen plan required for original palm SDFs")
    hands = {side: audit_side(row, scene, nodes, side, screened["hands"][side]["candidates"],
                             plan["frozen_files"], project) for side in ("left", "right")}
    result = dict(version="dh116-power-partial-opposition-endpoints-v1", archive=sources, clock=clock,
                  screen_sha256=archive_sha, hands=hands, continuous_route_verified=False, new_contact_verified=False,
                  native_action_performed=False, teacher_admission=False, grasp_qualified=False)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
