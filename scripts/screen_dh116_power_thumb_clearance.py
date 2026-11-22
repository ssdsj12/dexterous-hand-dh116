"""Screen joint Power thumb abduction and flexion on one archived wire boundary."""

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import active_limits
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import CONTRACT as WRIST
from scripts.audit_dh116_pinch_initial_pose import checked_json
from scripts.audit_dh116_power_fixed_wrist_digits import (
    align_measured_wrist, load_geometry, load_measured_palm_boundary, measured_palm_boundary,
)
from scripts.audit_dh116_wire_contact_witness import endpoint_gap
from scripts.diagnose_dh116_teacher_initialization import digest
from scripts.diagnose_dh116_teacher_initialization import digest
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_grid
from scripts.prepare_dh116_wire_pose_candidate import _pad_gap_interval
from scripts.screen_dh116_power_measured_reseat import (
    PAD_PENETRATION_M, REACH_M, _locked_grids, palm_anchor, support_adjacent_nodes,
)
from scripts.screen_dh116_power_partial_opposition import (
    NON_TARGET_CLEARANCE_M, OPPOSITION_COS, THUMB, clearance_status, endpoint, staged_joints,
)
from scripts.verify_dh116_power_staged_palm_witness import MAX_PALM_PENETRATION_M

READINESS = "docs/contracts/dh116_teacher_power_partial_opposition_stage_v1.json"
MIDDLE = "finger13_link"
ROUTE_INTERVALS = 16


def thumb_joint_grid(active, sample_counts=(41, 61)):
    low, high = active_limits()
    axes = [np.unique(np.r_[active[i], np.linspace(low[i], high[i], count)])
            for i, count in enumerate(sample_counts)]
    return np.asarray(list(product(*axes)))


def opposing_thumb_witnesses(gap, dots, allowed, *, partner_node):
    """Return one eligible rod node per joint sample, in material-node order."""
    local = allowed & (np.abs(np.arange(gap.shape[1]) - partner_node) <= 1)
    deep = np.any(gap < -PAD_PENETRATION_M, axis=1)
    eligible = local[None] & (np.abs(gap) <= REACH_M) & (dots <= OPPOSITION_COS) & ~deep[:, None]
    rows = np.flatnonzero(eligible.any(axis=1))
    return rows, eligible[rows].argmax(axis=1)


def middle_link_clearance(hull, nodes, radius):
    """Distinguish proved clearance, proved rejection, and a straddling bound."""
    required = NON_TARGET_CLEARANCE_M
    bound = _pad_gap_interval(hull, nodes, radius)
    if bound["lower_m"] <= required <= bound["upper_m"]:
        bound = endpoint_gap(hull, nodes, radius)
    status = clearance_status(lower_m=bound["lower_m"], upper_m=bound["upper_m"], required_m=required)
    return dict(bound, required_m=required, status=status)


def thumb_hulls(model, active, settings, tip_offset):
    unit = np.array([0., 0., 0., 1., 0., 0., 0.])
    vertices = model["source_colliders"][MIDDLE]["vertices"]
    tip, middle = [], []
    for angles in settings:
        q = active.copy()
        q[:2] = angles
        links = fk_poses(model, unit, q)
        tip.append((links[THUMB] @ tip_offset)[:3, :4])
        middle.append(vertices @ links[MIDDLE][:3, :3].T + links[MIDDLE][:3, 3])
    return np.asarray(tip), np.asarray(middle)


def wrist_samples(source, target, projection, side, fractions, *, retreat_world_m=None):
    """Sample a static route in the original cumulative intrinsic-XYZ convention."""
    index = ("left", "right").index(side)
    old_target = projection["projected_root_poses"][0, index]
    seed = Rotation.from_quat(old_target[[4, 5, 6, 3]]) * Rotation.from_euler(
        "XYZ", projection["projected_angles"][0, index]).inv()
    start, end = [(seed.inv() * Rotation.from_quat(p[[4, 5, 6, 3]])).as_euler("XYZ")
                  for p in (source, target)]
    angles = start + fractions[:, None] * (end - start)
    positions = source[:3] + fractions[:, None] * (target[:3] - source[:3])
    if retreat_world_m is not None:
        positions -= np.sin(np.pi * fractions)[:, None] * retreat_world_m
    orientations = (seed * Rotation.from_euler("XYZ", angles)).as_quat()
    return np.column_stack((positions, orientations[:, [3, 0, 1, 2]]))


def thumb_path(active, target, fractions, *, flex_exponent):
    joints = active + fractions[:, None] * (target - active)
    flex = 1. - (1. - fractions)**flex_exponent
    joints[:, 1] = active[1] + flex * (target[1] - active[1])
    return joints


def sample_route(model, source, wrist, active, middle_q, close_q, pair, side, row, nodes,
                 scene, palm_grid, other_bounds, *, mode, retreat_world_m=None, flex_exponent=1.,
                 intervals=ROUTE_INTERVALS):
    """Find rejected samples against the archived rod, before any live route proposal."""
    fractions = np.linspace(0., 1., intervals + 1)
    projection = row["wrist_projection"]
    index = ("left", "right").index(side)
    origin = projection["projected_root_poses"][0, index, :3] - projection["palm_offsets_m"][0, index]
    wrists = wrist_samples(source, wrist, projection, side, fractions, retreat_world_m=retreat_world_m)
    fixed_wrist = np.broadcast_to(wrist, wrists.shape)
    fixed_active = np.broadcast_to(active, (len(fractions), len(active)))
    middle_qs = thumb_path(active, middle_q, fractions, flex_exponent=flex_exponent)
    close_qs = middle_q + fractions[:, None] * (close_q - middle_q)
    stages = [
        ("wrist_reseat", wrists, fixed_active, ()),
        (mode, fixed_wrist, middle_qs, (THUMB,) if mode == "thumb_first" else ()),
        ("two_pad_closure", fixed_wrist, close_qs, (THUMB, pair["partner"])),
    ]
    result = {}
    for name, poses, joints, targets in stages:
        failures = []
        for fraction, pose, q in zip(fractions, poses, joints, strict=True):
            check = endpoint(model, pose, q, nodes, scene, palm_grid, targets=targets)
            bounds = check["open_hull_guards"]["hand_y_bounds_m"]
            separation = bounds[0] - other_bounds[1] if side == "left" else other_bounds[0] - bounds[1]
            offset = pose[:3] - origin
            within_workspace = bool(np.all(offset >= WRIST["palm_lower_m"])
                                    and np.all(offset <= WRIST["palm_upper_m"]))
            gap = check["palm_gap_m"]
            palm_clear = gap is not None and gap >= -MAX_PALM_PENETRATION_M
            if (not check["open_hull_guards"]["passed"] or check["limiting_rod"] is not None
                    or not palm_clear or separation <= 0. or not within_workspace):
                failures.append(dict(fraction=float(fraction), separation_y_m=separation,
                                     within_workspace=within_workspace, check=check))
        result[name] = dict(samples=len(fractions), failed=len(failures),
                            first_failed=failures[0] if failures else None)
    return result


def audit_side(row, scene, nodes, side, earlier, transport, files, project, other_bounds):
    model = load_geometry(project, side=side)
    source, active, _, offsets = align_measured_wrist(model, side, row, apply_to_geometry=True)
    grids, hashes = _locked_grids(row["rod"], side, files)
    settings = thumb_joint_grid(active)
    tip, middle = thumb_hulls(model, active, settings, offsets[THUMB])
    indices = np.arange(len(nodes))
    allowed = indices > len(nodes)//2 if side == "left" else indices < len(nodes)//2
    allowed &= ~support_adjacent_nodes(row["rod"], nodes, scene)
    radius = scene["wire_radius_m"]
    proposals = []
    for item in earlier["proposals"]:
        if not earlier["source_boundary"]["passed"] or not item["endpoints"]["wrist_only"]["passed"]:
            continue
        pair = item["pair"]
        pose = np.asarray(item["wrist_wxyz_m"])
        orientation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        rotations = orientation @ tip[:, :, :3]
        translations = tip[:, :, 3] @ orientation.T + pose[:3]
        local = np.einsum("knj,kji->kni", nodes[None] - translations[:, None], rotations)
        phi, gradient, valid = sample_grid(grids[THUMB], local.reshape(-1, 3), dtype=np.float64)
        shape = (len(tip), len(nodes))
        gap = np.where(valid.reshape(shape), phi.reshape(shape) - radius, np.nan)
        normals = np.einsum("kij,knj->kni", rotations, gradient.reshape(*shape, 3))
        partner = transport["candidates"][item["candidate_index"]]["witnesses"][pair["partner"]]
        lengths = np.linalg.norm(normals, axis=-1)
        dots = np.divide(normals @ partner["normal_world"], lengths,
                         out=np.full(shape, np.nan), where=lengths > 0.)
        rows, stations = opposing_thumb_witnesses(gap, dots, allowed, partner_node=pair["partner_node"])
        counts = Counter()
        attempts = []
        for index, station in zip(rows, stations, strict=True):
            hull = middle[index] @ orientation.T + pose[:3]
            clearance = middle_link_clearance(hull, nodes, radius)
            counts[clearance["status"]] += 1
            if clearance["status"] != "clear":
                continue
            option = dict(pair, thumb_joints_rad=settings[index].tolist(), thumb_node=int(station),
                          opposition_cos=float(dots[index, station]))
            abduct, close = staged_joints(active, option)
            closed = endpoint(model, pose, close, nodes, scene, grids["base_link"],
                              targets=(THUMB, pair["partner"]))
            opening = endpoint(model, pose, abduct, nodes, scene, grids["base_link"])
            record = dict(pair=option, middle_clearance=clearance, thumb_gap_m=float(gap[index, station]),
                          closing=closed, abduction=opening)
            if closed["passed"]:
                thumb_first = active.copy()
                thumb_first[:2] = close[:2]
                first = endpoint(model, pose, thumb_first, nodes, scene, grids["base_link"], targets=(THUMB,))
                record["thumb_first"] = first
                routes = {}
                for mode, target_q, intermediate in (
                    ("thumb_abduction", abduct, opening), ("thumb_first", thumb_first, first)
                ):
                    if intermediate["passed"]:
                        routes[mode] = sample_route(
                            model, source, pose, active, target_q, close, option, side, row, nodes,
                            scene, grids["base_link"], other_bounds, mode=mode)
                record["routes"] = routes
                if first["passed"]:
                    retreat = (.0015 * np.asarray(palm_anchor(row["rod"], nodes, side)["normal_world"])
                               if side == "left" else None)
                    flex_exponent = 2. if side == "right" else 1.
                    detour = sample_route(
                        model, source, pose, active, thumb_first, close, option, side, row, nodes,
                        scene, grids["base_link"], other_bounds, mode="thumb_first",
                        retreat_world_m=retreat, flex_exponent=flex_exponent, intervals=32)
                    record["thumb_first_detour"] = dict(
                        retreat_world_m=None if retreat is None else retreat.tolist(),
                        flex_exponent=flex_exponent, stages=detour,
                        sampled_route_passed=all(stage["failed"] == 0 for stage in detour.values()))
            attempts.append(record)
        proposals.append(dict(candidate_index=item["candidate_index"], wrist_wxyz_m=pose.tolist(),
                              source_q_rad=active.tolist(), clearance_counts=dict(counts), attempts=attempts))
    return dict(angles=len(settings), source_sdf_sha256=hashes[THUMB], proposals=proposals)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("palm-receipt", "partial-audit", "transport-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    readiness = json.loads((project / READINESS).read_bytes())["endpoint_audit"]
    partial = checked_json(args.partial_audit, readiness["artifact_sha256"])
    transport = checked_json(args.transport_audit, readiness["transport_sha256"])
    row, scene, archive = load_measured_palm_boundary(args.palm_receipt, project=project)
    clock, nodes, _ = measured_palm_boundary(row)
    if any(audit["clock"] != clock or audit["archive"] != archive for audit in (partial, transport)):
        raise ValueError("audits must bind the same archived exchange")
    outer = json.loads(args.palm_receipt.read_bytes())
    plan = json.loads(args.palm_receipt.with_name("plan.json").read_bytes())
    if digest(plan) != outer["frozen_sha256"]:
        raise ValueError("native frozen plan required for original thumb SDFs")
    hands = {}
    for side, other in (("left", "right"), ("right", "left")):
        opposite = partial["hands"][other]["source_boundary"]["open_hull_guards"]
        hands[side] = audit_side(row, scene, nodes, side, partial["hands"][side], transport["hands"][side],
                                 plan["frozen_files"], project, opposite["hand_y_bounds_m"])
    result = dict(version="dh116-power-thumb-clearance-v1", archive=archive, clock=clock,
                  wire_diameter_m=2*scene["wire_radius_m"], hand_results=hands,
                  partial_sha256=readiness["artifact_sha256"], transport_sha256=readiness["transport_sha256"],
                  continuous_route_verified=False, native_runs=0, teacher_admission=False,
                  grasp_qualified=False, generalization_qualified=False)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
