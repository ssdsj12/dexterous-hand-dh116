
"""Screen Power wrist re-seating on the archived dual-palm contact boundary."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import CONTRACT as WRIST
from scripts.audit_dh116_power_fixed_wrist_digits import (
    DISTAL_PAD_LINKS, active_limits, align_measured_wrist, digest as file_sha,
    fk_poses, load_geometry, load_measured_palm_boundary, measured_palm_boundary, shapes_world,
)
from scripts.diagnose_dh116_teacher_initialization import digest
from scripts.prepare_dh116_pinch_closure_margin import sample_grid
from scripts.audit_dh116_pinch_initial_pose import box_relation
from scripts.prepare_dh116_power_palm_seed import thumb_clearance
from scripts.verify_dh116_power_staged_palm_witness import MAX_PALM_PENETRATION_M

ANGLES_DEG = range(-20, 21, 5)
SLIDES_M = (-.008, -.004, 0., .004, .008)
REACH_M = .0005
PAD_PENETRATION_M = .00015
PALM_ALIGNMENT_COS = .85
PALM_RESEAT_LIMIT_M = .002
PALM_RESIDUAL_M = 5e-6


def palm_anchor(rod, nodes, side):
    name = f"{side}.base_link"
    pairs = [pair for pair in rod["rod_mesh_contact_pairs"]["pairs"]
             if pair["mesh_name"] == name and pair["normal_force_on_rod_n"] > 0.]
    stations = {pair["node_index"] for pair in pairs}
    if len(stations) != 1 or {pair["substep"] for pair in pairs} != set(range(12)):
        raise ValueError("one same-boundary palm node supported by all solver substeps required")
    station, = stations
    if not (20 < station < 40 if side == "left" else 0 < station < 20):
        raise ValueError("interior palm node on its assigned wire half required")
    weights = [pair["normal_force_on_rod_n"] for pair in pairs]
    pivot = np.average([pair["contact_point_world_m"] for pair in pairs], axis=0, weights=weights)
    normal = np.average([pair["outward_normal_world"] for pair in pairs], axis=0, weights=weights)
    tangent = nodes[station+1] - nodes[station-1]
    axes = (normal, tangent, np.cross(tangent, normal))
    norms = [np.linalg.norm(axis) for axis in axes]
    if min(norms) <= 1e-10 or not np.isfinite([*pivot, *norms]).all():
        raise ValueError("finite nondegenerate measured palm frame required")
    normal, tangent, transverse = (axis / norm for axis, norm in zip(axes, norms, strict=True))
    return dict(node_id=station, pivot_world_m=pivot, normal_world=normal,
                tangent_world=tangent, transverse_world=transverse)


def pick_pad_witness(gaps, normal_dots, allowed, *, thumb):
    """Keep a near same-side node only on a pose compatible with the full rod."""
    safe_pose = ~np.any(gaps < -PAD_PENETRATION_M, axis=1)
    mask = np.isfinite(gaps) & allowed[None] & safe_pose[:, None] & (
        np.isfinite(normal_dots) if thumb else normal_dots < 0.)
    if not mask.any():
        return None
    score = np.where(mask, np.abs(gaps), np.inf)
    return np.unravel_index(np.argmin(score), score.shape)


def _locked_grids(rod, side, files):
    grids, hashes = {}, {}
    for link in ("base_link", *DISTAL_PAD_LINKS):
        name = f"{side}.{link}"
        expected = rod["mesh_bindings"][name]["sdf_sha256"]
        paths = [Path(path) for path, sha in files.items()
                 if sha == expected and Path(path).name == f"{name}.npz"]
        if len(paths) != 1 or file_sha(paths[0]) != expected:
            raise ValueError(f"{name} must use its archived registered SDF")
        with np.load(paths[0], allow_pickle=False) as data:
            grids[link] = {key: data[key].copy() for key in ("origin", "spacing", "values")}
        hashes[link] = expected
    return grids, hashes


def _finger_samples(geometry, wrist, active, offsets):
    lower, upper = active_limits()
    rotation = Rotation.from_quat(wrist[[4, 5, 6, 3]]).as_matrix()
    results = {}
    for index, pad in enumerate(DISTAL_PAD_LINKS, 1):
        angles = np.unique(np.r_[active[index], np.linspace(lower[index], upper[index], 45)])
        abductions = np.unique(np.r_[active[0], np.linspace(lower[0], upper[0], 13)]) if index == 1 else [active[0]]
        matrices, choices = [], []
        for angle, abduction in product(angles, abductions):
            q = active.copy()
            q[index], q[0] = angle, abduction
            fk = fk_poses(geometry, wrist, q)[pad] @ offsets[pad]
            matrices.append(np.c_[rotation.T @ fk[:3, :3], rotation.T @ (fk[:3, 3] - wrist[:3])])
            choices.append([abduction, angle])
        results[pad] = np.stack(matrices), np.asarray(choices)
    return results


def _palm_candidate(wrist, anchor, nodes, grid, radius, rotation, slide):
    pivot, normal, tangent = (anchor[key] for key in ("pivot_world_m", "normal_world", "tangent_world"))
    root = pivot + rotation.apply(wrist[:3] - pivot) + slide * tangent
    provisional = root.copy()
    orient = rotation.as_matrix() @ Rotation.from_quat(wrist[[4, 5, 6, 3]]).as_matrix()
    station = anchor["node_id"]
    point = nodes[station:station+1]
    for _ in range(4):
        phi, gradient, valid = sample_grid(grid, (point-root) @ orient, dtype=np.float64)
        if not valid[0] or np.linalg.norm(gradient[0]) < .5:
            return None
        error = float(phi[0] - radius)
        if abs(error) < PALM_RESIDUAL_M:
            break
        world = orient @ gradient[0]
        root += error * world / float(world @ world)
    move = float(np.linalg.norm(root - provisional))
    if move > PALM_RESEAT_LIMIT_M:
        return None
    phi, gradient, valid = sample_grid(grid, (nodes-root) @ orient, dtype=np.float64)
    gaps = phi - radius
    if (not valid[station] or gaps[station] > REACH_M
            or np.any(gaps < -MAX_PALM_PENETRATION_M)):
        return None
    direction = orient @ gradient[station]
    norm = np.linalg.norm(direction)
    if norm < .5 or direction @ normal / norm < PALM_ALIGNMENT_COS:
        return None
    return root, orient, direction / norm, float(gaps[station]), move


def _support_clearance_lower(hulls, supports):
    """Use certified world-axis separation first and the original SAT near contact."""
    span = [(vertices, vertices.min(axis=0), vertices.max(axis=0)) for vertices in hulls.values()]
    gap = float("inf")
    for support in supports:
        center, size = np.asarray(support["center_m"]), np.asarray(support["size_m"])
        low, high = center - size / 2, center + size / 2
        for vertices, bottom, top in span:
            bound = float(np.maximum(low - top, bottom - high).max())
            if bound < CLEARANCE_M:
                bound = box_relation(vertices, center, size)["max_separating_axis_gap_m"]
            gap = min(gap, bound)
    return gap


def _wrist_clearance(geometry, wrist, q, scene):
    hulls = shapes_world(geometry, wrist, q)
    table = min(float(points[:, 2].min()) for points in hulls.values()) - scene["table_top_m"]
    support = _support_clearance_lower(hulls, scene["supports"])
    inter_digit = thumb_clearance(hulls)
    return dict(table_m=table, support_m=support, thumb_clearance_m=inter_digit,
                passed=bool(table >= CLEARANCE_M and support >= CLEARANCE_M and inter_digit > 0.))


def screen_side(row, nodes, scene, side, files, project):
    geometry = load_geometry(project, side=side)
    wrist, active, correction, offsets = align_measured_wrist(geometry, side, row, apply_to_geometry=True)
    anchor = palm_anchor(row["rod"], nodes, side)
    grids, hashes = _locked_grids(row["rod"], side, files)
    alternatives = _finger_samples(geometry, wrist, active, offsets)
    # The arm is projected in the original Reset frame, including its wrist angles.
    side_id = ("left", "right").index(side)
    projection = row["wrist_projection"]
    target = projection["projected_root_poses"][0, side_id]
    reset_pos = target[:3] - projection["palm_offsets_m"][0, side_id]
    reset_rot = Rotation.from_quat(target[[4, 5, 6, 3]]) * Rotation.from_euler(
        "XYZ", projection["projected_angles"][0, side_id]).inv()
    allowed = np.arange(len(nodes)) > 20 if side == "left" else np.arange(len(nodes)) < 20
    radius = scene["wire_radius_m"]
    rows = []
    rejected = dict(palm=0, action_bounds=0, open_geometry=0, combined_pose=0)
    for roll, yaw, pitch, slide in product(ANGLES_DEG, ANGLES_DEG, ANGLES_DEG, SLIDES_M):
        tilt = Rotation.from_rotvec(np.deg2rad(roll)*anchor["tangent_world"]) * Rotation.from_rotvec(
            np.deg2rad(yaw)*anchor["normal_world"]) * Rotation.from_rotvec(
            np.deg2rad(pitch)*anchor["transverse_world"])
        palm = _palm_candidate(wrist, anchor, nodes, grids["base_link"], radius, tilt, slide)
        if palm is None:
            rejected["palm"] += 1
            continue
        root, orient, palm_normal, palm_gap, correction_m = palm
        offset = root - reset_pos
        attitude = Rotation.from_matrix(orient)
        angles = (reset_rot.inv()*attitude).as_euler("XYZ")
        if (np.any(offset < WRIST["palm_lower_m"]) or np.any(offset > WRIST["palm_upper_m"])
                or np.any(np.abs(angles) > np.pi/2)):
            rejected["action_bounds"] += 1
            continue
        pose = np.r_[root, attitude.as_quat()[[3, 0, 1, 2]]]
        guards = _wrist_clearance(geometry, pose, active, scene)
        if not guards["passed"]:
            rejected["open_geometry"] += 1
            continue
        witnesses = {}
        for pad, (transforms, joint_values) in alternatives.items():
            rotations = orient @ transforms[:, :, :3]
            xyz = transforms[:, :, 3] @ orient.T + root
            local = np.einsum("knj,kji->kni", nodes[None]-xyz[:, None], rotations)
            phi, gradients, _ = sample_grid(grids[pad], local.reshape(-1, 3), dtype=np.float64)
            gaps = phi.reshape(len(xyz), -1) - radius
            direction = np.einsum("kij,knj->kni", rotations, gradients.reshape(len(xyz), -1, 3))
            with np.errstate(invalid="ignore", divide="ignore"):
                dots = direction @ palm_normal / np.linalg.norm(direction, axis=-1)
            witness = pick_pad_witness(gaps, dots, allowed, thumb=(pad == DISTAL_PAD_LINKS[0]))
            if witness is not None:
                i, j = witness
                unit = direction[i, j] / np.linalg.norm(direction[i, j])
                witnesses[pad] = dict(gap_m=float(gaps[i,j]), node_id=int(j),
                                      normal_world=unit.tolist(), joints_rad=joint_values[i].tolist())
        near = [pad for pad, gap in witnesses.items() if abs(gap["gap_m"]) <= REACH_M]
        combined = None
        if len(near) == len(DISTAL_PAD_LINKS):
            q = np.r_[witnesses[DISTAL_PAD_LINKS[0]]["joints_rad"][0],
                      [witnesses[pad]["joints_rad"][1] for pad in DISTAL_PAD_LINKS]]
            combined = _wrist_clearance(geometry, pose, q, scene)
            thumb = np.asarray(witnesses[DISTAL_PAD_LINKS[0]]["normal_world"])
            opposed = any(float(thumb @ witnesses[pad]["normal_world"]) < -.2 for pad in DISTAL_PAD_LINKS[1:])
            combined.update(q_rad=q.tolist(), thumb_opposed=bool(opposed))
            if not combined["passed"] or not opposed:
                rejected["combined_pose"] += 1
        gap_order = [abs(gap["gap_m"]) for gap in witnesses.values()]
        rows.append(dict(roll_deg=roll, yaw_deg=yaw, pitch_deg=pitch, slide_m=slide,
                         position_correction_m=correction_m, palm_gap_m=palm_gap, wrist_wxyz_m=pose.tolist(),
                         open_endpoint=guards, independent_near_pads=near, witnesses=witnesses,
                         combined_endpoint=combined,
                         rank=[-len(near), 5-len(gap_order), max(gap_order,default=1.), sum(gap_order)]))
    return dict(anchor={k: v.tolist() if isinstance(v,np.ndarray) else v for k,v in anchor.items()},
                pad_pose_alignment=correction, sdf_sha256=hashes, rejections=rejected,
                enumerated=len(ANGLES_DEG)**3 * len(SLIDES_M), candidates=sorted(rows,key=lambda item:item["rank"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palm-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    row, scene, sources = load_measured_palm_boundary(args.palm_receipt, project=project)
    clock, nodes, _ = measured_palm_boundary(row)
    outer = json.loads(args.palm_receipt.read_bytes())
    plan = json.loads(args.palm_receipt.with_name("plan.json").read_bytes())
    if digest(plan) != outer["frozen_sha256"]:
        raise ValueError("expected native frozen plan for the SDF source binding")
    hands = {side: screen_side(row, nodes, scene, side, plan["frozen_files"], project)
             for side in ("left", "right")}
    result = dict(version="dh116-power-measured-reseat-screen-v1", archive=sources, clock=clock,
                  hands=hands, continuous_route_verified=False, native_action_performed=False,
                  teacher_admission=False, grasp_qualified=False)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
