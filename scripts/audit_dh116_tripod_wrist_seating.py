"""Screen left-wrist World XY seating on recorded Tripod contact endpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import CONTRACT as WRIST
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from go2_pvcnn.control.dh116.tabletop_training_reward import CONTROL_HZ


PADS = ("finger14_link", "finger23_link", "finger33_link")
FIRST_SDK, LAST_SDK = 1801, 2056


def choose_offset(offsets, gaps, normals, *, max_depth_m):
    """Prefer the shortest move giving geometry on every measured endpoint."""
    offsets = np.asarray(offsets, dtype=float)
    gaps = np.asarray(gaps, dtype=float)
    normals = np.asarray(normals, dtype=float)
    if (gaps.ndim != 3 or gaps.shape[2] != 3 or gaps.shape[1] < 1
            or offsets.shape != (len(gaps), 3) or normals.shape != (*gaps.shape, 3)
            or not all(np.isfinite(a).all() for a in (offsets, gaps, normals))
            or not math.isfinite(max_depth_m) or max_depth_m <= 0.):
        raise ValueError("finite measured Tripod candidate gaps, normals and original depth required")
    thumb, index, middle = np.moveaxis(normals, 2, 0)
    axis = thumb - (index + middle) / 2.
    length = np.linalg.norm(axis, axis=-1)
    axis /= np.maximum(length, 1e-12)[..., None]
    dots = lambda a, b: np.einsum("...i,...i->...", a, b)
    opposed = ((dots(thumb, index) <= -.25) & (dots(thumb, middle) <= -.25)
               & (dots(thumb, axis) >= .2) & (dots(index, axis) <= -.2) & (dots(middle, axis) <= -.2))
    geometry = (gaps < 0.).all(-1) & (gaps >= -max_depth_m).all(-1) & opposed
    valid = np.flatnonzero(geometry.all(-1))
    best = int(valid[np.linalg.norm(offsets[valid], axis=-1).argmin()]) if len(valid) else None
    return best, geometry.all(-1)


def world_xy_dose(offset):
    """Emit finite World XY increments through the existing 24D action path."""
    offset = np.asarray(offset, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all() or offset[2] != 0.:
        raise ValueError("finite left World XY seating shift required")
    distance_per_control = ACTION_CONTRACT["translation_rate_m_s"] / CONTROL_HZ
    count = math.ceil(np.linalg.norm(offset) / distance_per_control)
    actions = np.zeros((count, ACTION_CONTRACT["size"]))
    if count:
        left = ACTION_CONTRACT["slices"]["world_translation_velocity"][0]
        actions[:, left:left+2] = offset[:2] / (count * distance_per_control)
    return actions


def _read_cohort(run, proof_sha256):
    raw = checked_file(run / "post_timeout_observation.json", proof_sha256)
    inventory = json.loads(raw)["artifact_sha256"]

    def read(name):
        return decode_owned(json.loads(checked_file(run / "native" / name, inventory[name])))

    plan = read("frozen_plan.json")
    depth = plan["right_tripod_distal_contact"]["max_pad_penetration_m"]
    rows = []
    for tick in range(FIRST_SDK, LAST_SDK+1):
        row = read(f"tripod_control_{tick}.json")
        if (row["physics_step"] != tick or row["rod"]["physics_step"] != tick
                or row["sdk_hand_geometry"]["physics_step"] != tick
                or row["rod"]["rod_tick"] != 12*(tick-2) or row["guard_failure"] is not None):
            raise ValueError("same-clock original Tripod rod and hand geometry required")
        rows.append(row)
    geometry = read("tripod_reset_source_2.json")["evidence"]
    radius = geometry["rod_radius_m"]
    top = geometry["table_center_m"][2] + .5*geometry["table_size_m"][2]
    supports = read("tripod_support_scene.json")["supports"]
    return rows, depth, radius, top, supports


def _sample_pad(grid, nodes, poses, offsets, radius):
    from scripts.prepare_dh116_pinch_closure_margin import sample_grid

    xyz, xyzw = [np.asarray(a) for a in zip(*poses, strict=True)]
    rotation = Rotation.from_quat(xyzw).as_matrix()
    local = np.einsum("tni,tij->tnj", nodes - xyz[:, None], rotation)
    local_shift = np.einsum("ci,tij->ctj", offsets, rotation)
    gaps, normals = [], []
    for shift in np.array_split(local_shift, math.ceil(len(offsets) / 32)):
        points = local[None] - shift[:, :, None]
        shape = points.shape[:-1]
        phi, gradient, valid = sample_grid(grid, points.reshape(-1, 3), dtype=np.float64)
        lattice = np.where(valid, phi, np.inf).reshape(shape)
        min_index = lattice.argmin(axis=-1)[..., None]
        minimum = np.take_along_axis(lattice, min_index, axis=2)[..., 0]
        normal = np.take_along_axis(gradient.reshape(*shape, 3), min_index[..., None], axis=2)[:, :, 0]
        norm = np.linalg.norm(normal, axis=-1)
        if not np.isfinite(minimum).all() or not np.isfinite(norm).all() or (norm <= 0.).any():
            raise ValueError("locked SDF witness missing in an archived left material half")
        gaps.append(minimum - radius)
        normals.append(np.einsum("cti,tji->ctj", normal / norm[..., None], rotation))
    return np.concatenate(gaps), np.concatenate(normals)


def audit(run, proof_sha256, *, project):
    """Produce a kinematic candidate with archived support and original bounds."""
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import load_sdf_arrays, prepare_scene
    from scripts.screen_dh116_tripod_measured_precontact import archived_tube, measure_side

    rows, max_depth, radius, table_top, supports = _read_cohort(run, proof_sha256)
    nodes = np.stack([r["rod"]["node_positions"][0] for r in rows]).astype(float)
    x = np.arange(-10, 11) * .000025
    xy = np.stack(np.meshgrid(x, x, indexing="ij"), axis=-1).reshape(-1, 2)
    offsets = np.column_stack((xy, np.zeros(len(xy))))
    grids = load_sdf_arrays(prepare_scene(project))
    gaps, normals = [], []
    for link in PADS:
        name = "left." + link
        samples = _sample_pad(grids[name], nodes[:, 21:],
                              [r["sdk_hand_geometry"]["poses"][name] for r in rows], offsets, radius)
        gaps.append(samples[0])
        normals.append(samples[1])
    gaps = np.stack(gaps, axis=-1)
    normals = np.stack(normals, axis=2)
    chosen, feasible = choose_offset(offsets, gaps, normals, max_depth_m=max_depth)
    zero = len(offsets) // 2
    issued_z = rows[-1]["wrist_projection"]["palm_offsets_m"][0, :, 2]
    z_headroom = float(WRIST["palm_upper_m"][2] - max(issued_z))
    result = dict(version="dh116-tripod-left-wrist-seating-screen-v1", proof_sha256=proof_sha256,
                  archived_sdk_range=[FIRST_SDK, LAST_SDK], samples=len(rows), candidates=len(offsets),
                  feasible_geometry_offsets=int(feasible.sum()), pad_names=PADS,
                  original_max_pad_depth_m=max_depth, baseline_pad_max_gap_m=gaps[zero].max(0).tolist(),
                  synchronized_wrist_z_headroom_m=z_headroom, candidate=None,
                  native_contact_qualified=False, grasp_qualified=False, teacher_admission=False)
    if chosen is None:
        return result
    anchor = rows[-1]
    xyz, quat = anchor["sdk_hand_geometry"]["poses"]["left.base_link"]
    hand = anchor["action_readback"]["post"]["hands"]["left"]
    indices = hand["active_indices"]
    if [hand["joint_names"][i] for i in indices] != list(ACTIVE):
        raise ValueError("original left active joint order required")
    unit_xyzw = Rotation.from_quat(quat).as_quat()
    target = dict(pose_wxyz=np.r_[np.asarray(xyz) + offsets[chosen], np.roll(unit_xyzw, 1)],
                  active_q_rad=hand["q"][0, indices])
    boxes = [(np.asarray(box["center_m"]) - np.asarray(box["size_m"]) / 2,
              np.asarray(box["center_m"]) + np.asarray(box["size_m"]) / 2) for box in supports]
    route = measure_side("left", anchor, nodes[-1], archived_tube(nodes), boxes, grids,
                         target, table_top, pads=PADS, contact_pads=PADS)
    candidate = dict(offset_world_m=offsets[chosen].tolist(),
                     pad_min_gap_m=gaps[chosen].min(0).tolist(),
                     pad_max_gap_m=gaps[chosen].max(0).tolist(),
                     minimum_table_gap_m=min(row["table_clearance_m"] for row in route),
                     minimum_support_gap_m=min(row["support_clearance_m"] for row in route),
                     minimum_noncontact_rod_gap_m=min(row["noncontact_rod_clearance_m"] for row in route),
                     minimum_contact_hull_gap_lower_bound_m=min(row["contact_rod_clearance_m"] for row in route),
                     minimum_path_pad_sdf_gap_m=min(min(row["pad_gaps_m"]) for row in route))
    # The original supported Tripod diagnostic applies source envelopes to
    # noncontact links and the locked SDF limit to selected pads.
    candidate["route_screen_passed"] = (
        candidate["minimum_table_gap_m"] >= CLEARANCE_M
        and candidate["minimum_support_gap_m"] > 1e-10
        and candidate["minimum_noncontact_rod_gap_m"] > .00025
        and candidate["minimum_path_pad_sdf_gap_m"] >= -max_depth)
    candidate["proposed_action24"] = world_xy_dose(offsets[chosen]).tolist()
    result["candidate"] = candidate
    return result


def seating_schedule(actions, recorded_prefix, offset):
    """Append one bounded translation and 128 SDK steps of unchanged targets."""
    actions = np.asarray(actions, dtype=float)
    if (actions.ndim != 2 or actions.shape[1] != 24 or not np.isfinite(actions).all()
            or (np.abs(actions) > 1.).any() or not np.array_equal(actions, recorded_prefix)):
        raise ValueError("exact original Tripod action prefix required")
    dose = world_xy_dose(offset)
    if not 1 <= len(dose) <= 3:
        raise ValueError("screened left wrist seating requires at most three World XY increments")
    return np.vstack((actions, dose, np.zeros((16, 24))))


def load_seating_extension(run, proof_sha256, actions, *, start_sdk, project):
    """Bind the proposed dose to the actual archived prefix and terminal sample."""
    if start_sdk + 8 * len(actions) != LAST_SDK:
        raise ValueError("seating must follow the complete SDK2056 replay")
    # USD mesh reads belong in a separate CPU interpreter. The subsequent
    # AppLauncher process must start without a pre-imported USD runtime.
    with TemporaryDirectory(prefix="dh116-tripod-seat-screen-") as directory:
        result_path = Path(directory) / "screen.json"
        result = subprocess.run([sys.executable, "-B", "-m", "scripts.audit_dh116_tripod_wrist_seating",
            "--run", str(run.resolve()), "--proof-sha256", proof_sha256, "--output", str(result_path)],
            cwd=project, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError("left wrist seating CPU screen rejected: " + result.stderr[-1800:])
        screen = json.loads(result_path.read_bytes())
    candidate = screen["candidate"]
    if candidate is None or candidate["route_screen_passed"] is not True:
        raise ValueError("whole-hand route screen rejected the lateral seating proposal")
    proof = run / "post_timeout_observation.json"
    inventory = json.loads(checked_file(proof, proof_sha256))["artifact_sha256"]
    plan_path = run / "native/frozen_plan.json"
    plan = json.loads(checked_file(plan_path, inventory["frozen_plan.json"]))
    anchor_path = run / f"native/tripod_control_{LAST_SDK}.json"
    anchor = json.loads(checked_file(anchor_path, inventory[anchor_path.name]))
    extended = seating_schedule(actions, plan["tripod_approach"], candidate["offset_world_m"])
    dose_end_sdk = LAST_SDK + 8 * (len(extended) - len(actions) - 16)
    stage = dict(anchor_sdk=LAST_SDK, dose_end_sdk=dose_end_sdk,
                 hold_end_sdk=start_sdk + 8 * len(extended), anchor=anchor, screen=screen)
    names = [f"tripod_control_{tick}.json" for tick in range(FIRST_SDK, LAST_SDK + 1)]
    names += ["frozen_plan.json", "tripod_support_scene.json", "tripod_reset_source_2.json"]
    return extended, stage, (proof, *(run / "native" / name for name in names), Path(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--proof-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.proof_sha256, project=Path(__file__).resolve().parents[1])
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())
    return 0 if result["candidate"] and result["candidate"]["route_screen_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
