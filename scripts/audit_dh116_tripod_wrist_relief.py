"""Screen a left wrist relief seed on receipt-bound Tripod contact losses."""

import argparse
from collections import Counter
import io
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import VERSION as READBACK_VERSION
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_joint_target_readback import BACKEND_SOURCE
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import (
    IDENTITY_FIELDS, MIN_LINK_LOAD_N, MAX_OPPOSITION_COS, measure_supported_distal_contact,
)
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from go2_pvcnn.control.dh116.tabletop_training_reward import CONTROL_HZ
from scripts.audit_dh116_tripod_index_contact_loss import _contact_row, _tracking, _validated_pose
from scripts.prepare_dh116_pinch_closure_margin import sample_grid
from scripts.verify_dh116_supported_distal_transfer import check_reaction_delivery


LINKS = ("left.finger14_link", "left.finger23_link", "left.finger33_link")
THUMB, INDEX, MIDDLE = LINKS
LINEAR_STEP = ACTION_CONTRACT["translation_rate_m_s"] / CONTROL_HZ
ROTATION_STEP = ACTION_CONTRACT["rotation_rate_rad_s"] / CONTROL_HZ
SINGLE_SDK_FRACTION = CONTROL_HZ / 480


def _pose_evidence(nodes, poses, grids, witness_nodes, radius, *, pivot, translation, rotation):
    rigid = Rotation.from_rotvec(rotation)
    gaps, normals, all_minimums = [], [], []
    for link in LINKS:
        position, orientation = _validated_pose(poses[link])
        # Transform the entire hand rigidly about its measured palm.
        position = pivot + translation + rigid.apply(position - pivot)
        orientation = rigid * orientation
        local = orientation.inv().apply(nodes - position)
        phi, gradient, valid = sample_grid(grids[link], local, dtype=np.float64)
        node = witness_nodes[link]
        if not valid[node]:
            return None
        gaps.append(float(phi[node] - radius))
        normals.append(orientation.apply(gradient[node]))
        all_minimums.append(float(phi[valid].min() - radius))
    if np.any(np.linalg.norm(normals, axis=1) <= 0.):
        return None
    return np.array(gaps), np.array(normals), np.array(all_minimums)


def _opposed(normals):
    thumb, index, middle = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    axis = thumb - .5 * (index + middle)
    length = np.linalg.norm(axis)
    if length <= 0.:
        return False
    axis /= length
    return (thumb @ index <= MAX_OPPOSITION_COS and thumb @ middle <= MAX_OPPOSITION_COS
            and thumb @ axis >= .2 and index @ axis <= -.2 and middle @ axis <= -.2)


def propose_wrist_relief(*, palm, poses, grids, nodes, witness_nodes, radius, max_depth, old_index_gap):
    """Find a bounded sampled rigid pose seed with finger positions held."""
    pivot, _ = _validated_pose(palm)
    nodes = np.asarray(nodes, dtype=float)
    if (nodes.ndim != 2 or nodes.shape[1] != 3 or not np.isfinite(nodes).all()
            or set(witness_nodes) != set(LINKS)
            or any(type(i) is not int or not 0 <= i < len(nodes) for i in witness_nodes.values())
            or set(grids) != set(LINKS) or not np.isfinite(radius) or radius <= 0.
            or not np.isfinite(max_depth) or not 0. < max_depth <= .00025 or not np.isfinite(old_index_gap)):
        raise ValueError("finite, locked three-pad sampled pose source required")

    def evidence(translation, rotation):
        return _pose_evidence(nodes, poses, grids, witness_nodes, radius, pivot=pivot,
                              translation=translation, rotation=rotation)

    nominal = evidence(np.zeros(3), np.zeros(3))
    if nominal is None:
        return dict(status="missing_pad_sdf_witness")
    gap, normals, minimums = nominal
    result = dict(gap_at_start_m=gap.tolist(), old_index_gap_m=old_index_gap)
    if gap[1] <= 0. or old_index_gap >= 0.:
        return dict(result, status="no_index_gap_to_recover")
    if not ((gap[[0, 2]] < 0.).all() and (minimums >= -max_depth).all()):
        return dict(result, status="protected_contact_or_depth_already_invalid")

    points = nodes[[witness_nodes[link] for link in LINKS]]
    # Differentiate the SDF of fixed rod nodes under a rigid left-hand twist.
    jac = -np.hstack((LINEAR_STEP * normals,
                      ROTATION_STEP * np.cross(points - pivot, normals)))
    keep = jac[[0, 2]]
    index_change = jac[1]
    direction = keep.T @ np.linalg.lstsq(keep.T, index_change, rcond=None)[0] - index_change
    norm = max(np.linalg.norm(direction[:3]), np.linalg.norm(direction[3:]))
    if norm <= np.finfo(float).eps * np.linalg.norm(index_change):
        return dict(result, status="index_direction_conflicts_with_protected_pads")
    translation, rotation = np.split(direction / norm, [3])
    translation *= LINEAR_STEP
    rotation *= ROTATION_STEP
    single_sdk = evidence(SINGLE_SDK_FRACTION * translation, SINGLE_SDK_FRACTION * rotation)
    result["gap_after_one_sdk_rate_bound_m"] = None if single_sdk is None else float(single_sdk[0][1])
    closest = max(old_index_gap, -max_depth)
    hit_index = False
    best_index_gap = float(gap[1])
    stopping_guard = None
    for fraction in np.linspace(1. / 32, 1., 32):
        trial = evidence(fraction * translation, fraction * rotation)
        if trial is None:
            stopping_guard = "sample_left_sdf_lattice"
            break
        trial_gap, trial_normals, trial_mins = trial
        if not ((trial_gap[[0, 2]] < 0.).all() and (trial_mins >= -max_depth).all()):
            stopping_guard = "protected_sdf_gap_or_depth"
            break
        best_index_gap = min(best_index_gap, float(trial_gap[1]))
        hit_index |= bool(trial_gap[1] <= closest)
        if trial_gap[1] <= closest and _opposed(trial_normals):
            inverse = evidence(-fraction * translation, -fraction * rotation)
            return dict(result, status="sampled_pose_candidate",
                        world_translation_m=(fraction * translation).tolist(),
                        world_rotation_vector_rad=(fraction * rotation).tolist(),
                        gap_at_end_m=trial_gap.tolist(),
                        opposite_gap_at_end_m=None if inverse is None else inverse[0].tolist(),
                        sample_fraction=float(fraction))
    return dict(result, status="bounded_direction_did_not_restore_geometry",
                best_index_gap_m=best_index_gap, restored_old_index_gap=hit_index, first_rejected_sample=stopping_guard)


def _check_readback_clock(row, when, rod):
    readback = row["action_readback"]
    snapshot = readback[when]["identity"]
    if len(snapshot) != 1:
        raise ValueError("one same-clock articulation readback required")
    clock = snapshot[0]
    if (readback["version"] != READBACK_VERSION or readback["env_ids"] != [rod["env_id"]]
            or clock["clock_valid"] is not True or clock["policy_step"] != row["policy_step"]
            or clock["substep"] != row["substep"]
            or any(clock[key] != rod[key] for key in (*IDENTITY_FIELDS, "timestamp_s", "solver_time_s"))):
        raise ValueError("same-clock PhysX readback must agree with the archived rod")


def _archive_sample(before, after, radius, max_depth, grids):
    old_rod, rod = before["rod"], after["rod"]
    if (any(old_rod[key] != rod[key] for key in IDENTITY_FIELDS[:-2])
            or not np.array_equal(old_rod["rest_lengths_m"], rod["rest_lengths_m"])
            or not check_reaction_delivery(before, after)):
        raise ValueError("loss pair must deliver the same-episode preceding measured reaction")
    _check_readback_clock(before, "post", old_rod)
    _check_readback_clock(after, "pre", old_rod)
    _check_readback_clock(after, "post", rod)
    for side in ("left", "right"):
        old = before["action_readback"]["post"]["hands"][side]
        pre = after["action_readback"]["pre"]["hands"][side]
        new = after["action_readback"]["post"]["hands"][side]
        if (old["joint_names"] != pre["joint_names"] or new["joint_names"] != pre["joint_names"]
                or any(hand["backend_source"] != BACKEND_SOURCE
                       or hand["measured_source"] != "physx_backend_get_dof_positions/get_dof_velocities"
                       for hand in (old, pre, new))
                or any(not np.array_equal(old[k], pre[k]) for k in ("q", "qd", "backend_targets"))):
            raise ValueError("same-clock measured joint source changed across the loss boundary")
        wrist_ids = [i for i, name in enumerate(new["joint_names"]) if name.startswith("wrist_")]
        if len(wrist_ids) != 6 or not np.array_equal(pre["backend_targets"][:, wrist_ids],
                                                   new["backend_targets"][:, wrist_ids]):
            raise ValueError("archived loss must keep the original wrist targets fixed")

    first = measure_supported_distal_contact(old_rod, grasp="tripod")
    current = measure_supported_distal_contact(rod, grasp="tripod")
    if (not first["bilateral"] or current["wrong_side_contacts"] or current["unexpected_hand_contacts"]):
        raise ValueError("correct-side Tripod qualification must precede an unambiguous loss")
    tracks = _tracking(after)["joints"]
    minimums = {pad: min(row["per_pad"][pad]["normal_load_n"]
                         for row in current["per_hand"]["left"]["per_substep"])
                for pad in ("thumb", "index", "middle")}
    if minimums["index"] > MIN_LINK_LOAD_N:
        raise ValueError("left index did not lose required normal load")
    result = dict(sdk=after["physics_step"], minimum_left_normal_load_n=minimums,
                  right_loaded=current["per_hand"]["right"]["loaded"],
                  signed_index_target_lead_rad=tracks["finger21"]["backend_target"] - tracks["finger21"]["q_after"],
                  index_joint_motion={name: tracks[name] for name in ("finger21", "finger22")})
    if not (result["right_loaded"] and min(minimums["thumb"], minimums["middle"]) > MIN_LINK_LOAD_N
            and all(min(current["external_support_force_norm_n"][name]) > 0.
                    for name in ("support_0", "support_1"))):
        return dict(result, status="protected_measured_support_not_loaded")
    if not (result["signed_index_target_lead_rad"] > 0.
            and all(tracks[name]["q_after"] < tracks[name]["q_before"] and tracks[name]["qd_after_per_s"] < 0.
                    for name in ("finger21", "finger22"))):
        return dict(result, status="index_signed_retreat_not_observed")

    witness_nodes = {}
    for link in LINKS:
        source = old_rod if link == INDEX else rod
        witnesses = {pair["node_index"] for pair in source["rod_mesh_contact_pairs"]["pairs"]
                     if pair["mesh_name"] == link and pair["normal_force_on_rod_n"] > 0.}
        if len(witnesses) != 1:
            return dict(result, status="no_unique_measured_pad_witness")
        witness_nodes[link] = witnesses.pop()
    old_nodes = old_rod["node_positions"][0]
    p, r = _validated_pose(before["sdk_hand_geometry"]["poses"][INDEX])
    old_phi, _, valid = sample_grid(grids[INDEX], r.inv().apply(old_nodes[[witness_nodes[INDEX]]] - p),
                                    dtype=np.float64)
    if not valid[0]:
        return dict(result, status="missing_old_index_sdf_witness")
    poses = after["sdk_hand_geometry"]["poses"]
    return dict(result, witness_nodes=witness_nodes, **propose_wrist_relief(
        palm=poses["left.base_link"], poses=poses, grids=grids, nodes=rod["node_positions"][0],
        witness_nodes=witness_nodes, radius=radius, max_depth=max_depth, old_index_gap=float(old_phi[0] - radius)))


def _screen_hulls(after, *, seed, radius, max_depth, grids, boxes, table_top):
    from scripts.screen_dh116_tripod_measured_precontact import PADS, archived_tube, measure_side

    if radius != .005:
        raise ValueError("archived hull screen uses the original 10 mm wire only")
    position, orientation = _validated_pose(after["sdk_hand_geometry"]["poses"]["left.base_link"])
    rotation = Rotation.from_rotvec(seed["world_rotation_vector_rad"]) * orientation
    pose = np.r_[position + seed["world_translation_m"], np.roll(rotation.as_quat(), 1)]
    hand = after["action_readback"]["post"]["hands"]["left"]
    goal = dict(pose_wxyz=pose, active_q_rad=hand["q"][0, hand["active_indices"]])
    nodes = after["rod"]["node_positions"][0]
    route = measure_side("left", after, nodes, archived_tube(nodes[None]), boxes, grids,
                         goal, table_top, pads=PADS, contact_pads=PADS)
    limits = dict(minimum_table_gap_m=min(x["table_clearance_m"] for x in route),
                  minimum_support_gap_m=min(x["support_clearance_m"] for x in route),
                  minimum_noncontact_rod_gap_m=min(x["noncontact_rod_clearance_m"] for x in route),
                  minimum_selected_pad_sdf_gap_m=min(min(x["pad_gaps_m"]) for x in route))
    passed = (all(x["shape_count"] == 22 for x in route) and limits["minimum_table_gap_m"] >= CLEARANCE_M
              and limits["minimum_support_gap_m"] > 1e-10 and limits["minimum_noncontact_rod_gap_m"] > .00025
              and limits["minimum_selected_pad_sdf_gap_m"] >= -max_depth)
    return dict(limits, sampled_hull_screen_passed=bool(passed), samples=len(route), left_sampled_shapes=22)


def replay(run, *, receipt_sha256, cohort, cohort_sha256, screen_hulls):
    outer = json.loads(checked_file(run/"outer.json", receipt_sha256))
    losses = json.loads(checked_file(cohort, cohort_sha256))
    if (outer["exit_code"] != 0 or outer["forced_exit"] or outer["exceeded_budget"]
            or outer["debugger_status_verified"] is not True
            or outer["debugger_inferior_exit_code"] != 0 or outer["debugger_inferior_signal"] is not None
            or losses["receipt_sha256"] != receipt_sha256):
        raise ValueError("sealed original natural-exit Tripod run required")
    inventory = outer["artifact_sha256"]

    def read(name):
        return decode_owned(json.loads(checked_file(run/"native"/name, inventory[name])))

    close = read("app_close_returned.json")
    if close["app_close_returned"] is not True or close["close_mode"] != "normal":
        raise ValueError("sealed normal app.close required")
    reset = read("tripod_reset_source_2.json")["evidence"]
    radius = reset["rod_radius_m"]
    max_depth = read("frozen_plan.json")["right_tripod_distal_contact"]["max_pad_penetration_m"]
    if not losses["samples"]:
        raise ValueError("measured adjacent first-loss cohort required")

    bindings, grids, rows, accepted = None, {}, [], []
    seen = set()
    for sample in losses["samples"]:
        early, late = (sample[k] for k in ("previous_loaded_sdk", "first_loss_sdk"))
        if (late != early + 1 or late in seen
                or sample["artifact_sha256"] != inventory[f"tripod_control_{late}.json"]):
            raise ValueError("unique adjacent losses and source digests required")
        seen.add(late)
        before, after = (_contact_row(read, step) for step in (early, late))
        if bindings is None:
            bindings = {link: after["rod"]["mesh_bindings"][link] for link in LINKS}
            for link, binding in bindings.items():
                if binding["frame"] != "PhysX_link_origin" or binding["unit"] != "m":
                    raise ValueError("original metric link SDF required")
                raw = checked_file(Path(binding["sdf_path"]), binding["sdf_sha256"])
                with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                    grids[link] = {key: data[key] for key in ("values", "origin", "spacing")}
        if any(row["rod"]["mesh_bindings"][link] != bindings[link] for row in (before, after) for link in LINKS):
            raise ValueError("locked pad binding changed across loss")
        row = _archive_sample(before, after, radius, max_depth, grids)
        rows.append(row)
        if row["status"] == "sampled_pose_candidate":
            accepted.append((row, after))

    shortlist = sorted(accepted, key=lambda pair: (pair[0]["sample_fraction"], pair[0]["sdk"]))
    finalists = []
    if screen_hulls and shortlist:
        supports = read("tripod_support_scene.json")["supports"]
        boxes = [(np.asarray(x["center_m"]) - np.asarray(x["size_m"]) / 2,
                  np.asarray(x["center_m"]) + np.asarray(x["size_m"]) / 2) for x in supports]
        table_top = reset["table_center_m"][2] + .5 * reset["table_size_m"][2]
        for row, after in shortlist:
            result = _screen_hulls(after, seed=row, radius=radius, max_depth=max_depth, grids=grids,
                                   boxes=boxes, table_top=table_top)
            finalists.append(dict(sdk=row["sdk"], **result))
            if result["sampled_hull_screen_passed"]:
                break
    return dict(version="dh116-tripod-wrist-relief-replay-v1", source_head=outer["head"],
                receipt_sha256=receipt_sha256, cohort_sha256=cohort_sha256,
                sdf_sha256={name: binding["sdf_sha256"] for name, binding in bindings.items()},
                counts=dict(Counter(row["status"] for row in rows)), rows=rows, hull_screen=finalists,
                geometry_scope="sampled_frozen_native_rod_and_finger_endpoints",
                motion_semantics="measured_palm_world_pose_offset_not_intrinsic_xyz_action",
                new_native_steps=0, teacher_admission=False, grasp_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("run", "cohort", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    for field in ("receipt-sha256", "cohort-sha256"):
        parser.add_argument(f"--{field}", required=True)
    parser.add_argument("--screen-hulls", action="store_true")
    args = parser.parse_args()
    output = replay(args.run, receipt_sha256=args.receipt_sha256, cohort=args.cohort,
                    cohort_sha256=args.cohort_sha256, screen_hulls=args.screen_hulls)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(output, f, allow_nan=False, indent=2)
        f.write(chr(10))


if __name__ == "__main__":
    main()
