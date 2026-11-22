"""Certify a two-stage Tripod contact candidate on a sealed supported rod."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, active_limits
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import load_sdf_arrays, prepare_scene
from go2_pvcnn.control.dh116.tabletop_power_shared_rod import END_SUPPORT_NAMES
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from go2_pvcnn.control.dh116.tabletop_training_reward import CONTROL_HZ
from go2_pvcnn.control.dh116.tabletop_wire_continuous_clearance import ROUNDING_M, WireClearance
from scripts.audit_dh116_pinch_initial_pose import load_geometry, path_displacement_bound
from scripts.certify_dh116_tripod_distal_route import support_clearance_lower
from scripts.certify_dh116_tripod_tracking_envelope import certify_box
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_contact
from scripts.prepare_dh116_tripod_contact_seed import checked_file, read_receipt
from scripts.screen_dh116_tripod_measured_precontact import as_matrix
from scripts.verify_dh116_supported_distal_transfer import check_reaction_delivery

STATION = 35
THUMB = "finger14_link"
INDEX = "finger23_link"
GROUPS = ("source_colliders", "cooked_pads")


def rotation_wxyz(quaternion):
    return Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]])


def nominal_orientation(target_wxyz, start_angles, measured_wxyz, angles):
    """Reconstruct intrinsic XYZ with the initial measured target offset held fixed."""
    target = rotation_wxyz(target_wxyz)
    seed = target * Rotation.from_euler("XYZ", start_angles).inv()
    residual = target.inv() * rotation_wxyz(measured_wxyz)
    return seed * Rotation.from_euler("XYZ", angles) * residual


def pivot_commands(start, target_wxyz, start_angles, station, rotvec, *, windows):
    """Follow the pivot with world-space chords and original intrinsic XYZ increments."""
    if type(windows) is not int or not 1 <= windows <= 256:
        raise ValueError("one to 256 pivot control windows required")
    move = Rotation.from_rotvec(np.linspace(0., 1., windows + 1)[:, None] * rotvec)
    target = rotation_wxyz(target_wxyz)
    seed = target * Rotation.from_euler("XYZ", start_angles).inv()
    angles = (seed.inv() * move * target).as_euler("XYZ")
    angles[0] = start_angles
    positions = station + move.apply(start[:3] - station)
    positions[0] = start[:3]
    action = np.zeros((windows, ACTION_CONTRACT["size"]))
    action[:, 12:15] = np.diff(positions, axis=0) * CONTROL_HZ / ACTION_CONTRACT["translation_rate_m_s"]
    action[:, 18:21] = np.diff(angles, axis=0) * CONTROL_HZ / ACTION_CONTRACT["rotation_rate_rad_s"]
    if (np.abs(angles).max() > np.pi / 2
            or any(np.linalg.norm(action[:, slots], axis=1).max() > 1.
                   for slots in (slice(12, 15), slice(18, 21)))):
        raise ValueError("pivot path exceeds the original 24D wrist rates or angular limits")
    return positions, angles, action


def read_source(run, receipt_sha):
    receipt = read_receipt(run, receipt_sha)
    hashes = receipt["artifact_sha256"]

    def artifact(name):
        return decode_owned(json.loads(checked_file(run/"native"/name, hashes[name])))

    pulse = artifact("wire_pulse_result.json")
    transfer = artifact("settled_thumb_transfer.json")
    final = artifact("result.json")
    support = artifact("tripod_support_scene.json")
    sdk = pulse["end_sdk"]
    name = f"wire_pulse_step_{sdk}.json"
    row = artifact(name)
    preceding = artifact(f"wire_pulse_step_{sdk-1}.json")
    rod = row["rod"]
    geom = row["sdk_hand_geometry"]
    post = row["action_readback"]["post"]["identity"][0]
    keys = ("run_uuid", "episode_uid", "reset_id", "env_id", "rod_uid", "physics_step", "rod_tick")
    anchored = any(p["mesh_name"] == f"left.{THUMB}" and p["node_index"] == STATION
                   and p["normal_force_on_rod_n"] > 0. for p in transfer["first_pairs"])
    if (final["failure"] is not None or final["physics_steps"] != sdk
            or final["acquisition"]["scope"] != "measured_settled_wire_left_thumb"
            or pulse["contact_reached"] is not True or transfer["first_transfer_sdk"] != sdk
            or transfer["first_contact_sdk"] != sdk - 1 or not anchored
            or row["guard_failure"] is not None or row["physics_step"] != sdk
            or geom["physics_step"] != sdk or geom["rod_source_physics_step"] != sdk-1
            or pulse["terminal_rod_clearance_physics_step"] != sdk
            or rod["physics_step"] != sdk or rod["rod_tick"] != (sdk-2)*12
            or any(post[k] != rod[k] for k in keys)
            or any(support["reset_clock"][k] != rod[k] for k in keys[:5])
            or support["no_hidden_rod_pins"] is not True
            or np.linalg.norm(rod["external_support_wrenches_world"]["table"][:3]) > 1e-6
            or any(rod["external_support_wrenches_world"][s][2] >= 0. for s in END_SUPPORT_NAMES)
            or not check_reaction_delivery(preceding, row)):
        raise ValueError("sealed same-clock supported thumb and applied reaction required")
    basis = dict(sdk=sdk, rod_tick=rod["rod_tick"], receipt_sha256=receipt_sha,
                 live_sample_sha256=hashes[name], source_head=receipt["head"],
                 wire_spec_sha256=pulse["readback"]["spec_sha256"], station=STATION)
    return row, support["supports"], pulse["measured_radius_m"], basis


def corrected_geometry(model, start, q, measured):
    """Fold the same-clock rigid frame correction into the local hull vertices."""
    nominal = fk_poses(model, start, q)
    model = dict(model)
    transforms = {}
    for group in GROUPS:
        model[group] = deepcopy(model[group])
        for name, shape in model[group].items():
            if name not in measured:
                continue
            measured_pose = as_matrix(*measured[name])
            C = np.linalg.solve(nominal[name], measured_pose)
            transforms[name] = C
            shape["vertices"] = shape["vertices"] @ C[:3, :3].T + C[:3, 3]
    return model, transforms


def point_margins(envelope, hulls, boxes, table_top, selected, reserve):
    result = {}
    for label, vertices in hulls.items():
        witness = envelope.certify(vertices, selected=label in selected, reserve_m=reserve)
        result[f"rod:{label}"] = witness["lower_m"] - witness["threshold_m"] - reserve - ROUNDING_M
        result[f"table:{label}"] = float(vertices[:, 2].min() - table_top - .0021 - ROUNDING_M)
        result[f"support:{label}"] = min(support_clearance_lower(vertices, lo, hi) for lo, hi in boxes) - ROUNDING_M
    return result


def world_hulls(model, frames):
    return {f"{group}:{name}": shape["vertices"] @ frames[name][:3, :3].T + frames[name][:3, 3]
            for group in GROUPS for name, shape in model[group].items()}


def selected_hulls(*links):
    return {f"{group}:{name}" for group in GROUPS for name in links}


def cover(margins, motion):
    coefficients = {f"{key}:{name}": [value] for name, value in motion.items()
                    for key in ("rod", "table", "support")}
    proof = certify_box(margins, coefficients, [0.], [1.], max_samples=128, max_seconds=10.)
    return dict(proved=proof["proved"], evaluated_samples=len(proof["samples"]),
                certified_intervals=len(proof["certified_boxes"]),
                minimum_margin_m=min((box["minimum_margin_m"] for box in proof["certified_boxes"]), default=None),
                violations=proof["violations"], unresolved_intervals=len(proof["unresolved_boxes"]))


def certify_route(root, row, supports, radius, *, rotvec, index_increment, reserve_m, pivot_windows):
    measured = row["sdk_hand_geometry"]["poses"]
    nodes = np.asarray(row["rod"]["node_positions"], dtype=float)
    guard = WireClearance(nodes, radius_m=radius)
    hand = row["action_readback"]["post"]["hands"]["left"]
    ids = hand["active_indices"]
    if tuple(hand["joint_names"][i] for i in ids) != ACTIVE or len(supports) != 2:
        raise ValueError("original ordered DH116 joints and two supports required")
    q = hand["q"][0, ids].astype(float)
    low, high = active_limits()
    q_final = q.copy()
    q_final[ACTIVE.index("finger21")] += index_increment
    if not np.isfinite(index_increment) or not 0. < index_increment <= .12 or np.any(q_final < low) or np.any(q_final > high):
        raise ValueError("original active limits and a bounded positive index closure required")
    positions, quat = measured["left.base_link"]
    start = np.r_[positions, quat[[3, 0, 1, 2]]]
    projection = row["action_readback"]["submission"]
    target = projection["projected_root_poses"][0, 0, 3:]
    start_angles = projection["projected_angles"][0, 0]
    station = nodes[0, STATION]
    positions, angles, actions = pivot_commands(start, target, start_angles, station, rotvec, windows=pivot_windows)
    layout = prepare_scene(root)
    table = layout["table"]
    table_top = table["center_m"][2] + table["size_m"][2] / 2
    boxes = [(np.asarray(b["center_m"]) - np.asarray(b["size_m"]) / 2,
              np.asarray(b["center_m"]) + np.asarray(b["size_m"]) / 2) for b in supports]
    left, correction = corrected_geometry(load_geometry(root, side="left"), start, q, {
        key.split(".")[1]: value for key, value in measured.items() if key.startswith("left.")})
    right_hand = row["action_readback"]["post"]["hands"]["right"]
    right_ids = right_hand["active_indices"]
    if tuple(right_hand["joint_names"][i] for i in right_ids) != ACTIVE:
        raise ValueError("right DH116 joint order differs from the original hand")
    right_q = right_hand["q"][0, right_ids]
    right_pose, right_quat = measured["right.base_link"]
    right_root = np.r_[right_pose, right_quat[[3, 0, 1, 2]]]
    right, right_correction = corrected_geometry(load_geometry(root, side="right"), right_root, right_q, {
        key.split(".")[1]: value for key, value in measured.items() if key.startswith("right.")})
    fixed = world_hulls(right, fk_poses(right, right_root, right_q))
    moving = world_hulls(left, fk_poses(left, start, q))
    if len(moving) != 22 or len(fixed) != 22:
        raise ValueError("the original 34 source hulls and 10 cooked pads are required")
    right_margins = point_margins(guard, fixed, boxes, table_top, set(), reserve_m)
    lever = {name: float(np.linalg.norm(vertices - start[:3], axis=1).max()) for name, vertices in moving.items()}

    def evaluate(pos, euler, joints, selected):
        rotation = nominal_orientation(target, start_angles, start[3:], euler).as_quat()
        frames = fk_poses(left, np.r_[pos, rotation[[3, 0, 1, 2]]], joints)
        return point_margins(guard, world_hulls(left, frames), boxes, table_top, selected, reserve_m)

    pivot = []
    for i, (p, a) in enumerate(zip(positions[:-1], angles[:-1], strict=True)):
        dp, da = positions[i+1] - p, angles[i+1] - a
        motion = {name: float(np.linalg.norm(dp) + radius_m*np.abs(da).sum())
                  for name, radius_m in lever.items()}

        def margins(t):
            return evaluate(p + t[0]*dp, a + t[0]*da, q, selected_hulls(THUMB))

        pivot.append(cover(margins, motion))
        if not pivot[-1]["proved"]:
            break

    closing = None
    if all(box["proved"] for box in pivot) and min(right_margins.values()) >= 0.:
        delta = q_final - q
        motion = {f"{group}:{name}": value for group in GROUPS for name, value in
                  path_displacement_bound(left, q, delta, cooked=group == "cooked_pads").items()}

        def margins(t):
            return evaluate(positions[-1], angles[-1], q + t[0]*delta, selected_hulls(THUMB, INDEX))

        # A constant-velocity finger closure has the same continuous path across all windows.
        closing = cover(margins, motion)

    rotation = nominal_orientation(target, start_angles, start[3:], angles[-1]).as_quat()
    endpoint = fk_poses(left, np.r_[positions[-1], rotation[[3, 0, 1, 2]]], q_final)
    grids = load_sdf_arrays(layout)
    pads = {link: sample_contact(grids[f"left.{link}"], endpoint[link] @ correction[link], nodes[0],
                                radius=radius, dtype=np.float64) for link in (THUMB, INDEX)}
    geometrically_proved = bool(len(pivot) == pivot_windows and all(box["proved"] for box in pivot)
                                and closing and closing["proved"] and min(right_margins.values()) >= 0.)
    return dict(proved_static_hull_path=geometrically_proved, rod_displacement_reserve_m=reserve_m,
                world_rotvec_rad=rotvec.tolist(), index_increment_rad=index_increment,
                pivot=pivot, close=closing, terminal_sdf=pads,
                right_fixed_minimum_margin_m=min(right_margins.values()), issued_action_count=0,
                canonical_proposed_pivot_actions=actions.tolist(),
                sdk_measured_frame_corrections={"left": sorted(correction), "right": sorted(right_correction)},
                static_rod_and_nominal_mimic=True, native_measurement=False, grasp_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--world-rotvec", type=float, nargs=3, required=True)
    parser.add_argument("--index-increment", type=float, required=True)
    parser.add_argument("--reserve-m", type=float, required=True)
    parser.add_argument("--pivot-windows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row, supports, radius, basis = read_source(args.run, args.receipt_sha256)
    result = certify_route(Path.cwd(), row, supports, radius, rotvec=np.asarray(args.world_rotvec),
                           index_increment=args.index_increment, reserve_m=args.reserve_m,
                           pivot_windows=args.pivot_windows)
    result.update(version="dh116-tripod-static-opposition-route-v1", basis=basis, radius_m=radius)
    with args.output.open("x") as file:
        json.dump(result, file, allow_nan=False, indent=2)
        file.write("\n")
    return 0 if result["proved_static_hull_path"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
