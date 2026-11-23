"""Test staged Pinch index clearance from the sealed R5 issued-state blocker."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import active_limits
from scripts.certify_dh116_902_pinch_relief_route import (
    _coordinates, _frame_contact, _geometry, _paired_frames, _phase, _ports, _pose_maps, _read_endpoints,
)
from scripts.diagnose_dh116_teacher_initialization import digest
from scripts.fit_dh116_902_pinch_alignment import PADS, PAD_LIMIT_M, SIDES, _boundary_context
from scripts.screen_dh116_902_command_path import command_step
from scripts.screen_dh116_902_pinch_facets import _body_pose, _nearest_segment
from scripts.verify_dh116_902_flexible_reset import load_candidate, sha


R5_ROUTE_SHA = "593a9fc121306f04313c9e21ad5d440d5d4ca851896e62260b3c8cea7bff3d76"
R6_LEFT_ONLY_SHA = "deb5e2638727c9c22099f5bece9ae7493a2b77fc28ec0e48c4446c4de856becb"
INDEX_HULL = "source_colliders:finger23_link"
INDEX_OFFSETS_RAD = (-.03, .03, -.06, .06, -.09, .09)
ROD_ROLL_RAD = (-.08, .08)


def _geometry_failures(proof):
    return [failure for failure in proof["failures"] if failure != "opposed_pad_sdf"]


def _change_index(goal, offset_rad):
    q = np.asarray(goal["active_q_rad"], dtype=float).copy()
    q[2] += offset_rad
    return dict(goal, active_q_rad=q)


def paired_index_stages(start, finish, *, offsets):
    """Retain the side-specific index reserves through the wrist alignment."""
    return [
        ("withdraw_index", dict(start, **{s: _change_index(start[s], a) for s, a in offsets.items()})),
        ("aim_with_index_reserve", dict(finish, **{s: _change_index(finish[s], a) for s, a in offsets.items()})),
        ("seat_at_fixed_wrist", finish),
    ]


def index_stages(start, finish, *, side, offset_rad):
    """Construct the original one-sided form of the same staged action."""
    return paired_index_stages(start, finish, offsets={side: offset_rad})


def _probe(context, models, port, side, pose, q, station, substep, expected):
    coordinates = _coordinates(port, pose, q)
    original = port["measure"](coordinates)
    hull = original["rod_envelopes"][INDEX_HULL]
    if hull["refinement"] is None or hull["refinement"]["reason"] != expected:
        raise ValueError("known source index guard failed direct replay")
    local_source = models[side]["source_colliders"]["finger23_link"]["vertices"]
    local_cooked = models[side]["cooked_pads"]["finger23_link"]["vertices"]
    world = port["world_shapes"](coordinates)
    body = _body_pose(local_cooked, world["cooked_pads:finger23_link"])
    low, high = (21, 40) if side == "left" else (0, 19)
    segments = range(max(low, station - 4), min(station + 2, high))
    active_segment = _nearest_segment(world[INDEX_HULL], body, local_source, context["nodes"], segments)
    i = active_segment["segment_start_node"]
    tangent = context["nodes"][-1, i + 1] - context["nodes"][-1, i]
    tangent /= np.linalg.norm(tangent)
    orientation = Rotation.from_quat(pose[[4, 5, 6, 3]])
    modes = [(value, 0.) for value in INDEX_OFFSETS_RAD] + [(0., angle) for angle in ROD_ROLL_RAD]
    alternatives = []
    for offset, roll in modes:
        target = (Rotation.from_rotvec(tangent * roll) * orientation).as_quat()[[3, 0, 1, 2]]
        alternative = _change_index(dict(pose_wxyz=np.r_[pose[:3], target], active_q_rad=q), offset)
        coords = _coordinates(port, alternative["pose_wxyz"], alternative["active_q_rad"])
        if np.any(coords < port["lower"]) or np.any(coords > port["upper"]):
            continue
        proof = port["measure"](coords)
        failures = _geometry_failures(proof)
        gap = proof["rod_envelopes"][INDEX_HULL]["lower_m"]
        alternatives.append(dict(index_offset_rad=offset, rod_roll_rad=roll, source_index_gap_m=gap,
                                 improvement_m=gap - hull["lower_m"], full_geometry_passed=not failures,
                                 failures=failures))
    return dict(side=side, issued_substep=substep, pad_hull=INDEX_HULL, active_segment=active_segment,
                digit_rod_tangent_w=tangent.tolist(), original_refinement=hull["refinement"],
                original_source_index_gap_m=hull["lower_m"], alternatives=alternatives)


def _blocker_probe(context, models, ports, archive, stations):
    rejected = next(row for row in archive["routes"] if row["standoff_m"] == 0.)
    first = rejected["first_guard"]
    if first["ordinal"] != 210 or first["failures"] != ["left.pad_rod_clearance"]:
        raise ValueError("sealed direct R5 left index blocker required")
    actual_roots = np.asarray(first["issued_root_pose"])[0]
    actual_q = np.asarray(first["issued_q"])[0]
    return _probe(context, models, ports["left"], "left", actual_roots[0], actual_q[:6],
                  stations["left"], first["ordinal"], "subset_collision")


def _right_probe(context, models, ports, archive_path, source, stations):
    if sha(archive_path) != R6_LEFT_ONLY_SHA:
        raise ValueError("sealed one-sided index-reserve replay required")
    archive = json.loads(archive_path.read_bytes())
    if any(archive[k] != source[k] for k in ("outer_sha256", "last_row_sha256", "archive_sha256")):
        raise ValueError("new right blocker must use the same sealed material clock")
    attempt = next(row for row in archive["attempts"] if row["index_offset_rad"] == -.03)
    first = attempt["first_guard"]
    if first["failures"] != ["right.pad_rod_clearance"] or first["phase"] != "aim_with_index_reserve":
        raise ValueError("expected exact right index guard during left reserve approach")
    blocked = first["proofs"]["right"]
    return _probe(context, models, ports["right"], "right", np.asarray(blocked["pose_wxyz"]),
                  np.asarray(blocked["active_q_rad"]), stations["right"], first["ordinal"],
                  "full_envelope_unresolved")


def _issued_goals(row):
    start = {}
    for i, side in enumerate(SIDES):
        hand = row["action_readback"]["post"]["hands"][side]
        start[side] = dict(pose_wxyz=row["wrist_projection"]["projected_root_poses"][0, i],
                           active_q_rad=hand["backend_targets"][0, hand["active_indices"]])
    return start


def _offset_preflight(ports, start, targets, blocker):
    side = blocker["side"]
    port = ports[side]
    modes = []
    for row in blocker["alternatives"]:
        offset = row["index_offset_rad"]
        if offset == 0. or not row["full_geometry_passed"]:
            continue
        unsafe = []
        # At both ends check the index-only line before compiling any SDK command.
        for name, phase in (("source_open", start[side]), ("fixed_wrist_seat", targets[side])):
            for fraction in (.125, .25, .5, .75, 1.):
                proposed = _change_index(phase, offset * fraction)
                coords = _coordinates(port, np.asarray(proposed["pose_wxyz"]), proposed["active_q_rad"])
                if np.any(coords < port["lower"]) or np.any(coords > port["upper"]):
                    unsafe.append(dict(phase=name, fraction=fraction, failures=["original_workspace"]))
                    break
                failures = _geometry_failures(port["measure"](coords))
                if failures:
                    unsafe.append(dict(phase=name, fraction=fraction, failures=failures))
                    break
        modes.append(dict(index_offset_rad=offset, source_index_gap_at_blocker_m=row["source_index_gap_m"],
                          fixed_wrist_samples_clear=not unsafe, first_blockers=unsafe))
    modes.sort(key=lambda mode: (not mode["fixed_wrist_samples_clear"], abs(mode["index_offset_rad"])))
    return modes


def _replay(context, ports, grids, models, stations, config, targets, stages):
    row = context["row"]
    projection = row["wrist_projection"]
    hands = row["action_readback"]["post"]["hands"]
    poses = row["sdk_hand_geometry"]["poses"]
    roots = np.array([[np.r_[poses[s + ".base_link"][0], poses[s + ".base_link"][1][[3, 0, 1, 2]]] for s in SIDES]])
    command = np.concatenate([hands[s]["backend_targets"][:, hands[s]["active_indices"]] for s in SIDES], 1)
    q = np.concatenate([hands[s]["q"][:, hands[s]["active_indices"]] for s in SIDES], 1)
    anchors = np.stack([hands[s]["mount_anchor_pose_w"] for s in SIDES], axis=1)
    anchor_rotations = [Rotation.from_quat(anchors[0, i, [4, 5, 6, 3]]) for i in range(2)]
    clearance = np.asarray(row["sdk_hand_geometry"]["table_clearance_m"]).reshape(1, 2).copy()
    seed = np.array([[config["hands"][s]["pose_wxyz"] for s in SIDES]])
    projected, palms, angles = [projection[k].copy() for k in
                                 ("projected_root_poses", "palm_offsets_m", "projected_angles")]
    backend_start = dict(projected_roots=projected.tolist(), active_fingers=command.tolist(),
                         native_measured_roots=roots.tolist())
    phases, ordinal, min_cross, first_sdf = [], 0, float("inf"), None
    for name, goals in stages:
        controls = _phase(row, config, projected, command, angles, goals)
        phases.append(dict(name=name, windows=len(controls), action=controls[0].tolist()))
        for action in controls:
            for _ in range(8):
                out = command_step(action=action[None], backend_targets=command, measured_q=q,
                                   seed=seed, measured_roots=roots, clearance=clearance, palms=palms,
                                   angles=angles, mount_anchor_poses=anchors)
                command = out["submitted_finger_targets"].astype(float)
                q = command.copy()
                for i, basis in enumerate(anchor_rotations):
                    target = out["submitted_wrist_targets"][0, i].astype(float)
                    roots[0,i,:3] = anchors[0,i,:3] + basis.apply(target[:3])
                    roots[0,i,3:] = (basis * Rotation.from_euler("XYZ", target[3:])).as_quat()[[3, 0, 1, 2]]
                projected, palms, angles = [out["projection"][k] for k in
                                            ("projected_root_poses", "palm_offsets_m", "projected_angles")]
                ordinal += 1
                world, clearance, cross, _, failures = _geometry(context, ports, roots, q, stations, models)
                min_cross = min(min_cross, cross)
                bodies = _pose_maps(world, models)
                for sdk, nodes in enumerate(context["nodes"], 424):
                    for side in SIDES:
                        record = _frame_contact(nodes, grids[side], bodies[side], stations[side])
                        if any(record["negative_nodes"].values()) and first_sdf is None:
                            first_sdf = dict(ordinal=ordinal, sdk=sdk, phase=name, side=side, **record)
                        for pad, gap in zip(PADS, record["gaps_m"], strict=True):
                            if gap is not None and gap < -PAD_LIMIT_M:
                                failures.append(f"{side}.{pad}.sdf_penetration")
                            if any(k != stations[side] for k in record["negative_nodes"][pad]):
                                failures.append(f"{side}.{pad}.off_station_contact")
                if failures:
                    collision_proofs = {}
                    for i, side in enumerate(SIDES):
                        if not any(failure.startswith(side + ".") for failure in failures):
                            continue
                        port = ports[side]
                        trial = _coordinates(port, roots[0, i], q[0, i*6:i*6+6])
                        collision_proofs[side] = port["measure"](trial)
                    return dict(start=backend_start, phases=phases, action_sha256=digest(phases),
                                checked_substeps=ordinal, minimum_cross_hand_m=min_cross, first_sdf=first_sdf,
                                first_guard=dict(phase=name, ordinal=ordinal,
                                                 failures=sorted(set(failures)), proofs=collision_proofs),
                                path_passed=False)

    paired = _paired_frames(context, grids, bodies, stations, world)
    errors = {}
    for i, side in enumerate(SIDES):
        target = np.asarray(targets[side]["pose_wxyz"])
        r0 = Rotation.from_quat(roots[0, i, [4, 5, 6, 3]])
        r1 = Rotation.from_quat(target[[4, 5, 6, 3]])
        errors[side] = dict(position_m=float(np.linalg.norm(roots[0,i,:3] - target[:3])),
                            rotation_rad=float((r0.inv()*r1).magnitude()),
                            joint_rad=float(np.max(abs(q[0,i*6:i*6+6] - targets[side]["active_q_rad"]))))
    tracking = all(e["position_m"] <= 1e-6 and e["rotation_rad"] <= 1e-6 and e["joint_rad"] <= 1e-5
                   for e in errors.values())
    passed = tracking and all(h["passed"] for row in paired for h in row["hands"].values())
    return dict(start=backend_start, phases=phases, action_sha256=digest(phases),
                checked_substeps=ordinal, minimum_cross_hand_m=min_cross, first_sdf=first_sdf,
                first_guard=None, endpoint_errors=errors, endpoint_frames=paired, path_passed=passed)


def screen(receipt, left, right, blocker_path, *, run_routes=True, right_blocker=None):
    if sha(blocker_path) != R5_ROUTE_SHA:
        raise ValueError("sealed R5 direct-route proof required")
    archived = json.loads(blocker_path.read_bytes())
    context, models, source = _boundary_context(receipt, sides=SIDES)
    for name in ("outer_sha256", "last_row_sha256", "archive_sha256"):
        if archived[name] != source[name]:
            raise ValueError("R5 issued-state proof is for another physical boundary")
    old_candidates, counts = _read_endpoints((left, right), source)
    stations = {side: old_candidates[side]["station"] for side in SIDES}
    targets = {side: old_candidates[side]["endpoint"] for side in SIDES}
    start = _issued_goals(context["row"])
    ports = _ports(context, models)
    grids = {s: {p: context["grids"][f"{s}.{p}"] for p in PADS} for s in SIDES}
    blocker = _blocker_probe(context, models, ports, archived, stations)
    modes = _offset_preflight(ports, start, targets, blocker)
    secondary = None
    right_modes = []
    allowed_left = [mode for mode in modes if mode["fixed_wrist_samples_clear"]]
    if right_blocker is not None:
        secondary = _right_probe(context, models, ports, right_blocker, source, stations)
        right_modes = _offset_preflight(ports, start, targets, secondary)
        allowed_right = [mode for mode in right_modes if mode["fixed_wrist_samples_clear"]]
        offsets = []
        if allowed_left and allowed_right:
            offsets.append({"left": allowed_left[0]["index_offset_rad"],
                            "right": allowed_right[0]["index_offset_rad"]})
    else:
        offsets = [{"left": mode["index_offset_rad"]} for mode in allowed_left[:2]]
    attempts = []
    if run_routes:
        _, config, _ = load_candidate("pinch")
        for offsets_by_side in offsets:
            stages = paired_index_stages(start, targets, offsets=offsets_by_side)
            result = _replay(context, ports, grids, models, stations, config, targets, stages)
            attempts.append(dict(index_offsets_rad=offsets_by_side, **result))
            guard = result["first_guard"]
            if result["path_passed"] or (guard is not None and all(
                    reason.startswith("right.") for reason in guard["failures"])):
                break
    return dict(version="dh116-902-pinch-index-clearance-v1", **source,
                blocker_probe=blocker, index_offset_preflight=modes,
                right_blocker_probe=secondary, right_index_offset_preflight=right_modes, attempts=attempts,
                endpoint_proposals=counts, joint_endpoint_passed=archived["joint_endpoint_passed"],
                prior_route_sha256=R5_ROUTE_SHA, stage_source_sha256=sha(Path(__file__)),
                command_path_qualified=any(row["path_passed"] for row in attempts),
                scope="archived_rod_and_ideal_binary32_submitted_target_tracking",
                native_contact_verified=False, teacher_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "left", "right", "blocker", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--right-blocker", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh staged Pinch proof required")
    proof = screen(args.receipt, args.left, args.right, args.blocker, right_blocker=args.right_blocker)
    with args.output.open("x") as file:
        json.dump(proof, file, indent=2, allow_nan=False)
        file.write("\n")
    return 0 if proof["command_path_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
