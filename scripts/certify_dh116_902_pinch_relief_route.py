"""Check simultaneous Pinch endpoints and replay the original 24D approach offline."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import active_limits
from scripts.diagnose_dh116_teacher_initialization import digest
from scripts.fit_dh116_902_pinch_alignment import (
    MOVE_SCALE, OPPOSITION_LIMIT, PADS, PAD_LIMIT_M, ROD_RADIUS_M, SIDES, _boundary_context, _side_fit,
)
from scripts.screen_dh116_902_command_path import command_step
from scripts.screen_dh116_902_pinch_facets import _body_pose, _phi, _unit
from scripts.verify_dh116_902_flexible_reset import load_candidate, sha
from scripts.verify_dh116_tripod_supported_reset import waypoint_controls


RELIEF_SHA = {
    "left": "c606f84d04beadb939039dc39148e14cf15b050ac3d2795222aad43682cac89c",
    "right": "553e446fe5c67da938f05be4d5a9dc87ef13dd548d6bee1be994322541da2f38",
}
PROVENANCE = ("outer_sha256", "archive_sha256", "last_row_sha256")
ROUTE_WINDOWS_MAX = 240


def cross_hand_separation(left, right):
    """Signed all-vertex separating bound along the original left/right axis."""
    return float(min(points[:, 1].min() for points in left.values())
                 - max(points[:, 1].max() for points in right.values()))


def _ports(context, models):
    result = {}
    for side in SIDES:
        hand = context["row"]["action_readback"]["post"]["hands"][side]
        q = np.clip(hand["q"][0, hand["active_indices"]], *active_limits())

        def port(**contents):
            return dict(**contents, measured_q=q)

        result[side] = _side_fit(side, **context, model=models[side], max_evaluations=1, starts=1, optimize=port)
    return result


def _coordinates(port, pose, q):
    target = Rotation.from_quat(pose[[4, 5, 6, 3]])
    return np.r_[pose[:3] - port["origin"], (port["rotation"].inv() * target).as_rotvec(),
                 q[:3] - port["measured_q"][:3]] / MOVE_SCALE


def _frame_contact(nodes, grids, poses, station):
    normals, gaps, contacts, seen = [], [], {}, True
    for pad in PADS:
        phi, grad, valid = _phi(grids[pad], nodes, poses[pad])
        contacts[pad] = np.flatnonzero(valid & (phi < ROD_RADIUS_M)).tolist()
        ok = valid[station] and np.linalg.norm(grad[station]) > 1e-12
        seen = seen and ok
        gaps.append(float(phi[station] - ROD_RADIUS_M) if ok else None)
        if ok:
            normals.append(poses[pad][0] @ _unit(grad[station]))
    opposition = float(normals[0] @ normals[1]) if seen else None
    exclusivity = all(contacts[pad] == [station] for pad in PADS)
    passed = bool(seen and exclusivity and opposition < OPPOSITION_LIMIT and min(gaps) >= -PAD_LIMIT_M)
    return dict(station=station, gaps_m=gaps, negative_nodes=contacts, normal_dot=opposition, passed=passed)


def _pose_maps(world, models):
    return {side: {pad: _body_pose(models[side]["cooked_pads"][pad]["vertices"],
                                  world[side][f"cooked_pads:{pad}"]) for pad in PADS} for side in SIDES}


def _paired_frames(context, grids, poses, stations, world):
    gap = cross_hand_separation(world["left"], world["right"])
    return [dict(sdk=sdk, positive_y_separation_m=gap,
                 hands={side: _frame_contact(nodes, grids[side], poses[side], stations[side]) for side in SIDES})
            for sdk, nodes in enumerate(context["nodes"], 424)]


def _read_endpoints(paths, source):
    reports, counts = {}, {}
    for side, path in zip(SIDES, paths, strict=True):
        if sha(path) != RELIEF_SHA[side]:
            raise ValueError("sealed relief result required on each original side")
        report = json.loads(path.read_bytes())
        if report["side"] != side or any(report[k] != source[k] for k in PROVENANCE):
            raise ValueError("both endpoints must share the same native rod clock and side order")
        candidates = [row for row in report["candidates"] if row["endpoint_passed"]]
        if not candidates:
            raise ValueError("qualified original endpoint required")
        reports[side] = candidates[0]
        counts[side] = len(candidates)
    return reports, counts


def _phase(row, config, projected, command, angles, goals):
    records = {}
    for i, side in enumerate(SIDES):
        hand = row["action_readback"]["post"]["hands"][side]
        targets = hand["backend_targets"].copy()
        targets[:, hand["active_indices"]] = command[:, i*6:i*6+6]
        records[side] = dict(hand, backend_targets=targets)
    state = dict(action_readback=dict(post=dict(hands=records)),
                 wrist_projection=dict(projected_root_poses=projected, projected_angles=angles))
    slow = waypoint_controls(config, state, goals, windows=ROUTE_WINDOWS_MAX)[0]
    largest = max(float(np.max(np.abs(slow[:12]))),
                  *(float(np.linalg.norm(slow[i:i+3])) for i in (12, 15, 18, 21)))
    windows = max(1, int(np.ceil(largest * ROUTE_WINDOWS_MAX)) + 1)
    return waypoint_controls(config, state, goals, windows=min(ROUTE_WINDOWS_MAX, windows))


def retreat_direction(probe, *, distance_m):
    """Prefer maximum rod reserve among sampled clear withdrawal directions."""
    options = []
    for direction in product((-1., 0., 1.), repeat=3):
        length = np.linalg.norm(direction)
        if length <= 0.:
            continue
        offset = np.asarray(direction) * distance_m / length
        samples = [probe(offset * fraction) for fraction in (.125, .25, .5, .75, 1.)]
        if all(value is not None for value in samples):
            options.append(dict(offset_m=offset.tolist(), sampled_rod_reserve_mm=samples))
    if not options:
        return None
    return max(options, key=lambda option: (option["sampled_rod_reserve_mm"][-1],
                                            min(option["sampled_rod_reserve_mm"])))


def _waypoints(row, target, standoff_m, ports, stations):
    if standoff_m == 0.:
        return [("direct", target)], {}
    retract, align, choices = {}, {}, {}
    for i, side in enumerate(SIDES):
        issued = row["wrist_projection"]["projected_root_poses"][0, i].copy()
        hand = row["action_readback"]["post"]["hands"][side]
        q = hand["backend_targets"][0, hand["active_indices"]]
        aim = np.asarray(target[side]["pose_wxyz"]).copy()
        q_goal = np.asarray(target[side]["active_q_rad"])
        port = ports[side]

        def probe(base, joints):
            def clearance(offset):
                pose = np.r_[base[:3] + offset, base[3:]]
                coords = _coordinates(port, pose, joints)
                if np.any(coords < port["lower"]) or np.any(coords > port["upper"]):
                    return None
                margins = dict(zip(port["labels"], port["state"](coords, stations[side])[2:], strict=True))
                geometry = [v for k, v in margins.items() if k.startswith(("rod:", "support:", "table:"))]
                if min(geometry) <= 0.:
                    refined = port["measure"](coords)
                    if any(failure != "opposed_pad_sdf" for failure in refined["failures"]):
                        return None
                return min(v for k, v in margins.items() if k.startswith("rod:"))
            return clearance

        departure = retreat_direction(probe(issued, q), distance_m=standoff_m)
        approach = retreat_direction(probe(aim, q_goal), distance_m=standoff_m)
        choices[side] = dict(departure=departure, approach=approach)
        if departure is None or approach is None:
            return [], choices
        issued[:3] += departure["offset_m"]
        aim[:3] += approach["offset_m"]
        retract[side] = dict(pose_wxyz=issued, active_q_rad=q)
        align[side] = dict(pose_wxyz=aim, active_q_rad=q_goal)
    return [("retract", retract), ("align", align), ("fixed_orientation_approach", target)], choices


def _geometry(step, ports, roots, q, stations, models):
    world, margins, errors = {}, {}, []
    clearance = np.empty((1, 2))
    for i, side in enumerate(SIDES):
        port = ports[side]
        coords = _coordinates(port, roots[0, i], q[0, i*6:i*6+6])
        world[side] = port["world_shapes"](coords)
        margins[side] = dict(zip(port["labels"], port["state"](coords, stations[side])[2:], strict=True))
        geometry = {k: v for k, v in margins[side].items() if k.startswith(("rod:", "support:", "table:"))}
        limiting = min(geometry, key=geometry.get)
        if geometry[limiting] <= 0.:
            # A close selected-pad bound can refine under the original whole-envelope verifier.
            hand = port["measure"](coords)
            errors.extend(f"{side}.{r}" for r in hand["failures"] if r != "opposed_pad_sdf")
        clearance[0, i] = min(float(np.min(v[:, 2])) for v in world[side].values()) - step["table_top"]
    cross = cross_hand_separation(world["left"], world["right"])
    if cross <= 0.:
        errors.append("cross_hand_clearance")
    return world, clearance, cross, margins, errors


def _replay(context, ports, grids, models, stations, config, targets, standoff_m):
    row = context["row"]
    projection = row["wrist_projection"]
    hands = row["action_readback"]["post"]["hands"]
    poses = row["sdk_hand_geometry"]["poses"]
    def base(side):
        p, q = poses[f"{side}.base_link"]
        return np.r_[p, q[[3, 0, 1, 2]]]

    roots = np.array([[base(side) for side in SIDES]])
    command = np.concatenate([hands[s]["backend_targets"][:, hands[s]["active_indices"]] for s in SIDES], axis=1)
    initial_command = command.copy()
    q = np.concatenate([hands[s]["q"][:, hands[s]["active_indices"]] for s in SIDES], axis=1)
    anchors = np.stack([hands[s]["mount_anchor_pose_w"] for s in SIDES], axis=1)
    anchor_rotations = [Rotation.from_quat(anchors[0, i, [4, 5, 6, 3]]) for i in range(2)]
    clearance = np.asarray(row["sdk_hand_geometry"]["table_clearance_m"]).reshape(1, 2).copy()
    seed = np.array([[config["hands"][s]["pose_wxyz"] for s in SIDES]])
    projected, palms, angles = [projection[k].copy() for k in
                                 ("projected_root_poses", "palm_offsets_m", "projected_angles")]
    verified_start = dict(active_backend_targets_rad=initial_command.tolist(),
                          projected_root_poses=projected.tolist(), measured_root_poses=roots.tolist())
    phases, frames, windows_issued, cross_min, next_contact = [], 0, 0, float("inf"), None
    waypoints, retreats = _waypoints(row, targets, standoff_m, ports, stations)
    if not waypoints:
        return dict(standoff_m=standoff_m, retreats=retreats, phases=[], checked_substeps=0,
                    first_guard=dict(ordinal=0, phase="retreat_prescreen", failures=["no_clear_withdrawal"]),
                    path_passed=False)
    for name, targets_in_phase in waypoints:
        controls = _phase(row, config, projected, command, angles, targets_in_phase)
        phases.append(dict(name=name, control_windows=len(controls), action=controls[0].tolist()))
        for action in controls:
            windows_issued += 1
            for _ in range(8):
                out = command_step(action=action[None], backend_targets=command, measured_q=q,
                                   seed=seed, measured_roots=roots, clearance=clearance, palms=palms,
                                   angles=angles, mount_anchor_poses=anchors)
                command = out["submitted_finger_targets"].astype(float)
                q = command.copy()
                for i, anchor_rotation in enumerate(anchor_rotations):
                    issued = out["submitted_wrist_targets"][0, i].astype(float)
                    roots[0, i, :3] = anchors[0, i, :3] + anchor_rotation.apply(issued[:3])
                    roots[0, i, 3:] = (anchor_rotation * Rotation.from_euler("XYZ", issued[3:])).as_quat()[[3, 0, 1, 2]]
                projected, palms, angles = [out["projection"][k] for k in
                                            ("projected_root_poses", "palm_offsets_m", "projected_angles")]
                frames += 1
                world, clearance, cross, margins, errors = _geometry(context, ports, roots, q, stations, models)
                cross_min = min(cross_min, cross)
                bodies = _pose_maps(world, models)
                for mesh_idx, nodes in enumerate(context["nodes"]):
                    for side in SIDES:
                        material = _frame_contact(nodes, grids[side], bodies[side], stations[side])
                        for pad, gap in zip(PADS, material["gaps_m"], strict=True):
                            if gap is not None and gap < -PAD_LIMIT_M:
                                errors.append(f"{side}.{pad}.sdf_penetration")
                            if any(index != stations[side] for index in material["negative_nodes"][pad]):
                                errors.append(f"{side}.{pad}.off_station_contact")
                        if any(material["negative_nodes"].values()) and next_contact is None:
                            next_contact = dict(ordinal=frames, archive_sdk=424 + mesh_idx, side=side, phase=name)
                if errors:
                    blocker = dict(ordinal=frames, phase=name, failures=sorted(set(errors)),
                                   margins_mm=margins, issued_root_pose=roots.tolist(), issued_q=q.tolist())
                    return dict(standoff_m=standoff_m, retreats=retreats, phases=phases, checked_substeps=frames,
                                issued_control_windows=windows_issued, start=verified_start,
                                action_package_sha256=digest(phases), minimum_cross_hand_m=cross_min,
                                first_contact=next_contact, first_guard=blocker, path_passed=False)
    paired = _paired_frames(context, grids, bodies, stations, world)
    endpoint_error = {}
    for i, side in enumerate(SIDES):
        expected = np.asarray(targets[side]["pose_wxyz"])
        r0 = Rotation.from_quat(roots[0, i, [4, 5, 6, 3]])
        r1 = Rotation.from_quat(expected[[4, 5, 6, 3]])
        endpoint_error[side] = dict(position_m=float(np.linalg.norm(roots[0, i, :3] - expected[:3])),
                rotation_rad=float((r0.inv() * r1).magnitude()),
                joint_rad=float(np.max(np.abs(q[0, i*6:i*6+6] - targets[side]["active_q_rad"]))))
    tracking = all(v["position_m"] <= 1e-6 and v["rotation_rad"] <= 1e-6 and v["joint_rad"] <= 1e-5
                   for v in endpoint_error.values())
    return dict(standoff_m=standoff_m, retreats=retreats, phases=phases, checked_substeps=frames,
                issued_control_windows=windows_issued, start=verified_start, action_package_sha256=digest(phases),
                minimum_cross_hand_m=cross_min, first_contact=next_contact, first_guard=None,
                final_frame_contacts=paired, endpoint_error=endpoint_error,
                path_passed=bool(tracking and all(h["passed"] for row in paired for h in row["hands"].values())))


def screen(receipt, left, right, *, run_routes=True):
    context, models, source = _boundary_context(receipt, sides=SIDES)
    selections, counts = _read_endpoints((left, right), source)
    ports = _ports(context, models)
    grids = {s: {pad: context["grids"][f"{s}.{pad}"] for pad in PADS} for s in SIDES}
    targets = {side: selections[side]["endpoint"] for side in SIDES}
    stations = {side: selections[side]["station"] for side in SIDES}
    endpoint, world = {}, {}
    for side in SIDES:
        target = targets[side]
        config = _coordinates(ports[side], np.asarray(target["pose_wxyz"]), np.asarray(target["active_q_rad"]))
        endpoint[side] = ports[side]["measure"](config)
        world[side] = ports[side]["world_shapes"](config)
    paired = _paired_frames(context, grids, _pose_maps(world, models), stations, world)
    endpoint_passed = (all(e["endpoint_passed"] for e in endpoint.values()) and
                       all(f["positive_y_separation_m"] > 0. and all(h["passed"] for h in f["hands"].values())
                           for f in paired))
    routes = []
    if endpoint_passed and run_routes:
        _, config, _ = load_candidate("pinch")
        for standoff in (0., .008, .016):
            result = _replay(context, ports, grids, models, stations, config, targets, standoff)
            routes.append(result)
            if result["path_passed"]:
                break
    return dict(version="dh116-902-pinch-relief-joint-route-v1", **source, endpoint=endpoint, paired_frames=paired,
                qualification_inputs_sha256=RELIEF_SHA, route_source_sha256=sha(Path(__file__)),
                per_side_qualified_proposals=counts, qualified_cartesian_pairs=int(np.prod(list(counts.values()))),
                joint_pairs_checked=1, joint_endpoint_passed=endpoint_passed, routes=routes,
                command_path_qualified=any(route["path_passed"] for route in routes),
                scope="conditional_static_nine_frame_rod_and_ideal_submitted_target_tracking",
                native_contact_verified=False, teacher_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh joint endpoint and route evidence required")
    result = screen(args.receipt, args.left, args.right)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    return 0 if result["command_path_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
