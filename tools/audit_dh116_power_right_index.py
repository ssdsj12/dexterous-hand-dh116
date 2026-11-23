"""Bound and screen right-index entry on the archived supported Power rod."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from go2_pvcnn.control.dh116.linear_coupling import JOINTS, active_limits, coupling_matrix
from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts import _clock
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file, digest
from go2_pvcnn.control.dh116.tabletop_polyline_support_certificate import NUMERIC_GUARD
from scripts import certify_dh116_power_thumb_route as route
from scripts import prepare_dh116_power_preload as preload
from scripts import search_dh116_power_surface_pairs as search
from scripts.audit_dh116_pinch_initial_pose import checked_json, load_geometry, path_displacement_bound
from scripts.audit_dh116_wire_contact_witness import endpoint_gap
from scripts.prepare_dh116_pinch_closure_margin import fk_poses
from scripts.refit_dh116_power_left_contact import projected_segment_margin
from scripts.screen_dh116_power_measured_reseat import support_adjacent_nodes
from scripts.screen_dh116_power_partial_opposition import DISTAL_PAD_LINKS, endpoint, rod_clearance_required
from scripts.screen_dh116_tripod_measured_precontact import as_matrix
from tools.audit_dh116_power_bilateral_handoff import FROZEN_SHA256
from tools.audit_dh116_power_coupled_wrist import capsule_planes, material_frame, transform_wrist
from tools.audit_dh116_power_remaining_digits import eligible_nodes


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "docs/contracts/dh116_teacher_power_bilateral_detour_v1.json"
FRONTIER_SHA = "eff0f021b3dc16cdb77bed0775a5b11ff16a340eb5f8cb4768620eab0bd0b5bf"
PRIOR_CODE = {
    "tools/audit_dh116_power_remaining_digits.py": "5e285d589b08cb1a92e65f6653909f5f4555ac87d59edeeb413bb4de2f01298b",
    "tools/audit_dh116_power_coupled_wrist.py": "ef3c11f4e99b24692cc8041678d035394c3baec4b2db0bcd59c8efc1718c9299",
}
INDEX = "finger23_link"
LITTLE = "finger53_link"
PALM_NODE = 9
SHELL_M = .0005
TRANSLATION_COMPARISON_M = .006
SAMPLES = 257
WRIST_SCALE = np.array([.012, .012, .012, .28, .16, .32])


def required_tilt(displacement_m, node_lever_m, translation_m):
    """Minimum arbitrary rotation angle allowed by an inverse-point chord bound."""
    if not all(np.isfinite(value) and value >= 0. for value in (displacement_m, node_lever_m, translation_m)):
        raise ValueError("finite nonnegative displacement, lever and translation required")
    remaining = max(0., displacement_m - translation_m)
    if remaining == 0.:
        return 0.
    if remaining > 2. * node_lever_m:
        return None
    return float(2. * np.arcsin(remaining/(2.*node_lever_m)))


def distance_lower(hull, points):
    """Per-point convex-hull distance lower bounds from box and support projections."""
    lo, hi = hull.min(axis=0), hull.max(axis=0)
    box = np.linalg.norm(np.maximum(np.maximum(lo-points, points-hi), 0.), axis=1)
    directions = points - hull.mean(axis=0)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= np.maximum(norms, 1e-12)
    # A direction of length <=1 also gives a valid bound.
    plane = np.einsum("ij,ij->i", directions, points) - (directions @ hull.T).max(axis=1)
    return np.maximum(box, plane)


def project_workspace_axis(start, axis, lower, upper):
    """Bound the axial displacement of the wrist root within the existing workspace box."""
    a, b = axis*(lower-start), axis*(upper-start)
    return float(np.minimum(a, b).sum()), float(np.maximum(a, b).sum())


def placement_bounds(hulls, nodes, palm_node, reserve, radius):
    """Bound continuous flexion plus arbitrary slides and rolls along the palm tangent."""
    pivot = nodes[palm_node]
    tangent = nodes[palm_node+1] - nodes[palm_node-1]
    norm = np.linalg.norm(tangent)
    if norm <= 1e-10:
        raise ValueError("nondegenerate measured palm tangent required")
    tangent /= norm
    transverse = np.eye(3) - np.outer(tangent, tangent)
    projected_nodes = (nodes - pivot) @ transverse
    radial = np.linalg.norm(projected_nodes, axis=1)
    world = np.stack([distance_lower(hull, nodes) for hull in hulls])
    transversal = np.stack([distance_lower((hull-pivot) @ transverse, projected_nodes) for hull in hulls])
    demand = radius + SHELL_M + reserve + NUMERIC_GUARD
    # Axial translations disappear upon projection. A tangent-axis roll
    # can move the inverse query by at most twice its transverse radius.
    cut = transversal.min(axis=0) - demand - 2.*radial
    displaced = np.maximum(world.min(axis=0) - demand, 0.)
    angles = [required_tilt(float(gap), float(lever), TRANSLATION_COMPARISON_M)
              for gap, lever in zip(displaced, np.linalg.norm(nodes-pivot, axis=1), strict=True)]
    axial_hulls = np.concatenate([(hull-pivot) @ tangent for hull in hulls])
    axial_nodes = (nodes-pivot) @ tangent
    slide = np.column_stack((axial_nodes-axial_hulls.max()-demand, axial_nodes-axial_hulls.min()+demand))
    return dict(fixed_pose_translation_lower_m=displaced.tolist(), tilt_lower_rad=angles,
                required_axial_slide_interval_m=slide.tolist(),
                translation_comparison_m=TRANSLATION_COMPARISON_M, tangent_world=tangent.tolist(),
                transverse_radius_m=radial.tolist(), rotation_invariant_transverse_reserve_m=cut.tolist(),
                geometric_witness_shell_m=SHELL_M, unsampled_point_travel_reserve_m=reserve)


def index_hulls(hand, state):
    """Sweep the complete originally admissible index flexion interval."""
    model = hand["model"]
    link_path = model["source_colliders"][INDEX]["source"]["body_path"]
    parents = {row[2]: row for row in model["fk"].edges}
    coupling = coupling_matrix()
    index = DISTAL_PAD_LINKS.index(INDEX) + 1
    while link_path in parents:
        joint, parent, _, _, _, axis = parents[link_path]
        if axis is not None:
            slope = coupling[JOINTS.index(joint)]
            if np.any(np.delete(slope, index)):
                raise ValueError("right index reach bound requires independent coupled finger kinematics")
        link_path = parent
    lower, upper = active_limits()
    angles = np.linspace(lower[index], upper[index], SAMPLES)
    q = state[6:].copy()
    q[index] = angles[0]
    delta = np.zeros(6)
    delta[index] = angles[-1] - angles[0]
    reserve = float(path_displacement_bound(model, q, delta, cooked=True)[INDEX] / (2*(SAMPLES-1)))
    wrist = route.root_pose(state, hand["seed"])
    hull = model["cooked_pads"][INDEX]["vertices"]
    result = []
    for angle in angles:
        q[index] = angle
        frame = fk_poses(model, wrist, q)[INDEX]
        result.append(hull @ frame[:3, :3].T + frame[:3, 3])
    return angles, result, reserve


def measured_index(row, model):
    """Measure actual cooked geometry on a recorded SDK row, without qualifying contact loads."""
    sdk = row["physics_step"]
    rod = row["rod"]
    geometry = row["sdk_hand_geometry"]
    if rod["physics_step"] != sdk or geometry["physics_step"] != sdk or row["guard_failure"] is not None:
        raise ValueError("right index geometry and intact rod must be from the same SDK tick")
    clock = _clock(lambda: rod)
    binding = rod["mesh_bindings"][f"right.{INDEX}"]
    locked = model["cooked_pads"][INDEX]
    if (binding["capture_sha256"] != locked["capture_sha256"]
            or binding["source_geometry_sha256"] != locked["geometry_sha256"]):
        raise ValueError("measured right index differs from the locked cooked hull")
    pose = as_matrix(*geometry["poses"][f"right.{INDEX}"])
    hull = locked["vertices"] @ pose[:3, :3].T + pose[:3, 3]
    nodes = np.asarray(rod["node_positions"], dtype=float)
    if nodes.shape != (1, 41, 3) or not np.isfinite(nodes).all():
        raise ValueError("complete measured rod nodes required")
    radius = .005
    bound = placement_bounds([hull], nodes[0], PALM_NODE, 0., radius)
    return dict(clock=clock, whole_wire_cooked_gap=endpoint_gap(hull, nodes[0], radius),
                transverse_geometry=bound, actual_load_verified=False)


def right_endpoint(hand, state, nodes, scene, candidate):
    status, original = search.evaluate(hand, state, nodes, scene, side="right")
    stations = {**hand["preload_targets"], **candidate}
    witnesses, blocked = {}, {}
    for name, node in stations.items():
        try:
            witness = preload.station_contact(hand, state, name, node, nodes, radius=scene["wire_radius_m"])
            witnesses[name] = witness
            limit = preload.MAX_PALM_PENETRATION_M if name == "base_link" else preload.PAD_PENETRATION_M
            preload.checked_terminal_witness(witness, node=node, max_penetration_m=limit)
        except ValueError as error:
            blocked[name] = str(error)
    targets = tuple(key for key in stations if key != "base_link")
    geometry = endpoint(hand["model"], route.root_pose(state, hand["seed"]), state[6:], nodes, scene,
                        hand["palm_grid"], targets=targets)
    return dict(passed=bool(not blocked and status in ("accepted_static", "rod_envelope")
                            and geometry["passed"] and geometry["open_hull_guards"]["hand_y_bounds_m"][1] < 0.),
                retained_status=status, thumb_ring_cosine=original.get("thumb_partner_cos"),
                witnesses=witnesses, station_failures=blocked, geometry=geometry)


def fit_right(hand, state, nodes, scene, axes, added, node, initial_angle, seed):
    """Propose a right wrist entrance; exact geometry and stations remain the gate."""
    stations = {**hand["preload_targets"], added: node}
    targets = tuple(name for name in stations if name != "base_link")
    model = hand["model"]
    pivot = nodes[PALM_NODE]
    radius = scene["wire_radius_m"]
    fields = {name: search.grid_field(hand["grids"][name], track_outside=True) for name in stations}
    base = state.copy()
    base[7+DISTAL_PAD_LINKS.index(added)] = initial_angle
    slots = np.array([0, *[DISTAL_PAD_LINKS.index(link)+1 for link in targets]])
    q_lo, q_hi = active_limits()
    lo = np.r_[np.full(6, -1.), (q_lo[slots]-base[6+slots])/.35]
    hi = np.r_[np.full(6, 1.), (q_hi[slots]-base[6+slots])/.35]
    hull_basis = {f"{kind}:{name}": (name, shape["vertices"])
                  for kind, group in (("source", model["source_colliders"]), ("cooked", model["cooked_pads"]))
                  for name, shape in group.items()}

    def target(x):
        trial = transform_wrist(base, hand["seed"], pivot, axes, WRIST_SCALE*x[:6])
        trial[6+slots] = base[6+slots] + x[6:]*.35
        return trial

    x0 = np.r_[seed, np.zeros(len(slots))]
    seed_target = target(x0)
    seed_frames = fk_poses(model, route.root_pose(seed_target, hand["seed"]), seed_target[6:])

    def world_hulls(frames):
        return {label: vertices @ frames[name][:3, :3].T + frames[name][:3, 3]
                for label, (name, vertices) in hull_basis.items()}

    planes = capsule_planes(world_hulls(seed_frames), nodes, radius, targets)

    def residual(x):
        trial = target(x)
        frames = fk_poses(model, route.root_pose(trial, hand["seed"]), trial[6:])
        hulls = world_hulls(frames)
        margins = [projected_segment_margin(hulls[label], nodes[segment:segment+2], normal,
                    radius=radius, required=rod_clearance_required(label, targets))
                   for (label, segment), normal in planes.items()]
        low, high = route.WRIST["palm_lower_m"], route.WRIST["palm_upper_m"]
        offset = trial[:3] - hand["origin"]
        margins.extend(np.r_[offset-low, high-offset, radius*(route.WRIST["rotation_limit_rad"]-np.abs(trial[3:6]))])
        margins.extend(points[:, 2].min()-scene["table_top_m"]-CLEARANCE_M for points in hulls.values())
        margins.append(-max(points[:, 1].max() for points in hulls.values()))
        gaps, normals = [], {}
        for name, material in stations.items():
            frame = frames[name]
            if name != "base_link":
                frame = frame @ hand["pad_offsets"][name]
            local = (nodes-frame[:3, 3]) @ frame[:3, :3]
            gap = fields[name](local)-radius
            gaps.append(gap[material]+.000012)
            margins.extend(np.delete(gap, material)-.000002)
            vec = search.sliding_world_normal(fields[name], local[material], frame[:3, :3])
            normals[name] = np.zeros(3) if vec is None else vec
        prior_partner, = set(hand["preload_targets"])-{"base_link", preload.THUMB}
        margins.append(radius*(search.OPPOSITION_COS_LIMIT-normals[preload.THUMB] @ normals[prior_partner]))
        return np.r_[np.asarray(gaps)*2e3, np.minimum(margins, 0.)*4e3, x*.005]

    fit = least_squares(residual, x0, bounds=(lo, hi), max_nfev=105)
    def record(solution, *, refreshed):
        trial = target(solution.x)
        result = right_endpoint(hand, trial, nodes, scene, {added: node})
        return dict(link=added, node=node, endpoint=result, state=trial.tolist(),
                    initial_flexion_rad=initial_angle, initial_wrist_seed=seed.tolist(),
                    attained_wrist_delta_m_rad=(solution.x[:6]*WRIST_SCALE).tolist(),
                    fit_evaluations=int(solution.nfev), fit_cost=float(solution.cost),
                    plane_count=len(planes), refreshed_planes=refreshed)

    first = record(fit, refreshed=False)
    results = [first]
    if added == LITTLE and not first["endpoint"]["passed"] and not first["endpoint"]["station_failures"]:
        arrived = target(fit.x)
        arrived_frames = fk_poses(model, route.root_pose(arrived, hand["seed"]), arrived[6:])
        planes.update(capsule_planes(world_hulls(arrived_frames), nodes, radius, targets))
        refreshed = least_squares(residual, fit.x, bounds=(lo, hi), max_nfev=105)
        results.append(record(refreshed, refreshed=True))
    return results


def inspect_native(receipt, sha, requested_sdk):
    outer = json.loads(checked_file(receipt, sha))
    native = receipt.parent / "native"
    if native.is_symlink() or outer.get("frozen_sha256") != FROZEN_SHA256:
        raise ValueError("the released Power handoff receipt and local native directory are required")
    inventory = outer["artifact_sha256"]

    def read(name):
        return json.loads(checked_file(native / name, inventory[name]))

    plan = read("frozen_plan.json")
    if digest(plan) != FROZEN_SHA256:
        raise ValueError("native artifact plan digest differs from the released flight")
    end = plan["budget"]["sdk_steps"]
    ticks = requested_sdk or (end-1, end)
    model = load_geometry(ROOT, side="right")
    rows = []
    for tick in ticks:
        row = decode_owned(read(f"power_control_{tick}.json"))
        if row["physics_step"] != tick:
            raise ValueError("receipt-bound raw row differs from requested SDK step")
        rows.append(measured_index(row, model))
    return dict(scope="native_measured_cooked_geometry_only", receipt_sha256=sha,
                frozen_sha256=FROZEN_SHA256, rows=rows, actual_load_verified=False)


def screen(args):
    for name, sha in PRIOR_CODE.items():
        checked_file(ROOT / name, sha)
    pinned = json.loads((ROOT / CONTRACT).read_bytes())["replay"]
    prior = checked_json(args.fixed_wrist_screen, FRONTIER_SHA)
    detour = checked_json(args.bilateral_detour, pinned["artifact_sha256"])
    base, base_sha, _, nodes, scene, hands, stations, _ = preload.load_inputs(
        args.palm_receipt, args.continuous_audit, args.partial_audit, ROOT)
    if (any(item["archive"] != base["archive"] or item["clock"] != base["clock"] for item in (prior, detour))
            or detour["base_route_sha256"] != base_sha or detour["source_stations"] != stations
            or prior["bilateral_detour_sha256"] != pinned["artifact_sha256"]
            or detour["left_target_waypoints"][-1] != prior["start_targets"]
            or detour["continuous_target_geometry_proved"] is not True or detour["fixed_rod_only"] is not True
            or sum(detour["control_windows_by_stage"]) != 963):
        raise ValueError("the exact published bilateral target and archived rod are required")
    row, _, _ = route.load_measured_palm_boundary(args.palm_receipt, project=ROOT)
    blocked = support_adjacent_nodes(row["rod"], nodes, scene)
    eligible = eligible_nodes(nodes, blocked, "right")
    begin = np.asarray(prior["start_targets"], dtype=float)
    retained = {}
    for index, side in enumerate(route.SIDES):
        status, evidence = search.evaluate(hands[side], begin[index], nodes, scene, side=side)
        if status != "accepted_static":
            raise ValueError(f"resting published partial geometry fails its original gate: {side} {status}")
        retained[side] = evidence
    right = hands["right"]
    if right["preload_targets"]["base_link"] != PALM_NODE:
        raise ValueError("the original right palm node is required")
    state = begin[1]
    axes = material_frame(nodes, PALM_NODE, retained["right"]["sdf"]["base_link"]["normal_world"])
    angles, hulls, reserve = index_hulls(right, state)
    bounds = placement_bounds(hulls, nodes, PALM_NODE, reserve, scene["wire_radius_m"])
    allowed = np.flatnonzero(eligible)
    cut = [int(node) for node in allowed if bounds["rotation_invariant_transverse_reserve_m"][node] > 0.]
    axis = np.asarray(bounds["tangent_world"])
    workspace = route.WRIST
    origin = right["origin"]
    slide_min, slide_max = project_workspace_axis(state[:3], axis, origin + workspace["palm_lower_m"],
                                                   origin + workspace["palm_upper_m"])
    axial_cut = [int(node) for node in allowed if (bounds["required_axial_slide_interval_m"][node][0] > slide_max
                                                 or bounds["required_axial_slide_interval_m"][node][1] < slide_min)]
    nearest = sorted(map(int, allowed), key=lambda node: bounds["fixed_pose_translation_lower_m"][node])
    unoccupied = next(node for node in nearest if node not in right["preload_targets"].values())
    closing_little = next(trial["angle_rad"] for trial in prior["hands"]["right"]["scans"][LITTLE]["trials"]
                          if trial["node"] == 6 and trial["target_gap_m"] == -.000006)
    little = state.copy()
    little[7+DISTAL_PAD_LINKS.index(LITTLE)] = closing_little
    fixed_little = right_endpoint(right, little, nodes, scene, {LITTLE: 6})
    results = []
    seeds = (np.array([0., 0., -.5, .40, 0., -.5]), np.array([.45, .4, .5, -.4, 0., .5]))
    for link, node in ((INDEX, nearest[0]), (INDEX, unoccupied), (LITTLE, 6)):
        if not eligible[node]:
            raise ValueError("search must bind an unobstructed node on the measured right physical half")
        initial_angle = closing_little
        if link == INDEX:
            sample = min(range(len(angles)), key=lambda i: distance_lower(hulls[i], nodes[node:node+1])[0])
            initial_angle = float(angles[sample])
        for seed in seeds:
            for result in fit_right(right, state, nodes, scene, axes, link, node, initial_angle, seed):
                if result["endpoint"]["passed"]:
                    target = np.stack((begin[0], result["state"]))
                    _, actions = route.compile_stage(begin, target, retreat_world_m=np.zeros((2, 3)),
                                                     flex_exponents=(1., 1.))
                    result["rate_checked_control_windows"] = len(actions)
                results.append(result)
                print(link, node, result["endpoint"]["passed"], result["fit_cost"], flush=True)
    measured_old = measured_index(row, load_geometry(ROOT, side="right"))
    return dict(scope="right_power_wrist_archived_geometry", archive=base["archive"], clock=base["clock"],
                detour_sha256=pinned["artifact_sha256"], source_sha256=PRIOR_CODE,
                right_palm_node=PALM_NODE, preserved_target_stations=stations,
                certified_start_targets=begin.tolist(), left_hand_held=True,
                angle_samples=SAMPLES, index_joint_interval_rad=angles[[0,-1]].tolist(),
                right_index_bounds=bounds, eligible_nodes=allowed.tolist(), transverse_roll_excluded_nodes=cut,
                original_workspace_axial_slide_m=[slide_min, slide_max], axial_workspace_excluded_nodes=axial_cut,
                axial_slide_roll_excluded_nodes=sorted(set(cut)|set(axial_cut)),
                nearest_nodes_by_translation_demand=nearest, fixed_wrist_little=fixed_little,
                orthogonal_wrist_trust_m_rad=WRIST_SCALE.tolist(), endpoint_trials=results,
                independent_endpoint_count=sum(item["endpoint"]["passed"] for item in results),
                historical_actual_pose_geometry=measured_old, native_runs=0, actual_load_verified=False,
                continuous_route_proved=False, full_wrap_verified=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    cpu = modes.add_parser("screen")
    for key in ("palm-receipt", "continuous-audit", "partial-audit", "bilateral-detour", "fixed-wrist-screen"):
        cpu.add_argument("--" + key, type=Path, required=True)
    native = modes.add_parser("inspect")
    native.add_argument("--receipt", type=Path, required=True)
    native.add_argument("--receipt-sha256", required=True)
    native.add_argument("--sdk-step", type=int, action="append")
    for mode in (cpu, native):
        mode.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = screen(args) if args.mode == "screen" else inspect_native(args.receipt, args.receipt_sha256, args.sdk_step)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if args.mode == "inspect" or result["independent_endpoint_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
