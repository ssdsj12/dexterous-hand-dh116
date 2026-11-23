"""Verify synchronized right Power palm relocation and multi-finger motion on the archived rod."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import least_squares, linprog

from go2_pvcnn.control.dh116.linear_coupling import active_limits, expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from scripts import certify_dh116_power_thumb_route as route
from scripts import prepare_dh116_power_preload as preload
from scripts import search_dh116_power_surface_pairs as search
from scripts.audit_dh116_pinch_initial_pose import checked_json
from scripts.audit_dh116_cooked_hull_device_parity import _pad_hull
from scripts.audit_dh116_wire_contact_witness import endpoint_gap, shapes_world
from scripts.bake_dh116_power_right_palm_sdf import reference_distances
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_grid
from scripts.refit_dh116_power_left_contact import projected_segment_margin
from scripts.screen_dh116_power_measured_reseat import support_adjacent_nodes
from scripts.screen_dh116_power_partial_opposition import rod_clearance_required
from scripts.verify_dh116_power_bilateral_handoff import compile_commands
from scripts.verify_dh116_power_shared_rod import archive_geometry_digest
from tools.audit_dh116_power_coupled_wrist import capsule_planes
from tools.audit_dh116_power_relocated_palm import exact_gate
from tools.audit_dh116_power_remaining_digits import eligible_nodes


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "docs/contracts/dh116_teacher_power_bilateral_detour_v1.json"
R5_TOOL_SHA = "0299e409b95a2b6ce11a12f9d6043ee45b4e221e33b524eb673f3a98350122ef"
R5_ENTRANCE_SHA = "367fe3f2ae5316b93f91ed4604b7aec1a1b13adc303eac7ef6768970b5059b55"
RIGHT_PALM_COOKED_SHA = "afbb3a2ce69d723edd9eb075dc0f214b163b6a30cc82d34c551a3f6a56156773"
GOAL_STATIONS = {"base_link": 11, "finger14_link": 8, "finger43_link": 8, "finger23_link": 4}
PALM_CORRIDOR = (9, 10, 11)
TARGETS = tuple(name for name in GOAL_STATIONS if name != "base_link")
LEFT_ACTIONS = [0, 1, 2, 3, 4, 5, 12, 13, 14, 18, 19, 20]
SDK_PER_WINDOW = 8
MAX_PROOF_WINDOWS = 32


def compile_schedule(start, goals):
    """Compile exact joins under the existing normalized 60 Hz target contract."""
    start = np.asarray(start, dtype=float)
    expected_left = start[0]
    pieces, blocks, joins = [start[None, ...]], [], []
    current = start
    for goal in goals:
        goal = np.asarray(goal, dtype=float)
        if not np.array_equal(goal[0], expected_left):
            raise ValueError("synchronized Power relocation must hold the left hand")
        states, commands = route.compile_stage(current, goal, retreat_world_m=np.zeros((2, 3)),
                                               flex_exponents=(1., 1.))
        pieces.append(states[1:])
        blocks.append((states, commands))
        joins.append((joins[-1] if joins else 0)+len(commands))
        current = goal
    return np.concatenate(pieces), compile_commands(blocks), joins


def palm_progress(prior, active, corridor):
    """No foreign palm contact or return to a station behind the sampled advance."""
    if any(node not in corridor for node in active):
        raise ValueError("palm loaded outside its owned material corridor")
    if len(active) > 1 and max(active)-min(active) > 1:
        raise ValueError("palm loaded nonadjacent owned material stations")
    latest = max(active, default=prior)
    if latest < prior:
        raise ValueError("palm contact retreated to an earlier material station")
    return latest


def check_workspace(states, origin):
    limits = route.WRIST
    bound = limits["rotation_limit_rad"]
    lower = np.r_[limits["palm_lower_m"], [-bound]*3]
    upper = np.r_[limits["palm_upper_m"], [bound]*3]
    target = np.c_[states[:, :3]-origin, states[:, 3:6]]
    axes = ("world_x", "world_y", "world_z", "wrist_x", "wrist_y", "wrist_z")
    for window, axis in np.argwhere((target < lower) | (target > upper)):
        return dict(window=int(window), axis=axes[axis], actual=float(target[window, axis]),
                    lower=float(lower[axis]), upper=float(upper[axis]))
    return None


def projected_goal(seed, hand):
    """Enforce original workspace independently of the normalized command rate."""
    goal = np.asarray(seed, dtype=float).copy()
    origin = hand["origin"]
    limits = route.WRIST
    reserve = .000002
    goal[:3] = np.clip(goal[:3], origin+limits["palm_lower_m"]+reserve,
                       origin+limits["palm_upper_m"]-reserve)
    angle = limits["rotation_limit_rad"] - 1e-8
    goal[3:6] = np.clip(goal[3:6], -angle, angle)
    expand(goal[6:])
    return goal


def common_release_direction(normals):
    """A bounded translation whose first-order SDF gap increases at all incumbent sites."""
    normals = np.asarray(normals, dtype=float)
    if normals.shape != (3, 3) or not np.isfinite(normals).all() or not np.allclose(
            np.linalg.norm(normals, axis=1), 1., atol=1e-8, rtol=0.):
        raise ValueError("three finite original unit contact normals required")
    fit = linprog([0., 0., 0., -1.], A_ub=np.c_[normals, np.ones(3)], b_ub=np.zeros(3),
                  bounds=[(-1., 1.), (-1., 1.), (0., 1.), (0., 1.)], method="highs")
    if not fit.success or fit.x[3] <= 1e-8:
        raise ValueError("no common non-downwards release direction for the original sites")
    direction = fit.x[:3]
    return direction/np.linalg.norm(direction)


def palm_geometry_checkpoint(hand, state, nodes, radius, cooked_mesh):
    """Measure separate source-envelope, original cooked fan and registered SDF evidence."""
    model = hand["model"]
    frame = fk_poses(model, route.root_pose(state, hand["seed"]), state[6:])["base_link"]
    source = model["source_colliders"]["base_link"]["vertices"]
    world = source @ frame[:3, :3].T + frame[:3, 3]
    local = (nodes-frame[:3, 3]) @ frame[:3, :3]
    cooked_gap = reference_distances(cooked_mesh, local)-radius
    signed, _, valid = sample_grid(hand["palm_grid"], local, dtype=np.float64)
    registered_gap = signed-radius
    return dict(source_convex_envelope=endpoint_gap(world, nodes, radius),
                cooked_original_triangle_fan=dict(minimum_node_gap_m=float(cooked_gap.min()),
                                                   negative_nodes=np.flatnonzero(cooked_gap < 0.).tolist()),
                registered_sdf=dict(minimum_sampled_node_gap_m=float(registered_gap[valid].min()) if valid.any() else None,
                                    negative_nodes=np.flatnonzero(valid & (registered_gap < 0.)).tolist()),
                triangle_fan_and_registered_sdf_scope="fixed_target_nodes_at_checkpoint")


def refine_anchor(hand, initial, nodes, scene, stations, *, position_trust_m):
    """Jointly refit all wrist and finger coordinates inside the original workspace."""
    radius = scene["wire_radius_m"]
    model = hand["model"]
    q_lo, q_hi = active_limits()
    workspace = route.WRIST
    angle_limit = workspace["rotation_limit_rad"]
    lower = np.r_[hand["origin"]+workspace["palm_lower_m"]+.000002, [-angle_limit+1e-8]*3, q_lo]
    upper = np.r_[hand["origin"]+workspace["palm_upper_m"]-.000002, [angle_limit-1e-8]*3, q_hi]
    # Millimetre translation conditioning, local wrist tilt, all six coupled inputs.
    scale = np.array([.001]*3+[.12]*3+[.35]*6)
    trust = np.array([position_trust_m]*3+[.17]*3+[.40]*6)
    low = np.maximum(lower-initial, -trust)/scale
    high = np.minimum(upper-initial, trust)/scale
    fields = {name: search.grid_field(hand["grids"][name], track_outside=True) for name in stations}
    seed_pose = route.root_pose(initial, hand["seed"])
    planes = capsule_planes(shapes_world(model, seed_pose, initial[6:]), nodes, radius, TARGETS)
    local_hulls = {f"{kind}:{name}": (name, shape["vertices"])
                   for kind, group in (("source", model["source_colliders"]), ("cooked", model["cooked_pads"]))
                   for name, shape in group.items()}

    def residual(x):
        target = initial + scale*x
        frames = fk_poses(model, route.root_pose(target, hand["seed"]), target[6:])
        hulls = {label: points @ frames[link][:3, :3].T + frames[link][:3, 3]
                 for label, (link, points) in local_hulls.items()}
        margins = [projected_segment_margin(hulls[label], nodes[i:i+2], normal, radius=radius,
                    required=rod_clearance_required(label, TARGETS)) for (label, i), normal in planes.items()]
        margins.extend(points[:, 2].min()-scene["table_top_m"]-CLEARANCE_M for points in hulls.values())
        margins.append(-max(points[:, 1].max() for points in hulls.values()))
        errors, normals = [], {}
        for name, node in stations.items():
            frame = frames[name]
            if name != "base_link":
                frame = frame @ hand["pad_offsets"][name]
            local = (nodes-frame[:3, 3]) @ frame[:3, :3]
            gaps = fields[name](local)-radius
            weights = {"base_link": 5., "finger23_link": 2.5}
            errors.append((gaps[node]+.000012)*weights.get(name, 1.))
            margins.extend(np.delete(gaps, node)-.000002)
            normal = search.sliding_world_normal(fields[name], local[node], frame[:3, :3])
            normals[name] = np.zeros(3) if normal is None else normal
        margins.append(radius*(search.OPPOSITION_COS_LIMIT - normals[preload.THUMB] @ normals["finger43_link"]))
        return np.r_[np.asarray(errors)*1e3, np.minimum(margins,0.)*5e3, x*.008]

    fit = least_squares(residual, np.clip(np.zeros(12), low, high), bounds=(low, high), max_nfev=100)
    return initial+scale*fit.x, dict(evaluations=int(fit.nfev), cost=float(fit.cost), planes=len(planes))


def corridor_midpoint(hand, start, finish, nodes, scene, retreat_m):
    """Coordinated joint midpoint, translated toward the measured intermediate palm station."""
    state = projected_goal((start+finish)/2, hand)
    model = hand["model"]
    field = search.grid_field(hand["palm_grid"], track_outside=True)
    radius = scene["wire_radius_m"]
    for _ in range(3):
        frame = fk_poses(model, route.root_pose(state, hand["seed"]), state[6:])["base_link"]
        material = (nodes[PALM_CORRIDOR[1]]-frame[:3, 3]) @ frame[:3, :3]
        gap = float(field(material)[0]-radius)
        normal = search.sliding_world_normal(field, material, frame[:3, :3])
        if normal is None:
            break
        state[:3] += np.clip(gap-retreat_m, -.001, .001)*normal
        state = projected_goal(state, hand)
    return state


def sample_corridor(right, states, nodes, radius, permitted, end_window):
    """Check the original SDFs at every 480 Hz chord sample through the proof budget."""
    for name in permitted:
        try:
            route.validate_palm_lattice(right["grids"][name], radius=radius)
        except ValueError as error:
            return dict(last_verified_tick=-1, failure=dict(link=name, reason=str(error)), palm_trace=[])
    last_palm = PALM_CORRIDOR[0]
    palm_trace = []
    limit_tick = SDK_PER_WINDOW*end_window
    for tick in range(limit_tick+1):
        fraction = tick / (SDK_PER_WINDOW*(len(states)-1))
        state = route.state_at(states, fraction)
        frames = fk_poses(right["model"], route.root_pose(state, right["seed"]), state[6:])
        for name, owned in permitted.items():
            frame = frames[name]
            if name != "base_link":
                frame = frame @ right["pad_offsets"][name]
            phi, gradient, valid = sample_grid(right["grids"][name],
                (nodes-frame[:3, 3]) @ frame[:3, :3], dtype=np.float64)
            negative = np.flatnonzero(valid & (phi < radius)).tolist()
            limit = preload.MAX_PALM_PENETRATION_M if name == "base_link" else preload.PAD_PENETRATION_M
            reason = None
            if any(node not in owned for node in negative):
                reason = "registered SDF touched a foreign material node"
            elif any(phi[node] < radius-limit for node in negative):
                reason = "registered SDF exceeded original penetration allowance"
            elif any(not np.isfinite(gradient[node]).all() or np.linalg.norm(gradient[node]) < 1e-12
                     for node in negative):
                reason = "registered SDF contact needs a finite normal"
            elif name == "base_link":
                try:
                    last_palm = palm_progress(last_palm, negative, PALM_CORRIDOR)
                except ValueError as error:
                    reason = str(error)
                if not palm_trace or negative != palm_trace[-1]["negative_nodes"]:
                    palm_trace.append(dict(tick=tick, negative_nodes=negative))
            if reason is not None:
                return dict(last_verified_tick=tick-1, failure=dict(tick=tick, link=name, reason=reason,
                    negative_nodes=negative), palm_trace=palm_trace)
    return dict(last_verified_tick=limit_tick, failure=None, palm_trace=palm_trace)


def prove_prefix(right, states, nodes, scene, max_windows):
    """Only advance when a complete geometric interval was independently replayed."""
    first = 0
    accepted, attempts = [], []
    while first < max_windows:
        width = min(max_windows-first, 4 if first else 1)
        while width:
            last = first+width
            proof = route.prove_stage(right, states[first:last+1], TARGETS, nodes, scene,
                                      allow_concurrent=True, allow_joint_reversal=True, require_palm_reach=False,
                                      max_samples=512, max_seconds=10.)
            proof_ok = proof["endpoints_passed"] and proof["verified"] is not None
            attempts.append(dict(first_window=first, last_window=last, verified=proof_ok,
                endpoints_passed=proof["endpoints_passed"], sampled_violations=proof["proof"]["violations"],
                unresolved_boxes=len(proof["proof"]["unresolved_boxes"])))
            if proof_ok:
                accepted.append(dict(first_window=first, last_window=last, certificate=proof))
                break
            width //= 2
        if not width:
            break
        first = last
    return dict(verified_windows=first, attempted=attempts, accepted_blocks=accepted)


def audit(args):
    checked_file(ROOT / "tools/audit_dh116_power_relocated_palm.py", R5_TOOL_SHA)
    candidate = checked_json(args.relocated_entrance, R5_ENTRANCE_SHA)
    contract = json.loads((ROOT / CONTRACT).read_bytes())["replay"]
    detour = checked_json(args.bilateral_detour, contract["artifact_sha256"])
    base, base_sha, _, nodes, scene, hands, stations, _ = preload.load_inputs(
        args.palm_receipt, args.continuous_audit, args.partial_audit, ROOT)
    if (any(item["archive"] != base["archive"] or item["clock"] != base["clock"] for item in (candidate, detour))
            or candidate["detour_sha256"] != contract["artifact_sha256"]
            or detour["base_route_sha256"] != base_sha or detour["source_stations"] != stations
            or detour["continuous_target_geometry_proved"] is not True or detour["fixed_rod_only"] is not True):
        raise ValueError("same archived rod and exact published bilateral target required")
    start = np.asarray(detour["left_target_waypoints"][-1], dtype=float)
    if not np.array_equal(candidate["left_held_target"], start[0]):
        raise ValueError("relocated entrance must preserve the published left target")
    for index, side in enumerate(route.SIDES):
        if search.evaluate(hands[side], start[index], nodes, scene, side=side)[0] != "accepted_static":
            raise ValueError(f"original {side} static gate failed")
    right = hands["right"]
    row, _, _ = route.load_measured_palm_boundary(args.palm_receipt, project=ROOT)
    free = eligible_nodes(nodes, support_adjacent_nodes(row["rod"], nodes, scene), "right")
    if right["preload_targets"]["base_link"] != PALM_CORRIDOR[0]:
        raise ValueError("original right palm station differs")
    source, = [item for item in candidate["trials"] if item["station_map"] == GOAL_STATIONS]
    possible = {"base_link": list(PALM_CORRIDOR)}
    for name in TARGETS:
        possible[name] = sorted({GOAL_STATIONS[name], right["preload_targets"].get(name, GOAL_STATIONS[name])})
    if not all(free[node] for owned in possible.values() for node in owned):
        raise ValueError("contact corridors must remain on the same free material and world-Y half")
    initial = projected_goal(source["fitted_12d_target"], right)
    refit, fit_record = refine_anchor(right, initial, nodes, scene, GOAL_STATIONS, position_trust_m=.005)
    middle = corridor_midpoint(right, start[1], refit, nodes, scene, retreat_m=.00035)
    palm_retreat = start[1].copy()
    retained = search.evaluate(right, start[1], nodes, scene, side="right")[1]["sdf"]
    normal = retained["base_link"]["normal_world"]
    palm_retreat[:3] -= .0012*np.asarray(normal)
    palm_retreat = projected_goal(palm_retreat, right)
    release = common_release_direction([retained[name]["normal_world"] for name in
                                       ("base_link", preload.THUMB, "finger43_link")])
    release_midpoint = start[1]+.12*(refit-start[1])
    release_midpoint[:3] += .0024*release
    release_midpoint = projected_goal(release_midpoint, right)
    release_first = start[1].copy()
    release_first[:3] += .0012*release
    release_first = projected_goal(release_first, right)
    archived_fan = checked_json(args.right_palm_cooked_record, RIGHT_PALM_COOKED_SHA)
    binding = row["rod"]["mesh_bindings"]["right.base_link"]
    if (binding["capture_sha256"] != RIGHT_PALM_COOKED_SHA
            or binding["cooked_geometry_sha256"] != archive_geometry_digest(archived_fan)
            or archived_fan["source"]["geometry_sha256"] != binding["source_geometry_sha256"]
            or binding["source_geometry_sha256"] != right["model"]["source_colliders"]["base_link"]["source"]["geometry_sha256"]
            or binding["sdf_sha256"] != right["grid_sha256"]["base_link"]):
        raise ValueError("source palm, cooked palm and original registered SDF do not share the archived binding")
    hull, _ = _pad_hull(archived_fan)
    cooked_mesh = trimesh.Trimesh(vertices=hull["vertices"], faces=hull["faces"], process=False)
    if not cooked_mesh.is_watertight or not cooked_mesh.is_winding_consistent:
        raise ValueError("original cooked palm triangle fan must be closed and consistently oriented")
    palm_start = palm_geometry_checkpoint(right, start[1], nodes, scene["wire_radius_m"], cooked_mesh)
    # Compare an actual common release with the single-normal retreat and the direct chords.
    alternatives = {
        "projected_entrance": [initial],
        "palm_prioritized_direct": [refit],
        "midpoint_corridor": [middle, refit],
        "palm_retreat_midpoint_corridor": [palm_retreat, middle, refit],
        "common_release_synchronized": [release_midpoint, middle, refit],
        "common_release_before_synchronizing": [release_first, release_midpoint, middle, refit],
    }
    results = {}
    for name, goals in alternatives.items():
        goals = [np.stack((start[0], state)) for state in goals]
        states, commands, joins = compile_schedule(start, goals)
        failure = check_workspace(states[:, 1], right["origin"])
        if failure is not None:
            results[name] = dict(workspace_failure=failure, verified_windows=0)
            continue
        limit_windows = min(MAX_PROOF_WINDOWS, len(commands))
        sampled = sample_corridor(right, states[:, 1], nodes, scene["wire_radius_m"], possible, limit_windows)
        proof = prove_prefix(right, states[:, 1], nodes, scene, max(0, sampled["last_verified_tick"]//SDK_PER_WINDOW))
        count = proof["verified_windows"]
        accepted = None
        simultaneous = 0
        if count:
            inverse = -commands[:count][::-1].copy()
            compile_commands(((states[:count+1], commands[:count]), (states[:count+1][::-1], inverse)))
            wrist = np.c_[commands[:count, 15:18], commands[:count, 21:24]]
            simultaneous = int(np.count_nonzero(np.any(wrist != 0., axis=1)
                                               & (np.count_nonzero(commands[:count, 6:12], axis=1) >= 2)))
            accepted = dict(target_waypoints=states[:count+1].tolist(), normalized_24d=commands[:count].tolist(),
                            exact_reverse_24d=inverse.tolist(), inverse_geometry_is_same_target_chords=True,
                            verified_palm_trace=[entry for entry in sampled["palm_trace"]
                                                 if entry["tick"] <= SDK_PER_WINDOW*count],
                            end_palm_checkpoint=palm_geometry_checkpoint(right, states[count, 1], nodes,
                                scene["wire_radius_m"], cooked_mesh))
        initial_rotation = route.root_pose(start[1], right["seed"])[3:]
        final_rotation = route.root_pose(goals[-1][1], right["seed"])[3:]
        relative_angle = float(2*np.arccos(np.clip(abs(initial_rotation @ final_rotation), 0., 1.)))
        terminal = exact_gate(right, goals[-1][1], nodes, scene, GOAL_STATIONS, relative_angle)
        results[name] = dict(full_proposal_windows=len(commands), proposal_join_windows=joins,
            goal_states=[goal[1].tolist() for goal in goals], corridor_samples=sampled, proof=proof,
            verified_windows=count, simultaneous_wrist_multi_finger_windows=simultaneous,
            certified_commands=accepted, terminal_exact=terminal)
        print(name, count, len(commands), sampled["failure"], flush=True)
    verified = max(entry["verified_windows"] for entry in results.values())
    return dict(status="certified_fixed_rod_prefix" if verified else "no_certified_relocation_prefix",
                archive=base["archive"], clock=base["clock"], source_entrance_sha256=R5_ENTRANCE_SHA,
                source_tool_sha256=R5_TOOL_SHA, detour_sha256=contract["artifact_sha256"], left_target_held=True,
                certified_start_targets=start.tolist(), workspace_projection_delta_m=(
                    initial[:3]-np.asarray(source["fitted_12d_target"][:3])).tolist(),
                right_material_corridors=possible, anchor_refit=fit_record, max_certificate_windows=MAX_PROOF_WINDOWS,
                common_release_world_unit=release.tolist(), cooked_palm_record_sha256=RIGHT_PALM_COOKED_SHA,
                start_palm_checkpoint=palm_start,
                candidates=results, longest_verified_windows=verified, fixed_rod_only=True,
                material_ownership_sampled_only=True, native_ready=False, actual_load_verified=False,
                native_runs=0, full_wrap_verified=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("palm-receipt", "continuous-audit", "partial-audit", "bilateral-detour", "relocated-entrance", "right-palm-cooked-record", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if result["longest_verified_windows"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
