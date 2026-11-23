"""Restore right Power palm reserve after the certified fixed-rod target prefix."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import trimesh

from go2_pvcnn.control.dh116.linear_coupling import active_limits, expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from scripts import certify_dh116_power_thumb_route as route
from scripts import prepare_dh116_power_preload as preload
from scripts import search_dh116_power_surface_pairs as search
from scripts.audit_dh116_cooked_hull_device_parity import _pad_hull
from scripts.audit_dh116_pinch_initial_pose import checked_json
from scripts.audit_dh116_wire_contact_witness import shapes_world
from scripts.bake_dh116_power_right_palm_sdf import reference_distances
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_grid
from scripts.refit_dh116_power_left_contact import projected_segment_margin
from scripts.screen_dh116_power_measured_reseat import support_adjacent_nodes
from scripts.screen_dh116_power_partial_opposition import DISTAL_PAD_LINKS, endpoint, rod_clearance_required
from scripts.verify_dh116_power_bilateral_handoff import compile_commands
from scripts.verify_dh116_power_shared_rod import archive_geometry_digest
from tools.audit_dh116_power_coupled_wrist import capsule_planes
from tools.audit_dh116_power_remaining_digits import eligible_nodes
from tools.audit_dh116_power_synchronized_relocation import (
    RIGHT_PALM_COOKED_SHA, SDK_PER_WINDOW, check_workspace, compile_schedule, palm_geometry_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
BILATERAL = "docs/contracts/dh116_teacher_power_bilateral_detour_v1.json"
PREFIX_SHA = "ebaceeee4f56692937da2e9e5ce704a96e38b5fb21571cd2f7184a97691fe77f"
PREFIX_TOOL_SHA = "18c3273c91721f3ccbc0936971cd6a359e113cf8ca0af5be35ebe9daef00f387"
LITTLE = "finger53_link"
RING = "finger43_link"
CORE = {"base_link": 9, preload.THUMB: 4, RING: 3}
RETAINED = ("base_link", preload.THUMB)
CORE_PADS = (preload.THUMB, RING)
PALM_TARGET_M = -.000050


def owned_site(gaps, *, node, limit, must_touch):
    """Material ownership and depth checks on a sampled original SDF packet."""
    valid = np.isfinite(gaps)
    negative = np.flatnonzero(valid & (gaps < 0.)).tolist()
    if any(index != node for index in negative):
        failure = "foreign_material_station"
    elif any(gaps[index] < -limit for index in negative):
        failure = "original_penetration_limit"
    elif must_touch and negative != [node]:
        failure = "retained_site_open"
    else:
        failure = None
    return dict(failure=failure, negative_nodes=negative, owned_gap_m=float(gaps[node]) if valid[node] else None)


def hold_thumb(states, commands):
    """Switch only the right thumb target and commands to the inherited hold."""
    fixed, held = np.array(states, copy=True), np.array(commands, copy=True)
    fixed[:, 1, 6:8] = fixed[0, 1, 6:8]
    held[:, 6:8] = 0.
    compile_commands(((fixed, held),))
    return fixed, held


def digit_goal(start, link, target_angle):
    """Hold wrist and all other joint targets while closing the ring or little finger."""
    if link not in (RING, LITTLE):
        raise ValueError("only the ring or little finger is in scope")
    state = np.asarray(start, dtype=float).copy()
    slot = 7+DISTAL_PAD_LINKS.index(link)
    if not np.isfinite(target_angle) or target_angle <= state[slot]:
        raise ValueError("strictly closing coupled finger target required")
    state[slot] = target_angle
    expand(state[6:])
    return state


def reseat_target(hand, start, nodes, scene, reacquire_ring):
    """Find a minimal fixed-angle wrist translation plus incumbent flexion correction."""
    model = hand["model"]
    radius = scene["wire_radius_m"]
    targets = CORE if reacquire_ring else {name: CORE[name] for name in RETAINED}
    fields = {name: search.grid_field(hand["grids"][name], track_outside=True) for name in targets}
    low, high = active_limits()
    slots = np.array([0, 1, 4] if reacquire_ring else [0, 1])
    scale = np.r_[[.001]*3, [.03]*len(slots)]
    reserves = np.asarray([.0009, .0009, .0012])
    workspace = route.WRIST
    lower = np.r_[np.maximum(-reserves, hand["origin"]+workspace["palm_lower_m"]+.000002-start[:3]),
                  np.maximum(-.055, low[slots]-start[6+slots])]/scale
    upper = np.r_[np.minimum(reserves, hand["origin"]+workspace["palm_upper_m"]-.000002-start[:3]),
                  np.minimum(.055, high[slots]-start[6+slots])]/scale
    stored = dict((f"{kind}:{link}", (link, row["vertices"]))
                  for kind, group in (("source", model["source_colliders"]), ("cooked", model["cooked_pads"]))
                  for link, row in group.items())
    planes = capsule_planes(shapes_world(model, route.root_pose(start, hand["seed"]), start[6:]), nodes, radius, CORE_PADS)
    desired = {"base_link": PALM_TARGET_M, preload.THUMB: -.000045, RING: -.000035}

    def state(x):
        out = start.copy()
        delta = x*scale
        out[:3] += delta[:3]
        out[6+slots] += delta[3:]
        return out

    def residual(x):
        trial = state(x)
        frames = fk_poses(model, route.root_pose(trial, hand["seed"]), trial[6:])
        hulls = {label: vertices @ frames[link][:3, :3].T + frames[link][:3, 3]
                 for label, (link, vertices) in stored.items()}
        margins = [projected_segment_margin(hulls[label], nodes[i:i+2], normal, radius=radius,
                    required=rod_clearance_required(label, CORE_PADS))-.000002
                   for (label, i), normal in planes.items()]
        margins.extend(points[:, 2].min()-scene["table_top_m"]-CLEARANCE_M for points in hulls.values())
        margins.append(-max(points[:, 1].max() for points in hulls.values()))
        goals, normals = [], {}
        for name, node in targets.items():
            frame = frames[name]
            if name != "base_link":
                frame = frame @ hand["pad_offsets"][name]
            local = (nodes-frame[:3, 3]) @ frame[:3, :3]
            field = fields[name]
            gaps = field(local)-radius
            goals.append(gaps[node]-desired[name])
            margins.extend(np.delete(gaps, node)-.000002)
            direction = search.sliding_world_normal(field, local[node], frame[:3, :3])
            normals[name] = np.zeros(3) if direction is None else direction
        if reacquire_ring:
            margins.append(radius*(search.OPPOSITION_COS_LIMIT-normals[preload.THUMB] @ normals[RING]))
        return np.r_[np.asarray(goals)*5e3, np.minimum(margins, 0.)*8e3, x*.012]

    fit = least_squares(residual, np.clip(np.zeros(len(scale)), lower, upper), bounds=(lower, upper), max_nfev=100)
    return state(fit.x), dict(evaluations=int(fit.nfev), cost=float(fit.cost), capsule_planes=len(planes))


def sampled_trace(hand, states, nodes, scene, cooked, stations, *, ring_required, upper_window):
    """Track retained sites and flag the first regained ring or candidate little contact."""
    radius = scene["wire_radius_m"]
    steps = SDK_PER_WINDOW*(len(states)-1)
    first_touch = {}
    most_negative = {link: None for link in stations}
    worst_fan = 0.
    for name in stations:
        route.validate_palm_lattice(hand["grids"][name], radius=radius)
    for sdk in range(upper_window*SDK_PER_WINDOW+1):
        state = route.state_at(states, sdk/steps)
        frames = fk_poses(hand["model"], route.root_pose(state, hand["seed"]), state[6:])
        for name, node in stations.items():
            frame = frames[name]
            if name != "base_link":
                frame = frame @ hand["pad_offsets"][name]
            local = (nodes-frame[:3, 3]) @ frame[:3, :3]
            phi, normals, valid = sample_grid(hand["grids"][name], local, dtype=np.float64)
            gaps = np.where(valid, phi-radius, np.nan)
            limit = preload.MAX_PALM_PENETRATION_M if name == "base_link" else preload.PAD_PENETRATION_M
            must_touch = name in RETAINED or (name == RING and ring_required) or name in first_touch
            witness = owned_site(gaps, node=node, limit=limit, must_touch=must_touch)
            gap = witness["owned_gap_m"]
            if gap is not None:
                most_negative[name] = gap if most_negative[name] is None else min(gap, most_negative[name])
            if witness["failure"] is None and witness["negative_nodes"]:
                gradient = normals[node]
                if not np.isfinite(gradient).all() or np.linalg.norm(gradient) < 1e-12:
                    witness["failure"] = "invalid_contact_normal"
                else:
                    first_touch.setdefault(name, sdk)
            if name == "base_link" and witness["failure"] is None:
                fan_gap = reference_distances(cooked, local)-radius
                worst_fan = min(worst_fan, float(fan_gap.min()))
                fan = owned_site(fan_gap, node=node, limit=limit, must_touch=True)
                if fan["failure"] is not None:
                    witness = dict(fan, source="original_cooked_fan")
            if witness["failure"] is not None:
                return dict(last_verified_tick=sdk-1, first_contact_tick=first_touch, minimum_gaps_m=most_negative,
                            most_negative_fan_m=worst_fan, failure=dict(tick=sdk, link=name, **witness))
    return dict(last_verified_tick=upper_window*SDK_PER_WINDOW, first_contact_tick=first_touch,
                minimum_gaps_m=most_negative, most_negative_fan_m=worst_fan, failure=None)


def prove_blocks(hand, states, nodes, scene, targets, max_windows):
    """Independently replay geometric intervals and stop before failed clearance."""
    first = 0
    accepted, rejected = [], []
    while first < max_windows:
        width = min(max_windows-first, 8)
        while width:
            last = first+width
            result = route.prove_stage(hand, states[first:last+1], targets, nodes, scene,
                                       allow_concurrent=True, allow_joint_reversal=True,
                                       require_palm_reach=True, max_samples=512, max_seconds=13.)
            if result["endpoints_passed"] and result["verified"] is not None:
                accepted.append(dict(first_window=first, last_window=last, certificate=result))
                break
            rejected.append(dict(first_window=first, last_window=last, endpoints_passed=result["endpoints_passed"],
                                 violations=result["proof"]["violations"], unresolved=len(result["proof"]["unresolved_boxes"])))
            width //= 2
        if not width:
            break
        first = last
    return dict(windows=first, accepted_blocks=accepted, failed_blocks=rejected)


def endpoint_witness(hand, state, nodes, scene, stations, mesh):
    expected, failures = {}, {}
    for name, node in stations.items():
        try:
            expected[name] = preload.station_contact(hand, state, name, node, nodes, radius=scene["wire_radius_m"])
        except ValueError as error:
            failures[name] = str(error)
            expected[name] = dict(node=node, gap_m=None, negative_nodes=[], normal_world=None)
    normals = [expected[name]["normal_world"] for name in CORE_PADS]
    cosine = float(np.dot(*normals)) if all(axis is not None for axis in normals) else None
    pads = tuple(name for name in stations if name != "base_link")
    guard = endpoint(hand["model"], route.root_pose(state, hand["seed"]), state[6:], nodes, scene,
                     hand["palm_grid"], targets=pads)
    return dict(witnesses=expected, sampled_station_failures=failures,
                thumb_ring_cosine=cosine, original_hull_guard=guard,
                palm_checkpoint=palm_geometry_checkpoint(hand, state, nodes, scene["wire_radius_m"], mesh))


def assess_stage(hand, begin, goal, nodes, scene, cooked, stations, *, ring_required):
    states, commands, _ = compile_schedule(begin, [np.stack((begin[0], goal))])
    if check_workspace(states[:, 1], hand["origin"]) is not None:
        raise ValueError("appended command left original wrist bounds")
    trace = sampled_trace(hand, states[:, 1], nodes, scene, cooked, stations,
                          ring_required=ring_required, upper_window=min(len(commands), 64))
    targets = tuple(name for name in stations if name != "base_link")
    proof = prove_blocks(hand, states[:, 1], nodes, scene, targets,
                         max(0, trace["last_verified_tick"]//SDK_PER_WINDOW))
    count = proof["windows"]
    accepted = dict(targets=states[:count+1].tolist(), commands=commands[:count].tolist())
    if count*SDK_PER_WINDOW == trace["last_verified_tick"] and trace["failure"] is None:
        certified_trace = trace
    else:
        certified_trace = sampled_trace(hand, states[:, 1], nodes, scene, cooked, stations,
                                        ring_required=ring_required, upper_window=count)
    exact = endpoint_witness(hand, states[count, 1], nodes, scene, stations, cooked)
    frontier = None
    if count < len(commands):
        frontier = endpoint_witness(hand, states[count+1, 1], nodes, scene, stations, cooked)
    return dict(requested_windows=len(commands), sample_trace=trace, certified_sample_trace=certified_trace,
                proof=proof, accepted=accepted, exact_prefix_endpoint=exact, next_target_after_prefix=frontier)


def ring_trial(hand, begin, nodes, scene, cooked):
    """Find the original shallow ring-contact depth by an independent fixed-wrist closure."""
    start = begin[1]
    angle = start[7+DISTAL_PAD_LINKS.index(RING)]
    travel = min(.03, float(active_limits()[1][4]-angle))

    def gap(delta):
        goal = start if delta == 0. else digit_goal(start, RING, angle+delta)
        return preload.station_contact(hand, goal, RING, CORE[RING], nodes,
                                       radius=scene["wire_radius_m"])["gap_m"]

    try:
        delta = preload.solve_preload(gap, travel=travel)
    except ValueError as error:
        return dict(bracket_failure=str(error), exact_ring_reacquired=False, selected=False)
    trial = assess_stage(hand, begin, digit_goal(start, RING, angle+delta), nodes, scene, cooked, CORE,
                         ring_required=False)
    exact = trial["exact_prefix_endpoint"]
    ring = exact["witnesses"][RING]
    attained = bool(trial["proof"]["windows"] and ring["negative_nodes"] == [CORE[RING]]
                    and -preload.PAD_PENETRATION_M <= ring["gap_m"] < 0.)
    trial.update(exact_ring_reacquired=attained, selected=bool(attained and exact["thumb_ring_cosine"] is not None
                                                            and exact["thumb_ring_cosine"] <= search.OPPOSITION_COS_LIMIT))
    return trial


def audit(args):
    checked_file(ROOT/"tools/audit_dh116_power_synchronized_relocation.py", PREFIX_TOOL_SHA)
    artifact = checked_json(args.certified_prefix, PREFIX_SHA)
    ready = json.loads((ROOT/BILATERAL).read_bytes())["replay"]
    detour = checked_json(args.bilateral_detour, ready["artifact_sha256"])
    base, base_sha, _, nodes, scene, hands, stations, _ = preload.load_inputs(
        args.palm_receipt, args.continuous_audit, args.partial_audit, ROOT)
    if (any(item["archive"] != base["archive"] or item["clock"] != base["clock"] for item in (artifact, detour))
            or detour["base_route_sha256"] != base_sha or detour["source_stations"] != stations
            or artifact["detour_sha256"] != ready["artifact_sha256"]
            or detour["continuous_target_geometry_proved"] is not True or detour["fixed_rod_only"] is not True):
        raise ValueError("the original fixed rod and certified bilateral target must agree")
    prefix = artifact["candidates"]["common_release_synchronized"]
    cert = prefix["certified_commands"]
    states, commands, inverse = (np.asarray(cert[key], dtype=float) for key in
                                 ("target_waypoints", "normalized_24d", "exact_reverse_24d"))
    if (prefix["verified_windows"] != len(commands) or len(commands) != 5
            or not np.array_equal(states[0], detour["left_target_waypoints"][-1])
            or not np.array_equal(inverse, -commands[::-1])
            or not np.all(states[:, 0] == states[0, 0])):
        raise ValueError("require the exact five-window certified prefix")
    compile_commands(((states, commands), (states[::-1], inverse)))
    right = hands["right"]
    if right["preload_targets"] != CORE:
        raise ValueError("original right palm/thumb/ring ownership differs")
    row, _, _ = route.load_measured_palm_boundary(args.palm_receipt, project=ROOT)
    allowed = eligible_nodes(nodes, support_adjacent_nodes(row["rod"], nodes, scene), "right")
    if not all(allowed[node] for node in (*CORE.values(), 6)):
        raise ValueError("all Power sites must be on the original free right material/world half")
    raw = checked_json(args.right_palm_cooked_record, RIGHT_PALM_COOKED_SHA)
    binding = row["rod"]["mesh_bindings"]["right.base_link"]
    if (binding["capture_sha256"] != RIGHT_PALM_COOKED_SHA
            or binding["cooked_geometry_sha256"] != archive_geometry_digest(raw)
            or binding["sdf_sha256"] != right["grid_sha256"]["base_link"]
            or raw["source"]["geometry_sha256"] != binding["source_geometry_sha256"]
            or right["model"]["source_colliders"]["base_link"]["source"]["geometry_sha256"]
               != binding["source_geometry_sha256"]):
        raise ValueError("original cooked palm and registered SDF must bind the observed right palm")
    hull, _ = _pad_hull(raw)
    mesh = trimesh.Trimesh(vertices=hull["vertices"], faces=hull["faces"], process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("original cooked palm surface must be closed and consistently oriented")
    beginning = states[-1]
    original = endpoint_witness(right, beginning[1], nodes, scene, CORE, mesh)
    restore = {}
    for name, reacquire in (("palm_thumb", False), ("palm_thumb_ring", True)):
        target, fit = reseat_target(right, beginning[1], nodes, scene, reacquire)
        result = assess_stage(right, beginning, target, nodes, scene, mesh, CORE, ring_required=False)
        witness = result["exact_prefix_endpoint"]["witnesses"][RING]
        result.update(fit=fit, ring_reacquired=bool(result["proof"]["windows"]
                      and witness["negative_nodes"] == [CORE[RING]] and -preload.PAD_PENETRATION_M <= witness["gap_m"] < 0.))
        restore[name] = result
        print(name, result["proof"]["windows"], result["ring_reacquired"], result["sample_trace"]["failure"], flush=True)
    def rank(name):
        case = restore[name]
        return (case["proof"]["windows"] > 0, case["ring_reacquired"],
                case["exact_prefix_endpoint"]["witnesses"]["base_link"]["gap_m"] or -1.)
    selected = max(restore, key=rank)
    stage = restore[selected]
    comparison, ring_follow, added = None, None, None
    upper = active_limits()[1][5]
    if stage["proof"]["windows"]:
        held, _ = hold_thumb(stage["accepted"]["targets"], stage["accepted"]["commands"])
        comparison = dict(sample_trace=sampled_trace(right, held[:, 1], nodes, scene, mesh, CORE,
            ring_required=False, upper_window=len(held)-1),
            fixed_thumb_endpoint=endpoint_witness(right, held[-1, 1], nodes, scene, CORE, mesh))
        follow_start = np.asarray(stage["accepted"]["targets"][-1])
        ring_kept = stage["ring_reacquired"]
        if not ring_kept:
            ring_follow = ring_trial(right, follow_start, nodes, scene, mesh)
            print("fixed_wrist_ring", ring_follow["exact_ring_reacquired"], ring_follow["selected"], flush=True)
            if ring_follow["selected"]:
                follow_start = np.asarray(ring_follow["accepted"]["targets"][-1])
                ring_kept = True
        goal = digit_goal(follow_start[1], LITTLE, upper)
        added = assess_stage(right, follow_start, goal, nodes, scene, mesh, {**CORE, LITTLE: 6},
                             ring_required=ring_kept)
        endpoint_data = added["exact_prefix_endpoint"]
        end_little = endpoint_data["witnesses"][LITTLE]
        added["little_geometric_contact"] = bool(added["proof"]["windows"] and end_little["negative_nodes"] == [6]
                                   and -preload.PAD_PENETRATION_M <= end_little["gap_m"] < 0.)
        print("fixed_wrist_little", added["proof"]["windows"], added["little_geometric_contact"],
              added["sample_trace"]["failure"], flush=True)
    accepted = [(states, commands)]
    for result in (stage, ring_follow if ring_follow and ring_follow["selected"] else None, added):
        if result and result["proof"]["windows"]:
            accepted.append((np.asarray(result["accepted"]["targets"]), np.asarray(result["accepted"]["commands"])))
    forward = compile_commands(accepted)
    all_targets = np.vstack([accepted[0][0], *(part[0][1:] for part in accepted[1:])])
    reverse = -forward[::-1].copy()
    compile_commands(((all_targets, forward), (all_targets[::-1], reverse)))
    return dict(status="certified_reseat_extension" if len(forward) > len(commands) else "no_verified_extension",
        archive=base["archive"], clock=base["clock"], certified_prefix_sha256=PREFIX_SHA,
        detour_sha256=ready["artifact_sha256"], original_cutoff=original, selected_reseat=selected, reseat_proposals=restore,
        matched_thumb_hold=comparison, fixed_wrist_ring_trial=ring_follow, fixed_wrist_little=added,
        left_held=True, assembled_targets=all_targets.tolist(),
        assembled_forward_24d=forward.tolist(), assembled_reverse_24d=reverse.tolist(),
        original_windows=len(commands), appended_windows=len(forward)-len(commands), fixed_rod_only=True,
        material_ownership_and_cooked_fan_sampled=True, measured_loads=False, native_ready=False,
        full_wrap_verified=False, native_runs=0, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("palm-receipt", "continuous-audit", "partial-audit", "bilateral-detour", "certified-prefix",
                 "right-palm-cooked-record", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if result["appended_windows"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
