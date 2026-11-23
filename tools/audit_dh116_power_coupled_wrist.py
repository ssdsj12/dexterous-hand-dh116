"""CPU refit of the left Power wrist and added pads on the certified archived rod."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import active_limits
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_polyline_closest_support import refine
from scripts import certify_dh116_power_thumb_route as route
from scripts import prepare_dh116_power_preload as preload
from scripts import search_dh116_power_surface_pairs as search
from scripts.audit_dh116_pinch_initial_pose import checked_json
from scripts.audit_dh116_wire_contact_witness import endpoint_gap, shapes_world
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_grid
from scripts.refit_dh116_power_left_contact import NEAR_ROD_M, projected_segment_margin
from scripts.screen_dh116_power_measured_reseat import support_adjacent_nodes
from scripts.screen_dh116_power_partial_opposition import DISTAL_PAD_LINKS, endpoint, rod_clearance_required
from scripts.verify_dh116_power_bilateral_handoff import compile_commands
from tools.audit_dh116_power_remaining_digits import eligible_nodes


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "docs/contracts/dh116_teacher_power_bilateral_detour_v1.json"
FRONTIER = "eff0f021b3dc16cdb77bed0775a5b11ff16a340eb5f8cb4768620eab0bd0b5bf"
ADDITIONS = {"finger33_link": 35, "finger43_link": 33}
WRIST_SCALE = np.array([.006] * 3 + [.10, .18, .10])
FLEX_SCALE = .25
MAX_ITERATIONS = 110
SDF_RESERVE_M = .000002


def material_frame(nodes, station, normal):
    """Orthonormal normal, increasing material tangent and transverse axes."""
    nodes, normal = np.asarray(nodes, dtype=float), np.asarray(normal, dtype=float)
    if (nodes.shape != (41, 3) or not np.isfinite(nodes).all() or normal.shape != (3,)
            or not np.isfinite(normal).all() or type(station) is not int or not 0 < station < len(nodes)-1):
        raise ValueError("finite measured rod and interior palm normal required")
    tangent = nodes[station+1] - nodes[station-1]
    length = np.linalg.norm(tangent)
    if length < 1e-9:
        raise ValueError("degenerate material tangent")
    tangent /= length
    transverse = np.cross(normal, tangent)
    length = np.linalg.norm(transverse)
    if length < 1e-9:
        raise ValueError("palm normal does not define a transverse material direction")
    transverse /= length
    return np.column_stack((np.cross(tangent, transverse), tangent, transverse))


def transform_wrist(start, seed, pivot, axes, delta):
    """Apply independent local translation and rotation around the material pivot."""
    attitude = Rotation.from_rotvec(axes @ delta[3:])
    current = seed * Rotation.from_euler("XYZ", start[3:6])
    target = start.copy()
    target[:3] = pivot + attitude.apply(start[:3]-pivot) + axes @ delta[:3]
    target[3:6] = (seed.inv() * attitude * current).as_euler("XYZ")
    return target


def command_pair(start, end):
    if not np.array_equal(start[1], end[1]):
        raise ValueError("the right Power hand must hold its certified target")
    waypoints, forward = route.compile_stage(start, end, retreat_world_m=np.zeros((2, 3)), flex_exponents=(1., 1.))
    inverse = -forward[::-1].copy()
    compile_commands(((waypoints, forward), (waypoints[::-1], inverse)))
    return waypoints, forward, inverse


def capsule_planes(hulls, nodes, radius, targets):
    """Collect local optimizer planes without assuming a new separating direction exists."""
    planes = {}
    for label, hull in hulls.items():
        if label == "source:base_link":
            continue
        threshold = rod_clearance_required(label, targets) + NEAR_ROD_M
        if endpoint_gap(hull, nodes, radius)["lower_m"] > threshold:
            continue
        packet = refine(hull, nodes, max_segments=12)
        for row, original in zip(packet["segments"], packet["original_certificate"]["segments"], strict=True):
            if row["lower_m"] <= threshold:
                normal = row["new_normal"] if row["new_normal"] is not None else original["normal"]
                planes[label, row["segment"]] = np.asarray(normal)
    return planes


def fit_endpoint(hand, start, nodes, scene, axes, additions, initial_flexions, local_seed):
    """Fit in the bounded material frame, then hand back original 12D targets."""
    station_map = {**hand["preload_targets"], **additions}
    palm_station = station_map["base_link"]
    pivot = nodes[palm_station]
    targets = tuple(link for link in station_map if link != "base_link")
    fields = {link: search.grid_field(hand["grids"][link], track_outside=True) for link in station_map}
    model = hand["model"]
    vertices = dict((f"{kind}:{link}", (link, collider["vertices"]))
                    for kind, meshes in (("source", model["source_colliders"]), ("cooked", model["cooked_pads"]))
                    for link, collider in meshes.items())
    active_slots = np.array([0] + [DISTAL_PAD_LINKS.index(link)+1 for link in targets])
    lower, upper = active_limits()
    base = start.copy()
    for link, angle in initial_flexions.items():
        base[7+DISTAL_PAD_LINKS.index(link)] = angle
    min_q = np.maximum(lower[active_slots], base[6+active_slots] - FLEX_SCALE)
    max_q = np.minimum(upper[active_slots], base[6+active_slots] + FLEX_SCALE)
    # A previously open auxiliary pad is free to close but cannot open past
    # the certified bilateral target during this local search.
    for link in additions:
        idx = DISTAL_PAD_LINKS.index(link)+1
        min_q[active_slots == idx] = max(float(min_q[active_slots == idx][0]), start[6+idx])
    bounds = np.column_stack((np.r_[np.full(6, -1.), (min_q-base[6+active_slots])/FLEX_SCALE],
                              np.r_[np.full(6, 1.), (max_q-base[6+active_slots])/FLEX_SCALE]))
    radius = scene["wire_radius_m"]
    angle_limit = route.WRIST["rotation_limit_rad"]
    lower_xyz = hand["origin"] + route.WRIST["palm_lower_m"]
    upper_xyz = hand["origin"] + route.WRIST["palm_upper_m"]
    seed_state = transform_wrist(base, hand["seed"], pivot, axes, local_seed * WRIST_SCALE)
    seed_hulls = shapes_world(model, route.root_pose(seed_state, hand["seed"]), seed_state[6:])
    planes = capsule_planes(seed_hulls, nodes, radius, targets)
    desired = np.array([-.000015 if link == "base_link" else -.000010 for link in station_map])

    def state(x):
        result = transform_wrist(base, hand["seed"], pivot, axes, x[:6]*WRIST_SCALE)
        result[6+active_slots] = base[6+active_slots] + x[6:]*FLEX_SCALE
        return result

    def measure(x):
        trial = state(x)
        frames = fk_poses(model, route.root_pose(trial, hand["seed"]), trial[6:])
        hulls = {key: local @ frames[link][:3, :3].T + frames[link][:3, 3]
                 for key, (link, local) in vertices.items()}
        margins = [projected_segment_margin(hulls[label], nodes[segment:segment+2], normal,
                    radius=radius, required=rod_clearance_required(label, targets)) - SDF_RESERVE_M
                   for (label, segment), normal in planes.items()]
        margins.extend(points[:, 2].min() - scene["table_top_m"] - CLEARANCE_M - SDF_RESERVE_M
                       for points in hulls.values())
        margins.extend((min(points[:, 1].min() for points in hulls.values()),))
        margins.extend(np.r_[trial[:3]-lower_xyz, upper_xyz-trial[:3],
                             radius*(angle_limit-np.abs(trial[3:6]))])
        gaps, normals = [], {}
        for link, node in station_map.items():
            frame = frames[link]
            if link != "base_link":
                frame = frame @ hand["pad_offsets"][link]
            local = (nodes-frame[:3, 3]) @ frame[:3, :3]
            gap = fields[link](local) - radius
            normal = search.sliding_world_normal(fields[link], local[node], frame[:3, :3])
            normals[link] = np.zeros(3) if normal is None else normal
            gaps.append(gap[node])
            limit = preload.MAX_PALM_PENETRATION_M if link == "base_link" else preload.PAD_PENETRATION_M
            margins.extend(np.r_[np.delete(gap, node)-SDF_RESERVE_M,
                                 -gap[node]-SDF_RESERVE_M, gap[node]+limit-SDF_RESERVE_M])
        partner, = set(hand["preload_targets"]) - {"base_link", preload.THUMB}
        margin = search.OPPOSITION_COS_LIMIT - normals[preload.THUMB] @ normals[partner]
        margins.append(radius*(margin-.002))
        return np.asarray(margins)*1e3, (np.asarray(gaps)-desired)*1e3

    cache_x = cache_value = None

    def cached(x):
        nonlocal cache_x, cache_value
        if cache_x is None or not np.array_equal(cache_x, x):
            cache_x, cache_value = x.copy(), measure(x)
        return cache_value

    def objective(x):
        residual = cached(x)[1]
        return residual @ residual + 1e-5 * (x @ x)

    initial = np.clip(np.r_[local_seed, np.zeros(len(active_slots))], bounds[:, 0], bounds[:, 1])
    warm = None
    if np.any(local_seed[[0, 2, 3, 5]]):
        def residual(x):
            margins, gaps = cached(x)
            return np.r_[np.minimum(margins, 0.) * 4., gaps, x * .005]

        warm = least_squares(residual, initial, bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=100)
        initial = warm.x
    result = minimize(objective, initial, bounds=bounds,
                      constraints=[{"type": "ineq", "fun": lambda x: cached(x)[0]}], method="SLSQP",
                      options=dict(maxiter=MAX_ITERATIONS, ftol=1e-10))
    return state(result.x), dict(converged=bool(result.success), iterations=int(result.nit),
        message=str(result.message), local_delta=(result.x[:6]*WRIST_SCALE).tolist(),
        start_seed_normalized=local_seed.tolist(), plane_count=len(planes),
        warm_start_evaluations=warm.nfev if warm is not None else 0,
        minimum_fit_margin_m=float(cached(result.x)[0].min()*1e-3), objective=float(result.fun))


def qualify_endpoint(hand, state, nodes, scene, additions):
    """Require the original retained hand gate, exact added stations and whole envelopes."""
    target_stations = {**hand["preload_targets"], **additions}
    witnesses, rejected = {}, {}
    for link, node in target_stations.items():
        limit = preload.MAX_PALM_PENETRATION_M if link == "base_link" else preload.PAD_PENETRATION_M
        try:
            witnesses[link] = preload.station_contact(hand, state, link, node, nodes, radius=scene["wire_radius_m"])
            preload.checked_terminal_witness(witnesses[link], node=node, max_penetration_m=limit)
        except ValueError as error:
            rejected[link] = str(error)
    original, retained = search.evaluate(hand, state, nodes, scene, side="left")
    pad_targets = tuple(link for link in target_stations if link != "base_link")
    geometry = endpoint(hand["model"], route.root_pose(state, hand["seed"]), state[6:], nodes, scene,
                        hand["palm_grid"], targets=pad_targets)
    # Re-evaluate the original opposition/workspace regardless of whether
    # its old clearance whitelist alone would reject the added pad.
    hard_scope_ok = original in ("accepted_static", "rod_envelope")
    registered_side = geometry["open_hull_guards"]["hand_y_bounds_m"][0] > 0.
    return dict(passed=bool(hard_scope_ok and not rejected and geometry["passed"] and registered_side),
                witnesses=witnesses, rejected_stations=rejected, original_scope=original,
                retained_thumb_partner_cosine=retained.get("thumb_partner_cos"), geometry=geometry)


def sample_route_stations(hand, waypoints, nodes, scene, additions):
    stations = {**hand["preload_targets"], **additions}
    radius = scene["wire_radius_m"]
    for link in stations:
        try:
            route.validate_palm_lattice(hand["grids"][link], radius=radius)
        except ValueError as error:
            return dict(passed=False, link=link, reason=str(error))
    partner, = set(hand["preload_targets"]) - {"base_link", preload.THUMB}
    steps = 8 * (len(waypoints)-1)
    for tick in range(steps+1):
        state = route.state_at(waypoints, tick/steps)
        frames = fk_poses(hand["model"], route.root_pose(state, hand["seed"]), state[6:])
        normals = {}
        for link, node in stations.items():
            frame = frames[link]
            if link != "base_link":
                frame = frame @ hand["pad_offsets"][link]
            phi, gradient, valid = sample_grid(
                hand["grids"][link], (nodes-frame[:3, 3]) @ frame[:3, :3], dtype=np.float64)
            active = np.flatnonzero(valid & (phi < radius)).tolist()
            limit = preload.MAX_PALM_PENETRATION_M if link == "base_link" else preload.PAD_PENETRATION_M
            if (any(idx != node for idx in active) or any(phi[idx] < radius-limit for idx in active)
                    or ((link not in additions or tick == steps) and active != [node])):
                return dict(passed=False, tick=tick, link=link, negative_nodes=active,
                            owned_gap_m=float(phi[node]-radius) if valid[node] else None)
            if node in active:
                vector = frame[:3, :3] @ gradient[node]
                norm = np.linalg.norm(vector)
                if not np.isfinite(norm) or norm < 1e-12:
                    return dict(passed=False, tick=tick, link=link, reason="missing_contact_normal")
                normals[link] = vector / norm
        cosine = float(normals[partner] @ normals[preload.THUMB])
        if cosine > search.OPPOSITION_COS_LIMIT:
            return dict(passed=False, tick=tick, reason="retained_opposition", cosine=cosine)
    return dict(passed=True, ticks=steps+1)


def audit(args, *, project=ROOT):
    pinned = json.loads((project / CONTRACT).read_bytes())["replay"]
    fixed = checked_json(args.fixed_wrist_screen, FRONTIER)
    detour = checked_json(args.bilateral_detour, pinned["artifact_sha256"])
    base, base_sha, _, nodes, scene, hands, stations, _ = preload.load_inputs(
        args.palm_receipt, args.continuous_audit, args.partial_audit, project)
    if (base["archive"] != detour["archive"] or base["clock"] != detour["clock"]
            or base["archive"] != fixed["archive"] or base["clock"] != fixed["clock"]
            or fixed["bilateral_detour_sha256"] != pinned["artifact_sha256"] or detour["base_route_sha256"] != base_sha
            or detour["continuous_target_geometry_proved"] is not True
            or detour["fixed_rod_only"] is not True or detour["source_stations"] != stations
            or detour["left_target_waypoints"][-1] != fixed["start_targets"]):
        raise ValueError("the measured SDK boundary and exact published bilateral target are required")
    start = np.asarray(fixed["start_targets"], dtype=float)
    row, _, _ = route.load_measured_palm_boundary(args.palm_receipt, project=project)
    eligible = eligible_nodes(nodes, support_adjacent_nodes(row["rod"], nodes, scene), "left")
    hand = hands["left"]
    retained = {}
    for index, side in enumerate(route.SIDES):
        status, evidence = search.evaluate(hands[side], start[index], nodes, scene, side=side)
        if status != "accepted_static":
            raise ValueError(f"published partial endpoint must pass the original exact gate: {side} {status}")
        retained[side] = evidence
    palm = hand["preload_targets"]["base_link"]
    axes = material_frame(nodes, palm, retained["left"]["sdf"]["base_link"]["normal_world"])
    results = []
    other = fixed["hands"]["left"]["scans"]
    presets = tuple(np.asarray(values) for values in (
        (0., 0., 0., 0., 0., 0.),
        (0., .22, 0., 0., -.4, 0.),
        (0., -.22, 0., 0., .4, 0.),
        (.20, 0., .24, .30, 0., -.45),
        (-.20, 0., -.24, -.30, 0., .45),
    ))
    for keys in (("finger33_link",), ("finger43_link",), tuple(ADDITIONS)):
        additions = {link: ADDITIONS[link] for link in keys}
        if not all(eligible[node] for node in (*additions.values(), *hand["preload_targets"].values())):
            raise ValueError("all target stations must be on the measured, unobstructed left physical wire half")
        flexions = {link: next(trial["angle_rad"] for trial in other[link]["trials"]
                             if trial["node"] == node and trial["target_gap_m"] == -.000006)
                    for link, node in additions.items()}
        for seed in presets:
            state, fit = fit_endpoint(hand, start[0], nodes, scene, axes, additions, flexions, seed)
            endpoint_gate = qualify_endpoint(hand, state, nodes, scene, additions)
            item = dict(new_stations=additions, state=state.tolist(), fit=fit, endpoint=endpoint_gate)
            if endpoint_gate["passed"]:
                target = np.stack((state, start[1]))
                points, forward, inverse = command_pair(start, target)
                sampled = sample_route_stations(hand, points[:, 0], nodes, scene, additions)
                item["sampled_route"] = sampled
                if sampled["passed"]:
                    targets = tuple(link for link in (*hand["preload_targets"], *additions) if link != "base_link")
                    proof = route.prove_stage(hand, points[:, 0], targets, nodes, scene, allow_concurrent=True,
                                              max_samples=2048, max_seconds=40.)
                    item["continuous_hull_proof"] = proof
                    if proof["endpoints_passed"] and proof["verified"]:
                        item["geometry_route"] = dict(waypoints=points.tolist(), forward=forward.tolist(),
                                                      inverse=inverse.tolist(), control_hz=route.CONTROL_HZ)
            results.append(item)
            print(keys, seed.tolist(), fit["minimum_fit_margin_m"], endpoint_gate["passed"],
                  "geometry_route" in item, flush=True)
    endpoint_count = sum(item["endpoint"]["passed"] for item in results)
    route_count = sum("geometry_route" in item for item in results)
    return dict(version="dh116-power-coupled-wrist-screen-v1",
                status="fixed_rod_direction_certified" if route_count else
                "static_endpoint_without_route" if endpoint_count else "blocked_coupled_wrist_screen",
                archive=base["archive"], clock=base["clock"], detour_sha256=pinned["artifact_sha256"],
                fixed_wrist_screen_sha256=FRONTIER, material_axes=axes.tolist(), wrist_trust_m_rad=WRIST_SCALE.tolist(),
                fixed_rod_only=True, right_hand_held=True, retained_stations=stations, results=results,
                endpoint_count=endpoint_count, route_count=route_count, native_runs=0, full_wrap_verified=False,
                actual_load_verified=False, grasp_qualified=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("palm-receipt", "continuous-audit", "partial-audit", "bilateral-detour", "fixed-wrist-screen", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if result["route_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
