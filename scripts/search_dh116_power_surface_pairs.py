"""Search bilateral Power contact poses using additional local pad surfaces."""

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import active_limits
from go2_pvcnn.control.dh116.tabletop_polyline_support_certificate import certify
from scripts.audit_dh116_pinch_initial_pose import checked_json, pose_matrix
from scripts.audit_dh116_wire_contact_witness import endpoint_gap, shapes_world
from scripts import prepare_dh116_power_preload as preload
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_grid
from scripts.screen_dh116_power_partial_opposition import endpoint

route = preload.route
READINESS = "docs/contracts/dh116_teacher_power_preload_envelope_v1.json"
RAYS_PER_PAD = 10
STATE_RADIUS_RAD = .35
REFIT_BUDGET_PER_SHAPE = 2
REFIT_RESERVE_M = .00002


def anchored_translation(rotation, point_in_hand, node_world):
    """Place one owned wire node at the given wrist-local point."""
    return node_world - rotation @ point_in_hand


def plane_ray_candidates(source, support, directions, grid, *, radius):
    """Generate node points on the locked SDF with a local support-plane reserve."""
    protrusion = (source @ directions.T).max(axis=0) - np.sum(support * directions, axis=1)
    lo = np.full(len(directions), radius - .0004)
    hi = np.full(len(directions), radius + .0004)
    target = radius + preload.TARGET_GAP_M

    def sample(length):
        return sample_grid(grid, support + length[:, None] * directions, dtype=np.float64)

    left, _, left_valid = sample(lo)
    right, _, right_valid = sample(hi)
    valid = left_valid & right_valid & (left <= target) & (target <= right)
    for _ in range(25):
        length = (lo + hi) / 2.
        phi, _, inside = sample(length)
        valid &= inside
        advance = phi < target
        lo = np.where(advance, length, lo)
        hi = np.where(advance, hi, length)
    length = (lo + hi) / 2.
    _, gradient, inside = sample(length)
    norm = np.linalg.norm(gradient, axis=1)
    alignment = np.divide(np.sum(gradient * directions, axis=1), norm,
                          out=np.zeros_like(norm), where=norm > 0.)
    # The supporting plane is only a seed generator. Final clearance to the
    # whole curved rod uses the actual source and cooked hulls at the fitted FK.
    retained = (valid & inside & (alignment >= .98)
                & (lo - protrusion > radius - preload.PAD_PENETRATION_M + .00001))
    return (support + length[:, None] * directions)[retained], directions[retained]


def pad_rays(geometry, grid, link, *, radius):
    count = 1536
    index = np.arange(count) + .5
    z = 1. - 2. * index / count
    phi = index * np.pi * (3. - np.sqrt(5.))
    ring = np.sqrt(1. - z**2)
    lattice = np.column_stack((ring * np.cos(phi), ring * np.sin(phi), z))
    source = geometry["source_colliders"][link]["vertices"]
    cooked = geometry["cooked_pads"][link]["vertices"]
    hull = ConvexHull(cooked)
    normals = hull.equations[:, :3]
    directions = np.vstack((lattice, normals))
    vertices = cooked[(cooked @ directions.T).argmax(axis=0)]
    # Face-centre rays allow contacts away from the arbitrary support vertex.
    support = np.vstack((vertices, cooked[hull.simplices].mean(axis=1)))
    return plane_ray_candidates(source, support, np.vstack((directions, normals)), grid, radius=radius)


def nearby_rays(points, normals, origin, normal):
    angle = np.arccos(np.clip(normals @ normal, -1., 1.))
    ranked = np.argsort(np.linalg.norm(points - origin, axis=1) + .008 * angle)
    chosen = []
    for i in ranked:
        if all(np.linalg.norm(points[i] - points[j]) > .0012 or normals[i] @ normals[j] < .995
               for j in chosen):
            chosen.append(i)
        if len(chosen) == RAYS_PER_PAD:
            break
    return np.stack((points[chosen], normals[chosen]), axis=1)


def axial_contact_error(world_normals, material_tangents, *, radius):
    """Express a tilt toward the material tangent as a radius-scaled residual."""
    return radius * np.sum(world_normals * material_tangents, axis=1)


def separation_deficit(vertices, first, second, normals, *, radius, required):
    """Soft separation penalty on the active rod segment and its valid supports."""
    projections = (vertices @ normals.T).min(axis=0)
    segment = np.maximum(first @ normals.T, second @ normals.T)
    gap = (projections - segment).max() - radius
    return max(0., required - gap)


def selected_station_residuals(gaps, *, owned_node, target):
    """Keep the selected pad on its original material node without pinning a surface ray."""
    residuals = np.maximum(-gaps, 0.)
    residuals[owned_node] = gaps[owned_node] - target
    return residuals


def grid_field(grid):
    axes = tuple(grid["origin"][dim] + float(grid["spacing"]) * np.arange(n)
                 for dim, n in enumerate(grid["values"].shape))
    return RegularGridInterpolator(axes, grid["values"], bounds_error=False, fill_value=.02)


def fit_pair(hand, start, nodes, thumb_ray, partner_ray, *, radius, align_wire_normals, planes=(),
             sliding_contact=False):
    """Fit wrist and active joints while preserving live material station identities."""
    targets = hand["preload_targets"]
    model = hand["model"]
    palm_node = nodes[targets["base_link"]]
    partner, = set(targets) - {"base_link", preload.THUMB}
    node_ids = [targets[name] for name in (preload.THUMB, partner)]
    thumb_node, partner_node = nodes[node_ids]
    tangents = np.stack([nodes[index + 1] - nodes[index - 1] for index in node_ids])
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    inverse = np.linalg.inv(pose_matrix(model["pose"]))
    seed = hand["seed"].as_matrix()
    active = start[6:]
    slots = np.arange(6) if planes else np.array([0, 1, preload.DISTAL_PAD_LINKS.index(partner) + 1])
    centre = np.r_[start[3:6], active[slots]]
    low, high = active_limits()
    angle_limit = route.WRIST["rotation_limit_rad"]
    lower = np.maximum(centre - STATE_RADIUS_RAD, np.r_[[-angle_limit] * 3, low[slots]])
    upper = np.minimum(centre + STATE_RADIUS_RAD, np.r_[[angle_limit] * 3, high[slots]])
    palm_field = grid_field(hand["palm_grid"])
    partner_field = grid_field(hand["grids"][partner]) if sliding_contact else None
    shaped = {"source": model["source_colliders"], "cooked": model["cooked_pads"]}
    vertices = []
    for plane in planes:
        kind, link = plane["shape"].split(":", 1)
        vertices.append((link, shaped[kind][link]["vertices"]))

    def state_at(var):
        q = active.copy()
        q[slots] = var[3:]
        fk = {path.rsplit("/", 1)[-1]: inverse @ pose for path, pose in model["fk"].poses(q).items()}
        attitude = seed @ Rotation.from_euler("XYZ", var[:3]).as_matrix()
        thumb_fk = fk[preload.THUMB] @ hand["pad_offsets"][preload.THUMB]
        partner_fk = fk[partner] @ hand["pad_offsets"][partner]
        origin = anchored_translation(attitude, thumb_fk[:3, :3] @ thumb_ray[0] + thumb_fk[:3, 3], thumb_node)
        partner_rotation = attitude @ partner_fk[:3, :3]
        partner_origin = attitude @ partner_fk[:3, 3] + origin
        projected_partner = partner_rotation @ partner_ray[0] + partner_origin
        station_penalty = np.empty(0)
        if partner_field is not None:
            local_nodes = (nodes - partner_origin) @ partner_rotation
            station_penalty = selected_station_residuals(partner_field(local_nodes) - radius,
                                                         owned_node=node_ids[1], target=preload.TARGET_GAP_M)
        normal_world = np.stack((attitude @ thumb_fk[:3, :3] @ thumb_ray[1],
                                 attitude @ partner_fk[:3, :3] @ partner_ray[1]))
        palm = fk["base_link"]
        palm_rotation = attitude @ palm[:3, :3]
        palm_origin = attitude @ palm[:3, 3] + origin
        local_palm_node = (palm_node - palm_origin) @ palm_rotation
        penalties = []
        for plane, (link, points) in zip(planes, vertices, strict=True):
            frame = fk[link]
            # The model vertices already carry the measured pad alignment.
            world = points @ (attitude @ frame[:3, :3]).T + attitude @ frame[:3, 3] + origin
            segment = plane["segment"]
            penalties.append(separation_deficit(
                world, nodes[segment], nodes[segment + 1], np.asarray(plane["normals"]),
                radius=radius, required=plane["required_m"] + REFIT_RESERVE_M))
        return (np.r_[origin, var[:3], q], projected_partner - partner_node,
                float(palm_field(local_palm_node)[0]), normal_world, penalties, station_penalty)

    def residual(var):
        state, partner_error, palm_distance, normals, penalties, station_penalty = state_at(var)
        offset = state[:3] - hand["origin"]
        reserve = .00003 if sliding_contact else 0.
        over = np.maximum(offset - np.asarray(route.WRIST["palm_upper_m"]) + reserve, 0.)
        under = np.maximum(np.asarray(route.WRIST["palm_lower_m"]) + reserve - offset, 0.)
        axial = axial_contact_error(normals, tangents, radius=radius) * 1e3 if align_wire_normals else np.empty(0)
        ray_weight, wrist_weight, axial_weight = (1e2, 12e3, .4) if sliding_contact else (1e3, 4e3, 1.)
        return np.r_[partner_error * ray_weight, (palm_distance - radius - preload.TARGET_GAP_M) * 1e3,
                     (over - under) * wrist_weight, axial * axial_weight, np.asarray(penalties) * 2e3,
                     station_penalty * 2e3, (var - centre) * .04]

    solution = least_squares(residual, centre, bounds=(lower, upper), max_nfev=40,
                             ftol=2e-7, xtol=2e-7, gtol=2e-7)
    state, error, _, _, _, _ = state_at(solution.x)
    return state, float(np.linalg.norm(error))


def evaluate(hand, state, nodes, scene, *, side):
    radius = scene["wire_radius_m"]
    targets = hand["preload_targets"]
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (12,) or not np.isfinite(state).all():
        return "invalid_state", {}
    offset = state[:3] - hand["origin"]
    if (np.any(offset < route.WRIST["palm_lower_m"])
            or np.any(offset > route.WRIST["palm_upper_m"])
            or np.any(np.abs(state[3:6]) > route.WRIST["rotation_limit_rad"])):
        return "wrist_workspace", {}
    try:
        route.expand(state[6:])
    except ValueError as error:
        return "coupled_joint_limits", {"reason": str(error)}
    try:
        measurements = {link: preload.station_contact(hand, state, link, node, nodes, radius=radius)
                        for link, node in targets.items()}
        for link, item in measurements.items():
            limit = preload.MAX_PALM_PENETRATION_M if link == "base_link" else preload.PAD_PENETRATION_M
            preload.checked_terminal_witness(item, node=targets[link], max_penetration_m=limit)
    except ValueError as error:
        return "locked_sdf_or_station", {"reason": str(error)}
    distal = tuple(link for link in targets if link != "base_link")
    cosine = float(np.dot(*(measurements[link]["normal_world"] for link in distal)))
    evidence = dict(sdf=measurements, thumb_partner_cos=cosine)
    if cosine > -.25:
        return "not_opposed", evidence
    pose = route.root_pose(state, hand["seed"])
    result = endpoint(hand["model"], pose, state[6:], nodes, scene, hand["palm_grid"], targets=distal)
    evidence["geometry"] = result
    if not result["passed"]:
        return "rod_envelope" if result["limiting_rod"] else "table_support_or_hand", evidence
    low, high = result["open_hull_guards"]["hand_y_bounds_m"]
    if (low <= 0. if side == "left" else high >= 0.):
        return "bilateral_separation", evidence
    return "accepted_static", evidence


def active_separation_planes(hand, state, nodes, scene, requirements):
    """Refresh separating directions on the currently limiting original capsules."""
    pose = route.root_pose(state, hand["seed"])
    shapes = shapes_world(hand["model"], pose, state[6:])
    planes = []
    for shape, required in sorted(requirements.items()):
        hull = shapes[shape]
        gap = endpoint_gap(hull, nodes, scene["wire_radius_m"])
        witness = gap["witness"]
        segments = certify(hull, nodes)["segments"]
        if witness is not None:
            segment = segments[witness["segment"]]
        else:
            segment = min(segments, key=lambda result: result["lower_m"])
        directions = [segment["normal"]]
        if witness is not None:
            delta = np.asarray(witness["hull_point_m"]) - np.asarray(witness["rod_axis_point_m"])
            length = np.linalg.norm(delta)
            if length > 1e-9:
                directions.append((delta / length).tolist())
        planes.append(dict(shape=shape, required_m=required, segment=segment["segment"],
                           normals=directions, reference_lower_m=gap["lower_m"],
                           reference_upper_m=gap["upper_m"]))
    return planes


def refit_candidate(hand, seed, nodes, scene, rays, *, side, rounds, align_wire_normals, sliding_contact):
    """Try a finite number of clearance-aware fits without changing the hard gate."""
    state = np.array(seed["state"])
    previous_limiting = seed["geometry"]["limiting_rod"]
    requirements = {previous_limiting["shape"]: previous_limiting["required_m"]}
    attempts = []
    for _ in range(rounds):
        planes = active_separation_planes(hand, state, nodes, scene, requirements)
        state, error = fit_pair(hand, state, nodes, *rays, radius=scene["wire_radius_m"],
                                align_wire_normals=align_wire_normals, planes=planes, sliding_contact=sliding_contact)
        result, evidence = evaluate(hand, state, nodes, scene, side=side)
        accepted = dict(state=state.tolist(), partner_fit_m=error, result=result, planes=planes, **evidence)
        attempts.append(accepted)
        if result == "accepted_static":
            break
        geometry = evidence.get("geometry")
        if geometry is None:
            pose = route.root_pose(state, hand["seed"])
            targets = tuple(link for link in hand["preload_targets"] if link != "base_link")
            geometry = endpoint(hand["model"], pose, state[6:], nodes, scene, hand["palm_grid"], targets=targets)
            accepted["geometry"] = geometry
        limiting = geometry["limiting_rod"]
        if limiting is None:
            break
        requirements[limiting["shape"]] = limiting["required_m"]
    return attempts


def search_side(project, side, hand, start, nodes, scene, prior, *, align_wire_normals,
                clearance_rounds=0, sliding_contact=False):
    geometry = route.load_geometry(project, side=side)  # Untransformed vertices in the registered SDF basis.
    fk = fk_poses(hand["model"], route.root_pose(start, hand["seed"]), start[6:])
    rays = {}
    for link, node in hand["preload_targets"].items():
        if link == "base_link":
            continue
        points, normals = pad_rays(geometry, hand["grids"][link], link, radius=scene["wire_radius_m"])
        transform = fk[link] @ hand["pad_offsets"][link]
        rotation = transform[:3, :3]
        original_point = (nodes[node] - transform[:3, 3]) @ rotation
        original_normal = np.asarray(prior[link]["normal_world"]) @ rotation
        rays[link] = nearby_rays(points, normals, original_point, original_normal)
    partner, = set(rays) - {preload.THUMB}
    accepted = []
    rejected = Counter()
    limiting_shapes = Counter()
    below_upper_bound = Counter()
    best_error = None
    fits_at_geometry_gate = 0
    examples = {}
    refit_seeds = {}
    for (i, thumb), (j, finger) in product(enumerate(rays[preload.THUMB]), enumerate(rays[partner])):
        state, error = fit_pair(hand, start, nodes, thumb, finger, radius=scene["wire_radius_m"],
                                align_wire_normals=align_wire_normals)
        best_error = error if best_error is None else min(best_error, error)
        if error > .0001:
            rejected["partner_fit"] += 1
            continue
        fits_at_geometry_gate += 1
        result, evidence = evaluate(hand, state, nodes, scene, side=side)
        record = dict(ray_indices=[i, j], partner_fit_m=error, state=state.tolist(), **evidence)
        if result == "accepted_static":
            accepted.append(record)
        else:
            rejected[result] += 1
            if result == "rod_envelope":
                bound = evidence["geometry"]["limiting_rod"]
                limiting_shapes[bound["shape"]] += 1
                if bound["upper_m"] < bound["required_m"]:
                    below_upper_bound[bound["shape"]] += 1
                if clearance_rounds:
                    refit_seeds.setdefault(bound["shape"], []).append(record)
            if result not in examples or error < examples[result]["partner_fit_m"]:
                examples[result] = record
    refits = []
    for shape, options in sorted(refit_seeds.items()):
        def proximity(record):
            bound = record["geometry"]["limiting_rod"]
            return (bound["upper_m"] - bound["required_m"], -record["partner_fit_m"])

        for seed_record in sorted(options, key=proximity, reverse=True)[:REFIT_BUDGET_PER_SHAPE]:
            i, j = seed_record["ray_indices"]
            attempts = refit_candidate(hand, seed_record, nodes, scene,
                                       (rays[preload.THUMB][i], rays[partner][j]),
                                       side=side, rounds=clearance_rounds, align_wire_normals=align_wire_normals,
                                       sliding_contact=sliding_contact)
            refits.append(dict(initial_shape=shape, ray_indices=[i, j], initial=seed_record, attempts=attempts))
            if attempts[-1]["result"] == "accepted_static":
                accepted.append(dict(ray_indices=[i, j], **attempts[-1]))
    return dict(selected_rays={link: points.tolist() for link, points in rays.items()},
                best_partner_fit_m=best_error, evaluated_geometry=fits_at_geometry_gate,
                rejection_counts=dict(rejected), rejected_examples=examples, accepted=accepted,
                limiting_rod_shapes=dict(limiting_shapes), proved_below_rod_limit=dict(below_upper_bound),
                clearance_refits=refits)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("palm-receipt", "continuous-audit", "partial-audit", "preload-audit", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--align-wire-normals", action="store_true",
                        help="Prefer transverse ray normals during fitting without changing endpoint criteria.")
    parser.add_argument("--clearance-rounds", type=int, choices=range(4), default=0,
                        help="Refit selected rod rejections using up to three refreshed support penalties.")
    parser.add_argument("--slide-selected-pad", action="store_true",
                        help="Use the selected material node's SDF during clearance refits.")
    args = parser.parse_args(argv)
    if args.slide_selected_pad and not args.clearance_rounds:
        parser.error("--slide-selected-pad requires --clearance-rounds")
    project = Path(__file__).resolve().parents[1]
    entry = json.loads((project / READINESS).read_bytes())
    prior = checked_json(args.preload_audit, entry["artifact_sha256"])
    certified, sha, _, nodes, scene, hands, stations, _ = preload.load_inputs(
        args.palm_receipt, args.continuous_audit, args.partial_audit, project)
    if (prior["continuous_parent_sha256"] != sha or prior["archive"] != certified["archive"]
            or prior["clock"] != certified["clock"] or prior["native_execution_ready"] is not False
            or prior["selected_station_by_link"] != stations):
        raise ValueError("a SHA-bound candidate from the original SDK 6504 motion is required")
    seed = np.array(prior["stages"][-1]["target_waypoints"][-1])
    results = {}
    for k, (side, hand) in enumerate(hands.items()):
        results[side] = search_side(project, side, hand, seed[k], nodes, scene,
                                    prior["endpoints"][side]["terminal"], align_wire_normals=args.align_wire_normals,
                                    clearance_rounds=args.clearance_rounds, sliding_contact=args.slide_selected_pad)
        print(side, results[side]["best_partner_fit_m"], results[side]["rejection_counts"],
              "accepted", len(results[side]["accepted"]), flush=True)
    bilateral = all(result["accepted"] for result in results.values())
    output = dict(version="dh116-power-surface-pair-search-v1", fixed_rod_only=True,
                  archive=certified["archive"], clock=certified["clock"],
                  preload_audit_sha256=entry["artifact_sha256"], stations=stations, hands=results,
                  bilateral_static_candidates=bilateral, axial_fit_residual=args.align_wire_normals,
                  clearance_rounds=args.clearance_rounds, sliding_contact=args.slide_selected_pad,
                  continuous_route_verified=False,
                  actual_load_verified=False, native_execution_ready=False, teacher_admission=False,
                  power_grasp_qualified=False, generalization_qualified=False, native_runs=0)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if bilateral else 1


if __name__ == "__main__":
    raise SystemExit(main())
