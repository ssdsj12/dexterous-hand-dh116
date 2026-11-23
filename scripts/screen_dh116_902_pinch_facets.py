"""Screen alternate 902 Pinch pad facets against the sealed flexible rod geometry."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import CONTRACT as WRIST
from scripts.audit_dh116_certified_segment_distance import segment_hull_bounds
from scripts.audit_dh116_wire_contact_witness import endpoint_gap
from scripts.fit_dh116_902_pinch_alignment import (
    MOVE_SCALE, OPPOSITION_LIMIT, PADS, PAD_LIMIT_M, ROD_RADIUS_M, _boundary_context, _side_fit,
)
from scripts.prepare_dh116_pinch_closure_margin import sample_grid
from scripts.search_dh116_power_surface_pairs import pad_rays, plane_ray_candidates
from scripts.verify_dh116_902_flexible_reset import load_candidate, sha


BASELINE_SHA = {
    "left": "6c7f4c9f62ab1f936cf85218903cc5656eabba5c5fb06db1c6eb687bf3e99b73",
    "right": "7fc275f62c076260f4c30a55790a76bbcb598bd1e9b5a05e4ed0537fd5ec38f1",
}
HULLS = ("source_colliders", "cooked_pads")
ROTATION_LIMIT_RAD = np.deg2rad(70.)
PAIR_LIMIT_M = .006
PAIR_SCAN_LIMIT = 64
SPINS_DEG = (0., -10., 10., -20., 20., -40., 40.)


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError("nonzero finite direction required")
    return vector / length


def sdf_locus(grid, seed, normal):
    """Locate a bracketed phi=radius crossing on a bounded, directed body-space ray."""
    start, direction = np.asarray(seed, dtype=float), _unit(normal)
    offsets = np.linspace(-.002, .002, 17)
    phi, _, valid = sample_grid(grid, start + offsets[:, None] * direction, dtype=np.float64)
    gaps = phi - ROD_RADIUS_M
    intervals = [i for i in range(len(offsets) - 1)
                 if valid[i] and valid[i+1] and gaps[i] <= 0. <= gaps[i+1]]
    if not intervals:
        return None
    index = min(intervals, key=lambda i: abs(offsets[i] + offsets[i+1]))

    def ray_gap(offset):
        value, _, inside = sample_grid(grid, (start + offset * direction)[None], dtype=np.float64)
        if not inside[0]:
            raise ValueError("SDF crossing left its authenticated lattice")
        return float(value[0] - ROD_RADIUS_M)

    offset = brentq(ray_gap, offsets[index], offsets[index+1], xtol=1e-12)
    center = start + offset * direction
    _, gradient, _ = sample_grid(grid, center[None], dtype=np.float64)
    if np.linalg.norm(gradient[0]) <= 1e-12:
        return None
    return dict(offset_m=float(offset), center_body_m=center.tolist(),
                normal_body=_unit(gradient[0]).tolist())


def align_pair(thumb, index, target_center, tangent, origin, *, spin=0.):
    """Aim the two selected pad loci along the same local rod segment."""
    thumb, index, origin = [np.asarray(v, dtype=float) for v in (thumb, index, origin)]
    source, target = _unit(index - thumb), _unit(tangent)
    if source @ target < 0.:
        target = -target
    rotation = Rotation.from_rotvec(target * spin) * Rotation.align_vectors(target[None], source[None])[0]
    new_origin = np.asarray(target_center, dtype=float) - rotation.apply((thumb + index) / 2. - origin)
    return rotation, new_origin, float(rotation.magnitude())


def _body_pose(local, world):
    """Recover the rigid transform from corresponding original hull vertices."""
    a, b = local.mean(axis=0), world.mean(axis=0)
    u, _, vt = np.linalg.svd((local - a).T @ (world - b))
    sign = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ np.diag([1., 1., sign]) @ u.T
    origin = b - rotation @ a
    if not np.allclose(local @ rotation.T + origin, world, atol=1e-9, rtol=0.):
        raise ValueError("locked pad vertices do not reproduce their measured-derived body frame")
    return rotation, origin


def _phi(grid, points, pose):
    rotation, origin = pose
    return sample_grid(grid, (points - origin) @ rotation, dtype=np.float64)


def _facet(vertices, hull, index):
    return dict(index=int(index), normal=hull.equations[index, :3].tolist(),
                triangle_body_m=vertices[hull.simplices[index]].tolist())


def _face_at(vertices, body_point, body_normal):
    hull = ConvexHull(vertices)
    offsets = abs(hull.equations[:, :3] @ body_point + hull.equations[:, 3])
    on_surface = np.flatnonzero(offsets <= offsets.min() + 1e-7)
    index = on_surface[np.argmax(hull.equations[on_surface, :3] @ body_normal)]
    return _facet(vertices, hull, index)


def _nearest_segment(vertices, pose, body_vertices, nodes, segments):
    samples, unresolved = [], []
    for frame, line in enumerate(nodes):
        for start in segments:
            try:
                bound = segment_hull_bounds(vertices, line[start], line[start+1])
            except ValueError:
                unresolved.append([frame, start])
            else:
                samples.append((bound, frame, start))
    if not samples:
        raise ValueError("no certified original hull-to-rod segment witness")
    bound, frame, start = min(samples, key=lambda item: item[0]["upper_m"])
    rotation, origin = pose
    body_point = (np.asarray(bound["hull_point_m"]) - origin) @ rotation
    direction = _unit(np.asarray(bound["segment_point_m"]) - bound["hull_point_m"])
    return dict(sdk=424 + frame, segment_start_node=start,
                gap_bounds_m=[bound["lower_m"] - ROD_RADIUS_M, bound["upper_m"] - ROD_RADIUS_M],
                face_rod_direction_body=(direction @ rotation).tolist(),
                facet_body=_face_at(body_vertices, body_point, direction @ rotation),
                hull_point_w_m=bound["hull_point_m"], rod_point_w_m=bound["segment_point_m"],
                segments_certified=len(samples), segments_unresolved=unresolved)


def _point_gap(vertices, point):
    try:
        bound = segment_hull_bounds(vertices, point, point)
    except ValueError:
        return None
    return [bound[k] - ROD_RADIUS_M for k in ("lower_m", "upper_m")]


def _active_facets(local, world, poses, grids, nodes, station, segments):
    mapped = {}
    for pad in PADS:
        pose = poses[pad]
        point = nodes[-1, station]
        phi, gradient, valid = _phi(grids[pad], point[None], pose)
        if not valid[0]:
            raise ValueError("sealed endpoint no longer has the original pad SDF witness")
        center_body = (point - pose[1]) @ pose[0]
        sdf_normal = _unit(gradient[0])
        locus = sdf_locus(grids[pad], center_body, sdf_normal)
        envelopes = {}
        for hull in HULLS:
            label = f"{hull}:{pad}"
            result = _nearest_segment(world[label], pose, local[label], nodes, segments)
            result["terminal_rod_envelope"] = endpoint_gap(
                world[label], nodes[-1, segments[0]:segments[-1] + 2], ROD_RADIUS_M)
            result["normal_dot_sdf"] = float(np.asarray(result["facet_body"]["normal"]) @ sdf_normal)
            result["at_sdf_locus_gap_bounds_m"] = (_point_gap(local[label], locus["center_body_m"])
                                                     if locus else None)
            envelopes[hull] = result
        mapped[pad] = dict(station=station, sdf_gap_m=float(phi[0] - ROD_RADIUS_M),
                           sdf_normal_body=sdf_normal.tolist(), sdf_locus=locus, active_envelopes=envelopes)
    return mapped


def _atlas(pad, model, local, grid, active):
    source, cooked = (local[f"{group}:{pad}"] for group in HULLS)
    hull = ConvexHull(cooked)
    centers = cooked[hull.simplices].mean(axis=1)
    normals = hull.equations[:, :3]
    guarded_centers, guarded_normals = plane_ray_candidates(
        source, centers, normals, grid, radius=ROD_RADIUS_M)
    ray_centers, ray_normals = pad_rays(model, grid, pad, radius=ROD_RADIUS_M)
    # Keep one forward ray per nearest cooked face normal instead of ranking
    # thousands of closely spaced rays as independent physical alternatives.
    rays_by_face = {}
    active_normal = np.asarray(active["sdf_normal_body"])
    for point, direction in zip(ray_centers, ray_normals, strict=True):
        index = int(np.argmax(normals @ direction))
        if index not in rays_by_face or direction @ active_normal < rays_by_face[index][1] @ active_normal:
            rays_by_face[index] = (point, direction)

    starts = [(i, "cooked_centroid", center + normal * ROD_RADIUS_M, normal)
              for i, (center, normal) in enumerate(zip(centers, normals, strict=True))]
    starts.extend((int(np.argmax(normals @ direction)), "guarded_centroid_ray", point, direction)
                  for point, direction in zip(guarded_centers, guarded_normals, strict=True))
    starts.extend((i, "guarded_pad_ray", point, direction)
                  for i, (point, direction) in rays_by_face.items())
    atlas = []
    for idx, seed, point, normal in starts:
        locus = sdf_locus(grid, point, normal)
        if locus is None:
            continue
        direction = np.asarray(locus["normal_body"])
        if direction @ normal < .45:
            continue
        gaps = {group: _point_gap(vertices, locus["center_body_m"])
                for group, vertices in zip(HULLS, (source, cooked), strict=True)}
        atlas.append(dict(feature_id=len(atlas), seed=seed, facet=_facet(cooked, hull, idx), **locus,
                          envelope_gap_bounds_m=gaps, same_normal_dot=float(direction @ active_normal),
                          local_guard_possible=all(bound is not None and bound[0] >= -PAD_LIMIT_M
                                                   for bound in gaps.values())))
    return atlas


def _material_points(nodes, segments, subdivisions):
    fractions = np.arange(subdivisions+1)[:, None] / subdivisions
    return np.concatenate([(1. - fractions) * nodes[i] + fractions * nodes[i+1] for i in segments])


def _continuous_contact(poses, grids, points):
    values = [_phi(grids[pad], points, poses[pad]) for pad in PADS]
    valid = values[0][2] & values[1][2]
    gradients = [v[1][valid] @ poses[pad][0].T for pad, v in zip(PADS, values, strict=True)]
    lengths = [np.linalg.norm(v, axis=1) for v in gradients]
    nz = (lengths[0] > 0.) & (lengths[1] > 0.)
    dots = np.full(len(points), np.inf)
    dots[np.flatnonzero(valid)[nz]] = (np.sum(gradients[0][nz] * gradients[1][nz], axis=1)
                                       / (lengths[0][nz] * lengths[1][nz]))
    opposed = valid & (dots < OPPOSITION_LIMIT)
    if not opposed.any():
        return None
    gaps = [v[0] - ROD_RADIUS_M for v in values]
    scores = np.where(opposed, np.maximum(*[np.abs(g + .00003) for g in gaps]), np.inf)
    i = int(np.argmin(scores))
    return float(scores[i]), dict(continuous_center_w_m=points[i].tolist(),
                                  continuous_pad_gaps_m=[float(g[i]) for g in gaps],
                                  continuous_pad_normal_dot=float(dots[i]))


def _shift_poses(poses, delta, new_origin, origin):
    return {pad: (delta.as_matrix() @ pose[0], new_origin + delta.apply(pose[1] - origin))
            for pad, pose in poses.items()}


def _candidate_poses(atlas, poses, grids, nodes, segments, origin, budget, workspace):
    pairs = []
    for thumb in atlas[PADS[0]]:
        for index in atlas[PADS[1]]:
            features = [thumb, index]
            centers = [pose[0] @ feature["center_body_m"] + pose[1]
                       for pose, feature in zip(poses.values(), features, strict=True)]
            normals = [pose[0] @ feature["normal_body"]
                       for pose, feature in zip(poses.values(), features, strict=True)]
            separation = np.linalg.norm(centers[0] - centers[1])
            alternate = min(thumb["same_normal_dot"], index["same_normal_dot"]) < np.cos(np.deg2rad(10))
            if alternate and 1e-8 < separation <= PAIR_LIMIT_M and normals[0] @ normals[1] < OPPOSITION_LIMIT:
                pairs.append((not all(f["local_guard_possible"] for f in features), separation, features, centers))
    pairs.sort(key=lambda row: row[:2])
    points = _material_points(nodes, segments, subdivisions=16)
    proposals = []
    counts = dict(compatible_pairs=len(pairs), considered_pairs=min(PAIR_SCAN_LIMIT, len(pairs)),
                  rotation_rejections=0, no_common_opposed_sample=0, workspace_rejections={})
    for unsafe_loci, distance, features, centers in pairs[:PAIR_SCAN_LIMIT]:
        for i in segments:
            tangent = nodes[i+1] - nodes[i]
            fractions = (0., .25, .5, .75, 1.) if i == segments[-1] else (0., .25, .5, .75)
            for fraction in fractions:
                target = nodes[i] + fraction * tangent
                for spin_deg in SPINS_DEG:
                    delta, position, angle = align_pair(*centers, target, tangent, origin,
                                                       spin=np.deg2rad(spin_deg))
                    if angle > ROTATION_LIMIT_RAD:
                        counts["rotation_rejections"] += 1
                        continue
                    coordinates, blocked = workspace(delta, position)
                    if blocked is not None:
                        rejected = counts["workspace_rejections"]
                        rejected[blocked] = rejected.get(blocked, 0) + 1
                        continue
                    shifted = _shift_poses(poses, delta, position, origin)
                    result = _continuous_contact(shifted, grids, points)
                    if result is None:
                        counts["no_common_opposed_sample"] += 1
                        continue
                    score, contact = result
                    proposals.append(((unsafe_loci, score), coordinates, shifted,
                                      dict(feature_ids=[f["feature_id"] for f in features],
                                           facet_ids=[f["facet"]["index"] for f in features],
                                           target_segment=i, target_fraction=fraction, axial_spin_deg=spin_deg,
                                           locus_center_separation_m=float(distance), wrist_angle_rad=angle,
                                           wrist_position_w_m=position.tolist(), **contact)))
    proposals.sort(key=lambda row: row[0])
    shortlist, remaining, pairs_seen = [], [], set()
    for proposal in proposals:
        feature_pair = tuple(proposal[-1]["feature_ids"])
        if feature_pair in pairs_seen:
            remaining.append(proposal)
        else:
            shortlist.append(proposal)
            pairs_seen.add(feature_pair)
    return (shortlist + remaining)[:budget], dict(**counts, pose_proposals=len(proposals))


def _read_baseline(path, side, source):
    if sha(path) != BASELINE_SHA[side]:
        raise ValueError("sealed side-specific coupled Pinch endpoint required")
    data = json.loads(path.read_bytes())
    if any(data[k] != source[k] for k in ("outer_sha256", "last_row_sha256", "archive_sha256")):
        raise ValueError("facet screening requires the exact same archived approach")
    return data


def screen(receipt, baseline, *, side, pair_budget=12):
    if side not in BASELINE_SHA or type(pair_budget) is not int or not 1 <= pair_budget <= 24:
        raise ValueError("explicit side and bounded facet shortlist required")
    context, models, source = _boundary_context(receipt, sides=(side,))
    old = _read_baseline(baseline, side, source)
    _, config, _ = load_candidate("pinch")
    seed_rotation = Rotation.from_quat(np.asarray(config["hands"][side]["pose_wxyz"])[[4, 5, 6, 3]])
    model = models[side]
    local = {f"{hull}:{pad}": model[hull][pad]["vertices"] for hull in HULLS for pad in PADS}
    nodes = context["nodes"]
    grids = {pad: context["grids"][f"{side}.{pad}"] for pad in PADS}
    station = old["station"]
    first, last = (21, 40) if side == "left" else (0, 19)
    segments = list(range(max(first, station - 2), min(last, station + 2)))

    def evaluate(*, rotation, origin, measure, lower, upper, world_shapes, **_):
        final = old["best"]
        pose = np.asarray(final["pose_wxyz"])
        prior = Rotation.from_quat(pose[[4, 5, 6, 3]])
        q = context["row"]["action_readback"]["post"]["hands"][side]
        actual = q["q"][0, q["active_indices"]]
        warm = np.r_[pose[:3] - origin, (rotation.inv() * prior).as_rotvec(),
                     np.asarray(final["active_q_rad"][:3]) - actual[:3]] / MOVE_SCALE
        replay = measure(warm)
        if replay["pad_stations"] != [station, station] or not np.allclose(
                replay["pad_gaps_m"], final["pad_gaps_m"], atol=1e-10, rtol=0.):
            raise ValueError("prior pad gaps failed direct replay")
        world = world_shapes(warm)
        poses = {pad: _body_pose(local[f"cooked_pads:{pad}"], world[f"cooked_pads:{pad}"]) for pad in PADS}
        facets = _active_facets(local, world, poses, grids, nodes, station, segments)
        atlas = {pad: _atlas(pad, model, local, grids[pad], facets[pad]) for pad in PADS}
        def workspace(delta, position):
            target = delta * prior
            coordinates = np.r_[position - origin, (rotation.inv() * target).as_rotvec(),
                                warm[6:] * MOVE_SCALE[6:]] / MOVE_SCALE
            axes = ("x", "y", "z")
            for axis in range(3):
                if coordinates[axis] < lower[axis]:
                    return coordinates, axes[axis] + "_lower"
                if coordinates[axis] > upper[axis]:
                    return coordinates, axes[axis] + "_upper"
            angles = (seed_rotation.inv() * target).as_euler("XYZ")
            if np.any(abs(angles) > WRIST["rotation_limit_rad"]):
                return coordinates, "intrinsic_rotation"
            return coordinates, None

        proposals, counts = _candidate_poses(atlas, poses, grids, nodes[-1], segments, pose[:3],
                                             pair_budget, workspace)
        candidates = []
        for _, coordinates, candidate_poses, result in proposals:
            proof = measure(coordinates)
            contact_nodes = {}
            for pad in PADS:
                phi, _, valid = _phi(grids[pad], nodes[-1], candidate_poses[pad])
                contact_nodes[pad] = np.flatnonzero(valid & (phi < ROD_RADIUS_M)).tolist()
            if (proof["endpoint_passed"]
                    and any(contact_nodes[pad] != [proof["pad_stations"][0]] for pad in PADS)):
                proof["failures"].append("nonexclusive_material_station")
                proof["endpoint_passed"] = False
            candidates.append(dict(**result, endpoint=proof, negative_nodes=contact_nodes,
                                   failures=proof["failures"]))
        return dict(facets=facets, atlas=atlas, candidates=candidates, neighboring_segments=segments,
                    **counts, endpoint_passed=any(row.get("endpoint", {}).get("endpoint_passed", False)
                                                  for row in candidates))

    result = _side_fit(side, **context, model=model, max_evaluations=1, starts=1, optimize=evaluate)
    return dict(version="dh116-902-pinch-facet-screen-v1", **source, **result, side=side,
                facet_source_sha256=sha(Path(__file__)), baseline_sha256=BASELINE_SHA[side],
                shortlist_limit=pair_budget, command_path_qualified=False, native_contact_verified=False,
                teacher_admission=False, scope="cpu_facet_and_geometric_endpoint_only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--side", required=True, choices=tuple(BASELINE_SHA))
    parser.add_argument("--pair-budget", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh facet screen output required")
    report = screen(args.receipt, args.baseline, side=args.side, pair_budget=args.pair_budget)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    return 0 if report["endpoint_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
