"""Audit Pinch native registration and search new trilinear SDF contact loci."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import active_limits
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import CONTRACT as WRIST
from scripts.audit_dh116_pinch_initial_pose import CAPTURE
from scripts.bake_dh116_deformx_pad_sdfs import checked_json, input_packets
from scripts.fit_dh116_902_pinch_alignment import (
    MOVE_SCALE, OPPOSITION_LIMIT, PADS, PAD_LIMIT_M, ROD_RADIUS_M, _boundary_context, _side_fit,
)
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_grid
from scripts.screen_dh116_902_pinch_facets import BASELINE_SHA, HULLS, _body_pose, _phi, _point_gap, _unit
from scripts.verify_dh116_902_flexible_reset import BASE, ROOT, load_candidate, sha


FACET_SHA = {
    "left": "4221aa7939e175b8b1b45d0e0d55cd577624b5fdfd3f68beffdfde4b4183a9b5",
    "right": "9a894bf3bf2f1808b9135f023ac9bb078f737233e367eb69d337f9b7298fc503",
}
ATLAS_SIZE = 48
NEARBY_ATLAS_SIZE = 24
PAIR_FIT_LIMIT = 32


def iso_edges(grid, *, level=ROD_RADIUS_M):
    """Intersect every crossing voxel edge with the same trilinear SDF level."""
    values = np.asarray(grid["values"], dtype=float)
    origin = np.asarray(grid["origin"], dtype=float)
    spacing = float(grid["spacing"])
    points = []
    for axis in range(3):
        a = np.moveaxis(values, axis, 0)
        starts, ends = a[:-1], a[1:]
        crosses = ((starts < level) & (ends >= level)) | ((ends < level) & (starts >= level))
        indices = np.argwhere(crosses)
        if not len(indices):
            continue
        first, last = starts[crosses], ends[crosses]
        axes = (axis, *(i for i in range(3) if i != axis))
        offsets = indices[:, np.argsort(axes)].astype(float)
        offsets[:, axis] += (level - first) / (last - first)
        points.append(origin + offsets * spacing)
    if not points:
        return np.empty((0, 3)), np.empty((0, 3))
    positions = np.unique(np.concatenate(points), axis=0)
    radii, gradients, valid = sample_grid(grid, positions, dtype=np.float64)
    lengths = np.linalg.norm(gradients, axis=1)
    high = origin + spacing * (np.asarray(values.shape) - 1)
    inside_high = np.all(positions < high - 1e-12, axis=1)
    keep = valid & inside_high & (abs(radii - level) <= 1e-9) & (lengths > 1e-12)
    return positions[keep], gradients[keep] / lengths[keep, None]


def directed_gap(vertices, points, normals):
    """Signed lower bound against a unit support direction for a rod-radius sphere."""
    return np.sum(points * normals, axis=1) - np.max(vertices @ normals.T, axis=0) - ROD_RADIUS_M


def transverse_aim(normal, tangent, *, twist=0.):
    axis, tangent = _unit(normal), _unit(tangent)
    target = _unit(axis - tangent * (axis @ tangent))
    return Rotation.from_rotvec(target * twist) * Rotation.align_vectors(target[None], axis[None])[0]


def geometry_guard(state, labels):
    """Rank complete physical envelope reserve before the independent SDF gate."""
    guarded = [(float(margin), label) for margin, label in zip(state[2:], labels, strict=True)
               if label.startswith(("rod:", "support:", "table:"))]
    margin, label = min(guarded)
    return max(0., -margin), label if margin < 0. else None


def _registration(receipt, side, context, model, active):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene

    lock = ROOT / "docs/contracts/dh116_deformx_pad_sdf_evidence_lock.json"
    lock_bytes = json.loads(lock.read_bytes())
    packets = input_packets(CAPTURE, lock_bytes)
    prepared = scene.prepare_scene(ROOT)
    bake = checked_json(scene.SDF_DIR / "result.json", scene.SDF_SHA)
    outer = json.loads(receipt.read_bytes())
    native = receipt.parent / "native"

    def artifact(name):
        digest = outer["artifact_sha256"][name]
        return checked_json(native / name, digest), digest

    reset, reset_sha = artifact("pinch_reset_source_3.json")
    plan, plan_sha = artifact("frozen_plan.json")
    _, config, _ = load_candidate("pinch")
    radius = reset["evidence"]["rod_radius_m"]
    scene_digests = [v for k, v in plan["frozen_files"].items()
                     if k.endswith("/configs/dh116_tasks/flexible_power_supported_contact_v1.json")]
    if radius != ROD_RADIUS_M or config["support_scene"]["wire_radius_m"] != radius or scene_digests != [sha(BASE)]:
        raise ValueError("the sealed Reset, original scene and CPU capsule must share one radius")

    native_bindings = context["row"]["rod"]["mesh_bindings"]
    measured = context["row"]["sdk_hand_geometry"]["poses"]
    base_position, base_quat = measured[f"{side}.base_link"]
    source_root = np.r_[base_position, base_quat[[3, 0, 1, 2]]]
    source_fk = fk_poses(model, source_root, active)
    pads = {}
    for pad in PADS:
        key = f"{side}.{pad}"
        record, _, basis_delta, digest = packets[key]
        binding = prepared["pads"][key]
        registered = bake["pads"][key]
        grid = context["grids"][key]
        sdk = np.asarray(record["sdk_body"]["link_to_world_row_major"])
        usd = record["usd_transforms_row_major"]
        f4_delta = float(np.max(abs(sdk - usd["body_path"])))
        mesh_delta = float(np.max(abs(np.asarray(usd["body_path"]) - usd["mesh_path"])))
        mesh_to_body = np.asarray(record["source"]["mesh_to_body"])
        local_delta = float(np.max(abs(mesh_to_body - np.eye(4))))
        if (any(native_bindings[key].get(k) != v for k, v in binding.items())
                or sha(Path(binding["sdf_path"])) != binding["sdf_sha256"]
                or registered["gpu_row_sha256"] != digest or binding["capture_sha256"] != digest
                or registered["rod_radius_m"] != radius or registered["spacing_m"] != grid["spacing"]
                or registered["frame"] != "original_pad_coincident_local_basis_m"
                or registered["round_m"] != 0. or registered["thickness_m"] != 0.
                or binding["frame"] != "PhysX_link_origin" or binding["unit"] != "m"
                or record["source"]["geometry_sha256"] != binding["source_geometry_sha256"]
                or not np.array_equal(np.asarray(registered["origin_link_m"]), grid["origin"])
                or max(f4_delta, mesh_delta, basis_delta, local_delta) > 5e-7):
            raise ValueError("archive and F4 disagree on pad registration or physical radius")
        pads[pad] = {k: binding[k] for k in ("source_usd_sha256", "source_geometry_sha256",
                                               "cooked_geometry_sha256", "capture_sha256", "sdf_sha256")}
        sdk_position, sdk_quat = measured[key]
        sdk_rotation = Rotation.from_quat(sdk_quat)
        nominal = source_fk[pad]
        pads[pad].update(sdk_to_usd_matrix_delta=f4_delta, mesh_to_body_matrix_delta=local_delta,
                         usd_mesh_to_body_world_matrix_delta=mesh_delta,
                         maximum_basis_delta=max(f4_delta, mesh_delta, basis_delta, local_delta),
                         archived_source_fk_position_error_m=float(np.linalg.norm(sdk_position - nominal[:3, 3])),
                         archived_source_fk_rotation_error_rad=float(
                             (sdk_rotation.inv() * Rotation.from_matrix(nominal[:3, :3])).magnitude()),
                         sdf_hull_vertex_residual_m=registered["max_abs_sdf_at_hull_vertices_m"],
                         lattice_spacing_m=registered["spacing_m"])
    return dict(radius_m=radius, original_scene_sha256=sha(BASE), native_reset_source_sha256=reset_sha,
                native_frozen_plan_artifact_sha256=plan_sha, f4_lock_sha256=sha(lock),
                f4_parity_sha256=lock_bytes["parity_sha256"], sdf_registration_sha256=scene.SDF_SHA,
                mesh_binding_provenance=context["row"]["rod"]["binding_provenance"], pads=pads)


def _loci(pad, local, grid, previous):
    points, normals = iso_edges(grid)
    gaps = [directed_gap(local[f"{hull}:{pad}"], points, normals) for hull in HULLS]
    departure = np.linalg.norm(points - previous, axis=1)
    # Rank by capsule guards first and keep spatially distinct contact regions.
    violations = np.maximum(0., -PAD_LIMIT_M - np.minimum(*gaps))
    eligible = np.flatnonzero(departure >= .0025)
    global_rank = eligible[np.lexsort((np.abs(gaps[1][eligible]), violations[eligible]))]
    near_rank = eligible[np.lexsort((departure[eligible], violations[eligible]))]
    loci, kept = [], []
    for rank, limit, spacing, band in ((global_rank, ATLAS_SIZE, .0025, "global"),
                                      (near_rank, NEARBY_ATLAS_SIZE, .0018, "nearby")):
        added = 0
        for i in rank:
            if kept and np.min(np.linalg.norm(points[kept] - points[i], axis=1)) < spacing:
                continue
            bounds = [_point_gap(local[f"{hull}:{pad}"], points[i]) for hull in HULLS]
            if any(bound is None for bound in bounds):
                continue
            kept.append(i)
            exact_violation = max(0., -PAD_LIMIT_M - min(bound[0] for bound in bounds))
            loci.append(dict(center=points[i].tolist(), normal=normals[i].tolist(), atlas_band=band,
                             departure_m=float(departure[i]), exact_violation_m=exact_violation,
                             envelope_gap_bounds_m=dict(zip(HULLS, bounds, strict=True))))
            added += 1
            if added >= limit:
                break
    loci.sort(key=lambda a: a["exact_violation_m"])
    return dict(iso_edge_count=len(points), distinct_edge_count=len(eligible), loci=loci)


def _pad_centers(loci, poses):
    normals = np.array([poses[pad][0] @ point["normal"] for pad, point in zip(PADS, loci, strict=True)])
    centers = np.array([poses[pad][0] @ (np.asarray(point["center"]) - .00003 * np.asarray(point["normal"]))
                        + poses[pad][1] for pad, point in zip(PADS, loci, strict=True)])
    return centers, normals


def _fit_pairs(atlas, pad_poses, warm, lower, upper):
    poses = pad_poses(warm)
    pairs = []
    for thumb in atlas[PADS[0]]["loci"]:
        for index in atlas[PADS[1]]["loci"]:
            loci = (thumb, index)
            centers, normals = _pad_centers(loci, poses)
            distance = np.linalg.norm(centers[0] - centers[1])
            if distance < .015 and normals[0] @ normals[1] < -.1:
                violation = max(point["exact_violation_m"] for point in loci)
                pairs.append((violation, distance, loci))
    pairs.sort(key=lambda item: item[:2])
    accepted = []
    bounds = (np.maximum(warm[6:] - .2 / MOVE_SCALE[6:], lower[6:]),
              np.minimum(warm[6:] + .2 / MOVE_SCALE[6:], upper[6:]))
    for violation, initial_gap, loci in pairs[:PAIR_FIT_LIMIT]:
        def objective(q):
            fitted = np.r_[warm[:6], q]
            centers, normals = _pad_centers(loci, pad_poses(fitted))
            return np.r_[(centers[0] - centers[1]) * 1000,
                         2. * max(0., normals[0] @ normals[1] - OPPOSITION_LIMIT),
                         .03 * (q - warm[6:])]
        fit = least_squares(objective, warm[6:], bounds=bounds, max_nfev=24, diff_step=.0001)
        fitted = np.r_[warm[:6], fit.x]
        geometry = pad_poses(fitted)
        centers, normals = _pad_centers(loci, geometry)
        distance = np.linalg.norm(centers[0] - centers[1])
        if distance <= .0015 and normals[0] @ normals[1] < OPPOSITION_LIMIT:
            accepted.append(dict(loci=loci, initial_separation_m=initial_gap, separation_m=float(distance),
                                 local_violation_m=violation, coordinates=fitted, poses=geometry,
                                 midpoint=centers.mean(axis=0), normal=_unit(normals[0] - normals[1])))
    return accepted, len(pairs), min(PAIR_FIT_LIMIT, len(pairs))


def _angles(rotation, reference):
    return (reference.inv() * rotation).as_euler("XYZ")


def screen(receipt, facet_archive, *, side, pair_budget=16):
    if side not in FACET_SHA or type(pair_budget) is not int or not 1 <= pair_budget <= 24:
        raise ValueError("original hand side and bounded endpoint shortlist required")
    if sha(facet_archive) != FACET_SHA[side]:
        raise ValueError("sealed facet comparison required")
    prior = json.loads(facet_archive.read_bytes())
    context, models, source = _boundary_context(receipt, sides=(side,))
    if any(prior[k] != source[k] for k in ("outer_sha256", "last_row_sha256", "archive_sha256")):
        raise ValueError("relief screen requires the same nine material frames")
    baseline_path = facet_archive.with_name(f"sync-pinch-round3-coupled-{side}-v4.json")
    if sha(baseline_path) != BASELINE_SHA[side]:
        raise ValueError("sealed coupled alignment required beside its facet archive")
    baseline = json.loads(baseline_path.read_bytes())
    hand = context["row"]["action_readback"]["post"]["hands"][side]
    raw_active = hand["q"][0, hand["active_indices"]]
    q_low, q_high = active_limits()
    active = np.clip(raw_active, q_low, q_high)
    if np.max(abs(raw_active - active)) > 1e-8:
        raise ValueError("archived active joints exceed original hand limits")
    registration = _registration(receipt, side, context, models[side], active)
    _, config, _ = load_candidate("pinch")
    seed = Rotation.from_quat(np.asarray(config["hands"][side]["pose_wxyz"])[[4, 5, 6, 3]])
    model = models[side]
    local = {f"{hull}:{pad}": model[hull][pad]["vertices"] for hull in HULLS for pad in PADS}
    grids = {pad: context["grids"][f"{side}.{pad}"] for pad in PADS}
    nodes = context["nodes"][-1]

    def evaluate(*, rotation, origin, measure, lower, upper, world_shapes, state, labels, **_):
        def coordinates(row):
            pose = np.asarray(row["pose_wxyz"])
            target = Rotation.from_quat(pose[[4, 5, 6, 3]])
            return np.r_[pose[:3] - origin, (rotation.inv() * target).as_rotvec(),
                         np.asarray(row["active_q_rad"][:3]) - active[:3]] / MOVE_SCALE

        def pad_poses(state):
            world = world_shapes(state)
            return {pad: _body_pose(local[f"cooked_pads:{pad}"], world[f"cooked_pads:{pad}"]) for pad in PADS}

        final = baseline["best"]
        warm = coordinates(final)
        if not np.allclose(measure(warm)["pad_gaps_m"], final["pad_gaps_m"], atol=1e-10, rtol=0.):
            raise ValueError("sealed coupled alignment failed direct replay")
        warm_rotation = rotation * Rotation.from_rotvec(warm[3:6] * MOVE_SCALE[3:6])
        wrist = np.asarray(final["pose_wxyz"][:3])
        if side == "left":
            unsafe = next(x for x in prior["candidates"] if x["failures"] == ["pad_rod_clearance"])
            comparison = dict(kind="exclusive_opposed_node_36", endpoint=measure(coordinates(unsafe["endpoint"])))
        else:
            locus = prior["facets"][PADS[0]]
            outward = pad_poses(warm)[PADS[0]][0] @ locus["sdf_normal_body"]
            trial = warm.copy()
            trial[:3] -= outward * locus["sdf_locus"]["offset_m"] / MOVE_SCALE[:3]
            comparison = dict(kind="old_thumb_normal_to_first_sdf_contact", endpoint=measure(trial))

        atlas = {pad: _loci(pad, local, grids[pad], prior["facets"][pad]["sdf_locus"]["center_body_m"])
                 for pad in PADS}
        fitted, pair_count, fitted_count = _fit_pairs(atlas, pad_poses, warm, lower, upper)
        proposals = []
        workspace_rejections = 0
        for pair in fitted:
            for station in range(prior["neighboring_segments"][0], prior["neighboring_segments"][-1] + 2):
                tangent = nodes[station+1] - nodes[station-1]
                for degrees in (0, -15, 15, -30, 30, -45, 45, -60, 60):
                    delta = transverse_aim(pair["normal"], tangent, twist=np.deg2rad(degrees))
                    new_rotation = delta * warm_rotation
                    root = nodes[station] - delta.apply(pair["midpoint"] - wrist)
                    candidate = np.r_[root - origin, (rotation.inv() * new_rotation).as_rotvec(),
                                      pair["coordinates"][6:] * MOVE_SCALE[6:]] / MOVE_SCALE
                    if (np.any(candidate < lower) or np.any(candidate > upper)
                            or np.any(abs(_angles(new_rotation, seed)) > WRIST["rotation_limit_rad"])):
                        workspace_rejections += 1
                        continue
                    poses = {pad: (delta.as_matrix() @ r, root + delta.apply(p - wrist))
                             for pad, (r, p) in pair["poses"].items()}
                    gaps, normals, negative = [], [], {}
                    for pad in PADS:
                        phi, grad, valid = _phi(grids[pad], nodes, poses[pad])
                        if not valid[station] or np.linalg.norm(grad[station]) <= 1e-12:
                            break
                        gaps.append(float(phi[station] - ROD_RADIUS_M))
                        normals.append(poses[pad][0] @ _unit(grad[station]))
                        negative[pad] = np.flatnonzero(valid & (phi < ROD_RADIUS_M)).tolist()
                    if len(gaps) < 2 or normals[0] @ normals[1] >= OPPOSITION_LIMIT:
                        continue
                    score = max(abs(gap + .00003) for gap in gaps)
                    exclusive = all(negative[pad] == [station] and gap >= -PAD_LIMIT_M
                                    for pad, gap in zip(PADS, gaps, strict=True))
                    guard_cost, blocker = geometry_guard(state(candidate, station), labels)
                    proposals.append(((guard_cost, not exclusive, pair["local_violation_m"], score), candidate,
                                      dict(station=station, twist_deg=degrees, pad_gaps_m=gaps,
                                           geometric_lower_shortfall_m=guard_cost * .001,
                                           limiting_lower_guard=blocker,
                                           negative_nodes=negative, exclusive_opposition=exclusive,
                                           pad_normal_dot=float(normals[0] @ normals[1]),
                                           joint_fit_separation_m=pair["separation_m"],
                                           local_violation_m=pair["local_violation_m"], loci=pair["loci"])))
        proposals.sort(key=lambda row: row[0])
        sdf_comparison = None
        overlap = [row for row in proposals if all(-PAD_LIMIT_M <= gap < 0. for gap in row[2]["pad_gaps_m"])]
        if overlap:
            _, sample, observed = min(overlap, key=lambda row: (not row[2]["exclusive_opposition"], row[0]))
            sdf_comparison = dict(**observed, endpoint=measure(sample))
        candidates = []
        for _, configuration, proposal in proposals[:pair_budget]:
            proof = measure(configuration)
            passed = bool(proof["endpoint_passed"] and proposal["exclusive_opposition"]
                          and proof["pad_stations"] == [proposal["station"]] * 2)
            candidates.append(dict(**proposal, endpoint=proof, endpoint_passed=passed))
        return dict(unsafe_comparison=comparison, same_node_sdf_comparison=sdf_comparison,
                    atlas=atlas, compatible_pairs=pair_count,
                    pairs_fit=fitted_count, pairs_colocated=len(fitted), workspace_rejections=workspace_rejections,
                    proposals=len(proposals), exclusive_sdf_proposals=sum(row[2]["exclusive_opposition"] for row in proposals),
                    candidates=candidates, endpoint_passed=any(x["endpoint_passed"] for x in candidates))

    result = _side_fit(side, **context, model=model, max_evaluations=1, starts=1, optimize=evaluate)
    return dict(version="dh116-902-pinch-relief-screen-v1", **source, **result, side=side,
                registration=registration, relief_source_sha256=sha(Path(__file__)),
                prior_facet_sha256=FACET_SHA[side], baseline_sha256=BASELINE_SHA[side], shortlist_limit=pair_budget,
                command_path_qualified=False, native_contact_verified=False, teacher_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--facet-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--side", required=True, choices=tuple(FACET_SHA))
    parser.add_argument("--pair-budget", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh contact-locus output required")
    evidence = screen(args.receipt, args.facet_archive, side=args.side, pair_budget=args.pair_budget)
    with args.output.open("x") as output:
        json.dump(evidence, output, indent=2, allow_nan=False)
        output.write("\n")
    return 0 if evidence["endpoint_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
