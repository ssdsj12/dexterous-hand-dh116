"""Fit CPU-only 902 Pinch endpoints around a sealed, hands-clear native boundary."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, active_limits
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import CONTRACT as WRIST
from scripts.audit_dh116_certified_segment_distance import rod_envelope_clearance
from scripts.certify_dh116_tripod_distal_route import support_clearance_lower
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_grid
from scripts.screen_dh116_902_command_contact import support_bound
from scripts.screen_dh116_902_flexible_contact import boundary
from scripts.screen_dh116_tripod_measured_precontact import as_matrix
from scripts.screen_dh116_tripod_supported_reset import archived_tube, separation
from scripts.verify_dh116_902_flexible_reset import ROOT, load_candidate, sha


PADS = ("finger14_link", "finger23_link")
SIDES = ("left", "right")
ROD_RADIUS_M = .005
PAD_LIMIT_M = .00015
NONPAD_RESERVE_M = .00025
SUPPORT_RESERVE_M = .001
TABLE_RESERVE_M = .0021
OPPOSITION_LIMIT = -.2
MOVE_SCALE = np.array([.004] * 3 + [.1] * 3 + [.15] * 3)


def opposed_at_station(side, gaps, stations, opposition):
    """Check a geometric two-pad pinch at one original-side material station."""
    if (side not in SIDES or len(gaps) != 2 or len(stations) != 2
            or any(type(station) is not int for station in stations)):
        return False
    lo, hi = (21, 41) if side == "left" else (0, 20)
    return bool(stations[0] == stations[1] and lo <= stations[0] < hi
                and np.isfinite([*gaps, opposition]).all()
                and -PAD_LIMIT_M <= min(gaps) <= max(gaps) < 0.
                and opposition < OPPOSITION_LIMIT)


def _side_fit(side, row, nodes, tube, boxes, grids, table_top, model, *, max_evaluations, starts):
    # Align candidate FK to the same-clock measured links. For the unobserved
    # colliders use original USD FK. The result remains a CPU endpoint estimate.
    hand = row["action_readback"]["post"]["hands"][side]
    indices = hand["active_indices"]
    if [hand["joint_names"][i] for i in indices] != list(ACTIVE):
        raise ValueError("original active joint order required")
    low, high = active_limits()
    active = hand["q"][0, indices]
    if np.max(np.maximum(low - active, active - high)) > 1e-8:
        raise ValueError("measured finger state exceeds original coupled limits")
    active = np.clip(active, low, high)
    observed = {key.split(".")[1]: as_matrix(*pose)
                for key, pose in row["sdk_hand_geometry"]["poses"].items() if key.startswith(side + ".")}
    palm = observed["base_link"]
    rotation = Rotation.from_matrix(palm[:3, :3])
    origin = palm[:3, 3]
    pose = np.r_[origin, rotation.as_quat()[[3, 0, 1, 2]]]
    nominal_fk = fk_poses(model, pose, active)
    corrections = {name: np.linalg.solve(nominal_fk[name], measured) for name, measured in observed.items()}
    shapes = {f"{group}:{name}": (name, shape["vertices"])
              for group in ("source_colliders", "cooked_pads") for name, shape in model[group].items()}
    if len(shapes) != 22 or any(pad not in observed for pad in PADS):
        raise ValueError("all 22 original hulls and both observed Pinch pads are required")

    projection = row["wrist_projection"]
    index = SIDES.index(side)
    reset_origin = projection["projected_root_poses"][0, index, :3] - projection["palm_offsets_m"][0, index]
    lower = np.r_[reset_origin + WRIST["palm_lower_m"] - origin, [-np.inf] * 3, low[:3] - active[:3]]
    upper = np.r_[reset_origin + WRIST["palm_upper_m"] - origin, [np.inf] * 3, high[:3] - active[:3]]
    lower = np.maximum(-1., lower / MOVE_SCALE)
    upper = np.minimum(1., upper / MOVE_SCALE)
    if np.any(lower >= upper) or np.any(lower > 0) or np.any(upper < 0):
        raise ValueError("measured state is outside the original 902 wrist or finger workspace")

    def geometry(coordinates):
        delta = coordinates * MOVE_SCALE
        target = np.r_[origin + delta[:3],
                       (rotation * Rotation.from_rotvec(delta[3:6])).as_quat()[[3, 0, 1, 2]]]
        q = active.copy()
        q[:3] += delta[6:]
        transforms = fk_poses(model, target, q)
        for name, correction in corrections.items():
            transforms[name] = transforms[name] @ correction
        world = {label: vertices @ transforms[name][:3, :3].T + transforms[name][:3, 3]
                 for label, (name, vertices) in shapes.items()}
        return target, q, transforms, world

    def pad_metrics(transforms, station=None):
        gaps, stations, normals = [], [], []
        for pad in PADS:
            matrix = transforms[pad]
            local = (nodes[-1] - matrix[:3, 3]) @ matrix[:3, :3]
            phi, grad, valid = sample_grid(grids[f"{side}.{pad}"], local, dtype=np.float64)
            selected = int(np.nanargmin(phi)) if station is None and valid.any() else station
            if selected is None or not valid[selected]:
                gaps.append(.025)
                stations.append(None)
                normals.append(np.zeros(3))
                continue
            normal = matrix[:3, :3] @ grad[selected]
            length = float(np.linalg.norm(normal))
            gaps.append(float(phi[selected] - ROD_RADIUS_M))
            stations.append(int(selected))
            normals.append(normal / length if length > 0 else np.zeros(3))
        opposition = float(normals[0] @ normals[1])
        return gaps, stations, opposition

    def measure(coordinates):
        target, q, transforms, world = geometry(coordinates)
        gaps, stations, opposition = pad_metrics(transforms)
        envelopes, pad_labels, nonpad_labels, supports = {}, [], [], []
        for label, vertices in world.items():
            fast = separation(vertices, *tube)
            is_pad = shapes[label][0] in PADS
            (pad_labels if is_pad else nonpad_labels).append(label)
            refined = (rod_envelope_clearance(vertices, nodes, max_penetration_m=PAD_LIMIT_M)
                       if is_pad and fast < -PAD_LIMIT_M else None)
            envelopes[label] = dict(lower_m=fast if refined is None else refined["lower_m"],
                                    fast_lower_m=fast, refinement=refined)
            supports.extend(dict(hull=label, support=i, **support_bound(vertices, lo, hi, reserve_m=SUPPORT_RESERVE_M))
                            for i, (lo, hi) in enumerate(boxes))
        nonpad = min(nonpad_labels, key=lambda label: envelopes[label]["lower_m"])
        pad = min(pad_labels, key=lambda label: envelopes[label]["lower_m"])
        support = min(supports, key=lambda evidence: evidence["lower_m"])
        table = min(float(vertices[:, 2].min() - table_top) for vertices in world.values())
        failures = []
        if not opposed_at_station(side, gaps, stations, opposition):
            failures.append("opposed_pad_sdf")
        if envelopes[nonpad]["lower_m"] <= NONPAD_RESERVE_M:
            failures.append("nonpad_rod_clearance")
        if any(envelopes[label]["lower_m"] < -PAD_LIMIT_M or (
                envelopes[label]["refinement"] is not None and not envelopes[label]["refinement"]["passed"])
               for label in pad_labels):
            failures.append("pad_rod_clearance")
        if support["lower_m"] < SUPPORT_RESERVE_M:
            failures.append("support_clearance")
        if table < TABLE_RESERVE_M:
            failures.append("table_clearance")
        return dict(pose_wxyz=target.tolist(), active_q_rad=q.tolist(), pad_gaps_m=gaps,
                    pad_stations=stations, pad_normal_dot=opposition,
                    rod_envelopes=envelopes, limiting_nonpad_hull=nonpad,
                    limiting_pad_hull=pad, limiting_support=support, table_clearance_m=table,
                    failures=failures, endpoint_passed=not failures)

    measured = measure(np.zeros(9))
    station = measured["pad_stations"][0]
    if station is None:
        raise ValueError("measured thumb SDF has no valid material station")
    pad_mask = np.array([name in PADS for name, _ in shapes.values()])
    rod_limits = np.where(pad_mask, -.00012, NONPAD_RESERVE_M + .00003)

    def residual(coordinates):
        _, _, transforms, world = geometry(coordinates)
        gaps, _, opposition = pad_metrics(transforms, station)
        rod = np.array([separation(vertices, *tube) for vertices in world.values()])
        support = np.array([support_clearance_lower(vertices, lo, hi)
                            for vertices in world.values() for lo, hi in boxes])
        table = np.array([vertices[:, 2].min() - table_top for vertices in world.values()])
        return np.r_[(np.asarray(gaps) + .00004) * 1000,
                     np.maximum(rod_limits - rod, 0.) * 12000,
                     np.maximum(SUPPORT_RESERVE_M - support, 0.) * 12000,
                     np.maximum(TABLE_RESERVE_M - table, 0.) * 12000,
                     2. * max(opposition - OPPOSITION_LIMIT, 0.), .02 * coordinates]

    rng = np.random.default_rng(902)
    seeds = [np.zeros(9), *rng.uniform(-.3, .3, size=(starts - 1, 9))]
    candidates = []
    for seed in seeds:
        result = least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper),
                               max_nfev=max_evaluations)
        report = measure(result.x)
        candidates.append(dict(**report, residual_cost=float(result.cost), solver_nfev=result.nfev))
    best = min(candidates, key=lambda candidate: (not candidate["endpoint_passed"], candidate["residual_cost"]))
    return dict(measured=measured, best=best, starts=len(candidates))


def fit(receipt, *, max_evaluations=48, starts=3):
    """Search near both measured hands with the original per-link clearance tests."""
    if type(max_evaluations) is not int or not 1 <= max_evaluations <= 96:
        raise ValueError("explicit bounded CPU iteration count required")
    if type(starts) is not int or not 1 <= starts <= 6:
        raise ValueError("explicit bounded CPU starts required")
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from scripts.audit_dh116_pinch_initial_pose import load_geometry

    reference, config, config_files = load_candidate("pinch")
    raw, supports, archive_files, sdk = boundary(receipt, "pinch")
    if supports != config["support_scene"]["supports"]:
        raise ValueError("measured supports differ from the original scene")
    rows = [decode_owned(row) for row in raw]
    nodes = np.stack([row["rod"]["node_positions"][0] for row in rows])
    tube = archived_tube(nodes)
    boxes = [(np.asarray(b["center_m"]) - np.asarray(b["size_m"]) / 2,
              np.asarray(b["center_m"]) + np.asarray(b["size_m"]) / 2) for b in supports]
    table = reference["table"]
    top = table["center_m"][2] + table["size_m"][2] / 2
    models = {side: load_geometry(ROOT, side=side) for side in SIDES}
    if any(models[side]["usd"]["sha256"] != reference["hand_usd_sha256"][side] for side in SIDES):
        raise ValueError("original side-specific USD required")
    grids = scene.load_sdf_arrays(scene.prepare_scene(ROOT))
    hands = {side: _side_fit(side, rows[-1], nodes, tube, boxes, grids, top, models[side],
                            max_evaluations=max_evaluations, starts=starts) for side in SIDES}
    return dict(version="dh116-902-pinch-measured-alignment-v1", sdk_range=[sdk - 8, sdk],
                fit_source_sha256=sha(Path(__file__)),
                budget=dict(starts=starts, max_evaluations_per_start=max_evaluations),
                guards=dict(pad_penetration_limit_m=PAD_LIMIT_M, nonpad_rod_reserve_m=NONPAD_RESERVE_M,
                            support_reserve_m=SUPPORT_RESERVE_M, table_reserve_m=TABLE_RESERVE_M,
                            pad_normal_dot_limit=OPPOSITION_LIMIT),
                scope="cpu_geometric_endpoint_only", outer_sha256=sha(receipt),
                last_row_sha256=sha(archive_files[-2]),
                archive_sha256=[sha(path) for path in archive_files[3:-1]],
                configuration_sha256={str(path.relative_to(ROOT)): sha(path) for path in config_files},
                geometry_sources={side: model["source_sha256"] for side, model in models.items()},
                hands=hands, bilateral_endpoint_passed=all(h["best"]["endpoint_passed"] for h in hands.values()),
                command_path_qualified=False, native_contact_verified=False, teacher_admission=False)


def main():
    parser = argparse.ArgumentParser(description="Refit a bounded geometric Pinch endpoint from the sealed 902 approach")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--max-evaluations", type=int, default=48)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh alignment evidence output required")
    report = fit(args.receipt, starts=args.starts, max_evaluations=args.max_evaluations)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0 if report["bilateral_endpoint_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
