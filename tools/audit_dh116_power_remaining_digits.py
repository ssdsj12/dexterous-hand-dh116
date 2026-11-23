"""Screen remaining Power pads from the certified fixed-rod bilateral target."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from go2_pvcnn.control.dh116.linear_coupling import active_limits, expand
from scripts import certify_dh116_power_thumb_route as route
from scripts import prepare_dh116_power_preload as preload
from scripts.audit_dh116_pinch_initial_pose import checked_json, path_displacement_bound
from scripts.audit_dh116_wire_contact_witness import endpoint_gap
from scripts.certify_dh116_power_bilateral_detour import LEFT_READINESS
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_grid
from scripts.prepare_dh116_wire_pose_candidate import _pad_gap_interval
from scripts.screen_dh116_power_measured_reseat import support_adjacent_nodes
from scripts.screen_dh116_power_partial_opposition import DISTAL_PAD_LINKS, PAD_PENETRATION_M, endpoint
from scripts.search_dh116_power_surface_pairs import evaluate


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = "docs/contracts/dh116_teacher_power_bilateral_detour_v1.json"
SAMPLES = 257
TARGET_GAPS_M = (-.000006, -.00003, -.00008)
WITNESS_SHELL_M = .0005


def continuous_reach_bound(angles, lower_gaps, *, point_travel_m):
    """Deduct a uniform point-travel bound to cover the unsampled angle interval."""
    angles, gaps = np.asarray(angles, dtype=float), np.asarray(lower_gaps, dtype=float)
    if (angles.ndim != 1 or len(angles) < 2 or gaps.shape != angles.shape
            or not np.isfinite(angles).all() or not np.isfinite(gaps).all()
            or not np.all(np.diff(angles) > 0.) or not np.isfinite(point_travel_m) or point_travel_m < 0.):
        raise ValueError("finite increasing samples and a conservative full-interval travel bound required")
    fraction = float(np.diff(angles).max() / (2. * (angles[-1] - angles[0])))
    reserve = float(point_travel_m * fraction)
    lower = float(gaps.min() - reserve)
    return dict(continuous_lower_m=lower, between_samples_reserve_m=reserve,
                witness_shell_m=WITNESS_SHELL_M, outside_witness_shell=lower > WITNESS_SHELL_M)


def eligible_nodes(nodes, blocked, side):
    """Require interior unblocked material on the assigned physical wire half."""
    nodes = np.asarray(nodes, dtype=float)
    blocked = np.asarray(blocked)
    if (nodes.shape != (41, 3) or not np.isfinite(nodes).all() or blocked.shape != (41,)
            or blocked.dtype != np.bool_ or side not in route.SIDES or nodes[0, 1] >= nodes[-1, 1]):
        raise ValueError("the original oriented 41-node rod and support adjacency mask are required")
    sign = 1 if side == "left" else -1
    ids = np.arange(len(nodes))
    center_y = (nodes[0, 1] + nodes[-1, 1]) / 2.
    return (~blocked & (ids > 0) & (ids < len(nodes)-1)
            & (sign * (ids - len(nodes)//2) > 0) & (sign * (nodes[:, 1] - center_y) > 0.))


def station_crossings(gaps_at, angles, allowed):
    """Refine finite closing brackets at the original locked SDF surface."""
    sampled = np.stack([gaps_at(angle) for angle in angles])
    valid = np.isfinite(sampled) & np.asarray(allowed, dtype=bool)
    candidates, unresolved = [], []
    for gap in TARGET_GAPS_M:
        for index, node in np.argwhere(valid[:-1] & valid[1:] & (sampled[:-1] > gap) & (sampled[1:] <= gap)):
            low, high = angles[index:index+2]
            try:
                angle = brentq(lambda q: gaps_at(q)[node] - gap, low, high, xtol=1e-12)
            except ValueError:
                unresolved.append(dict(node=int(node), bracket_rad=[float(low), float(high)], target_gap_m=gap))
                continue
            candidates.append(dict(node=int(node), angle_rad=float(angle), target_gap_m=gap,
                                   gap_m=float(gaps_at(angle)[node])))
    nearest = None
    if valid.any():
        angular_index, node = np.unravel_index(np.where(valid, sampled, np.inf).argmin(), sampled.shape)
        nearest = dict(node=int(node), angle_rad=float(angles[angular_index]), gap_m=float(sampled[angular_index, node]))
    return dict(candidates=candidates, unresolved_brackets=unresolved, nearest_sample=nearest)


def compile_extension(start, side, link, angle):
    """Compile one original coupled flexion, with exact hold on the other 23 controls."""
    start = np.asarray(start, dtype=float)
    side_index = route.SIDES.index(side)
    slot = 7 + DISTAL_PAD_LINKS.index(link)
    if (start.shape != (2, 12) or not np.isfinite(start).all() or link == preload.THUMB
            or not np.isfinite(angle) or angle <= start[side_index, slot]):
        raise ValueError("one strictly closing non-thumb command from the bilateral endpoint required")
    end = start.copy()
    end[side_index, slot] = angle
    for state in (*start, *end):
        expand(state[6:])
    return route.compile_stage(start, end, retreat_world_m=np.zeros((2, 3)), flex_exponents=(1., 1.))


def screen_pad(start, side, link, hand, nodes, scene, allowed, retained):
    slot = 7 + DISTAL_PAD_LINKS.index(link)
    active = start[route.SIDES.index(side)]
    pose = route.root_pose(active, hand["seed"])
    radius = scene["wire_radius_m"]
    _, upper = active_limits()
    angles = np.linspace(active[slot], upper[slot-6], SAMPLES)
    if angles[-1] <= angles[0]:
        return dict(result="no_remaining_closing_travel", angle_rad=float(angles[0]), trials=[])

    def state(angle):
        target = active.copy()
        target[slot] = angle
        return target

    def transform(angle):
        return fk_poses(hand["model"], pose, state(angle)[6:])[link]

    def cooked(angle):
        frame = transform(angle)
        return hand["model"]["cooked_pads"][link]["vertices"] @ frame[:3, :3].T + frame[:3, 3]

    def gaps_at(angle):
        frame = transform(angle) @ hand["pad_offsets"][link]
        phi, _, valid = sample_grid(hand["grids"][link], (nodes-frame[:3, 3]) @ frame[:3, :3], dtype=np.float64)
        return np.where(valid, phi-radius, np.nan)

    hull = [_pad_gap_interval(cooked(angle), nodes, radius) for angle in angles]
    point_travel = path_displacement_bound(hand["model"], active[6:], state(angles[-1])[6:]-active[6:],
                                          cooked=True)[link]
    reach = continuous_reach_bound(angles, [item["lower_m"] for item in hull], point_travel_m=point_travel)
    best_index = min(range(len(angles)), key=lambda index: hull[index]["upper_m"])
    best_angle = float(angles[best_index])
    best_hull = dict(angle_rad=best_angle, **endpoint_gap(cooked(best_angle), nodes, radius))
    scan = station_crossings(gaps_at, angles, allowed)
    trials = []
    previous = hand["preload_targets"]
    targets = (*[name for name in previous if name != "base_link"], link)
    for candidate in scan.pop("candidates"):
        angle, node = candidate["angle_rad"], candidate["node"]
        terminal = state(angle)
        witness = preload.station_contact(hand, terminal, link, node, nodes, radius=radius)
        _, actions = compile_extension(start, side, link, angle)
        for old_link, old_node in previous.items():
            if preload.station_contact(hand, terminal, old_link, old_node, nodes, radius=radius) != retained[old_link]:
                raise ValueError(f"independent flexion moved existing Power witness {side}.{old_link}")
        guards = endpoint(hand["model"], pose, terminal[6:], nodes, scene, hand["palm_grid"], targets=targets)
        low_y, high_y = guards["open_hull_guards"]["hand_y_bounds_m"]
        side_ok = low_y > 0. if side == "left" else high_y < 0.
        try:
            preload.checked_terminal_witness(witness, node=node, max_penetration_m=PAD_PENETRATION_M)
        except ValueError as error:
            station_failure = str(error)
        else:
            station_failure = None
        cosine = float(np.dot(witness["normal_world"], retained[preload.THUMB]["normal_world"])) if (
            witness["normal_world"] is not None) else None
        trials.append(dict(candidate, witness=witness, station_failure=station_failure, whole_hand_guard=guards,
                           thumb_added_pad_cosine=cosine, control_windows=len(actions),
                           max_normalized_24d_component=float(np.abs(actions).max()),
                           static_endpoint_passed=bool(station_failure is None and guards["passed"] and side_ok)))
    accepted = any(trial["static_endpoint_passed"] for trial in trials)
    result = ("static_endpoint_candidate" if accepted else
              "fixed_wrist_cooked_reach_excluded" if reach["outside_witness_shell"] else
              "screened_endpoints_blocked" if trials else "no_sampled_station_crossing")
    return dict(result=result, angle_bounds_rad=[float(angles[0]), float(angles[-1])], angle_samples=len(angles),
                eligible_material_nodes=np.flatnonzero(allowed).tolist(), cooked_reach=reach, best_sampled_cooked=best_hull,
                sdf_scan=scan, trials=trials)


def audit(args, *, project=ROOT):
    contract = json.loads((project / HANDOFF).read_bytes())["replay"]
    refit_sha = json.loads((project / LEFT_READINESS).read_bytes())["replay"]["artifact_sha256"]
    detour = checked_json(args.bilateral_detour, contract["artifact_sha256"])
    refit = checked_json(args.left_refit, refit_sha)
    base, base_sha, partial_sha, nodes, scene, hands, stations, _ = preload.load_inputs(
        args.palm_receipt, args.continuous_audit, args.partial_audit, project)
    if (any(item["archive"] != base["archive"] or item["clock"] != base["clock"]
            or item["source_stations"] != stations or item["fixed_rod_only"] is not True for item in (detour, refit))
            or detour["base_route_sha256"] != base_sha or detour["left_refit_sha256"] != refit_sha
            or detour["right_route_sha256"] != refit["right_continuous_sha256"]
            or detour["continuous_target_geometry_proved"] is not True
            or detour["left_target_sdf_samples"]["passed"] is not True
            or refit["bilateral_partial_static_passed"] is not True
            or detour["control_windows_by_stage"] != contract["stage_control_windows"]
            or detour["left_target_waypoints"][-1] != refit["endpoint_states_left_then_right"]):
        raise ValueError("published bilateral trajectory and measured-rod refit must share the exact endpoint")
    row, _, _ = route.load_measured_palm_boundary(args.palm_receipt, project=project)
    blocked = support_adjacent_nodes(row["rod"], nodes, scene)
    start = np.asarray(detour["left_target_waypoints"][-1], dtype=float)
    results = {}
    for index, side in enumerate(route.SIDES):
        hand = hands[side]
        status, prior = evaluate(hand, start[index], nodes, scene, side=side)
        if status != "accepted_static":
            raise ValueError(f"certified bilateral endpoint failed the original gate: {side} {status}")
        allowed = eligible_nodes(nodes, blocked, side)
        if not all(allowed[node] for node in hand["preload_targets"].values()):
            raise ValueError("certified partial stations must be on their own free measured wire half")
        digits = {
            link: screen_pad(start, side, link, hand, nodes, scene, allowed, prior["sdf"])
            for link in DISTAL_PAD_LINKS if link not in hand["preload_targets"]
        }
        results[side] = dict(retained_stations=hand["preload_targets"], retained_witnesses=prior["sdf"],
                             retained_thumb_partner_cosine=prior["thumb_partner_cos"], scans=digits,
                             sdf_sha256=hand["grid_sha256"], geometry_sha256=hand["model"]["source_sha256"])
    accepted = sum(trial["static_endpoint_passed"] for hand in results.values()
                   for digit in hand["scans"].values() for trial in digit["trials"])
    return dict(version="dh116-power-remaining-digits-screen-v1",
                status="static_endpoint_candidates_need_route_proof" if accepted else "blocked_fixed_wrist_screen",
                archive=base["archive"], clock=base["clock"], bilateral_detour_sha256=contract["artifact_sha256"],
                left_refit_sha256=refit_sha, base_route_sha256=base_sha, partial_audit_sha256=partial_sha,
                certified_prefix_windows=sum(detour["control_windows_by_stage"]), start_targets=start.tolist(),
                frozen_rod=True, one_flexion_at_a_time=True, wrist_held=True, sampled_sdf_brackets=True,
                support_adjacent_nodes=np.flatnonzero(blocked).tolist(), hands=results,
                candidate_count=int(accepted), new_continuous_route_proved=False,
                full_wrap_verified=False, grasp_qualified=False, teacher_admission=False, native_runs=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("palm-receipt", "continuous-audit", "partial-audit", "bilateral-detour", "left-refit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if result["candidate_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
