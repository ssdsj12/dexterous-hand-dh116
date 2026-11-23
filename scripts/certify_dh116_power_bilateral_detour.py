"""Certify the left Power detour after the fixed-rod right approach."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts import certify_dh116_power_thumb_route as route
from scripts import prepare_dh116_power_preload as preload
from scripts.audit_dh116_pinch_initial_pose import checked_json
from scripts.certify_dh116_power_right_loaded_target import prove_command_sections
from scripts.screen_dh116_power_partial_opposition import endpoint
from scripts.search_dh116_power_surface_pairs import evaluate

LEFT_READINESS = "docs/contracts/dh116_teacher_power_left_contact_refit_v1.json"
RIGHT_READINESS = "docs/contracts/dh116_teacher_power_right_concurrent_route_v1.json"
PALM_DETOUR_M = .0025
THUMB_ABDUCTION_RAD = .05
THUMB_OPENING_RAD = .035
RIGHT_ACTIONS = [6, 7, 8, 9, 10, 11, 15, 16, 17, 21, 22, 23]


def left_detour(start, goal, axis):
    start, goal, axis = (np.asarray(value, dtype=float) for value in (start, goal, axis))
    if (start.shape != (2, 12) or goal.shape != (2, 12)
            or not np.isfinite(start).all() or not np.isfinite(goal).all()
            or not np.array_equal(start[1], goal[1])):
        raise ValueError("left detour must preserve the finite certified right endpoint")
    if axis.shape != (3,) or not np.isclose(np.linalg.norm(axis), 1., atol=1e-9, rtol=0.):
        raise ValueError("measured unit palm direction required")
    palm_detour = np.stack((PALM_DETOUR_M * axis, np.zeros(3)))
    thumb_detour = np.zeros((2, 6))
    thumb_detour[0, :2] = [THUMB_ABDUCTION_RAD, -THUMB_OPENING_RAD]
    states, actions = route.compile_stage(start, goal, retreat_world_m=palm_detour,
                                         flex_exponents=(1., 1.), joint_detour_rad=thumb_detour)
    if np.any(actions[:, RIGHT_ACTIONS] != 0.):
        raise ValueError("left continuation cannot command the right hand")
    return states, actions


def route_station_samples(hand, states, nodes, radius):
    """Read locked SDFs at each target sample, retaining missing and unsafe rows."""
    sections = 8 * (len(states) - 1)
    counts = dict(contact_samples=0, open_samples=0, missed_samples=0, unsafe_samples=0)
    failures = []
    owned = hand["preload_targets"]
    for tick in range(sections + 1):
        state = route.state_at(states, tick / sections)
        for link, node in owned.items():
            try:
                item = preload.station_contact(hand, state, link, node, nodes, radius=radius)
            except ValueError as error:
                if str(error) != "owned material node is outside its locked SDF":
                    raise
                counts["missed_samples"] += 1
                failures.append(dict(tick=tick, link=link, out_of_lattice=True))
                continue
            limit = preload.MAX_PALM_PENETRATION_M if link == "base_link" else preload.PAD_PENETRATION_M
            if (any(i != node for i in item["negative_nodes"]) or item["minimum_gap_m"] < -limit
                    or item["normal_world"] is None):
                counts["unsafe_samples"] += 1
                failures.append(dict(tick=tick, link=link, **item))
            elif item["negative_nodes"]:
                counts["contact_samples"] += 1
            else:
                counts["open_samples"] += 1
    return dict(ticks=sections + 1, station_count=len(owned), **counts, failures=failures, passed=not failures)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("palm-receipt", "continuous-audit", "partial-audit", "right-route", "left-refit", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--proof-time-limit", type=float, default=240.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.proof_time_limit) or not 0. < args.proof_time_limit <= 600.:
        parser.error("proof time per CPU section must be in (0, 600] seconds")

    project = Path(__file__).resolve().parents[1]
    right_sha = json.loads((project / RIGHT_READINESS).read_bytes())["archived_proof"]["final_artifact_sha256"]
    left_sha = json.loads((project / LEFT_READINESS).read_bytes())["replay"]["artifact_sha256"]
    right = checked_json(args.right_route, right_sha)
    refit = checked_json(args.left_refit, left_sha)
    base, base_sha, _, nodes, scene, hands, stations, terminal = preload.load_inputs(
        args.palm_receipt, args.continuous_audit, args.partial_audit, project)
    if (not right["continuous_target_geometry_proved"] or not refit["bilateral_partial_static_passed"]
            or any(x["archive"] != base["archive"] or x["clock"] != base["clock"]
                   or not x["fixed_rod_only"] or x["native_execution_ready"] for x in (right, refit))
            or right["source_stations"] != stations or refit["source_stations"] != stations
            or right["previous_route_sha256"] != base_sha or refit["right_continuous_sha256"] != right_sha
            or not np.array_equal(right["target_waypoints"][0], terminal)):
        raise ValueError("the left endpoint and right approach must share the original fixed-rod boundary")
    start = np.asarray(right["target_waypoints"][-1])
    goal = np.asarray(refit["endpoint_states_left_then_right"])
    if not np.array_equal(start[0], terminal[0]):
        raise ValueError("right approach must have held the original left terminal target")
    left, right_hand = (hands[side] for side in route.SIDES)
    left_targets, right_targets = (tuple(link for link in hand["preload_targets"] if link != "base_link")
                                   for hand in (left, right_hand))
    for index, hand in enumerate(hands.values()):
        if evaluate(hand, goal[index], nodes, scene, side=route.SIDES[index])[0] != "accepted_static":
            raise ValueError("independent original partial endpoint revaluation failed")
        route.validate_palm_lattice(hand["palm_grid"], radius=scene["wire_radius_m"])
    left_start = endpoint(left["model"], route.root_pose(start[0], left["seed"]), start[0, 6:], nodes, scene,
                          left["palm_grid"], targets=left_targets)
    if not left_start["passed"]:
        raise ValueError("the independent original left start gate failed")
    palm = preload.station_contact(left, (start[0] + goal[0]) / 2., "base_link",
                                   left["preload_targets"]["base_link"], nodes, radius=scene["wire_radius_m"])
    states, actions = left_detour(start, goal, palm["normal_world"])
    sdf = route_station_samples(left, states[:, 0], nodes, scene["wire_radius_m"])
    print("left station samples", sdf["ticks"], sdf["unsafe_samples"], sdf["missed_samples"], flush=True)
    right_hold = route.prove_stage(right_hand, states[:, 1], right_targets, nodes, scene)
    left_sections = prove_command_sections(left, states[:, 0], left_targets, nodes, scene,
                                         count=4, seconds=args.proof_time_limit,
                                         allow_joint_reversal=True, require_palm_reach=False)
    for item in left_sections:
        print("left", item["first_control_window"], item["last_control_window_exclusive"],
              item["proof"]["proved"], len(item["proof"]["samples"]), item["endpoints_passed"], flush=True)
    proved = all(item["endpoints_passed"] and item["verified"] for item in [right_hold, *left_sections])
    output = dict(version="dh116-power-bilateral-detour-v1", archive=base["archive"], clock=base["clock"],
                  base_route_sha256=base_sha, right_route_sha256=right_sha, left_refit_sha256=left_sha,
                  source_stations=stations, control_hz=route.CONTROL_HZ,
                  control_windows_by_stage=[base["plan_control_steps"], right["plan_control_steps"], len(actions)],
                  left_target_waypoints=states.tolist(), left_planned_24d_commands=actions.tolist(),
                  palm_detour_world_unit=palm["normal_world"], palm_detour_m=PALM_DETOUR_M,
                  thumb_abduction_rad=THUMB_ABDUCTION_RAD, thumb_opening_rad=THUMB_OPENING_RAD,
                  left_target_sdf_samples=sdf, left_section_certificates=left_sections, right_hold_certificate=right_hold,
                  fixed_rod_only=True, continuous_target_geometry_proved=proved,
                  continuous_material_ownership_proved=False, native_execution_ready=False,
                  full_wrap_verified=False, actual_load_verified=False, native_runs=0,
                  grasp_qualified=False, teacher_admission=False, generalization_qualified=False)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, allow_nan=False, indent=2)
        stream.write("\n")
    return 0 if proved and sdf["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
