"""Certify a right-only Power approach to a sealed partial-contact pose."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from scripts import certify_dh116_power_thumb_route as route
from scripts import prepare_dh116_power_preload as preload
from scripts.audit_dh116_pinch_initial_pose import checked_json
from scripts.search_dh116_power_surface_pairs import evaluate

READINESS = "docs/contracts/dh116_teacher_power_free_wrist_refit_v1.json"


def candidate_for_boundary(packet, certified, stations):
    if (packet["archive"] != certified["archive"] or packet["clock"] != certified["clock"]
            or packet["stations"] != stations or not packet["fixed_rod_only"]
            or packet["native_runs"] != 0 or packet["native_execution_ready"] is not False):
        raise ValueError("right endpoint belongs to a different archived Power boundary")
    right, = packet["hands"]["right"]["accepted"]
    if right["result"] != "accepted_static" or not right["geometry"]["passed"]:
        raise ValueError("sealed static Power endpoint required")
    return np.asarray(right["state"], dtype=float)


def right_extension(terminal, target, *, opening_exponent):
    """Keep the left target fixed while the right thumb opens slightly ahead."""
    if (not np.isfinite(opening_exponent) or not 1. <= opening_exponent <= 2.
            or terminal.shape != (2, 12) or target.shape != (12,)
            or not np.isfinite(terminal).all() or not np.isfinite(target).all()
            or target[7] >= terminal[1, 7]):
        raise ValueError("finite original targets and a bounded right thumb opening lead required")
    end = np.stack((terminal[0], target))
    states, actions = route.compile_stage(terminal, end, retreat_world_m=np.zeros((2, 3)),
                                         flex_exponents=(1., opening_exponent))
    if np.any(actions[:, [0, 1, 2, 3, 4, 5, 12, 13, 14, 18, 19, 20]] != 0.):
        raise ValueError("right Power continuation must not command the left hand")
    return states, actions


def command_sections(states, *, count):
    """Partition whole control windows, retaining both endpoint states."""
    windows = len(states) - 1
    if type(count) is not int or not 1 <= count <= min(4, windows):
        raise ValueError("one to four nonempty contiguous certificate sections required")
    cuts = [j * windows // count for j in range(count + 1)]
    for first, last in zip(cuts[:-1], cuts[1:], strict=True):
        yield first, last, states[first:last + 1]


def _prove_right_section(item):
    hand, states, targets, nodes, scene, seconds, first, last = item
    result = route.prove_stage(hand, states, targets, nodes, scene, allow_concurrent=True,
                               max_samples=4096, max_seconds=seconds)
    result["first_control_window"] = first
    result["last_control_window_exclusive"] = last
    return result


def prove_right_sections(hand, states, targets, nodes, scene, *, count, seconds):
    section_args = [(hand, section, targets, nodes, scene, seconds, first, last)
                    for first, last, section in command_sections(states, count=count)]
    if count == 1:
        return [_prove_right_section(section_args[0])]
    with ProcessPoolExecutor(max_workers=count, mp_context=get_context("spawn")) as pool:
        return list(pool.map(_prove_right_section, section_args))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("palm-receipt", "continuous-audit", "partial-audit", "right-refit-audit", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--thumb-opening-exponent", type=float, default=1.15)
    parser.add_argument("--proof-time-limit", type=float, default=600.,
                        help="CPU wall time for the right target geometry certificate")
    parser.add_argument("--proof-sections", type=int, default=4, choices=(1, 2, 3, 4))
    args = parser.parse_args(argv)
    if not math.isfinite(args.proof_time_limit) or not 0. < args.proof_time_limit <= 600.:
        parser.error("--proof-time-limit requires a finite number of seconds in (0, 600]")
    project = Path(__file__).resolve().parents[1]
    ready = json.loads((project / READINESS).read_bytes())
    refit_sha = ready["archived_replay"]["artifact_sha256"]
    packet = checked_json(args.right_refit_audit, refit_sha)
    certified, previous_sha, _, nodes, scene, hands, stations, terminal = preload.load_inputs(
        args.palm_receipt, args.continuous_audit, args.partial_audit, project)
    target = candidate_for_boundary(packet, certified, stations)
    result, endpoint = evaluate(hands["right"], target, nodes, scene, side="right")
    if result != "accepted_static":
        raise ValueError(f"revalidated right endpoint rejected: {result}")
    states, actions = right_extension(terminal, target, opening_exponent=args.thumb_opening_exponent)
    left_hand, right_hand = (hands[side] for side in route.SIDES)
    left_targets, right_targets = (tuple(name for name in hand["preload_targets"] if name != "base_link")
                                   for hand in (left_hand, right_hand))
    for hand in (left_hand, right_hand):
        route.validate_palm_lattice(hand["palm_grid"], radius=scene["wire_radius_m"])
    left = route.prove_stage(left_hand, states[:, 0], left_targets, nodes, scene)
    right = prove_right_sections(right_hand, states[:, 1], right_targets, nodes, scene,
                                 count=args.proof_sections, seconds=args.proof_time_limit)
    for item in right:
        print("right", item["first_control_window"], item["last_control_window_exclusive"],
              item["proof"]["proved"], len(item["proof"]["samples"]), item["endpoints_passed"], flush=True)
    selected_sdf = preload.sample_selected_sdfs(right_hand, states[:, 1], nodes, scene["wire_radius_m"])
    passed = all(item["endpoints_passed"] and item["verified"] for item in [left, *right])
    output = dict(version="dh116-power-right-concurrent-target-v1",
                  archive=packet["archive"], clock=packet["clock"], source_stations=stations,
                  previous_route_sha256=previous_sha, right_refit_sha256=refit_sha,
                  right_static_endpoint=endpoint, right_thumb_opening_exponent=args.thumb_opening_exponent,
                  control_hz=route.CONTROL_HZ, plan_control_steps=len(actions),
                  right_proof_budget=dict(sections=args.proof_sections, max_samples_per_section=4096,
                                          max_seconds_per_section=args.proof_time_limit),
                  right_selected_sdf_samples=selected_sdf,
                  target_waypoints=states.tolist(), planned_24d_commands=actions.tolist(),
                  left_static_certificate=left, right_section_certificates=right,
                  fixed_rod_only=True, left_target_held=True, continuous_target_geometry_proved=passed,
                  continuous_material_ownership_proved=False, native_execution_ready=False, native_runs=0,
                  actual_load_verified=False, grasp_qualified=False, teacher_admission=False,
                  generalization_qualified=False)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, allow_nan=False, indent=2)
        stream.write("\n")
    return 0 if passed and selected_sdf["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
