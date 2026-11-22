"""Prepare a small bilateral Power preload on the original measured wire boundary."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from go2_pvcnn.control.dh116.linear_coupling import active_limits
from scripts import certify_dh116_power_thumb_route as route
from scripts.audit_dh116_pinch_initial_pose import checked_json
from scripts.audit_dh116_wire_contact_witness import endpoint_gap, shapes_world
from scripts.diagnose_dh116_teacher_initialization import digest
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_grid
from scripts.screen_dh116_power_measured_reseat import PAD_PENETRATION_M, _locked_grids, support_adjacent_nodes
from scripts.screen_dh116_power_partial_opposition import DISTAL_PAD_LINKS, THUMB
from scripts.verify_dh116_power_partial_stage import _selected_links
from scripts.verify_dh116_power_staged_palm_witness import MAX_PALM_PENETRATION_M

READINESS = "docs/contracts/dh116_teacher_power_continuous_thumb_route_v1.json"
TARGET_GAP_M = -.00005
WRIST_TRAVEL_M = .00015
FINGER_TRAVEL_RAD = .011
SETTLE_WINDOWS = 1


def solve_preload(gap_at, *, travel):
    """Find the chosen signed gap strictly within a finite closing interval."""
    if not np.isfinite(travel) or travel <= 0.:
        raise ValueError("bounded positive preload travel required")

    def residual(delta):
        gap = gap_at(delta)
        if not np.isfinite(gap):
            raise ValueError("owned material node is outside its locked SDF")
        return gap - TARGET_GAP_M

    if residual(0.) <= 0. or residual(travel) > 0.:
        raise ValueError("requested preload gap cannot reach its target from the open side within the bound")
    return float(brentq(residual, 0., travel, xtol=1e-12, rtol=1e-12))


def checked_terminal_witness(report, *, node, max_penetration_m):
    if (report["node"] != node or report["most_negative_node"] != node
            or report["negative_nodes"] != [node]):
        raise ValueError("contact geometry must belong exclusively to its owned material station")
    if not -max_penetration_m < report["gap_m"] < 0.:
        raise ValueError("locked SDF preload exceeds its original penetration interval")
    normal = np.asarray(report["normal_world"])
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1.):
        raise ValueError("finite unit world contact normal required")


def station_contact(hand, state, link, node, nodes, *, radius):
    pose = route.root_pose(state, hand["seed"])
    transform = fk_poses(hand["model"], pose, state[6:])[link]
    if link != "base_link":
        # Alignment of the audit hull vertices does not change the FK pose.
        transform = transform @ hand["pad_offsets"][link]
    rotation = transform[:3, :3]
    local = (nodes - transform[:3, 3]) @ rotation
    phi, gradients, valid = sample_grid(hand["grids"][link], local, dtype=np.float64)
    if not valid[node]:
        raise ValueError("owned material node is outside its locked SDF")
    gradient = rotation @ gradients[node]
    norm = np.linalg.norm(gradient)
    closest = int(np.nanargmin(phi))
    return dict(node=node, gap_m=float(phi[node] - radius), most_negative_node=closest,
                minimum_gap_m=float(phi[closest] - radius),
                negative_nodes=np.flatnonzero(valid & (phi < radius)).tolist(),
                normal_world=(gradient / norm).tolist() if norm > 0. else None)


def adjust_hand(hand, start, targets, nodes, radius):
    """Seat the original palm, then close only the two selected active flexions."""
    initial = {link: station_contact(hand, start, link, node, nodes, radius=radius)
               for link, node in targets.items()}
    direction = np.asarray(initial["base_link"]["normal_world"], dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("preload must use the original measured palm SDF gradient")

    def palm_state(delta):
        state = start.copy()
        state[:3] += delta * direction
        return state

    palm_shift = solve_preload(lambda delta: station_contact(
        hand, palm_state(delta), "base_link", targets["base_link"], nodes, radius=radius)["gap_m"],
        travel=WRIST_TRAVEL_M)
    seated = palm_state(palm_shift)
    closed = seated.copy()
    _, upper = active_limits()
    flex = {}
    for link, node in targets.items():
        if link == "base_link":
            continue
        slot = DISTAL_PAD_LINKS.index(link) + 1

        def finger_gap(delta):
            state = closed.copy()
            state[6 + slot] += delta
            return station_contact(hand, state, link, node, nodes, radius=radius)["gap_m"]

        movement = solve_preload(finger_gap, travel=min(FINGER_TRAVEL_RAD, float(upper[slot] - closed[6 + slot])))
        closed[6 + slot] += movement
        flex[link] = movement
    terminal = {link: station_contact(hand, closed, link, node, nodes, radius=radius)
                for link, node in targets.items()}
    for link, witness in terminal.items():
        limit = MAX_PALM_PENETRATION_M if link == "base_link" else PAD_PENETRATION_M
        checked_terminal_witness(witness, node=targets[link], max_penetration_m=limit)
    thumb, partner = (row["normal_world"] for link, row in terminal.items() if link != "base_link")
    cosine = float(np.dot(thumb, partner))
    if cosine > -.25:
        raise ValueError("preload lost thumb and opposing pad orientation")
    return dict(seated=seated, closed=closed, initial=initial, terminal=terminal,
                translation_world_m=(palm_shift * direction).tolist(), flexion_rad=flex,
                thumb_partner_cos=cosine)


def load_inputs(receipt, continuous, partial, project):
    readiness = json.loads((project / READINESS).read_bytes())
    route_sha = readiness["continuous_artifact_sha256"]
    certified = checked_json(continuous, route_sha)
    prior = json.loads((project / route.READINESS).read_bytes())
    partial_sha = prior["audit"]["partial_sha256"]
    proposals = checked_json(partial, partial_sha)
    row, scene, archive = route.load_measured_palm_boundary(receipt, project=project)
    clock, nodes, _ = route.measured_palm_boundary(row)
    if (certified["continuous_target_geometry_proved"] is not True or not certified["fixed_rod_only"]
            or certified["native_execution_ready"] is not False
            or any(item["archive"] != archive or item["clock"] != clock for item in (certified, proposals))
            or [stage["stage"] for stage in certified["stages"]] != list(route.STAGES)):
        raise ValueError("certified trajectory and the original SDK 6504 boundary required")
    outer = checked_json(receipt, archive["outer_sha256"])
    frozen = json.loads(receipt.with_name("plan.json").read_bytes())
    if digest(frozen) != outer["frozen_sha256"]:
        raise ValueError("locked original palm and distal SDF resources required")
    hands, station_map = {}, {}
    blocked = support_adjacent_nodes(row["rod"], nodes, scene)
    terminal = np.asarray(certified["stages"][-1]["target_waypoints"][-1], dtype=float)
    for k, side in enumerate(route.SIDES):
        selected = prior["audit"]["hands"][side]["selected"]
        pair = selected["pair"]
        initial_pair, = [trial for trial in proposals["hands"][side]["proposals"]
                         if trial["candidate_index"] == selected["candidate_index"]
                         and trial["pair"]["partner"] == pair["partner"]]
        targets = {"base_link": initial_pair["palm_target_node"], THUMB: pair["thumb_node"],
                   pair["partner"]: pair["partner_node"]}
        if any(blocked[node] for node in targets.values()):
            raise ValueError("candidate preload station belongs to external support")
        station_map.update({f"{side}.{link}": [node] for link, node in targets.items()})
        h = route.read_hand(project, row, nodes, frozen["frozen_files"], selected, side)
        reference = route.load_geometry(project, side=side)
        _, _, _, offsets = route.align_measured_wrist(reference, side, row, apply_to_geometry=False)
        h["pad_offsets"] = offsets
        h["grids"], h["grid_sha256"] = _locked_grids(row["rod"], side, frozen["frozen_files"])
        if not np.array_equal(h["waypoints"][-1], terminal[k]):
            raise ValueError("preload must begin at the exact certified final wrist and joints")
        h["preload_targets"] = targets
        hands[side] = h
    _selected_links(station_map, len(nodes))
    return certified, route_sha, partial_sha, nodes, scene, hands, station_map, terminal



def selected_envelopes(hand, state, nodes, radius):
    pose = route.root_pose(state, hand["seed"])
    shapes = shapes_world(hand["model"], pose, state[6:])
    records = {}
    for link in hand["preload_targets"]:
        if link == "base_link":
            continue
        for prefix in ("source", "cooked"):
            key = f"{prefix}:{link}"
            gap = endpoint_gap(shapes[key], nodes, radius)
            records[key] = dict(lower_m=gap["lower_m"], upper_m=gap["upper_m"],
                                proved_below_original_limit=gap["upper_m"] < -PAD_PENETRATION_M)
    return records


def sample_selected_sdfs(hand, states, nodes, radius):
    targets = hand["preload_targets"]
    total = (len(states) - 1) * 8
    failures = []
    for index in range(total + 1):
        state = route.state_at(states, index / total)
        for link, node in targets.items():
            witness = station_contact(hand, state, link, node, nodes, radius=radius)
            limit = MAX_PALM_PENETRATION_M if link == "base_link" else PAD_PENETRATION_M
            if (witness["minimum_gap_m"] < -limit or witness["most_negative_node"] != node
                    or any(candidate != node for candidate in witness["negative_nodes"])
                    or witness["normal_world"] is None):
                failures.append(dict(sdk_offset=index, link=link, **witness))
    return dict(sample_count=total + 1, failures=failures, passed=not failures)

def prepare(receipt, continuous, partial, project):
    certified, sha, partial_sha, nodes, scene, hands, stations, terminal = load_inputs(
        receipt, continuous, partial, project)
    radius = scene["wire_radius_m"]
    results = {side: adjust_hand(hands[side], terminal[k], hands[side]["preload_targets"], nodes, radius)
               for k, side in enumerate(route.SIDES)}
    seated = np.stack([results[side].pop("seated") for side in route.SIDES])
    closed = np.stack([results[side].pop("closed") for side in route.SIDES])
    stages = []
    for name, start, end in (("palm_preload", terminal, seated), ("distal_preload", seated, closed)):
        states, actions = route.compile_stage(start, end, retreat_world_m=np.zeros((2, 3)), flex_exponents=(1., 1.))
        proofs = {}
        for k, (side, hand) in enumerate(hands.items()):
            result = route.prove_stage(hand, states[:, k], hand["targets"][-1], nodes, scene)
            result["selected_sdf_samples"] = sample_selected_sdfs(hand, states[:, k], nodes, radius)
            result["selected_end_envelopes"] = selected_envelopes(hand, end[k], nodes, radius)
            violations = result["proof"]["violations"]
            if violations:
                fraction = violations[0]["point"][0]
                failure_state = route.state_at(states[:, k], fraction)
                result["selected_envelopes_at_first_failed_sample"] = selected_envelopes(
                    hand, failure_state, nodes, radius)
            proofs[side] = result
        stages.append(dict(stage=name, control_steps=len(actions), planned_24d_commands=actions.tolist(),
                           target_waypoints=states.tolist(), hands=proofs))
    passed = all(result["verified"] and result["endpoints_passed"] and result["selected_sdf_samples"]["passed"]
                 for stage in stages for result in stage["hands"].values())
    return dict(version="dh116-power-static-preload-extension-v1", archive=certified["archive"],
                clock=certified["clock"], continuous_parent_sha256=sha, partial_sha256=partial_sha,
                selected_station_by_link=stations, target_gap_m=TARGET_GAP_M, endpoints=results,
                original_continuous_command_windows=certified["plan_control_steps"],
                nominal_preload_windows=sum(stage["control_steps"] for stage in stages),
                suggested_zero_action_observe_windows=SETTLE_WINDOWS, stages=stages,
                source_grid_sha256={side: h["grid_sha256"] for side, h in hands.items()},
                fixed_rod_only=True, extension_geometry_verified=passed, native_execution_ready=False,
                actual_load_verified=False, power_grasp_qualified=False, teacher_admission=False,
                generalization_qualified=False, native_runs=0, ppo_updates=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("palm-receipt", "continuous-audit", "partial-audit", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.palm_receipt, args.continuous_audit, args.partial_audit, Path(__file__).resolve().parents[1])
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if result["extension_geometry_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
