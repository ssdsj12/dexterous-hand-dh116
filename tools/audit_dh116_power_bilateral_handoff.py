"""Qualify the frozen supported Power partial-contact diagnostic from its native receipt."""

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np

from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts import IDENTITY, _clock
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import SHA, checked_file, digest
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import SUBSTEPS, measure_power_contact
from go2_pvcnn.control.dh116.tabletop_power_shared_rod import END_SUPPORT_NAMES
from scripts.audit_dh116_power_fixed_wrist_digits import POWER_TASK, POWER_TASK_SHA
from scripts.verify_dh116_power_bilateral_handoff_probe import verify_anchor
from scripts.verify_dh116_power_partial_stage import confirm_local_opposition


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA256 = "6e3c66dcb2fecb69e5c5397865cb6f9c2874846e76e26fc1276a4e96edb7c333"
GDB_STATUS = "DH116_GDB_INFERIOR_STATUS "
FINAL_STAGE = "fixed_target_observation"
SDK_PER_CONTROL = 8


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _exact_ints(record, **expected):
    return all(type(record.get(key)) is int and record[key] == value for key, value in expected.items())


def _verify_exit(receipt, run, wall_seconds):
    elapsed = receipt["elapsed_s"]
    _require(_exact_ints(receipt, exit_code=0, debugger_inferior_exit_code=0)
             and receipt["debugger_status_verified"] is True and receipt["debugger_backtrace_captured"] is False
             and receipt["debugger_inferior_signal"] is None and receipt["forced_exit"] is False
             and receipt["exceeded_budget"] is False
             and type(elapsed) in (int, float) and 0 < elapsed <= wall_seconds,
             "natural exit 0 within the frozen wall budget required")
    log = checked_file(run / "native.log", receipt["native_log_sha256"]).decode("utf-8", errors="replace")
    lines = log.splitlines()
    status = [json.loads(line[len(GDB_STATUS):]) for line in lines if line.startswith(GDB_STATUS)]
    normal = re.findall(r"^\[Inferior \d+ \(process \d+\) exited normally\]$", log, flags=re.MULTILINE)
    _require(len(normal) == 1 and len(status) == 1 and isinstance(status[0], dict)
             and status[0].keys() == {"exit_code", "signal"}
             and _exact_ints(status[0], exit_code=0) and status[0]["signal"] is None
             and "DH116_GDB_FATAL_BACKTRACE" not in log
             and not re.search(r"(?:received|terminated with) signal SIG[A-Z0-9]+\b", log),
             "GDB capture must show one natural inferior exit and no signal or backtrace")


def _verify_close(read, end):
    before = read("before_app_close.json")
    returned = read("app_close_returned.json")
    cleanup = read("cleanup_stages.json")
    clocks = [row for row in before["clocks"] if row["phase"] == "before_stop"]
    stages = ("acquire", "stop", "before_close_marker", "close", "close_returned_marker", "observe_after_close")
    _require(before["case"] == returned["case"] == "explicit"
             and before["close_mode"] == returned["close_mode"] == cleanup["close_mode"] == "normal"
             and before["lifetime_released"] is True and before["play_simulations"] is False
             and len(clocks) == 1 and clocks[0]["clock_available"] is True
             and _exact_ints(clocks[0], sdk_step=end)
             and returned["app_close_returned"] is True and cleanup["failure"] is None
             and all(cleanup["stages"][key] == dict(attempted=True, returned=True, error=None) for key in stages),
             "normal app close with clean cleanup and the frozen pre-stop SDK clock required")


def _verify_clock(record, owner, step):
    clock = _clock(lambda: record)
    _require(all(clock[key] == owner[key] for key in IDENTITY[:5])
             and _exact_ints(clock, physics_step=step, rod_tick=(step - owner["physics_step"]) * SUBSTEPS),
             "same owner and exact frozen SDK/solver clock required")
    return clock


def _verify_budget(read, inventory, plan, owner):
    budget = plan["budget"]
    request = plan["power_high_route"]["bilateral_partial"]
    setup, end, count = (budget[key] for key in ("sdk_setup_steps", "sdk_steps", "control_steps"))
    controlled = SDK_PER_CONTROL * count
    start, anchor = end - controlled, request["anchor_sdk"]
    stations, phases = request["final_stations"], request["phases"]
    ticks = [anchor, *(phase["end_sdk"] for phase in phases)]
    hold = plan["power_retreat_then_open"][-1]
    _require(budget["mode"] == "power_bilateral_partial_contact" and setup <= start <= anchor < end
             and _exact_ints(budget, episode_resets=1, num_envs=1, teacher_queries=0, ppo_updates=0,
                             model_loads=0, auto_retries=0, solver_steps=(end - setup) * SUBSTEPS)
             and budget["auto_kill"] is False and len(plan["power_retreat_then_open"]) == count
             and len(hold) == 24 and all(value == 0. for value in hold)
             and phases[-1] == dict(stage=FINAL_STAGE, end_sdk=end, stations=stations)
             and ticks[-2] == end - SDK_PER_CONTROL
             and all((after - before) > 0 and (after - before) % SDK_PER_CONTROL == 0
                     for before, after in zip(ticks, ticks[1:])),
             "frozen Power stage and SDK/control/solver budget differ")
    shared = read("power_shared_result.json")
    control = read("power_control_result.json")
    result = control["result"]
    _require(shared["scope"] == "measured_supported_power_action_handoff"
             and _exact_ints(shared, rods_per_env=1, sdk_steps=end, measured_exchanges=end-setup,
                             solver_substeps=budget["solver_steps"], executed_control_windows=count,
                             controlled_sdk_steps=controlled, parked_sdk_steps=start-setup, ppo_updates=0)
             and _exact_ints(result, physical_samples=controlled) and result["termination_reason"] is None
             and len(result["windows"]) == count
             and all(_exact_ints(window, control_step=index, physical_samples=SDK_PER_CONTROL)
                     and window["termination_reason"] is None and window["training_reward"] is None
                     and window["teacher_admission"] is False for index, window in enumerate(result["windows"])),
             "finished native coverage of the frozen SDK/control/solver budget required")
    _verify_clock(shared["supported_seed_clock"], owner, setup)
    _verify_clock(control["start"], owner, start)
    final_clock = _verify_clock(control["end"], owner, end)
    # Check full index coverage from the receipt without rereading the trajectory.
    for prefix, low, high in (("power_exchange", setup+1, start), ("power_control", start+1, end),
                              ("power_palm_probe", anchor+1, end)):
        step_names = {name: sha for name, sha in inventory.items()
                      if re.fullmatch(rf"{prefix}_\d+\.json", name) and int(name.rsplit("_", 1)[-1][:-5]) >= low}
        _require(len(step_names) == high-low+1 and all(
            isinstance(sha := step_names.get(f"{prefix}_{sdk}.json"), str) and SHA.fullmatch(sha)
            for sdk in range(low, high+1)), f"receipt inventory does not cover the frozen {prefix} range")
    return final_clock, hold, stations, anchor


def _hand_owner(hands):
    _require(hands.keys() == {"left", "right"}, "same owner required for both hand readbacks")
    keys = ("instance_object_id", "view_object_id", "prim_paths", "row_indices")
    pair = tuple(tuple(hands[side][key] for key in keys) for side in ("left", "right"))
    _require(all(type(value) is int and value > 0 for hand in pair for value in hand[:2])
             and all(isinstance(hand[2], list) and len(hand[2]) == 1 and hand[2][0] and hand[3] == [0] for hand in pair)
             and all(pair[0][i] != pair[1][i] for i in range(3)), "same owner requires two distinct live hand views")
    return pair


def _terminal_row(read, owner, sdk, final_sdk, control_steps, hold):
    row = decode_owned(read(f"power_control_{sdk}.json"))
    clock = _verify_clock(row["rod"], owner, sdk)
    action = row["action_readback"]
    pre, post = (action[key] for key in ("pre", "post"))
    _require(_exact_ints(row, physics_step=sdk, policy_step=control_steps-1, substep=SDK_PER_CONTROL-1+sdk-final_sdk)
             and row["guard_failure"] is None and _exact_ints(row["sdk_hand_geometry"], physics_step=sdk)
             and action["version"] == "dh116-deformable-bimanual-step-v1" and action["env_ids"] == [owner["env_id"]]
             and _exact_ints(action, observed_sdk_step_delta=1, observed_rod_tick_delta=SUBSTEPS)
             and np.array_equal(action["request"]["action24"], [hold])
             and len(pre["identity"]) == len(post["identity"]) == 1,
             "complete terminal control readback of the frozen hold command required")
    _verify_clock(pre["identity"][0], owner, sdk - 1)
    _require(_verify_clock(post["identity"][0], owner, sdk) == clock,
             "same owner clock required in both rod and action readbacks")
    hands = _hand_owner(post["hands"])
    _require(hands == _hand_owner(pre["hands"]), "same owner required across the terminal hand readback")

    rod = row["rod"]
    measured = measure_power_contact(rod)
    _require(not measured["wrong_side_contacts"], "measured terminal contact crossed its assigned wire side")
    center = measured["material_side_split_world_y_m"]
    positions_y = rod["node_positions"][0, :, 1]
    retained = set()
    for pair in rod["rod_mesh_contact_pairs"]["pairs"]:
        load = float(np.dot(pair["force_on_rod_world_n"], pair["outward_normal_world"]))
        _require(math.isclose(load, pair["normal_force_on_rod_n"], rel_tol=1e-12, abs_tol=1e-12),
                 "terminal solver pair normal load disagrees with its captured vectors")
        name = pair["mesh_name"]
        if name in END_SUPPORT_NAMES and load > 0.:
            sign = (-1, 1)[END_SUPPORT_NAMES.index(name)]
            _require(sign * (positions_y[pair["node_index"]] - center) > 0,
                     "end support loaded on the wrong wire side")
            retained.add((name, pair["substep"]))
    _require(retained == {(name, tick) for name in END_SUPPORT_NAMES for tick in range(SUBSTEPS)},
             "both end supports must retain positive normal load at every terminal solver substep")
    return row, clock, hands


def _audit(receipt_path, receipt_sha256):
    outer = json.loads(checked_file(receipt_path, receipt_sha256))
    _require(outer.get("frozen_sha256") == FROZEN_SHA256, "outer receipt does not name the released frozen plan")
    root = receipt_path.parent
    native = root / "native"
    _require(not native.is_symlink(), "native evidence directory must be local to the receipt")
    inventory = outer["artifact_sha256"]
    _require("power_failure.json" not in inventory and not (native / "power_failure.json").exists(),
             "native failure evidence precludes partial contact qualification")

    def read(name):
        return json.loads(checked_file(native / name, inventory.get(name)))

    plan = read("frozen_plan.json")
    _require(digest(plan) == FROZEN_SHA256, "receipt-bound native plan differs from the released frozen plan")
    budget = plan["budget"]
    end = budget["sdk_steps"]
    _verify_exit(outer, root, budget["soft_wall_seconds"])
    _verify_close(read, end)

    config_path = str(Path(plan["project"]) / POWER_TASK)
    _require(plan["frozen_files"].get(config_path) == POWER_TASK_SHA, "frozen support scene config SHA differs")
    scene = json.loads(checked_file(ROOT / POWER_TASK, POWER_TASK_SHA))["support_scene"]
    support = read("power_support_scene.json")
    owner = _clock(lambda: support["reset_clock"])
    _require(support["supports"] == scene["supports"] and support["no_hidden_rod_pins"] is True
             and support["authored_rod_force"] is False and support["reset_seed_sha256"] == plan["power_seed_sha256"]
             and _exact_ints(owner, physics_step=budget["sdk_setup_steps"], rod_tick=0, env_id=0, reset_id=1),
             "the frozen free-rod end support scene and one actual Reset required")

    final_clock, hold, stations, anchor = _verify_budget(read, inventory, plan, owner)
    boundary = decode_owned(read(f"power_control_{anchor}.json"))
    _verify_clock(boundary["rod"], owner, anchor)
    verify_anchor(boundary, plan["power_high_route"]["bilateral_partial"]["archive"], anchor)
    prior, _, first_hands = _terminal_row(read, owner, end-1, end, budget["control_steps"], hold)
    current, clock, last_hands = _terminal_row(read, owner, end, end, budget["control_steps"], hold)
    _require(clock == final_clock and first_hands == last_hands, "same owner required for both terminal exchanges")
    terminal = confirm_local_opposition(prior, current, stations, scene=scene)
    _require(terminal["local_opposition_with_reaction"] is True,
             "terminal bilateral local opposition and delivered reaction were not independently verified")
    for sdk in (end-1, end):
        probe = read(f"power_palm_probe_{sdk}.json")
        _require(_exact_ints(probe, sdk_step=sdk) and probe["stage"] == FINAL_STAGE
                 and probe["completed_live_rod_guard"] is True and probe["supports_retained"] is True
                 and probe["station_violations"] == probe["support_adjacent_pairs"] == []
                 and probe["terminal_opposition"] == (terminal if sdk == end else None),
                 "terminal probe disagrees with the receipt-bound Power exchange")
    summary = read("power_palm_probe_summary.json")
    _require(summary["scope"] == "power_supported_partial_contact" and summary["anchor_matches"] is True
             and _exact_ints(summary, live_geometry_and_reaction_steps=end-anchor)
             and type(summary["nonzero_reaction_transfer_steps"]) is int
             and 0 < summary["nonzero_reaction_transfer_steps"] <= end-anchor
             and summary["local_opposition_with_reaction"] is True and summary["terminal"] == terminal
             and all(summary[key] is False for key in ("full_wrap_verified", "grasp_qualified", "teacher_admission")),
             "terminal summary disagrees with the independently measured opposition or planned coverage")
    return dict(status="supported_partial_contact_verified", receipt_sha256=receipt_sha256, frozen_sha256=FROZEN_SHA256,
                contact_sdk_step=end-1, reaction_sdk_step=end, hands=terminal["hands"],
                full_wrap_verified=False, grasp_qualified=False, lift_qualified=False, teacher_admission=False)


def audit_receipt(receipt_path, receipt_sha256):
    """Read the authenticated receipt and qualify only supported partial contact."""
    try:
        return _audit(Path(receipt_path), receipt_sha256)
    except (OSError, KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError(f"missing or malformed receipt-bound evidence: {exc}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True, help="outer.json beside native.log and native/")
    parser.add_argument("--receipt-sha256", required=True, help="independently supplied SHA-256 of outer.json")
    args = parser.parse_args(argv)
    try:
        result = audit_receipt(args.receipt, args.receipt_sha256)
    except ValueError as exc:
        parser.exit(1, f"Power receipt rejected: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
