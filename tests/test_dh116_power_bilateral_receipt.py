import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import digest
from test_dh116_power_palm_contact_evidence import build


ROOT = Path(__file__).resolve().parents[1]
TASK = "configs/dh116_tasks/flexible_power_supported_contact_v1.json"
STAGE = "fixed_target_observation"
STATIONS = {
    f"{side}.{name}": [node]
    for side, node in (("left", 33), ("right", 7))
    for name in ("base_link", "finger14_link", "finger53_link")
}
OWNER = dict(run_uuid="test_run", episode_uid="test_run:0", env_id=0, rod_uid="rod0", reset_id=1)
STATUS = '[Inferior 1 (process 123) exited normally]\nDH116_GDB_INFERIOR_STATUS {"exit_code": 0, "signal": null}\n'


def _clock(step):
    return dict(OWNER, physics_step=step, rod_tick=(step - 2) * 12, timestamp_s=step / 480,
                solver_time_s=(step - 2) / 480, clock_valid=True)


def _row(step, *, right_support=True):
    rod = build(proximal=True, table=True, hand_contact_links={"base_link", "finger14_link", "finger53_link"})
    capture = rod["rod_mesh_contact_pairs"]
    pairs = [pair for pair in capture["pairs"]
             if pair["mesh_name"] != "table" and (right_support or pair["mesh_name"] != "support_1")]
    forces = np.zeros((1, 41, 3))
    wrench = {name: np.zeros(6) for name in rod["mesh_names"]}
    for pair in pairs:
        name = pair["mesh_name"]
        if name in ("support_0", "support_1"):
            node = 1 if name == "support_0" else 39
            pair.update(node_index=node, outward_normal_world=[0., 0., 1.],
                        force_on_rod_world_n=[0., 0., .1], normal_force_on_rod_n=.1)
        forces[0, pair["node_index"]] += np.asarray(pair["force_on_rod_world_n"]) / 12
        wrench[name][:3] -= np.asarray(pair["force_on_rod_world_n"]) / 12
    capture["solver_buffer"]["attempted"] = len(pairs)
    capture.update(pairs=pairs, physics_step=step, rod_tick=(step - 2) * 12)
    routes = {}
    hands = {}
    for index, side in enumerate(("left", "right")):
        links = [name for name in rod["mesh_bindings"] if name.startswith(side + ".")]
        routes[side] = dict(body_ids=[rod["mesh_bindings"][name]["body_index"] for name in links],
                            is_global=True, forces=np.asarray([[wrench[name][:3] for name in links]]),
                            torques=np.zeros((1, len(links), 3)))
        hands[side] = dict(instance_object_id=index+1, view_object_id=index+3, row_indices=[0],
                           prim_paths=[f"/World/{side}Hand/wrist_fixed"], q=np.zeros((1, 17)), qd=np.zeros((1, 17)),
                           backend_targets=np.zeros((1, 17)), native_anchor_pose_w=np.zeros((1, 7)))
    rod.update(_clock(step), distributed_contact=forces, named_wrenches=wrench,
               table_wrench_world=wrench["table"], hand_wrenches_com_world=routes,
               velocities=np.zeros((1, 41, 3)), rest_lengths_m=np.full(40, .0125),
               external_support_wrenches_world={name: wrench[name] for name in ("table", "support_0", "support_1")})
    policy, substep = divmod(step - 11, 8)
    readback = dict(version="dh116-deformable-bimanual-step-v1", env_ids=[0],
                    pre=dict(identity=[_clock(step-1)], hands=deepcopy(hands)),
                    post=dict(identity=[_clock(step)], hands=hands),
                    request=dict(action24=np.zeros((1, 24))), observed_sdk_step_delta=1, observed_rod_tick_delta=12)
    return dict(physics_step=step, policy_step=policy, substep=substep, guard_failure=None,
                sdk_hand_geometry=dict(physics_step=step), rod=rod, action_readback=readback,
                reaction_source_physics_step=step-1, issued_composer_wrench=deepcopy(routes), wrist_projection={})


@pytest.fixture
def attempt(tmp_path, monkeypatch):
    from tools import audit_dh116_power_bilateral_handoff as audit
    from scripts.verify_dh116_power_partial_stage import confirm_local_opposition

    native = tmp_path / "native"
    native.mkdir()
    plan = dict(project=str(ROOT), output=str(native),
                budget=dict(mode="power_bilateral_partial_contact", sdk_steps=26, sdk_setup_steps=2,
                            control_steps=2, solver_steps=288, soft_wall_seconds=30, episode_resets=1,
                            num_envs=1, teacher_queries=0, ppo_updates=0, model_loads=0, auto_retries=0, auto_kill=False),
                power_high_route=dict(bilateral_partial=dict(anchor_sdk=18, archive=encode_owned(_row(18)), final_stations=STATIONS,
                    phases=[dict(stage=STAGE, end_sdk=26, stations=STATIONS)])),
                power_seed_sha256="1"*64, power_retreat_then_open=[[0.]*24 for _ in range(2)],
                frozen_files={str(ROOT / TASK): hashlib.sha256((ROOT / TASK).read_bytes()).hexdigest()})
    frozen = digest(plan)
    monkeypatch.setattr(audit, "FROZEN_SHA256", frozen)
    scene = json.loads((ROOT / TASK).read_bytes())["support_scene"]
    first, final = _row(25), _row(26)
    terminal = confirm_local_opposition(first, final, STATIONS, scene=scene)
    assert terminal["local_opposition_with_reaction"]
    passed = dict(attempted=True, returned=True, error=None)
    records = {
        "frozen_plan.json": plan,
        "before_app_close.json": dict(case="explicit", close_mode="normal", lifetime_released=True,
            play_simulations=False, clocks=[dict(phase="before_stop", clock_available=True, sdk_step=26)]),
        "app_close_returned.json": dict(case="explicit", app_close_returned=True, close_mode="normal"),
        "cleanup_stages.json": dict(close_mode="normal", failure=None, stages={key: passed for key in
            ("acquire", "stop", "before_close_marker", "close", "close_returned_marker", "observe_after_close")}),
        "power_support_scene.json": dict(reset_clock=_clock(2), supports=scene["supports"],
            reset_seed_sha256=plan["power_seed_sha256"], no_hidden_rod_pins=True, authored_rod_force=False),
        "power_shared_result.json": dict(scope="measured_supported_power_action_handoff", rods_per_env=1,
            sdk_steps=26, measured_exchanges=24, solver_substeps=288, executed_control_windows=2,
            controlled_sdk_steps=16, parked_sdk_steps=8, ppo_updates=0, supported_seed_clock=_clock(2)),
        "power_control_result.json": dict(start=_clock(10), end=_clock(26), result=dict(physical_samples=16,
            termination_reason=None, windows=[dict(control_step=index, physical_samples=8,
                termination_reason=None, training_reward=None, teacher_admission=False) for index in range(2)])),
        "power_palm_probe_summary.json": dict(scope="power_supported_partial_contact", anchor_matches=True,
            live_geometry_and_reaction_steps=8, nonzero_reaction_transfer_steps=8, terminal=terminal,
            local_opposition_with_reaction=True, full_wrap_verified=False, grasp_qualified=False,
            teacher_admission=False),
        **{f"power_exchange_{step}.json": {} for step in range(3, 11)},
        **{f"power_control_{step}.json": {} for step in range(11, 25)},
        **{f"power_palm_probe_{step}.json": {} for step in range(19, 25)},
        "power_control_18.json": encode_owned(_row(18)),
        "power_control_25.json": encode_owned(first),
        "power_control_26.json": encode_owned(final),
    }
    for step in (25, 26):
        records[f"power_palm_probe_{step}.json"] = dict(sdk_step=step, stage=STAGE,
            completed_live_rod_guard=True, supports_retained=True, station_violations=[], support_adjacent_pairs=[],
            previous_reaction_transferred=True, terminal_opposition=terminal if step == 26 else None)
    outer = dict(exit_code=0, forced_exit=False, exceeded_budget=False, elapsed_s=12., frozen_sha256=frozen,
                 debugger_status_verified=True, debugger_backtrace_captured=False,
                 debugger_inferior_exit_code=0, debugger_inferior_signal=None)
    return tmp_path, records, outer


def _seal(attempt, log=STATUS):
    root, records, outer = attempt
    inventory = {}
    for name, record in records.items():
        raw = json.dumps(record).encode()
        (root / "native" / name).write_bytes(raw)
        inventory[name] = hashlib.sha256(raw).hexdigest()
    raw_log = log.encode()
    (root / "native.log").write_bytes(raw_log)
    raw = json.dumps(dict(outer, native_log_sha256=hashlib.sha256(raw_log).hexdigest(),
                          artifact_sha256=inventory)).encode()
    path = root / "outer.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _audit(attempt, **kwargs):
    from tools.audit_dh116_power_bilateral_handoff import audit_receipt

    return audit_receipt(*_seal(attempt, **kwargs))


def test_naturally_closed_attempt_qualifies_only_supported_partial_contact(attempt):
    result = _audit(attempt)
    assert result["status"] == "supported_partial_contact_verified"
    assert (result["contact_sdk_step"], result["reaction_sdk_step"]) == (25, 26)
    assert not result["full_wrap_verified"]
    assert not result["grasp_qualified"]
    assert not result["lift_qualified"]
    assert not result["teacher_admission"]


@pytest.mark.parametrize(("field", "value"), [
    ("exit_code", 1), ("exit_code", False), ("forced_exit", True), ("exceeded_budget", True),
    ("elapsed_s", 31), ("debugger_status_verified", False), ("debugger_backtrace_captured", True),
    ("debugger_inferior_exit_code", 1), ("debugger_inferior_signal", 11), ("frozen_sha256", "2"*64),
])
def test_abnormal_or_different_outer_attempt_is_rejected(attempt, field, value):
    attempt[2][field] = value
    with pytest.raises(ValueError, match="frozen plan|natural exit"):
        _audit(attempt)


@pytest.mark.parametrize("log", [STATUS.splitlines()[0], STATUS + STATUS, "Thread 1 received signal SIGBUS\n" + STATUS])
def test_outer_booleans_cannot_substitute_for_the_gdb_capture(attempt, log):
    with pytest.raises(ValueError, match="GDB"):
        _audit(attempt, log=log)


@pytest.mark.parametrize("name", ["native.log", "outer.json", "native/frozen_plan.json", "native/power_control_25.json"])
def test_receipt_or_consumed_bytes_cannot_be_changed_after_sealing(attempt, name):
    from tools.audit_dh116_power_bilateral_handoff import audit_receipt

    path, sha = _seal(attempt)
    target = path.parent / name
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA"):
        audit_receipt(path, sha)


@pytest.mark.parametrize(("name", "message"), [
    ("power_control_26.json", "inventory"), ("power_exchange_6.json", "inventory"),
    ("power_palm_probe_21.json", "inventory"), ("app_close_returned.json", "SHA"),
])
def test_missing_evidence_is_rejected(attempt, name, message):
    del attempt[1][name]
    with pytest.raises(ValueError, match=message):
        _audit(attempt)


def test_native_plan_cannot_be_swapped_and_resealed(attempt):
    attempt[1]["frozen_plan.json"]["budget"]["sdk_steps"] -= 1
    with pytest.raises(ValueError, match="frozen plan"):
        _audit(attempt)


def test_close_marker_must_show_return(attempt):
    attempt[1]["app_close_returned.json"]["app_close_returned"] = False
    with pytest.raises(ValueError, match="close"):
        _audit(attempt)


def test_window_counts_and_stop_clock_must_cover_the_frozen_budget(attempt):
    attempt[1]["power_control_result.json"]["result"]["windows"][0]["physical_samples"] -= 1
    with pytest.raises(ValueError, match="budget"):
        _audit(attempt)


@pytest.mark.parametrize(("fault", "message"), [
    ("reset", "same owner"), ("action", "terminal control"), ("owner_view", "same owner"),
    ("wire_side", "wire side"), ("support", "end support"), ("reaction", "applied world wrench"),
    ("capture", "solver buffer"), ("summary", "terminal"),
])
def test_resealed_but_inconsistent_terminal_evidence_is_rejected(attempt, fault, message):
    records = attempt[1]
    row = records["power_control_26.json"]
    if fault == "reset":
        row["rod"]["reset_id"] += 1
    elif fault == "action":
        row["action_readback"]["request"]["action24"]["__dh116_array__"][0][0] = 1.
    elif fault == "owner_view":
        row["action_readback"]["post"]["hands"]["left"]["view_object_id"] += 50
    elif fault == "wire_side":
        row["rod"]["node_positions"]["__dh116_array__"][0][33][1] = -.12
    elif fault == "support":
        records["power_control_26.json"] = encode_owned(_row(26, right_support=False))
    elif fault == "reaction":
        row["issued_composer_wrench"]["left"]["forces"]["__dh116_array__"][0][0][0] += 1.
    elif fault == "capture":
        row["rod"]["rod_mesh_contact_pairs"]["solver_buffer"]["attempted"] -= 1
    else:
        records["power_palm_probe_summary.json"]["terminal"]["contact_sdk_step"] -= 1
    with pytest.raises(ValueError, match=message):
        _audit(attempt)


def test_resealed_anchor_cannot_differ_from_the_frozen_prefix(attempt):
    attempt[1]["power_control_18.json"]["rod"]["node_positions"]["__dh116_array__"][0][0][0] += .01
    with pytest.raises(ValueError, match="sealed waypoint boundary"):
        _audit(attempt)
