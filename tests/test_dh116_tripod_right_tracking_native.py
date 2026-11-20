import json
import hashlib

import numpy as np
import pytest

from scripts.verify_dh116_tripod_right_tracking import (
    BUDGET, read_tracking_plan, record_tracking_window, validate_schedule, validate_tracking_sample,
)


def archived_shape():
    return dict(physics_step=19, rod_tick=204, policy_step=2, substep=0,
                timestamp_s=19 / 480, action24=np.r_[np.zeros(7), .1, .1, .1, np.zeros(14)],
                hand_loaded_substeps=[], guard_failure=None)


def test_tracking_sample_enforces_original_clock_and_action():
    row = archived_shape()
    assert validate_tracking_sample(row, row["action24"], ordinal=17) is None


@pytest.mark.parametrize("defect,match", [
    ("step", "clock"),
    ("left", "action"),
    ("contact", "hand-rod"),
    ("guard", "clearance"),
])
def test_tracking_terminates_for_out_of_scope_measurements(defect, match):
    row = archived_shape()
    target = row["action24"].copy()
    if defect == "step":
        row["rod_tick"] += 1
    if defect == "left":
        row["action24"][0] = .02
    if defect == "contact":
        row["hand_loaded_substeps"] = [8]
    if defect == "guard":
        row["guard_failure"] = "post_step_source_clearance:right"
    with pytest.raises(ValueError, match=match):
        validate_tracking_sample(row, target, ordinal=17)


def schedule():
    actions = np.zeros((22, 24))
    actions[1:-1, 7:10] = .1
    actions[1:-1, 17] = -.21
    return dict(actions24=actions, budget=BUDGET.copy(),
                version="dh116-right-tripod-noncontact-tracking-plan-v1",
                original_clearance_floor_m=.0021)


@pytest.mark.parametrize("defect", ["left", "rotation", "initial", "end", "budget"])
def test_invalid_schedule_rejects_before_emitting_action(defect):
    plan = schedule()
    if defect == "left":
        plan["actions24"][1, 12] = .01
    if defect == "rotation":
        plan["actions24"][1, 21] = .01
    if defect == "initial":
        plan["actions24"][0, 7] = .01
    if defect == "end":
        plan["actions24"][-1, 17] = .01
    if defect == "budget":
        plan["budget"]["sdk_steps"] += 1
    with pytest.raises(ValueError):
        validate_schedule(plan)


def test_exchange_records_terminal_row_before_abort():
    recorded = []
    audit = record_tracking_window(schedule(), recorded.append)
    physical = dict(kind="physical", exchange=dict(guard_failure="mimic_error"))
    with pytest.raises(ValueError, match="mimic_error"):
        audit(physical)
    assert recorded == [physical]


def test_unrelated_rows_preserved_without_making_measurement():
    recorded = []
    audit = record_tracking_window(schedule(), recorded.append)
    audit(dict(kind="setup"))
    assert recorded == [dict(kind="setup")]
    with pytest.raises(ValueError, match="incomplete"):
        audit.finish()


def physical_sample(ordinal, action):
    clocks = dict(run_uuid="run", env_id=0, episode_uid="episode", reset_id=1, rod_uid="rod",
                  physics_step=ordinal + 2, rod_tick=ordinal * 12, timestamp_s=(ordinal + 2) / 480)
    hand = dict(measured_source="physx_backend_get_dof_positions/get_dof_velocities",
                joint_names=[f"q{i}" for i in range(17)],
                units_by_name={f"q{i}": "rad" for i in range(17)}, mounted=True,
                q=np.zeros((1, 17)), qd=np.zeros((1, 17)), backend_targets=np.zeros((1, 17)),
                native_anchor_pose_w=np.array([[0., 0., 0., 1., 0., 0., 0.]]))
    rod = dict(clocks, clock_valid=True, source="RopeBatch.positions/velocities/contact_forces/mesh_wrenches",
               pair_capture_status="validated_solver_samples", node_positions=np.zeros((1, 41, 3)),
               velocities=np.zeros((1, 41, 3)), rod_mesh_contact_pairs=dict(
                   physics_step=clocks["physics_step"], rod_tick=clocks["rod_tick"],
                   pairs=[dict(substep=i, normal_force_on_rod_n=1., mesh_name="table")
                          for i in range(12)]))
    return dict(kind="physical", exchange=dict(physics_step=clocks["physics_step"],
        policy_step=(ordinal - 1) // 8, substep=(ordinal - 1) % 8, guard_failure=None,
        action_readback=dict(request=dict(action24=action[None]), post=dict(
            hands=dict(left=hand, right=hand.copy()), identity=[clocks.copy()])), rod=rod))


def test_all_windows_need_current_contact_and_mounted_joint_readback():
    from go2_pvcnn.control.dh116.tabletop_tripod_quiet_gate import TripodQuietGate

    plan = schedule()
    ledger = []
    audit = record_tracking_window(plan, ledger.append, quiet=TripodQuietGate(reset_nodes_m=np.zeros((41, 3))))
    for ordinal in range(1, 177):
        audit(physical_sample(ordinal, plan["actions24"][(ordinal - 1) // 8]))
    assert audit.finish() == len(ledger) == 176


@pytest.mark.parametrize("defect,match", [
    ("identity", "identity"),
    ("joint", "17"),
    ("target", "finite"),
    ("contact", "hand-rod"),
    ("rod", "actual complete"),
])
def test_invalid_native_evidence_is_saved_and_stops_the_next_sample(defect, match):
    plan = schedule()
    ledger = []
    audit = record_tracking_window(plan, ledger.append)
    audit(physical_sample(1, plan["actions24"][0]))
    payload = physical_sample(2, plan["actions24"][0])
    row = payload["exchange"]
    post = row["action_readback"]["post"]
    if defect == "identity":
        post["identity"][0]["episode_uid"] = row["rod"]["episode_uid"] = "other"
    elif defect == "joint":
        post["hands"]["right"]["mounted"] = False
    elif defect == "target":
        post["hands"]["right"]["backend_targets"] = np.zeros((1, 16))
    elif defect == "contact":
        row["rod"]["rod_mesh_contact_pairs"]["pairs"][0]["mesh_name"] = "right.finger14_link"
    else:
        row["rod"]["pair_capture_status"] = "missing"
    with pytest.raises(ValueError, match=match):
        audit(payload)
    assert len(ledger) == 2
    with pytest.raises(RuntimeError, match="stopped"):
        audit(physical_sample(3, plan["actions24"][0]))
    assert len(ledger) == 2


def test_frozen_launch_string_path_uses_the_real_plan_reader(tmp_path):
    from scripts.prepare_dh116_tripod_right_tracking import ENDPOINT_SHA, REFERENCE_SHA

    plan = schedule()
    plan["actions24"] = plan["actions24"].tolist()
    plan.update(endpoint_sha256=ENDPOINT_SHA, reference_sha256=REFERENCE_SHA,
                proved_right_sweep=dict(proved=True), native_qualification=False)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded, actions = read_tracking_plan(str(path), digest)
    assert loaded == plan
    np.testing.assert_array_equal(actions, plan["actions24"])
