import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, MIMIC
from go2_pvcnn.control.dh116.tabletop_bimanual_joint_tracking_audit import (
    JointTrackingAuditWriter,
    capture_joint_tracking,
)


def native_arrays():
    # Deliberately reorder both hands' articulation indices. A hard-coded SDK
    # finger12 index would diagnose the wrong measured joint.
    names = [list(reversed(JOINTS)), list(JOINTS[4:] + JOINTS[:4])]
    hands = []
    q = np.zeros((1, 2, 11))
    qd = np.zeros_like(q)
    mimic_ids = []
    ids = []
    for h, hand_names in enumerate(names):
        ids.append([hand_names.index(name) for name in ACTIVE])
        mimic_ids.append([(hand_names.index(f), hand_names.index(m), r, o)
                          for f, m, r, o in MIMIC])
        for name in ACTIVE:
            q[0, h, hand_names.index(name)] = .04
        for follower, master, ratio, offset in MIMIC:
            q[0, h, hand_names.index(follower)] = ratio * .04 + offset
        hands.append(SimpleNamespace(joint_names=hand_names, data=SimpleNamespace(
            applied_torque=np.ones((1, 11)) * .2,
            computed_torque=np.ones((1, 11)) * .3)))
    q[0, 1, names[1].index('finger12')] = -.013
    qd[0, 1, names[1].index('finger12')] = -.8
    q[0, 1, names[1].index('finger13')] = .1
    bounds = np.empty((1, 2, 11, 2))
    bounds[..., 0] = .007912623
    bounds[..., 1] = 1.03
    world = SimpleNamespace(n=1, hands=hands, q=q, qd=qd,
        w=SimpleNamespace(ids=ids, sim=SimpleNamespace(current_time=.5)),
        _physics_steps=240, dt=1/480, commands=np.ones((1, 2, 6)) * .04,
        limits=bounds, mimic_ids=mimic_ids, parameters=SimpleNamespace(effort_limit_nm=1.),
        pad_measurements={'effective_pad_loads_n': np.ones((1, 2, 5)) * .1,
            'effective_pad_vectors_w': np.ones((1, 2, 5, 3)) * .02,
            'effective_pad_valid': np.ones((1, 2, 5), dtype=bool),
            'effective_pad_bad_env': np.zeros(1, dtype=bool)})
    actions = np.zeros((1, 24))
    actions[0, 7] = -.6
    return world, actions


def capture(world, actions, **kwargs):
    fields = dict(env_ids=[0], episode_uids=['run:0:3'], policy_step=30,
        actor_raw24=actions, action_before_inverse24=actions,
        action_after_inverse24=actions, live=[True], terminal=[True], failure_codes=[4])
    fields.update(kwargs)
    return capture_joint_tracking(world, {'hand_object_loads': np.ones((1, 2))}, **fields)[0]


def test_tracks_real_sdk_finger12_with_legal_target_and_out_of_bounds_actual():
    world, actions = native_arrays()
    row = capture(world, actions)
    finger = row['right_finger12']
    assert finger['sdk_joint_index'] == world.hands[1].joint_names.index('finger12')
    assert finger['physical_action_index'] == 7
    assert finger['physical_normalized_velocity_command'] == -.6
    assert finger['position_rad'] == -.013
    assert finger['velocity_rad_s'] == -.8
    assert finger['sdk_target_rad'] == pytest.approx(.04)
    assert finger['target_minus_actual_rad'] == pytest.approx(.053)
    assert finger['target_within_nominal_limits']
    assert finger['actual_below_guard']
    assert row['observed_signatures']['legal_target_actual_guard_violation']
    assert row['observed_signatures']['passive_coupling_residual_present']
    assert row['root_cause_proven'] is False
    assert row['identity']['episode_uid'] == 'run:0:3'
    assert row['identity']['physics_step'] == 240
    assert row['torque']['source'] == 'implicit_actuator_estimate_not_hardware_measurement'
    assert row['torque']['hardware_measured_torque_nm'] is None
    assert row['contacts']['impulse_source'] == 'force_times_physics_dt_estimate_not_raw_solver_impulse'
    assert row['contacts']['raw_solver_impulse_ns'] is None
    assert row['contacts']['accepted_normal_impulse_ns'][1][0] == pytest.approx(.1 / 480)
    json.dumps(row, allow_nan=False)


def test_rejects_wrong_active_order_instead_of_silently_relabeling_commands():
    world, actions = native_arrays()
    world.w.ids[1] = list(reversed(world.w.ids[1]))
    row = capture(world, actions)
    assert row['joint_mapping_valid'] is False
    assert row['right_finger12']['sdk_target_rad'] is None
    assert row['observed_signatures']['mapping_contract_mismatch']


def test_unavailable_or_nonfinite_torque_and_contact_have_explicit_validity():
    world, actions = native_arrays()
    del world.hands[1].data.applied_torque
    world.pad_measurements['effective_pad_bad_env'][0] = True
    row = capture(world, actions)
    assert row['torque']['applied_estimate_nm'][1] is None
    assert row['torque']['applied_valid'][1] is False
    assert row['contacts']['effective_pad_measurement_valid'] is False
    world.qd[0, 1, world.hands[1].joint_names.index('finger12')] = np.nan
    row = capture(world, actions)
    assert row['right_finger12']['velocity_rad_s'] is None
    assert 'right_full_velocity_rad_s.0' in row['nonfinite_fields'] or row['nonfinite_fields']
    json.dumps(row, allow_nan=False)


def test_writer_retains_terminal_and_rejects_duplicate_or_regressing_clocks(tmp_path):
    world, actions = native_arrays()
    terminal = capture(world, actions)
    path = tmp_path / 'joint.jsonl'
    with JointTrackingAuditWriter(path) as writer:
        assert writer.append(terminal)
        late = copy.deepcopy(terminal)
        late['identity']['physics_step'] += 1
        late['identity']['timestamp_s'] += world.dt
        late['live'] = False
        late['terminal'] = False
        late['right_finger12']['position_rad'] = .04
        assert writer.append(late) is False
        assert writer.terminal_rows['run:0:3']['right_finger12']['position_rad'] == -.013
        assert writer.skipped_closed_episode_rows == 1
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['terminal']
    with pytest.raises(FileExistsError):
        JointTrackingAuditWriter(path)
    with JointTrackingAuditWriter(tmp_path / 'clock.jsonl') as writer:
        row = copy.deepcopy(terminal)
        row['terminal'] = False
        assert writer.append(row)
        with pytest.raises(ValueError, match='clock'):
            writer.append(row)
        row['identity']['physics_step'] += 1
        row['identity']['timestamp_s'] -= .1
        with pytest.raises(ValueError, match='clock'):
            writer.append(row)


def test_writer_accepts_unique_reset_episode_and_never_reopens_old_uid(tmp_path):
    world, actions = native_arrays()
    first = capture(world, actions)
    with JointTrackingAuditWriter(tmp_path / 'reset.jsonl') as writer:
        writer.append(first)
        reset = copy.deepcopy(first)
        reset['identity']['episode_uid'] = 'run:0:4'
        reset['event_kind'] = 'reset'
        reset['terminal'] = False
        writer.append(reset)
        reset['identity']['episode_uid'] = 'run:0:3'
        with pytest.raises(ValueError, match='reused'):
            writer.append(reset)


def test_unique_uid_cannot_be_reassigned_to_another_environment(tmp_path):
    world, actions = native_arrays()
    row = capture(world, actions)
    with JointTrackingAuditWriter(tmp_path / 'owner.jsonl') as writer:
        writer.append(row)
        row['identity']['env_id'] = 1
        row['event_kind'] = 'reset'
        row['terminal'] = False
        with pytest.raises(ValueError, match='different environment'):
            writer.append(row)


def test_invalid_joint_measurement_never_reported_as_inside_guard():
    world, actions = native_arrays()
    world.qd[0, 1, world.hands[1].joint_names.index('finger12')] = np.nan
    row = capture(world, actions)
    assert row['right_finger12']['measured_joint_valid'] is False
    assert row['right_finger12']['joint_limit_observation'] == 'right_finger12_measurement_invalid'


def test_sdk_readback_target_takes_priority_over_integrated_double_command():
    world, actions = native_arrays()
    for hand in world.hands:
        hand.data.joint_pos_target = np.full((1, 11), .05, dtype=np.float32)
    row = capture(world, actions)
    assert row['right_finger12']['sdk_target_rad'] == float(np.float32(.05))
    assert row['right_finger12']['integrated_command_before_float32_rad'] == .04
    assert row['right_finger12']['sdk_target_source'] == 'native_articulation_data_joint_pos_target'
    assert row['sdk_target_native_readback_valid'] == [True, True]


def test_missing_sdk_readback_uses_explicit_float32_submission_reconstruction():
    world, actions = native_arrays()
    row = capture(world, actions)
    assert row['right_finger12']['sdk_target_rad'] == float(np.float32(.04))
    assert row['right_finger12']['sdk_target_source'] == 'float32_submission_reconstructed_from_integrated_command'
    assert row['sdk_target_native_readback_valid'] == [False, False]


def test_native_base_pose_and_contact_point_have_independent_validity():
    world, actions = native_arrays()
    world.roots = np.array([[[1., 2., 3., 1., 0., 0., 0.], [4., 5., 6., 1., 0., 0., 0.]]])
    world.pad_measurements['effective_pad_points_w'] = np.ones((1, 2, 5, 3)) * .123
    row = capture(world, actions)
    assert row['actual_base_link_pose_w'][1][:3] == [4., 5., 6.]
    assert row['actual_base_link_pose_valid'] == [True, True]
    assert row['contacts']['effective_pad_points_w_m'][1][0] == [.123] * 3
    assert row['contacts']['effective_pad_point_valid'][1][0] is True
    world.pad_measurements['effective_pad_points_w'][0, 1, 0, 0] = np.nan
    row = capture(world, actions)
    assert row['contacts']['effective_pad_point_valid'][1][0] is False
    assert row['contacts']['effective_pad_measurement_valid'] is True
