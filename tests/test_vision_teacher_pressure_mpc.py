"""Pressure-profile native adapter contracts; CPU geometry is not physical success."""
import importlib

import pytest
import torch

from test_vision_teacher_mpc import native_env
from go2_pvcnn.control.dh116.tabletop_vision_teacher_absolute_mpc import AbsoluteTeacherReferenceMPC


PINCH = 'pinch_thumb_index'
TRIPOD = 'tripod_thumb_index_middle'


def pressure_env(n=1):
    env = native_env(n)
    env.task.m['bottom'][:] = .36
    jac = env.world.hands[0].root_physx_view.get_jacobians()
    jac.zero_()
    jac[:, 0, 0, 0] = .05
    jac[:, 1, 0, 2] = .05
    jac[:, 2, 0, 3] = .05
    loads = env.task.m['effective_pad_loads_n']
    loads[:] = loads.new_tensor([5., .75, 5., 0., 0.])
    vectors = env.task.m['effective_pad_vectors_w']; vectors.zero_()
    vectors[..., 0] = loads * loads.new_tensor([1., -1., -1., 0., 0.])
    env.task.m['effective_pad_valid'] = loads > 0
    env.task.m['effective_pad_depth_m'] = torch.zeros_like(loads)
    env.task.m['rejected_contact_loads_n'] = torch.zeros(n, 2, dtype=loads.dtype)
    return env


def optimizer(env, sequence, styles=None):
    module = importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_teacher_pressure_mpc')
    return module.PressureTeacherReferenceMPC(env, 'a' * 64, sequence,
        grasp_styles=styles or [TRIPOD] * env.world.n)


@pytest.mark.parametrize('case', ['active', 'below_air', 'invalid_q', 'invalid_command', 'outside_trust'])
def test_pinch_output_is_bitwise_equal_to_stage_a_including_fallbacks(case):
    a = native_env(); b = native_env()
    sequence = torch.zeros(1, 5, 24); sequence[:, :, 12:] = .2
    reference = AbsoluteTeacherReferenceMPC(a, 'a' * 64, sequence)
    candidate = optimizer(b, sequence, [PINCH])
    for env in (a, b):
        if case == 'below_air': env.task.m['bottom'][:] = .36
        elif case == 'invalid_q': env.task.m['q'][0, 0] = float('nan')
        elif case == 'invalid_command': env.task.command[0, 0] = float('nan')
        elif case == 'outside_trust': env.task.command[0, 0] += .02
    expected, old = reference.refine(sequence[:, 0], context={'tick': 0})
    actual, new = candidate.refine(sequence[:, 0], context={'tick': 0})
    assert torch.equal(actual, expected)
    assert new['per_env'][0]['active'] == old['per_env'][0]['active']
    assert new['per_env'][0]['reason'] == old['per_env'][0]['reason']


def test_tripod_activates_before_lift_and_uses_signed_normals_for_unloading():
    env = pressure_env(); sequence = torch.zeros(1, 5, 24)
    controller = optimizer(env, sequence)
    action, diag = controller.refine(sequence[:, 0], context={'tick': 0})
    assert diag['per_env'][0]['active']
    assert action[0, 0] > 0  # This fixture's thumb unloads by increasing this angle.
    assert action[0, 3] < 0  # Its middle unloads with the opposite angle sign.
    normal = torch.tensor(diag['per_env'][0]['pressure_predicted_relative_normal_m'])
    assert (normal[..., [0, 2]] <= 1e-10).all()
    assert diag['force_prediction'] is False and diag['optimized_action_dim'] == 12
    assert diag['pressure_goal_source_sha256'] == 'fea884d24928026d006669fcef8af1565eb12ad25e0c35130035a8bc0378274b'


def test_current_points_and_current_command_replace_old_compressed_anchor_and_absolute_box():
    env = pressure_env(); sequence = torch.zeros(1, 5, 24)
    controller = optimizer(env, sequence)
    env.task.command += .03
    env.task.m['effective_pad_points_w'] += .04
    captured = []; original = controller.native.pressure_planner.plan
    def record(**kwargs):
        captured.append(kwargs)
        return original(**kwargs)
    controller.native.pressure_planner.plan = record
    _, diag = controller.refine(sequence[:, 0], context={'tick': 0})
    assert captured and diag['per_env'][0]['active']
    assert torch.equal(captured[0]['points'], env.task.m['effective_pad_points_w'])
    torch.testing.assert_close(captured[0]['nominal'], env.task.command.reshape(1, 1, 2, 6).expand(-1, 4, -1, -1))
    assert captured[0].get('reference_radius') is None
    assert diag['per_env'][0]['pressure_reference_radius_rad'] is None


def test_nominal_masks_bearing_digits_but_keeps_future_source_actions_for_missing_digits():
    env = pressure_env(); sequence = torch.zeros(1, 5, 24)
    sequence[0, :, 2] = torch.tensor([.2, -.3, .4, -.5, .6])
    sequence[:, :, 0:2] = .5
    env.task.m['effective_pad_loads_n'][:, 0, 1] = 0
    env.task.m['effective_pad_valid'][:, 0, 1] = False
    env.task.m['effective_pad_vectors_w'][:, 0, 1] = 0
    controller = optimizer(env, sequence)
    calls = []; original = controller.native.pressure_planner.plan
    def record(**kwargs): calls.append(kwargs); return original(**kwargs)
    controller.native.pressure_planner.plan = record
    controller.refine(sequence[:, 0], context={'tick': 0})
    assert calls
    torch.testing.assert_close(calls[0]['nominal'][0, :, 0, 2],
        torch.tensor([.201, .1995, .2015, .199], dtype=torch.float64))
    torch.testing.assert_close(calls[0]['nominal'][0, :, 0, :2], torch.full((4, 2), .2, dtype=torch.float64))


def test_failed_pressure_model_holds_loaded_fingers_instead_of_restoring_overcompressed_teacher_target():
    env = pressure_env(); sequence = torch.full((1, 4, 24), .2)
    controller = optimizer(env, sequence); env.task.command -= .03
    def fail(**kwargs): raise ValueError('fixture failed local model')
    controller.native.pressure_planner.plan = fail
    action, diag = controller.refine(sequence[:, 0], context={'tick': 0})
    assert not action[0, [0, 1, 2, 3, 6, 7, 8, 9]].any()
    assert torch.equal(action[:, 12:], sequence[:, 0, 12:])
    assert diag['per_env'][0]['reason'] == 'pressure_model_fallback_hold'


def test_predicted_overloaded_inward_motion_is_rejected_and_fallback_holds():
    env = pressure_env(); sequence = torch.zeros(1, 4, 24)
    controller = optimizer(env, sequence); original = controller.native.pressure_planner.plan
    def inward(**kwargs):
        target, diag = original(**kwargs)
        diag['predicted_points'][:, :, :, 0, 0] = kwargs['points'][:, None, :, 0, 0] - .0001
        return target, diag
    controller.native.pressure_planner.plan = inward
    action, diag = controller.refine(sequence[:, 0], context={'tick': 0})
    assert not action[:, :12].any()
    assert not controller.native.pending.any()
    assert diag['per_env'][0]['reason'] == 'pressure_direction_rejected'


def test_rejected_whole_hand_pressure_requests_relief_even_for_underloaded_pads():
    env = pressure_env(); sequence = torch.full((1, 4, 24), .3)
    env.task.m['effective_pad_loads_n'][:] = env.task.m['effective_pad_loads_n'].new_tensor([.5, .2, .2, 0, 0])
    env.task.m['rejected_contact_loads_n'][0, 0] = .1
    controller = optimizer(env, sequence)
    action, diag = controller.refine(sequence[:, 0], context={'tick': 0})
    row = diag['per_env'][0]
    speed = torch.tensor(row['pressure_speed_m_s'])
    assert (speed[0, :3] < 0).all()
    assert not torch.tensor(row['pressure_fallback_actions'][:6]).any()
    assert torch.equal(action[:, 12:], sequence[:, 0, 12:])


def test_unknown_normal_holds_affected_hand_and_never_uses_absolute_restore():
    env = pressure_env(); sequence = torch.full((1, 4, 24), .3)
    env.task.m['effective_pad_vectors_w'][0, 0, 0] = 0
    controller = optimizer(env, sequence); env.task.command -= .03
    action, diag = controller.refine(sequence[:, 0], context={'tick': 0})
    assert not action[0, :6].any()
    assert torch.isfinite(action).all()
    assert diag['per_env'][0]['pressure_unknown_normal_hands'] == [True, False]


def test_pressure_targets_respect_existing_joint_lead_and_rate_limits():
    env = pressure_env(); sequence = torch.ones(1, 4, 24)
    env.task.command[:] = env.task.m['q'] + .099
    controller = optimizer(env, sequence)
    before = env.task.command.clone()
    action, _ = controller.refine(sequence[:, 0], context={'tick': 0})
    target = before + action[:, :12].double() * .005
    assert (target <= env.task.m['q'] + .1 + 1e-9).all()
    assert (target >= env.task.m['q'] - .1 - 1e-9).all()
    assert (target - before).abs().max() <= .005 + 1e-9


def test_mixed_pressure_invalid_row_and_terminal_reset_do_not_change_pinch_route():
    env = pressure_env(3); sequence = torch.zeros(3, 4, 24)
    controller = optimizer(env, sequence, [PINCH, TRIPOD, TRIPOD])
    env.task.m['q'][1, 0] = float('nan')
    action, diag = controller.refine(sequence[:, 0], context={'tick': 0})
    assert torch.isfinite(action).all() and not diag['per_env'][1]['active']
    assert diag['per_env'][2]['active']
    controller.observe_result(torch.tensor([False, True, False]))
    action, diag = controller.refine(sequence[:, 1], context={'tick': 1}, alive=torch.tensor([True, False, True]))
    assert not action[1].any() and diag['per_env'][1]['reason'] == 'inactive_episode'


def test_tripod_invalid_command_holds_all_fingers_and_preserves_teacher_wrists():
    env = pressure_env(); sequence = torch.full((1, 4, 24), .3)
    controller = optimizer(env, sequence)
    env.task.command[0, 0] = float('nan')
    action, diag = controller.refine(sequence[:, 0], context={'tick': 0})
    assert not action[:, :12].any()
    assert torch.equal(action[:, 12:], sequence[:, 0, 12:])
    assert diag['per_env'][0]['reason'] == 'invalid_absolute_command'


def test_mixed_batch_pinch_action_is_bitwise_equal_to_standalone_absolute_route():
    mixed = native_env(2); single = native_env()
    sequence = torch.zeros(2, 5, 24); sequence[:, :, 12:] = .3
    pressure = optimizer(mixed, sequence, [PINCH, TRIPOD])
    reference = AbsoluteTeacherReferenceMPC(single, 'a' * 64, sequence[:1])
    for tick in range(4):
        actual, _ = pressure.refine(sequence[:, tick], context={'tick': tick})
        expected, _ = reference.refine(sequence[:1, tick], context={'tick': tick})
        assert torch.equal(actual[:1], expected)
        mixed.task.command += actual[:, :12].double() * .005
        single.task.command += expected[:, :12].double() * .005


def test_repeated_pressure_relief_can_accumulate_past_old_absolute_reference_box():
    env = pressure_env(); sequence = torch.full((1, 24, 24), .3)
    controller = optimizer(env, sequence)
    measured = {key: value.clone() for key, value in env.task.m.items()}
    for tick in range(20):
        before = env.task.command.clone()
        action, diag = controller.refine(sequence[:, tick], context={'tick': tick})
        assert diag['per_env'][0]['active']
        assert action[0, 3] < 0
        assert torch.equal(action[:, 12:], sequence[:, tick, 12:])
        assert (action[:, :12].double() * .005).abs().max() <= .005
        env.task.command += action[:, :12].double() * .005
        assert (env.task.command - before).abs().max() <= .005
    assert abs(float(env.task.command[0, 3] - controller.absolute_commands[0, 19, 0, 3])) > .001
    for key in measured:
        assert torch.equal(env.task.m[key], measured[key])


def test_unknown_hand_hold_box_cannot_bypass_existing_measured_joint_lead():
    env = pressure_env(); sequence = torch.zeros(1, 4, 24)
    controller = optimizer(env, sequence)
    env.task.command[0, :6] = env.task.m['q'][0, :6] + .12
    env.task.m['effective_pad_vectors_w'][0, 0, 0] = 0
    action, diag = controller.refine(sequence[:, 0], context={'tick': 0})
    assert not controller.native.pending.any()
    assert not action[:, :12].any()
    assert diag['per_env'][0]['reason'] == 'pressure_model_fallback_hold'
    assert controller.native.stats['model_fallback_env_steps'] == 1
