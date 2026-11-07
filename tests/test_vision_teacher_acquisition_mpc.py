"""Opt-in CAD acquisition: real contacts and original wrist intent stay separate."""
import importlib

import pytest
import torch

from test_vision_teacher_pressure_mpc import pressure_env, optimizer as pressure_optimizer, PINCH, TRIPOD


def fixture(n=1, gap=.0002, facing=True):
    env = pressure_env(n)
    env.task.m['effective_pad_loads_n'].zero_()
    env.task.m['effective_pad_vectors_w'].zero_()
    env.task.m['effective_pad_valid'].zero_()
    env.task.m['effective_pad_points_w'].zero_()
    env.world.hands[0].data.body_link_pose_w[:, 1:, 0] = .008 + gap
    atlas = dict(geometry_only=True, sides={side: {
        name: dict(center_local_m=[0., 0., 0.], reference_outward_normal_local=[-1. if facing else 1., 0., 0.], geometry_only=True)
        for name in env.world.hands[0].body_names[1:]} for side in ('left', 'right')})
    return env, atlas


def controller(env, atlas, sequence, styles=None):
    module = importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_teacher_acquisition_mpc')
    return module.AcquisitionTeacherReferenceMPC(env, 'a' * 64, sequence,
        grasp_styles=styles or [TRIPOD] * env.world.n, cylinder_radius_m=.008,
        cylinder_length_m=.5, pad_atlas=atlas)


def test_missing_pads_acquire_without_manufacturing_measured_contact_and_preserve_wrists():
    env, atlas = fixture(); sequence = torch.zeros(1, 5, 24)
    sequence[:, :, 13] = .1  # Cylinder-axis wrist transport does not help radial closure.
    candidate = controller(env, atlas, sequence)
    measured = {k: v.clone() for k, v in env.task.m.items()}
    action, diag = candidate.refine(sequence[:, 0], context={'tick': 0})
    assert diag['per_env'][0]['active']
    assert (action[0, [0, 2, 3, 6, 8, 9]] < 0).all()
    assert torch.equal(action[:, 12:], sequence[:, 0, 12:])
    assert diag['force_prediction'] is False and diag['optimized_action_dim'] == 12
    assert diag['geometry_points_kind'] == 'CAD_FK_geometry_only_not_measured_contact'
    for key, value in measured.items(): assert torch.equal(env.task.m[key], value)
    assert not env.task.m['effective_pad_valid'].any()


@pytest.mark.parametrize('case', ['active', 'below_air', 'invalid_q', 'invalid_command', 'bad_native_sensor'])
def test_pinch_is_bitwise_pressure_profile_including_invalid_fallback(case):
    old_env, _ = fixture(); env, atlas = fixture()
    for item in (old_env, env):
        item.task.m['effective_pad_loads_n'][:] = item.task.m['effective_pad_loads_n'].new_tensor([1., .5, .5, 0., 0.])
        item.task.m['effective_pad_vectors_w'][..., 0] = item.task.m['effective_pad_loads_n'] * item.task.m['effective_pad_loads_n'].new_tensor([1., -1., -1., 0., 0.])
        item.task.m['effective_pad_valid'] = item.task.m['effective_pad_loads_n'] > 0
        if case == 'active': item.task.m['bottom'][:] = .39
    sequence = torch.full((1, 5, 24), .1)
    old = pressure_optimizer(old_env, sequence, [PINCH]); new = controller(env, atlas, sequence, [PINCH])
    for item in (old_env, env):
        if case == 'invalid_q': item.task.m['q'][0, 0] = float('nan')
        if case == 'invalid_command': item.task.command[0, 0] = float('nan')
        if case == 'bad_native_sensor':
            item.world.pad_measurements = {'effective_pad_bad_env': torch.ones(1,dtype=torch.bool)}
            item.world.adapter_fault_bits = torch.ones(1,dtype=torch.long)
    expected, old_diag = old.refine(sequence[:, 0], context={'tick': 0})
    actual, new_diag = new.refine(sequence[:, 0], context={'tick': 0})
    assert torch.equal(actual, expected)
    assert old_diag['per_env'][0]['reason'] == new_diag['per_env'][0]['reason']


@pytest.mark.parametrize('bad', ['wrong_facing', 'inside', 'end_face'])
def test_geometry_ineligible_missing_fingers_hold_instead_of_blind_source_closure(bad):
    env, atlas = fixture(gap=-.0002 if bad == 'inside' else .0002, facing=bad != 'wrong_facing')
    if bad == 'end_face': env.world.hands[0].data.body_link_pose_w[:, 1:, 1] = .251
    sequence = torch.full((1, 5, 24), .1)
    candidate = controller(env, atlas, sequence)
    action, diag = candidate.refine(sequence[:, 0], context={'tick': 0})
    assert not action[0, [0, 1, 2, 3, 6, 7, 8, 9]].any()
    assert not torch.tensor(diag['per_env'][0]['geometry_eligible']).any()
    assert torch.equal(action[:, 12:], sequence[:, 0, 12:])


def test_geometry_world_target_does_not_move_with_teacher_wrist_transport():
    env, atlas = fixture(); sequence = torch.zeros(1, 5, 24); sequence[:, :, 12] = .1
    candidate = controller(env, atlas, sequence)
    calls = []; original = candidate.native.pressure_planner.plan
    def record(**kwargs): calls.append(kwargs); return original(**kwargs)
    candidate.native.pressure_planner.plan = record
    candidate.refine(sequence[:, 0], context={'tick': 0})
    assert calls
    assert calls[0]['wrist_motion'][0, 0, 0, 1, 0] > 0
    assert calls[0]['target_points'][0, 0, 0, 1, 0] < calls[0]['points'][0, 0, 1, 0]


def test_bearing_pad_pressure_behavior_matches_original_when_no_missing_required_pad():
    env, atlas = fixture(); old_env, _ = fixture()
    for item in (env, old_env):
        item.task.m['effective_pad_loads_n'][:] = item.task.m['effective_pad_loads_n'].new_tensor([5., .75, 5., 0., 0.])
        item.task.m['effective_pad_vectors_w'][..., 0] = item.task.m['effective_pad_loads_n'] * item.task.m['effective_pad_loads_n'].new_tensor([1., -1., -1., 0., 0.])
        item.task.m['effective_pad_valid'] = item.task.m['effective_pad_loads_n'] > 0
    sequence = torch.zeros(1, 5, 24)
    old = pressure_optimizer(old_env, sequence); new = controller(env, atlas, sequence)
    a, _ = old.refine(sequence[:, 0], context={'tick': 0})
    b, d = new.refine(sequence[:, 0], context={'tick': 0})
    assert torch.equal(a, b)
    assert not torch.tensor(d['per_env'][0]['geometry_eligible']).any()
    assert d['per_env'][0]['geometry_prediction_current'] is False
    assert d['per_env'][0]['geometry_predicted_gap_m'] is None


@pytest.mark.parametrize('bad', ['rejected', 'unknown', 'overloaded'])
def test_bad_hand_never_acquires_and_other_hand_is_isolated(bad):
    env, atlas = fixture(); sequence = torch.zeros(1, 5, 24)
    if bad == 'rejected': env.task.m['rejected_contact_loads_n'][0, 0] = .1
    else:
        env.task.m['effective_pad_loads_n'][0, 0, 0] = 5. if bad == 'overloaded' else 1.
        env.task.m['effective_pad_valid'][0, 0, 0] = True
        if bad == 'overloaded': env.task.m['effective_pad_vectors_w'][0, 0, 0, 0] = 5.
    candidate = controller(env, atlas, sequence)
    _, diag = candidate.refine(sequence[:, 0], context={'tick': 0})
    eligible = torch.tensor(diag['per_env'][0]['geometry_eligible'])
    assert not eligible[0].any() and eligible[1, :3].all()


def test_unreachable_missing_geometry_holds_and_reports_infeasible():
    env, atlas = fixture(); sequence = torch.zeros(1, 5, 24)
    env.world.hands[0].root_physx_view.get_jacobians().zero_()
    candidate = controller(env, atlas, sequence)
    action, diag = candidate.refine(sequence[:, 0], context={'tick': 0})
    assert not action[:, :12].any()
    assert torch.tensor(diag['per_env'][0]['geometry_model_infeasible'])[..., :3].all()


def test_request_budgets_do_not_count_bearing_time_or_reset_on_contact_flicker():
    env, atlas = fixture(); sequence = torch.zeros(1, 120, 24)
    candidate = controller(env, atlas, sequence)
    spent = []
    for tick in range(90):
        if tick % 2:
            env.task.m['effective_pad_loads_n'][:] = env.task.m['effective_pad_loads_n'].new_tensor([1.5,.75,.75,0,0])
            env.task.m['effective_pad_vectors_w'][..., 0] = env.task.m['effective_pad_loads_n'] * env.task.m['effective_pad_loads_n'].new_tensor([1.,-1.,-1.,0.,0.])
            env.task.m['effective_pad_valid'] = env.task.m['effective_pad_loads_n'] > 0
        else:
            env.task.m['effective_pad_loads_n'].zero_(); env.task.m['effective_pad_vectors_w'].zero_(); env.task.m['effective_pad_valid'].zero_()
        _, diag = candidate.refine(sequence[:, tick], context={'tick': tick})
        spent.append(torch.tensor(diag['per_env'][0]['geometry_requested_approach_m']))
    assert all(bool((b >= a).all()) for a, b in zip(spent, spent[1:]))
    assert spent[-1].max() <= .001 + 1e-9
    assert not torch.tensor(diag['per_env'][0]['geometry_eligible']).any()
    assert (torch.tensor(diag['per_env'][0]['geometry_no_progress_s'])[..., :3] >= .5-1e-8).all()


def test_geometry_request_time_and_displacement_are_bounded_when_actual_gap_progresses():
    env, atlas = fixture(gap=.001); sequence = torch.zeros(1, 90, 24)
    candidate = controller(env, atlas, sequence)
    for tick in range(70):
        env.world.hands[0].data.body_link_pose_w[:, 1:, 0] = .009 - tick * .000002
        action, diag = candidate.refine(sequence[:, tick], context={'tick': tick})
    row = diag['per_env'][0]
    assert torch.tensor(row['geometry_requested_approach_m']).max() <= .001+1e-9
    assert torch.tensor(row['geometry_request_time_s']).max() <= 1.+1e-9
    assert not torch.tensor(row['geometry_eligible']).any()
    assert not action[0, [0,1,2,3,6,7,8,9]].any()


def test_invalid_sibling_and_terminal_death_are_isolated_and_prediction_matches_action():
    env, atlas = fixture(3); sequence = torch.zeros(3, 5, 24)
    candidate = controller(env, atlas, sequence)
    env.task.m['object_quat'][1] = float('nan')
    before = env.task.command.clone()
    action, diag = candidate.refine(sequence[:, 0], context={'tick': 0})
    assert torch.isfinite(action).all() and not action[1, :12].any()
    assert diag['per_env'][0]['active'] and diag['per_env'][2]['active']
    mask = candidate.native.pending
    expected = env.task.m['q'].reshape(3,2,6) + .5 * action[:, :12].double().reshape(3,2,6) * .005
    torch.testing.assert_close(candidate.native.last_prediction[mask], expected[mask])
    target = before + action[:, :12].double() * .005
    assert (target - env.task.m['q']).abs().max() <= .1
    assert (target - before).abs().max() <= .005
    candidate.observe_result(torch.tensor([True, False, False]))
    action, _ = candidate.refine(sequence[:, 1], context={'tick': 1}, alive=torch.tensor([False, True, True]))
    assert not action[0].any()
    with pytest.raises(ValueError, match='ended'):
        candidate.refine(sequence[:, 2], context={'tick': 2}, alive=torch.ones(3,dtype=torch.bool))


@pytest.mark.parametrize('source', ['world_bad_env', 'world_fault', 'measurement_bad_env'])
def test_invalid_native_sensor_row_cannot_acquire_and_siblings_remain_active(source):
    env, atlas = fixture(3); sequence = torch.zeros(3, 5, 24)
    bad = torch.tensor([False, True, False])
    if source.startswith('world'):
        env.world.pad_measurements = {'effective_pad_bad_env': bad if source == 'world_bad_env' else torch.zeros_like(bad)}
        env.world.adapter_fault_bits = bad.long() * (8 if source == 'world_fault' else 0)
        # Native world validity takes precedence over a stale task placeholder.
        env.task.m['effective_pad_bad_env'] = torch.zeros_like(bad)
    else: env.task.m['effective_pad_bad_env'] = bad
    candidate = controller(env, atlas, sequence)
    action, diagnostics = candidate.refine(sequence[:, 0], context={'tick': 0})
    assert not action[1, :12].any()
    assert not torch.tensor(diagnostics['per_env'][1]['geometry_eligible']).any()
    assert diagnostics['per_env'][1]['geometry_sensor_invalid'] is True
    assert diagnostics['per_env'][1]['reason'] == 'invalid_geometry_sensor_hold'
    assert diagnostics['per_env'][0]['active'] and diagnostics['per_env'][2]['active']


@pytest.mark.parametrize('bad', [torch.zeros(1, 1, dtype=torch.bool), torch.zeros(1)])
def test_native_sensor_flag_schema_is_checked(bad):
    env, atlas = fixture(); sequence = torch.zeros(1, 5, 24)
    env.world.pad_measurements = {'effective_pad_bad_env': bad}
    env.world.adapter_fault_bits = torch.zeros(1, dtype=torch.long)
    candidate = controller(env, atlas, sequence)
    with pytest.raises(ValueError, match='shape/dtype/device'):
        candidate.refine(sequence[:, 0], context={'tick': 0})


def test_native_missing_validity_fails_closed_and_prediction_never_survives_a_fallback():
    env, atlas = fixture(); sequence = torch.zeros(1, 5, 24)
    candidate = controller(env, atlas, sequence)
    _, first = candidate.refine(sequence[:, 0], context={'tick': 0})
    assert first['per_env'][0]['geometry_prediction_current'] is True
    env.world.pad_measurements = {}
    env.world.adapter_fault_bits = torch.zeros(1, dtype=torch.long)
    action, second = candidate.refine(sequence[:, 1], context={'tick': 1})
    assert not action[:, :12].any()
    row = second['per_env'][0]
    assert row['geometry_sensor_invalid'] is True
    assert row['geometry_prediction_current'] is False
    assert row['geometry_predicted_gap_m'] is None


def test_every_resolve_rechecks_remaining_geometry_and_stops_coupled_bad_targets(monkeypatch):
    env, atlas = fixture(); sequence = torch.zeros(1, 5, 24)
    candidate = controller(env, atlas, sequence)
    pressure = importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_teacher_pressure_mpc')
    calls = []
    def coupled_solve(self, ids, **kwargs):
        calls.append(kwargs)
        predicted = kwargs['points'][:, None].expand(-1, self.horizon, -1, -1, -1).clone()
        predicted[..., 0] -= .00001
        # First solve rejects thumb. Removing it changes the index optimum;
        # that new violation must be caught by a second admission pass.
        digit = min(len(calls)-1, 2)
        if len(calls) < 4: predicted[:, :, 0, digit, 0] += .0001
        q = kwargs['q'][:, None].expand(-1, self.horizon, -1, -1)
        return kwargs['command'].clone(), dict(predicted_points=predicted,
            predicted_joints=q, nominal_cost=torch.zeros(1), optimized_cost=torch.zeros(1))
    monkeypatch.setattr(pressure._PressureNativeSurfaceMPC, '_plan', coupled_solve)
    _, diag = candidate.refine(sequence[:, 0], context={'tick': 0})
    assert len(calls) == 4
    row = diag['per_env'][0]
    assert torch.tensor(row['geometry_model_infeasible'])[0, :3].all()
    assert not torch.tensor(row['geometry_eligible'])[0].any()
    # Infeasible digits are held inside the original bounds on every re-solve.
    assert torch.equal(calls[-1]['lower'][0, :4], env.task.command[0, :4])
    assert torch.equal(calls[-1]['upper'][0, :4], env.task.command[0, :4])


def test_reacquisition_progress_uses_current_request_segment_not_old_bearing_negative_gap():
    env, atlas = fixture(gap=.0006); sequence = torch.zeros(1, 60, 24)
    candidate = controller(env, atlas, sequence)
    candidate.refine(sequence[:, 0], context={'tick': 0})
    env.world.hands[0].data.body_link_pose_w[:, 1:, 0] = .0078
    env.task.m['effective_pad_loads_n'][:] = env.task.m['effective_pad_loads_n'].new_tensor([1.5,.75,.75,0,0])
    env.task.m['effective_pad_vectors_w'][..., 0] = env.task.m['effective_pad_loads_n'] * env.task.m['effective_pad_loads_n'].new_tensor([1.,-1.,-1.,0.,0.])
    env.task.m['effective_pad_valid'] = env.task.m['effective_pad_loads_n'] > 0
    candidate.refine(sequence[:, 1], context={'tick': 1})
    env.task.m['effective_pad_loads_n'].zero_(); env.task.m['effective_pad_vectors_w'].zero_(); env.task.m['effective_pad_valid'].zero_()
    for tick in range(2, 39):
        env.world.hands[0].data.body_link_pose_w[:, 1:, 0] = .0086 - (tick-2)*.000002
        _, diag = candidate.refine(sequence[:, tick], context={'tick': tick})
    row = diag['per_env'][0]
    assert torch.tensor(row['geometry_eligible'])[..., :3].all()
    assert torch.tensor(row['geometry_no_progress_s']).max() < .05
    assert torch.tensor(row['geometry_request_time_s'])[..., :3].min() > .6
    assert torch.tensor(row['geometry_requested_approach_m'])[..., :3].min() > .0006


def test_real_positive_gap_progress_at_contact_acquisition_counts_without_replenishing_budget():
    env, atlas = fixture(gap=.0006); sequence = torch.zeros(1, 90, 24)
    candidate = controller(env, atlas, sequence)
    for tick in range(72):
        # Every request actually moves 2 um and creates contact in the next
        # observed state. Those are real progress samples even while bearing.
        env.world.hands[0].data.body_link_pose_w[:, 1:, 0] = .0086-((tick+1)//2)*.000002
        if tick % 2:
            env.task.m['effective_pad_loads_n'][:] = env.task.m['effective_pad_loads_n'].new_tensor([1.5,.75,.75,0,0])
            env.task.m['effective_pad_vectors_w'][..., 0] = env.task.m['effective_pad_loads_n'] * env.task.m['effective_pad_loads_n'].new_tensor([1.,-1.,-1.,0.,0.])
            env.task.m['effective_pad_valid'] = env.task.m['effective_pad_loads_n'] > 0
        else:
            env.task.m['effective_pad_loads_n'].zero_(); env.task.m['effective_pad_vectors_w'].zero_(); env.task.m['effective_pad_valid'].zero_()
        _, diag = candidate.refine(sequence[:, tick], context={'tick': tick})
    row = diag['per_env'][0]
    assert not torch.tensor(row['geometry_stop_code']).any()
    assert torch.tensor(row['geometry_no_progress_s']).max() == 0
    assert torch.tensor(row['geometry_request_time_s'])[..., :3].min() > .59
    assert torch.tensor(row['geometry_requested_approach_m'])[..., :3].min() > .00059


def test_overflowing_finite_CAD_quaternion_norm_cannot_create_geometry_action():
    env, atlas = fixture(2); sequence = torch.zeros(2, 5, 24)
    env.world.hands[0].data.body_link_pose_w[1, 1:, 3:] = 1e308
    candidate = controller(env, atlas, sequence)
    action, diag = candidate.refine(sequence[:, 0], context={'tick': 0})
    assert not action[1, :12].any()
    assert not torch.tensor(diag['per_env'][1]['geometry_eligible']).any()
    assert diag['per_env'][0]['active'] and (action[0,:12] != 0).any()
