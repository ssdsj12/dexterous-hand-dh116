"""CPU reward settlement over explicit source schemas, without scalar engines."""
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from dh116.rl.reward import load_config
from dh116.rl.reward import SourceFailure as SourceFailure
from dh116.rl.reward import train_rewar__ERROR_FIELDS as ERROR_FIELDS
from dh116.rl.reward import TensorTrainingReward
from dh116.rl.reward import TensorLaneSchema
from dh116.rl.reward import TensorRewardFrame
from dh116.rl.reward import measured_finger_errors
ROOT = Path(__file__).resolve().parents[1]
GUARD = 'original_guard_joint_force_clearance_geometry'

def setup():
    config = load_config(ROOT / 'configs/training/reward_pinch.json', ppo_gamma=0.995)
    specs = (('distance', 'm', 'measured_material_distance'), ('numerical_health', 'dimensionless', 'actual_solver_health'), ('constraint_failure', 'dimensionless', 'original_constraint_guard'), (GUARD, 'dimensionless', 'original_geometry_guard'))
    specs += tuple(((name, 'rad', 'measured_named_active_q') for name in ERROR_FIELDS))
    schemas = tuple((TensorLaneSchema(f'ep-{i}', i, f'reset-{i}', f'rod-{i}', 'cpu_fixture', 'fixture_endpoint', tuple((str(j) for j in range(41))), specs) for i in range(2)))
    reward = TensorTrainingReward(config, num_envs=2, device='cpu')
    reward.reset_selected((0, 1), schemas, frame(0), previous_request=torch.zeros(2, 24, dtype=torch.float64), tokens=('reset', 'reset'), policy_uids=('policy', 'policy'))
    assert not reward._terminal.any()
    return (reward, schemas, config)

def frame(tick, *, distance=(0.02, 0.02), error=0.01, fail_lane=None):
    fields = dict(distance=torch.tensor(distance, dtype=torch.float64), numerical_health=torch.ones(2, dtype=torch.bool), constraint_failure=torch.zeros(2, dtype=torch.bool), **{GUARD: torch.zeros(2, dtype=torch.bool)})
    fields.update({name: torch.full((2,), error, dtype=torch.float64) for name in ERROR_FIELDS})
    if fail_lane is not None:
        fields['constraint_failure'][fail_lane] = True
        fields[GUARD][fail_lane] = True
    return TensorRewardFrame(torch.full((2,), tick, dtype=torch.int64), torch.full((2,), tick / 480, dtype=torch.float64), fields)

def begin(reward, schemas, tick=0, token='w0', **changes):
    args = dict(request=torch.zeros(2, 24, dtype=torch.float64), tokens=(token, token), policy_uids=('policy', 'policy'), physics_tick=torch.full((2,), tick, dtype=torch.int64), solver_time_s=torch.full((2,), tick / 480, dtype=torch.float64), remaining_substeps=8)
    args.update(changes)
    reward.begin(schemas, **args)

def test_complete_eight_substeps_numeric_shaping_and_external_boundaries():
    (reward, schemas, config) = setup()
    begin(reward, schemas, request=torch.stack((torch.ones(24), torch.zeros(24))).double())
    for tick in range(1, 8):
        assert not reward.observe(frame(tick, distance=(0.01, 0.02))).any()
    with pytest.raises(RuntimeError, match='incomplete'):
        reward.settle_once(('w0', 'w0'))
    assert reward.observe(frame(8, distance=(0.01, 0.02))).all()
    packet = reward.settle_once(('w0', 'w0'), truncated=torch.tensor([False, True]), buffer_full=torch.tensor([True, False]))
    torch.testing.assert_close(packet.reward, torch.tensor([0.988125, -0.006666666666666667], dtype=torch.float64), atol=1e-13, rtol=0)
    torch.testing.assert_close(packet.cost_per_second, torch.tensor([[0.1125] * 8, [0.1] * 8], dtype=torch.float64), atol=1e-15, rtol=0)
    assert packet.substeps.tolist() == [8, 8] and packet.done.tolist() == [False, True] and (not packet.terminated.any())
    torch.testing.assert_close(packet.duration_s, torch.full((2,), 1 / 60, dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(packet.bootstrap(torch.ones(2, dtype=torch.float64)), torch.full((2,), config.gamma, dtype=torch.float64), atol=0, rtol=0)
    assert not packet.native_qualification
    with pytest.raises(RuntimeError, match='already settled|absent'):
        reward.settle_once(('w0', 'w0'))

def test_true_terminal_absorbs_potential_and_cost_while_truncation_bootstraps():
    (reward, schemas, config) = setup()
    begin(reward, schemas)
    for tick in range(1, 9):
        observed = frame(tick, distance=(0.01, 0.01), fail_lane=0 if tick == 3 else None)
        if tick > 3:
            observed.fields['distance'][0] = float('nan')
            observed.physics_tick[0] = 999
        reward.observe(observed)
    packet = reward.settle_once(('w0', 'w0'), truncated=torch.tensor([False, True]))
    assert packet.substeps.tolist() == [3, 8] and packet.event.tolist() == [-40.0, 0.0]
    assert packet.terminated.tolist() == [True, False] and packet.done.tolist() == [True, True]
    torch.testing.assert_close(packet.reward, torch.tensor([-41.000625, 0.9883333333333333], dtype=torch.float64), atol=1e-13, rtol=0)
    torch.testing.assert_close(packet.bootstrap(torch.ones(2, dtype=torch.float64)), torch.tensor([0.0, config.gamma], dtype=torch.float64), atol=0, rtol=0)
    assert packet.endpoint_phi.tolist() == [0.0, 2.0] and torch.isnan(packet.cost_per_second[0, 3:]).all()

def test_arrival_event_cannot_be_farmed_by_leaving_and_returning_to_goal():
    (reward, schemas, config) = setup()
    events = []
    assert config.finger_goal.arrival_steps == 16 and config.finger_goal.arrival_reward == 2.0
    for (window, error) in enumerate((0.01, 0.01, 0.4, 0.01, 0.01)):
        token = f'w{window}'
        begin(reward, schemas, tick=window * 8, token=token)
        for offset in range(1, 9):
            reward.observe(frame(window * 8 + offset, error=error))
        packet = reward.settle_once((token, token))
        events.append(packet.event.tolist())
    assert events == [[0.0, 0.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    assert not packet.done.any() and (not packet.native_qualification)

@pytest.mark.parametrize('fault', ['source', 'reset_identity', 'pre_tick', 'pre_time', 'interval'])
def test_wrong_source_or_actual_stamp_quarantines_without_a_ppo_packet(fault):
    (reward, schemas, _) = setup()
    with pytest.raises(SourceFailure):
        if fault == 'source':
            begin(reward, (replace(schemas[0], source='other_reader'), schemas[1]))
        elif fault == 'reset_identity':
            begin(reward, (schemas[0], replace(schemas[1], reset_uid='other-reset')))
        elif fault == 'pre_tick':
            begin(reward, schemas, physics_tick=torch.tensor([0, 1]))
        elif fault == 'pre_time':
            begin(reward, schemas, solver_time_s=torch.tensor([0.0, 1 / 480], dtype=torch.float64))
        else:
            begin(reward, schemas)
            reward.observe(frame(2))
    assert reward.quarantined.all()
    with pytest.raises(SourceFailure, match='quarantin'):
        reward.settle_once(('w0', 'w0'))

def test_measured_goal_errors_use_named_hand_and_participating_joint_axes():
    (_, _, config) = setup()
    goal = config.finger_goal
    q = torch.tensor([goal.target_rad] * 2, dtype=torch.float64)
    q[0, 0, goal.participating[0]] += 0.07
    q[1, 1, goal.participating[-1]] += 0.11
    torch.testing.assert_close(measured_finger_errors(q, goal), torch.tensor([[0.07, 0.0], [0.0, 0.11]], dtype=torch.float64), atol=1e-15, rtol=0)
    q[0, 0, 5] = float('nan')
    assert torch.isnan(measured_finger_errors(q, goal)[0]).all()

def test_goal_full_recipe_preserves_hold_standard_and_a_32_window_rollout():
    for grasp in ('pinch', 'tripod', 'power'):
        config = load_config(ROOT / f'configs/training/reward_{grasp}.json', ppo_gamma=0.995, full_grasp=True)
        engine = TensorTrainingReward(config, num_envs=2, device='cpu')
        assert engine.full and config.finger_goal.grasp == grasp
        assert config.ppo['rollout_steps'] == 32
        assert config.events['trusted_constraint_failure'] == -40
        assert dict(config.success) == {'net_lift_m': 0.03, 'hold_s': 5.0, 'max_slip_m': 0.0005}


def goal_full_setup(*, defer_checks=True, first_contact=1):
    from dh116.rl.reward import GOAL_SOFT_COSTS, train_rewar___FULL_UNITS

    config = load_config(ROOT / 'configs/training/reward_pinch.json', full_grasp=True)
    reward = TensorTrainingReward(config, num_envs=2, device='cpu', defer_checks=defer_checks)
    units = {**train_rewar___FULL_UNITS, 'load_progress_left': 'dimensionless',
             'load_progress_right': 'dimensionless', 'support_unload': 'dimensionless',
             'hand_lift_load': 'dimensionless', 'balanced_hand_lift_load': 'dimensionless', 'slip': 'm',
             **dict.fromkeys(GOAL_SOFT_COSTS, 'dimensionless')}
    specs = (*((name, 'rad', 'fixture') for name in ERROR_FIELDS),
             ('distance', 'm', 'fixture'),
             *((name, 'dimensionless', 'fixture') for name in
               ('numerical_health', 'constraint_failure', GUARD)),
             *((name, unit, 'fixture') for name, unit in units.items()))
    schemas = tuple(TensorLaneSchema(
        f'ep-{i}', i, f'reset-{i}', f'rod-{i}', 'cpu_fixture', 'fixture', ('0', '1'), specs,
        required_pad_ids=(('left', ('thumb', 'index')), ('right', ('thumb', 'index'))),
        anchor_specs=tuple((f'{i}:{anchor}', 'drift', 'uncertainty') for anchor in range(4)),
    ) for i in range(2))

    def endpoint(tick, *, touch=0., carried=False, detached=False, upward=0., min_z=0., median_z=0.01):
        base = frame(tick)
        fields = dict(base.fields)
        for name, unit in units.items():
            if name != 'slip':
                dtype = (torch.float64 if unit == 'm' or name.startswith(('load_progress', 'coverage'))
                         or name in (*GOAL_SOFT_COSTS, 'support_unload', 'hand_lift_load',
                                     'balanced_hand_lift_load') else torch.bool)
                fields[name] = torch.zeros(2, dtype=dtype)
        fields['load_progress_left'].fill_(touch)
        fields['load_progress_right'].fill_(touch)
        if carried:
            for name in ('coverage_left', 'coverage_right', 'bilateral_support', 'opposed_left', 'opposed_right', 'off_table'):
                fields[name].fill_(1)
            fields['support_unload'].fill_(0.8)
            fields['hand_lift_load'].fill_(upward)
            fields['balanced_hand_lift_load'].fill_(upward)
            fields['median_z'].fill_(median_z)
            fields['min_z'].fill_(min_z)
            fields['no_external_support'].fill_(detached)
        return replace(base, fields=fields, anchor_present=torch.full((2,), tick >= first_contact),
                       anchor_drift=torch.zeros((2, 4), dtype=torch.float64),
                       anchor_uncertainty=torch.zeros((2, 4), dtype=torch.float64),
                       anchor_reference=torch.zeros((2, 4, 1), dtype=torch.float64),
                       anchor_reference_tick=torch.full((2, 4), first_contact, dtype=torch.long))

    reward.reset_selected((0, 1), schemas, endpoint(0), previous_request=torch.zeros(2, 24),
                          tokens=('reset', 'reset'), policy_uids=('policy', 'policy'))
    return reward, schemas, endpoint


@pytest.mark.parametrize('defer_checks', (False, True))
def test_goal_full_rewards_light_contact_and_continuous_measured_lift(defer_checks):
    reward, schemas, endpoint = goal_full_setup(defer_checks=defer_checks, first_contact=9)
    phases = (
        dict(touch=.25), dict(touch=1., carried=True, min_z=.005),
        dict(touch=.55), dict(touch=1., carried=True, upward=.8, min_z=.005),
        dict(touch=1., carried=True, upward=.8, min_z=.005, detached=True),
        dict(touch=.55), dict(touch=.55), dict(touch=.55), dict(touch=.55), dict(touch=.55), {},
    )
    packets = []
    for index, phase in enumerate(phases):
        token = f'w{index}'
        begin(reward, schemas, tick=8 * index, token=token)
        for tick in range(8 * index + 1, 8 * index + 9):
            measured = endpoint(tick, **phase)
            if index == 0:
                measured.fields['load_progress_left'][0] = 0.50
                measured.fields['load_progress_right'][0] = 0.0
            if index in (2, 5, 6, 7, 8, 9):
                measured.fields['hand_lift_load'].fill_(.8)
                measured.fields['balanced_hand_lift_load'].fill_(0. if index == 9 else .8)
                measured.fields['support_unload'].fill_(.8)
                measured.fields['min_z'].fill_(-.037 if index == 8 else -.019)
                measured.fields['median_z'].fill_(-.009)
                measured.fields['off_table'].fill_(index != 6)
                if index == 5:
                    measured.fields['load_progress_right'].zero_()
                if index == 7:
                    measured.fields['soft_non_target_contact'][0] = .1
                    measured.fields['soft_reaction_force'][1] = 1.0
            reward.observe(measured)
        packets.append(reward.settle_once((token, token)))

    progress = packets[0].terms['load_progress']['contribution']
    assert progress[1] > progress[0] > 0
    for index in (0, 1, 5, 6, 7, 8, 9, 10):
        assert not packets[index].terms['prelift']['contribution'].any()
    transfer = packets[2].terms['prelift']['contribution']
    assert (transfer > 0).all()
    assert not packets[2].terms['lift']['contribution'].any()
    assert (packets[3].terms['prelift']['contribution'] > transfer).all()
    for name in ('prelift', 'lift'):
        torch.testing.assert_close(packets[3].terms[name]['contribution'],
                                   packets[4].terms[name]['contribution'], atol=0., rtol=0.)
    assert all(not packet.terminated.any() for packet in packets)

def test_goal_full_clean_contact_and_unsaturated_force_are_required_at_success_boundary():
    from dh116.physics.runtime import NativeTensorPhysicsRuntime

    reward, schemas, endpoint = goal_full_setup()
    runtime = NativeTensorPhysicsRuntime.__new__(NativeTensorPhysicsRuntime)
    runtime._hand_mesh_slots = (0, 2)
    wrench = torch.zeros((2, 3, 6), dtype=torch.float64)
    force_limit = wrench.new_tensor(40.)
    wrench[:, 0, 0] = torch.nextafter(force_limit, wrench.new_tensor(0.))
    wrench[:, 1, 0] = 80.  # External support is tracked by separate contact readouts.
    below_limit = runtime._reaction_force_cost(wrench)
    wrench[1, 2, 1] = force_limit
    at_limit = runtime._reaction_force_cost(wrench)
    assert (below_limit < 1.0).all() and at_limit[1] == 1.0
    for window in range(301):
        token = f'hold-{window}'
        begin(reward, schemas, tick=window * 8, token=token)
        for tick in range(window * 8 + 1, window * 8 + 9):
            measured = endpoint(tick, carried=True, detached=True, upward=1., min_z=.031, median_z=.031)
            measured.fields['hold_motion'].fill_(True)
            measured.fields['soft_reaction_force'] = at_limit if tick == 2401 else below_limit
            if tick == 2401:
                measured.fields['soft_non_target_contact'][0] = 0.01
            reward.observe(measured)
        packet = reward.settle_once((token, token))
        if window == 299:
            torch.testing.assert_close(packet.post_context['hold_s'], torch.full((2,), 2399 / 480, dtype=torch.float64))
    assert packet.event.tolist() == [0.0, 0.0]
    assert packet.terminated.tolist() == [False, False]
    torch.testing.assert_close(packet.post_context['hold_s'], torch.full((2,), 6 / 480, dtype=torch.float64))
    assert packet.substep_terms['soft_non_target_contact']['contribution'][0, 0] > 0
    assert packet.substep_terms['soft_reaction_force']['contribution'][1, 0] > 0


@pytest.mark.parametrize('already_terminal', (False, True))
def test_goal_full_nan_tail_is_only_ignored_after_trusted_terminal(already_terminal):
    reward, schemas, endpoint = goal_full_setup()
    begin(reward, schemas)
    for tick in range(1, 9):
        measured = endpoint(tick, carried=True)
        if already_terminal and tick == 2:
            measured.fields['constraint_failure'][0] = True
        if tick == 3:
            measured.fields['numerical_health'][0] = False
            measured.fields['distance'][0] = float('nan')
        reward.observe(measured)
    if not already_terminal:
        with pytest.raises(SourceFailure, match='invalid source'):
            reward.settle_once(('w0', 'w0'))
        assert reward.quarantined.all()
        return
    packet = reward.settle_once(('w0', 'w0'))
    assert packet.event.tolist() == [-40.0, 0.0]
    assert packet.substeps.tolist() == [2, 8]
    assert packet.post_context['physics_tick'].tolist() == [2, 8]
    assert torch.isfinite(packet.reward).all() and not reward.quarantined.any()


def test_wrist_reference_is_invariant_to_lane_origin_and_rod_yaw():
    from dh116.physics.runtime import rod_local_wrists
    rod = torch.zeros((1, 41, 3), dtype=torch.float64)
    rod[0, :, 1] = torch.linspace(-0.25, 0.25, 41, dtype=rod.dtype)
    wrists = torch.tensor([[[0.012, -0.20, 0.025, 1, 0, 0, 0],
                            [0.018, 0.19, 0.040, 1, 0, 0, 0]]], dtype=rod.dtype)
    ids = torch.tensor([[4, 36]])
    position, rotation = rod_local_wrists(rod, wrists, ids)
    yaw = rod.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    origin = rod.new_tensor([17, 23, 0.5])
    shifted = wrists.clone()
    shifted[..., :3] = wrists[..., :3] @ yaw.T + origin
    shifted[..., 3:] = rod.new_tensor([1, 0, 0, 1]) / (2 ** 0.5)
    moved_position, moved_rotation = rod_local_wrists(rod @ yaw.T + origin, shifted, ids)
    torch.testing.assert_close(moved_position, position, rtol=0, atol=1e-13)
    torch.testing.assert_close(moved_rotation, rotation, rtol=0, atol=1e-13)
