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

def test_wrist_reference_is_invariant_to_lane_origin_and_rod_yaw():
    from dh116.physics.runtime import rod_local_wrists
    rod = torch.zeros((1, 41, 3), dtype=torch.float64)
    rod[0, :, 1] = torch.linspace(-0.25, 0.25, 41, dtype=rod.dtype)
    wrists = torch.tensor([[[0.012, 0.20, 0.025, 1, 0, 0, 0],
                            [0.018, -0.19, 0.040, 1, 0, 0, 0]]], dtype=rod.dtype)
    ids = torch.tensor([[36, 4]])
    position, rotation = rod_local_wrists(rod, wrists, ids)
    yaw = rod.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    origin = rod.new_tensor([17, 23, 0.5])
    shifted = wrists.clone()
    shifted[..., :3] = wrists[..., :3] @ yaw.T + origin
    shifted[..., 3:] = rod.new_tensor([1, 0, 0, 1]) / (2 ** 0.5)
    moved_position, moved_rotation = rod_local_wrists(rod @ yaw.T + origin, shifted, ids)
    torch.testing.assert_close(moved_position, position, rtol=0, atol=1e-13)
    torch.testing.assert_close(moved_rotation, rotation, rtol=0, atol=1e-13)
