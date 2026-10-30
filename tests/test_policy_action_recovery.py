"""CPU collection-only regressions; never invoke an optimizer or Isaac."""
import ast
import copy
import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_tensor_training as training
from go2_pvcnn.control.dh116.tabletop_tensor_task import FAILURE_CODES


def recovery(counts=None):
    name = 'go2_pvcnn.control.dh116.tabletop_policy_recovery'
    assert importlib.util.find_spec(name) is not None, 'bounded policy recovery is missing'
    return importlib.import_module(name).PolicyActionRecovery(counts)


class Env:
    continuous_motion = recovery = True

    def __init__(self, broken=None):
        self.actions = []
        self.resets = []
        self.tick = 0
        self.broken = broken

    def step(self, action):
        self.actions.append(action.clone())
        self.tick += 1
        invalid = ~torch.isfinite(action).all(1)
        if invalid.any():
            self.resets.append(invalid.nonzero().flatten().tolist())
        obs = torch.full((2, 6), float(self.tick))
        reward = torch.full((2,), float(self.tick))
        done = invalid.clone()
        recovered = invalid.clone()
        if invalid.any():
            if self.broken == 'obs': obs[0, 0] = float('nan')
            if self.broken == 'reward': reward[0] = float('inf')
            if self.broken == 'done': done[0] = False
            if self.broken == 'ack': recovered[0] = False
        if self.broken == 'transient_reward' and self.tick == 2:
            reward[0] = float('nan')
        return obs, reward, done, dict(recovered=recovered, phase=torch.ones(2, dtype=torch.long),
            failure=torch.where(invalid, FAILURE_CODES['invalid_action'], 0),
            pressure_limited=torch.zeros(2, dtype=torch.bool), terms={'test': reward.clone()},
            terminal_observation=obs.clone())


def setup_algorithm(monkeypatch, steps=3):
    obs = training.observations(torch.zeros(2, 6))
    algorithm = training.make_algorithm(obs, steps)
    def forbidden(*args, **kwargs):
        pytest.fail('collection tests must never optimize')
    monkeypatch.setattr(algorithm, 'update', forbidden)
    monkeypatch.setattr(algorithm.optimizer, 'step', forbidden)
    return algorithm, obs


def inject(monkeypatch, algorithm, field, calls=(2,), row=0, value=float('nan')):
    original = algorithm.act
    count = 0
    def act(obs):
        nonlocal count
        count += 1
        action = original(obs)
        if count in calls:
            # Distribution sigma can have stride 0 across the batch. Inject a
            # single row without accidentally corrupting every expanded row.
            tensor = getattr(algorithm.transition, field).clone()
            setattr(algorithm.transition, field, tensor)
            tensor.reshape(2, -1)[row, 0] = value
            if field == 'actions': action = tensor
        return action
    monkeypatch.setattr(algorithm, 'act', act)


@pytest.mark.parametrize('field,value', [
    ('actions', float('nan')), ('actions', float('inf')), ('actions', -float('inf')),
    ('values', float('nan')), ('actions_log_prob', float('inf')),
    ('action_mean', float('nan')), ('action_sigma', float('inf')), ('action_sigma', 0.),
])
def test_bad_policy_row_discards_whole_partial_rollout(monkeypatch, field, value):
    algorithm, obs = setup_algorithm(monkeypatch)
    before = copy.deepcopy(algorithm.policy.state_dict())
    inject(monkeypatch, algorithm, field, value=value)
    guard, env = recovery(), Env()
    writes = []
    add = algorithm.storage.add_transitions
    def checked_add(transition):
        for name in ('actions', 'values', 'actions_log_prob', 'action_mean', 'action_sigma'):
            assert torch.isfinite(getattr(transition, name)).all()
        writes.append(float(transition.rewards[0]))
        add(transition)
    monkeypatch.setattr(algorithm.storage, 'add_transitions', checked_add)
    step = env.step
    def checked_step(action):
        if not torch.isfinite(action).all():
            assert algorithm.storage.step == 0
            assert algorithm.transition.actions is None
            assert algorithm.transition.actions_log_prob is None
        return step(action)
    monkeypatch.setattr(env, 'step', checked_step)
    result = guard.collect(algorithm, env, obs, 3)
    assert env.resets == [[0]]
    if field == 'actions':
        bad = env.actions[1][0, 0]
        assert torch.isnan(bad) if value != value else bad == value
    assert algorithm.storage.step == 3
    # Only ticks 3,4,5 survive: no temporal credit from tick 1 across reset tick 2.
    torch.testing.assert_close(algorithm.storage.rewards[:, :, 0], torch.tensor([[3.,3.],[4.,4.],[5.,5.]]))
    for name in ('actions', 'values', 'actions_log_prob', 'mu', 'sigma', 'returns', 'advantages'):
        assert torch.isfinite(getattr(algorithm.storage, name)).all()
    assert result['reward_sum'] == 12.
    assert guard.counts['discarded_control_steps'] == 2
    assert guard.counts['discarded_transitions'] == 4
    assert guard.counts['discarded_rollouts'] == guard.counts['recoveries'] == 1
    assert guard.counts['env_control_steps'] == 5
    assert writes == [1., 3., 4., 5.]
    for key, value in before.items():
        torch.testing.assert_close(algorithm.policy.state_dict()[key], value)
    assert not algorithm.optimizer.state


def test_finite_out_of_range_keeps_raw_ppo_action(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    inject(monkeypatch, algorithm, 'actions', calls=(1,), value=2.)
    guard, env = recovery(), Env()
    guard.collect(algorithm, env, obs, 3)
    assert env.actions[0][0, 0] == 1.
    assert algorithm.storage.actions[0, 0, 0] == 2.
    assert not env.resets and not guard.counts['recoveries']


def test_same_slot_cannot_retry_after_one_good_tick(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    inject(monkeypatch, algorithm, 'actions', calls=(2, 4))
    guard, env = recovery(), Env()
    with pytest.raises(RuntimeError, match='recurr'):
        guard.collect(algorithm, env, obs, 3)
    assert env.resets == [[0]]
    assert algorithm.storage.step == 0
    assert algorithm.transition.actions is None
    assert getattr(algorithm.transition, 'actions_log_prob', None) is None


def test_retry_budget_renews_only_after_full_collection(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    inject(monkeypatch, algorithm, 'actions', calls=(1, 5))
    guard, env = recovery(), Env()
    result = guard.collect(algorithm, env, obs, 3)
    algorithm.storage.clear()  # emulate the collection boundary, never optimize
    guard.collect(algorithm, env, result['obs'], 3)
    assert env.resets == [[0], [0]]
    assert guard.counts['recoveries'] == 2


@pytest.mark.parametrize('broken', ['obs', 'reward', 'done', 'ack'])
def test_invalid_reset_result_clears_every_pending_sample(monkeypatch, broken):
    algorithm, obs = setup_algorithm(monkeypatch)
    inject(monkeypatch, algorithm, 'actions')
    guard, env = recovery(), Env(broken)
    with pytest.raises(RuntimeError, match='reset'):
        guard.collect(algorithm, env, obs, 3)
    assert algorithm.storage.step == 0
    assert algorithm.transition.actions is None
    assert getattr(algorithm.transition, 'actions_log_prob', None) is None
    assert len(env.actions) == 2


def test_bad_final_bootstrap_value_restarts_without_cross_gap_gae(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    evaluate = algorithm.policy.evaluate
    calls = 0
    def bad_once(obs):
        nonlocal calls
        calls += 1
        value = evaluate(obs)
        if calls == 4: value[0, 0] = float('nan')
        return value
    monkeypatch.setattr(algorithm.policy, 'evaluate', bad_once)
    guard, env = recovery(), Env()
    guard.collect(algorithm, env, obs, 3)
    assert env.resets == [[0]]
    assert algorithm.storage.step == 3
    assert algorithm.storage.rewards[0, 0, 0] == 5.
    assert torch.isfinite(algorithm.storage.returns).all()


def test_nonfinite_reward_is_reset_before_it_can_reach_storage(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    guard, env = recovery(), Env('transient_reward')
    guard.collect(algorithm, env, obs, 3)
    assert env.resets == [[0]]
    assert guard.counts['discarded_control_steps'] == 3
    assert algorithm.storage.rewards[:, 0, 0].tolist() == [4., 5., 6.]


def test_distinct_slots_each_have_one_attempt_before_full_collection(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    act, calls = algorithm.act, 0
    def faulty(obs):
        nonlocal calls
        calls += 1
        action = act(obs)
        if calls in (2, 4): action[0 if calls == 2 else 1, 0] = float('nan')
        return action
    monkeypatch.setattr(algorithm, 'act', faulty)
    guard, env = recovery(), Env()
    guard.collect(algorithm, env, obs, 3)
    assert env.resets == [[0], [1]]
    assert guard.counts['recoveries'] == 2
    assert algorithm.storage.rewards[:, 0, 0].tolist() == [5., 6., 7.]


def test_collection_callback_counts_discarded_physics_ticks(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    inject(monkeypatch, algorithm, 'actions')
    guard, env, ticks = recovery(), Env(), []
    guard.collect(algorithm, env, obs, 3, ticks.append)
    assert ticks == [1, 2, 3, 4, 5]


def test_broken_std_is_unrecoverable_without_optimizer_or_world_reset(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    with torch.no_grad(): algorithm.policy.log_std[0] = float('nan')
    guard, env = recovery(), Env()
    with pytest.raises(RuntimeError, match='policy.*unrecoverable'):
        guard.collect(algorithm, env, obs, 3)
    assert not env.actions
    assert algorithm.storage.step == 0 and algorithm.transition.actions is None


def test_persistent_nan_actor_weights_cannot_be_repaired_by_world_reset(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    with torch.no_grad(): algorithm.policy.actor[0].weight[0, 0] = float('nan')
    guard, env = recovery(), Env()
    with pytest.raises(RuntimeError, match='recurr|policy.*unrecoverable'):
        guard.collect(algorithm, env, obs, 3)
    assert len(env.actions) <= 1
    assert algorithm.storage.step == 0
    assert not algorithm.optimizer.state
    assert torch.isnan(algorithm.policy.actor[0].weight[0, 0])


def test_bootstrap_uses_the_single_checked_critic_result(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    evaluate, calls = algorithm.policy.evaluate, []
    def counted(obs):
        calls.append(obs['policy'].clone())
        return evaluate(obs)
    monkeypatch.setattr(algorithm.policy, 'evaluate', counted)
    recovery().collect(algorithm, Env(), obs, 3)
    assert len(calls) == 4
    assert torch.isfinite(algorithm.storage.returns).all()


def test_real_tensor_env_receives_nan_and_only_resets_bad_slot(monkeypatch):
    from test_dh116_tensor_env import World
    from test_dh116_tensor_task import template
    from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
    from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv
    task = TensorTask(template(), 2, 'cpu')
    task.continuous_motion = True
    world = World(task)
    env = TensorEnv(world, task, recovery=True)
    obs = training.observations(env.reset())
    world.resets.clear()
    world.tick = 100  # base fixture injects an unrelated 31 N fault at tick 3
    algorithm = training.make_algorithm(obs, 3)
    inject(monkeypatch, algorithm, 'actions')
    result = recovery().collect(algorithm, env, obs, 3)
    assert world.resets == [[0]]
    assert env.numerical_resets == 1
    assert algorithm.storage.step == 3
    assert torch.isfinite(algorithm.storage.actions).all()
    assert result['failures'][FAILURE_CODES['invalid_action']] == 1


class FailureEnv(Env):
    def __init__(self, scheduled, broken=None):
        super().__init__(broken)
        self.scheduled = scheduled

    def step(self, action):
        obs, reward, done, extra = super().step(action)
        for row, code in self.scheduled.get(self.tick, []):
            done[row] = True
            extra['failure'][row] = code
        return obs, reward, done, extra


def test_failure_counts_include_discarded_fragment_quarantine_and_final_rollout(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    inject(monkeypatch, algorithm, 'actions')
    env = FailureEnv({1: [(0, FAILURE_CODES['joint_limit'])],
                      2: [(1, FAILURE_CODES['excess_force'])],
                      3: [(1, FAILURE_CODES['object_dropped'])], 4: [(1, 0)]})
    result = recovery().collect(algorithm, env, obs, 3)
    counts = result['failures']
    for name in ('joint_limit', 'invalid_action', 'excess_force', 'object_dropped'):
        assert counts[FAILURE_CODES[name]] == 1
    assert counts.sum() == 4 and counts[0] == 0
    assert result['reward_sum'] == result['terms']['test'] == 12.
    assert algorithm.storage.rewards[:, 0, 0].tolist() == [3., 4., 5.]


def test_failure_counts_survive_multiple_restarts_without_double_counting(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    original, calls = algorithm.act, 0
    def act(obs):
        nonlocal calls
        calls += 1
        action = original(obs)
        if calls in (2, 4): action[0 if calls == 2 else 1, 0] = float('nan')
        return action
    monkeypatch.setattr(algorithm, 'act', act)
    env = FailureEnv({tick: [(0, FAILURE_CODES['joint_limit'])] for tick in (1, 3, 5)})
    result = recovery().collect(algorithm, env, obs, 3)
    assert result['failures'][FAILURE_CODES['joint_limit']] == 3
    assert result['failures'][FAILURE_CODES['invalid_action']] == 2
    assert result['failures'].sum() == 5
    assert result['reward_sum'] == result['terms']['test'] == 18.


def test_failure_counts_include_bad_reward_result_before_quarantine(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    env = FailureEnv({2: [(1, FAILURE_CODES['excess_force'])]}, broken='transient_reward')
    result = recovery().collect(algorithm, env, obs, 3)
    assert result['failures'][FAILURE_CODES['excess_force']] == 1
    assert result['failures'][FAILURE_CODES['invalid_action']] == 1
    assert result['failures'].sum() == 2
    assert result['reward_sum'] == result['terms']['test'] == 15.


def test_failure_counts_restart_at_next_collect_boundary(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    guard = recovery()
    first = guard.collect(algorithm, FailureEnv({1: [(0, FAILURE_CODES['joint_limit'])]}), obs, 3)
    algorithm.storage.clear()
    second = guard.collect(algorithm, Env(), first['obs'], 3)
    assert first['failures'][FAILURE_CODES['joint_limit']] == 1
    assert second['failures'].sum() == 0


@pytest.mark.parametrize('done,code,expected', [
    (True, float('nan'), 0), (True, float('inf'), 0), (True, -1., 0),
    (True, 1.5, 0), (True, 99999., 0), (False, 3., 0), (float('nan'), 3., 0),
    (True, 3., 1),
])
def test_failure_histogram_only_accepts_valid_done_and_failure_codes(monkeypatch, done, code, expected):
    algorithm, obs = setup_algorithm(monkeypatch)
    env = Env()
    step = env.step
    def annotated(action):
        obs, reward, dones, extra = step(action)
        dones = dones.float()
        extra['failure'] = extra['failure'].float()
        if env.tick == 1:
            dones[0] = done
            extra['failure'][0] = code
        return obs, reward, dones, extra
    monkeypatch.setattr(env, 'step', annotated)
    result = recovery().collect(algorithm, env, obs, 3)
    # A malformed done also needs quarantine; its valid invalid_action result
    # counts once, whereas the malformed annotation itself never counts.
    assert result['failures'][FAILURE_CODES['joint_limit']] == expected
    assert result['failures'][FAILURE_CODES['invalid_action']] == int(done != done)


class RecoveredEnv(Env):
    def __init__(self, code, broken=None, repeat=False):
        super().__init__()
        self.code, self.bad_reset, self.repeat = code, broken, repeat

    def step(self, action):
        obs, reward, done, extra = super().step(action)
        if self.tick == 2 or (self.repeat and self.tick == 4):
            done[0] = True
            extra['recovered'][0] = True
            extra['failure'][0] = self.code
            self.resets.append([0])
            if self.bad_reset == 'obs': obs[0, 0] = float('nan')
            if self.bad_reset == 'reward': reward[0] = float('nan')
            if self.bad_reset == 'done': done[0] = False
            if self.bad_reset == 'ack': extra['recovered'][0] = False
        return obs, reward, done, extra


@pytest.mark.parametrize('name', ['invalid_action', 'nonfinite_state', 'nonfinite_joint_state'])
def test_env_completed_invalid_state_reset_discards_without_another_reset(monkeypatch, name):
    algorithm, obs = setup_algorithm(monkeypatch)
    guard, env = recovery(), RecoveredEnv(FAILURE_CODES[name])
    result = guard.collect(algorithm, env, obs, 3)
    assert env.resets == [[0]] and len(env.actions) == 5
    assert all(torch.isfinite(action).all() for action in env.actions)
    assert algorithm.storage.rewards[:, 0, 0].tolist() == [3., 4., 5.]
    assert result['reward_sum'] == result['terms']['test'] == 12.
    assert result['failures'][FAILURE_CODES[name]] == 1
    assert guard.counts['discarded_control_steps'] == 2
    assert guard.counts['recoveries'] == 1


@pytest.mark.parametrize('broken', ['obs', 'reward', 'done', 'ack'])
def test_env_completed_reset_requires_valid_result_without_sending_marker(monkeypatch, broken):
    algorithm, obs = setup_algorithm(monkeypatch)
    guard, env = recovery(), RecoveredEnv(FAILURE_CODES['nonfinite_state'], broken=broken)
    with pytest.raises(RuntimeError, match='reset'):
        guard.collect(algorithm, env, obs, 3)
    assert env.resets == [[0]] and len(env.actions) == 2
    assert algorithm.storage.step == 0 and algorithm.transition.actions is None


def test_independent_env_completed_resets_do_not_consume_policy_retry_budget(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    guard, env = recovery(), RecoveredEnv(FAILURE_CODES['nonfinite_state'], repeat=True)
    result = guard.collect(algorithm, env, obs, 3)
    assert len(env.actions) == 7 and all(torch.isfinite(a).all() for a in env.actions)
    assert env.resets == [[0], [0]]
    assert algorithm.storage.rewards[:, 0, 0].tolist() == [5., 6., 7.]
    assert result['failures'][FAILURE_CODES['nonfinite_state']] == 2
    assert guard.counts['recoveries'] == 2


def test_physical_reset_does_not_spend_a_later_policy_recovery_attempt(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    inject(monkeypatch, algorithm, 'actions', calls=(4,))
    guard, env = recovery(), RecoveredEnv(FAILURE_CODES['nonfinite_state'])
    result = guard.collect(algorithm, env, obs, 3)
    assert env.resets == [[0], [0]]
    assert result['failures'][FAILURE_CODES['nonfinite_state']] == 1
    assert result['failures'][FAILURE_CODES['invalid_action']] == 1
    assert guard.counts['recoveries'] == 2


def test_independent_nonfinite_env_rewards_do_not_spend_policy_retry_budget(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    env = Env()
    step = env.step
    def bad_rewards(action):
        obs, reward, done, extra = step(action)
        if env.tick in (2, 5): reward[0] = float('nan')
        return obs, reward, done, extra
    monkeypatch.setattr(env, 'step', bad_rewards)
    guard = recovery()
    result = guard.collect(algorithm, env, obs, 3)
    assert env.resets == [[0], [0]]
    assert algorithm.storage.rewards[:, 0, 0].tolist() == [7., 8., 9.]
    assert result['failures'][FAILURE_CODES['invalid_action']] == 2
    assert guard.counts['recoveries'] == 2


@pytest.mark.parametrize('packet_field', ['obs', 'reward'])
@pytest.mark.parametrize('row', [0, 1])
def test_persistent_bad_ordinary_packet_with_valid_markers_resets_only_once(monkeypatch, packet_field, row):
    algorithm, obs = setup_algorithm(monkeypatch)
    env = Env()
    step = env.step
    def persistent_packet(action):
        result = step(action)
        if env.tick > 8:
            raise RuntimeError('test guard: unbounded ordinary packets')
        if torch.isfinite(action).all():  # marker replies themselves are valid
            if packet_field == 'obs': result[0][row, 0] = float('nan')
            else: result[1][row] = float('nan')
        return result
    monkeypatch.setattr(env, 'step', persistent_packet)
    guard = recovery()
    with pytest.raises(RuntimeError, match='packet.*recurr'):
        guard.collect(algorithm, env, obs, 3)
    assert env.resets == [[row]] and len(env.actions) == 3
    assert guard.counts['recoveries'] == 1
    assert guard.counts['discarded_control_steps'] == 3
    assert algorithm.storage.step == 0 and algorithm.transition.actions is None
    assert getattr(algorithm.transition, 'actions_log_prob', None) is None
    assert not algorithm.optimizer.state


def test_packet_pending_clears_per_row_only_after_finite_ordinary_result(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    env, ordinary = Env(), 0
    step = env.step
    def independent_packets(action):
        nonlocal ordinary
        result = step(action)
        if torch.isfinite(action).all():
            ordinary += 1
            if ordinary in (1, 2, 3):
                result[1][0 if ordinary in (1, 3) else 1] = float('nan')
        return result
    monkeypatch.setattr(env, 'step', independent_packets)
    guard = recovery()
    result = guard.collect(algorithm, env, obs, 3)
    assert env.resets == [[0], [1], [0]]
    assert guard.counts['recoveries'] == 3
    assert result['failures'][FAILURE_CODES['invalid_action']] == 3
    assert algorithm.storage.rewards[:, 0, 0].tolist() == [7., 8., 9.]


def test_env_still_controls_immediate_physical_recurrence_failure(monkeypatch):
    algorithm, obs = setup_algorithm(monkeypatch)
    env = RecoveredEnv(FAILURE_CODES['nonfinite_state'])
    step = env.step
    def guarded(action):
        if env.tick == 2:
            raise RuntimeError('Env post_reset_or_terminal immediate physical recurrence')
        return step(action)
    monkeypatch.setattr(env, 'step', guarded)
    with pytest.raises(RuntimeError, match='Env post_reset_or_terminal'):
        recovery().collect(algorithm, env, obs, 3)
    assert env.resets == [[0]]
    assert algorithm.storage.step == 0 and algorithm.transition.actions is None


def startup_world(fields):
    from types import SimpleNamespace
    pose = torch.zeros((2, 7), dtype=torch.float64)
    wire = SimpleNamespace(data=SimpleNamespace(root_pose_w=pose.clone()))
    wire.write_root_pose_to_sim = lambda value: setattr(wire.data, 'root_pose_w', value.clone())
    world = SimpleNamespace(n=2, device='cpu', continuous_motion=True,
        wire=wire, wp=pose[:, :3].clone())
    world.measure = lambda: dict(q=torch.zeros(2, 2), **fields)
    world.refresh = lambda **kwargs: setattr(world, 'wp', wire.data.root_pose_w[:, :3].clone())
    def reset(ids):
        wire.data.root_pose_w[ids] = pose[ids]
        world.refresh()
    world.reset = reset
    legacy = SimpleNamespace(refresh=lambda **kwargs: None,
        measure_batch=lambda: dict(q=torch.zeros(2, 2)),
        sim=SimpleNamespace(physics_sim_view=SimpleNamespace(update_articulations_kinematic=lambda: None)))
    return legacy, world


def test_startup_adapter_explicitly_validates_tensor_only_fields():
    legacy, world = startup_world(dict(adapter_fault_bits=torch.zeros(2, dtype=torch.long),
                                      transport_warning=torch.tensor([False, True])))
    result = training.validate_world_adapter(legacy, world, None)
    assert result['selected_reset_isolation'] and result['asymmetric_mapping']
    assert result['tensor_only_startup_fields']['adapter_fault_bits']['nonzero_rows'] == 0
    assert result['tensor_only_startup_fields']['transport_warning']['nonzero_rows'] == 1


@pytest.mark.parametrize('key,value', [
    ('adapter_fault_bits', torch.tensor([float('nan'), 0.])),
    ('adapter_fault_bits', torch.tensor([float('inf'), 0.])),
    ('adapter_fault_bits', torch.tensor([1, 0])),
    ('adapter_fault_bits', torch.tensor([0, 4])),
    ('adapter_fault_bits', torch.tensor([0, 8])),
    ('adapter_fault_bits', torch.zeros(2, 1)),
    ('transport_warning', torch.tensor([float('nan'), 0.])),
    ('transport_warning', torch.tensor([float('inf'), 0.])),
    ('transport_warning', torch.tensor([2., 0.])),
    ('transport_warning', torch.zeros(1)),
])
def test_startup_tensor_only_fields_cannot_hide_invalid_data(key, value):
    legacy, world = startup_world({key: value})
    with pytest.raises(ValueError, match=key):
        training.validate_world_adapter(legacy, world, None)


def test_startup_still_rejects_unknown_tensor_only_fields():
    legacy, world = startup_world(dict(unrecognized=torch.zeros(2)))
    with pytest.raises(KeyError, match='unrecognized'):
        training.validate_world_adapter(legacy, world, None)


def test_continuous_contract_overrides_legacy_recovery_requirements():
    from go2_pvcnn.control.dh116.tabletop_tensor_env import OBJECT_DROP_RESET_CONTRACT
    contract = annotation(dict(object_drop_reset=dict(OBJECT_DROP_RESET_CONTRACT)), True)
    assert contract['bounded_failure_reset']['requires'] == 'invalid_live_row_without_pending_reset_then_validated_isolated_reset'
    assert contract['object_drop_reset']['other_guards'] == 'actual_actuator_limits_and_clearance_preserved_finite_mimic_transport_diagnostic_invalid_state_validated_isolated_reset'
    assert contract['object_drop_reset']['bottom_threshold_m'] == OBJECT_DROP_RESET_CONTRACT['bottom_threshold_m']


def test_continuous_contract_disables_early_active_joint_failure_margin():
    from go2_pvcnn.control.dh116.tabletop_tensor_env import JOINT_LIMIT_RESET_CONTRACT
    contract = annotation({}, True)
    assert contract['joint_limit_reset']['early_active_joint_margin'] is None
    assert 'training_joint_limit' not in contract
    # The ordinary mechanical joint guard and legacy metadata stay intact.
    assert contract['joint_limit_reset']['joint_margin'] == JOINT_LIMIT_RESET_CONTRACT['joint_margin']
    assert JOINT_LIMIT_RESET_CONTRACT['early_active_joint_margin'] == .005


def annotation(contract, enabled):
    function = getattr(training, 'apply_continuous_motion_contract', None)
    assert callable(function), 'shared continuous-mode contract annotation is missing'
    return function(contract, enabled)


def test_common_annotation_preserves_reset_semantics_and_rejects_old_models():
    old = dict(training_joint_limit={'legacy': True}, bounded_failure_reset={'legacy': True})
    current = copy.deepcopy(old)
    annotation(current, True)
    assert current['continuous_motion']['version'] == 'continuous_actuator_limits_v1'
    bounds = current['bounded_failure_reset']
    assert bounds['strict_mimic_rad'] is None
    assert bounds['max_mimic_exclusive_rad'] is None
    assert bounds['max_force_exclusive_n'] is None
    assert bounds['hard_fault'] == 'abort_only_when_reset_invalid_or_immediate_recurrence'
    assert 'training_joint_limit' not in current
    with pytest.raises(ValueError, match='contract mismatch'):
        training.checkpoint_contract_changes(old, current)
    assert training.checkpoint_contract_changes(current, current) == []
    assert annotation(copy.deepcopy(current), True) == current


def test_disabled_contract_is_unchanged():
    old = dict(training_joint_limit={'legacy': True}, continuous_motion={'sentinel': True})
    current = copy.deepcopy(old)
    assert annotation(current, False) == old


def test_common_annotation_records_transport_diagnostics_and_adapter_bits():
    from go2_pvcnn.control.dh116.tabletop_palm_mount import CONTRACT
    current = annotation({}, True)
    mode = current['continuous_motion']
    assert mode.get('adapter_fault_bits') == dict(object_reference=1, normal=2, friction=4, transport=8)
    transport = mode.get('transport', {})
    assert transport.get('tracking_warning_m') == CONTRACT['tracking_guard_m']
    assert transport.get('linear_speed_warning_m_s') == CONTRACT['linear_speed_guard_m_s']
    assert transport.get('finite_exceedance') == 'diagnostic_only'
    assert transport.get('invalid') == 'nonfinite_actual_target_or_velocity_isolated_reset'
    assert transport.get('command_velocity_limit_m_s') == CONTRACT['velocity_limit_m_s']
    assert transport.get('effort_limit_n') == CONTRACT['effort_limit_n']


def test_cli_still_rejects_tensor_startup_only_before_app_launch(monkeypatch, capsys, tmp_path):
    import runpy
    import sys
    import types
    app_module = types.ModuleType('isaaclab.app')
    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument('--device', default='cuda:0')
            parser.add_argument('--headless', action='store_true')
        def __init__(self, args):
            pytest.fail('CLI rejection must happen before Isaac launch')
    app_module.AppLauncher = AppLauncher
    monkeypatch.setitem(sys.modules, 'isaaclab', types.ModuleType('isaaclab'))
    monkeypatch.setitem(sys.modules, 'isaaclab.app', app_module)
    script = Path(__file__).resolve().parents[1]/'scripts/train_dh116_tabletop_gpu_ppo.py'
    output = tmp_path/'unused'
    monkeypatch.setattr(sys, 'argv', [str(script), '--pose-summary', 'unused.json', '--output', str(output),
        '--continuous-motion', '--tensor-pipeline', '--full-hand-contact', '--force-feedback',
        '--training-episode-resets', '--reset-height-jitter-m', '0', '--grasp-controller', 'policy',
        '--palm-transport', 'driven_carriage', '--pressure-relief', '--skip-hold',
        '--grasp-readiness', 'measured_load', '--palm-exploration-axis', 'xyz', '--startup-only', '--headless'])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name='__main__')
    assert error.value.code == 2
    assert '--tensor-pipeline requires rigid full-hand force feedback' in capsys.readouterr().err
    assert not output.exists()


def test_cli_publishes_explicit_flag_and_common_contract_before_configuration():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root/'scripts/train_dh116_tabletop_gpu_ppo.py').read_text(encoding='utf-8-sig'))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    annotations = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'apply_continuous_motion_contract']
    assert annotations, 'configuration must use the shared mode annotation'
    write = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'write_json'
                 and n.args and 'configuration.json' in ast.unparse(n.args[0]))
    assert annotations[0].lineno < write.lineno
    reports = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'dict'
               and any(k.arg == 'continuous_motion' and ast.unparse(k.value) == 'args.continuous_motion' for k in n.keywords)]
    assert reports, 'startup report must explicitly record true/false flag identity'
    # The approved review fix must not broaden --startup-only to tensor mode.
    guards = [n for n in ast.walk(tree) if isinstance(n, ast.If) and 'args.tensor_pipeline and' in ast.unparse(n.test)]
    assert any('args.startup_only' in ast.unparse(n.test) for n in guards)


def test_recovery_counts_can_resume_without_resetting_existing_counts():
    original = recovery()
    original.counts.update(recoveries=3, discarded_control_steps=9)
    resumed = recovery(original.counts)
    assert resumed.counts == original.counts
    resumed.counts['recoveries'] += 1
    assert original.counts['recoveries'] == 3


def test_stopped_monitor_includes_discard_counts_and_preserves_completed_progress(tmp_path):
    import json
    assert 'policy_action_recovery' in inspect.signature(training.publish_stopped_monitor).parameters
    path = tmp_path/'monitor.json'
    path.write_text(json.dumps(dict(successes=7, numerical_resets=2)), encoding='utf-8')
    counts = dict(discarded_control_steps=9, recoveries=3)
    training.publish_stopped_monitor(path, RuntimeError('persistent fault'), torch.zeros(4),
        4, 24, 2, policy_action_recovery=counts)
    report = json.loads(path.read_text())
    assert report['iteration'] == 4 and report['timesteps'] == 24
    assert report['successes'] == 7 and report['numerical_resets'] == 2
    assert report['policy_action_recovery'] == counts


@pytest.mark.parametrize('field', ['obs','reward','done','terms','terminal_observation'])
def test_invalid_packet_immediately_after_quarantine_stops_without_reset_loop(monkeypatch, field):
    algorithm, obs=setup_algorithm(monkeypatch)
    guard, env=recovery(), Env()
    step=env.step
    def broken_packets(action):
        if env.tick>=8:raise RuntimeError('bounded test sentinel: reset loop')
        obs,reward,done,extra=step(action)
        if torch.isfinite(action).all():
            if field=='obs':obs[0,0]=float('nan')
            elif field=='reward':reward[0]=float('nan')
            elif field=='done':
                done=done.float();done[0]=float('nan')
            elif field=='terms':extra['terms']['test'][0]=float('nan')
            else:extra['terminal_observation'][0,0]=float('nan')
        return obs,reward,done,extra
    monkeypatch.setattr(env,'step',broken_packets)
    with pytest.raises(RuntimeError,match='post_reset_invalid_packet'):
        guard.collect(algorithm,env,obs,3)
    assert env.resets==[[0]] and env.tick==3
    assert algorithm.storage.step==0 and algorithm.transition.actions is None
