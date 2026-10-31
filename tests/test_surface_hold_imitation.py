"""Regression for persistent velocity bias in cloned native hold commands."""
import copy
import math

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_surface_imitation as bc


def data():
    g = torch.Generator().manual_seed(81)
    obs = torch.randn(64, 351, generator=g) * .1
    actions = torch.zeros(64, 24)
    actions[32:, [14, 17]] = .3
    return dict(version=bc.DATASET_VERSION,
        contract=dict(observation_size=351, action_size=24, policy_command_size=24,
            observation_version=bc.OBSERVATION_VERSION, num_envs=2, physics_hz=480),
        observations=obs, actions=actions, episode_ids=torch.zeros(64, dtype=torch.long),
        qualification=dict(qualified=True, successful_replays=1, proof_path='native/proof.json'))


def test_zero_extra_weights_preserve_existing_bc_objective():
    target = torch.zeros(3, 24)
    pred = torch.arange(72, dtype=torch.float32).reshape(3, 24) / 100
    actual = bc.action_fit_loss(pred, target)
    assert torch.equal(actual, (pred - target).square().mean())


def test_holding_velocity_bias_receives_stronger_corrective_gradient():
    target = torch.zeros(2, 24)
    target[1, 14] = .3
    pred = (target + .02).requires_grad_()
    loss = bc.action_fit_loss(pred, target, holding_weight=20., wrist_weight=4.)
    loss.backward()
    assert pred.grad[0, 14] > pred.grad[1, 14] * 10
    assert pred.grad[0, 14] > pred.grad[0, 0] * 3
    assert pred.grad[0, 14] > 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize('kwargs', [dict(holding_weight=-1), dict(wrist_weight=0),
                                  dict(holding_weight=float('nan')),
                                  dict(wrist_weight=float('inf'))])
def test_invalid_objective_is_rejected(kwargs):
    with pytest.raises(ValueError):
        bc.action_fit_loss(torch.zeros(2, 24), torch.zeros(2, 24), **kwargs)


def test_warm_start_retains_real_initial_actor_and_records_objective(tmp_path):
    sample = data()
    actor = bc.make_actor()
    with torch.no_grad():
        for p in actor.parameters():
            p.zero_()
        actor[-2].bias.fill_(math.atanh(.02))
    source = tmp_path / 'source.pt'
    torch.save(dict(version=bc.ACTOR_VERSION, contract=copy.deepcopy(sample['contract']),
                    actor=actor.state_dict()), source)
    before = source.read_bytes()
    expected = float((actor(sample['observations']) - sample['actions']).square().mean())
    path = tmp_path / 'fine_tuned.pt'
    metrics = bc.pretrain(sample, path, steps=2, learning_rate=1e-4,
                          warm_start=source, holding_weight=20., wrist_weight=4.)
    assert metrics['initial_mse'] == pytest.approx(expected)
    assert metrics['warm_start']['path'] == str(source.resolve())
    assert metrics['holding_weight'] == 20.
    assert metrics['wrist_weight'] == 4.
    assert source.read_bytes() == before
    assert path.is_file()


def test_warm_start_physics_mismatch_does_not_create_output(tmp_path):
    sample = data()
    source = tmp_path / 'bad.pt'
    torch.save(dict(version=bc.ACTOR_VERSION,
                    contract=dict(sample['contract'], physics_hz=240),
                    actor=bc.make_actor().state_dict()), source)
    output = tmp_path / 'must_not_exist.pt'
    with pytest.raises(ValueError, match='contract'):
        bc.pretrain(sample, output, steps=2, warm_start=source)
    assert not output.exists()


def test_independent_teacher_step_ignores_opposing_ppo_momentum():
    from types import SimpleNamespace
    sample = data()
    sample['actions'].zero_()
    actor = bc.make_actor()
    with torch.no_grad():
        for p in actor.parameters():
            p.zero_()
        actor[-2].bias.fill_(math.atanh(.02))
    policy = SimpleNamespace(actor=actor)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    # Seed PPO history with a gradient that pushes hold velocity upward.
    for p in actor.parameters():
        p.grad = -torch.ones_like(p)
    optimizer.step()
    policy_before = copy.deepcopy(optimizer.state_dict())
    algorithm = SimpleNamespace(policy=policy, optimizer=optimizer)
    regularizer = bc.DemonstrationRegularizer(sample, sample['contract'], 'cpu',
        independent_optimizer=True, holding_weight=20., wrist_weight=4.)
    with torch.no_grad():
        before = bc.action_fit_loss(actor(sample['observations']), sample['actions'],
                                    holding_weight=20., wrist_weight=4.).item()
    result = regularizer.update(algorithm, 0, initial_weight=.1, decay=1., batch_size=64)
    with torch.no_grad():
        after = bc.action_fit_loss(actor(sample['observations']), sample['actions'],
                                   holding_weight=20., wrist_weight=4.).item()
    assert after < before
    assert result['optimizer'] == 'independent_actor_adam'
    assert result['learning_rate'] == pytest.approx(1e-4)
    for key, state in policy_before['state'].items():
        for field, old in state.items():
            value = optimizer.state_dict()['state'][key][field]
            assert torch.equal(old, value) if isinstance(old, torch.Tensor) else old == value
    assert algorithm.surface_imitation_optimizer is not optimizer


def test_independent_teacher_optimizer_state_round_trips():
    from types import SimpleNamespace
    sample = data()
    actor = bc.make_actor()
    algorithm = SimpleNamespace(policy=SimpleNamespace(actor=actor),
                                optimizer=torch.optim.Adam(actor.parameters()))
    regularizer = bc.DemonstrationRegularizer(sample, sample['contract'], 'cpu', independent_optimizer=True)
    regularizer.update(algorithm, 0)
    state = copy.deepcopy(algorithm.surface_imitation_optimizer.state_dict())
    restored = bc.DemonstrationRegularizer(sample, sample['contract'], 'cpu', independent_optimizer=True)
    restored.optimizer_for(algorithm).load_state_dict(state)
    assert len(restored.optimizer_for(algorithm).state_dict()['state']) == len(state['state'])


def test_tensor_checkpoint_keeps_independent_supervision_optimizer(tmp_path):
    from types import SimpleNamespace
    from tensordict import TensorDict
    from go2_pvcnn.control.dh116.tabletop_tensor_training import make_algorithm, save_checkpoint
    sample = data()
    obs = TensorDict({'policy': torch.zeros(2, 351)}, batch_size=[2], device='cpu')
    algorithm = make_algorithm(obs, steps=2, action_size=24)
    regularizer = bc.DemonstrationRegularizer(sample, sample['contract'], 'cpu', independent_optimizer=True)
    regularizer.update(algorithm, 0)
    algorithm.surface_imitation_metadata = dict(update='independent_actor_adam_stable_hold_v2')
    env = SimpleNamespace(**{name: torch.zeros(2, dtype=torch.long) for name in
                             ('episodes', 'successes', 'recovered', 'reached_counts', 'transition_counts')})
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, algorithm, sample['contract'], 17, 1234, env)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['iteration'] == 17
    assert saved['surface_imitation'] == algorithm.surface_imitation_metadata
    expected = algorithm.surface_imitation_optimizer.state_dict()
    for key, state in expected['state'].items():
        for field, old in state.items():
            value = saved['surface_imitation_optimizer']['state'][key][field]
            assert torch.equal(old, value) if isinstance(old, torch.Tensor) else old == value


def test_approved_exploration_contract_change_does_not_invalidate_identical_teacher_physics(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_policy_noise import CONTRACT
    sample = data()
    sample['contract']['policy_noise'] = dict(CONTRACT)
    current = copy.deepcopy(sample['contract'])
    current['policy_noise'] = dict(CONTRACT, version='bounded_surface_policy_noise_v2',
                                   min_std=.001, max_std=.3, initial_std=.005)
    bc.validate_dataset(sample, current)
    current['policy_noise']['action_log_probability'] = 'clipped_sample'
    with pytest.raises(ValueError, match='contract'):
        bc.validate_dataset(sample, current)


def test_actor_loading_honors_explicit_surface_noise_configuration(tmp_path):
    from tensordict import TensorDict
    from go2_pvcnn.control.dh116.tabletop_tensor_training import make_algorithm
    from go2_pvcnn.control.dh116.tabletop_policy_noise import configure_surface_noise
    sample = data()
    path = tmp_path / 'actor.pt'
    bc.pretrain(sample, path, steps=1)
    obs = TensorDict({'policy': torch.zeros(2, 351)}, batch_size=[2], device='cpu')
    algorithm = make_algorithm(obs, steps=2, action_size=24)
    configure_surface_noise(algorithm)
    bc.load_actor(algorithm, path, sample['contract'])
    torch.testing.assert_close(algorithm.policy.log_std.exp(), torch.full((24,), .005))


def test_saved_supervision_resumes_with_the_same_next_update(tmp_path):
    from types import SimpleNamespace
    from tensordict import TensorDict
    from go2_pvcnn.control.dh116.tabletop_tensor_training import make_algorithm, save_checkpoint
    sample = data()
    obs = TensorDict({'policy': torch.zeros(2, 351)}, batch_size=[2], device='cpu')
    one = make_algorithm(obs, steps=2, action_size=24)
    sum(p.square().sum() for p in one.policy.parameters()).backward()
    one.optimizer.step()
    regularizer = bc.DemonstrationRegularizer(sample, sample['contract'], 'cpu',
        independent_optimizer=True, holding_weight=20., wrist_weight=4.)
    regularizer.update(one, 0)
    env = SimpleNamespace(**{name: torch.zeros(2, dtype=torch.long) for name in
        ('episodes','successes','recovered','reached_counts','transition_counts')})
    path = tmp_path / 'round_trip.pt'
    save_checkpoint(path, one, sample['contract'], 17, 1234, env)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    two = make_algorithm(obs, steps=2, action_size=24)
    two.policy.load_state_dict(saved['policy'])
    two.optimizer.load_state_dict(saved['optimizer'])
    restored = bc.DemonstrationRegularizer(sample, sample['contract'], 'cpu',
        independent_optimizer=True, holding_weight=20., wrist_weight=4.)
    restored.optimizer_for(two).load_state_dict(saved['surface_imitation_optimizer'])
    rng = torch.get_rng_state()
    regularizer.update(one, 18)
    torch.set_rng_state(rng)
    restored.update(two, 18)

    def same(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:same(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):same(x, y)
        else:
            assert a == b
    same(one.policy.state_dict(), two.policy.state_dict())
    same(one.optimizer.state_dict(), two.optimizer.state_dict())
    same(one.surface_imitation_optimizer.state_dict(), two.surface_imitation_optimizer.state_dict())
