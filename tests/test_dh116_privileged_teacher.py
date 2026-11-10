"""Pure CPU Phase 1 contracts. These tests provide no native qualification."""
import copy
import importlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

torch.set_num_threads(1)
PREFIX = 'go2_pvcnn.control.dh116.tabletop_privileged_'


def module(name):
    target = PREFIX + name
    assert importlib.util.find_spec(target) is not None, f'Phase 1 module missing: {target}'
    return importlib.import_module(target)


def teacher():
    torch.manual_seed(31)
    return module('teacher').PrivilegedTeacher()


def trainer(steps=3, environments=2, model=None, epochs=2, minibatches=8):
    ppo = module('ppo')
    return ppo.PPOTrainer(teacher() if model is None else model,
        ppo.PPOConfig(epochs=epochs, minibatches=minibatches),
        rollout_steps=steps, num_envs=environments, device='cpu')


def fill(engine, steps=None):
    for tick in range(steps or engine.rollout_steps):
        obs = torch.full((engine.num_envs, 371), tick / 10.)
        sample = engine.collect_step(obs, obs)
        engine.record_step(torch.arange(engine.num_envs).float() + tick,
            torch.zeros(engine.num_envs, dtype=torch.bool),
            torch.zeros(engine.num_envs), executed_actions=sample['actions'])


def test_teacher_topology_distribution_and_separate_critic():
    model = teacher()
    actor = [m for m in model.actor if isinstance(m, nn.Linear)]
    critic = [m for m in model.critic if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in actor] == [(371, 256), (256, 256), (256, 24)]
    assert [(m.in_features, m.out_features) for m in critic] == [(371, 256), (256, 256), (256, 1)]
    assert model.log_std.shape == (24,)
    torch.testing.assert_close(model.log_std.exp(), torch.full((24,), .1))
    obs = torch.randn(5, 371)
    sampled = model.sample(obs)
    assert sampled['actions'].shape == (5, 24)
    assert torch.equal(sampled['actions'], sampled['latent_actions'].tanh())
    assert sampled['log_prob'].shape == (5,)
    torch.testing.assert_close(model.deterministic(obs), model.latent_mean(obs).tanh())
    evaluated = model.evaluate_latents(obs, obs, sampled['latent_actions'])
    torch.testing.assert_close(sampled['log_prob'], evaluated['log_prob'])
    assert evaluated['value'].shape == (5,) and evaluated['entropy'].shape == (5,)
    assert not (set(map(id, model.actor.parameters())) & set(map(id, model.critic.parameters())))


@pytest.mark.parametrize('dimension', [261, 351, 3844])
def test_teacher_rejects_legacy_or_visual_observation(dimension):
    with pytest.raises(ValueError, match='371'):
        teacher().sample(torch.zeros(2, dimension))


def test_ppo_baseline_config_is_explicit_and_validated():
    config = module('ppo').PPOConfig()
    assert (config.gamma, config.lam, config.clip, config.lr) == (.995, .95, .2, 3e-4)
    assert (config.epochs, config.minibatches, config.entropy_coef, config.value_coef, config.max_grad_norm) == (5, 8, .005, .5, .5)
    for kwargs in ({'epochs': 0}, {'lr': float('nan')}, {'gamma': 1.2}, {'minibatches': 0}):
        with pytest.raises(ValueError):
            module('ppo').PPOConfig(**kwargs)


def test_pending_tokens_prevent_external_or_changed_issued_action():
    engine = trainer(steps=1)
    with pytest.raises(RuntimeError, match='pending'):
        engine.record_step(torch.ones(2), torch.zeros(2, dtype=torch.bool), torch.zeros(2))
    sample = engine.collect_step(torch.zeros(2, 371), torch.zeros(2, 371))
    with pytest.raises(RuntimeError, match='pending'):
        engine.collect_step(torch.zeros(2, 371), torch.zeros(2, 371))
    with pytest.raises(ValueError, match='owner'):
        engine.record_step(torch.ones(2), torch.zeros(2, dtype=torch.bool), torch.zeros(2), action_owner='mpc')
    with pytest.raises(ValueError, match='issued|sample'):
        engine.record_step(torch.ones(2), torch.zeros(2, dtype=torch.bool), torch.zeros(2),
            executed_actions=sample['actions'] + .01)
    # Mutating returned data never mutates the private sampled pending record.
    issued = sample['actions'].clone()
    sample['latent_actions'].fill_(999.)
    sample['log_prob'].fill_(999.)
    engine.record_step(torch.ones(2), torch.zeros(2, dtype=torch.bool), torch.zeros(2), executed_actions=issued)
    assert engine.update()['samples'] == 2


def test_storage_is_detached_cpu_float_and_complete_batch_is_used():
    engine = trainer(steps=7, environments=1)
    before = {name: value.clone() for name, value in engine.model.state_dict().items()}
    fill(engine)
    batch = engine.rollout.batch()
    for name in ('critic_obs', 'actions', 'latent_actions', 'old_log_prob', 'old_mu', 'old_std', 'values', 'rewards'):
        value = batch[name]
        assert value.device.type == 'cpu' and value.dtype == torch.float32 and not value.requires_grad
    assert torch.equal(batch['actions'], batch['latent_actions'].tanh())
    result = engine.update()
    assert result['samples'] == 7 and result['optimized_samples'] == 14
    assert result['minibatch_updates'] == 14  # min(8,7) singleton batches, two epochs.
    assert result['policy_version'] == 1 and len(engine.rollout) == 0
    assert all(math.isfinite(result[name]) for name in ('policy_loss', 'value_loss', 'entropy', 'kl'))
    assert any(not torch.equal(before[name], value) for name, value in engine.model.state_dict().items())


def test_frozen_likelihood_detects_parameter_change_before_update():
    engine = trainer(steps=1)
    fill(engine)
    with torch.no_grad():
        engine.model.actor[-1].bias.add_(.1)
    with pytest.raises(RuntimeError, match='policy|likelihood|on.policy'):
        engine.update()


def test_incomplete_rollout_and_pending_update_are_rejected():
    engine = trainer(steps=2)
    with pytest.raises(RuntimeError, match='complete|empty'):
        engine.update()
    engine.collect_step(torch.zeros(2, 371), torch.zeros(2, 371))
    with pytest.raises(RuntimeError, match='pending'):
        engine.update()


def test_terminal_and_truncated_bootstrap_are_explicit_and_gae_does_not_bridge_reset():
    engine = trainer(steps=1)
    engine.collect_step(torch.zeros(2, 371), torch.zeros(2, 371))
    with pytest.raises(ValueError, match='bootstrap'):
        engine.record_step(torch.tensor([2., 3.]), torch.tensor([True, True]), torch.tensor([9., 9.]),
            bootstrap_mask=torch.tensor([True, True]))
    engine.record_step(torch.tensor([2., 3.]), torch.tensor([True, True]), torch.tensor([9., 9.]),
        truncated=torch.tensor([False, True]), bootstrap_mask=torch.tensor([False, True]))
    batch = engine.rollout.batch()
    returns, advantages = engine.rollout.returns_and_advantages(engine.config)
    torch.testing.assert_close(returns, torch.tensor([2., 3. + .995 * 9.]))
    assert not advantages.requires_grad
    assert batch['done'].tolist() == [True, True]


class HistoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(12, 24)
        self.critic = nn.Linear(371, 1)
        self.log_std = nn.Parameter(torch.full((24,), math.log(.1)))

    def latent_mean(self, obs):
        return self.actor(obs['history'].reshape(len(obs['history']), -1))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def test_generic_dictionary_actor_and_privileged_critic_are_kept_separate():
    engine = trainer(steps=2, model=HistoryModel(), minibatches=3)
    for _ in range(2):
        engine.collect_step({'history': torch.ones(2, 3, 4), 'valid': torch.ones(2, 3, dtype=torch.bool),
            'frame_ids': torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.long)}, torch.ones(2, 371))
        engine.record_step(torch.ones(2), torch.zeros(2, dtype=torch.bool), torch.zeros(2))
    batch = engine.rollout.batch()
    assert set(batch['actor_obs']) == {'history', 'valid', 'frame_ids'}
    assert batch['actor_obs']['history'].shape == (4, 3, 4)
    assert batch['actor_obs']['valid'].dtype == torch.bool
    assert batch['actor_obs']['frame_ids'].dtype == torch.long
    assert batch['critic_obs'].shape == (4, 371)
    assert engine.update()['optimized_samples'] == 8


def test_noise_projection_clears_only_changed_adam_coordinates():
    engine = trainer(steps=1)
    fill(engine)
    engine.update()
    state = engine.optimizer.state[engine.model.log_std]
    with torch.no_grad():
        engine.model.log_std[0] = math.log(4.)
        engine.model.log_std[1] = math.log(.00001)
    state['exp_avg'].fill_(1.)
    state['exp_avg_sq'].fill_(1.)
    engine.project_noise()
    torch.testing.assert_close(engine.model.log_std[:2].exp(), torch.tensor([.3, .001]))
    assert state['exp_avg'][:2].eq(0).all() and state['exp_avg_sq'][:2].eq(0).all()
    assert state['exp_avg'][2:].eq(1).all()


def test_actor_only_bc_has_independent_optimizer_and_refuses_dirty_ppo():
    engine = trainer(steps=1)
    data = module('dataset')
    bc = data.ActorBCTrainer(engine.model, ppo_trainer=engine)
    critic_before = copy.deepcopy(engine.model.critic.state_dict())
    actor_before = copy.deepcopy(engine.model.actor.state_dict())
    observations, targets = torch.ones(2, 371), torch.full((2, 24), .2)
    # Arithmetic training API is explicit; production datasets must separately
    # pass validate_expert_dataset and retain native qualification provenance.
    bc.step(observations, targets, torch.ones(2, 24, dtype=torch.bool), schema_only=True)
    assert not engine.optimizer.state
    assert all(torch.equal(critic_before[k], v) for k, v in engine.model.critic.state_dict().items())
    assert any(not torch.equal(actor_before[k], v) for k, v in engine.model.actor.state_dict().items())
    engine.collect_step(observations, observations)
    with pytest.raises(RuntimeError, match='PPO|pending|rollout'):
        bc.step(observations, targets, torch.ones(2, 24, dtype=torch.bool), schema_only=True)


@pytest.mark.parametrize('bad', ['schema_only', 'partial', 'later_episode', 'recovered', 'restored', 'missing_proof'])
def test_expert_success_admission_rejects_ineligible_evidence(bad):
    dataset = module('dataset')
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
    identity = dict(state_contract_sha256='a' * 64,
        action_contract_sha256=module('checkpoint').canonical_sha256(ACTION_CONTRACT))
    payload = dict(version=dataset.DATASET_VERSION, observations=torch.zeros(2, 371),
        actions=torch.zeros(2, 24), valid_mask=torch.ones(2, 24, dtype=torch.bool),
        identity=identity, schema_only=False,
        episodes=[dict(episode_id='one', start=0, end=2, complete=True, first_episode=True,
            recovered=False, restored=False, native_success=True,
            source_sha256='c' * 64, physical_config_sha256='d' * 64)])
    if bad == 'schema_only': payload['schema_only'] = True
    elif bad == 'partial': payload['episodes'][0]['complete'] = False
    elif bad == 'later_episode': payload['episodes'][0]['first_episode'] = False
    elif bad == 'recovered': payload['episodes'][0]['recovered'] = True
    elif bad == 'restored': payload['episodes'][0]['restored'] = True
    with pytest.raises(ValueError):
        dataset.validate_expert_dataset(payload, expected_identity=identity)


def test_native_r0_is_passed_through_once_and_trace_closes():
    reward = module('reward')
    native = torch.tensor([2., -4.])
    result, trace = reward.compose_native_reward(native, {'one': torch.tensor([3., -1.]), 'two': torch.tensor([-1., -3.])})
    assert torch.equal(result, native)
    assert torch.equal(trace['total_contribution'], native)
    assert trace['profile'] == 'teacher_r0_existing'
    assert set(trace['terms']) == {'one', 'two'}
    with pytest.raises(ValueError, match='sum|close|total'):
        reward.compose_native_reward(native, {'incomplete': torch.zeros(2)})


@pytest.mark.parametrize('profile', ['teacher_r1_geometry', 'teacher_r2_support_recovery'])
def test_optional_reward_requires_qualified_provenance_and_rejects_na(profile):
    reward = module('reward')
    with pytest.raises(ValueError, match='qualified|qualification|provenance'):
        reward.compose_native_reward(torch.zeros(2), {'native': torch.zeros(2)}, profile=profile,
            shaping={'geometry_potential_before': torch.zeros(2), 'geometry_potential_after': torch.ones(2)})


def test_checkpoint_exclusive_identity_optimizer_rng_and_history_reset(tmp_path):
    checkpoints = module('checkpoint')
    engine = trainer(steps=1)
    fill(engine)
    engine.update()
    identity = checkpoints.build_identity(engine.config, state_contract_sha256='a' * 64,
        source_sha256={'audited_source.py': 'b' * 64})
    path = tmp_path / 'teacher.pt'
    checkpoints.save_checkpoint(path, engine, identity, history_reset='clear')
    expected_random = torch.rand(5)
    with pytest.raises(FileExistsError):
        checkpoints.save_checkpoint(path, engine, identity, history_reset='clear')
    resumed = trainer(steps=1)
    restored = checkpoints.load_checkpoint(path, resumed, identity, history_reset='clear')
    assert restored['history_reset'] == 'clear'
    assert resumed.policy_version == engine.policy_version
    assert resumed.counters == engine.counters
    assert resumed.optimizer.state_dict()['param_groups'] == engine.optimizer.state_dict()['param_groups']
    assert resumed.optimizer.state_dict()['state']
    assert all(torch.equal(engine.model.state_dict()[k], v) for k, v in resumed.model.state_dict().items())
    torch.testing.assert_close(torch.rand(5), expected_random)
    mismatch = dict(identity, state_contract_sha256='c' * 64)
    with pytest.raises(ValueError, match='identity|contract'):
        checkpoints.load_checkpoint(path, resumed, mismatch, history_reset='clear')
    legacy = tmp_path / 'visual.pt'
    torch.save({'version': 'dh116-vision-ppo', 'policy': {}}, legacy)
    with pytest.raises(ValueError, match='version|checkpoint'):
        checkpoints.load_checkpoint(legacy, resumed, identity, history_reset='clear')


def test_checkpoint_refuses_pending_rollout_and_implicit_history_policy(tmp_path):
    checkpoints = module('checkpoint')
    engine = trainer(steps=1)
    identity = checkpoints.build_identity(engine.config, state_contract_sha256='a' * 64, source_sha256={})
    engine.collect_step(torch.zeros(2, 371), torch.zeros(2, 371))
    with pytest.raises(RuntimeError, match='pending|rollout|boundary'):
        checkpoints.save_checkpoint(tmp_path / 'dirty.pt', engine, identity, history_reset='clear')


def test_saturated_tanh_uses_saved_latents_without_arctanh_reconstruction():
    model = teacher()
    with torch.no_grad():
        model.actor[-1].weight.zero_()
        model.actor[-1].bias.fill_(30.)
    obs = torch.zeros(1, 371)
    sample = model.sample(obs)
    assert sample['actions'].eq(1.).all()
    assert torch.isfinite(sample['log_prob']).all()
    evaluated = model.evaluate_latents(obs, obs, sample['latent_actions'])
    torch.testing.assert_close(evaluated['log_prob'], sample['log_prob'])


def test_invalid_reward_does_not_consume_pending_and_input_storage_is_owned():
    engine = trainer(steps=1)
    observations = torch.ones(2, 371)
    engine.collect_step(observations, observations)
    observations.fill_(99.)
    with pytest.raises(ValueError, match='finite'):
        engine.record_step(torch.tensor([float('nan'), 1.]), torch.zeros(2, dtype=torch.bool), torch.zeros(2))
    engine.record_step(torch.ones(2), torch.zeros(2, dtype=torch.bool), torch.zeros(2))
    batch = engine.rollout.batch()
    assert batch['actor_obs'].eq(1.).all() and batch['critic_obs'].eq(1.).all()


def test_group_bc_normalization_does_not_weight_twelve_joints_twice():
    data = module('dataset')
    predictions = torch.zeros(2, 24, requires_grad=True)
    targets = torch.cat((torch.ones(2, 12), torch.full((2, 6), 2.), torch.full((2, 6), 3.)), -1)
    loss = data.masked_group_normalized_loss(predictions, targets, torch.ones(2, 24, dtype=torch.bool))
    assert loss.item() == pytest.approx((1. + 4. + 9.) / 3.)
    loss.backward()
    assert predictions.grad is not None and torch.isfinite(predictions.grad).all()
    mask = torch.zeros(2, 24, dtype=torch.bool)
    assert data.masked_group_normalized_loss(predictions, targets, mask).item() == 0.


@pytest.mark.parametrize('corruption', ['adam_shape', 'adam_missing', 'rng', 'noise'])
def test_corrupted_checkpoint_rejected_before_live_model_or_rng_changes(tmp_path, corruption):
    checkpoints = module('checkpoint')
    engine = trainer(steps=1)
    fill(engine)
    engine.update()
    identity = checkpoints.build_identity(engine.config, state_contract_sha256='a' * 64, source_sha256={})
    clean = tmp_path / 'clean.pt'
    checkpoints.save_checkpoint(clean, engine, identity, history_reset='clear')
    payload = torch.load(clean, map_location='cpu', weights_only=False)
    first = next(iter(payload['optimizer']['state']))
    if corruption == 'adam_shape': payload['optimizer']['state'][first]['exp_avg'] = torch.zeros(9)
    elif corruption == 'adam_missing': payload['optimizer']['state'][first].pop('exp_avg_sq')
    elif corruption == 'rng': payload['rng']['torch'] = torch.zeros(1)
    elif corruption == 'noise': payload['policy']['log_std'].fill_(10.)
    broken = tmp_path / 'broken.pt'
    torch.save(payload, broken)
    recipient = trainer(steps=1)
    before = copy.deepcopy(recipient.model.state_dict())
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        checkpoints.load_checkpoint(broken, recipient, identity, history_reset='clear')
    assert all(torch.equal(before[k], v) for k, v in recipient.model.state_dict().items())
    assert torch.equal(rng_before, torch.get_rng_state())


def test_configs_keep_reward_qualification_opt_in_and_actual_ppo_values():
    root = Path(__file__).resolve().parents[1]
    config_root = root / 'configs/dh116_training'
    payload = json.loads((config_root / 'privileged_teacher_ppo_v1.json').read_text())
    config = module('ppo').PPOConfig(**payload['ppo'])
    assert config == module('ppo').PPOConfig()
    assert payload['reward_profile'] == 'teacher_r0_existing'
    assert payload['bc']['auxiliary_weight'] == 0.
    for index in range(3):
        reward = json.loads((config_root / f'privileged_teacher_reward_r{index}.json').read_text())
        assert reward['enabled'] is (index == 0)


def test_resume_reproduces_next_on_policy_update_including_adam_and_rng(tmp_path):
    checkpoints = module('checkpoint')
    original = trainer(steps=2)
    fill(original)
    original.update()
    identity = checkpoints.build_identity(original.config, state_contract_sha256='a' * 64, source_sha256={})
    path = tmp_path / 'resume.pt'
    checkpoints.save_checkpoint(path, original, identity, history_reset='clear')
    fill(original)
    expected = original.update()
    resumed = trainer(steps=2)
    checkpoints.load_checkpoint(path, resumed, identity, history_reset='clear')
    fill(resumed)
    actual = resumed.update()
    assert actual == expected
    assert original.counters == resumed.counters
    assert all(torch.equal(v, resumed.model.state_dict()[k]) for k, v in original.model.state_dict().items())


def test_self_declared_native_json_and_arbitrary_source_cannot_admit_success_pool(tmp_path):
    import hashlib
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
    dataset = module('dataset')
    identity = dict(state_contract_sha256='a' * 64,
        action_contract_sha256=module('checkpoint').canonical_sha256(ACTION_CONTRACT))
    source = tmp_path / 'not_native.bin'
    source.write_bytes(b'explicitly synthetic source, not a native episode')
    episode = dict(episode_id='synthetic', start=0, end=2, complete=True, first_episode=True,
        recovered=False, restored=False, native_success=True, source_path=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), physical_config_sha256='d' * 64)
    proof = dict(episode, identity=identity, evidence_kind='native_isaac_physx', schema_only=False,
        success_metrics=dict(net_lift_m=.03, continuous_hold_s=5., formal_anchor_slip_m=0., native_task_success=True))
    path = tmp_path / 'self_declared.json'
    path.write_text(json.dumps(proof))
    episode.update(proof_path=str(path), proof_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    payload = dict(version=dataset.DATASET_VERSION, observations=torch.zeros(2, 371), actions=torch.zeros(2, 24),
        valid_mask=torch.ones(2, 24, dtype=torch.bool), identity=identity, schema_only=False, episodes=[episode])
    with pytest.raises(ValueError, match='qualification|raw|native episode'):
        dataset.validate_expert_dataset(payload, expected_identity=identity)
