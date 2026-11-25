"""Device PPO algorithms and complete checkpoint continuation contracts."""
import ast
from copy import deepcopy
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from test_core_teacher import teacher_data
from test_core_teacher import seed
from test_core_teacher import digest_tree
from test_core_teacher import rng
from test_core_teacher import collect
from test_core_teacher import parameter_digest
from test_core_teacher import BASELINE
from test_core_teacher import cpu_threads
from dh116.rl.device import DeviceDeformablePPOTrainer as DeviceDeformablePPOTrainer
from dh116.rl.ppo import PPOConfig
from dh116.rl.ppo import PPOTrainer
from dh116.env.contracts import runti_io__ROOT as ROOT

def test_device_trainer_directly_uses_frozen_ppo_without_legacy_checkpoint_imports():
    assert DeviceDeformablePPOTrainer.__bases__ == (PPOTrainer,)
    for name in ('rl/device.py',):
        path = ROOT / 'src/dh116' / name
        imports = [n.module or '' for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.ImportFrom)]
        assert not [m for m in imports if any((m.endswith('tabletop_' + old) for old in ('deformable_privileged_ppo', 'deformable_privileged_checkpoint', 'native_teacher_profile', 'privileged_checkpoint')))]

@pytest.mark.parametrize('guarded', [False, True])
def test_two_actual_updates_match_original_all_parameters_adam_gae_metrics_rng(monkeypatch, guarded):
    (model, obs, *_) = teacher_data(monkeypatch)
    trainer = DeviceDeformablePPOTrainer(model, PPOConfig(epochs=2, minibatches=2, lr=0.0001, target_kl=0.02 if guarded else None, kl_backtrack_steps=2 if guarded else 0), rollout_steps=3, num_envs=2)
    for (update, expected) in enumerate(BASELINE['updates'][str(guarded)]):
        seed(199 + update)
        collect(trainer, obs)
        assert digest_tree(trainer.rollout.batch()) == expected['batch']
        assert digest_tree(trainer.rollout.returns_and_advantages(trainer.config)) == expected['gae']
        seed(800 + update)
        metrics = trainer.update()
        assert parameter_digest(trainer.model) == expected['parameters']
        assert digest_tree(trainer.optimizer.state_dict()) == expected['adam']
        assert digest_tree(metrics) == expected['metrics']
        assert digest_tree(rng()) == expected['rng']
        assert trainer.counters == expected['counters'] and trainer.policy_version == update + 1
        assert trainer.clean_boundary

def trainer_for(model, *, fast_update=False):
    return DeviceDeformablePPOTrainer(deepcopy(model), PPOConfig(lr=0.0001, epochs=1, minibatches=1, target_kl=0.02, kl_backtrack_steps=3), rollout_steps=3, num_envs=2, fast_update=fast_update)

def test_fast_update_rejects_oversized_adam_step_and_reports_accepted_kl(monkeypatch):
    model, obs, *_ = teacher_data(monkeypatch)
    config = PPOConfig(lr=0.001, epochs=2, minibatches=2, target_kl=0.001, kl_backtrack_steps=3)
    trainer = DeviceDeformablePPOTrainer(model, config, rollout_steps=3, num_envs=2, fast_update=True)
    seed(35)
    collect(trainer, obs)
    batch = trainer.rollout.batch()
    metrics = trainer.update()

    actual_kl = trainer._minibatch_exact_kl(batch, torch.arange(len(batch['values']))).item()
    assert actual_kl <= config.target_kl
    assert metrics['kl'] == pytest.approx(actual_kl)
    assert metrics['kl_backtracks'] > 0
    assert 0 < metrics['minibatch_updates'] <= 4
    assert metrics['optimized_samples'] == 3 * metrics['minibatch_updates']
    assert {int(state['step']) for state in trainer.optimizer.state.values()} == {metrics['minibatch_updates']}
    assert trainer.policy_version == trainer.counters['updates'] == 1
    assert trainer.clean_boundary


def test_fast_update_discards_nonfinite_gradient_without_stepping_adam(monkeypatch):
    model, obs, *_ = teacher_data(monkeypatch)
    trainer = trainer_for(model, fast_update=True)
    collect(trainer, obs)
    before = parameter_digest(trainer.model)
    with trainer.model.actor_head.weight.register_hook(lambda grad: torch.full_like(grad, torch.nan)):
        metrics = trainer.update()

    assert metrics['early_stopped'] and metrics['stop_reason'] == 'nonfinite_gradient'
    assert metrics['minibatch_updates'] == metrics['optimized_samples'] == 0
    assert parameter_digest(trainer.model) == before
    assert not trainer.optimizer.state
    assert trainer.policy_version == trainer.counters['updates'] == 0
    assert trainer.clean_boundary

def test_exact_kl_audits_full_4096_in_one_actor_forward(monkeypatch):
    from dh116.rl.ppo import latent_gaussian_kl as latent_gaussian_kl
    (model, obs, *_) = teacher_data(monkeypatch)
    trainer = trainer_for(model)
    full = {k: v.repeat((2048,) + (1,) * (v.ndim - 1)) for (k, v) in obs.items()}
    with torch.no_grad():
        mu = model.latent_mean(full)
        std = model.log_std.exp().expand_as(mu).clone()
    batch = dict(actor_obs=full, old_mu=mu + 0.001, old_std=std, values=torch.zeros(4096))
    calls = []
    handle = trainer.model.actor.register_forward_pre_hook(lambda module, args: calls.append(len(args[0])))
    trainer.max_visual_batch = 7
    try:
        actual = trainer._minibatch_exact_kl(batch, torch.arange(4096))
    finally:
        handle.remove()
    expected = latent_gaussian_kl(batch['old_mu'], std, mu, std).clamp_min(0.0).sum().double() / 4096
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert calls == [4096]

def test_gae_terminal_blocks_bootstrap_truncation_keeps_value_and_both_stop_carry(monkeypatch):
    from dh116.rl.device import DeviceOnPolicyRollout as DeviceOnPolicyRollout
    capability = object()
    rollout = DeviceOnPolicyRollout(2, 2, session_id='s', policy_version=0, capability=capability)
    for t in range(2):
        row = {name: torch.zeros(2, 24) for name in ('actor_obs', 'critic_obs', 'actions', 'latent_actions', 'old_mu', 'old_std')}
        row.update(old_log_prob=torch.zeros(2), values=torch.tensor([0.2, 0.3]) if t == 0 else torch.tensor([0.4, 0.5]), rewards=torch.tensor([1.0, 2.0]) if t == 0 else torch.tensor([3.0, 4.0]), next_value=torch.tensor([100.0, 100.0]) if t == 0 else torch.tensor([5.0, 6.0]), done=torch.tensor([True, True]) if t == 0 else torch.tensor([False, False]), bootstrap_mask=torch.tensor([False, True]), session_id='s', policy_version=0, collection_token=str(t))
        if t == 1:
            row['bootstrap_mask'][:] = True
        rollout._append(row, capability)
    (returns, adv) = rollout.returns_and_advantages(PPOConfig(gamma=0.9, lam=0.8))
    torch.testing.assert_close(returns, torch.tensor([1.0, 92.0, 7.5, 9.4]), atol=1e-05, rtol=0)
    torch.testing.assert_close(adv, torch.tensor([0.8, 91.7, 7.1, 8.9]), atol=1e-05, rtol=0)

@pytest.mark.parametrize('raises', [False, True])
def test_rejected_or_exception_candidate_restores_full_policy_adam_and_all_rng(monkeypatch, raises):
    (model, obs, *_) = teacher_data(monkeypatch)
    trainer = trainer_for(model)
    seed(15)
    collect(trainer, obs)
    seed(18)
    assert trainer.update()['minibatch_updates'] == 1
    seed(21)
    collect(trainer, obs)
    batch = trainer.rollout.batch()
    for parameter in trainer.model.parameters():
        parameter.grad = torch.ones_like(parameter) * 0.03
    before_policy = parameter_digest(trainer.model)
    before_adam = digest_tree(trainer.optimizer.state_dict())
    before_rng = digest_tree(rng())

    def audit(*args):
        torch.rand(3)
        random.random()
        np.random.rand()
        if torch.cuda.is_initialized():
            for i in range(torch.cuda.device_count()):
                torch.rand(3, device='cuda:' + str(i))
        if raises:
            raise RuntimeError('audit failed')
        return torch.tensor(0.5)
    monkeypatch.setattr(trainer, '_minibatch_exact_kl', audit)
    if raises:
        with pytest.raises(RuntimeError, match='audit failed'):
            trainer._optimizer_step(batch)
    else:
        (accepted, attempts) = trainer._optimizer_step(batch)
        assert accepted is None and len(attempts) == 4
        assert [a['lr'] for a in attempts] == [trainer.config.lr * 0.1 ** i for i in range(4)]
    assert parameter_digest(trainer.model) == before_policy
    assert digest_tree(trainer.optimizer.state_dict()) == before_adam
    assert digest_tree(rng()) == before_rng

def test_frozen_ppo_byte_gate_rejects_changed_source(monkeypatch, tmp_path):
    import dh116.rl.device as module
    (model, *_) = teacher_data(monkeypatch)
    bad = tmp_path / 'ppo.py'
    bad.write_text('changed policy source')
    monkeypatch.setattr(module, 'source_file', lambda p: bad)
    with pytest.raises(ValueError, match='frozen base PPO'):
        trainer_for(model)

def checkpoint_context(profile):
    return dict(recipe_sha256='a' * 64, receipt_sha256='b' * 64, seed_sha256='c' * 64, backend_profile=profile, policy_seed=21)

def draw():
    return (torch.rand(9), random.random(), np.random.rand(9))

@pytest.mark.parametrize('fast_update', [False, True])
def test_checkpoint_continues_next_actual_update_with_policy_adam_rng_and_new_session(tmp_path, monkeypatch, fast_update):
    import dh116.rl.device as cp
    from dh116.env.contracts import runti_io__file_sha256 as file_sha256
    (model, obs, profile, *_) = teacher_data(monkeypatch)
    context = checkpoint_context(profile)
    original = trainer_for(model, fast_update=fast_update)
    seed(101)
    collect(original, obs)
    assert original.update()['minibatch_updates'] == 1
    path = tmp_path / 'tensor.pt'
    before_rng = digest_tree(rng())
    before_policy = parameter_digest(original.model)
    before_adam = digest_tree(original.optimizer.state_dict())
    old_session = original.session_id
    old_capability = original._capability
    old_rollout = original.rollout
    with monkeypatch.context() as save_audit:
        if fast_update:
            def unexpected_roundtrip(*args, **kwargs):
                pytest.fail('fast checkpoint save read the serialized snapshot')
            save_audit.setattr(torch, 'load', unexpected_roundtrip)
        report = cp.save_checkpoint(path, original, **context)
    assert digest_tree(rng()) == before_rng and parameter_digest(original.model) == before_policy
    assert digest_tree(original.optimizer.state_dict()) == before_adam
    assert original.session_id == old_session and original._capability is old_capability and (original.rollout is old_rollout)
    assert report['sha256'] == file_sha256(path) and report['physical_state_saved'] is False and (report['restart_requires_verified_reset'] is True)
    if fast_update:
        with pytest.raises(ValueError, match='identity'):
            cp.load_checkpoint(path, trainer_for(model), expected_sha256=report['sha256'], **context)
    expected_draw = digest_tree(draw())
    collect(original, obs)
    expected_batch = digest_tree(original.rollout.batch())
    expected_metrics = digest_tree(original.update())
    expected_rng = digest_tree(rng())
    restored = trainer_for(model, fast_update=fast_update)
    seed(999)
    draw()
    destination_session = restored.session_id
    destination_capability = restored._capability
    destination_rollout = restored.rollout
    cp.load_checkpoint(path, restored, expected_sha256=report['sha256'], **context)
    assert parameter_digest(restored.model) == before_policy and digest_tree(restored.optimizer.state_dict()) == before_adam
    assert digest_tree(rng()) == before_rng and restored.session_id not in (destination_session, old_session)
    assert restored._capability is not destination_capability and restored.rollout is not destination_rollout and restored.clean_boundary
    assert restored.rollout.session_id == restored.session_id and restored.rollout._capability is restored._capability
    assert digest_tree(draw()) == expected_draw
    collect(restored, obs)
    assert digest_tree(restored.rollout.batch()) == expected_batch
    assert digest_tree(restored.update()) == expected_metrics
    assert parameter_digest(restored.model) == parameter_digest(original.model)
    assert digest_tree(restored.optimizer.state_dict()) == digest_tree(original.optimizer.state_dict())
    assert digest_tree(rng()) == expected_rng and restored.policy_version == 2 and (restored.counters == original.counters)

def test_best_snapshot_preserves_the_policy_that_sampled_the_completed_rollout(tmp_path, monkeypatch):
    from dh116.rl.device import save_checkpoint

    model, obs, profile, *_ = teacher_data(monkeypatch)
    trainer = trainer_for(model, fast_update=True)
    seed(51)
    collect(trainer, obs)
    assert trainer.update()['minibatch_updates'] == 1
    collect(trainer, obs)
    before = parameter_digest(trainer.model)
    before_adam = digest_tree(trainer.optimizer.state_dict())
    path = tmp_path / 'best.pt'
    with pytest.raises(RuntimeError, match='clean PPO'):
        save_checkpoint(path, trainer, **checkpoint_context(profile))
    report = save_checkpoint(path, trainer, evaluated_rollout=True, **checkpoint_context(profile))
    assert trainer.rollout.full
    assert trainer.update()['minibatch_updates'] == 1
    assert trainer.policy_version == 2 and parameter_digest(trainer.model) != before
    snapshot = torch.load(path, map_location='cpu', weights_only=True)
    assert snapshot['policy_version'] == report['policy_version'] == 1
    restored = trainer_for(model, fast_update=True)
    from dh116.rl.device import load_checkpoint
    load_checkpoint(path, restored, expected_sha256=report['sha256'], **checkpoint_context(profile))
    assert parameter_digest(restored.model) == before
    assert digest_tree(restored.optimizer.state_dict()) == before_adam


def test_invalid_checkpoint_rejects_each_corruption_before_any_live_mutation(tmp_path, monkeypatch):
    import dh116.rl.device as cp
    from dh116.env.contracts import runti_io__file_sha256 as file_sha256
    (model, obs, profile, *_) = teacher_data(monkeypatch)
    context = checkpoint_context(profile)
    original = trainer_for(model)
    seed(101)
    collect(original, obs)
    original.update()
    path = tmp_path / 'original.pt'
    report = cp.save_checkpoint(path, original, **context)
    restored = trainer_for(model)
    for damage in ('namespace', 'source', 'counter', 'coverage', 'moment', 'adam_setting', 'rng', 'physical_state'):
        payload = torch.load(path, weights_only=True)
        if damage == 'namespace':
            payload['identity']['model']['checkpoint_namespace'] = 'tensor_native_teacher_power_v1'
        elif damage == 'source':
            payload['identity']['source_sha256']['tabletop_device_ppo.py'] = 'f' * 64
        elif damage == 'counter':
            payload['policy_version'] += 1
        elif damage == 'coverage':
            payload['counters']['environment_steps'] = 0
        elif damage == 'moment':
            del payload['optimizer']['state'][0]['exp_avg']
        elif damage == 'adam_setting':
            payload['optimizer']['param_groups'][0]['amsgrad'] = True
        elif damage == 'rng':
            payload['rng']['torch'] = torch.zeros(2, dtype=torch.uint8)
        else:
            payload['identity']['physical_state_saved'] = True
        bad = tmp_path / (damage + '.pt')
        torch.save(payload, bad)
        before = (parameter_digest(restored.model), digest_tree(restored.optimizer.state_dict()), digest_tree(rng()), deepcopy(restored.counters), restored.policy_version, restored.session_id, restored._capability, restored.rollout)
        with pytest.raises(ValueError):
            cp.load_checkpoint(bad, restored, expected_sha256=file_sha256(bad), **context)
        after = (parameter_digest(restored.model), digest_tree(restored.optimizer.state_dict()), digest_tree(rng()), restored.counters, restored.policy_version, restored.session_id, restored._capability, restored.rollout)
        assert after == before
    with pytest.raises(FileExistsError):
        cp.save_checkpoint(path, original, **context)
    with pytest.raises(ValueError, match='SHA'):
        cp.load_checkpoint(path, restored, expected_sha256='0' * 64, **context)
    original.collect_step(obs, obs)
    with pytest.raises(RuntimeError, match='boundary'):
        cp.save_checkpoint(tmp_path / 'pending.pt', original, **context)
    with pytest.raises(RuntimeError, match='boundary'):
        cp.load_checkpoint(path, original, expected_sha256=report['sha256'], **context)
