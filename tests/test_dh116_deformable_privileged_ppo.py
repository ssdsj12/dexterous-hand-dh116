import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_deformable_privileged_checkpoint import load_checkpoint, save_checkpoint
from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
from go2_pvcnn.control.dh116.tabletop_deformable_privileged_teacher import (
    DeformablePrivilegedTeacher, load_deformable_teacher_config,
)
from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig, PPOTrainer
from go2_pvcnn.control.dh116.tabletop_privileged_teacher import PrivilegedTeacher
from test_dh116_deformable_privileged_teacher import config_path, inputs


def same(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def make_trainer(grasp="pinch", ppo=None):
    config, identity = load_deformable_teacher_config(config_path(grasp), Path.cwd())
    model = DeformablePrivilegedTeacher(config, identity)
    trainer = DeformablePPOTrainer(model, ppo or PPOConfig(epochs=1, minibatches=1),
                                   rollout_steps=2, num_envs=2)
    common, rod, clock, mask = inputs(identity)
    return trainer, model.pack(common, rod, clock, mask)


def record(trainer, sample, *, terminal=False):
    if terminal:
        done = torch.tensor([True, True])
        truncated = torch.tensor([False, True])
        bootstrap = torch.tensor([False, True])
    else:
        done = truncated = torch.zeros(2, dtype=torch.bool)
        bootstrap = ~done
    trainer.record_step(torch.tensor([1., 2.]), done, torch.tensor([100., 3.]), truncated=truncated,
                        bootstrap_mask=bootstrap, executed_actions=sample["actions"],
                        collection_token=sample["collection_token"])


def populate(trainer, obs):
    for terminal in (True, False):
        sample = trainer.collect_step(obs, obs)
        record(trainer, sample, terminal=terminal)


def test_owned_tree_batches_terminal_gae_and_full_update():
    trainer, obs = make_trainer()
    initial = copy.deepcopy(trainer.model.state_dict())
    old_nodes = obs["rod_nodes"].clone()
    sample = trainer.collect_step(obs, obs)
    record(trainer, sample, terminal=True)
    obs["rod_nodes"].zero_()
    record(trainer, trainer.collect_step(obs, obs))
    batch = trainer.rollout.batch()
    assert isinstance(batch["critic_obs"], dict) and not isinstance(batch["critic_obs"], torch.Tensor)
    chunk = batch["critic_obs"][torch.tensor([1, 0])].to("cpu")
    assert chunk.keys() == {"common", "rod_nodes", "rod_mask"}
    torch.testing.assert_close(chunk["rod_nodes"], old_nodes.flip(0))
    batch["critic_obs"]["rod_nodes"].zero_()
    torch.testing.assert_close(trainer.rollout.batch()["critic_obs"]["rod_nodes"][:2], old_nodes)
    returns, _ = trainer.rollout.returns_and_advantages(trainer.config)
    torch.testing.assert_close(returns[:2], torch.tensor([1., 2. + 3. * trainer.config.gamma]))
    metrics = trainer.update()
    assert metrics["samples"] == 4 and metrics["minibatch_updates"] == 1
    assert metrics["frozen_policy_check"]["max_abs_log_probability_error"] < 1e-6
    assert trainer.counters == {"environment_steps": 4, "updates": 1}
    for name in ("actor_rod", "critic_rod"):
        assert any(not torch.equal(param, initial[key]) for key, param in trainer.model.state_dict().items()
                   if key.startswith(name))
    assert DeformablePPOTrainer.update is PPOTrainer.update
    assert DeformablePPOTrainer._optimizer_step is PPOTrainer._optimizer_step


class CommonOnlyFixture(DeformablePrivilegedTeacher):
    """The same legacy weights/equations behind an explicitly structured CPU fixture."""

    def __init__(self, weights):
        PrivilegedTeacher.__init__(self)
        self.load_state_dict(weights)

    def latent_mean(self, obs):
        return PrivilegedTeacher.latent_mean(self, obs["common"])

    def value(self, obs):
        return PrivilegedTeacher.value(self, obs["common"])


def test_common_only_equivalence_preserves_legacy_sampling_gae_and_optimizer():
    legacy = PrivilegedTeacher()
    structured = CommonOnlyFixture(legacy.state_dict())
    ppo = PPOConfig(epochs=1, minibatches=2)
    old = PPOTrainer(legacy, ppo, rollout_steps=2, num_envs=2)
    new = DeformablePPOTrainer(structured, ppo, rollout_steps=2, num_envs=2)
    _, obs = make_trainer()
    for step, terminal in enumerate((True, False)):
        torch.manual_seed(71 + step)
        a = old.collect_step(obs["common"], obs["common"])
        torch.manual_seed(71 + step)
        b = new.collect_step(obs, obs)
        for key in ("actions", "latent_actions", "log_prob", "value"):
            assert torch.equal(a[key], b[key])
        record(old, a, terminal=terminal)
        record(new, b, terminal=terminal)
    assert same(old.rollout.returns_and_advantages(ppo), new.rollout.returns_and_advantages(ppo))
    torch.manual_seed(99)
    old_metrics = old.update()
    torch.manual_seed(99)
    assert new.update() == old_metrics
    assert same(old.model.state_dict(), new.model.state_dict())
    assert same(old.optimizer.state_dict(), new.optimizer.state_dict())


def test_kl_rejection_restores_live_adam_and_all_cpu_rng(monkeypatch):
    trainer, obs = make_trainer(ppo=PPOConfig(epochs=1, minibatches=1, target_kl=10., kl_backtrack_steps=1))
    populate(trainer, obs)
    assert trainer.update()["minibatch_updates"] == 1
    policy, adam = copy.deepcopy(trainer.model.state_dict()), copy.deepcopy(trainer.optimizer.state_dict())
    populate(trainer, obs)
    original = trainer.optimizer.step
    entering = []

    def step():
        entering.append((torch.get_rng_state().clone(), random.getstate(), np.random.get_state()))
        torch.rand(1)
        random.random()
        np.random.rand()
        return original()

    monkeypatch.setattr(trainer.optimizer, "step", step)
    monkeypatch.setattr(trainer, "_minibatch_exact_kl", lambda *_: 20.)
    metrics = trainer.update()
    assert metrics["early_stopped"] and metrics["minibatch_updates"] == 0 and len(entering) == 2
    assert same(entering[0], entering[1])
    assert same(entering[0], (torch.get_rng_state(), random.getstate(), np.random.get_state()))
    assert same(policy, trainer.model.state_dict())
    assert same(adam, trainer.optimizer.state_dict())
    assert trainer.counters["updates"] == 1


def test_exclusive_checkpoint_roundtrip_matches_cpu_rollout_and_rejects_cross_grasp(tmp_path):
    trainer, obs = make_trainer()
    populate(trainer, obs)
    trainer.update()
    checkpoint = tmp_path / "pinch.pt"
    saved = save_checkpoint(checkpoint, trainer)
    assert len(saved["sha256"]) == 64
    with pytest.raises(FileExistsError):
        save_checkpoint(checkpoint, trainer)
    reference_rng = (torch.rand(4), np.random.rand(4), random.random())
    populate(trainer, obs)
    expected_metrics = trainer.update()
    target, _ = make_trainer()
    load_checkpoint(checkpoint, target)
    assert same(reference_rng, (torch.rand(4), np.random.rand(4), random.random()))
    populate(target, obs)
    assert target.update() == expected_metrics
    assert same(trainer.model.state_dict(), target.model.state_dict())
    assert same(trainer.optimizer.state_dict(), target.optimizer.state_dict())
    assert trainer.counters == target.counters
    tripod, _ = make_trainer("tripod")
    before = copy.deepcopy(tripod.model.state_dict())
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(checkpoint, tripod)
    assert same(before, tripod.model.state_dict())
    old = tmp_path / "legacy.pt"
    torch.save({"version": "dh116-privileged-teacher-checkpoint-v1"}, old)
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(old, target)


@pytest.mark.parametrize(("field", "value"), (("amsgrad", True), ("betas", (.5, .9))))
def test_checkpoint_rejects_changed_adam_settings_before_restore(tmp_path, field, value):
    trainer, obs = make_trainer()
    populate(trainer, obs)
    assert trainer.update()["minibatch_updates"] == 1
    checkpoint = tmp_path / "pristine.pt"
    save_checkpoint(checkpoint, trainer)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["optimizer"]["param_groups"][0][field] = value
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    target, _ = make_trainer()
    policy, adam = copy.deepcopy(target.model.state_dict()), copy.deepcopy(target.optimizer.state_dict())
    with pytest.raises(ValueError, match="Adam parameter-group settings"):
        load_checkpoint(tampered, target)
    assert same(target.model.state_dict(), policy) and same(target.optimizer.state_dict(), adam)
    assert target.counters == {"environment_steps": 0, "updates": 0}


@pytest.mark.parametrize("environment_steps", [0, 5])
def test_checkpoint_rejects_impossible_full_rollout_counters(tmp_path, environment_steps):
    trainer, obs = make_trainer()
    populate(trainer, obs)
    assert trainer.update()["minibatch_updates"] == 1
    checkpoint = tmp_path / "pristine.pt"
    save_checkpoint(checkpoint, trainer)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["counters"]["environment_steps"] = environment_steps
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    target, _ = make_trainer()
    with pytest.raises(ValueError, match="environment steps.*rollouts"):
        load_checkpoint(tampered, target)
    assert target.counters == {"environment_steps": 0, "updates": 0}


def test_checkpoint_allows_collected_rollout_with_no_accepted_kl_update(tmp_path, monkeypatch):
    config = PPOConfig(epochs=1, minibatches=1, target_kl=10., kl_backtrack_steps=1)
    trainer, obs = make_trainer(ppo=config)
    populate(trainer, obs)
    monkeypatch.setattr(trainer, "_minibatch_exact_kl", lambda *_: 20.)
    assert trainer.update()["minibatch_updates"] == 0
    assert trainer.counters == {"environment_steps": 4, "updates": 0}
    checkpoint = tmp_path / "rejected.pt"
    save_checkpoint(checkpoint, trainer)
    target, _ = make_trainer(ppo=config)
    load_checkpoint(checkpoint, target)
    assert target.counters == trainer.counters
