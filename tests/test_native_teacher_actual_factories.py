"""CPU test seams for factory composition without any native qualification."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_factories as factories_module
from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_factories import build_actual_factories
from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_process import ActualWorkerLease
from go2_pvcnn.control.dh116.tabletop_native_teacher_session import (
    ModelContract, NativeEpisodeOwner, NativeFactories, NativeSessionConfig, StaticContract,
)
from go2_pvcnn.control.dh116.tabletop_native_teacher_training import VERSION
from go2_pvcnn.control.dh116.tabletop_training_reward import TrainingReward, load_config
from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
from go2_pvcnn.control.dh116.tabletop_native_teacher_context import TaskContextOwner
from test_native_teacher_context import ROOT, backend_profile
from test_dh116_flexible_measured_common_teacher import setup

SMOKE = "approach_wiring_smoke_v1"


@pytest.fixture
def boundaries(monkeypatch, tmp_path, backend_profile):
    """Only the borrowed owner, live Reset marker, and lease are test doubles."""
    model_config, *_ = setup("pinch")
    preflight = dict(style="pinch", profile=SMOKE)
    config = NativeSessionConfig(preflight, model_config, tmp_path / "train.jsonl", 2., 32, ROOT)
    reward = load_config(ROOT / "configs/dh116_training/approach_wiring_smoke_v1.json", ppo_gamma=.995)
    ppo = PPOConfig(lr=.0001, gamma=.995, epochs=1, minibatches=1)
    static = StaticContract(config, backend_profile, None, {}, ppo, reward, {}, None, "")
    lease = object.__new__(ActualWorkerLease)
    lease._config = dict(schema=VERSION, preflight=preflight, model_config=model_config,
                         episode_cap_s=2., rollout_steps=32)
    lease._closed = False
    lease_events = []

    def revalidate(self):
        if self._closed:
            raise ValueError("expired test lease")
        lease_events.append("revalidated")

    monkeypatch.setattr(ActualWorkerLease, "revalidate", revalidate)
    owner = object.__new__(NativeEpisodeOwner)
    owner.contract, owner.backend_profile = static, backend_profile
    owner._closed = False
    owner.episode = SimpleNamespace(_state="new")
    owner.sources = SimpleNamespace(scope="owner_source_unqualified")
    owner.context = TaskContextOwner(backend_profile=backend_profile, mode=SMOKE, grasp="pinch",
        scope="owner_source_unqualified", episode_cap_s=2., external_cap_s=5.)
    calls = []
    fs = build_actual_factories(lease=lease, episode_factory=lambda contract: owner,
        prepare_source_pack=lambda held, contract: (lambda policy: calls.append((held, policy))))
    return fs, lease, owner, static, calls, lease_events


def _finished_live_check(owner):
    owner.episode._state = "active"
    # Stand-in for C04 having called reset_from_verified_state. This fixture
    # calls no state reader and does not construct a native witness.
    owner.context._current = object()


def test_real_factories_use_c04_owner_and_original_components(boundaries):
    fs, lease, owner, static, calls, lease_events = boundaries
    assert type(fs) is NativeFactories and fs.episode(static) is owner
    _finished_live_check(owner)
    live = ModelContract(static.config.model_config, static.backend_profile, SMOKE)
    policy = fs.policy(live)
    trainer = fs.trainer(policy, static.ppo_config)
    reward = fs.reward(static.reward_config)
    assert calls == [(owner, policy)] and lease_events
    assert type(trainer) is DeformablePPOTrainer and trainer.model is policy
    assert trainer.rollout_steps == 32 and trainer.num_envs == 1
    assert trainer.counters["updates"] == 0 and trainer.device.type == "cpu"
    assert type(reward) is TrainingReward and reward.config.profile == SMOKE
    assert policy.model_identity["native_ready"] is False


def test_rejects_stale_or_foreign_boundaries_before_model_creation(boundaries):
    fs, lease, owner, static, calls, _ = boundaries
    live = ModelContract(static.config.model_config, static.backend_profile, SMOKE)
    with pytest.raises(ValueError, match="issued process config"):
        fs.episode(replace(static, config=replace(static.config, rollout_steps=16)))
    with pytest.raises(ValueError, match="same live episode"):
        fs.policy(live)
    fs.episode(static)
    with pytest.raises(ValueError, match="same live episode"):
        fs.policy(live)
    _finished_live_check(owner)
    with pytest.raises(ValueError, match="model contract"):
        fs.policy(replace(live, config=dict(live.config)))
    lease._closed = True
    with pytest.raises(ValueError, match="expired test lease"):
        fs.policy(live)
    assert calls == []


def test_trainer_and_reward_reject_foreign_model_or_contract(boundaries):
    fs, _, owner, static, calls, _ = boundaries
    fs.episode(static)
    _finished_live_check(owner)
    live = ModelContract(static.config.model_config, static.backend_profile, SMOKE)
    policy = fs.policy(live)
    with pytest.raises(ValueError, match="same policy"):
        fs.trainer(object(), static.ppo_config)
    with pytest.raises(ValueError, match="approved PPO"):
        fs.trainer(policy, replace(static.ppo_config, lr=.5))
    with pytest.raises(ValueError, match="approved reward"):
        fs.reward(object())
    assert calls == [(owner, policy)]




def test_unavailable_source_capability_fails_before_model_initialization(boundaries, monkeypatch):
    _, lease, owner, static, _, _ = boundaries

    def absent_source(held, contract):
        assert held is owner and contract.backend_profile is static.backend_profile
        raise ValueError("actual source capability unavailable")

    def forbid_model(*args, **kwargs):
        pytest.fail("model built before actual source capability")

    fs = build_actual_factories(lease=lease, episode_factory=lambda contract: owner,
                                prepare_source_pack=absent_source)
    monkeypatch.setattr(factories_module, "NativeContextTeacher", forbid_model)
    fs.episode(static)
    _finished_live_check(owner)
    with pytest.raises(ValueError, match="actual source capability unavailable"):
        fs.policy(ModelContract(static.config.model_config, static.backend_profile, SMOKE))
