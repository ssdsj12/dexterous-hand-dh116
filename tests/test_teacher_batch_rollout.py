"""Exercise grouped PPO writes through the physical exchange using CPU fixtures."""
from dataclasses import replace

import pytest
import torch

from test_dh116_flexible_physical_exchange import setup
from test_native_teacher_context import backend_profile
from test_native_teacher_model import cpu_model
from test_native_teacher_reward import config as reward_config, endpoint
from test_teacher_batch_inputs import lanes

from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
from go2_pvcnn.control.dh116.tabletop_teacher_batch_inputs import pack_teacher_group
from go2_pvcnn.control.dh116.tabletop_teacher_physical_group import TeacherPhysicalGroup
from go2_pvcnn.control.dh116.tabletop_training_reward import SourceStamp, SubmittedRequest, TrainingReward


def measured(clock, **fields):
    stamp = SourceStamp(clock["episode_uid"], clock["physics_step"], clock["solver_time_s"],
                        "cpu_fixture", "test_reader", env_id=clock["env_id"],
                        reset_uid=str(clock["reset_id"]), rod_uid=clock["rod_uid"])
    sample = endpoint(clock["physics_step"], **fields)
    return replace(sample, stamp=stamp, fields={
        key: replace(value, stamp=stamp) for key, value in sample.fields.items()
    })


@pytest.mark.parametrize("grasp", ["pinch", "tripod", "power"])
def test_64_lane_windows_feed_existing_ppo(monkeypatch, backend_profile, grasp):
    from go2_pvcnn.control.dh116.tabletop_teacher_batch_rollout import record_group_window
    torch.manual_seed(31)
    count = 64
    cfg = reward_config()
    model = cpu_model(backend_profile, grasp=grasp)[0]
    trainer = DeformablePPOTrainer(model, PPOConfig(gamma=cfg.gamma, epochs=1, minibatches=1, lr=1e-5),
                                   rollout_steps=1, num_envs=count)
    obs = pack_teacher_group(model, lanes(backend_profile, grasp, count), backend_profile=backend_profile)
    sample = trainer.collect_step(obs, obs)
    first, sim, _, _ = setup(monkeypatch, env_id=0)
    exchanges = [first, *(setup(monkeypatch, env_id=i, sim=sim)[0] for i in range(1, count))]
    rewards = {}
    rows = {}
    token = sample["collection_token"]
    policy_id = "cpu-policy"
    for i, exchange in enumerate(exchanges):
        rewards[i] = reward = TrainingReward(cfg)
        rows[i] = []
        pre = measured(exchange._rod.clock())
        def posted(label, command):
            return SubmittedRequest(label, policy_id, pre.stamp, tuple(command),
                                    "normalized_submitted_command24_preprojection")
        reward.reset_from_verified_state(pre, previous_request=posted(f"reset-{i}", [0.] * 24))
        reward.begin(posted(token, sample["actions"][i].tolist()), remaining_substeps=8)

    def observe(row):
        env = row["rod"]["env_id"]
        rows[env].append(row)
        last = row["substep"] == 7
        terminal = env == 0 and last
        rewards[env].observe(measured(row["rod"], distance=.009 - env * .00003, failure=terminal))
        if terminal:
            return "trusted_constraint_failure"
        if env == 1 and last:
            return "time_limit"
        return None

    def commit(report):
        settlement = {i: reward.settle_once(token, truncated=i == 1, buffer_full=i > 1)
                      for i, reward in rewards.items()}
        boundaries = {
            i: {key: rows[i][-1]["rod"][key] for key in (
                "env_id", "episode_uid", "reset_id", "rod_uid", "physics_step", "rod_tick", "timestamp_s"
            )}
            for i in range(1, count)
        }
        values = {i: (float(i) / 16., dict(clock, phase="post_action_pre_reset"))
                  for i, clock in boundaries.items()}
        pristine = rows[0][4]["action_readback"]["request"]["action24"]
        rows[0][4]["action_readback"]["request"]["action24"] = pristine.clone().zero_()
        with pytest.raises(ValueError, match="sampled policy action"):
            record_group_window(trainer, sample, report, rows, settlement, values,
                                policy_uid=policy_id, reward_config=cfg)
        assert len(trainer.rollout) == 0
        rows[0][4]["action_readback"]["request"]["action24"] = pristine
        bad_values = dict(values)
        bad_values[1] = (0., dict(values[1][1], reset_id=99))
        with pytest.raises(ValueError, match="pre-reset"):
            record_group_window(trainer, sample, report, rows, settlement, bad_values,
                                policy_uid=policy_id, reward_config=cfg)
        assert len(trainer.rollout) == 0
        bad_rewards = dict(settlement)
        bad_rewards[2] = settlement[3]
        with pytest.raises(ValueError, match="reward endpoints"):
            record_group_window(trainer, sample, report, rows, bad_rewards, values,
                                policy_uid=policy_id, reward_config=cfg)
        assert len(trainer.rollout) == 0
        record_group_window(trainer, sample, report, rows, settlement, values,
                            policy_uid=policy_id, reward_config=cfg)

    report = TeacherPhysicalGroup(exchanges).control(sample["actions"], observe=observe, commit=commit)
    assert report["env_ids"] == list(range(64))
    assert sim.current_time_step_index == 10
    ledger = trainer.rollout.batch()
    assert ledger["done"].tolist() == [True, True] + [False] * 62
    assert ledger["bootstrap_mask"].tolist() == [False] + [True] * 63
    assert ledger["next_value"][:2].tolist() == [0., 1. / 16.]
    assert trainer.update()["samples"] == 64
