from types import SimpleNamespace
import pytest


def budget(args,iteration,timesteps,episodes):
    from go2_pvcnn.control.dh116.tabletop_tensor_training import training_budget_reached
    return training_budget_reached(args,iteration,timesteps,episodes)


def test_exact_2048_updates_override_default_episode_and_timestep_budgets():
    args=SimpleNamespace(max_iterations=2048,episodes_per_env=1024,timesteps=1)
    assert not budget(args,2047,201228288,9999)
    assert budget(args,2048,201326592,0)


def test_legacy_episode_and_timestep_budgets_preserved():
    args=SimpleNamespace(episodes_per_env=1024,timesteps=1)
    assert not budget(args,2000,99999,1023)
    assert budget(args,1,1,1024)
    args.episodes_per_env=None
    assert budget(args,1,1,0)


def test_old_contract_cannot_silently_resume_new_reward():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    from go2_pvcnn.control.dh116.tabletop_stable_lift_reward import CONTRACT
    saved=dict(observation_size=186,observation_version='dh116-policy-lift-hold-v1')
    current=dict(saved,stable_lift_reward=CONTRACT,observation_size=188,
                 observation_version='dh116-stable-lift-reward-v1')
    with pytest.raises(ValueError,match='contract mismatch'):
        checkpoint_contract_changes(saved,current)
