import pytest

from go2_pvcnn.control.dh116.tabletop_surface_exploration import (
    SURFACE_REWARD_PROFILES, reward_ablation_contract, validate_reward_profile)


def test_reward_profiles_follow_documented_cumulative_stage_order():
    assert SURFACE_REWARD_PROFILES[0] == 'baseline'
    for profile in SURFACE_REWARD_PROFILES:
        contract = reward_ablation_contract(profile)
        assert contract['profile'] == profile
        if profile == 'baseline': assert contract['enabled_terms'] == []
        else: assert contract['enabled_terms']
    assert len(reward_ablation_contract('early_hold')['enabled_terms'])==3
    assert len(reward_ablation_contract('support_deficit')['enabled_terms'])==7


def test_unknown_or_all_factor_profile_is_rejected():
    with pytest.raises(ValueError):
        validate_reward_profile('all_new_rewards')
    with pytest.raises(ValueError):
        reward_ablation_contract('all_new_rewards')
