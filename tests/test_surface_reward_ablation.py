import pytest

from go2_pvcnn.control.dh116.tabletop_surface_exploration import (
    SURFACE_REWARD_PROFILES, reward_ablation_contract, validate_reward_profile)


def test_reward_profiles_are_single_factor_stages():
    assert SURFACE_REWARD_PROFILES[0] == 'baseline'
    for profile in SURFACE_REWARD_PROFILES:
        contract = reward_ablation_contract(profile)
        assert contract['profile'] == profile
        assert len(contract['enabled_terms']) <= 1
        if profile == 'baseline':
            assert contract['enabled_terms'] == []
        else:
            assert len(contract['enabled_terms']) == 1


def test_unknown_or_all_factor_profile_is_rejected():
    with pytest.raises(ValueError):
        validate_reward_profile('all_new_rewards')
    with pytest.raises(ValueError):
        reward_ablation_contract('all_new_rewards')
