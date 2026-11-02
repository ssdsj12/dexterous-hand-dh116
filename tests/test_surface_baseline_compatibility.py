import copy
import torch
from test_surface_exploration import surface_task
from test_surface_contract import legacy
from go2_pvcnn.control.dh116.tabletop_surface_contract import bind_contract,semantic_sha256
from go2_pvcnn.control.dh116.tabletop_surface_imitation import _check_contract
from go2_pvcnn.control.dh116.tabletop_surface_exploration import apply_contract,enable_stable_hold_rewards
from go2_pvcnn.control.dh116.tabletop_surface_reward_ablation import enable,STAGE_TERMS


def test_baseline_does_not_add_contract_fields_or_mutable_reward_state():
    t,w=surface_task();c={};apply_contract(c,t)
    assert 'surface_reward_ablation' not in c
    assert not hasattr(t,'surface_reward_ablation')


def test_371_bc_comparison_validates_then_canonicalizes_batch_sha():
    a=bind_contract(legacy());b=copy.deepcopy(a)
    b['num_envs']=16;b['surface_contract_sha256']=semantic_sha256(b)
    _check_contract(a,b)


def test_registered_single_reward_runtime_uses_native_inputs_and_retains_old_bc_baseline():
    for profile in ('lift_progress','early_hold','contact_break','slip_relative','support_deficit'):
        t,w=surface_task();enable_stable_hold_rewards(t)
        t.surface_reward_profile=profile;enable(t)
        assert all(term in t.terms for term in STAGE_TERMS[profile])
