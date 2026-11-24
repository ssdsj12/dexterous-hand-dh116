"""Owned device rollout and original CPU GAE/flatten parity."""
from copy import deepcopy
import importlib.util
import os

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_privileged_ppo import OnPolicyRollout, PPOConfig

MODULE = "go2_pvcnn.control.dh116.tabletop_device_rollout"


def row(t, n, device="cpu"):
    x = torch.arange(n, device=device).float()+t*.25
    return dict(actor_obs={"x": x[:,None].expand(n,3).clone(), "valid": torch.ones(n,3,dtype=torch.bool,device=device)},
        critic_obs={"x": (x+3)[:,None].expand(n,3).clone()}, actions=x[:,None].expand(n,24).clone(),
        latent_actions=(x+1)[:,None].expand(n,24).clone(), old_log_prob=x-2,
        old_mu=(x+2)[:,None].expand(n,24).clone(), old_std=torch.ones(n,24,device=device)*.2,
        values=x*.3, rewards=x*.7, done=torch.tensor([(i+t)%3 == 0 for i in range(n)],device=device),
        next_value=x*.4, bootstrap_mask=torch.tensor([(i+t)%4 != 0 for i in range(n)],device=device),
        session_id="session", policy_version=3, collection_token="t"+str(t))


def test_device_rollout_api_is_available():
    assert importlib.util.find_spec(MODULE) is not None, "owned device rollout is missing"


@pytest.mark.parametrize("n", [2, 128])
def test_cpu_exact_gae_flatten_and_retained_ledger_ownership(n):
    from go2_pvcnn.control.dh116.tabletop_device_rollout import DeviceOnPolicyRollout
    cap = object()
    args = dict(session_id="session", policy_version=3, capability=cap)
    cpu, device = OnPolicyRollout(5,n,**args), DeviceOnPolicyRollout(5,n,device="cpu",**args)
    rows = [row(t,n) for t in range(5)]
    for value in rows:
        cpu._append(deepcopy(value),cap); device._append(value,cap)
    assert len(device) == 5 and device.full
    returns, advantage = device.returns_and_advantages(PPOConfig())
    old_returns, old_advantage = cpu.returns_and_advantages(PPOConfig())
    torch.testing.assert_close(returns, old_returns, atol=0, rtol=0)
    torch.testing.assert_close(advantage, old_advantage, atol=0, rtol=0)
    expected = cpu.batch()
    # Caller mutation after append cannot rewrite a transition.
    rows[0]["actor_obs"]["x"].fill_(99); rows[0]["values"].fill_(99)
    batch = device.batch()
    for key, value in expected.items():
        if isinstance(value,dict):
            for field in value: torch.testing.assert_close(batch[key][field],value[field],atol=0,rtol=0)
        else: torch.testing.assert_close(batch[key],value,atol=0,rtol=0)
    # Legacy structured Critic indexing protocol remains available.
    selected = batch["critic_obs"][torch.tensor([n,0])].to("cpu")
    torch.testing.assert_close(selected["x"], expected["critic_obs"]["x"][[n,0]])
    batch["actor_obs"]["x"].fill_(-5); batch["values"].fill_(-5)
    torch.testing.assert_close(device.batch()["values"], expected["values"], atol=0, rtol=0)
    assert not returns.requires_grad and not advantage.requires_grad


def test_capability_policy_empty_partial_and_full_guards():
    from go2_pvcnn.control.dh116.tabletop_device_rollout import DeviceOnPolicyRollout
    cap = object()
    rollout = DeviceOnPolicyRollout(1,2,device="cpu",session_id="session",policy_version=3,capability=cap)
    with pytest.raises(RuntimeError,match="empty"): rollout.batch()
    with pytest.raises(RuntimeError,match="complete"): rollout.returns_and_advantages(PPOConfig())
    with pytest.raises(ValueError,match="capability"): rollout._append(row(0,2),object())
    wrong = row(0,2); wrong["policy_version"] += 1
    with pytest.raises(ValueError,match="policy"): rollout._append(wrong,cap)
    rollout._append(row(0,2),cap)
    with pytest.raises(RuntimeError,match="complete"): rollout._append(row(1,2),cap)


def test_rollout_rejects_wrong_lane_count_or_nonboolean_boundaries():
    from go2_pvcnn.control.dh116.tabletop_device_rollout import DeviceOnPolicyRollout
    cap = object()
    rollout = DeviceOnPolicyRollout(2,2,device="cpu",session_id="session",policy_version=3,capability=cap)
    for damaged in (row(0,3), dict(row(0,2),done=torch.zeros(2))):
        with pytest.raises(ValueError): rollout._append(damaged,cap)
    assert len(rollout) == 0


def test_two_terminal_lanes_do_not_bootstrap_but_explicit_truncation_does():
    from go2_pvcnn.control.dh116.tabletop_device_rollout import DeviceOnPolicyRollout
    cap = object()
    rollout = DeviceOnPolicyRollout(1,2,device="cpu",session_id="session",policy_version=3,capability=cap)
    value = row(0,2)
    value.update(rewards=torch.tensor([1.,2.]),done=torch.ones(2,dtype=torch.bool),
        next_value=torch.tensor([99.,3.]),bootstrap_mask=torch.tensor([False,True]))
    rollout._append(value,cap)
    returns, advantage = rollout.returns_and_advantages(PPOConfig())
    torch.testing.assert_close(returns,torch.tensor([1.,2.+.995*3.]),atol=0,rtol=0)
    torch.testing.assert_close(advantage,returns-value["values"],atol=0,rtol=0)


def test_public_device_clone_rejects_transfer_and_append_rejects_schema_changes():
    from go2_pvcnn.control.dh116.tabletop_device_rollout import DeviceOnPolicyRollout, own_device_tree
    with pytest.raises(ValueError,match="device"):
        own_device_tree(torch.zeros(2),"meta")
    cap = object()
    rollout = DeviceOnPolicyRollout(2,2,device="cpu",session_id="session",policy_version=3,capability=cap)
    rollout._append(row(0,2),cap)
    broken = row(1,2); broken["actor_obs"]["x"] = torch.zeros(2,4)
    with pytest.raises(ValueError,match="schema"): rollout._append(broken,cap)
    assert len(rollout) == 1


def test_rollout_uses_the_resolved_device_for_storage_and_gae():
    from go2_pvcnn.control.dh116.tabletop_device_rollout import DeviceOnPolicyRollout
    cap = object()
    rollout = DeviceOnPolicyRollout(1,2,device="cpu:0",session_id="session",policy_version=3,capability=cap)
    assert rollout.device == torch.zeros(0).device
    rollout._append(row(0,2),cap)
    all_value = rollout.returns_and_advantages(PPOConfig())
    assert all(t.device == rollout.device for t in all_value)


@pytest.mark.skipif(os.environ.get("DH116_RUN_DEVICE_TESTS") != "1", reason="operator must allocate a CUDA device")
def test_cuda_gae_and_owned_batch_never_leave_collection_device():
    from go2_pvcnn.control.dh116.tabletop_device_rollout import DeviceOnPolicyRollout
    cap = object(); device = os.environ.get("DH116_TEST_DEVICE", "cuda:0")
    args = dict(session_id="session",policy_version=3,capability=cap)
    old = OnPolicyRollout(5,128,**args)
    new = DeviceOnPolicyRollout(5,128,device=device,**args)
    for t in range(5):
        old._append(row(t,128),cap); new._append(row(t,128,device),cap)
    for actual, expected in zip(new.returns_and_advantages(PPOConfig()),old.returns_and_advantages(PPOConfig())):
        assert actual.device == torch.device(device)
        torch.testing.assert_close(actual.cpu(),expected,atol=1e-6,rtol=1e-6)
    for value in new.batch().values():
        for tensor in (value.values() if isinstance(value,dict) else (value,)):
            assert tensor.device == torch.device(device)
