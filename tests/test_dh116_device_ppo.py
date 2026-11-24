"""Original PPO numerical oracle plus explicit, operator-allocated CUDA tests."""
from copy import deepcopy
import importlib.util
import os
import random

import numpy as np
import pytest
import torch

from test_native_teacher_context import backend_profile


MODULE = "go2_pvcnn.control.dh116.tabletop_device_ppo"


@pytest.fixture(scope="module")
def teacher_data(backend_profile):
    from test_dh116_shared_teacher_source import reader
    from test_native_teacher_model import cpu_model
    from go2_pvcnn.control.dh116.tabletop_shared_native_teacher import SharedNativeTeacher
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import ROOT
    from go2_pvcnn.control.dh116.tabletop_training_reward import load_config
    source, _, common, context = reader(backend_profile)
    _, *_, config, _ = cpu_model(backend_profile)
    reward = load_config(ROOT/"configs/dh116_training/pinch_finger_goal_reward_v1.json",ppo_gamma=.995)
    model = SharedNativeTeacher(config,backend_profile=source.profile,num_envs=2,reward_config=reward)
    obs = model.pack_shared(source.capture(common,context),source)
    return model,obs


def seed(value):
    torch.manual_seed(value); random.seed(value); np.random.seed(value)


def state_equal(actual, expected, *, atol=0):
    if isinstance(actual,torch.Tensor):
        torch.testing.assert_close(actual,expected,atol=atol,rtol=atol,equal_nan=True)
    elif isinstance(actual,dict):
        assert actual.keys() == expected.keys()
        for key in actual: state_equal(actual[key],expected[key],atol=atol)
    elif isinstance(actual,(tuple,list)):
        assert len(actual) == len(expected)
        for a,b in zip(actual,expected): state_equal(a,b,atol=atol)
    elif isinstance(actual,float):
        assert actual == pytest.approx(expected,abs=atol,rel=atol)
    else: assert actual == expected


def all_on(value, device):
    if isinstance(value,torch.Tensor): assert value.device == torch.device(device)
    elif isinstance(value,dict):
        for item in value.values(): all_on(item,device)
    elif isinstance(value,(tuple,list)):
        for item in value: all_on(item,device)


def collect(trainer, obs, *, count=3):
    for t in range(count):
        sample = trainer.collect_step(obs,obs)
        all_on(trainer._pending,trainer.device)
        n = trainer.num_envs
        reward = torch.arange(n,device=trainer.device).float()+1+t*.25
        done = torch.zeros(n,dtype=torch.bool,device=trainer.device)
        trunc = torch.zeros_like(done); bootstrap = ~done
        if t == 1:
            done[:2] = True; trunc[1] = True; bootstrap[0] = False
        next_value = torch.ones(n,device=trainer.device)*(t+.5)
        trainer.record_step(reward,done,next_value,truncated=trunc,bootstrap_mask=bootstrap,
            executed_actions=sample["actions"],collection_token=sample["collection_token"])
        all_on(trainer.rollout._rows,trainer.device)
    all_on(trainer.rollout.batch(),trainer.device)


def test_device_trainer_api_is_available():
    assert importlib.util.find_spec(MODULE) is not None, "complete device PPO adapter is missing"


@pytest.mark.parametrize("guarded", [False, True])
def test_cpu_two_updates_exact_parameters_optimizer_metrics_and_rng(teacher_data, guarded):
    from go2_pvcnn.control.dh116.tabletop_device_ppo import DeviceDeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
    model,obs = teacher_data
    config = PPOConfig(epochs=2,minibatches=2,lr=1e-4,target_kl=.02 if guarded else None,
        kl_backtrack_steps=2 if guarded else 0)
    original = DeformablePPOTrainer(deepcopy(model),config,rollout_steps=3,num_envs=2,device="cpu")
    device = DeviceDeformablePPOTrainer(deepcopy(model),config,rollout_steps=3,num_envs=2,device="cpu")
    for update in range(2):
        seed(199+update); collect(original,obs)
        seed(199+update); collect(device,obs)
        for key in original.rollout.batch(): state_equal(device.rollout.batch()[key],original.rollout.batch()[key])
        seed(800+update); old_metrics = original.update()
        old_rng = torch.get_rng_state().clone()
        seed(800+update); new_metrics = device.update()
        state_equal(device.model.state_dict(),original.model.state_dict())
        state_equal(device.optimizer.state_dict(),original.optimizer.state_dict())
        state_equal(new_metrics,old_metrics)
        assert torch.equal(old_rng,torch.get_rng_state())
        assert device.counters == original.counters
        assert device.policy_version == original.policy_version
        assert device.clean_boundary


def test_collector_guards_and_retained_observation_ownership(teacher_data):
    from go2_pvcnn.control.dh116.tabletop_device_ppo import DeviceDeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
    model,obs = teacher_data
    trainer = DeviceDeformablePPOTrainer(deepcopy(model),PPOConfig(),rollout_steps=1,num_envs=2)
    supplied = deepcopy(obs)
    sample = trainer.collect_step(supplied,supplied)
    supplied["rod_nodes"].fill_(99)
    torch.testing.assert_close(trainer._pending["actor_obs"]["rod_nodes"],obs["rod_nodes"],atol=0,rtol=0)
    with pytest.raises(RuntimeError,match="pending"): trainer.collect_step(obs,obs)
    with pytest.raises(RuntimeError,match="pending"): trainer.update()
    args = (torch.ones(2),torch.ones(2,dtype=torch.bool),torch.zeros(2))
    with pytest.raises(ValueError,match="owner"): trainer.record_step(*args,action_owner="other")
    with pytest.raises(ValueError,match="token"): trainer.record_step(*args,collection_token="wrong")
    with pytest.raises(ValueError,match="bootstrap"):
        trainer.record_step(*args,bootstrap_mask=torch.ones(2,dtype=torch.bool))
    with pytest.raises(ValueError,match="truncation"):
        trainer.record_step(*args,truncated=torch.ones(2,dtype=torch.bool))
    with pytest.raises(ValueError,match="issued"):
        trainer.record_step(*args,executed_actions=sample["actions"]+1)
    trainer.record_step(*args,executed_actions=sample["actions"],collection_token=sample["collection_token"])
    with pytest.raises(RuntimeError,match="complete"): trainer.collect_step(obs,obs)
    with pytest.raises(RuntimeError,match="pending"): trainer.record_step(*args)


def test_constructor_normalizes_device_to_actual_model_device(teacher_data):
    from go2_pvcnn.control.dh116.tabletop_device_ppo import DeviceDeformablePPOTrainer
    model,obs = teacher_data
    trainer = DeviceDeformablePPOTrainer(deepcopy(model),rollout_steps=1,num_envs=2,device="cpu:0")
    assert trainer.device == next(trainer.model.parameters()).device
    trainer.collect_step(obs,obs)


def test_exact_kl_full_4096_is_one_unchunked_actor_forward(teacher_data):
    from go2_pvcnn.control.dh116.tabletop_device_ppo import DeviceDeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_squashed_gaussian import latent_gaussian_kl
    model,obs = teacher_data
    trainer = DeviceDeformablePPOTrainer(deepcopy(model),num_envs=2)
    full = {k:v.repeat((2048,)+(1,)*(v.ndim-1)) for k,v in obs.items()}
    with torch.no_grad():
        mu = trainer.model.latent_mean(full)
        std = trainer.model.log_std.exp().expand_as(mu).clone()
    batch = dict(actor_obs=full,old_mu=mu+.001,old_std=std,values=torch.zeros(4096))
    calls = []
    handle = trainer.model.actor.register_forward_pre_hook(lambda module,args:calls.append(len(args[0])))
    trainer.max_visual_batch = 7  # full audit must ignore legacy visual chunk size
    try: actual = trainer._minibatch_exact_kl(batch,torch.arange(4096))
    finally: handle.remove()
    expected = latent_gaussian_kl(batch["old_mu"],std,mu,std).clamp_min(0).sum().double()/4096
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert calls == [4096]


def test_rejected_candidates_restore_adam_parameters_and_all_rng(teacher_data):
    from go2_pvcnn.control.dh116.tabletop_device_ppo import DeviceDeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
    model,obs = teacher_data
    config = PPOConfig(epochs=1,minibatches=1,lr=1e-4,target_kl=.02,kl_backtrack_steps=2)
    original = DeformablePPOTrainer(deepcopy(model),config,rollout_steps=3,num_envs=2)
    device = DeviceDeformablePPOTrainer(deepcopy(model),config,rollout_steps=3,num_envs=2)
    for trainer in (original,device):
        # Warm Adam state, then force real candidate audits to reject.
        seed(15); collect(trainer,obs); seed(18); trainer.update()
        seed(21); collect(trainer,obs)
    before = deepcopy(device.model.state_dict())
    adam_before = deepcopy(device.optimizer.state_dict())
    def fail_audit(batch,indices):
        torch.rand(3); random.random(); np.random.rand()
        return torch.tensor(.5,device=device.device)
    original._minibatch_exact_kl = fail_audit
    device._minibatch_exact_kl = fail_audit
    seed(51); old = original.update(); old_rng = torch.get_rng_state().clone()
    old_python, old_np = random.getstate(), deepcopy(np.random.get_state())
    seed(51); new = device.update()
    state_equal(new,old)
    state_equal(device.model.state_dict(),before)
    # LR retains the last rejected trial exactly as the legacy rollback does.
    state_equal(device.optimizer.state_dict(),original.optimizer.state_dict())
    state_equal(device.optimizer.state_dict()["state"],adam_before["state"])
    assert torch.equal(torch.get_rng_state(),old_rng)
    assert random.getstate() == old_python
    assert np.array_equal(np.random.get_state()[1],old_np[1])
    assert new["minibatch_updates"] == 0 and new["early_stopped"]


@pytest.mark.skipif(os.environ.get("DH116_RUN_DEVICE_TESTS") != "1", reason="operator must allocate CUDA")
def test_cuda_complete_update_owns_device_tensors_and_reports_scalar_stats(teacher_data,monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_device_ppo import DeviceDeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
    model,obs = teacher_data
    device = os.environ.get("DH116_TEST_DEVICE","cuda:0")
    trainer = DeviceDeformablePPOTrainer(deepcopy(model),PPOConfig(epochs=1,minibatches=2,
        lr=1e-5,target_kl=.02,kl_backtrack_steps=2),rollout_steps=3,num_envs=2,device=device)
    obs = {k:v.to(device) for k,v in obs.items()}
    def forbidden(*args,**kwargs): pytest.fail("training tensors must not be copied to CPU")
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor,"cpu",forbidden)
        patch.setattr(torch.Tensor,"numpy",forbidden)
        collect(trainer,obs)
        result = trainer.update()
    assert result["minibatch_updates"] == 2 and trainer.clean_boundary
    all_on(trainer.model.state_dict(),device)
    # Preserve original non-capturable Adam: only its scalar step is CPU.
    for state in trainer.optimizer.state_dict()["state"].values():
        all_on({k:v for k,v in state.items() if k != "step"},device)
    def no_tensors(value):
        assert not isinstance(value,torch.Tensor)
        if isinstance(value,dict):
            for v in value.values(): no_tensors(v)
        elif isinstance(value,(tuple,list)):
            for v in value: no_tensors(v)
    no_tensors(result)


@pytest.mark.skipif(os.environ.get("DH116_RUN_DEVICE_TESTS") != "1", reason="operator must allocate CUDA")
def test_cuda_4096_guard_once_and_rejected_update_restores_cuda_rng(teacher_data):
    from go2_pvcnn.control.dh116.tabletop_device_ppo import DeviceDeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
    model,obs = teacher_data
    device = os.environ.get("DH116_TEST_DEVICE","cuda:0")
    trainer = DeviceDeformablePPOTrainer(deepcopy(model),PPOConfig(epochs=1,minibatches=1,
        lr=1e-5,target_kl=.02,kl_backtrack_steps=2),rollout_steps=32,num_envs=128,device=device)
    obs = {k:v.repeat((64,)+(1,)*(v.ndim-1)).to(device) for k,v in obs.items()}
    seed(92); collect(trainer,obs,count=32)
    batch = trainer.rollout.batch()
    calls = []
    handle = trainer.model.actor.register_forward_pre_hook(lambda module,args:calls.append(len(args[0])))
    try: kl = trainer._minibatch_exact_kl(batch,torch.arange(4096,device=device))
    finally: handle.remove()
    assert calls == [4096] and kl.device == torch.device(device)
    seed(93); result = trainer.update()
    assert result["samples"] == 4096 and result["minibatch_updates"] == 1
    # A rejected candidate can execute CUDA/random hooks without consuming
    # their RNG streams or mutating the initialized Adam moments.
    seed(94); collect(trainer,obs,count=32)
    before = deepcopy(trainer.model.state_dict())
    adam = deepcopy(trainer.optimizer.state_dict())
    batch = trainer.rollout.batch()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.model.latent_mean(obs).square().mean().backward()
    def reject(batch,indices):
        torch.rand(4,device=device); torch.rand(4); random.random(); np.random.rand()
        return torch.tensor(.5,device=device)
    trainer._minibatch_exact_kl = reject
    seed(95)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    python_rng,np_rng = random.getstate(),deepcopy(np.random.get_state())
    lr,attempts = trainer._optimizer_step(batch)
    assert lr is None and len(attempts) == 3
    state_equal(trainer.model.state_dict(),before)
    state_equal(trainer.optimizer.state_dict()["state"],adam["state"])
    assert torch.equal(cpu_rng,torch.get_rng_state())
    assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1],np_rng[1])
