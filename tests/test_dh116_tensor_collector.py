"""Collector-control tests use real reward and PPO writers with CPU physics fixtures."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_native_teacher_context import backend_profile
from test_dh116_device_ppo import teacher_data
from test_tensor_training_reward import start, config, row, batch
from go2_pvcnn.control.dh116 import tabletop_tensor_collector as collector
from go2_pvcnn.control.dh116.tabletop_device_ppo import DeviceDeformablePPOTrainer
from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig


def test_raw_exact_fallback_both_sides_shared_batch(monkeypatch):
    vertices=np.array([[0,0,0],[.03,.07,.005]],dtype=float)
    names=tuple("link"+str(i) for i in range(23))
    hands={s:dict(source_colliders={k:dict(vertices=vertices) for k in names[:17]}) for s in collector.SIDES}
    guard=collector.RawTableClearance(dict(hands=hands),(names,names),"cpu")
    poses=torch.zeros(2,2,23,7,dtype=torch.float64)
    poses[...,3]=1
    poses[...,2]=.3+collector.CLEARANCE_M+.000001
    oracle=collector.collision_clearance
    calls=[]
    def checked(p,s):
        calls.append(len(p))
        return oracle(p,s)
    monkeypatch.setattr(collector,"collision_clearance",checked)
    actual=guard(poses)
    torch.testing.assert_close(actual,torch.tensor(oracle(poses.numpy(),guard.shapes)[0]),rtol=0,atol=0)
    assert calls==[2]
    poses[1,0,0,2]-=.0000015
    with pytest.raises(ValueError,match="table clearance"):
        guard(poses)
    assert calls==[2,2]


@pytest.mark.parametrize("inject_failure",[True,False])
def test_substep_aborts_before_next_sdk_or_records_pre_reset_value(teacher_data,inject_failure):
    model,obs=teacher_data
    c=collector.NativeTensorCollector.__new__(collector.NativeTensorCollector)
    c.probe_dir=None
    c.n,c.controls,c.runtime=2,0,SimpleNamespace(physics_step=0)
    c.ages=torch.zeros(2,dtype=torch.long)
    c.reward,_,prior=start(config(),n=2)
    c.last=batch(prior)[1]
    c.model=deepcopy(model)
    c.trainer=DeviceDeformablePPOTrainer(c.model,PPOConfig(),rollout_steps=32,num_envs=2)
    c.observation=lambda: obs
    submitted=[]
    c.source=SimpleNamespace(set_previous_request=lambda actions,token:submitted.append((actions.clone(),token)))
    def step(actions,end):
        c.runtime.physics_step+=1
        k=c.runtime.physics_step
        c.last=batch([row(config(),i,k,failure=inject_failure and i==1 and k==3) for i in range(2)])[1]
        return c.reward.observe(c.last)
    c.sdk=step
    reset=[]
    c.reset=reset.append
    if inject_failure:
        with pytest.raises(ValueError,match="terminal lane"):
            c.window(8)
        assert c.runtime.physics_step==3 and len(c.trainer.rollout)==0 and not submitted
        return
    value=c.model.value(obs).detach()
    c.window(8,probe_reset=True)
    assert reset==[(1,)] and c.runtime.physics_step==8 and len(submitted)==1
    actual=c.trainer.rollout._rows[0]
    assert torch.equal(actual["actions"],submitted[0][0])
    assert torch.equal(actual["next_value"],value)
    assert actual["done"].tolist()==[False,True] and actual["bootstrap_mask"].all()



def test_flat_teacher_optimizer_rng_checkpoint_roundtrip(tmp_path,backend_profile,monkeypatch):
    import json
    from dataclasses import replace
    from test_dh116_tensor_native_inputs import native_fixture
    from test_dh116_device_ppo import collect
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import ROOT
    from go2_pvcnn.control.dh116.tabletop_tensor_backend_profile import TensorBackendProfile
    from go2_pvcnn.control.dh116.tabletop_tensor_native_teacher import TensorNativeTeacher
    from go2_pvcnn.control.dh116.tabletop_tensor_native_inputs import pack_shared_tensor_inputs
    from scripts.train_dh116_tensor_native_teacher import save_snapshot
    q,qd,kw,_,source=native_fixture(backend_profile,2)
    p=source.profile
    candidate=TensorBackendProfile(p.binary_path,p.binary_sha256,p.source_tree_sha256,p.patch_sha256,"d"*64,"fixture")
    monkeypatch.setattr(TensorBackendProfile,"verify",lambda self:self)
    cfg=json.loads((ROOT/"configs/dh116_training/privileged_deformable_teacher_pinch_v1.json").read_text())
    model=TensorNativeTeacher(cfg,backend_profile=candidate,num_envs=2,reward_config=config())
    kw["source_identity"]=replace(kw["source_identity"],profile_sha256=candidate.digest)
    kw["backend_profile_sha256"]=candidate.digest
    packet=pack_shared_tensor_inputs(q,qd,**kw)
    inputs=packet.for_teacher(model,current_identity=packet.source)
    trainer=DeviceDeformablePPOTrainer(model,PPOConfig(lr=1e-4,epochs=1,minibatches=1,target_kl=.02,
        kl_backtrack_steps=3),rollout_steps=32,num_envs=2)
    collect(trainer,inputs,count=32)
    assert trainer.update()["minibatch_updates"]>0
    path=tmp_path/"tensor.pt"
    state=torch.get_rng_state()
    digest=save_snapshot(path,trainer,SimpleNamespace(receipt_sha256="a"*64,policy_seed=21),candidate,{"sha256":"b"*64})
    assert len(digest)==64 and torch.equal(torch.get_rng_state(),state)
    payload=torch.load(path,weights_only=True)
    assert payload["counters"]["updates"]==1
    assert payload["model"]["checkpoint_namespace"]=="tensor_native_teacher_pinch_v1"
