from dataclasses import replace
import json
import pytest
import torch
from test_native_teacher_context import backend_profile
from test_dh116_tensor_native_inputs import native_fixture
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import ROOT
from go2_pvcnn.control.dh116.tabletop_training_reward import load_config

def test_tensor_identity_keeps_original_network_and_isolates_checkpoint(backend_profile,monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_tensor_native_teacher import TensorNativeTeacher
    from go2_pvcnn.control.dh116.tabletop_tensor_backend_profile import TensorBackendProfile
    from go2_pvcnn.control.dh116.tabletop_shared_native_teacher import SharedNativeTeacher
    from go2_pvcnn.control.dh116.tabletop_tensor_native_inputs import pack_shared_tensor_inputs
    q,qd,kw,oracle,source=native_fixture(backend_profile,2)
    p=source.profile
    candidate=TensorBackendProfile(p.binary_path,p.binary_sha256,p.source_tree_sha256,
        p.patch_sha256,"d"*64,"fixture-only")
    monkeypatch.setattr(TensorBackendProfile,"verify",lambda self:self)
    cfg=json.loads((ROOT/"configs/dh116_training/privileged_deformable_teacher_pinch_v1.json").read_text())
    reward=load_config(ROOT/"configs/dh116_training/pinch_finger_goal_reward_v1.json",ppo_gamma=.995)
    torch.manual_seed(18);old=SharedNativeTeacher(cfg,backend_profile=p,num_envs=2,reward_config=reward)
    torch.manual_seed(18);new=TensorNativeTeacher(cfg,backend_profile=candidate,num_envs=2,reward_config=reward)
    kw["source_identity"]=replace(kw["source_identity"],profile_sha256=candidate.digest)
    kw["backend_profile_sha256"]=candidate.digest
    packet=pack_shared_tensor_inputs(q,qd,**kw)
    inputs=packet.for_teacher(new,current_identity=packet.source)
    torch.testing.assert_close(new.latent_mean(inputs),old.latent_mean(inputs),rtol=0,atol=0)
    torch.testing.assert_close(new.value(inputs),old.value(inputs),rtol=0,atol=0)
    assert new.model_identity["checkpoint_namespace"]!=old.model_identity["checkpoint_namespace"]
    with pytest.raises(ValueError):new.load_state_dict(old.state_dict())
    assert not new.model_identity["teacher_admission"]
    with pytest.raises(ValueError):TensorNativeTeacher(cfg,backend_profile=p,num_envs=2,reward_config=reward)
