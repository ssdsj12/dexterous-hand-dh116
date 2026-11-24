"""CPU execution proof for tensor-native checkpoint restart; no native/GPU admission."""
from copy import deepcopy
from dataclasses import replace
import importlib
import importlib.util
import json
import random
import os

import numpy as np
import pytest
import torch

from test_native_teacher_context import backend_profile
from test_dh116_device_ppo import collect, seed, state_equal
from test_dh116_tensor_native_inputs import native_fixture
from test_tensor_training_reward import config
from go2_pvcnn.control.dh116.tabletop_device_ppo import DeviceDeformablePPOTrainer
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import ROOT
from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
from go2_pvcnn.control.dh116.tabletop_tensor_backend_profile import TensorBackendProfile
from go2_pvcnn.control.dh116.tabletop_tensor_native_inputs import pack_shared_tensor_inputs
from go2_pvcnn.control.dh116.tabletop_tensor_native_teacher import TensorNativeTeacher
from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256

MODULE = "go2_pvcnn.control.dh116.tabletop_tensor_checkpoint"


def api():
    assert importlib.util.find_spec(MODULE) is not None, "tensor checkpoint resume module is missing"
    return importlib.import_module(MODULE)


@pytest.fixture
def tensor_data(backend_profile, monkeypatch):
    q, qd, kw, _, source = native_fixture(backend_profile, 2)
    p = source.profile
    profile = TensorBackendProfile(p.binary_path, p.binary_sha256, p.source_tree_sha256,
        p.patch_sha256, "d"*64, "fixture")
    monkeypatch.setattr(TensorBackendProfile, "verify", lambda self: self)
    cfg = json.loads((ROOT/"configs/dh116_training/privileged_deformable_teacher_pinch_v1.json").read_text())
    model = TensorNativeTeacher(cfg, backend_profile=profile, num_envs=2, reward_config=config())
    kw["source_identity"] = replace(kw["source_identity"], profile_sha256=profile.digest)
    kw["backend_profile_sha256"] = profile.digest
    packet = pack_shared_tensor_inputs(q, qd, **kw)
    inputs = packet.for_teacher(model, current_identity=packet.source)
    context = dict(recipe_sha256="a"*64, receipt_sha256="b"*64, backend_profile=profile,
        seed_sha256="c"*64, policy_seed=21)
    return model, inputs, context


def trainer_for(model):
    return DeviceDeformablePPOTrainer(deepcopy(model), PPOConfig(lr=1e-4, epochs=1,
        minibatches=1, target_kl=.02, kl_backtrack_steps=3), rollout_steps=3, num_envs=2)


def updated(tensor_data):
    model, obs, context = tensor_data
    trainer = trainer_for(model)
    seed(101)
    collect(trainer, obs)
    assert trainer.update()["minibatch_updates"] == 1
    return trainer, obs, context


def rng_state():
    a, b, c, d, e = np.random.get_state()
    return dict(torch=torch.get_rng_state().clone(), python=random.getstate(),
        numpy=(a, torch.from_numpy(b.astype(np.int64)), c, d, e))


def draw():
    return dict(torch=torch.rand(9), python=random.random(), numpy=torch.from_numpy(np.random.rand(9)))


def test_resume_rebuilds_session_rollout_and_next_real_update_matches_continuation(tmp_path, tensor_data):
    checkpoint = api()
    original, obs, context = updated(tensor_data)
    path = tmp_path/"tensor.pt"
    before_rng = rng_state()
    before_policy = deepcopy(original.model.state_dict())
    before_adam = deepcopy(original.optimizer.state_dict())
    old_rollout, old_session, old_capability = original.rollout, original.session_id, original._capability
    report = checkpoint.save_checkpoint(path, original, **context)
    state_equal(rng_state(), before_rng)
    state_equal(original.model.state_dict(), before_policy)
    state_equal(original.optimizer.state_dict(), before_adam)
    assert original.rollout is old_rollout and original.session_id == old_session
    assert original._capability is old_capability
    assert report["sha256"] == file_sha256(path)
    assert report["physical_state_saved"] is False and report["restart_requires_verified_reset"] is True
    expected_draw = draw()
    collect(original, obs)
    expected_batch = original.rollout.batch()
    expected_metrics = original.update()
    expected_rng = rng_state()
    # Deliberately consume every CPU stream and initialize a fresh trainer first.
    restored = trainer_for(tensor_data[0])
    seed(999)
    draw()
    destination_session = restored.session_id
    destination_rollout = restored.rollout
    destination_capability = restored._capability
    result = checkpoint.load_checkpoint(path, restored, expected_sha256=report["sha256"], **context)
    state_equal(restored.model.state_dict(), before_policy)
    state_equal(restored.optimizer.state_dict(), before_adam)
    state_equal(rng_state(), before_rng)
    assert restored.policy_version == 1 and restored.counters == {"environment_steps": 6, "updates": 1}
    assert restored.rollout is not destination_rollout and restored.session_id not in (destination_session, old_session)
    assert restored._capability is not destination_capability
    assert restored.rollout.session_id == restored.session_id
    assert restored.rollout.policy_version == restored.policy_version
    assert restored.rollout._capability is restored._capability
    assert restored.clean_boundary
    assert result["physical_state_saved"] is False and result["restart_requires_verified_reset"] is True
    state_equal(draw(), expected_draw)
    collect(restored, obs)
    state_equal(restored.rollout.batch(), expected_batch)
    state_equal(restored.update(), expected_metrics)
    state_equal(restored.model.state_dict(), original.model.state_dict())
    state_equal(restored.optimizer.state_dict(), original.optimizer.state_dict())
    state_equal(rng_state(), expected_rng)
    assert restored.policy_version == 2 and restored.counters == original.counters


@pytest.mark.parametrize("damage", ["namespace", "source", "counter", "coverage", "moment", "adam_setting",
    "rng", "physical_state"])
def test_invalid_packet_rejects_before_live_trainer_or_rng_mutation(tmp_path, tensor_data, damage):
    checkpoint = api()
    original, _, context = updated(tensor_data)
    path = tmp_path/"original.pt"
    checkpoint.save_checkpoint(path, original, **context)
    payload = torch.load(path, weights_only=True)
    if damage == "namespace": payload["identity"]["model"]["checkpoint_namespace"] = "tensor_native_teacher_power_v1"
    elif damage == "source": payload["identity"]["source_sha256"]["tabletop_device_ppo.py"] = "f"*64
    elif damage == "counter": payload["policy_version"] += 1
    elif damage == "coverage": payload["counters"]["environment_steps"] = 0
    elif damage == "moment": del payload["optimizer"]["state"][0]["exp_avg"]
    elif damage == "adam_setting": payload["optimizer"]["param_groups"][0]["amsgrad"] = True
    elif damage == "rng": payload["rng"]["torch"] = torch.zeros(2, dtype=torch.uint8)
    else: payload["identity"]["physical_state_saved"] = True
    bad = tmp_path/"bad.pt"
    torch.save(payload, bad)
    restored = trainer_for(tensor_data[0])
    policy, adam, rng = deepcopy(restored.model.state_dict()), deepcopy(restored.optimizer.state_dict()), rng_state()
    session, rollout, capability = restored.session_id, restored.rollout, restored._capability
    with pytest.raises(ValueError):
        checkpoint.load_checkpoint(bad, restored, expected_sha256=file_sha256(bad), **context)
    state_equal(restored.model.state_dict(), policy)
    state_equal(restored.optimizer.state_dict(), adam)
    state_equal(rng_state(), rng)
    assert restored.session_id == session and restored.rollout is rollout and restored._capability is capability


@pytest.mark.parametrize("key", ["recipe_sha256", "receipt_sha256", "seed_sha256", "policy_seed", "backend_profile"])
def test_explicit_context_mismatch_rejects(tmp_path, tensor_data, key):
    checkpoint = api()
    original, _, context = updated(tensor_data)
    path = tmp_path/"tensor.pt"
    report = checkpoint.save_checkpoint(path, original, **context)
    wrong = dict(context)
    wrong[key] = (replace(context[key], digest="e"*64) if key == "backend_profile"
        else context[key]+1 if key == "policy_seed" else "f"*64)
    with pytest.raises(ValueError):
        checkpoint.load_checkpoint(path, trainer_for(tensor_data[0]), expected_sha256=report["sha256"], **wrong)


def test_actual_power_model_cannot_load_pinch_namespace(tmp_path, tensor_data):
    checkpoint = api()
    original, _, context = updated(tensor_data)
    path = tmp_path/"pinch.pt"
    report = checkpoint.save_checkpoint(path, original, **context)
    cfg = json.loads((ROOT/"configs/dh116_training/privileged_deformable_teacher_power_v1.json").read_text())
    power = TensorNativeTeacher(cfg, backend_profile=context["backend_profile"], num_envs=2,
        reward_config=config("power"))
    with pytest.raises(ValueError, match="identity|namespace"):
        checkpoint.load_checkpoint(path, trainer_for(power), expected_sha256=report["sha256"], **context)


def test_hash_exclusive_path_and_clean_boundary_guards(tmp_path, tensor_data):
    checkpoint = api()
    original, obs, context = updated(tensor_data)
    path = tmp_path/"tensor.pt"
    report = checkpoint.save_checkpoint(path, original, **context)
    with pytest.raises(FileExistsError): checkpoint.save_checkpoint(path, original, **context)
    with pytest.raises(ValueError, match="SHA"):
        checkpoint.load_checkpoint(path, trainer_for(tensor_data[0]), expected_sha256="0"*64, **context)
    original.collect_step(obs, obs)
    with pytest.raises(RuntimeError, match="boundary"):
        checkpoint.save_checkpoint(tmp_path/"pending.pt", original, **context)
    with pytest.raises(RuntimeError, match="boundary"):
        checkpoint.load_checkpoint(path, original, expected_sha256=report["sha256"], **context)

@pytest.mark.skipif(os.environ.get("DH116_TENSOR_CHECKPOINT_CUDA")!="1",reason="explicit CUDA continuation test")
def test_cuda_rng_and_adam_continue_exactly_after_explicit_load(tmp_path,tensor_data):
    cp=api()
    model,obs,context=tensor_data
    device=torch.device("cuda:0")
    obs={k:v.to(device) for k,v in obs.items()}
    original=DeviceDeformablePPOTrainer(deepcopy(model),PPOConfig(lr=1e-4,epochs=1,minibatches=1,target_kl=.02,kl_backtrack_steps=3),
        rollout_steps=3,num_envs=2,device=device)
    seed(101);torch.cuda.manual_seed_all(101)
    collect(original,obs)
    assert original.update()["minibatch_updates"]==1
    path=tmp_path/"cuda.pt"
    before=torch.cuda.get_rng_state_all()
    report=cp.save_checkpoint(path,original,**context)
    assert all(torch.equal(a,b) for a,b in zip(before,torch.cuda.get_rng_state_all()))
    expected_draw=torch.rand(11,device=device)
    collect(original,obs);expected_batch=original.rollout.batch();expected_metrics=original.update()
    expected_state=deepcopy(original.model.state_dict());expected_adam=deepcopy(original.optimizer.state_dict())
    final_rng=torch.cuda.get_rng_state_all()
    restored=DeviceDeformablePPOTrainer(deepcopy(model),original.config,rollout_steps=3,num_envs=2,device=device)
    torch.cuda.manual_seed_all(999)
    cp.load_checkpoint(path,restored,expected_sha256=report["sha256"],**context)
    assert torch.equal(torch.rand(11,device=device),expected_draw)
    collect(restored,obs)
    state_equal(restored.rollout.batch(),expected_batch)
    state_equal(restored.update(),expected_metrics)
    state_equal(restored.model.state_dict(),expected_state)
    state_equal(restored.optimizer.state_dict(),expected_adam)
    assert all(torch.equal(a,b) for a,b in zip(final_rng,torch.cuda.get_rng_state_all()))
