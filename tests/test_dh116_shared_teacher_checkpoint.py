"""Independent shared checkpoints keep original policy/Adam/RNG and fresh Reset."""
from copy import deepcopy
import pytest
import torch
from test_native_teacher_context import backend_profile
from test_native_teacher_model import cpu_model
from test_dh116_shared_teacher_source import reader
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import ROOT
from go2_pvcnn.control.dh116.tabletop_training_reward import load_config


def make(profile,device="cpu"):
    from go2_pvcnn.control.dh116.tabletop_shared_native_teacher import SharedNativeTeacher
    from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
    source,_,common,context=reader(profile)
    _,*_,config,_=cpu_model(profile)
    reward=load_config(ROOT/"configs/dh116_training/pinch_finger_goal_reward_v1.json",ppo_gamma=.995)
    model=SharedNativeTeacher(config,backend_profile=source.profile,num_envs=2,reward_config=reward)
    obs=model.pack_shared(source.capture(common,context),source)
    trainer=DeformablePPOTrainer(model,PPOConfig(epochs=1,minibatches=1,lr=1e-5),
        rollout_steps=2,num_envs=2,device=device)
    return trainer,obs


@pytest.mark.parametrize("device",["cpu","cuda:0"])
def test_shared_resume_restores_original_adam_rng_and_rejects_other_recipe(backend_profile,tmp_path,device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("actual CUDA RNG required")
    from go2_pvcnn.control.dh116.tabletop_shared_teacher_checkpoint import save_checkpoint,load_checkpoint
    trainer,obs=make(backend_profile,device)
    for _ in range(2):
        sample=trainer.collect_step(obs,obs)
        trainer.record_step(torch.tensor([1.,2.]),torch.ones(2,dtype=torch.bool),torch.zeros(2),
            executed_actions=sample["actions"],collection_token=sample["collection_token"])
    assert trainer.update()["minibatch_updates"]==1
    path=tmp_path/"shared.pt"
    saved=save_checkpoint(path,trainer,recipe_sha256="a"*64)
    expected=trainer.collect_step(obs,obs)
    restored,_=make(backend_profile,device)
    before=deepcopy(restored.model.state_dict())
    with pytest.raises(ValueError):
        load_checkpoint(path,restored,recipe_sha256="b"*64,expected_sha256=saved["sha256"])
    assert torch.equal(restored.model.actor_head.weight,before["actor_head.weight"])
    result=load_checkpoint(path,restored,recipe_sha256="a"*64,expected_sha256=saved["sha256"])
    actual=restored.collect_step(obs,obs)
    assert torch.equal(actual["latent_actions"],expected["latent_actions"])
    assert torch.equal(actual["log_prob"],expected["log_prob"])
    assert result["restart_requires_verified_reset"] is True
    assert restored.counters==trainer.counters
    assert restored.optimizer.state_dict()["state"].keys()==trainer.optimizer.state_dict()["state"].keys()
    from go2_pvcnn.control.dh116.tabletop_shared_teacher_checkpoint import verify_roundtrip
    verify_roundtrip(trainer,restored)
    first=next(iter(restored.optimizer.state.values()))
    first["exp_avg"].view(-1)[0]+=1.
    with pytest.raises(ValueError,match="optimizer"):
        verify_roundtrip(trainer,restored)
    with pytest.raises(FileExistsError):
        # restored currently has a pending transition, so use a clean new instance.
        clean,_=make(backend_profile,device)
        save_checkpoint(path,clean,recipe_sha256="a"*64)
