"""Independent measured-common checkpoint boundaries."""
import random

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_flexible_measured_checkpoint import load_checkpoint, save_checkpoint
from go2_pvcnn.control.dh116.tabletop_flexible_measured_common_teacher import MeasuredCommonTeacher
from go2_pvcnn.control.dh116.tabletop_deformable_privileged_ppo import DeformablePPOTrainer
from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
from test_dh116_flexible_measured_common_teacher import setup
from test_dh116_deformable_privileged_ppo import make_trainer, populate, same


def measured(grasp="pinch"):
    cfg, base, ledger, valid, rod, clock, mask = setup(grasp)
    model = MeasuredCommonTeacher(cfg,base)
    obs = model.pack_measured(ledger,valid,rod,clock,mask)
    return DeformablePPOTrainer(model,PPOConfig(epochs=1,minibatches=1),
                                rollout_steps=2,num_envs=2), obs


def rng():
    return (torch.get_rng_state(), np.random.get_state(), random.getstate())


def test_continuation_preserves_policy_adam_counters_and_rng(tmp_path):
    source, obs = measured()
    target, _ = measured()
    populate(source, obs)
    source.update()
    path = tmp_path/"test.pt"
    receipt = save_checkpoint(path,source)
    assert len(receipt["sha256"]) == 64
    previous = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_checkpoint(path,source)
    assert path.read_bytes() == previous
    saved_rng = rng()
    populate(source, obs)
    expected = source.update()
    load_checkpoint(path,target)
    assert same(rng(),saved_rng)
    populate(target,obs)
    assert target.update() == expected
    assert same(source.model.state_dict(),target.model.state_dict())
    assert same(source.optimizer.state_dict(),target.optimizer.state_dict())
    assert source.counters == target.counters


def test_cross_style_and_existing_teacher_rejected_before_mutation(tmp_path):
    original, _ = measured()
    other, _ = measured("tripod")
    path = tmp_path/"pinch.pt"
    save_checkpoint(path,original)
    before = rng()
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(path,other)
    assert same(before,rng()) and other.counters["updates"] == 0
    legacy, _ = make_trainer()
    with pytest.raises(ValueError,match="measured"):
        load_checkpoint(path,legacy)


def test_malformed_checkpoint_rejected_before_mutation(tmp_path):
    trainer, obs = measured()
    path = tmp_path/"original.pt"
    save_checkpoint(path,trainer)
    packet = torch.load(path,weights_only=True)
    packet["counters"]["environment_steps"] = 3
    tampered = tmp_path/"tampered.pt"
    torch.save(packet,tampered)
    before = rng()
    with pytest.raises(ValueError,match="rollout"):
        load_checkpoint(tampered,trainer)
    assert same(before,rng()) and trainer.counters["environment_steps"] == 0
    trainer.collect_step(obs,obs)
    with pytest.raises(RuntimeError,match="clean"):
        save_checkpoint(tmp_path/"pending.pt",trainer)
