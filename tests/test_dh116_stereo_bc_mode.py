"""BC/DAgger use training checkpointing without changing deployment mode."""
from copy import deepcopy

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_bc import StereoBCTrainer
from go2_pvcnn.control.dh116.tabletop_stereo_dagger_trainer import StereoDAggerTrainer
from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_stereo_actor
from go2_pvcnn.control.dh116.tabletop_stereo_checkpoint import save_actor_package,load_deployment_actor
from go2_pvcnn.control.dh116.tabletop_stereo_contract import make_contract
from go2_pvcnn.control.dh116.tabletop_stereo_supervision_dataset import batch_dataset
from test_dh116_stereo_training_cli import dataset

torch.set_num_threads(1)


def deployment_actor(tmp_path):
    network=dict(embed_dim=32,history=4,num_heads=4,num_layers=1,dropout=0.,
        image_resolution_wh=[32,32],image_processing_batch_size=2)
    actor=make_stereo_actor(network)
    path=tmp_path/'actor.pt'
    save_actor_package(path,actor,make_contract('bc',dict(network=network),calibration_sha256='c'*64),network_config=network)
    return load_deployment_actor(path)


@pytest.mark.parametrize('trainer_type',[StereoBCTrainer,StereoDAggerTrainer])
def test_deployment_warmstart_uses_chunk_checkpointing_for_supervised_update(tmp_path,monkeypatch,trainer_type):
    import torch.utils.checkpoint
    actor=deployment_actor(tmp_path)
    assert actor.training is False and actor.estimator.encoder.training is False
    modes=[module.training for module in actor.modules()]
    observed=[];calls=[]
    hook=actor.estimator.encoder.register_forward_pre_hook(lambda module,args:observed.append((module.training,torch.is_grad_enabled())))
    original=torch.utils.checkpoint.checkpoint
    def checkpoint(*args,**kwargs):
        calls.append(kwargs.get('use_reentrant'))
        return original(*args,**kwargs)
    monkeypatch.setattr(torch.utils.checkpoint,'checkpoint',checkpoint)
    trainer=trainer_type(actor,schema_only=True)
    before=actor.estimator.encoder.backbone.stem[0].weight.detach().clone()
    trainer.step(batch_dataset(dataset(),[0]))
    hook.remove()
    assert observed and all(training and grad for training,grad in observed)
    assert calls and all(value is False for value in calls)
    assert [module.training for module in actor.modules()]==modes
    assert not torch.equal(before,actor.estimator.encoder.backbone.stem[0].weight)


def test_failed_supervised_update_restores_exact_module_modes(tmp_path):
    actor=deployment_actor(tmp_path)
    actor.estimator.encoder.backbone.train()
    modes=[module.training for module in actor.modules()]
    batch=batch_dataset(dataset(),[0]);batch['teacher_actions']=deepcopy(batch['teacher_actions'])+.01
    with pytest.raises(ValueError,match='differs'):
        StereoBCTrainer(actor,schema_only=True).step(batch)
    assert [module.training for module in actor.modules()]==modes
