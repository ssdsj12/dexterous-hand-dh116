"""Explicit model-adapter identity; old raw actors must never change silently."""
from copy import deepcopy

import pytest
import torch

MODE='wrist_quaternion_hemisphere_v1'
FIELD='actor_wrist_quaternion_preprocessing'


def test_contract_records_only_explicit_versioned_adapter(tmp_path):
    from test_vision_training import fixture
    from go2_pvcnn.control.dh116.tabletop_vision_training import _contract
    _,_,_,args=fixture(tmp_path)
    args.vision_action_distribution='tanh_gaussian'
    old=_contract(args,{'observation_size':371})
    assert FIELD not in old
    args.vision_wrist_quaternion_preprocessing=MODE
    current=_contract(args,{'observation_size':371})
    assert current[FIELD]==MODE
    assert {k:v for k,v in current.items() if k!=FIELD}==old
    args.vision_action_distribution='gaussian_clip'
    with pytest.raises(ValueError,match='quaternion|tanh'):
        _contract(args,{'observation_size':371})


def test_unknown_adapter_contract_is_rejected(tmp_path):
    from test_vision_action_derivation import source_checkpoint
    from go2_pvcnn.control.dh116.tabletop_vision_resume import _validate_contract
    _,saved=source_checkpoint(tmp_path)
    saved['contract'][FIELD]='invented'
    with pytest.raises(ValueError,match='quaternion'):
        _validate_contract(saved['contract'])


def test_adapter_mode_is_checked_before_kit_and_static_resume(tmp_path):
    from types import SimpleNamespace
    from test_vision_action_derivation import source_checkpoint
    from go2_pvcnn.control.dh116.tabletop_vision_action_derivation import derive_action_checkpoint
    from go2_pvcnn.control.dh116.tabletop_vision_action_contract import validate_vision_action_options
    with pytest.raises(ValueError,match='quaternion'):
        validate_vision_action_options(SimpleNamespace(vision_wrist_quaternion_preprocessing=MODE))
    source,_=source_checkpoint(tmp_path);target=tmp_path/'canonical.pt'
    derive_action_checkpoint(source,target)
    saved=torch.load(target,weights_only=False);saved['contract'][FIELD]=MODE
    torch.save(saved,target)
    with pytest.raises(ValueError,match='quaternion'):
        validate_vision_action_options(SimpleNamespace(vision_action_distribution='tanh_gaussian',resume=target))


def test_canonical_inference_projects_batch_but_rejects_raw_contract(tmp_path):
    from test_vision_action_derivation import source_checkpoint,observations
    from go2_pvcnn.control.dh116.tabletop_vision_action_derivation import derive_action_checkpoint
    from go2_pvcnn.control.dh116.tabletop_vision_inference import load_vision_inference
    from go2_pvcnn.control.dh116.tabletop_vision_evaluation_contract import evaluation_contract
    source,_=source_checkpoint(tmp_path);target=tmp_path/'canonical.pt'
    derive_action_checkpoint(source,target)
    saved=torch.load(target,weights_only=False);saved['contract'][FIELD]=MODE
    torch.save(saved,target)
    current=evaluation_contract(saved['contract'],4)
    assert current[FIELD]==MODE
    actor=load_vision_inference(target,current)
    obs=observations()['policy']
    a=obs.clone();b=obs.clone()
    for frame in range(4):
        a[:,frame*961+923:frame*961+927]=torch.tensor([.1,.2,.3,.4])
        b[:,frame*961+923:frame*961+927]=-torch.tensor([.1,.2,.3,.4])
    assert torch.equal(actor.act(a),actor.act(b))
    bad=deepcopy(current);bad.pop(FIELD)
    with pytest.raises(ValueError,match='contract'):
        load_vision_inference(target,bad)


def test_canonical_ppo_factory_uses_adapter_without_extra_parameters():
    from test_vision_action_derivation import observations
    from go2_pvcnn.control.dh116.tabletop_vision_squashed_policy import make_squashed_vision_algorithm
    torch.set_num_threads(1)
    algorithm=make_squashed_vision_algorithm(observations(),2,
        wrist_quaternion_preprocessing=MODE)
    actor=algorithm.policy.actor
    assert actor.wrist_quaternion_preprocessing==MODE
    assert len(list(actor.state_dict()))==12


def test_new_adapter_checkpoint_resume_requires_explicit_same_mode(tmp_path):
    from types import SimpleNamespace
    from test_vision_training import fixture,Camera
    from go2_pvcnn.control.dh116.tabletop_vision_training import run_vision_training
    torch.set_num_threads(1)
    dw,task,env,args=fixture(tmp_path,maximum=1)
    args.vision_action_distribution='tanh_gaussian'
    args.vision_wrist_quaternion_preprocessing=MODE
    run_vision_training(SimpleNamespace(),dw,task,env,args,{'observation_size':371},camera=Camera(2))
    path=args.output/'vision_student_ppo.pt'
    assert torch.load(path,weights_only=False)['contract'][FIELD]==MODE
    args.resume=path;args.output=tmp_path/'resume';args.max_iterations=2
    args.vision_wrist_quaternion_preprocessing='raw'
    with pytest.raises(ValueError,match='contract'):
        run_vision_training(SimpleNamespace(),dw,task,env,args,{'observation_size':371},camera=Camera(2))


def test_front_launcher_forwards_explicit_adapter_without_starting_isaac(tmp_path,monkeypatch):
    import os,runpy,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1];script=root/'scripts/run_dh116_surface_student.py'
    cfg=root/'configs/dh116_training';calls=[]
    monkeypatch.setattr(sys,'argv',[str(script),'--output',str(tmp_path/'front'),
        '--vision-student','--surface-feedback','--vision-action-distribution','tanh_gaussian',
        '--vision-wrist-quaternion-preprocessing',MODE,
        '--depth-camera-config',str(cfg/'front_depth_camera_v2.json'),
        '--pregrasp-candidates',str(cfg/'effective_pad_center_pregrasp_v1.json'),
        '--mixed-pregrasp-candidates',str(cfg/'effective_pad_triad_pregrasp_v1.json')])
    monkeypatch.setattr(os,'execv',lambda program,cmd:calls.append(cmd))
    runpy.run_path(str(script),run_name='__main__')
    assert len(calls)==1
    command=calls[0]
    assert command[command.index('--vision-wrist-quaternion-preprocessing')+1]==MODE
