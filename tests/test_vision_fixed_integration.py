from pathlib import Path
from types import SimpleNamespace
import json,os,runpy,sys
import pytest
import torch
from go2_pvcnn.control.dh116.tabletop_vision_training import _contract,run_vision_training


def test_fixed_contract_records_one_stationary_camera_per_environment(tmp_path):
    camera=tmp_path/'camera.json';camera.write_text('{}')
    seed=tmp_path/'seed.json';seed.write_text('{}')
    args=SimpleNamespace(num_envs=1024,rollout_steps=32,vision_width=320,vision_height=240,
        depth_camera_config=camera,grasp_initialization=seed,mixed_grasp_initialization=seed,
        vision_camera_mode='fixed',vision_camera_batch_size=64,vision_visibility_mode='all')
    c=_contract(args,{'observation_size':371})
    assert c['camera_mode']=='fixed_per_env'
    assert c['camera_batch_size']==1024
    assert c['camera_batching']=='fixed_per_environment_real_RGBD'
    assert c['visibility_mode']=='all' and c['pinned_source_environment'] is None
    args.vision_camera_mode='bounded'
    assert 'camera_mode' not in _contract(args,{'observation_size':371})


@pytest.mark.parametrize('n',[1024,2048])
def test_fixed_launcher_forwards_mode_resume_and_native_view_limit(tmp_path,monkeypatch,n):
    root=Path(__file__).resolve().parents[1];cfg=root/'configs/dh116_training'
    script=root/'scripts/run_dh116_surface_student.py';calls=[]
    monkeypatch.setattr(sys,'argv',[str(script),'--output',str(tmp_path/'fixed'),
        '--num-envs',str(n),'--updates','2048','--vision-student','--surface-feedback',
        '--depth-camera-config',str(cfg/'front_depth_camera_v2.json'),
        '--mixed-pregrasp-candidates',str(cfg/'effective_pad_triad_pregrasp_v1.json'),
        '--vision-camera-mode','fixed','--physics-clone-mode','usd_parse',
        '--resume',str(tmp_path/'old.pt'),'--vision-resume-downsize'])
    monkeypatch.setattr(os,'execv',lambda program,cmd:calls.append(cmd))
    runpy.run_path(str(script),run_name='__main__')
    cmd=calls[0]
    assert cmd[cmd.index('--vision-camera-mode')+1]=='fixed'
    assert '--vision-resume-downsize' in cmd
    assert cmd[cmd.index('--physics-clone-mode')+1]=='usd_parse'
    kit=next(x for x in cmd if x.startswith('--kit_args='))
    assert f'--/rtx/viewTile/limit={n}' in kit


def test_fixed_launcher_rejects_4096_before_native_launch(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1];cfg=root/'configs/dh116_training'
    script=root/'scripts/run_dh116_surface_student.py';calls=[]
    monkeypatch.setattr(sys,'argv',[str(script),'--output',str(tmp_path/'too_many'),
        '--num-envs','4096','--vision-student','--surface-feedback',
        '--depth-camera-config',str(cfg/'front_depth_camera_v2.json'),
        '--mixed-pregrasp-candidates',str(cfg/'effective_pad_triad_pregrasp_v1.json'),
        '--vision-camera-mode','fixed'])
    monkeypatch.setattr(os,'execv',lambda program,cmd:calls.append(cmd))
    with pytest.raises(SystemExit):runpy.run_path(str(script),run_name='__main__')
    assert not calls


def test_training_uses_fixed_camera_and_keeps_atomic_checkpoint(tmp_path,monkeypatch):
    from test_vision_training import fixture,Camera
    from go2_pvcnn.control.dh116 import tabletop_vision_fixed_camera as fixed
    from go2_pvcnn.control.dh116 import tabletop_vision_audit as audit
    from go2_pvcnn.control.dh116 import tabletop_vision_memory as memory
    torch.set_num_threads(1);dw,task,env,args=fixture(tmp_path,maximum=1)
    args.vision_camera_mode='fixed';calls=[]
    def provider(world,state,config,**kwargs):
        calls.append(kwargs);assert state is dw
        return Camera(2)
    monkeypatch.setattr(fixed,'FixedVisionCamera',provider)
    monkeypatch.setattr(memory,'memory_snapshot',lambda *unused,**kw:None)
    monkeypatch.setattr(audit,'audit_initial_points',lambda *unused,**kw:dict(passed=True))
    result=run_vision_training(SimpleNamespace(),dw,task,env,args,{'observation_size':371})
    assert result['iteration']==1 and len(calls)==1
    saved=torch.load(args.output/'vision_student_ppo.pt',weights_only=False)
    assert saved['contract']['camera_mode']=='fixed_per_env'
