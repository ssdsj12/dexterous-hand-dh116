import os
from pathlib import Path
import runpy,sys
import pytest


def test_4096_launch_forwards_bounded_memory_controls(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    script=root/'scripts/run_dh116_surface_student.py'
    cfg=root/'configs/dh116_training'
    calls=[]
    monkeypatch.setattr(sys,'argv',[str(script),'--output',str(tmp_path/'run'),
        '--num-envs','4096','--updates','2048','--vision-student','--surface-feedback',
        '--depth-camera-config',str(cfg/'front_depth_camera_v2.json'),
        '--pregrasp-candidates',str(cfg/'effective_pad_center_pregrasp_v1.json'),
        '--mixed-pregrasp-candidates',str(cfg/'effective_pad_triad_pregrasp_v1.json'),
        '--vision-camera-batch-size','32','--vision-ppo-microbatch-size','128',
        '--vision-inference-batch-size','64','--vision-visibility-mode','all'])
    monkeypatch.setattr(os,'execv',lambda program,cmd:calls.append(cmd))
    runpy.run_path(str(script),run_name='__main__')
    cmd=calls[0]
    for flag,value in [('--num-envs','4096'),('--max-iterations','2048'),
        ('--vision-camera-batch-size','32'),('--vision-ppo-microbatch-size','128'),
        ('--vision-inference-batch-size','64'),('--vision-visibility-mode','all')]:
        assert cmd[cmd.index(flag)+1]==value
    assert '--enable_cameras' in cmd
    assert '--surface-demonstrations' not in cmd
    assert '--surface-student-actor' not in cmd
