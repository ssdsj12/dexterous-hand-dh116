import os
from pathlib import Path
import runpy
import sys


def test_depth_probe_routes_to_camera_enabled_native_entry(tmp_path, monkeypatch):
    root=Path(__file__).resolve().parents[1]
    script=root/'scripts/run_dh116_surface_student.py'
    configs=root/'configs/dh116_training'
    calls=[]
    monkeypatch.setattr(sys,'argv',[str(script),'--output',str(tmp_path/'capture'),
        '--num-envs','2','--updates','1','--surface-feedback',
        '--imitation-mode','stable_hold_v2','--noise-mode','holding_v2',
        '--demonstrations',str(tmp_path/'demo.pt'),
        '--depth-coordinate-probe','--depth-camera-config',str(configs/'front_depth_camera_v2.json'),
        '--pregrasp-candidates',str(configs/'effective_pad_center_pregrasp_v1.json'),
        '--mixed-pregrasp-candidates',str(configs/'effective_pad_triad_pregrasp_v1.json')])
    monkeypatch.setattr(os,'execv',lambda executable,args:calls.append(args))
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(script),run_name='__main__')
    cmd=calls[0]
    assert '--depth-coordinate-probe' in cmd
    assert '--enable_cameras' in cmd
    assert Path(cmd[cmd.index('--depth-camera-config')+1]).name=='front_depth_camera_v2.json'
    assert '--resume' not in cmd
