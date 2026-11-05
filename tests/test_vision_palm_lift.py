from types import SimpleNamespace
import json
import torch
from go2_pvcnn.control.dh116.tabletop_vision_training import run_vision_training
from test_vision_training import fixture,Camera


def test_visual_training_records_measured_hand_peaks_and_tensorboard_mm(tmp_path,monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_vision_training as training
    torch.set_num_threads(1)
    dw,task,env,args=fixture(tmp_path,maximum=1)
    enabled=[];tags={}
    env.enable_palm_lift_telemetry=lambda:enabled.append(True)
    original=env.step
    def step(action):
        obs,reward,done,extra=original(action)
        # True per-substep measured displacement; terminal/last values are lower.
        extra['palm_lift_telemetry']=dict(
            last_m=torch.tensor([[.001,.002],[.003,.004]]),
            peak_m=torch.tensor([[.011,.022],[.033,.044]]),
            valid=torch.ones(2,2,dtype=torch.bool))
        return obs,reward,done,extra
    env.step=step
    class Writer:
        def __init__(self,*unused):pass
        def add_scalar(self,name,value,iteration):tags[name]=float(value)
        def close(self):pass
    monkeypatch.setattr(training,'SummaryWriter',Writer)
    result=run_vision_training(SimpleNamespace(),dw,task,env,args,{'observation_size':371},camera=Camera(2))
    assert enabled==[True]
    row=json.loads((args.output/'monitor.json').read_text())
    pinch=row['grasp_styles']['pinch_thumb_index'];tripod=row['grasp_styles']['tripod_thumb_index_middle']
    assert abs(pinch['left_palm_lift_max_m']-.011)<1e-8
    assert abs(pinch['right_palm_lift_max_m']-.022)<1e-8
    assert abs(tripod['left_palm_lift_max_m']-.033)<1e-8
    assert abs(tripod['right_palm_lift_max_m']-.044)<1e-8
    assert pinch['left_palm_lift_max_env_id']==0 and tripod['right_palm_lift_max_env_id']==1
    assert abs(tags['lift_mm/pinch_thumb_index/left_palm_peak']-11.)<1e-5
    assert abs(tags['lift_mm/tripod_thumb_index_middle/right_palm_peak']-44.)<1e-5
    assert row['palm_lift_measurement']['baseline']=='measured_after_physical_episode_reset'
    saved=torch.load(args.output/'vision_student_ppo.pt',weights_only=False)
    assert saved['contract']['actor_observation_size']==3844
    assert 'palm_lift_measurement' not in saved['contract']


def test_visual_legacy_env_does_not_invent_unmeasured_hand_lift(tmp_path):
    torch.set_num_threads(1)
    dw,task,env,args=fixture(tmp_path,maximum=1)
    run_vision_training(SimpleNamespace(),dw,task,env,args,{'observation_size':371},camera=Camera(2))
    row=json.loads((args.output/'monitor.json').read_text())
    assert 'left_palm_lift_max_m' not in row['grasp_styles']['pinch_thumb_index']
