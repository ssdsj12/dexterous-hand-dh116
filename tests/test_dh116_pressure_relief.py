import torch
import json
import numpy as np
from go2_pvcnn.control.dh116.tabletop_pressure_relief import pressure_limited_target


def test_overload_overrides_positive_policy_target_without_affecting_other_hand():
    command=torch.ones((3,2,6),dtype=torch.float64)*.4
    desired=command+.2
    loads=torch.tensor([[20.,0.],[.12,.12],[0.,0.]])
    target=torch.full((3,2),.1)
    direction=torch.tensor([[[.2,.6,.77,0,0,0]]*2]*3)
    actual,active=pressure_limited_target(command,desired,loads,target,direction,1/480,.02)
    assert active.tolist()==[[True,False],[False,False],[False,False]]
    assert (actual[0,0,:3]<command[0,0,:3]).all()
    assert (actual[0,0,3:]==command[0,0,3:]).all()
    assert torch.equal(actual[0,1],desired[0,1])
    assert torch.equal(actual[1:],desired[1:])


def test_existing_opening_is_preserved_and_reversed_closure_direction_supported():
    command=torch.ones((1,2,6),dtype=torch.float64)*.4
    direction=torch.tensor([[[-.2,.6,.77,0,0,0]]*2])
    sign=torch.where(direction==0,1,direction.sign())
    desired=command-.1*sign
    actual,_=pressure_limited_target(command,desired,torch.full((1,2),20.),torch.ones((1,2)),direction,1/480,.02)
    assert torch.equal(actual,desired)


def test_actual_controller_saturated_feedback_cannot_keep_closing_under_load():
    from test_dh116_tensor_task import template
    from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
    from pathlib import Path
    profile=json.loads((Path(__file__).parents[1]/'configs/dh116_joint_envelopes/free_grasp_geometry_v1.json').read_text())
    e=template();e.joint_envelope=profile
    e.reference=np.array(profile['center_rad']);e.command=e.reference.copy();e.previous=e.command.copy();e.measurement['q']=e.command.copy()
    t=TensorTask(e,2,'cpu',pressure_relief=True)
    t.f_offset[:]=-.08;t.m['hand_object_loads'][0]=20.
    t.m['hand_object_loads'][1]=0.
    a=torch.ones((2,20),dtype=torch.float64);before=t.command.clone()
    allowed=t.control(a,~t.done,1/480)
    assert (t.command[0]<=before[0]).all() and (t.command[0]<before[0]).any()
    assert (t.command[1]>before[1]).all()
    assert allowed.tolist()==[False,True]
    assert (abs(t.command-before)<=.20/480+1e-12).all()
    assert (t.command>=t.lower).all() and (t.command<=t.upper).all()
    t.reset(torch.tensor([0]),t.m)
    assert not t.pressure_limited[0].any()


def test_overload_pauses_controller_lift_as_well_as_adaptive_palm_motion():
    from test_dh116_tensor_task import template
    from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
    t=TensorTask(template(),2,'cpu',pressure_relief=True)
    t.phase[:]=3;t.m['hand_object_loads'][0]=2.
    t.m['hand_object_loads'][1]=.11
    before=t.offset.clone()
    allowed=t.control(torch.zeros((2,20),dtype=torch.float64),~t.done,1/480)
    assert t.offset[0]==before[0]
    assert t.offset[1]>before[1]
    assert not allowed[0] and allowed[1]
