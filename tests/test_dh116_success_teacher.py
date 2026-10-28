import pytest
import torch
from go2_pvcnn.control.dh116.tabletop_success_teacher import TeacherController

def controller(**changes):
    case=dict(name='test',contact_z_m=-.002,force_n=[2.,12.],joint_offset_rad=[0.]*12)
    case.update(changes)
    return TeacherController([case],torch.zeros(12),-torch.ones(12),torch.zeros(12),torch.ones(12),'cpu')

def test_policy_actions_bounded_and_synchronized():
    c=controller()
    a=c.actions({})
    assert a.shape==(1,20) and a.abs().max()<=1
    assert a[0,14]==a[0,17] and a[0,14]<0
    assert a[0,18:].tolist()==[-1.,1.]

def test_actual_contact_feedback_moves_shared_z_only():
    c=controller(); before=c.actions({}).clone()
    m={'contact_valid':torch.ones(1,2,dtype=torch.bool),'contact_center_relative_m':torch.zeros(1,2,3)}
    after=c.actions(m)
    assert after[0,14]<before[0,14] and after[0,14]==after[0,17]
    assert torch.equal(after[:,:12],before[:,:12])

def test_missing_contacts_no_fake_feedback():
    c=controller(); before=c.actions({}).clone()
    m={'contact_valid':torch.zeros(1,2,dtype=torch.bool),'contact_center_relative_m':torch.full((1,2,3),float('nan'))}
    assert torch.equal(c.actions(m),before)

@pytest.mark.parametrize('change',[{'force_n':[1.,4.]},{'contact_z_m':.001},{'xy_m':[[.001,0],[0,0]]},{'joint_offset_rad':[float('nan')]*12}])
def test_invalid_case(change):
    with pytest.raises(ValueError):controller(**change)

def test_reset_restores_initial_search():
    c=controller(); initial=c.actions({}).clone()
    c.actions({'contact_valid':torch.ones(1,2,dtype=torch.bool),'contact_center_relative_m':torch.zeros(1,2,3)})
    c.reset(torch.tensor([True])); assert torch.equal(initial,c.actions({}))

def test_double_physics_feedback_keeps_policy_dtype_and_reset():
    c=controller()
    c.actions({'contact_valid':torch.ones(1,2,dtype=torch.bool),'contact_center_relative_m':torch.zeros(1,2,3,dtype=torch.float64)})
    c.reset(torch.tensor([True]));assert c.z.dtype==torch.float32

def test_no_lift_summary_is_json_serializable():
    import json
    from go2_pvcnn.control.dh116.tabletop_control_probe import phase3_lift_summary
    result=phase3_lift_summary([0.],[1],[False],[0.],[0.])
    json.dumps(result,allow_nan=False)
    assert result['mean_lift_force_n'] is None
