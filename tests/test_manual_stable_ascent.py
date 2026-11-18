import pytest
import torch
from test_force_region_integration import constrained


@pytest.mark.parametrize('error,allow',[(.013,True),(.0199,True),(.02,False)])
def test_manual_stable_grip_can_ascend_inside_existing_hard_mimic_bound(error,allow):
    t=constrained();t.manual_stable_grip_ascent=True
    t.m['mimic_errors'].fill_(error)
    a=torch.zeros((2,20),dtype=torch.float64);a[:,14]=.2;a[:,17]=.2
    assert bool(t.control(a,~t.done,1/480).all())==allow


@pytest.mark.parametrize('kind',['ordinary_policy','horizontal','descending','overforce'])
def test_manual_ascent_exception_does_not_exempt_other_requests(kind):
    t=constrained();t.manual_stable_grip_ascent=kind!='ordinary_policy'
    t.m['mimic_errors'].fill_(.013)
    a=torch.zeros((2,20),dtype=torch.float64);a[:,14]=.2;a[:,17]=.2
    if kind=='horizontal':a[:,12]=1.
    if kind=='descending':a[:,14]=-.5
    if kind=='overforce':t.m['hand_object_loads'].fill_(13.)
    assert not t.control(a,~t.done,1/480).any()
