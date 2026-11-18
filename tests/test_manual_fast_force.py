import pytest
import torch
from test_force_region_integration import constrained


@pytest.mark.parametrize('load,valid,below',[(0.,False,False),(.2,True,False),(.2,True,True)])
def test_manual_closing_can_find_contact_without_changing_palm(load,valid,below):
    from go2_pvcnn.control.dh116.tabletop_policy_force_region import adjust
    t=constrained();t.manual_fast_force=True
    t.m['hand_object_loads'].fill_(load);t.m['contact_valid'].fill_(valid)
    t.m['contact_all_below_centerline'].fill_(below)
    target=t.command.clone();palms=t.palm_targets.clone()
    out=adjust(t,target,torch.full((2,2),5.),torch.tensor([True,False]),.01)
    torch.testing.assert_close(t.f_offset[0],torch.full_like(t.f_offset[0],.0012))
    assert not t.f_offset[1].any()
    torch.testing.assert_close(out[0]-target[0],(.0012*t.direction[0]).flatten())
    torch.testing.assert_close(t.palm_targets,palms)


def test_manual_fast_force_stops_closing_at_target_and_releases_overforce():
    from go2_pvcnn.control.dh116.tabletop_policy_force_region import adjust
    t=constrained();t.manual_fast_force=True;t.m['hand_object_loads'].fill_(5.)
    adjust(t,t.command,torch.full((2,2),5.),~t.done,.01)
    assert not t.f_offset.any()
    t.m['hand_object_loads'].fill_(13.)
    adjust(t,t.command,torch.full((2,2),5.),~t.done,.01)
    assert (t.f_offset<0).all()


def test_severe_mimic_still_blocks_tightening_and_closure_is_bounded():
    from go2_pvcnn.control.dh116.tabletop_policy_force_region import adjust
    t=constrained();t.manual_fast_force=True;t.m['hand_object_loads'].zero_()
    t.m['mimic_errors'].fill_(.013)
    adjust(t,t.command,torch.full((2,2),5.),~t.done,.01)
    assert not t.f_offset.any()
    t.m['mimic_errors'].zero_();t.f_offset.fill_(.0799)
    adjust(t,t.command,torch.full((2,2),5.),~t.done,.01)
    assert (t.f_offset<=.08).all()
