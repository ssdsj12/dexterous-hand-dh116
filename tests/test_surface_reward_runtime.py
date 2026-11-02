import pytest
import torch
from go2_pvcnn.control.dh116.tabletop_surface_reward_ablation import RewardAblation


def step(r, *, bottom=.36, qualified=True, early=True, duration=0., drift=0.,
         relative=0., support=1.2, weight=1., dt=1/480):
    x=lambda v:torch.tensor([v],dtype=torch.float64)
    return r.update(bottom=x(bottom),qualified=torch.tensor([qualified]),
        early_window_good=torch.tensor([early]),early_duration=x(duration),
        drift=x(drift),relative_speed=x(relative),support=x(support),weight=x(weight),
        valid=torch.tensor([True]),dt=dt)


def test_height_progress_is_qualified_new_height_and_capped_at_eight():
    r=RewardAblation('lift_progress',torch.tensor([.36],dtype=torch.float64))
    assert step(r,bottom=.39)['surface_lift_progress'].item()==pytest.approx(8*(.0295/.03))
    assert step(r,bottom=.37)['surface_lift_progress'].item()==0
    assert step(r,bottom=.3905)['surface_lift_progress'].item()==pytest.approx(8*(.0005/.03))
    r.reset(torch.tensor([0]),torch.tensor([.36],dtype=torch.float64))
    assert step(r,bottom=.39,qualified=False)['surface_lift_progress'].item()==0


def test_hold_rate_and_milestones_are_early_and_one_shot():
    r=RewardAblation('early_hold',torch.tensor([.36],dtype=torch.float64))
    first=step(r,bottom=.364,duration=.1)
    assert first['surface_early_hold_rate'].item()==pytest.approx(2/480)
    assert first['surface_hold_milestone'].item()==pytest.approx(1.)
    assert step(r,bottom=.364,duration=.1)['surface_hold_milestone'].item()==0
    assert step(r,bottom=.364,duration=.3)['surface_hold_milestone'].item()==pytest.approx(2.)


def test_slip_and_relative_motion_use_real_valid_three_mm_window():
    r=RewardAblation('slip_relative',torch.tensor([.36],dtype=torch.float64))
    out=step(r,bottom=.364,drift=.001,relative=.004)
    assert out['surface_slip_window'].item()<0
    assert out['surface_relative_motion'].item()<0
    out=step(r,bottom=.364,early=False,drift=.001,relative=.004)
    assert out['surface_slip_window'].item()==0


def test_contact_break_debounces_and_caps_events():
    r=RewardAblation('contact_break',torch.tensor([.36],dtype=torch.float64))
    step(r,bottom=.364,duration=.1)
    assert step(r,bottom=.364,early=False)['surface_contact_break'].item()==0
    assert step(r,bottom=.364,early=False)['surface_contact_break'].item()==pytest.approx(-4.)
    assert step(r,bottom=.364,early=False)['surface_contact_break'].item()==0


def test_support_deficit_requires_positive_runtime_weight():
    r=RewardAblation('support_deficit',torch.tensor([.36],dtype=torch.float64))
    assert step(r,bottom=.364,support=.8,weight=1.)['surface_support_deficit'].item()<0
    with pytest.raises(ValueError):step(r,weight=0.)
