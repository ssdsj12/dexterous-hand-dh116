"""Once really airborne, preserve the acquired pose while the native drives hold."""
import importlib
import torch
def m():return importlib.import_module('go2_pvcnn.control.dh116.tabletop_pose_carry')

def test_only_real_airborne_opposed_grip_latches_and_reset_releases_it():
    active=torch.tensor([False,False,True,True])
    height=torch.tensor([.004,.002,0.,.04],dtype=torch.float64)
    paired=torch.tensor([[True,True],[True,True],[False,False],[True,True]])
    reset=torch.tensor([False,False,False,True])
    assert m().carry_latch(active,height,paired,reset).tolist()==[True,False,True,False]

def test_carry_commands_keep_all_fingers_xy_and_rotation_targets_unchanged():
    action=torch.ones(2,24,dtype=torch.float64)*.5
    lo=torch.ones(2,2,12,dtype=torch.float64)*-.01;hi=-lo
    result,up=m().carry_action(action,torch.tensor([True,False]),torch.tensor([.02,.02]),lo,hi,.003)
    assert torch.equal(result[1],action[1])
    expected=torch.zeros(24,dtype=torch.float64);expected[[14,17]]=.3
    torch.testing.assert_close(result[0],expected)

def test_actual_rod_height_controls_upward_stop_and_original_workspace_limits_still_win():
    a=torch.ones(2,24,dtype=torch.float64)
    lo=torch.ones(2,2,12,dtype=torch.float64)*-.01;hi=-lo;hi[0,1,8]=.001
    result,up=m().carry_action(a,torch.ones(2,dtype=torch.bool),torch.tensor([.03,.032]),lo,hi,.003)
    assert result[0,14]==.3 and result[0,17]==.1
    assert not result[1].any()


def test_prelift_retains_a_real_loaded_two_digit_grip_without_force_precision_gate():
    loads=torch.tensor([[[1.2,1.1,0.,0.,0.],[1.6,.8,0.,0.,0.]]],dtype=torch.float64)
    eligible=loads>0
    active=torch.zeros(1,dtype=torch.bool);duration=torch.zeros(1,dtype=torch.float64)
    for _ in range(12):
        active,duration=m().prelift_retention(active,duration,loads,torch.ones(1,2,dtype=torch.bool),
            eligible,torch.zeros(1,dtype=torch.bool),1/60)
    assert active.item()
    assert .19999999<float(duration[0])<.20000001


def test_prelift_requires_both_selected_pads_and_ignores_stale_reset_interval():
    loads=torch.tensor([[[2.5,.01,0.,0.,0.],[1.,1.,0.,0.,0.]]],dtype=torch.float64)
    eligible=torch.tensor([[[True,True,False,False,False]]]).expand(1,2,5)
    args=(torch.zeros(1,dtype=torch.bool),torch.ones(1,dtype=torch.float64),loads,
        torch.ones(1,2,dtype=torch.bool),eligible,torch.zeros(1,dtype=torch.bool),1/60)
    active,duration=m().prelift_retention(*args)
    assert not active.any() and not duration.any()
    args=list(args);args[0]=torch.ones(1,dtype=torch.bool);args[5]=torch.ones(1,dtype=torch.bool)
    active,duration=m().prelift_retention(*args)
    assert not active.any() and not duration.any()
