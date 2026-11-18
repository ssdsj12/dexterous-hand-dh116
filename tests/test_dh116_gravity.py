import numpy as np
import pytest
import torch
from go2_pvcnn.control.dh116.linear_coupling import JOINTS,ACTIVE
from go2_pvcnn.control.dh116.gravity import compensated_efforts

NAMES=tuple(f'left_panda_joint{i}' for i in range(1,8))+JOINTS

def test_gravity_folds_follower_torques_and_preserves_arm():
    g=torch.zeros((1,18),dtype=torch.float64)
    g[0,0]=3.
    g[0,NAMES.index('finger21')]=.01
    g[0,NAMES.index('finger22')]=.02
    out=compensated_efforts(g,NAMES,'left',40.,.1)
    assert out[0,0]==3.
    assert out[0,NAMES.index('finger21')]==pytest.approx(.01+.9206*.02)
    assert out[0,NAMES.index('finger22')]==0.
    order=np.random.default_rng(42).permutation(18)
    perm=compensated_efforts(g[:,order],tuple(NAMES[i] for i in order),'left',40.,.1)
    torch.testing.assert_close(perm,out[:,order])

@pytest.mark.parametrize('problem',['nan','arm_limit','hand_limit','shape','names','limit'])
def test_gravity_rejects_unsafe_input(problem):
    g=torch.zeros((1,18));names=NAMES;limit=40.
    if problem=='nan':g[0,0]=float('nan')
    if problem=='arm_limit':g[0,0]=41.
    if problem=='hand_limit':g[0,NAMES.index('finger21')]=.2
    if problem=='shape':g=torch.zeros(18)
    if problem=='names':names=names[:-1]+(names[0],)
    if problem=='limit':limit=-1.
    with pytest.raises(ValueError):compensated_efforts(g,names,'left',limit,.1)


def test_gravity_failure_identifies_joint_and_folded_demand():
    g=torch.zeros((1,18),dtype=torch.float64);g[0,NAMES.index('finger31')]=.2
    with pytest.raises(ValueError,match=r'left.*finger31.*0.2.*0.1'):
        compensated_efforts(g,NAMES,'left',40.,.1)


def test_drive_headroom_reserves_feedforward_in_same_motor_budget():
    from go2_pvcnn.control.dh116.gravity import drive_headroom
    ff=torch.zeros((1,18),dtype=torch.float64)
    ff[0,1]=45.;ff[0,NAMES.index('finger31')]=-.13
    remaining=drive_headroom(ff,NAMES,'left',60.,.2)
    assert remaining[0,1]==15.
    assert remaining[0,NAMES.index('finger31')]==pytest.approx(.07)
    assert remaining[0,NAMES.index('finger32')]==0
    ff[0,1]=61.
    with pytest.raises(ValueError):drive_headroom(ff,NAMES,'left',60.,.2)
