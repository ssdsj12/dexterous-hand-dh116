import copy
from types import SimpleNamespace
import pytest
import torch
from test_dh116_tensor_env import World
from test_dh116_tensor_task import template
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask,FAILURE_CODES
from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv,physical_guard


class CombinedWorld(World):
    def __init__(self,t,force=30.031951291739396,mimic=.021337531524056732,repeat=False):
        super().__init__(t)
        self.force,self.mimic,self.repeat=force,mimic,repeat

    def advance(self,t,allowed,live):
        self.tick+=1
        if self.tick==3 or (self.repeat and self.resets):
            self.m['object_link_loads'][0,0,0]=self.force
            self.m['grouped_object_loads'][0,0,5]=self.force
            self.m['hand_object_loads'][0,0]=self.force
            self.m['mimic_errors'][0,0]=self.mimic
            self.m['joint_failure'][0]=FAILURE_CODES['mimic_error']
        return self.m

    def reset(self,ids):
        super().reset(ids)
        self.m['mimic_errors'][ids]=0.
        self.m['joint_failure'][ids]=0
        self.q[ids]=0.
        return self.m


def test_recorded_combined_fault_resets_failed_environment_once_preserving_penalty():
    t=TensorTask(template(),2,'cpu');w=CombinedWorld(t);e=TensorEnv(w,t)
    obs,reward,done,extra=e.step(torch.zeros((2,20)))
    assert done.tolist()==[True,False]
    assert w.resets==[[0]] and w.tick==8
    assert extra['terms']['failure'][0]==-25 and reward[0]<-20
    assert extra['failure'][0]==FAILURE_CODES['mimic_error']
    assert e.recovered.tolist()==[1,0] and e.episodes.tolist()==[1,0]
    assert e.combined_failure_resets==1
    assert e.physics_peaks[0]==pytest.approx(w.force)
    assert e.physics_peaks[3]==pytest.approx(w.mimic)
    assert not t.done.any() and torch.isfinite(obs).all()
    e.step(torch.zeros((2,20)))
    assert w.resets==[[0]]


@pytest.mark.parametrize('force,mimic',[(45.,.021),(31.,.03),(45.,.03)])
def test_combined_hard_bounds_still_abort(force,mimic):
    t=TensorTask(template(),2,'cpu');w=CombinedWorld(t,force,mimic);e=TensorEnv(w,t)
    with pytest.raises(RuntimeError,match='independent_guard'):e.step(torch.zeros((2,20)))
    assert not w.resets


def test_combined_fault_after_reset_still_aborts():
    t=TensorTask(template(),2,'cpu');w=CombinedWorld(t,repeat=True);e=TensorEnv(w,t)
    with pytest.raises(RuntimeError,match='post_reset_or_terminal'):e.step(torch.zeros((2,20)))
    assert w.resets==[[0]] and w.tick==4


@pytest.mark.parametrize('force',[29.,30.,30.031951291739396,44.999,45.,46.])
@pytest.mark.parametrize('mimic',[0.,.02,.021337531524056732,.029999,.03,.031])
def test_force_mimic_grid_has_no_combined_recovery_hole(force,mimic):
    t=TensorTask(template(),2,'cpu');w=World(t)
    w.m['object_link_loads'][:]=force;w.m['mimic_errors'][:]=mimic
    strict,rf,rm=physical_guard(w,w.m)
    expected_safe=force<=30. and mimic<=.02
    expected_reset=not expected_safe and force<45. and mimic<.03
    assert bool(strict.all())==expected_safe
    assert bool((rf|rm).all())==expected_reset


@pytest.mark.parametrize('fault',['nan','negative_load','clearance'])
def test_combined_reset_never_masks_independent_fault(fault):
    t=TensorTask(template(),2,'cpu');w=CombinedWorld(t)
    w.tick=2
    if fault=='nan':w.qd[0,0,0]=float('nan')
    if fault=='negative_load':w.m['object_link_loads'][0,1,0]=-1.
    if fault=='clearance':w.m['clearance'][0]=-.002
    e=TensorEnv(w,t)
    with pytest.raises(RuntimeError):e.step(torch.zeros((2,20)))
    assert not w.resets


def test_combined_recovery_requires_explicit_checkpoint_migration():
    from go2_pvcnn.control.dh116.tabletop_tensor_env import BOUNDED_FAILURE_RESET_CONTRACT,TERMINAL_RESET_CONTRACT
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    old=dict(terminal_reset=TERMINAL_RESET_CONTRACT)
    new=dict(old,bounded_failure_reset=BOUNDED_FAILURE_RESET_CONTRACT)
    assert 'joint_bounded_failure_reset_v2' in checkpoint_contract_changes(old,new)
    bad=copy.deepcopy(new);bad['bounded_failure_reset']['max_force_exclusive_n']=50.
    with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(old,bad)


@pytest.mark.parametrize('amount',[-1.021,1.021,2.])
def test_finite_joint_limit_violation_resets_only_that_environment(amount):
    class LimitWorld(World):
        def advance(self,t,allowed,live):
            self.tick+=1
            if self.tick==3:self.q[0,0,0]=amount
            return self.m
        def reset(self,ids):
            super().reset(ids);self.q[ids]=0.
            self.m['joint_failure'][ids]=0
            return self.m
    t=TensorTask(template(),2,'cpu');w=LimitWorld(t);e=TensorEnv(w,t)
    _,r,d,extra=e.step(torch.zeros((2,20)))
    assert d.tolist()==[True,False] and w.resets==[[0]]
    assert extra['failure'][0]==FAILURE_CODES['joint_limit']
    assert extra['terms']['failure'][0]==-25 and r[0]<-20
    assert w.tick==8 and e.joint_limit_resets==1


def test_joint_limit_does_not_mask_hard_force_fault():
    t=TensorTask(template(),2,'cpu');w=CombinedWorld(t,force=45.)
    w.q[0,0,0]=1.021;w.tick=2;e=TensorEnv(w,t)
    with pytest.raises(RuntimeError,match='physical safety failure'):e.step(torch.zeros((2,20)))
    assert not w.resets


def test_joint_limit_with_combined_bounded_fault_preserves_original_failure():
    t=TensorTask(template(),2,'cpu');w=CombinedWorld(t)
    w.q[0,0,0]=1.021;w.tick=2;e=TensorEnv(w,t)
    _,_,done,extra=e.step(torch.zeros((2,20)))
    assert done.tolist()==[True,False] and w.resets==[[0]]
    assert extra['failure'][0]==FAILURE_CODES['mimic_error']
    assert e.joint_limit_resets==1
