import pytest
import torch
from test_dh116_tensor_env import World
from test_dh116_tensor_task import template
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv


class DropWorld(World):
    def __init__(self,t,force=0.,mimic=0.,clearance=.01):
        super().__init__(t);self.force=force;self.mimic=mimic;self.clearance=clearance
    def advance(self,t,allowed,live):
        self.tick+=1
        if self.tick==3 or (self.tick==4 and self.fail_again):
            self.m['bottom'][0]=.289
            self.m['clearance'][0]=self.clearance
            self.m['object_link_loads'][0,0,0]=self.force
            self.m['grouped_object_loads'][0,0,5]=self.force
            self.m['hand_object_loads'][0,0]=self.force
            self.m['mimic_errors'][0,0]=self.mimic;self.errors[0,0,0]=self.mimic
        return self.m
    def reset(self,ids):
        super().reset(ids)
        self.m['bottom'][ids]=.30;self.m['clearance'][ids]=.01
        self.m['mimic_errors'][ids]=0.;self.errors[ids]=0.
        return self.m


@pytest.mark.parametrize('force,mimic',[(0.,0.),(31.,0.),(0.,.025),(31.,.025)])
def test_object_drop_resets_only_affected_row_and_retains_penalty(force,mimic):
    t=TensorTask(template(),2,'cpu');w=DropWorld(t,force,mimic);e=TensorEnv(w,t)
    obs,r,d,extra=e.step(torch.zeros((2,20)))
    assert d.tolist()==[True,False] and w.resets==[[0]]
    assert extra['terms']['failure'].tolist()==[-25.,0.]
    assert e.recovered.tolist()==[1,0] and int(e.object_drop_resets)==1
    assert w.tick==8 and not t.done.any() and torch.isfinite(obs).all()


@pytest.mark.parametrize('force,mimic,clearance',[(45.,0.,.01),(0.,.03,.01),(0.,0.,-.002),(float('nan'),0.,.01)])
def test_drop_never_exempts_other_hard_faults(force,mimic,clearance):
    t=TensorTask(template(),2,'cpu');w=DropWorld(t,force,mimic,clearance);e=TensorEnv(w,t)
    with pytest.raises(RuntimeError):e.step(torch.zeros((2,20)))
    assert not w.resets


def test_drop_recovery_requires_enabled_resets():
    t=TensorTask(template(),2,'cpu');w=DropWorld(t);e=TensorEnv(w,t,recovery=False)
    with pytest.raises(RuntimeError,match='independent_guard'):e.step(torch.zeros((2,20)))


def test_bad_physical_reset_still_fails_validation():
    t=TensorTask(template(),2,'cpu');w=DropWorld(t);w.fail_again=True;e=TensorEnv(w,t)
    with pytest.raises(RuntimeError,match='post_reset_or_terminal'):e.step(torch.zeros((2,20)))
    assert w.resets==[[0]]


@pytest.mark.parametrize('excursion',[.006,.025])
def test_drop_combined_with_finite_joint_excursion_is_isolated_reset(excursion):
    from test_dh116_tensor_env import JointExcursionWorld
    class DroppedJointWorld(JointExcursionWorld):
        def advance(self,t,allowed,live):
            result=super().advance(t,allowed,live)
            if self.tick==3:result['bottom'][0]=.289
            return result
        def reset(self,ids):
            result=super().reset(ids);result['bottom'][ids]=.30
            return result
    t=TensorTask(template(),2,'cpu');w=DroppedJointWorld(t,excursion=excursion)
    e=TensorEnv(w,t,training_joint_limit=True)
    _,r,d,extra=e.step(torch.zeros((2,20)))
    assert d.tolist()==[True,False] and w.resets==[[0]]
    assert extra['terms']['failure'].tolist()==[-25.,0.]
    assert int(e.object_drop_resets)==1
