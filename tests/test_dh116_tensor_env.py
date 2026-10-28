from types import SimpleNamespace
import pytest
import torch
from test_dh116_tensor_task import template, measurement
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
from go2_pvcnn.control.dh116.tabletop_tensor_env import physical_guard, TensorEnv


class World:
    n=2;device='cpu';dt=1/480;decimation=8
    def __init__(self,t):
        self.q=torch.zeros((2,2,6));self.qd=torch.zeros_like(self.q)
        self.limits=torch.stack((self.q-1,self.q+1),-1);self.errors=torch.zeros((2,2,5))
        self.m=measurement(t);self.tick=0;self.resets=[];self.fail_again=False
    def measure(self):return self.m
    def reset(self,ids):
        self.resets.append(ids.tolist())
        for key in ('object_link_loads','grouped_object_loads','hand_object_loads'):self.m[key][ids]=0.
        return self.m
    def advance(self,t,allowed,live):
        self.tick+=1
        if self.tick==3 or (self.tick==4 and self.fail_again):
            self.m['object_link_loads'][0,0,0]=31.
            self.m['grouped_object_loads'][0,0,5]=31.
            self.m['hand_object_loads'][0,0]=31.
        return self.m


def test_failed_transition_resets_only_once_and_keeps_terminal_penalty():
    t=TensorTask(template(),2,'cpu');w=World(t);e=TensorEnv(w,t)
    obs,r,d,extra=e.step(torch.zeros((2,20)))
    assert d.tolist()==[True,False] and r[0]<-20
    assert w.tick==8 and w.resets==[[0]] and e.recovered.tolist()==[1,0]
    assert extra['terms']['failure'].tolist()==[-25,0]
    assert e.episodes.tolist()==[1,0] and not t.done.any()


def test_immediately_repeated_fault_is_fatal():
    t=TensorTask(template(),2,'cpu');w=World(t);w.fail_again=True;e=TensorEnv(w,t)
    with pytest.raises(RuntimeError,match='post_reset_or_terminal'):e.step(torch.zeros((2,20)))


def test_float32_observation_overflow_is_global_failure():
    t=TensorTask(template(),2,'cpu');w=World(t);e=TensorEnv(w,t)
    w.m['object_relative'][0,0]=1e100
    with pytest.raises(RuntimeError,match='nonfinite_task_or_observation'):e.step(torch.zeros((2,20)))


def test_nonfinite_action_is_not_an_ordinary_training_reset():
    t=TensorTask(template(),2,'cpu');w=World(t);e=TensorEnv(w,t)
    actions=torch.zeros((2,20));actions[0,0]=float('nan')
    with pytest.raises(RuntimeError,match='invalid_policy_action'):e.step(actions)
    assert w.tick==0


@pytest.mark.parametrize('force,mimic,eligible',[(31,.0,True),(45,0,False),(31,.021,False),(0,.025,True),(0,.03,False)])
def test_independent_guard_recovery_bounds(force,mimic,eligible):
    t=TensorTask(template(),2,'cpu');w=World(t)
    w.m['object_link_loads'][:]=force;w.m['mimic_errors'][:]=mimic
    strict,rf,rm=physical_guard(w,w.m)
    assert not strict.any() and bool((rf|rm).all())==eligible
    w.qd[:]=3.1
    assert bool((physical_guard(w,w.m)[1]|physical_guard(w,w.m)[2]).all())==eligible
    w.qd[:]=float('nan')
    assert not (physical_guard(w,w.m)[1]|physical_guard(w,w.m)[2]).any()


@pytest.mark.parametrize('velocity',[3.000006675720215,-3.000006675720215,3.1714,-8.8])
def test_terminal_reset_accepts_finite_measured_speed(velocity):
    t=TensorTask(template(),2,'cpu');w=World(t);e=TensorEnv(w,t)
    w.qd[:]=velocity
    e.reset()
    t.elapsed[:]=4.01
    # Exercise termination plus frozen terminal rows before batched reset.
    w.tick=100
    _,reward,done,_=e.step(torch.zeros((2,20)))
    assert done.all() and (reward<-20).all()
    assert e.episodes.tolist()==[1,1]
    e.step(torch.zeros((2,20)))  # first physical substep after reset


@pytest.mark.parametrize('velocity',[float('nan'),float('inf'),-float('inf')])
def test_reset_velocity_rejects_nonfinite_values(velocity):
    t=TensorTask(template(),2,'cpu');w=World(t);e=TensorEnv(w,t)
    w.qd[:]=velocity
    with pytest.raises(RuntimeError,match='initial_reset'):e.reset()


def test_finite_velocity_does_not_block_force_recovery():
    t=TensorTask(template(),2,'cpu');w=World(t)
    w.m['object_link_loads'][:]=31.
    w.qd[0]=3.000006675720215;w.qd[1]=3.00002
    _,recover,_=physical_guard(w,w.m)
    assert recover.tolist()==[True,True]


def test_finite_speed_on_either_side_of_diagnostic_line_is_accepted():
    from go2_pvcnn.control.dh116.tabletop_tensor_env import reset_velocity_safe
    t=TensorTask(template(),2,'cpu');w=World(t)
    w.qd=w.qd.double()
    w.qd[0]=3.+1e-5;w.qd[1]=3.+1e-5+1e-10
    assert reset_velocity_safe(w).tolist()==[True,True]


class TerminalWorld(World):
    """Old terminal targets accelerate unless reset before the next substep."""
    def __init__(self,t,post_reset_fault=None):
        super().__init__(t)
        self.task=t;self.reset_ticks=[];self.post_reset_fault=post_reset_fault
        self.m['bottom'][0]=.34
    def reset(self,ids):
        self.reset_ticks.append(self.tick)
        super().reset(ids)
        self.qd[ids]=0.;self.m['bottom'][ids]=.30
        return self.m
    def advance(self,t,allowed,live):
        self.tick+=1
        if not live[0]:
            assert t.m['bottom'][0]==.34  # terminal snapshot survives world.reset
            if not self.resets:
                self.qd[0,0,0]=3.000018358230591
            elif self.post_reset_fault=='velocity':self.qd[0,0,0]=3.00002
            elif self.post_reset_fault=='force':self.m['object_link_loads'][0,0,0]=31.
            elif self.post_reset_fault=='mimic':self.m['mimic_errors'][0,0]=.021
            elif self.post_reset_fault=='nonfinite':self.q[0,0,0]=float('nan')
            elif self.post_reset_fault=='joint':self.q[0,0,0]=2.
            elif self.post_reset_fault=='clearance':self.m['clearance'][0]=-.002
            elif self.post_reset_fault=='bottom':self.m['bottom'][0]=.28
        return self.m


@pytest.mark.parametrize('reason',['grasp_deadline','timeout','success'])
def test_ordinary_terminal_resets_before_more_physics_and_preserves_transition(reason):
    t=TensorTask(template(),2,'cpu');w=TerminalWorld(t);e=TensorEnv(w,t)
    if reason=='grasp_deadline':t.elapsed[0]=4.01
    else:
        t.phase[0]=3
        if reason=='timeout':t.elapsed[0]=15.01
        else:t.success_time[0]=.5
    obs,reward,done,extra=e.step(torch.zeros((2,20)))
    assert w.tick==8 and w.resets==[[0]] and w.reset_ticks==[1]
    assert done.tolist()==[True,False] and e.episodes.tolist()==[1,0]
    assert e.recovered.tolist()==[0,0]
    assert bool(extra['success'][0])==(reason=='success')
    assert extra['used_time'][0]==w.dt and extra['used_time'][1]==8*w.dt
    assert not torch.equal(obs[0],extra['terminal_observation'][0])
    assert t.m['bottom'][0]==.30
    if reason=='success':assert extra['terms']['success'][0]==25 and reward[0]>20
    else:assert extra['terms']['failure'][0]==-25 and reward[0]<-20


@pytest.mark.parametrize('fault',['force','mimic','nonfinite','joint','clearance','bottom'])
def test_ordinary_reset_still_rejects_every_post_reset_guard(fault):
    t=TensorTask(template(),2,'cpu');w=TerminalWorld(t,fault);e=TensorEnv(w,t)
    t.elapsed[0]=4.01
    with pytest.raises(RuntimeError,match='post_reset_or_terminal'):e.step(torch.zeros((2,20)))
    assert w.tick==2 and w.resets==[[0]]


def test_final_substep_terminal_checks_first_next_physics_without_resetting_twice():
    class NextTickWorld(TerminalWorld):
        decimation=1
        def advance(self,*args):
            m=super().advance(*args)
            if self.resets:self.qd[0,0,0]=float('nan')
            return m
    t=TensorTask(template(),2,'cpu');w=NextTickWorld(t);e=TensorEnv(w,t)
    t.elapsed[0]=4.01
    e.step(torch.zeros((2,20)))
    assert e.pending.tolist()==[True,False]
    with pytest.raises(RuntimeError,match='post_reset_or_terminal'):e.step(torch.zeros((2,20)))
    assert w.tick==2 and w.resets==[[0]]


def test_mixed_ordinary_and_recovered_terminals_keep_separate_accounting():
    t=TensorTask(template(),2,'cpu');w=World(t);e=TensorEnv(w,t)
    t.elapsed[1]=4.01
    _,reward,done,extra=e.step(torch.zeros((2,20)))
    assert w.resets==[[1],[0]] and w.tick==8
    assert done.all() and (reward<-20).all()
    assert e.recovered.tolist()==[1,0] and e.episodes.tolist()==[1,1]
    assert extra['failure'].tolist()==[6,9]


def test_physical_failure_with_recovery_disabled_never_resets():
    t=TensorTask(template(),2,'cpu');w=World(t);e=TensorEnv(w,t,recovery=False)
    with pytest.raises(RuntimeError,match='independent_guard'):e.step(torch.zeros((2,20)))
    assert not w.resets


def test_physics_peak_telemetry_keeps_substep_spike_after_immediate_reset():
    t=TensorTask(template(),2,'cpu');w=World(t);e=TensorEnv(w,t)
    w.m['qd'][:]=2.5;w.qd[:]=1.7;w.m['mimic_errors'][:]=.002
    e.step(torch.zeros((2,20)))
    assert w.m['object_link_loads'][0].max()==0.
    assert e.physics_peaks.tolist()==pytest.approx([31.,2.5,1.7,.002])
    w.m['qd'].zero_();w.qd.zero_();w.m['mimic_errors'].zero_()
    e.step(torch.zeros((2,20)))
    assert e.physics_peaks.tolist()==pytest.approx([31.,2.5,1.7,.002])


class JointExcursionWorld(World):
    """A failed grasp slowly pushes one active joint beyond its nominal limit."""
    def __init__(self,t,excursion=.006,side='lower',reset_excursion=0.,repeat=False,joint=7):
        super().__init__(t)
        self.task=t;self.excursion=excursion;self.side=side
        self.reset_excursion=reset_excursion;self.repeat=repeat
        self.joint=joint
        self.initial_q=self.m['q'].clone()
        self.q=self.m['q'].reshape(2,2,6)
        self.limits=torch.stack((t.low,t.high),-1).reshape(1,2,6,2).repeat(2,1,1,1)
        self.reset_ticks=[]

    def displace(self,amount):
        bound=self.task.low[self.joint] if self.side=='lower' else self.task.high[self.joint]
        self.m['q'][0,self.joint]=bound+(-amount if self.side=='lower' else amount)

    def advance(self,t,allowed,live):
        self.tick+=1
        if self.tick==3 or (self.repeat and self.resets):self.displace(self.excursion)
        return self.m

    def reset(self,ids):
        self.resets.append(ids.tolist());self.reset_ticks.append(self.tick)
        self.m['q'][ids]=self.initial_q[ids]
        if self.reset_excursion:self.displace(self.reset_excursion)
        return self.m


@pytest.mark.parametrize('side',['lower','upper'])
@pytest.mark.parametrize('joint',[1,7])
def test_training_joint_excursion_ends_only_failed_grasp_before_hard_guard(side,joint):
    from go2_pvcnn.control.dh116.tabletop_tensor_task import FAILURE_CODES
    t=TensorTask(template(),2,'cpu');w=JointExcursionWorld(t,side=side,joint=joint);e=TensorEnv(w,t,training_joint_limit=True)
    _,reward,done,extra=e.step(torch.zeros((2,20)))
    assert done.tolist()==[True,False]
    assert w.tick==8 and w.resets==[[0]] and w.reset_ticks==[3]
    assert extra['failure'].tolist()==[FAILURE_CODES['joint_limit'],0]
    assert extra['terms']['failure'].tolist()==[-25.,0.] and reward[0]<-20.
    assert e.episodes.tolist()==[1,0] and e.recovered.tolist()==[1,0]
    assert extra['recovered'].tolist()==[True,False]
    assert torch.equal(w.m['q'][0],w.initial_q[0])


@pytest.mark.parametrize('excursion',[0.,.0049,.005])
@pytest.mark.parametrize('side',['lower','upper'])
def test_training_joint_excursion_tolerance_does_not_reset_healthy_grasp(excursion,side):
    t=TensorTask(template(),2,'cpu');w=JointExcursionWorld(t,excursion,side);e=TensorEnv(w,t,training_joint_limit=True)
    _,_,done,_=e.step(torch.zeros((2,20)))
    assert not done.any() and not w.resets


@pytest.mark.parametrize('fault',['hard_position','nonfinite_velocity','nonfinite','force','mimic','clearance','bottom'])
def test_training_joint_excursion_cannot_hide_simultaneous_hard_fault(fault):
    t=TensorTask(template(),2,'cpu');w=JointExcursionWorld(t);e=TensorEnv(w,t,training_joint_limit=True)
    w.tick=2  # Both the excursion and the hard fault occur on this next sample.
    if fault=='hard_position':w.excursion=.0201
    elif fault=='nonfinite_velocity':w.qd[0,0,0]=float('inf')
    elif fault=='nonfinite':w.errors[0,0,0]=float('nan')
    elif fault=='force':
        w.m['object_link_loads'][0,0,0]=45.
        w.m['grouped_object_loads'][0,0,5]=45.
        w.m['hand_object_loads'][0,0]=45.
    elif fault=='mimic':w.m['mimic_errors'][0,0]=.03
    elif fault=='clearance':w.m['clearance'][0]=-.002
    elif fault=='bottom':w.m['bottom'][0]=.28
    with pytest.raises(RuntimeError,match='physical safety failure'):e.step(torch.zeros((2,20)))
    assert not w.resets and not e.episodes.any()


def test_early_joint_boundary_preserves_prior_failure_reason():
    from go2_pvcnn.control.dh116.tabletop_tensor_task import FAILURE_CODES
    t=TensorTask(template(),2,'cpu');w=JointExcursionWorld(t);w.tick=2
    w.m['joint_failure'][0]=FAILURE_CODES['mimic_error']
    e=TensorEnv(w,t,training_joint_limit=True)
    # This fixture retains a classified failure despite a now-small mimic
    # residual; the earlier reason must not be relabeled as a joint excursion.
    _,_,done,extra=e.step(torch.zeros((2,20)))
    assert done.tolist()==[True,False] and w.resets==[[0]]
    assert extra['failure'][0]==FAILURE_CODES['mimic_error']
    assert e.recovered.tolist()==[0,0]


def test_training_joint_excursion_is_disabled_without_episode_recovery():
    t=TensorTask(template(),2,'cpu');w=JointExcursionWorld(t);e=TensorEnv(w,t,recovery=False,training_joint_limit=True)
    _,_,done,_=e.step(torch.zeros((2,20)))
    assert not done.any() and not w.resets


def test_early_joint_boundary_does_not_treat_prismatic_wrist_metres_as_radians():
    t=TensorTask(template(),2,'cpu');w=JointExcursionWorld(t,excursion=0.)
    w.q=torch.cat((torch.zeros((2,2,3)),w.q),dim=2)
    wrists=torch.tensor([-1.,1.]).expand(2,2,3,2)
    w.limits=torch.cat((wrists,w.limits),dim=2)
    w.q[0,0,0]=-1.006
    e=TensorEnv(w,t,training_joint_limit=True)
    _,_,done,_=e.step(torch.zeros((2,20)))
    assert not done.any() and not w.resets


@pytest.mark.parametrize('when',['initial','immediate','next_physics','next_tick'])
def test_joint_excursion_reset_must_clear_early_boundary(when):
    t=TensorTask(template(),2,'cpu');w=JointExcursionWorld(t);e=TensorEnv(w,t,training_joint_limit=True)
    if when in ('initial','immediate'):w.reset_excursion=.006
    else:w.repeat=True
    if when=='initial':
        with pytest.raises(RuntimeError,match='initial_reset'):e.reset()
    elif when=='immediate':
        with pytest.raises(RuntimeError,match='immediate_reset'):e.step(torch.zeros((2,20)))
    else:
        if when=='next_tick':
            w.decimation=3
            e.step(torch.zeros((2,20)))
        with pytest.raises(RuntimeError,match='post_reset_or_terminal'):e.step(torch.zeros((2,20)))
    assert w.resets==([[0,1]] if when=='initial' else [[0]])
