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


@pytest.mark.parametrize('velocity',[3.1,-3.1,8.834162712097168])
def test_live_velocity_spike_stops_before_reward_or_another_physics_step(velocity):
    t=TensorTask(template(),2,'cpu');w=World(t);e=TensorEnv(w,t)
    w.qd[0]=velocity
    safe,rf,rm=physical_guard(w,w.m)
    assert safe.tolist()==[False,True] and not (rf|rm).any()
    with pytest.raises(RuntimeError,match='live_velocity'):e.step(torch.zeros((2,20)))
    assert w.tick==1 and not w.resets
    assert not t.elapsed.any() and not e.episodes.any()
    assert e.physics_peaks[2]==pytest.approx(abs(velocity))


@pytest.mark.parametrize('force,mimic,eligible',[(31,.0,True),(45,0,False),(31,.021,False),(0,.025,True),(0,.03,False)])
def test_independent_guard_recovery_bounds(force,mimic,eligible):
    t=TensorTask(template(),2,'cpu');w=World(t)
    w.m['object_link_loads'][:]=force;w.m['mimic_errors'][:]=mimic
    strict,rf,rm=physical_guard(w,w.m)
    assert not strict.any() and bool((rf|rm).all())==eligible
    w.qd[:]=3.1
    assert not (physical_guard(w,w.m)[1]|physical_guard(w,w.m)[2]).any()


@pytest.mark.parametrize('velocity',[3.000006675720215,-3.000006675720215])
def test_terminal_reset_accepts_float32_solver_velocity_tolerance(velocity):
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


@pytest.mark.parametrize('velocity',[3.00002,-3.00002,3.1714026927948,float('nan'),float('inf')])
def test_reset_velocity_rejects_excess_beyond_numerical_tolerance(velocity):
    t=TensorTask(template(),2,'cpu');w=World(t);e=TensorEnv(w,t)
    w.qd[:]=velocity
    with pytest.raises(RuntimeError,match='initial_reset'):e.reset()


def test_force_recovery_uses_same_tiny_velocity_tolerance():
    t=TensorTask(template(),2,'cpu');w=World(t)
    w.m['object_link_loads'][:]=31.
    w.qd[0]=3.000006675720215;w.qd[1]=3.00002
    _,recover,_=physical_guard(w,w.m)
    assert recover.tolist()==[True,False]


def test_velocity_tolerance_boundary_is_fixed_and_inclusive():
    from go2_pvcnn.control.dh116.tabletop_tensor_env import reset_velocity_safe
    t=TensorTask(template(),2,'cpu');w=World(t)
    w.qd=w.qd.double()
    w.qd[0]=3.+1e-5;w.qd[1]=3.+1e-5+1e-10
    assert reset_velocity_safe(w).tolist()==[True,False]


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


@pytest.mark.parametrize('fault',['velocity','force','mimic','nonfinite','joint','clearance','bottom'])
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
            if self.resets:self.qd[0,0,0]=3.00002
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
