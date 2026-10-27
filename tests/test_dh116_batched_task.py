"""Differential tests against the existing task, not new formula copies."""
import copy
import numpy as np
import pytest
from go2_pvcnn.control.dh116.tabletop_env import TabletopEnv
from go2_pvcnn.control.dh116.tabletop_batched_task import BatchTask
from go2_pvcnn.control.dh116.linear_coupling import active_limits


class Backend:
    dt=1/480
    decimation=8
    def __init__(self, force=.055, parameters=None):
        self.parameters=parameters
        lo,hi=active_limits();self.grasp=np.tile((lo+hi)/2,2)
        self.initial_command=self.grasp.copy();self.force=force
        self.adaptive_palm_offset=np.zeros((2,3));self.grip_stiffness=np.full(2,10.)
    def reset(self,offset):
        self.offset=offset;self.q=self.initial_command.copy();self.t=0
    def advance(self,offset,command,**kw):
        self.offset=offset;self.q=command.copy();self.t+=self.dt
        from go2_pvcnn.control.dh116.tabletop_adaptive_backend import adaptive_step
        self.adaptive_palm_offset,self.grip_stiffness=adaptive_step(palm_offset=self.adaptive_palm_offset,
            stiffness=self.grip_stiffness,palm_action=kw['palm_action'],grip_action=kw['grip_action'],
            hand_clearance=np.full(2,.01),dt=self.dt,palm_motion_allowed=kw['palm_motion_allowed'],parameters=self.parameters)
    def measure(self):
        return dict(q=self.q.copy(),qd=np.zeros(12),forces=np.full(4,self.force),clearance=.01,
            object_relative=np.array([0,0,-.02]),object_quat=np.array([1,0,0,0]),
            object_velocity=np.zeros(6),bottom=.30+max(0.,self.offset-.004),approach_error=abs(self.offset),
            joint_failure=None,diagnostics=[],mimic_errors=np.zeros(10),palm_adjustment=np.zeros(4),
            hand_clearance=np.full(2,.01),lowest_link='synthetic',adaptive_palm_offset=self.adaptive_palm_offset.copy(),
            grip_stiffness=self.grip_stiffness.copy(),pad_relative=np.zeros((4,3)))


def make_env(force, compensation=None, parameters=None):
    e=TabletopEnv(Backend(force,parameters),stage='full',reset_phase='close',control_mode='adaptive_grasp',
        grasp_mode='object_stability',initial_offset=.004,reset_jitter=0,
        force_direction=np.ones((2,6))/np.sqrt(6), force_palm_compensation=compensation,load_parameters=parameters)
    e.reset(seed=11);return e


@pytest.mark.parametrize(
    'maximum,expected',
    [(12., (2., 12.)), (1., (.1, 1.))],
)
def test_batch_force_policy_action_mapping_preserves_capacity_modes(maximum, expected):
    from go2_pvcnn.control.dh116.tabletop_load_parameters import LoadParameters
    parameters=LoadParameters(max_force_n=maximum,force_rate_n_s=None)
    task=BatchTask([make_env(.055,parameters=parameters)])
    actions=np.tile(np.r_[np.zeros(18),-1.,-1.],(1,1))
    task.control(actions,np.ones(1,dtype=bool),1/480)
    np.testing.assert_allclose(task.f_requested,[[expected[0],expected[0]]])
    actions[:,18:]=1.
    task.control(actions,np.ones(1,dtype=bool),1/480)
    np.testing.assert_allclose(task.f_requested,[[expected[1],expected[1]]])


@pytest.mark.parametrize('force,random_actions,compensated,profile',[(.055,False,False,False),(.055,True,False,False),(0.,False,False,False),(.055,True,True,False),(.4,True,True,True)])
def test_batch_matches_legacy_control_observation_reward_and_phase(force,random_actions,compensated,profile):
    from go2_pvcnn.control.dh116.tabletop_load_parameters import LoadParameters
    parameters=LoadParameters(max_force_n=4.,force_rate_n_s=2.,deadband_n=.05,stiffness_max=50.,stiffness_rate=40.,effort_limit_nm=.5) if profile else None
    compensation=np.array([[.01,.015,-.038],[-.028,-.014,-.03]]) if compensated else None
    legacy=[make_env(force,compensation,parameters),make_env(force*1.05,compensation,parameters)]
    fast=[make_env(force,compensation,parameters),make_env(force*1.05,compensation,parameters)]
    rng=np.random.default_rng(8)
    for tick in range(380):
        actions=rng.uniform(-1,1,(2,20)) if random_actions else np.tile(np.r_[np.zeros(18),-1.,-1.],(2,1))
        expected=[e.step(a) for e,a in zip(legacy,actions)]
        batch=BatchTask(fast)
        def advance(offset,command,allowed,active):
            for i,e in enumerate(fast):
                if active[i]:
                    e.backend.advance(offset[i],command[i],phase='close',palm_motion_allowed=bool(allowed[i]),
                        palm_action=batch.palm_actions[i],grip_action=np.ones(2))
            return batch.stack_measurements([e.backend.measure() for e in fast])
        actual=batch.step(actions,advance)
        for i,(x,y) in enumerate(zip(expected,actual)):
            np.testing.assert_allclose(x[0],y[0],rtol=0,atol=2e-6,err_msg=f'tick {tick} env {i}')
            assert x[1] == pytest.approx(y[1],abs=1e-10)
            assert x[2:] == y[2:]
            np.testing.assert_allclose(legacy[i].command,fast[i].command,atol=1e-12)
        if any(x[2] for x in expected):
            break


@pytest.mark.parametrize('kind',['mimic_warning','joint_limit','mimic_error','table_penetration','excess_force',
    'object_dropped','nonfinite_velocity','overflow_velocity','slip','contact_chatter','invalid_action'])
def test_batch_preserves_warning_and_failure_semantics(kind):
    legacy=[make_env(.055)];fast=[make_env(.055)]
    def instrument(e):
        original=e.backend.measure
        def measure():
            m=original()
            if e.backend.t>.022:
                if kind=='mimic_warning':m['mimic_errors'][0]=.006
                if kind in ('joint_limit','mimic_error'):m['joint_failure']=kind
                if kind=='table_penetration':m['clearance']=-.0011
                if kind=='excess_force':m['forces'][0]=31.
                if kind=='object_dropped':m['bottom']=.289
                if kind=='nonfinite_velocity':m['object_velocity'][0]=np.nan
                if kind=='overflow_velocity':m['object_velocity'][0]=1e300
                if kind=='slip':m['object_relative'][0]=max(0.,e.backend.t-.7)*.003
                if kind=='contact_chatter' and int(e.backend.t/.1)%3==2:m['forces'][:]=0.
            return m
        e.backend.measure=measure
    for e in legacy+fast:instrument(e)
    for tick in range(80):
        actions=np.tile(np.r_[np.zeros(18),-1.,-1.],(1,1))
        if kind=='invalid_action' and tick==2:actions[0,3]=np.nan
        expected=[legacy[0].step(actions[0])]
        batch=BatchTask(fast)
        def advance(offset,command,allowed,active):
            if active[0]:fast[0].backend.advance(offset[0],command[0],palm_motion_allowed=bool(allowed[0]),
                palm_action=np.zeros((2,3)),grip_action=np.ones(2))
            return batch.stack_measurements([fast[0].backend.measure()])
        actual=batch.step(actions,advance)
        x,y=expected[0],actual[0]
        np.testing.assert_allclose(x[0],y[0],atol=2e-6,rtol=0,err_msg=kind)
        assert x[1]==pytest.approx(y[1],abs=1e-10)
        assert x[2:]==y[2:]
        if x[2]:break


def test_controller_observation_overflow_terminates_at_same_substep():
    legacy=make_env(.055);fast=make_env(.055)
    for e in (legacy,fast):
        e.task.phase='lift';e.grip_relative=np.zeros(3)
        original=e.backend.measure
        def measure(original=original):
            m=original();m['object_relative'][0]=1e37;return m
        e.backend.measure=measure
    actions=np.tile(np.r_[np.zeros(18),-1.,-1.],(1,1))
    expected=legacy.step(actions[0]);batch=BatchTask([fast])
    def advance(offset,command,allowed,active):
        if active[0]:fast.backend.advance(offset[0],command[0],palm_motion_allowed=bool(allowed[0]),
            palm_action=np.zeros((2,3)),grip_action=np.ones(2))
        return batch.stack_measurements([fast.backend.measure()])
    actual=batch.step(actions,advance)[0]
    assert expected[2:] == actual[2:]
    assert legacy.task.elapsed == fast.task.elapsed
    np.testing.assert_array_equal(expected[0],actual[0])


def test_invalid_first_action_preserves_reset_filter_state():
    legacy=make_env(.055);fast=make_env(.055)
    for e in (legacy,fast):
        e.backend.initial_command=e.reference-.01;e.reset(seed=1)
    action=np.full((1,20),np.nan)
    expected=legacy.step(action[0]);batch=BatchTask([fast])
    actual=batch.step(action,lambda *args:batch.stack_measurements([fast.backend.measure()]))[0]
    np.testing.assert_array_equal(expected[0],actual[0])
    assert expected[1:] == actual[1:]


def test_early_terminal_measurement_is_frozen_while_other_slot_advances():
    legacy=[make_env(.055),make_env(.055)];fast=[make_env(.055),make_env(.055)]
    for e in (legacy[0],fast[0]):
        original=e.backend.measure
        def measure(original=original,e=e):
            m=original()
            if e.backend.t>2*e.backend.dt:m['joint_failure']='mimic_error'
            return m
        e.backend.measure=measure
    actions=np.tile(np.r_[np.zeros(18),-1.,-1.],(2,1))
    expected=[e.step(a) for e,a in zip(legacy,actions)]
    task=BatchTask(fast)
    def advance(offset,command,allowed,live):
        for i,e in enumerate(fast):
            e.backend.advance(offset[i],command[i],palm_motion_allowed=bool(allowed[i]),
                palm_action=np.zeros((2,3)),grip_action=np.ones(2))
            if not live[i]:e.backend.q+=.02
        return task.stack_measurements([e.backend.measure() for e in fast])
    actual=task.step(actions,advance)
    for x,y in zip(expected,actual):
        np.testing.assert_array_equal(x[0],y[0])
        assert x[1] == pytest.approx(y[1],abs=1e-12)
        assert x[2:] == y[2:]
    assert fast[0].task.elapsed < fast[1].task.elapsed
