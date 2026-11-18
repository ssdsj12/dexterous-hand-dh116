"""Software scheduling and guards with synthetic physics; not grasp evidence."""
import numpy as np
import pytest
from go2_pvcnn.control.dh116.linear_coupling import active_limits, expand, JOINTS
from go2_pvcnn.control.dh116.tabletop_env import TabletopEnv
from go2_pvcnn.control.dh116.tabletop_protected_vec_env import ProtectedMimicVecEnv, UnrecoverablePhysicsError

class World:
    def __init__(self, fault_ticks=(1,), terminal_overrides=None, after_reset_overrides=None, faults_by_env=None):
        self.ticks=0; self.events=[]; self.fault_ticks=set(fault_ticks)
        self.faults_by_env=faults_by_env if faults_by_env is not None else {0:self.fault_ticks}
        self.terminal_overrides=terminal_overrides or {}; self.after_reset_overrides=after_reset_overrides or {}
        self.backends=[Backend(self,i) for i in range(2)]
    def advance(self, requests):
        for b,r in zip(self.backends,requests):
            if r is not None:
                b.q=r[1].copy(); self.events.append(('command',self.ticks,b.index))
        self.events.append(('physics',self.ticks)); self.ticks+=1
        for b in self.backends:
            b.fault=self.ticks in self.faults_by_env.get(b.index, ())
    def guard(self,i):
        b=self.backends[i]
        if b.fault:
            return ['mimic_error',self.terminal_overrides.get('other_hand_failure')]
        return [None,None]

class Backend:
    dt=1/240; decimation=4
    def __init__(self,w,i):
        self.world=w; self.index=i; self.resets=0; self.fault=False
        lo,hi=active_limits(); self.grasp=np.tile((lo+hi)/2,2); self.initial_command=self.grasp.copy()
    def reset(self,offset):
        self.resets+=1; self.fault=False; self.q=self.initial_command.copy(); self.offset=offset
        self.reset_tick=self.world.ticks; self.world.events.append(('reset',self.world.ticks,self.index))
    def measure(self):
        age=self.world.ticks-self.reset_tick
        m=dict(q=self.q.copy(),qd=np.zeros(12),forces=np.zeros(4),clearance=.01,
          object_relative=np.array([age*.001,0.,-.02]),object_quat=np.array([1.,0.,0.,0.]),
          object_velocity=np.zeros(6),bottom=.30,approach_error=0.,joint_failure=None,
          mimic_errors=np.zeros(10),lowest_link='fake',palm_adjustment=np.zeros(4),
          diagnostics=[dict(joints_rad=dict(zip(JOINTS,expand(self.q[:6]))),
                            joint_vel_rad_s=np.zeros(11).tolist()) for _ in range(2)])
        if self.fault:
            m['joint_failure']='mimic_error';m['mimic_errors'][0]=.021
            m.update({k:v for k,v in self.world.terminal_overrides.items() if k!='other_hand_failure'})
        if self.index==0 and self.resets>1 and age>0:
            m.update(self.world.after_reset_overrides)
        return m

def make(**kwargs):
    w=World(**kwargs)
    envs=[TabletopEnv(b,stage='full',reset_phase='close',control_mode='joint_target',
             grasp_mode='object_stability',reset_jitter=0) for b in w.backends]
    vec=ProtectedMimicVecEnv(envs,w.advance,joint_guard=w.guard);vec.reset()
    return w,vec

def test_mimic_reset_is_immediate_remaining_substeps_are_checked_and_original_failure_returned():
    w,v=make();obs,r,d,info=v.step(np.zeros((2,12)))
    assert w.ticks==4 and d.tolist()==[True,False]
    assert ('reset',1,0) in w.events and w.backends[1].resets==1
    assert [e for e in w.events if e[0]=='command' and e[2]==0]==[('command',0,0)]
    assert info[0]['safety_failure']=='mimic_error' and info[0]['recovered_mimic'] is True
    assert info[0]['physical']['max_mimic_error_rad']==pytest.approx(.021)
    assert info[0]['reset_validation']['checked_physics_substeps']==3
    assert info[0]['reset_validation']['passed'] is True
    assert r[0]<-24 and np.array_equal(obs[0],v.envs[0].observe())
    assert not np.array_equal(obs[0],info[0]['terminal_observation'])
    assert v.envs[0].task.elapsed==0 and v.envs[1].task.elapsed==pytest.approx(4/240)
    obs,r,d,info=v.step(np.zeros((2,12)))
    assert w.ticks==8 and not d.any() and v.envs[0].task.elapsed==pytest.approx(4/240)

def test_twenty_repeated_faults_reset_locally_and_other_slot_keeps_running():
    w,v=make(fault_ticks=range(1,81,4))
    for _ in range(20):
        _,_,d,info=v.step(np.zeros((2,12)))
        assert d.tolist()==[True,False] and info[0]['reset_validation']['passed']
    assert w.ticks==80 and w.backends[0].resets==21 and w.backends[1].resets==1
    assert v.envs[1].task.elapsed==pytest.approx(80/240)

def test_last_substep_recovery_reports_zero_checked_physics_steps():
    w,v=make(fault_ticks=(4,));_,_,_,info=v.step(np.zeros((2,12)))
    assert info[0]['reset_validation']['checked_physics_substeps']==0
    assert ('reset',4,0) in w.events
    assert not v.step(np.zeros((2,12)))[2].any()

@pytest.mark.parametrize('overrides,failure',[
    (dict(mimic_errors=np.array([.03]+[0.]*9)),'mimic_error'),
    (dict(other_hand_failure='joint_limit'),'joint_limit'),
    (dict(forces=np.array([31.,0.,0.,0.])),'excess_force'),
    (dict(clearance=-.002),'table_penetration'),
    (dict(bottom=.28),'object_dropped'),
    (dict(qd=np.full(12,np.nan)),'nonfinite_state'),
    (dict(diagnostics=[dict(joint_vel_rad_s=[3.01])]),'joint_velocity'),
    (dict(diagnostics=[dict(joint_vel_rad_s=[np.nan])]),'nonfinite_state'),
])
def test_other_or_large_faults_stop_before_another_physics_step(overrides,failure):
    w,v=make(terminal_overrides=overrides)
    with pytest.raises(UnrecoverablePhysicsError) as e:v.step(np.zeros((2,12)))
    assert e.value.global_stop is True and e.value.failure==failure
    assert e.value.info['global_stop'] is True
    assert w.ticks==1 and w.backends[0].resets==1

@pytest.mark.parametrize('overrides',[
    dict(clearance=-.002),dict(forces=np.array([31.,0.,0.,0.])),
    dict(joint_failure='joint_limit'),dict(mimic_errors=np.array([.021]+[0.]*9),joint_failure='mimic_error'),
])
def test_guard_failure_during_post_reset_hold_stops_immediately(overrides):
    w,v=make(after_reset_overrides=overrides)
    with pytest.raises(UnrecoverablePhysicsError) as e:v.step(np.zeros((2,12)))
    assert w.ticks==2 and w.backends[0].resets==2
    assert e.value.info['safety_failure']=='mimic_error'
    assert e.value.info['reset_validation']['passed'] is False


def test_ordinary_terminal_slot_is_guarded_during_remaining_shared_substeps():
    w,v=make(fault_ticks=())
    v.envs[0].task.elapsed=14.999
    backend=w.backends[0]; original=backend.measure
    def measured():
        result=original()
        if w.ticks>=2: result['forces']=np.array([31.,0.,0.,0.])
        return result
    backend.measure=measured
    with pytest.raises(UnrecoverablePhysicsError) as e:v.step(np.zeros((2,12)))
    assert w.ticks==2 and backend.resets==1
    assert e.value.failure=='excess_force' and e.value.info['failure']=='timeout'
    assert e.value.info['physical']['max_force_n']==0.


def test_reset_overwrites_old_drive_target_and_no_old_command_is_reissued():
    w,v=make(); actions=np.ones((2,12))
    _,_,_,info=v.step(actions)
    assert np.array_equal(w.backends[0].q,w.backends[0].initial_command)
    assert [e for e in w.events if e[0]=='command' and e[2]==0]==[('command',0,0)]
    assert info[0]['reset_validation']['validation_scope']=='reset_and_remaining_substeps'


def test_last_substep_scope_explicitly_reports_only_immediate_reset_verified():
    w,v=make(fault_ticks=(4,));_,_,_,info=v.step(np.zeros((2,12)))
    assert info[0]['reset_validation']['validation_scope']=='immediate_reset_only'
    assert info[0]['reset_validation']['checked_physics_substeps']==0


@pytest.mark.parametrize('late_tick', [1, 2, 4])
def test_multiple_slots_can_reset_at_distinct_substeps_in_one_shared_tick(late_tick):
    w,v=make(faults_by_env={0:{1},1:{late_tick}})
    obs,rewards,dones,infos=v.step(np.ones((2,12)))
    assert w.ticks==4 and dones.tolist()==[True,True]
    assert ('reset',1,0) in w.events and ('reset',late_tick,1) in w.events
    assert [b.resets for b in w.backends]==[2,2]
    assert [i['reset_validation']['checked_physics_substeps'] for i in infos]==[3,4-late_tick]
    assert all(i['recovered_mimic'] and i['safety_failure']=='mimic_error' for i in infos)
    assert all(env.task.elapsed==0 for env in v.envs)
    assert all(np.array_equal(obs[i],env.observe()) for i,env in enumerate(v.envs))
    assert np.all(rewards < -24)


@pytest.mark.parametrize('mimic_terminal', [True,False])
def test_reset_exception_is_a_global_stop_with_original_terminal_info(mimic_terminal):
    w,v=make(fault_ticks=(1,) if mimic_terminal else ())
    if not mimic_terminal: v.envs[0].task.elapsed=14.999
    def fail_reset(offset): raise ValueError('injected reset failure')
    w.backends[0].reset=fail_reset
    with pytest.raises(UnrecoverablePhysicsError) as e:v.step(np.zeros((2,12)))
    assert e.value.failure=='reset_failed' and e.value.global_stop and v.global_stop
    assert e.value.info['failure']==('mimic_error' if mimic_terminal else 'timeout')
    assert w.ticks==(1 if mimic_terminal else 4)
    with pytest.raises(RuntimeError,match='globally stopped'):v.step(np.zeros((2,12)))
    assert w.ticks==(1 if mimic_terminal else 4)


@pytest.mark.parametrize('failure', ['joint_limit','excess_force'])
def test_last_substep_reset_followed_by_fatal_first_next_substep_stops_world(failure):
    overrides=(dict(joint_failure=failure) if failure=='joint_limit'
               else dict(forces=np.array([31.,0.,0.,0.])))
    w,v=make(fault_ticks=(4,),after_reset_overrides=overrides)
    _,_,_,info=v.step(np.zeros((2,12)))
    assert info[0]['reset_validation']['checked_physics_substeps']==0
    assert info[0]['reset_validation']['validation_scope']=='immediate_reset_only'
    with pytest.raises(UnrecoverablePhysicsError) as e:v.step(np.zeros((2,12)))
    assert w.ticks==5 and e.value.failure==failure
    assert w.backends[0].resets==2


def test_last_substep_reset_cannot_recover_again_before_first_post_reset_physics_passes():
    w,v=make(fault_ticks=(4,5))
    _,_,_,info=v.step(np.zeros((2,12)))
    assert info[0]['recovered_mimic'] and info[0]['reset_validation']['checked_physics_substeps']==0
    with pytest.raises(UnrecoverablePhysicsError) as e:v.step(np.zeros((2,12)))
    assert w.ticks==5 and w.backends[0].resets==2
    assert e.value.failure=='mimic_error'
    assert e.value.info['first_post_reset_physics']['passed'] is False
    # A later failure cannot mutate the previously returned transition.
    assert info[0]['reset_validation']['verified'] is True
    assert info[0]['reset_validation']['validation_scope']=='immediate_reset_only'


def test_first_safe_post_reset_substep_clears_pending_before_a_later_new_mimic_failure():
    w,v=make(fault_ticks=(4,6))
    v.step(np.zeros((2,12)))
    _,_,dones,info=v.step(np.zeros((2,12)))
    assert w.ticks==8 and w.backends[0].resets==3
    assert dones.tolist()==[True,False] and info[0]['recovered_mimic']
    assert info[0]['reset_validation']['terminal_physics_substep']==2
    assert info[0]['reset_validation']['checked_physics_substeps']==2
