"""Behavioral tests of failed transitions and shared-clock reset validation."""
from types import SimpleNamespace
import numpy as np
import pytest
from test_dh116_batched_task import make_env
from go2_pvcnn.control.dh116.tabletop_batched_task import BatchTask
from go2_pvcnn.control.dh116.tabletop_gpu_vec_env import GpuTaskVecEnv
from go2_pvcnn.control.dh116.tabletop_task import joint_safety_failure
from go2_pvcnn.control.dh116.linear_coupling import active_limits


class World:
    def __init__(self, reason='excess_force', at=3, secondary=None):
        self.envs=[make_env(.055),make_env(.055)]
        self.backends=[e.backend for e in self.envs]
        self.n=2;self.tick=0;self.resets=[0,0];self.calls=[]
        self.reason=reason;self.at=at;self.secondary=secondary
        self.full_hand_contact=False
        self.palms=np.zeros((2,2,3));self.gains=np.full((2,2),10.)
        low,high=active_limits();self.limits=np.tile(np.stack([low,high],axis=1),(2,2,1,1))
        for i,b in enumerate(self.backends):
            physical=b.reset
            def reset(offset,i=i,b=b,physical=physical):
                prepared=b.__dict__.pop('_prepared_reset_offset',None)
                if prepared is None:
                    self.resets[i]+=1
                    physical(offset)
            b.reset=reset
        self.refresh()

    def refresh(self):
        self.q=np.stack([b.q.reshape(2,6) for b in self.backends])
        self.qd=np.zeros_like(self.q)
        self.errors=np.zeros((2,2,5))
        if self.fault_active() and self.reason=='mimic_error':self.errors[1,0,0]=.021

    def fault_active(self):
        return self.tick>=self.at and self.resets[1]==1 and self.envs[1].mode=='training'

    def reset_slots(self,indices,offsets):
        self.calls.append(list(indices))
        for i,offset in zip(indices,offsets):self.backends[i].reset(offset)
        self.refresh()

    def advance(self,requests):raise AssertionError('unexpected scalar physics')

    def advance_batch(self,offset,command,allowed,live,palm_actions):
        self.tick+=1
        for i,b in enumerate(self.backends):
            if live[i]:b.advance(offset[i],command[i],palm_motion_allowed=bool(allowed[i]),palm_action=palm_actions[i],grip_action=np.ones(2))
        self.refresh()

    def measure_batch(self):
        m=BatchTask.stack_measurements([b.measure() for b in self.backends])
        m['mimic_errors']=self.errors.reshape(2,10).copy()
        if self.fault_active():
            if self.reason=='excess_force':m['forces'][1,0]=31.
            if self.reason=='mimic_error':m['joint_failure'][1]='mimic_error'
            if self.secondary=='penetration':m['clearance'][1]=-.002
            if self.secondary=='velocity':self.qd[1,0,0]=4.
            if self.secondary=='nonfinite':m['object_velocity'][1,0]=np.nan
            if self.secondary=='severe_force':m['forces'][1,0]=46.
            if self.secondary=='both':m['forces'][1,0]=31.;m['mimic_errors'][1,0]=.021;self.errors[1,0,0]=.021
        if self.secondary=='post_reset' and self.resets[1]>1 and self.tick>self.at:
            m['clearance'][1]=-.002
        return m


def vector(reason='excess_force',at=3,secondary=None):
    world=World(reason,at,secondary)
    vec=GpuTaskVecEnv(world.envs,world,recover_training_failures=True)
    vec.reset();world.calls.clear()
    vec.env_method('set_mode','training')
    return world,vec


def actions():return np.tile(np.r_[np.zeros(18),-1.,-1.],(2,1))


@pytest.mark.parametrize('reason',['excess_force','mimic_error'])
@pytest.mark.parametrize('at',[1,3,8])
def test_failed_slot_resets_once_original_failure_is_retained(reason,at):
    from go2_pvcnn.control.dh116.tabletop_gpu_recovery import validated_recovery,VERSION
    world,vec=vector(reason,at)
    obs,rewards,dones,infos=vec.step(actions())
    assert world.tick==8 and world.calls==[[1]]
    assert world.resets==[1,2]
    assert dones.tolist()==[False,True] and rewards[1]<-20
    info=infos[1]
    assert info['failure']==info['safety_failure']==reason
    assert not info['is_success'] and info['recovered_physical']
    assert not np.array_equal(obs[1],info['terminal_observation'])
    assert vec.envs[1].task.elapsed==0 and vec.envs[0].task.elapsed==pytest.approx(8/480)
    v=info['reset_validation'];assert v['checked_physics_substeps']==8-at
    contract={'training_failure_handling':VERSION,'physics_hz':480,'control_hz':60,'observation_size':136}
    assert validated_recovery(info,True,contract,(136,))
    assert not validated_recovery(info,False,contract,(136,))
    assert not validated_recovery(info,True,{},(136,))
    vec.step(actions());assert world.tick==16 and world.resets==[1,2]


@pytest.mark.parametrize('secondary',['penetration','velocity','nonfinite','severe_force','both','post_reset'])
def test_other_or_post_reset_faults_stop_the_world(secondary):
    from go2_pvcnn.control.dh116.tabletop_protected_vec_env import UnrecoverablePhysicsError
    world,vec=vector(secondary=secondary)
    with pytest.raises(UnrecoverablePhysicsError):vec.step(actions())
    assert world.tick<=4
    with pytest.raises(RuntimeError):vec.step(actions())


def test_last_substep_reset_must_pass_next_physics_before_recovery_again():
    from go2_pvcnn.control.dh116.tabletop_protected_vec_env import UnrecoverablePhysicsError
    world,vec=vector(at=8,secondary='post_reset')
    _,_,_,infos=vec.step(actions())
    assert infos[1]['reset_validation']['validation_scope']=='immediate_reset_only'
    with pytest.raises(UnrecoverablePhysicsError):vec.step(actions())
    assert world.tick==9


def test_evaluation_does_not_accept_training_recovery():
    from go2_pvcnn.control.dh116.tabletop_vec_training import _safety
    world,vec=vector()
    obs,rewards,_,infos=vec.step(actions())
    assert _safety(infos,obs,rewards)=='excess_force'


def test_validator_rejects_tampering():
    import copy
    from go2_pvcnn.control.dh116.tabletop_gpu_recovery import validated_recovery,VERSION
    _,vec=vector();_,_,_,infos=vec.step(actions());info=infos[1]
    contract=dict(training_failure_handling=VERSION,physics_hz=480,control_hz=60)
    for key,value in [('post_reset_guards_safe',False),('checked_physics_substeps',0),('terminal_physics_substep',True),('version','unknown')]:
        bad=copy.deepcopy(info);bad['reset_validation'][key]=value
        assert not validated_recovery(bad,True,contract,(136,))
    bad=copy.deepcopy(info);bad['terminal_observation'][0]=np.nan
    assert not validated_recovery(bad,True,contract,(136,))


def test_real_ppo_consumes_failed_transition_and_updates(tmp_path):
    import json
    from go2_pvcnn.control.dh116.tabletop_gpu_recovery import VERSION
    from go2_pvcnn.control.dh116.tabletop_vec_training import run_vector_training
    world=World();world.armed=False;world.spent=False
    world.fault_active=lambda:world.armed and not world.spent
    advance=world.advance_batch;reset=world.reset_slots
    def wrapped_advance(*args):
        if world.envs[1].mode=='training':world.armed=True
        return advance(*args)
    def wrapped_reset(indices,offsets):
        if world.armed and 1 in indices:world.spent=True
        return reset(indices,offsets)
    world.advance_batch=wrapped_advance;world.reset_slots=wrapped_reset
    vec=GpuTaskVecEnv(world.envs,world,recover_training_failures=True)
    contract=dict(training_failure_handling=VERSION,physics_hz=480,control_hz=60,num_envs=2)
    report,model=run_vector_training(vec,tmp_path,contract,timesteps=16,eval_episodes=2,
        checkpoint_every=8,direct_training=True,ppo_kwargs=dict(n_steps=4,batch_size=8,n_epochs=1))
    assert model._n_updates>=2 and model.num_timesteps==16
    assert report.get('safety_failure') is None
    m=json.loads((tmp_path/'monitor.json').read_text())
    assert m['per_env'][1]['failure_counts']['excess_force']==1
    assert m['per_env'][1]['recovered_physical_episodes']==1
    assert m['per_env'][1]['last_return']<-20
    assert m['recovered_physical_episodes']==1
    assert m['recovered_mimic_episodes']==0


@pytest.mark.parametrize('terminal',['grasp_not_established','success'])
def test_ordinary_terminal_slots_remain_guarded_during_remaining_substeps(terminal):
    from go2_pvcnn.control.dh116.tabletop_protected_vec_env import UnrecoverablePhysicsError
    world,vec=vector(at=2)
    if terminal=='grasp_not_established':
        vec.envs[1].task.elapsed=6.
    else:
        vec.envs[1].task.success=True
    with pytest.raises(UnrecoverablePhysicsError,match='excess_force'):
        vec.step(actions())
    assert world.tick==2
