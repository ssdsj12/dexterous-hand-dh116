"""CPU SDK doubles with the real DualHandRodRuntime reset; no simulation."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from test_dual_hand_rod_runtime import setup as rod_setup, Grid, PADS
from test_dh116_deformable_bimanual_step import Hand as ActionHand
from go2_pvcnn.control.dh116.linear_coupling import JOINTS, ACTIVE, expand
from go2_pvcnn.control.dh116.tabletop_palm_mount import MountView
from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import HandBinding
from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import DualHandRodRuntime
from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import (
    DualHandResetTransaction, prepare_reset_state, RESET_OWNERS,
)


class Composer:
    def __init__(self, bodies):
        self.composed_force_as_torch = torch.ones((1,bodies,3))
        self.composed_torque_as_torch = torch.ones((1,bodies,3))
        self.active = True
    def reset(self, env_ids=None):
        self.composed_force_as_torch.zero_()
        self.composed_torque_as_torch.zero_()
        if env_ids is None:
            self.active = False
    def set_forces_and_torques(self,forces,torques,**kwargs):
        self.composed_force_as_torch[:] = forces
        self.composed_torque_as_torch[:] = torques
        self.active = True


class Hand(ActionHand):
    def __init__(self,side,events):
        super().__init__(side,(7,),mounted=True)
        self.side,self.events = side,events
        self.body_names = list(PADS)+['base_link']+[f'aux{i}' for i in range(17)]
        self.root_physx_view.prim_paths=[f'/World/{side}Hand/wrist_fixed']
        self.root_physx_view.link_paths=[[f'/World/{side}Hand/{n}' for n in self.body_names]]
        self.link_poses = torch.zeros((1,23,7)); self.link_poses[...,6]=1.
        self.root_velocity = torch.ones((1,6))
        self.stiffness = torch.full((1,17),17.)
        self.effort_limit = torch.full((1,17),.15)
        self.actuation = torch.ones((1,17))
        self.effort_target = torch.ones((1,17))
        self.velocity_target = torch.ones((1,17))
        self.actuators={'active':SimpleNamespace(stiffness=torch.full((1,6),17.))}
        self.permanent_wrench_composer=Composer(23)
        self.instantaneous_wrench_composer=Composer(23)
        self.native_zero_wrench_writes=0
        self.root_physx_view.get_link_transforms=lambda:self.link_poses
        self.root_physx_view.get_root_velocities=lambda:self.root_velocity
        self.root_physx_view.get_dof_stiffnesses=lambda:self.stiffness
        self.root_physx_view.get_dof_max_forces=lambda:self.effort_limit
        self.root_physx_view.get_dof_actuation_forces=lambda:self.actuation
        self.root_physx_view.get_dof_velocity_targets=lambda:self.velocity_target
    def reset(self,env_ids=None):
        self.events.append(self.side+':reset')
        self.permanent_wrench_composer.reset(env_ids)
        self.instantaneous_wrench_composer.reset(env_ids)
    def write_joint_state_to_sim(self,q,qd,env_ids=None):
        self.events.append(self.side+':joint')
        self.measured[:] = q;self.velocity[:] = qd
        self.data.joint_pos[:]=q;self.data.joint_vel[:]=qd
    def write_root_pose_to_sim(self,pose,env_ids=None):
        self.events.append(self.side+':root')
        self.data.root_pose_w[:]=pose
        self.root_native[:]=pose[:,[0,1,2,4,5,6,3]]
    def write_root_velocity_to_sim(self,value,env_ids=None):
        self.root_velocity[:]=value
    def write_joint_stiffness_to_sim(self,values,joint_ids,env_ids=None):
        self.stiffness[:,joint_ids]=values
    def write_joint_effort_limit_to_sim(self,values,joint_ids,env_ids=None):
        self.effort_limit[:,joint_ids]=values
    def set_joint_effort_target(self,values,joint_ids=None,env_ids=None):
        if joint_ids is None:self.effort_target[:]=values
        else:self.effort_target[:,joint_ids]=values
    def set_joint_velocity_target(self,values,joint_ids=None,env_ids=None):
        if joint_ids is None:self.velocity_target[:]=values
        else:self.velocity_target[:,joint_ids]=values
    def set_joint_position_target(self,values,joint_ids,env_ids=None):
        super().set_joint_position_target(values,joint_ids)
    def write_data_to_sim(self):
        self.events.append(self.side+':flush')
        super().write_data_to_sim()
        self.actuation[:]=self.effort_target
        if self.permanent_wrench_composer.active:
            assert not self.permanent_wrench_composer.composed_force_as_torch.any()
            assert not self.permanent_wrench_composer.composed_torque_as_torch.any()
            self.native_zero_wrench_writes+=1
        self.instantaneous_wrench_composer.reset()
    def update(self,dt):
        self.events.append(self.side+':cache_refresh')


def setup(monkeypatch):
    events=[]
    base,bindings,poses,_=rod_setup(monkeypatch)
    rod=DualHandRodRuntime(base,bindings=bindings,right_grids={n:Grid() for n in PADS},
        right_poses={n:poses['right.'+n] for n in PADS},run_uuid='run-cpu',env_id=7)
    hands=[Hand(s,events) for s in ('left','right')]
    hand_bindings=[HandBinding(s,h,{7:f'/World/{s}Hand/wrist_fixed'},MountView(h))
                   for s,h in zip(('left','right'),hands)]
    def refresh():
        events.append('kinematic_refresh')
        for h in hands:
            h.link_poses[:]=h.root_native[:,None]
            h.link_poses[0,:,1]+=torch.arange(23)*.001
    sim=SimpleNamespace(current_time_step_index=2,current_time=2/480,
        get_physics_dt=lambda:1/480,physics_sim_view=SimpleNamespace(update_articulations_kinematic=refresh))
    source=prepare_reset_state(
        hand_roots={s:[.5, y, .6,1.,0.,0.,0.] for s,y in [('left',.1),('right',-.1)]},
        origin=[.01,.02,.03],root_xy=[[.001,.002],[-.001,-.002]],initial_offset_m=.04,
        joint_positions=[expand(np.full(6,.4)).tolist(),expand(np.full(6,.5)).tolist()],
        joint_velocities=[np.linspace(0,.01,11).tolist(),np.linspace(0,.02,11).tolist()],
        source_description='explicit CPU fixture; no safe pose qualification')
    owners={name:(lambda context,n=name:dict(owner=n,status='cleared',
        source='CPU explicit owner',episode_uid=context['episode_uid'])) for name in RESET_OWNERS}
    return events,rod,hands,hand_bindings,sim,source,owners


def test_real_reset_order_preserves_same_rod_clock_and_named_seed(monkeypatch):
    events,rod,hands,bindings,sim,source,owners=setup(monkeypatch)
    old_rope=id(rod.rope)
    txn=DualHandResetTransaction(source,episode_uid='run-cpu:reset-a')
    result=txn.run(sim,rod,bindings,owner_resets=owners)
    assert id(rod.rope)==old_rope and rod.rope.resets==1 and rod.rope.steps==0
    assert result['new_sdk_steps']==result['new_solver_steps']==0
    assert result['after_clock']['physics_step']==2 and result['after_clock']['rod_tick']==0
    assert result['after_clock']['reset_id']==1 and result['after_clock']['episode_uid']=='run-cpu:reset-a'
    assert result['initial_pose_qualified'] is result['teacher_admission'] is False
    assert result['external_body_force_native_readback']['status']=='unavailable_no_getter'
    assert result['r0_reset_complete'] is False
    assert events.index('kinematic_refresh')>events.index('right:flush')
    for hi,(side,hand) in enumerate(zip(('left','right'),hands)):
        packet=result['hands'][side]
        for i,name in enumerate(JOINTS):
            actual=hand.joint_names.index(name)
            assert packet['q'][0,actual]==pytest.approx(source['joint_positions'][hi][i],abs=3e-8)
            assert packet['qd'][0,actual]==pytest.approx(source['joint_velocities'][hi][i],abs=3e-8)
        assert hand.native_zero_wrench_writes==1
        assert not hand.permanent_wrench_composer.active
        assert not hand.instantaneous_wrench_composer.active
        assert not packet['explicit_actuation_force'].any()
        assert np.allclose(packet['root_pose_wxyz'],np.asarray(source['reset_root_pose_w'])[hi][None],atol=1e-6)
    # Rod receives the post-refresh actual pad poses, not the source/old pose.
    actual=dict((slot,p) for slot,p,_,teleport in rod.rope.poses if teleport)
    assert np.allclose(actual[0],hands[0].link_poses[0,0,:3].numpy())
    assert np.allclose(actual[6],hands[1].link_poses[0,0,:3].numpy())
    with pytest.raises(RuntimeError):
        txn.run(sim,rod,bindings,owner_resets=owners)


@pytest.mark.parametrize('fault',['missing_owner','wrong_side','bad_seed','wrong_mount'])
def test_prevalidation_never_mutates_either_hand_or_rod(monkeypatch,fault):
    events,rod,hands,bindings,sim,source,owners=setup(monkeypatch)
    if fault=='missing_owner':owners.pop('contact')
    elif fault=='wrong_side':bindings.reverse()
    elif fault=='bad_seed':source['joint_positions'][0][0]=9.
    else:bindings[1].mount.hand=hands[0]
    with pytest.raises(ValueError):
        DualHandResetTransaction(source,episode_uid='run-cpu:reset-a').run(
            sim,rod,bindings,owner_resets=owners)
    assert not events and rod.rope.resets==0


@pytest.mark.parametrize('fault',['hand_write','kinematic','rod_reset','owner','readback','sdk_tick','solver_time'])
def test_partial_failure_poisoned_without_claiming_rollback(monkeypatch,fault):
    events,rod,hands,bindings,sim,source,owners=setup(monkeypatch)
    def fail(*a,**k):raise RuntimeError('injected reset failure')
    if fault=='hand_write':hands[1].write_joint_state_to_sim=fail
    elif fault=='kinematic':sim.physics_sim_view.update_articulations_kinematic=fail
    elif fault=='rod_reset':rod.rope.reset_envs=fail
    elif fault=='owner':owners['filter']=fail
    elif fault=='readback':
        original=hands[1].write_data_to_sim
        def wrong():
            original();hands[1].backend[:,hands[1].joint_names.index('finger11')]+=.01
        hands[1].write_data_to_sim=wrong
    elif fault=='sdk_tick':
        def advanced():
            sim.current_time_step_index+=1;sim.current_time+=1/480
        sim.physics_sim_view.update_articulations_kinematic=advanced
    else:
        old=rod.rope.reset_envs
        def advanced(ids):
            old(ids);rod.rope.time+=1/480
        rod.rope.reset_envs=advanced
    txn=DualHandResetTransaction(source,episode_uid='run-cpu:reset-a')
    with pytest.raises((ValueError,RuntimeError)):
        txn.run(sim,rod,bindings,owner_resets=owners)
    assert txn.status=='failed'
    assert rod.clock()['clock_valid'] is False
    assert rod.rope.steps==0
    with pytest.raises(RuntimeError):
        txn.run(sim,rod,bindings,owner_resets=owners)


def test_new_episode_reset_increments_once_and_rejects_reused_uid(monkeypatch):
    _,rod,hands,bindings,sim,source,owners=setup(monkeypatch)
    a=DualHandResetTransaction(source,episode_uid='run-cpu:reset-a').run(sim,rod,bindings,owner_resets=owners)
    b=DualHandResetTransaction(source,episode_uid='run-cpu:reset-b').run(sim,rod,bindings,owner_resets=owners)
    assert a['after_clock']['reset_id']==1 and b['after_clock']['reset_id']==2
    assert rod.rope.resets==2 and rod.rope.steps==0
    with pytest.raises(ValueError):
        DualHandResetTransaction(source,episode_uid='run-cpu:reset-a').run(sim,rod,bindings,owner_resets=owners)
    assert rod.rope.resets==2


def test_missing_sensor_source_is_explicitly_unknown_not_filled_with_zero(monkeypatch):
    _,rod,hands,bindings,sim,source,owners=setup(monkeypatch)
    owners['contact']=lambda context:dict(owner='contact',status='unavailable',
        source='no post-reset contact sample',episode_uid=context['episode_uid'])
    result=DualHandResetTransaction(source,episode_uid='run-cpu:reset-a').run(sim,rod,bindings,owner_resets=owners)
    assert result['owner_resets']['contact']['status']=='unavailable'
    assert result['rod_reset']['distributed_contact'] is None
    assert result['rod_reset']['named_wrenches'] is None
    assert result['r0_reset_complete'] is False


@pytest.mark.parametrize('fault',['missing_getter','pad_index','root_path','passive_limit'])
def test_sources_and_actual_pad_identity_fail_before_any_reset(monkeypatch,fault):
    _,rod,hands,bindings,sim,source,owners=setup(monkeypatch)
    if fault=='missing_getter':
        del hands[1].root_physx_view.get_dof_actuation_forces
    elif fault=='pad_index':
        rod.bindings['right.'+PADS[0]]['body_index']=10
    elif fault=='root_path':
        wrong='/World/unrelated/rightHand/wrist_fixed'
        hands[1].root_physx_view.prim_paths=[wrong]
        bindings[1].env_paths[7]=wrong
    else:
        params={k:source[k] for k in ('origin','root_xy','initial_offset_m',
            'joint_positions','joint_velocities','source_description')}
        params['joint_positions'][0][JOINTS.index('finger13')]=9.
        with pytest.raises(ValueError):
            prepare_reset_state(hand_roots=dict(zip(('left','right'),source['hand_roots'])),**params)
        return
    with pytest.raises((ValueError,AttributeError)):
        DualHandResetTransaction(source,episode_uid='run-cpu:reset-a').run(
            sim,rod,bindings,owner_resets=owners)
    assert not hands[0].events and rod.rope.resets==0


def test_authored_seed_offset_and_owned_receipt_do_not_retain_handles(monkeypatch):
    import gc,weakref
    events,rod,hands,bindings,sim,source,owners=setup(monkeypatch)
    roots=np.asarray(source['hand_roots'])
    expected=roots.copy()
    expected[:,:3]+=source['origin']
    expected[:,:2]+=source['root_xy']
    np.testing.assert_array_equal(source['authored_root_seed_w'],expected)
    expected[:,2]+=.04
    np.testing.assert_array_equal(source['reset_root_pose_w'],expected)
    refs=[weakref.ref(obj) for obj in (rod,sim.physics_sim_view.update_articulations_kinematic,*hands)]
    txn=DualHandResetTransaction(source,episode_uid='run-cpu:reset-a')
    result=txn.run(sim,rod,bindings,owner_resets=owners)
    hands[0].measured[:]=9
    assert not (result['hands']['left']['q']==9).all()
    del rod,hands,bindings,sim,owners
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_reset_can_recover_nonfinite_pre_state_without_claiming_it_valid(monkeypatch):
    _,rod,hands,bindings,sim,source,owners=setup(monkeypatch)
    hands[0].measured[:]=float('nan')
    result=DualHandResetTransaction(source,episode_uid='run-cpu:reset-a').run(
        sim,rod,bindings,owner_resets=owners)
    assert result['before_hands']['left']['all_finite'] is False
    assert np.isnan(result['before_hands']['left']['q']).all()
    assert result['hands']['left']['all_finite'] is True
