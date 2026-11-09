"""Fixed-wrist CPU control checks; geometry predictions never establish contact."""
import importlib
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from test_tripod_candidate_planner import fixture
from test_tripod_contact_recovery import api, observe, retry, reserve


def module():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_tripod_bilateral_recovery')


def test_explicit_scope_defaults_right_and_both_enables_only_tripod_three_digits():
    m = module(); styles = torch.tensor([0, 1, 1])
    right, fingers = m.recovery_scope(styles)
    assert right.shape == (3, 2, 5) and fingers.shape == (3, 12)
    assert right[1:, 1, :3].all() and not right[:, 0].any()
    both, fingers = m.recovery_scope(styles, 'both')
    assert both[1:, :, :3].all() and not both[0].any() and not both[:, :, 3:].any()
    assert fingers[1, [0, 1, 2, 3, 6, 7, 8, 9]].all()
    assert not fingers[:, [4, 5, 10, 11]].any()
    with pytest.raises(ValueError): m.recovery_scope(styles, 'mirror')


@pytest.mark.parametrize('hand', [0, 1])
def test_substep_lineage_cancels_only_acquired_digit_of_selected_hand(hand):
    from go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime import cancel_inapplicable_substep_actions
    f = api(); observe(f, contact=True)
    f.enabled_pad_mask[:, :, :3] = True
    t = NS(actions=torch.full((1, 24), -.1, dtype=torch.float64))
    origin = torch.zeros(1, 2, 3, dtype=torch.bool); origin[0, hand, 1] = True
    f.contact_known[:, :, :3] = True; f.measured_contact[:, :, :3] = True
    cancel_inapplicable_substep_actions(t, f, torch.tensor([True]), origin)
    assert t.actions[0, 6*hand+2] == 0
    assert t.actions[0, 6*(1-hand)+2] == -.1
    assert t.actions[0, 6*hand] == -.1  # original bearing pressure retained


def bilateral_planner():
    p, a = fixture(); m = module()
    p = m.BilateralTripodCandidatePlanner(p.geometry, active_lower=p.low,
        active_upper=p.high, full_lower=p.flow, full_upper=p.fhigh,
        limit_sources=p.limit_sources, max_candidates=1)
    return p, a


@pytest.mark.parametrize('hand', [0, 1])
def test_nonlinear_auditor_reports_independent_hand_command_arc_lengths(hand):
    p, a = bilateral_planner(); final = a['current_command'].copy()
    final[hand, 2] += np.float32(.00001)
    out = p.audit_commands(final_next_command=final, **a)
    assert out['accepted'], out['rejection_reasons']
    assert out['bilateral_approach_command_m'].shape == (2, 3)
    assert out['bilateral_approach_command_m'][hand, 1] > 0
    assert not out['bilateral_approach_command_m'][1-hand].any()
    assert not out['native_contact_verified']


@pytest.mark.parametrize('hand', [0, 1])
def test_missing_pad_uses_its_own_fk_and_preserves_other_hand_pressure(hand):
    p, a = bilateral_planner(); model = p.geometry.models[hand]; snap = a['snapshots'][hand]
    patch = model.patches['index']; pose = snap['poses'][patch['body']]
    c = pose[:3, :3] @ np.asarray(patch['center_local_m']) + pose[:3, 3]
    normal = pose[:3, :3] @ np.asarray(patch['reference_outward_normal_local'])
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS
    angle = snap['full_q'][JOINTS.index('finger22')]
    p.geometry.scene['rod'] = dict(pose_w=[*(c+normal*.01005), np.cos(angle/2), 0., 0., np.sin(angle/2)], radius_m=.01, length_m=.01)
    missing = np.zeros((2, 3), bool); missing[hand, 1] = True
    out = p.plan(missing_hands=missing, remaining_approach_m=np.ones((2, 3))*.001, substeps=8, **a)
    assert out['accepted'], (out.get('reason'), out.get('rejection_reasons'))
    np.testing.assert_array_equal(np.flatnonzero(out['action24']), [6*hand+2])
    assert out['objective']['bearing_weight'] == 10
    assert out['budget_consumed_m'] == 0 and not out['native_contact_verified']
    assert out['bilateral_approach_command_m'][hand, 1] <= .001/60


def test_fixed_wrist_planner_never_admits_moving_wrist_even_with_geometry_permission():
    p, a = bilateral_planner()
    with pytest.raises(ValueError, match='fixed wrist'):
        p.plan(missing_hands=np.ones((2, 3), bool), remaining_approach_m=np.ones((2, 3))*.001,
            wrist_motion_requested=True, cooked_live_qualified=True, **a)


def test_bilateral_ledger_flicker_does_not_refill_other_hand_or_episode_budget():
    m = module(); enabled, _ = m.recovery_scope(torch.tensor([1, 1]), 'both')
    from go2_pvcnn.control.dh116.tabletop_tripod_contact_recovery import TripodContactRecovery
    f = TripodContactRecovery(2, 'cpu', enabled_pad_mask=enabled)
    f.reset(torch.arange(2), episode_ids=torch.zeros(2, dtype=torch.long), source_sha='a'*64)
    observe(f); retry(f); ticket = reserve(f)
    amount = ticket['allowed_distance_m'].clone()
    f.commit_execution(episode_ids=f.episode_ids, reservation_id=ticket['reservation_id'], execution_id='bilateral',
        physics_step=1, approved_approach_m=amount, executed_mask=amount>0,
        requested_action24=torch.zeros(2,24), approved_action24=torch.zeros(2,24), final_target_rad=torch.zeros(2,12))
    spent = f.executed_approach_m.clone()
    observe(f, contact=True); observe(f, contact=False)
    assert torch.equal(f.executed_approach_m, spent)
    assert f.executed_approach_m[0,0,:3].sum() > 0 and f.executed_approach_m[0,1,:3].sum() > 0


def test_teacher_transfer_has_real_contact_dwell_and_finite_unloading_without_mg_gate():
    m = module(); c = m.TripodTeacherLift(1, enabled=True, source='teacher')
    kw = dict(real_stable_contact=torch.tensor([True]), tracking_safe=torch.tensor([True]),
        object_following=torch.tensor([True]), airborne=torch.tensor([False]),
        original_hold_qualified=torch.tensor([False]), mandatory_unload=torch.tensor([False]), terminal=torch.tensor([False]))
    out = c.step(dt_s=.1, **kw); assert out['phase'].item() == c.STABILIZE_CONTACT
    assert not out['common_ascent_m_s'].any()
    c.step(dt_s=.1, **kw); out = c.step(dt_s=.1, **kw)
    assert out['phase'].item() == c.LOAD_TRANSFER
    assert out['common_ascent_m_s'].item() == pytest.approx(.0005)
    for _ in range(41): out = c.step(dt_s=.1, **kw)
    assert out['phase'].item() == c.LIFT
    assert .001 <= out['common_ascent_m_s'].item() <= .003
    kw['mandatory_unload'][:] = True
    out = c.step(dt_s=.1, **kw)
    assert not out['common_ascent_m_s'].any() and out['mandatory_unload'].all()
    with pytest.raises(ValueError, match='teacher'):
        m.TripodTeacherLift(1, enabled=True, source='student')


def test_load_transfer_failure_terminates_instead_of_closing_forever():
    m=module(); c=m.TripodTeacherLift(1, enabled=True, source='teacher', max_duration_s=1.)
    kw = dict(real_stable_contact=torch.tensor([False]), tracking_safe=torch.tensor([True]),
        object_following=torch.tensor([False]), airborne=torch.tensor([False]),
        original_hold_qualified=torch.tensor([False]), mandatory_unload=torch.tensor([False]), terminal=torch.tensor([False]))
    for _ in range(11): out=c.step(dt_s=.1, **kw)
    assert out['phase'].item()==c.TERMINATED and not out['common_ascent_m_s'].any()


def test_transfer_permits_first_unloading_motion_before_object_can_follow_and_caps_two_mm():
    m=module();c=m.TripodTeacherLift(1,enabled=True,source='teacher')
    kw=dict(real_stable_contact=torch.tensor([True]),tracking_safe=torch.tensor([True]),
        object_following=torch.tensor([False]),airborne=torch.tensor([False]),
        original_hold_qualified=torch.tensor([False]),mandatory_unload=torch.tensor([False]),terminal=torch.tensor([False]))
    c.step(dt_s=.1,**kw);out=c.step(dt_s=.1,**kw)
    assert out['phase'].item()==c.LOAD_TRANSFER and out['common_ascent_m_s'].item()==pytest.approx(.0005)
    kw['object_following'][:]=True;total=out['common_ascent_m_s'].item()*.1
    while out['phase'].item()==c.LOAD_TRANSFER:
        out=c.step(dt_s=.1,**kw);total+=out['common_ascent_m_s'].item()*.1
    assert total==pytest.approx(.002)


def test_airborne_hold_requires_original_thirty_mm_height_and_original_hold_observer():
    m=module();c=m.TripodTeacherLift(1,enabled=True,source='teacher');c.phase.fill_(c.LIFT)
    kw=dict(real_stable_contact=torch.tensor([True]),tracking_safe=torch.tensor([True]),
        object_following=torch.tensor([True]),airborne=torch.tensor([True]),
        original_hold_qualified=torch.tensor([False]),mandatory_unload=torch.tensor([False]),terminal=torch.tensor([False]))
    out=c.step(dt_s=.1,object_net_lift_m=torch.tensor([.003],dtype=torch.float64),**kw)
    assert out['phase'].item()==c.LIFT and not out['original_hold_qualified'].any()
    out=c.step(dt_s=.1,object_net_lift_m=torch.tensor([.03],dtype=torch.float64),**kw)
    assert out['phase'].item()==c.AIRBORNE_HOLD and not out['original_hold_qualified'].any()


@pytest.mark.parametrize('hand',[0,1])
def test_bearing_pressure_is_rebuilt_on_each_hand_and_never_spends_recovery_budget(hand):
    p,a=bilateral_planner();model=p.geometry.models[hand];snap=a['snapshots'][hand]
    patch=model.patches['index'];pose=snap['poses'][patch['body']]
    center=pose[:3,:3]@np.asarray(patch['center_local_m'])+pose[:3,3]
    normal=pose[:3,:3]@np.asarray(patch['reference_outward_normal_local'])
    tangent=pose[:3,:3]@np.array([0.,1.,0.])
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS
    angle=snap['full_q'][JOINTS.index('finger22')]
    p.geometry.scene['rod']=dict(pose_w=[*(center+normal*.01005),np.cos(angle/2),0.,0.,np.sin(angle/2)],radius_m=.01,length_m=.01)
    valid=np.zeros((2,5),bool);valid[hand,1]=True
    loads=np.zeros((2,5));loads[hand,1]=.5
    vectors=np.zeros((2,5,3));vectors[hand,1]=-(.95*normal+np.sqrt(1-.95**2)*tangent)*.5
    points=np.zeros((2,5,3));points[hand,1]=center
    a.update(bearing_mask=valid,bearing_provenance=dict(source='synthetic-not-native',native_accepted_contacts=True,
        physics_step=a['snapshot_identity']['physics_step'],snapshot_identity=a['snapshot_identity']),
        effective_pad_loads_n=loads,effective_pad_valid=valid,effective_pad_vectors_w=vectors,
        effective_pad_points_w_m=points,effective_pad_depth_m=np.zeros((2,5)),
        rejected_contact_loads_n=np.zeros(2),upstream_pressure_action24=np.zeros(24,np.float32))
    hold=module().HandTripodPressureHold(p,hand);out=hold.plan(**a)
    assert out['accepted'],out
    assert np.flatnonzero(out['action24']).tolist()==[6*hand+2]
    assert out['reacquisition_budget_untouched'] and not out['native_contact_verified']
    assert out['audit']['coverage']['source_hulls']==[17,17]
    # Actual runtime supplies torch command/contact buffers, not only numpy.
    for key in ('current_command','bearing_mask','effective_pad_loads_n','effective_pad_valid',
        'effective_pad_vectors_w','effective_pad_points_w_m','effective_pad_depth_m',
        'rejected_contact_loads_n','upstream_pressure_action24'):
        a[key]=torch.as_tensor(a[key])
    out=hold.plan(**a)
    assert out['accepted'] and np.flatnonzero(out['action24']).tolist()==[6*hand+2]


def test_both_hands_are_billed_only_after_exact_postguard_native_submission():
    import io
    from go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime import TripodRecoveryRuntime,commit_submitted_targets
    from go2_pvcnn.control.dh116.tabletop_tripod_contact_recovery import TripodContactRecovery
    p,a=bilateral_planner();enabled,_=module().recovery_scope(torch.tensor([1]),'both')
    f=TripodContactRecovery(1,'cpu',enabled_pad_mask=enabled)
    f.reset(torch.tensor([0]),episode_ids=torch.tensor([0]),source_sha='a'*64)
    observe(f);retry(f)
    r=TripodRecoveryRuntime.__new__(TripodRecoveryRuntime)
    r.active=True;r.recovery_hands='both';r.n=1;r.device='cpu';r.tick=437
    r.tripod=torch.tensor([True]);r.ended=torch.tensor([False]);r.ledger=f
    r.episode_ids=f.episode_ids.clone();r.pressure_step_proposals=set();r.decisions=io.StringIO()
    r.w=NS(_physics_steps=0);r.requested_actions=torch.zeros(1,24,dtype=torch.float64)
    r._planner=lambda _:p;r._contexts=lambda _:{k:a[k] for k in ('snapshots','full_qd','snapshot_identity','bearing_mask','bearing_provenance')}
    command=torch.as_tensor(a['current_command']).reshape(1,12).double();governed=command.clone()
    governed[0,[2,8]]+=.000001
    mask=torch.zeros(1,12,dtype=torch.bool);mask[0,[2,8]]=True
    packet=dict(command_rad=command,governed_target_rad=governed,finger_mask=mask,live=torch.tensor([True]))
    result=r.filter_post_guard_targets(packet)
    assert not f.executed_approach_m.any() and r.pending is not None
    assert r.pending['approved_approach_m'][0,:,1].gt(0).all()
    commit_submitted_targets(f,r.pending,NS(_physics_steps=0,commands=result.reshape(1,2,6)))
    assert f.executed_approach_m[0,:,1].gt(0).all()
    assert not f.executed_approach_m[0,:,[0,2,3,4]].any()


def test_stage_b_scope_allows_left_but_protects_real_bearing_relief():
    from test_tripod_execution_guard import inputs
    from go2_pvcnn.control.dh116.tabletop_tripod_execution_guard import apply_post_guard_recovery
    t,p,mask,live,a=inputs();t.tripod_recovery_stage='B';mask[1,:4]=True
    def shorten(packet):
        return torch.where(packet['finger_mask'],packet['command_rad'],packet['governed_target_rad'])
    t.tripod_recovery_runtime=NS(recovery_hands='both',finger_mask=mask,filter_post_guard_targets=shorten)
    result=apply_post_guard_recovery(t,p,live,1/480,a)
    assert torch.equal(result[1,:4],t.command[1,:4])
    t.m['effective_pad_loads_n'][1,0,0]=2.;p[1,0]=t.command[1,0]-.0001
    result=apply_post_guard_recovery(t,p,live,1/480,a)
    assert result[1,0]==p[1,0]


def test_fresh_mandatory_release_cannot_disappear_when_actual_contact_is_lost():
    from go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime import TripodRecoveryRuntime
    r=TripodRecoveryRuntime.__new__(TripodRecoveryRuntime)
    r.active=True;r.pending=None;r.n=1;r.device='cpu';r.tick=437;r.last_before_step_tick=437
    r.ledger=api();observe(r.ledger,contact=False)
    r.tripod=torch.tensor([True]);r.ended=torch.tensor([False]);r.acquisition_origins=torch.zeros(1,2,3,dtype=torch.bool)
    r.substep_action_origin=torch.tensor([[[0,0,0],[0,2,0]]])
    r.pressure_mandatory_mask=torch.tensor([[[False,False,False],[False,True,False]]])
    r.t=NS(done=torch.tensor([False]),actions=torch.zeros(1,24,dtype=torch.float64))
    r.t.actions[0,8]=-.01;r.w=NS(pad_measurements={});r._settled=lambda _:True
    with pytest.raises(RuntimeError,match='mandatory pressure relief'):
        r.before_step()


@pytest.mark.parametrize('hands,stage',[('right','A'),('left','B'),('both','B')])
def test_real_runtime_owns_explicit_hand_axes_and_legacy_default_scope(hands,stage,tmp_path,monkeypatch):
    import go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime as runtime
    monkeypatch.setattr(runtime,'TripodNativeEvidenceBuilder',lambda *a,**k:NS(bindings=None))
    world=NS(n=2,device='cpu',mixed_reset_style_ids=torch.tensor([0,1]),w=NS(end_support_scene=None))
    env=NS(world=world,task=NS())
    r=runtime.TripodRecoveryRuntime(env,[], 'a'*64,tmp_path,recovery_hands=hands)
    for name in ('acquisition_origins','substep_action_origin','previous_fresh_mask','pressure_mandatory_mask'):
        assert getattr(r,name).shape==(2,2,3)
    assert env.task.tripod_recovery_stage==stage
    enabled,fingers=module().recovery_scope(world.mixed_reset_style_ids,hands)
    assert torch.equal(r.ledger.enabled_pad_mask,enabled) and torch.equal(r.finger_mask,fingers)
    r.close()


def test_bilateral_no_progress_watchdog_keeps_real_bearing_hand_out_of_attempt_budget():
    from test_tripod_contact_recovery import packet
    from go2_pvcnn.control.dh116.tabletop_tripod_contact_recovery import TripodContactRecovery
    enabled,_=module().recovery_scope(torch.tensor([1]),'both')
    f=TripodContactRecovery(1,'cpu',enabled_pad_mask=enabled)
    f.reset(torch.tensor([0]),episode_ids=torch.tensor([0]),source_sha='a'*64)
    for _ in range(10):
        p=packet(f,contact=True);f.observe_substep(**p)
    retry(f)
    for _ in range(241):
        f.observe_substep(**packet(f,contact=True))
    assert f.no_progress_s[0,0,:3].ge(.5-1e-12).all() and f.round_stopped[0,0,:3].all()
    assert not f.attempt_time_s[0,1,:3].any() and f.state[0,1,:3].eq(f.STABLE).all()
