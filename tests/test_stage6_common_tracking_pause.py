"""Independent stateful candidate, never changes the existing guard defaults."""
import importlib
import torch
import pytest
from test_bimanual_tracking_guard import state, actions


def module():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_tracking_guard')


def candidate(n=1):
    m=module(); assert hasattr(m, 'CommonTrackingPause'), 'stateful independent candidate missing'
    return m.CommonTrackingPause(n, teacher_error_p999_rad=torch.ones(12,dtype=torch.float64)*.002, enabled=True)


def feedback(n=1):
    return dict(pose_state=state(n), backend_target_rad=torch.zeros(n,12,dtype=torch.float64),
        measured_q_rad=torch.zeros(n,12,dtype=torch.float64), measured_qd_rad_s=torch.zeros(n,12,dtype=torch.float64),
        joint_lower_rad=torch.ones(12,dtype=torch.float64)*-1, joint_upper_rad=torch.ones(12,dtype=torch.float64),
        mandatory_release_mask=torch.zeros(n,12,dtype=torch.bool), overload=torch.zeros(n,dtype=torch.bool),
        terminal=torch.zeros(n,dtype=torch.bool), dt_s=1/480)


def test_per_joint_thresholds_and_disabled_path_preserve_existing_defaults():
    m=module(); assert hasattr(m,'CommonTrackingPause')
    c=m.CommonTrackingPause(1,teacher_error_p999_rad=torch.arange(12,dtype=torch.float64)*.002)
    torch.testing.assert_close(c.trigger_rad,torch.maximum(torch.ones(12,dtype=torch.float64)*.01,torch.arange(12,dtype=torch.float64)*.003))
    raw=actions(); out,info=c.project(raw)
    assert torch.equal(out,raw) and not info['pause_hand'].any()


def test_growing_tracking_error_pauses_both_and_resume_rebases_actual_targets():
    c=candidate(); f=feedback(); raw=actions(); f['backend_target_rad'][0,8]=.02
    out,info=c.project(raw,**f)
    assert info['state'].item()==c.PAUSE_REQUESTED and info['pause_hand'].all()
    assert info['tracking_error_growing'][0,8]
    out,info=c.project(raw,**f)
    assert info['state'].item()==c.PAUSED and not out[:,12:].any()
    assert not out[:,:12].any()  # no unbounded finger closure while paused
    f['backend_target_rad'].zero_(); f['pose_state'].current_palm_target_m.fill_(.009)
    f['pose_state'].measured_palm_offset_m.fill_(.001)
    for _ in range(24): out,info=c.project(raw,**f)
    assert info['state'].item()==c.RESUME_PENDING
    out,info=c.project(raw,**f)
    assert info['state'].item()==c.NORMAL
    assert info['cancel_translation'].all() and info['hold_offset_m'].eq(.001).all()
    assert not out[:,12:].any()  # resume step cannot apply an old queued target


def test_limit_prediction_and_mandatory_release_precede_pause():
    c=candidate(2);f=feedback(2);raw=actions(2)
    f['measured_q_rad'][0,0]=.999; f['measured_qd_rad_s'][0,0]=.1
    f['backend_target_rad'][0,0]=.999
    f['mandatory_release_mask'][0,1]=True
    f['terminal'][1]=True
    out,info=c.project(raw,**f)
    assert info['predicted_limit_warning'][0,0]
    assert out[0,1]==raw[0,1] and out[0,0]==0
    assert info['state'][1]==c.TERMINATED


def test_unsafe_velocity_cannot_satisfy_resume_dwell():
    c=candidate();f=feedback();raw=actions();f['backend_target_rad'][0,0]=.02
    c.project(raw,**f);c.project(raw,**f);f['backend_target_rad'].zero_()
    f['measured_qd_rad_s'][0,0]=.2
    for _ in range(30): _,info=c.project(raw,**f)
    assert info['state'].item()==c.PAUSED


def test_native_backend_target_requires_readback_and_finite_pause_times_out():
    m=module(); assert hasattr(m,'CommonTrackingPause')
    c=m.CommonTrackingPause(1,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64), enabled=True, max_pause_s=.1)
    f=feedback();f['backend_target_rad'][0,0]=.02
    for _ in range(55): out,info=c.project(actions(),**f)
    assert info['state'].item()==c.TERMINATED
    assert not out.any()
    f=feedback();f['backend_target_rad'][0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'): c.project(actions(),**f)


def test_actual_pose_cancellation_writes_both_target_caches_without_teleport():
    from types import SimpleNamespace as NS
    m=module();c=candidate();f=feedback();raw=actions()
    f['backend_target_rad'][0,0]=.02
    f['pose_state'].measured_palm_offset_m[:]=torch.tensor([[[.001,.002,.003],[-.001,-.002,.004]]])
    _,info=c.project(raw,**f)
    world=NS(n=1,palms=torch.ones(1,2,3,dtype=torch.float64)*.009,
        wrist_angles=torch.zeros(1,2,3,dtype=torch.float64),roots=f['pose_state'].actual_base_pose_w.clone())
    task=NS(palm_targets=world.palms.clone(),rotation_targets=world.wrist_angles.clone())
    before=world.roots.clone();m.apply_common_pause_wrist_targets(world,task,info)
    assert torch.equal(world.palms,f['pose_state'].measured_palm_offset_m)
    assert torch.equal(task.palm_targets,world.palms) and torch.equal(world.roots,before)


def test_original_mandatory_wrist_unloading_precedes_common_pause():
    c=candidate();f=feedback();raw=actions();f['backend_target_rad'][0,0]=.02
    original_unloading=torch.tensor([[False,True]])
    out,info=c.project(raw,mandatory_wrist_release_mask=original_unloading,**f)
    assert out[0,12:15].eq(0).all() and out[0,18:21].eq(0).all()
    assert torch.equal(out[0,15:18],raw[0,15:18]) and torch.equal(out[0,21:24],raw[0,21:24])
    assert not info['cancel_translation'][0,1] and not info['cancel_rotation'][0,1]


@pytest.mark.parametrize('reason',['timeout','unavailable_pose'])
def test_candidate_termination_preserves_original_mandatory_unloading(reason):
    m=module();c=m.CommonTrackingPause(1,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64),
        enabled=True,max_pause_s=.1)
    f=feedback();raw=actions();f['backend_target_rad'][0,0]=.02
    f['mandatory_release_mask'][0,1]=True
    wrist_release=torch.tensor([[False,True]])
    if reason=='unavailable_pose':f['pose_state'].measured_palm_offset_m.fill_(1.)
    for _ in range(55 if reason=='timeout' else 1):
        out,info=c.project(raw,mandatory_wrist_release_mask=wrist_release,**f)
    assert info['state'].item()==c.TERMINATED
    assert out[0,1]==raw[0,1]
    assert torch.equal(out[0,15:18],raw[0,15:18]) and torch.equal(out[0,21:24],raw[0,21:24])
    assert out[0,[0,2,3,4,5,6,7,8,9,10,11]].eq(0).all()
    f['terminal'][:]=True
    out,_=c.project(raw,mandatory_wrist_release_mask=wrist_release,**f)
    assert not out.any()  # original terminal retains final authority


@pytest.mark.parametrize('reason',['quaternion','target_chart'])
def test_unusable_actual_pose_terminates_even_when_joint_tracking_is_healthy(reason):
    c=candidate();f=feedback()
    if reason=='quaternion':f['pose_state'].actual_base_pose_w[...,3:]=0.
    else:f['pose_state'].measured_palm_offset_m.fill_(1.)
    out,info=c.project(actions(),**f)
    assert info['state'].item()==c.TERMINATED and not out.any()
