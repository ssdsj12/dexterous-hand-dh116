"""Pure native-boundary fixtures; they do not count as Isaac experiments."""
import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch


def module():
    name='go2_pvcnn.control.dh116.tabletop_tripod_execution_guard'
    assert importlib.util.find_spec(name) is not None, 'post-guard recovery boundary missing'
    return importlib.import_module(name)


def inputs():
    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    low,high=(torch.tensor(x,dtype=torch.float64).repeat(2) for x in active_limits())
    command=((low+high)/2)[None].repeat(2,1)
    task=SimpleNamespace(command=command.clone(),low=low,high=high,
        tripod_recovery_stage='A',
        free_wrist_world=SimpleNamespace(mixed_reset_style_ids=torch.tensor([0,1])),
        m={'q':command.clone(),'effective_pad_loads_n':torch.zeros(2,2,5,dtype=torch.float64)})
    mask=torch.zeros(2,12,dtype=torch.bool);mask[1,6:10]=True
    proposal=command+0.0005
    return task,proposal,mask,torch.ones(2,dtype=torch.bool),torch.zeros(2,24)


def test_absent_hook_preserves_original_tensor_and_legacy_actions():
    t,p,mask,live,a=inputs()
    assert module().apply_post_guard_recovery(t,p,live,1/480,torch.zeros(2,26)) is p


def test_only_enabled_post_guard_motion_can_be_reduced():
    t,p,mask,live,a=inputs();seen=[]
    def reduce(packet):
        seen.append(packet)
        return torch.where(mask,(packet['governed_target_rad']+packet['command_rad'])/2,
                           packet['governed_target_rad'])
    t.tripod_recovery_runtime=SimpleNamespace(finger_mask=mask,filter_post_guard_targets=reduce)
    result=module().apply_post_guard_recovery(t,p,live,1/480,a)
    assert torch.equal(result[~mask],p[~mask])
    torch.testing.assert_close(result[mask],((p+t.command)/2)[mask])
    assert len(seen)==1 and torch.equal(seen[0]['requested_action24'],a)
    assert torch.equal(t.command,t.m['q'])


@pytest.mark.parametrize('case',['pinch','left','ring','inactive'])
def test_scope_escape_is_rejected_before_native_submission(case):
    t,p,mask,live,a=inputs()
    if case=='inactive':live[1]=False
    def invalid(packet):
        out=packet['governed_target_rad'].clone()
        index={'pinch':(0,6),'left':(1,0),'ring':(1,10),'inactive':(1,6)}[case]
        out[index]=t.command[index]
        return out
    t.tripod_recovery_runtime=SimpleNamespace(finger_mask=mask,filter_post_guard_targets=invalid)
    with pytest.raises(ValueError,match='scope'):
        module().apply_post_guard_recovery(t,p,live,1/480,a)


@pytest.mark.parametrize('case',['faster','reverse','nan','shape','dtype'])
def test_callback_cannot_bypass_original_guard(case):
    t,p,mask,live,a=inputs()
    def invalid(packet):
        out=packet['governed_target_rad'].clone()
        if case=='faster':out[1,6]+=.0001
        if case=='reverse':out[1,6]=t.command[1,6]-.0001
        if case=='nan':out[1,6]=float('nan')
        if case=='shape':out=out[:,:11]
        if case=='dtype':out=out.float()
        return out
    t.tripod_recovery_runtime=SimpleNamespace(finger_mask=mask,filter_post_guard_targets=invalid)
    with pytest.raises(ValueError):module().apply_post_guard_recovery(t,p,live,1/480,a)


def test_callback_receives_owned_snapshots():
    t,p,mask,live,a=inputs();original=p.clone()
    def mutate(packet):
        packet['command_rad'].zero_();packet['measured_q_rad'].zero_()
        packet['requested_action24'].fill_(1.)
        return packet['governed_target_rad']
    t.tripod_recovery_runtime=SimpleNamespace(finger_mask=mask,filter_post_guard_targets=mutate)
    assert torch.equal(module().apply_post_guard_recovery(t,p,live,1/480,a),original)
    assert torch.equal(t.command,t.m['q']) and not bool((t.command==0).all())
    assert not bool(a.any())


def test_actual_bearing_pressure_relief_cannot_be_cancelled():
    t,p,mask,live,a=inputs()
    t.m['effective_pad_loads_n'][1,1,0]=2.
    p[1,6]=t.command[1,6]-.0001
    def cancel(packet):
        out=packet['governed_target_rad'].clone();out[1,6]=packet['command_rad'][1,6]
        return out
    t.tripod_recovery_runtime=SimpleNamespace(finger_mask=mask,filter_post_guard_targets=cancel)
    with pytest.raises(ValueError,match='pressure|scope'):
        module().apply_post_guard_recovery(t,p,live,1/480,a)


@pytest.mark.parametrize('bad_index',[(0,6),(1,0),(1,10)])
def test_runtime_mask_is_checked_against_native_style_and_stage(bad_index):
    t,p,mask,live,a=inputs();mask[bad_index]=True
    t.tripod_recovery_runtime=SimpleNamespace(finger_mask=mask,
        filter_post_guard_targets=lambda packet:packet['governed_target_rad'])
    with pytest.raises(ValueError,match='scope'):
        module().apply_post_guard_recovery(t,p,live,1/480,a)


def test_pause_cancels_both_queued_heights_against_measured_palms_only_for_tripod():
    t,p,mask,live,a=inputs()
    measured=torch.tensor([[[.01,.02,.003],[.02,.01,.004]],
                           [[.01,.02,.005],[.02,.01,.006]]],dtype=torch.float64)
    w=SimpleNamespace(palms=torch.full((2,2,3),.03,dtype=torch.float64))
    t.m['adaptive_palm_offset']=measured.clone()
    t.palm_targets=torch.full_like(w.palms,.04)
    oldw=w.palms.clone();oldt=t.palm_targets.clone()
    pause=torch.tensor([False,True])
    module().cancel_queued_common_ascent(w,t,pause)
    assert torch.equal(w.palms[0],oldw[0]) and torch.equal(t.palm_targets[0],oldt[0])
    assert torch.equal(w.palms[1,:,2],measured[1,:,2])
    assert torch.equal(t.palm_targets[1,:,2],measured[1,:,2])
    assert torch.equal(w.palms[1,:,:2],oldw[1,:,:2])
    assert torch.equal(t.palm_targets[1,:,:2],oldt[1,:,:2])
    assert torch.equal(t.m['adaptive_palm_offset'],measured)


def test_pause_with_unknown_measurement_fails_without_physical_state_write():
    t,p,mask,live,a=inputs();w=SimpleNamespace(palms=torch.ones(2,2,3))
    t.palm_targets=torch.ones(2,2,3);t.m['adaptive_palm_offset']=torch.full((2,2,3),float('nan'))
    before=w.palms.clone()
    with pytest.raises(ValueError):module().cancel_queued_common_ascent(w,t,torch.tensor([False,True]))
    assert torch.equal(w.palms,before)


def test_real_surface_control_invokes_callback_after_original_native_guard():
    from test_surface_exploration import surface_task
    from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import control
    t,w=surface_task();middle=(t.low+t.high)/2
    t.command[:]=middle;t.m['q'][:]=middle
    t.tripod_recovery_stage='A';w.mixed_reset_style_ids=torch.tensor([0,1])
    mask=torch.zeros(2,12,dtype=torch.bool);mask[1,6:10]=True;seen=[]
    def pause_missing(packet):
        seen.append(packet)
        return torch.where(packet['finger_mask'],packet['command_rad'],packet['governed_target_rad'])
    t.tripod_recovery_runtime=SimpleNamespace(finger_mask=mask,filter_post_guard_targets=pause_missing)
    action=torch.zeros(2,24,dtype=torch.float64);action[:,:12]=.1
    control(t,action,torch.ones(2,dtype=torch.bool),1/480)
    assert len(seen)==1
    assert (seen[0]['governed_target_rad']>middle).all()
    assert torch.equal(t.command[1,6:10],middle[6:10])
    assert (t.command[0]>middle).all()
