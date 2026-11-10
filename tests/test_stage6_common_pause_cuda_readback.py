"""Real CUDA Tensor masks + CPU SDK-format arrays, without importing Isaac.

Run explicitly with CUDA_VISIBLE_DEVICES=2. CPU-only test runs skip this file;
they never serve as evidence that the CUDA/native device boundary was covered.
"""
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real CUDA device required; CPU fixture is not CUDA coverage')
@pytest.mark.parametrize('active', [8, 5])
def test_after_advance_compares_cpu_sdk_readback_using_cuda_live_mask_without_changing_control(monkeypatch,active):
    from go2_pvcnn.control.dh116 import tabletop_bimanual_tracking_pause_runtime as module
    device=torch.device('cuda:0');n=8
    live=torch.arange(n,device=device)<active;original_live=live.clone()
    logical=torch.arange(n*12,device=device,dtype=torch.float64).reshape(n,12)/113.
    before=logical.clone();backend=logical.float().cpu()
    # Inactive slots may already contain original reset targets. Confirmation
    # applies only to the genuinely live submitted slots on this physics step.
    backend[active:]+=1.;backend_before=backend.clone()
    runtime=module.CommonTrackingPauseRuntime(n,teacher_error_p999_rad=torch.zeros(12,device=device,dtype=torch.float64),enabled=True)
    earlier=dict(sdk_physics_step=18,sdk_timestamp_s=0.03750000195577741)
    later=dict(sdk_physics_step=19,sdk_timestamp_s=0.03958333539776504)
    runtime.pending=dict(candidate_motion_qualified=True,physics_step_before=0,live=live.clone(),
        logical_command_after_pause_rad=logical.clone(),dt_s=1/480,
        target_readback_clock_before_control=earlier,counts_as_student_success=False)
    evidence=dict(backend_active_rad=backend,isaaclab_command_cache_active_rad=backend.clone(),hands=[],clock=later)
    monkeypatch.setattr(module,'read_common_pause_targets',lambda world:evidence)
    world=SimpleNamespace(n=n,_physics_steps=1,commands=logical.reshape(n,2,6),
        root_targets=torch.zeros(n,2,7,device=device,dtype=torch.float64))
    result=runtime.after_advance(world,SimpleNamespace(),live)
    assert result['native_submission_confirmed'] is True and result['counts_as_student_success'] is False
    assert result['target_readback_clock_after_advance']==later and result['target_readback_clock_before_control']==earlier
    assert result['backend_target_readback_after_advance_rad'].device.type=='cpu'
    assert result['logical_command_submitted_rad'].device.type=='cuda'
    assert torch.equal(backend,backend_before) and torch.equal(logical,before) and torch.equal(live,original_live)
    assert runtime.pending is None and runtime.records==[result]
