"""CPU entry contracts; fixtures never count as native recovery qualification."""
import importlib
import io
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from test_tripod_candidate_planner import fixture


def options(stage='A',hands='right'):
    return NS(tripod_recovery_stage=stage,tripod_recovery_hands=hands,
        vision_teacher_source='source.pt',vision_teacher_mpc=True,vision_teacher_mpc_profile='pressure_reference',
        vision_student=True,vision_action_distribution='tanh_gaussian',vision_camera_mode='fixed',num_envs=8,
        vision_teacher_max_control_steps=560,resume=None,vision_evaluation_checkpoint=None,
        surface_mpc=False,vision_lift_reward_experiment='R0')


@pytest.mark.parametrize('stage,hands',[('A','right'),('B','left'),('B','both')])
def test_native_staged_options_admit_only_explicit_fixed_hand_profiles(stage,hands):
    from go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime import validate_staged_options
    validate_staged_options(options(stage,hands))


@pytest.mark.parametrize('stage,hands',[('A','left'),('A','both'),('B','right'),('C','both')])
def test_staged_profiles_cannot_silently_change_A_or_enable_moving_stages(stage,hands):
    from go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime import validate_staged_options
    with pytest.raises(ValueError):validate_staged_options(options(stage,hands))


def test_B_contact_diagnostic_cannot_override_uncertified_actual_geometry():
    from go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime import TripodRecoveryRuntime
    from go2_pvcnn.control.dh116.tabletop_tripod_nonlinear_geometry import world_hull
    p,a=fixture();r=TripodRecoveryRuntime.__new__(TripodRecoveryRuntime)
    r.stage='B';r.active=True;r.tripod=torch.tensor([True]);r.ended=torch.tensor([False]);r.n=1
    r.latest_rows={0:{'stage_gate':{'pass':True},'six_pad_contiguous_s':.2}};r.geometry_blocked={};r.decisions=io.StringIO()
    r.episode_ids=torch.tensor([0]);r.tick=437;r.w=NS(_physics_steps=12)
    r._settled=lambda i:True;r._planner=lambda i:p;r._contexts=lambda i:{k:a[k] for k in ('snapshots','full_qd','snapshot_identity','bearing_mask','bearing_provenance')}
    model=p.geometry.models[0];hull=model.predict(a['snapshots'][0],np.zeros(6))['hulls'][0]
    p.geometry.obstacles.append(world_hull(hull['vertices'],np.eye(4),'original-obstacle-fixture'))
    assert not r.fixed_stage_qualified()
    assert 0 in r.geometry_blocked
    p.geometry.obstacles.pop()
    assert r.fixed_stage_qualified()


@pytest.mark.parametrize('stage,flags',[('A',(False,False,False,False)),('B',(True,False,False,False))])
def test_summary_names_only_the_fixed_stage_that_ran(stage,flags,tmp_path,monkeypatch):
    import go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime as runtime
    monkeypatch.setattr(runtime,'TripodNativeEvidenceBuilder',lambda *a,**k:NS(bindings=None))
    world=NS(n=1,device='cpu',mixed_reset_style_ids=torch.tensor([1]),w=NS(end_support_scene=None))
    r=runtime.TripodRecoveryRuntime(NS(world=world,task=NS()),[],'a'*64,tmp_path,
        recovery_hands='right' if stage=='A' else 'both')
    r.active=True
    report=r.close()
    assert tuple(report[f'stage_{name}_executed'] for name in 'BCDE')==flags
    assert report['stages_B_C_D_E_executed'] is False
