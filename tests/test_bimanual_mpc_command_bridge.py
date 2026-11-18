"""CPU command hypotheses exercise original kernels and named SDK encoding."""
from copy import deepcopy
from dataclasses import asdict,replace
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from test_bimanual_mpc_model_descriptor import fixture
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,MIMIC
from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_mounted_kinematics import descriptor_digest
from go2_pvcnn.control.dh116 import tabletop_vision_command_shadow as SHADOW
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import CONTRACT
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT

WRIST=('wrist_x','wrist_y','wrist_z','wrist_roll','wrist_pitch','wrist_yaw')


def api():
    path=Path(__file__).parents[1]/'src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_command_bridge.py'
    assert path.is_file(),'independent H8 command diagnostic bridge is missing'
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_mpc_command_bridge')


def context(*,dtype='float64',anchor_quaternion=(0.,0.,0.,1.)):
    f=fixture()
    for side,hand in f.hands.items():
        v=hand.root_physx_view;meta=v.shared_metatype
        if side=='right':
            names=meta.dof_names[5:]+meta.dof_names[:5]
            meta.dof_names=names;meta.dof_indices={n:i for i,n in enumerate(names)}
            meta.dof_types=[1 if n in WRIST[:3] else 0 for n in names]
            meta.joint_names=names+meta.joint_names[17:]
            meta.joint_indices={n:i for i,n in enumerate(meta.joint_names)}
            v.arrays['get_dof_types'][:]=meta.dof_types
            hand.joint_names=list(names)
        old=meta.link_names[-1];meta.link_names[-1]='base_link'
        meta.link_indices={n:i for i,n in enumerate(meta.link_names)}
        v.link_paths=[p.rsplit('/',1)[0]+'/base_link' if p.endswith('/'+old) else p for p in v.link_paths]
        hand.body_names=list(reversed(meta.link_names))
        names=meta.dof_names;q={n:.2 for n in ACTIVE}
        q.update({name:ratio*q[master]+offset for name,master,ratio,offset in MIMIC})
        q.update({n:0. for n in WRIST})
        v.arrays['get_dof_positions'][:]=[q[n] for n in names]
        v.arrays['get_dof_velocities'][:]=.0123
        v.arrays['get_link_transforms'][:,:,3:]=anchor_quaternion
    old=f.old.__class__(hands=f.hands,clock_reader=lambda i:f.now[i],source_reader=f.p.source_reader,enabled=True)
    samples=old.capture_batch(clocks=f.now)
    sample=f.p.capture_batch(clocks=f.now,articulation_samples=samples)[0]
    state=dict(joint_command_rad=[.2]*12,palm_command_offset_m=[[0.]*3]*2,wrist_command_angle_rad=[[0.]*3]*2)
    seed_pose=[[0.,0.,0.,1.,0.,0.,0.]]*2
    feedback=dict(root_seed_pose=seed_pose,prior_root_target_pose=seed_pose,clearance_m=[.05,.05],
        task_offset_m=0.,clearance_floor_m=.0021,live=True,allowed=True,joint_lower=[0.]*12,
        joint_upper=[1.]*12,teacher_force_controlled_digits=[[False]*5]*2)
    units=dict(joint_command_rad='rad',palm_command_offset_m='m',wrist_command_angle_rad='rad',
        root_seed_pose='world_xyz_wxyz',prior_root_target_pose='world_xyz_wxyz',clearance_m='m',
        task_offset_m='m',clearance_floor_m='m',live='boolean',allowed='boolean',joint_lower='rad',
        joint_upper='rad',teacher_force_controlled_digits='boolean')
    receipts={k:dict(clock=asdict(sample.clock),source='CPU current recorded '+k,unit=units[k],valid=True,
        descriptor_sha256=descriptor_digest(sample)) for k in (*state,*feedback)}
    profile=dict(version='dh116-conditional-command-v1',profile='surface_free_wrist_v1',
        action_contract=deepcopy(ACTION_CONTRACT),free_wrist_contract=deepcopy(CONTRACT),
        surface_policy_owned_targets=True,pinch_force_hold=False,redistribution=False,callbacks=[],
        physics_substeps=8,action_contract_sha256=SHADOW.action_contract_sha256(),
        controller_source_sha256=SHADOW.controller_source_sha256(),source_clock=asdict(sample.clock),
        descriptor_sha256=descriptor_digest(sample),source='CPU current recorded controller callback profile')
    return sample,state,feedback,receipts,profile,dtype


def run(m,c,actions=None,models=None,scope='supported_conditional_shadow'):
    s,state,feedback,receipts,profile,dtype=c
    if actions is None:actions=np.zeros((8,24))
    if models is None:models=m.freeze_feedback_models(s,actions=actions,source='explicit frozen current feedback hypothesis')
    return m.unfold_command_h8(s,current_clock=s.clock,actions=actions,command_state=state,
        current_feedback=feedback,source_receipts=receipts,feedback_models=models,
        controller_contract=profile,scope=scope,logical_dtype=dtype)


def test_64_steps_have_independent_closed_form_rates_eight_endpoints_and_twelve_direct_targets():
    m=api();c=context();u=np.zeros((8,24));u[:,:12]=.4;u[:,12]=.3;u[:,16]=-.2;u[:,18]=.25
    out=run(m,c,u)
    assert len(out.steps)==64
    assert [x.substep_index for x in out.steps if x.control_endpoint]==list(range(8,65,8))
    for k in (1,8,9,64):
        x=out.steps[k-1]
        np.testing.assert_allclose(x.command_state['joint_command_rad'],np.full(12,.2+.4*.3*k/480),atol=2e-15,rtol=0)
        np.testing.assert_allclose(x.command_state['palm_command_offset_m'],[[.3*.01*k/480,0,0],[0,-.2*.01*k/480,0]],atol=2e-15,rtol=0)
        assert x.command_state['wrist_command_angle_rad'][0][0]==pytest.approx(.25*.5*k/480)
        for side in ('left','right'):
            sdk=x.named_sdk_float32[side]
            assert sdk['joint_names']==ACTIVE+WRIST and len(sdk['values'])==12
            binding=c[0].descriptor['hands'][side]['binding']
            assert sdk['sdk_dof_indices']==tuple(list(binding['dof_names']).index(n) for n in ACTIVE+WRIST)
            assert sdk['hand_joint_indices']==tuple(list(binding['hand_joint_to_dof']).index(i) for i in sdk['sdk_dof_indices'])
            assert not any(n in sdk['joint_names'] for n,_,_,_ in MIMIC)
    assert out.steps[-1].prediction_offset_s==8/60 and not any(out.qualification.values())
    assert out.steps[-1].kind=='conditional_command_hypothesis'
    assert out.receipt['targets_are_actual_response'] is False


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_supported_profile_matches_original_shadow_each_of_64_substeps(dtype):
    m=api();c=context(dtype=dtype);u=np.linspace(-.5,.5,192).reshape(8,24)
    out=run(m,c,u);d=torch.float32 if dtype=='float32' else torch.float64
    state={k:torch.tensor(v,dtype=d)[None] for k,v in c[1].items()}
    for k,x in enumerate(out.steps):
        ctx={name:torch.tensor(m._thaw(value),dtype=torch.bool if name in ('live','allowed','teacher_force_controlled_digits','active_controller_flags','source_valid') else d)[None]
            for name,value in x.kernel_context.items()}
        state,expected=SHADOW.shadow_step(state,torch.tensor(u[k//8],dtype=d)[None],ctx,c[4])
        for key in state:assert np.array_equal(np.asarray(x.command_state[key]),state[key][0].numpy())
        assert np.array_equal(np.asarray(x.diagnostics['root_pose']),expected['root_pose'][0].numpy())
        assert np.array_equal(np.asarray(x.named_sdk_float32['left']['values'][:6]),expected['sdk_float32']['finger'][0,:6].numpy())
        assert np.array_equal(np.asarray(x.named_sdk_float32['right']['values'][:6]),expected['sdk_float32']['finger'][0,6:].numpy())


def test_mount_encoding_uses_measured_rotated_anchor_and_float32_not_world_translation():
    m=api();c=context(anchor_quaternion=(0.,0.,np.sqrt(.5),np.sqrt(.5)))
    u=np.zeros((8,24));u[:,12]=1.;out=run(m,c,u)
    last=out.steps[-1];sdk=last.named_sdk_float32['left']
    np.testing.assert_allclose(sdk['values'][6:9],[0.,-64*.01/480,0.],atol=2e-9,rtol=0)
    assert sdk['values'][6:9]!=last.command_state['palm_command_offset_m'][0]
    assert sdk['values'][7]==float(np.float32(-64*.01/480))


def test_frozen_q_lead_and_actual_clearance_projection_remain_visible_with_zero_action():
    m=api();c=context();c[1]['joint_command_rad']=[.6]*12;c[2]['clearance_m']=[-.001,.05]
    out=run(m,c);first=out.steps[0];last=out.steps[-1]
    assert first.diagnostics['raw_increment_finger'][0]==.6
    assert last.command_state['joint_command_rad'][0]==pytest.approx(.6-.3*64/480)
    assert first.command_state['palm_command_offset_m'][0][2]==pytest.approx(.0031)
    assert first.diagnostics['safe_z_m'][0]==pytest.approx(.0031)
    assert first.diagnostics['projection_reasons'][0]&2
    assert not out.receipt['complete_teacher_equivalence']


def test_external_full17_feedback_changes_lead_without_becoming_twelve_target_response():
    m=api();c=context();c[1]['joint_command_rad']=[.29]*12;u=np.zeros((8,24));u[:,:12]=1.
    models=m.freeze_feedback_models(c[0],actions=u,source='explicit freeze')
    frozen=run(m,c,u);base=m._thaw(frozen.initial_current['q_sdk17'])
    values=np.tile(base,(64,1,1));values[1:]+=.15
    models['q_sdk17'].update(mode='external_forecast',values=values.tolist(),source='CPU exogenous q forecast',candidate_independent=True)
    out=run(m,c,u,models)
    np.testing.assert_array_equal(out.steps[-1].feedback_hypotheses['q_sdk17'],values[-1])
    assert out.receipt['feedback_models']['q_sdk17']['mode']=='external_forecast'
    assert out.receipt['full17_response_generated_from_targets'] is False
    assert out.initial_current['qd_sdk17']!=out.steps[-1].named_sdk_float32['left']['values']
    assert out.steps[-1].command_state['joint_command_rad'][0]==pytest.approx(.33)
    assert out.steps[-1].command_state['joint_command_rad'][0]>frozen.steps[-1].command_state['joint_command_rad'][0]+.02


def test_teacher_callbacks_are_kept_and_can_only_use_explicit_kernel_subchain_scope():
    m=api();c=context();c[4]['callbacks']=['tripod_recovery_runtime','contact_effort','policy_lift_hold']
    c[4]['profile']='pressure_teacher_recorded_profile'
    with pytest.raises(ValueError,match='profile|callback'):run(m,c)
    out=run(m,c,scope='kernel_subchain_diagnostic')
    assert out.receipt['controller_callbacks']==tuple(c[4]['callbacks'])
    assert out.receipt['complete_teacher_equivalence'] is False
    assert out.receipt['mandatory_release_modeled'] is False
    assert not any(out.qualification.values())
    assert 'pressure' in out.receipt['uncovered_feedback']


def test_owned_output_and_all_inputs_are_unchanged():
    m=api();c=context();u=np.zeros((8,24));before=deepcopy(c[1:5]);sha=descriptor_digest(c[0]);out=run(m,c,u)
    assert c[1:5]==before and descriptor_digest(c[0])==sha and not u.any()
    c[1]['joint_command_rad'][0]=99.
    assert out.initial_current['command_state']['joint_command_rad'][0]==.2
    with pytest.raises(TypeError):out.steps[0].named_sdk_float32['left']['values'][0]=99.


@pytest.mark.parametrize('fault',['clock','descriptor','source','action_contract','substeps','version'])
def test_callback_subchain_cannot_relabel_current_profile_or_original_action_contract(fault):
    m=api();c=context();p=c[4];p['profile']='recorded_pressure_teacher';p['callbacks']=['contact_effort']
    if fault=='clock':p['source_clock']['sdk_step']+=1
    elif fault=='descriptor':p['descriptor_sha256']='0'*64
    elif fault=='source':p['source']=''
    elif fault=='action_contract':p['action_contract']['joint_rate_rad_s']=30.
    elif fault=='substeps':p['physics_substeps']=1
    else:p['version']='unknown_profile_schema'
    with pytest.raises(ValueError):run(m,c,scope='kernel_subchain_diagnostic')


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_kernel_subchain_matches_supported_shadow_math_without_claiming_callbacks(dtype):
    m=api();c=context(dtype=dtype);u=np.linspace(-.8,.8,192).reshape(8,24)
    full=run(m,c,u);part=run(m,c,u,scope='kernel_subchain_diagnostic')
    for actual,expected in zip(part.steps,full.steps):
        for key in actual.command_state:
            assert np.array_equal(np.asarray(actual.command_state[key]),np.asarray(expected.command_state[key]))
        assert actual.named_sdk_float32==expected.named_sdk_float32


def test_nonselected_inactive_sdk_limit_padding_is_not_an_observation_failure():
    m=api();c=context();s=c[0];desc=m._thaw(s.descriptor);raw=m._thaw(s.raw)
    for side in ('left','right'):
        raw['hands'][side]['articulation_input']['getters']['get_dof_limits']['value'][0]=[[float('nan')]*2]*17
    bad=replace(s,raw=m._freeze(raw));c=(bad,*c[1:])
    for r in c[3].values():r['descriptor_sha256']=descriptor_digest(bad)
    c[4]['descriptor_sha256']=descriptor_digest(bad)
    out=run(m,c)
    assert len(out.steps)==64 and not any(out.qualification.values())


def rebind(c,sample):
    for receipt in c[3].values():receipt['descriptor_sha256']=descriptor_digest(sample)
    c[4]['descriptor_sha256']=descriptor_digest(sample)
    return (sample,*c[1:])


@pytest.mark.parametrize('fault',['pose_source','pose_shape','pose_dtype','limit_shape','limit_dtype','hand_index_boolean'])
def test_actual_base_anchor_limits_and_hand_mapping_cannot_relabel_owned_metadata(fault):
    m=api();c=context();s=c[0];desc=m._thaw(s.descriptor);raw=m._thaw(s.raw)
    if fault.startswith('pose_'):
        for r in (desc['hands']['left']['readbacks']['get_link_transforms'],raw['hands']['left']['readbacks']['get_link_transforms']):
            if fault=='pose_source':r['source']='unrelated pose getter'
            elif fault=='pose_shape':r['raw_shape']=[2,7]
            else:r['raw_dtype']='unknown'
    elif fault.startswith('limit_'):
        r=raw['hands']['left']['articulation_input']['getters']['get_dof_limits']
        if fault=='limit_shape':r['raw_shape']=[17,2]
        else:r['raw_dtype']='unknown'
    else:
        desc['hands']['left']['binding']['hand_joint_to_dof'][-1]=False
        raw['hands']['left']['articulation_input']['binding']['hand_joint_to_dof'][-1]=False
    c=rebind(c,replace(s,descriptor=m._freeze(desc),raw=m._freeze(raw)))
    with pytest.raises(ValueError):run(m,c)


def test_full17_future_coordinate_units_must_follow_each_actual_sdk_row():
    m=api();c=context();u=np.zeros((8,24));models=m.freeze_feedback_models(c[0],actions=u,source='explicit freeze')
    models['q_sdk17']['unit']='all radians including actual prismatic rows'
    with pytest.raises(ValueError,match='unit'):run(m,c,u,models)


@pytest.mark.parametrize('fault',['missing','clock','source','unit','valid','descriptor'])
def test_current_command_and_feedback_receipts_are_mandatory_and_same_source(fault):
    m=api();c=context();r=c[3]['clearance_m']
    if fault=='missing':c[3].pop('clearance_m')
    elif fault=='clock':r['clock']['episode_uid']='some other episode'
    elif fault=='source':r['source']=''
    elif fault=='unit':r['unit']='stage_distance'
    elif fault=='valid':r['valid']=False
    else:r['descriptor_sha256']='0'*64
    with pytest.raises(ValueError):run(m,c)


@pytest.mark.parametrize('fault',['missing','clock','candidate','source','valid','freeze_value','q_na','q_shape','q_first'])
def test_future_feedback_cannot_hide_missing_response_or_wrong_assumption_binding(fault):
    m=api();c=context();u=np.zeros((8,24));models=m.freeze_feedback_models(c[0],actions=u,source='explicit freeze')
    r=models['q_sdk17']
    if fault=='missing':models.pop('q_sdk17')
    elif fault=='clock':r['start_clock']['reset_generation']+=1
    elif fault=='candidate':r['candidate_digest']='0'*64
    elif fault=='source':r['source']=''
    elif fault=='valid':r['valid_mask'][10]=False
    elif fault=='freeze_value':r['values']=[[[0.]*17]*2]*64
    else:
        q=[[list(c[0].descriptor['hands'][s]['measurements']['q']) for s in ('left','right')] for _ in range(64)]
        r.update(mode='external_forecast',values=q)
        if fault=='q_na':q[32][1][3]=None
        elif fault=='q_shape':q[-1][1].pop()
        else:q[0][1][3]+=.1
    with pytest.raises(ValueError):run(m,c,u,models)


def test_current_limits_and_canonical_candidate_are_not_clipped_or_defaulted():
    m=api();c=context();c[2]['joint_upper'][0]=11.
    with pytest.raises(ValueError,match='limit'):run(m,c)
    c=context()
    for bad in (np.full((8,24),1.01),[[False]*24]*8,np.zeros((64,24))):
        with pytest.raises(ValueError):run(m,c,bad)


def test_finite_external_feedback_that_overflows_original_logical_dtype_is_rejected():
    m=api();c=context(dtype='float32');u=np.zeros((8,24));models=m.freeze_feedback_models(c[0],actions=u,source='explicit freeze')
    q=[[list(c[0].descriptor['hands'][s]['measurements']['q']) for s in ('left','right')] for _ in range(64)]
    active_row=list(c[0].descriptor['hands']['left']['binding']['dof_names']).index(ACTIVE[0])
    q[30][0][active_row]=1.e100
    models['q_sdk17'].update(mode='external_forecast',values=q,source='explicit finite but unrepresentable external forecast')
    with pytest.raises(ValueError,match='dtype|finite'):run(m,c,u,models)


@pytest.mark.parametrize('cache_kind',['numpy','torch'])
def test_typed_float32_command_caches_cannot_be_relabelled_float64(cache_kind):
    m=api();c=context();u=np.linspace(-.8,.8,192).reshape(8,24)
    for key,value in c[1].items():
        c[1][key]=np.asarray(value,dtype=np.float32) if cache_kind=='numpy' else torch.tensor(value,dtype=torch.float32)
    with pytest.raises(ValueError,match='dtype'):run(m,c,u)
    correct=(*c[:5],'float32');actual=run(m,correct,u)
    reference=context(dtype='float32');expected=run(m,reference,u)
    for a,b in zip(actual.steps,expected.steps):
        assert a.command_state==b.command_state and a.named_sdk_float32==b.named_sdk_float32
