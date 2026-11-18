"""Pure H8 hand kinematic arithmetic; no SDK, control or physical validation."""
from copy import deepcopy
from dataclasses import replace
import importlib
from pathlib import Path

import numpy as np
import pytest

from test_bimanual_mpc_model_descriptor import fixture
from test_bimanual_mpc_mounted_kinematics import hypothesis
from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_mounted_kinematics import descriptor_digest
from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_objective import candidate_digest


def api():
    path=Path(__file__).parents[1]/'src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_dynamics.py'
    assert path.is_file(),'pure H8 mounted hand rollout core is missing'
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_mpc_dynamics')


def context():
    f=fixture();sample=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]
    kin=importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_mpc_mounted_kinematics')
    return sample,{side:hypothesis(kin,sample,side) for side in ('left','right')}


def forecast(m,s,actions=None):
    if actions is None:actions=np.zeros((8,24))
    values=np.empty((64,2,17))
    for k in range(64):
        for side in range(2):values[k,side]=(side+1)*np.arange(1,18)*.001+(k+1)*.0001
    return m.make_velocity_forecast(s,actions=actions,values=values,valid_mask=np.ones((64,2,17),dtype=bool),
        model_source='CPU full17 externally supplied response; numerical hypothesis only',
        model_kind='external_candidate_conditioned',candidate_independent=False)


def rollout(m,s,h,f,actions=None):
    return m.rollout_kinematic_h8(s,current_clock=s.clock,actions=np.zeros((8,24)) if actions is None else actions,
        coordinate_hypotheses=h,forecast=f)


def test_independent_hand_calculation_integrates_64_intervals_and_eight_control_endpoints():
    m=api();s,h=context();f=forecast(m,s);out=rollout(m,s,h,f)
    assert len(out.states)==64 and out.physics_period_s==1/480 and out.control_period_s==1/60
    assert out.horizon==8 and out.substeps_per_control==8
    assert [x.substep_index for x in out.states]==list(range(1,65))
    assert [x.substep_index for x in out.states if x.control_endpoint]==list(range(8,65,8))
    assert out.states[-1].prediction_offset_s==8/60
    assert not isinstance(out.states[0],type(s.clock))
    for side_index,side in enumerate(('left','right')):
        q0=np.asarray(s.descriptor['hands'][side]['measurements']['q'])
        for k in (1,8,9,32,64):
            expected=q0+((side_index+1)*np.arange(1,18)*.001*k+.0001*k*(k+1)/2)/480
            np.testing.assert_allclose(out.states[k-1].q_sdk17[side],expected,atol=2e-12,rtol=0)
        assert out.states[8].control_index==1 and out.states[8].within_control_substep==1
    assert out.kind=='kinematic_rollout_hypothesis' and not any(out.qualification.values())


def test_full_fk_has_independent_left_x_right_y_axes_and_mixed_units_nonzero_followers():
    m=api();s,h=context();f=forecast(m,s);out=rollout(m,s,h,f)
    last=out.states[-1]
    for side,axis in (('left',0),('right',1)):
        b=s.descriptor['hands'][side]['binding'];q=np.asarray(last.q_sdk17[side])
        p=[i for i,t in enumerate(b['dof_types']) if t==1];r=[i for i,t in enumerate(b['dof_types']) if t==0]
        assert len(p)==3 and len(r)==14
        expected=np.zeros(3);expected[axis]=sum(q[i] for i in p)
        pose=np.asarray(last.fk[side].link_poses_world)[22]
        np.testing.assert_allclose(pose[:3,3],expected,atol=1e-10,rtol=0)
        angle=sum(q[i] for i in r)
        c,t=np.cos(angle),np.sin(angle)
        rotation=np.array([[1,0,0],[0,c,-t],[0,t,c]]) if axis==0 else np.array([[c,0,t],[0,1,0],[-t,0,c]])
        np.testing.assert_allclose(pose[:3,:3],rotation,atol=2e-12,rtol=0)
        for follower in ('finger13','finger22','finger32','finger42','finger52'):
            row=list(b['dof_names']).index(follower)
            assert q[row]!=s.descriptor['hands'][side]['measurements']['q'][row]
        assert out.forecast_receipt['velocity_units'][side]==tuple('m/s' if t==1 else 'rad/s' for t in b['dof_types'])


def test_current_actual_is_separate_from_future_and_measured_constant_velocity_is_candidate_independent():
    m=api();s,h=context();u=np.zeros((8,24));v=u.copy();v[-1,23]=.75
    a=m.measured_constant_velocity_forecast(s,actions=u,model_source='explicit constant measured velocity exogenous assumption')
    b=m.measured_constant_velocity_forecast(s,actions=v,model_source='explicit constant measured velocity exogenous assumption')
    x=rollout(m,s,h,a,u);y=rollout(m,s,h,b,v)
    assert x.candidate_digest!=y.candidate_digest and a.candidate_independent is True
    for side in ('left','right'):
        q0=np.asarray(s.descriptor['hands'][side]['measurements']['q']);qd=np.asarray(s.descriptor['hands'][side]['measurements']['qd'])
        np.testing.assert_allclose(x.states[-1].q_sdk17[side],q0+qd*8/60,atol=2e-12,rtol=0)
        assert x.initial_current_actual['q_sdk17'][side]==tuple(q0)
        assert x.states[-1].q_sdk17[side]==y.states[-1].q_sdk17[side]
        assert x.initial_geometry_hypothesis[side].kind=='geometry_hypothesis'
    assert x.forecast_receipt['candidate_independent'] is True
    assert x.forecast_receipt['action_response_qualified'] is False


@pytest.mark.parametrize('bad',[np.zeros((64,24)),np.zeros((8,23)),[[False]*24]*8,[[None]*24]*8,
    np.full((8,24),np.nan),np.full((8,24),np.inf),np.full((8,24),1.0001),np.full((8,24),-1.0001)])
def test_canonical_candidate_shape_range_boolean_and_na_are_strict(bad):
    with pytest.raises(ValueError):api().canonical_candidate_digest(bad)


def test_candidate_digest_reuses_published_objective_protocol_exactly():
    m=api();u=np.linspace(-1,1,192).reshape(8,24)
    assert m.canonical_candidate_digest(u)==candidate_digest(u.tolist())


@pytest.mark.parametrize('field,value',[('candidate_digest','0'*64),('descriptor_sha256','0'*64),('model_source',''),
    ('start_clock',None)])
def test_wrong_candidate_descriptor_clock_or_missing_model_source_is_rejected(field,value):
    m=api();s,h=context();f=forecast(m,s)
    with pytest.raises(ValueError):rollout(m,s,h,replace(f,**{field:value}))


def test_wrong_source_binding_and_per_row_units_cannot_be_relabelled():
    m=api();s,h=context();f=forecast(m,s)
    changed=deepcopy(m._thaw(f.source_bindings));changed['right']['source_identity']['stage_identity']='different stage'
    with pytest.raises(ValueError,match='source'):rollout(m,s,h,replace(f,source_bindings=changed))
    units=m._thaw(f.velocity_units);units['left'][0]='m/s' if units['left'][0]=='rad/s' else 'rad/s'
    with pytest.raises(ValueError,match='unit'):rollout(m,s,h,replace(f,velocity_units=units))


def test_missing_passive_forecast_or_false_validity_is_not_zero_filled():
    m=api();s,h=context();f=forecast(m,s)
    values=m._thaw(f.values);mask=m._thaw(f.valid_mask);values[0][0][0]=None;mask[0][0][0]=False
    with pytest.raises(ValueError):rollout(m,s,h,replace(f,values=values,valid_mask=mask,data_valid=False,missing_reason='passive response unavailable'))
    with pytest.raises(ValueError):rollout(m,s,h,replace(f,values=np.zeros((8,2,17))))
    with pytest.raises(ValueError):rollout(m,s,h,replace(f,valid_mask=np.zeros((64,2,17),dtype=bool)))


@pytest.mark.parametrize('changes',[{'terminal':True},{'reset':True},{'episode_uid':'wrong episode'},{'sdk_step':999}])
def test_nonlive_or_mismatched_current_clock_cannot_seed_a_forecast(changes):
    m=api();s,h=context();f=forecast(m,s)
    with pytest.raises(ValueError):m.rollout_kinematic_h8(s,current_clock=replace(s.clock,**changes),
        actions=np.zeros((8,24)),coordinate_hypotheses=h,forecast=f)


def test_input_bytes_and_immutable_output_are_preserved_and_targets_never_replace_q():
    m=api();s,h=context();u=np.zeros((8,24));before=descriptor_digest(s);h0=deepcopy(h);f=forecast(m,s,u)
    values=m._thaw(f.values);out=rollout(m,s,h,f,u)
    assert descriptor_digest(s)==before and h==h0 and m._thaw(f.values)==values and not u.any()
    with pytest.raises(TypeError):out.initial_current_actual['q_sdk17']['left'][0]=99
    with pytest.raises(TypeError):out.states[0].q_sdk17['left'][0]=99
    with pytest.raises(ValueError):rollout(m,s,h,replace(f,model_kind='measured_constant_velocity',candidate_independent=False))


@pytest.mark.parametrize('fault',['unit','getter_source','articulation_source','root_readback'])
def test_current_actual_receipt_and_raw_source_cannot_be_relabelled(fault):
    m=api();s,_=context();desc=m._thaw(s.descriptor);raw=m._thaw(s.raw)
    if fault=='unit':desc['hands']['left']['measurement_receipts']['q']['unit']='N'
    elif fault=='getter_source':desc['hands']['left']['measurement_receipts']['qd']['source']='some unrelated getter'
    elif fault=='articulation_source':raw['hands']['left']['articulation_input']['source_identity']['stage_identity']='different source'
    else:raw['hands']['left']['readbacks']['get_link_transforms']['selected'][0][0]=123.
    bad=replace(s,descriptor=m._freeze(desc),raw=m._freeze(raw))
    with pytest.raises(ValueError):m.measured_constant_velocity_forecast(bad,actions=np.zeros((8,24)),model_source='explicit CV assumption')


def test_declared_valid_forecast_cannot_simultaneously_claim_missing_data():
    m=api();s,h=context();f=forecast(m,s)
    with pytest.raises(ValueError):rollout(m,s,h,replace(f,missing_reason='response unavailable'))


def test_unrelated_inactive_batch_nan_padding_does_not_invalidate_the_finite_selected_seed():
    m=api();f=fixture()
    for hand in f.hands.values():hand.root_physx_view.arrays['get_link_transforms'][0]=np.nan
    old=f.old.__class__(hands=f.hands,clock_reader=lambda i:f.now[i],source_reader=f.p.source_reader,enabled=True)
    samples=old.capture_batch(clocks=f.now,active=(True,False))
    s=f.p.capture_batch(clocks=f.now,articulation_samples=samples,active_mask=(True,False))[0]
    assert s.collection_valid
    # A CLOSED JSON/bits decoder creates distinct NaN objects in each copy.
    desc=m._thaw(s.descriptor);raw=m._thaw(s.raw)
    for side in ('left','right'):
        desc['hands'][side]['readbacks']['get_link_transforms']['value'][0][0][0]=float('nan')
        raw['hands'][side]['readbacks']['get_link_transforms']['value'][0][0][0]=float('nan')
    s=replace(s,descriptor=m._freeze(desc),raw=m._freeze(raw))
    forecast=m.measured_constant_velocity_forecast(s,actions=np.zeros((8,24)),model_source='explicit CV extrapolation')
    kin=importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_mpc_mounted_kinematics')
    out=rollout(m,s,{side:hypothesis(kin,s,side) for side in ('left','right')},forecast)
    assert len(out.states)==64 and not any(out.qualification.values())


def test_unknown_descriptor_schema_cannot_seed_an_old_coordinate_interpretation():
    m=api();s,_=context();desc=m._thaw(s.descriptor);desc['version']='unreviewed-frame-schema'
    bad=replace(s,descriptor=m._freeze(desc))
    with pytest.raises(ValueError):m.measured_constant_velocity_forecast(bad,actions=np.zeros((8,24)),model_source='explicit CV assumption')


def test_measured_constant_velocity_label_cannot_hide_a_different_supplied_response():
    m=api();s,h=context()
    f=m.measured_constant_velocity_forecast(s,actions=np.zeros((8,24)),model_source='explicit CV extrapolation')
    values=m._thaw(f.values);values[32][1][6]+=1.
    with pytest.raises(ValueError):rollout(m,s,h,replace(f,values=values))


@pytest.mark.parametrize('fault',['before_clock','after_clock','missing_before','missing_after',
    'missing_current_env','duplicate_current_env','after_error','missing_brackets'])
def test_current_seed_rejects_torn_or_missing_original_getter_clock_brackets(fault):
    m=api();s,_=context();raw=m._thaw(s.raw)
    receipt=next(r for r in raw['read_brackets'] if r['name']=='left.get_link_transforms')
    if fault in ('before_clock','after_clock'):
        edge=fault.split('_')[0]
        own=next(c for c in receipt[edge] if c['env_id']==s.clock.env_id)
        own['sdk_step']+=1
    elif fault in ('missing_before','missing_after'):receipt.pop(fault.split('_')[1])
    elif fault=='missing_current_env':
        receipt['before']=[c for c in receipt['before'] if c['env_id']!=s.clock.env_id]
        receipt['after']=deepcopy(receipt['before'])
    elif fault=='duplicate_current_env':
        own=next(c for c in receipt['before'] if c['env_id']==s.clock.env_id)
        receipt['before'].append(deepcopy(own));receipt['after']=deepcopy(receipt['before'])
    elif fault=='after_error':receipt['after_error']='RuntimeError: actual clock changed during current getter'
    else:raw.pop('read_brackets')
    bad=replace(s,raw=m._freeze(raw))
    with pytest.raises(ValueError,match='bracket'):
        m.measured_constant_velocity_forecast(bad,actions=np.zeros((8,24)),model_source='explicit CV assumption')
