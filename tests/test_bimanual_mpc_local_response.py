"""Independent arithmetic and source-bound CPU tests, never SDK/Native."""
from copy import deepcopy
from dataclasses import replace
import importlib
from pathlib import Path

import numpy as np
import pytest

from test_bimanual_mpc_model_descriptor import fixture,WRIST


def api():
    path=Path(__file__).parents[1]/'src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_local_response.py'
    assert path.is_file(),'independent full17 implicit local response core is missing'
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_mpc_local_response')


def parameters():
    return dict(q=np.zeros(17),qd=np.zeros(17),mass_matrix=np.diag(np.arange(1.,18.)),
        stiffness=np.zeros(17),damping=np.zeros(17),effort_caps=np.ones(17)*100.,drive_types=np.ones(17,dtype=int),
        position_targets=np.zeros(17),velocity_targets=np.zeros(17),generalized_load=np.zeros(17))


def context():
    f=fixture()
    for side,hand in f.hands.items():
        v=hand.root_physx_view;meta=v.shared_metatype
        if side=='right':
            meta.dof_names=meta.dof_names[5:]+meta.dof_names[:5]
            meta.dof_indices={n:i for i,n in enumerate(meta.dof_names)}
            meta.dof_types=[1 if n in WRIST[:3] else 0 for n in meta.dof_names]
            meta.joint_names=meta.dof_names+meta.joint_names[17:]
            meta.joint_indices={n:i for i,n in enumerate(meta.joint_names)}
            v.arrays['get_dof_types'][:]=meta.dof_types
            hand.joint_names=list(reversed(meta.dof_names))
        v.arrays['get_dof_positions'][:]=np.arange(17)*.01
        v.arrays['get_dof_velocities'][:]=np.arange(17)*.001
        v.arrays['get_dof_stiffnesses'][:]=4.
        v.arrays['get_dof_dampings'][:]=2.
        v.arrays['get_dof_max_forces'][:]=100.
        v.arrays['get_generalized_mass_matrices'][:]=np.diag(np.arange(1.,18.))
        v.arrays['get_dof_position_targets'][:]=.2
        v.arrays['get_dof_velocity_targets'][:]=.03
    old=f.old.__class__(hands=f.hands,clock_reader=lambda i:f.now[i],source_reader=f.p.source_reader,enabled=True)
    a=old.capture_batch(clocks=f.now)[0]
    s=f.p.capture_batch(clocks=(f.now[0],),articulation_samples=(a,))[0]
    return s,a


def model_and_targets(m,s,a,actions=None):
    if actions is None:actions=np.zeros((8,24))
    model=m.LocalHandResponse.from_current(s,a,current_clock=s.clock,model_source='CPU force/none implicit local mathematical hypothesis')
    pos=np.stack([np.asarray(s.descriptor['hands'][side]['measurements']['q'])+.1 for side in ('left','right')])
    targets=m.make_target_hypothesis(s,actions=actions,position_targets=np.tile(pos,(64,1,1)),
        velocity_targets=np.zeros((64,2,17)),source='CPU explicit full17 target series; five non-direct rows supplied independently',
        interval_assumption='row k is held on (t_k,t_k+1/480]; target read/write timing not Native calibrated')
    return model,targets


def predict(model,targets,actions=None,load=None):
    return model.predict_h8(actions=np.zeros((8,24)) if actions is None else actions,targets=targets,
        generalized_load=np.zeros((64,2,17)) if load is None else load,
        load_assumption='explicit zero external generalized-load mathematical baseline, not measured absence',
        compensation_assumption='current C/G raw retained; neither compensation sign applied',
        contact_assumption='no contact term in this mathematical baseline; no physical forecast claim')


def test_two_coordinate_implicit_pd_closed_form_embedded_in_full17():
    m=api();p=parameters();dt=1/480
    p['mass_matrix'][:2,:2]=[[2.,1.],[1.,2.]]
    p['q'][:2]=[.2,-.1];p['qd'][:2]=[.3,-.4]
    p['stiffness'][:2]=[9.,4.];p['damping'][:2]=[3.,2.]
    p['position_targets'][:2]=[.4,.2];p['velocity_targets'][:2]=[.5,.6]
    p['generalized_load'][:2]=[.7,-.8]
    a=2+3*dt+9*dt*dt;b=2+2*dt+4*dt*dt
    x=2*.3-.4+dt*(9*(.4-.2)+3*.5+.7)
    y=.3+2*(-.4)+dt*(4*(.2+.1)+2*.6-.8)
    expected=np.array([(b*x-y)/(a*b-1),(a*y-x)/(a*b-1)])
    out=m.implicit_force_drive_step(**p)
    np.testing.assert_allclose(out.qd_sdk17[:2],expected,rtol=0,atol=2e-15)
    np.testing.assert_allclose(out.q_sdk17[:2],p['q'][:2]+expected*dt,rtol=0,atol=2e-15)
    tau=p['stiffness']*(p['position_targets']-np.asarray(out.q_sdk17))+p['damping']*(p['velocity_targets']-np.asarray(out.qd_sdk17))
    residual=p['mass_matrix']@((np.asarray(out.qd_sdk17)-p['qd'])/dt)-tau-p['generalized_load']
    np.testing.assert_allclose(out.equation_residual,residual,rtol=0,atol=1e-13)
    assert np.max(np.abs(residual))<1e-10


def test_cap_is_inside_implicit_equation_and_none_follower_moves_through_mass_coupling():
    m=api();p=parameters();dt=1/480
    p['mass_matrix'][:2,:2]=[[2.,1.],[1.,2.]]
    p['stiffness'][:2]=[480.,999.];p['position_targets'][:2]=[1.,-50.]
    p['effort_caps'][0]=3.;p['drive_types'][1]=0
    out=m.implicit_force_drive_step(**p)
    np.testing.assert_allclose(out.qd_sdk17[:2],[2*dt,-dt],rtol=0,atol=1e-15)
    np.testing.assert_allclose(out.q_sdk17[:2],[2*dt*dt,-dt*dt],rtol=0,atol=1e-15)
    assert out.drive_effort_hypothesis[0]==3. and out.drive_effort_hypothesis[1]==0.
    assert out.q_sdk17[1]!=0 and not out.actual_motor_measurement


def test_opposite_targets_reverse_response_and_zero_cap_is_present_not_missing():
    m=api();p=parameters();p['stiffness'][:]=1000.;p['effort_caps'][:]=2.;p['position_targets'][:]=1.
    plus=m.implicit_force_drive_step(**p);p['position_targets']*=-1
    minus=m.implicit_force_drive_step(**p)
    np.testing.assert_array_equal(np.asarray(plus.qd_sdk17),-np.asarray(minus.qd_sdk17))
    p['effort_caps'][:]=0.;zero=m.implicit_force_drive_step(**p)
    assert not any(zero.qd_sdk17) and not any(zero.drive_effort_hypothesis)


def test_one_saturated_one_unsaturated_drive_has_independent_coupled_solution():
    m=api();p=parameters();dt=1/480
    p['mass_matrix'][:2,:2]=[[2.,.5],[.5,1.]];p['qd'][:2]=[.1,-.2]
    p['stiffness'][:2]=[1e5,4.];p['damping'][:2]=[0.,3.]
    p['position_targets'][:2]=[1.,.4];p['velocity_targets'][1]=-.1;p['effort_caps'][0]=.2
    p['generalized_load'][:2]=[-.3,.7]
    diagonal=1+3*dt+4*dt*dt
    rhs0=2*.1+.5*(-.2)+dt*(.2-.3)
    rhs1=.5*.1-.2+dt*(4*.4+3*(-.1)+.7)
    expected=[(diagonal*rhs0-.5*rhs1)/(2*diagonal-.25),(2*rhs1-.5*rhs0)/(2*diagonal-.25)]
    out=m.implicit_force_drive_step(**p)
    np.testing.assert_allclose(out.qd_sdk17[:2],expected,rtol=0,atol=2e-15)
    assert out.drive_effort_hypothesis[0]==.2 and abs(out.drive_effort_hypothesis[1])<100
    assert max(abs(x) for x in out.equation_residual)<1e-10


def test_asymmetric_raw_matrix_is_used_exactly_never_silently_symmetrized():
    m=api();p=parameters();p['mass_matrix'][:2,:2]=[[2.,1.],[.25,3.]];p['generalized_load'][:2]=[1.,2.]
    out=m.implicit_force_drive_step(**p);dt=1/480;det=6-.25
    np.testing.assert_allclose(out.qd_sdk17[:2],[(3-2)*dt/det,(4-.25)*dt/det],rtol=0,atol=1e-15)


@pytest.mark.parametrize('fault',['acceleration','unknown_drive','missing_passive','bool','negative_gain','negative_cap','singular','nonfinite'])
def test_unsupported_or_missing_math_never_uses_zero_or_identity_fallback(fault):
    m=api();p=parameters()
    if fault=='acceleration':p['drive_types'][4]=2
    elif fault=='unknown_drive':p['drive_types'][4]=9
    elif fault=='missing_passive':p['position_targets']=list(p['position_targets']);p['position_targets'][4]=None
    elif fault=='bool':p['damping']=list(p['damping']);p['damping'][0]=False
    elif fault=='negative_gain':p['stiffness'][0]=-1.
    elif fault=='negative_cap':p['effort_caps'][0]=-1.
    elif fault=='singular':p['mass_matrix'][0]=0.
    else:p['generalized_load'][0]=np.inf
    with pytest.raises(ValueError):m.implicit_force_drive_step(**p)


def test_nonconverged_saturation_reports_failure_instead_of_a_complete_step():
    m=api();p=parameters();p['stiffness'][:]=1e8;p['position_targets'][:]=1.;p['effort_caps'][:]=.01
    with pytest.raises(ValueError,match='converg|residual'):m.implicit_force_drive_step(**p,max_iterations=1)


def test_step_inputs_unchanged_and_outputs_owned():
    m=api();p=parameters();p['stiffness'][:]=4.;p['position_targets'][:]=1.
    before=deepcopy(p);out=m.implicit_force_drive_step(**p)
    for key in p:np.testing.assert_array_equal(p[key],before[key])
    p['q'][0]=99.;assert out.q_sdk17[0]!=99.
    with pytest.raises(TypeError):out.q_sdk17[0]=88.


def test_64_full17_recurrences_have_independent_scalar_reference_and_eight_control_endpoints():
    m=api();s,a=context();model,targets=model_and_targets(m,s,a);out=predict(model,targets);dt=1/480
    assert len(out.states)==64 and out.states[-1].prediction_offset_s==8/60
    assert [x.substep_index for x in out.states if x.control_endpoint]==list(range(8,65,8))
    for i,side in enumerate(('left','right')):
        q=np.asarray(s.descriptor['hands'][side]['measurements']['q']).copy()
        v=np.asarray(s.descriptor['hands'][side]['measurements']['qd']).copy();mass=np.arange(1.,18.)
        for k in range(64):
            v=(mass*v+dt*4*(np.asarray(targets.position_targets[k][i])-q))/(mass+2*dt+4*dt*dt)
            q=q+dt*v
            np.testing.assert_allclose(out.states[k].q_sdk17[side],q,rtol=0,atol=2e-13)
        assert out.states[-1].q_sdk17[side]!=out.states[7].q_sdk17[side]
    assert out.error_tube is None and not any(out.qualification.values())
    assert out.receipt['future_values_are_measured'] is False
    assert out.initial_current['kind']=='current_reported_actual_readback'


def test_actual_selected_rows_independent_order_mixed_m_rad_and_twelve_direct_are_not_seventeen_targets():
    m=api();s,a=context();model,t=model_and_targets(m,s,a);out=predict(model,t)
    left=s.descriptor['hands']['left']['binding'];right=s.descriptor['hands']['right']['binding']
    assert tuple(left['dof_names'])!=tuple(right['dof_names'])
    for side in ('left','right'):
        b=s.descriptor['hands'][side]['binding'];names=list(b['dof_names'])
        assert b['selected_view_index']==1
        assert t.position_units[side]==tuple('m' if x==1 else 'rad' for x in b['dof_types'])
        assert len(t.direct_sdk_dof_indices[side])==12
        assert set(t.direct_sdk_dof_indices[side])=={names.index(x) for x in m.DIRECT}
        for follower in ('finger13','finger22','finger32','finger42','finger52'):
            j=names.index(follower)
            assert j not in t.direct_sdk_dof_indices[side]
            assert out.states[-1].q_sdk17[side][j]!=out.initial_current['q_sdk17'][side][j]
        assert model.current_parameters[side]['mass_matrix'][0][0]==1.


@pytest.mark.parametrize('fault',['clock','descriptor','candidate','binding','units','direct_indices','shape12','shape8','missing_assumption'])
def test_targets_require_exact_source_candidate_units_and_64_full17_intervals(fault):
    m=api();s,a=context();model,t=model_and_targets(m,s,a)
    with pytest.raises(ValueError):
        if fault=='clock':predict(model,replace(t,source_clock=replace(s.clock,sdk_step=s.clock.sdk_step+1)))
        elif fault=='descriptor':predict(model,replace(t,descriptor_sha256='0'*64))
        elif fault=='candidate':predict(model,replace(t,candidate_digest='0'*64))
        elif fault=='binding':
            b=m._thaw(t.source_bindings);b['right']['source_identity']['stage_identity']='elsewhere';predict(model,replace(t,source_bindings=b))
        elif fault=='units':
            u=m._thaw(t.position_units);u['left'][0]='wrong';predict(model,replace(t,position_units=u))
        elif fault=='direct_indices':
            d=m._thaw(t.direct_sdk_dof_indices);d['left'][0]=d['left'][1];predict(model,replace(t,direct_sdk_dof_indices=d))
        elif fault=='shape12':predict(model,replace(t,position_targets=np.zeros((64,2,12))))
        elif fault=='shape8':predict(model,replace(t,velocity_targets=np.zeros((8,2,17))))
        else:predict(model,replace(t,interval_assumption=''))


@pytest.mark.parametrize('fault',['wrong_clock','articulation_raw','invalid_M','torn_bracket','terminal','missing_load'])
def test_source_and_na_boundaries_reject_before_a_response_can_be_labelled_current(fault):
    m=api();s,a=context()
    with pytest.raises(ValueError):
        if fault=='wrong_clock':m.LocalHandResponse.from_current(s,a,current_clock=replace(s.clock,sdk_step=0),model_source='CPU explicit')
        elif fault=='articulation_raw':
            raw=deepcopy(a.raw);raw['hands']['left']['getters']['get_generalized_mass_matrices']['value'][1][0][0]=999.
            m.LocalHandResponse.from_current(s,replace(a,raw=raw),current_clock=s.clock,model_source='CPU explicit')
        elif fault=='invalid_M':
            fields=dict(a.fields);fields['left.generalized_mass_matrix_raw']=replace(fields['left.generalized_mass_matrix_raw'],
                value=None,valid=False,kind='unavailable',missing_reason='getter failed',timestamp_s=None)
            m.LocalHandResponse.from_current(s,replace(a,fields=fields),current_clock=s.clock,model_source='CPU explicit')
        elif fault=='torn_bracket':
            raw=deepcopy(a.raw);raw['read_brackets'][0]['after'][0]['sdk_step']+=1
            m.LocalHandResponse.from_current(s,replace(a,raw=raw),current_clock=s.clock,model_source='CPU explicit')
        elif fault=='terminal':m.LocalHandResponse.from_current(s,a,current_clock=replace(s.clock,terminal=True),model_source='CPU explicit')
        else:
            model,t=model_and_targets(m,s,a);predict(model,t,load=np.full((64,2,17),np.nan))


def test_current_sources_targets_and_model_inputs_remain_unchanged():
    m=api();s,a=context();before=m._digest(s.raw);abefore=m._digest(a.raw);model,t=model_and_targets(m,s,a)
    loads=np.zeros((64,2,17));out=predict(model,t,load=loads)
    assert m._digest(s.raw)==before and m._digest(a.raw)==abefore and not loads.any()
    assert out.initial_current['backend_position_targets']['left']==tuple([float(np.float32(.2))]*17)
    assert out.states[-1].kind=='local_force_drive_response_hypothesis'
    with pytest.raises(TypeError):out.states[0].q_sdk17['left'][0]=99.
    with pytest.raises(TypeError):model.current_parameters['left']['mass_matrix'][0][0]=99.


@pytest.mark.parametrize('edge',['before','after'])
def test_current_articulation_bracket_missing_reset_cannot_use_clock_default(edge):
    m=api();s,a=context();raw=deepcopy(a.raw)
    own=next(c for c in raw['read_brackets'][0][edge] if c['env_id']==s.clock.env_id)
    assert own.pop('reset') is False
    with pytest.raises(ValueError,match='clock|receipt'):
        m.LocalHandResponse.from_current(s,replace(a,raw=raw),current_clock=s.clock,model_source='CPU explicit')


@pytest.mark.parametrize('name',['load_assumption','compensation_assumption','contact_assumption'])
def test_no_contact_or_compensation_sign_is_silently_assumed(name):
    m=api();s,a=context();model,t=model_and_targets(m,s,a)
    hypotheses=dict(load_assumption='explicit provided load',compensation_assumption='not automatically applied',
        contact_assumption='external hypothetical contact term only')
    hypotheses[name]=''
    with pytest.raises(ValueError):model.predict_h8(actions=np.zeros((8,24)),targets=t,
        generalized_load=np.zeros((64,2,17)),**hypotheses)


def test_named_root_translation_and_follower_targets_stay_full17_owned_hypotheses():
    m=api();s,a=context();model,t=model_and_targets(m,s,a)
    positions=np.array(t.position_targets);velocities=np.array(t.velocity_targets)
    for i,side in enumerate(('left','right')):
        names=list(s.descriptor['hands'][side]['binding']['dof_names'])
        positions[:,i,names.index('wrist_x')]=.008 if side=='left' else -.006
        positions[:,i,names.index('finger13')]=.45 if side=='left' else -.3
    target=m.make_target_hypothesis(s,actions=np.zeros((8,24)),position_targets=positions,velocity_targets=velocities,
        source='explicit independent SDK-order root/follower test',interval_assumption='unvalidated held target interval hypothesis')
    before=positions.copy();out=predict(model,target);positions[:]=99.;velocities[:]=88.
    assert np.array_equal(np.asarray(target.position_targets),before)
    assert not np.any(np.asarray(target.velocity_targets))
    for i,side in enumerate(('left','right')):
        names=list(s.descriptor['hands'][side]['binding']['dof_names']);j=names.index('wrist_x')
        mass=j+1.;q=float(model.initial_current['q_sdk17'][side][j]);v=float(model.initial_current['qd_sdk17'][side][j]);dt=1/480
        for k in range(64):v=(mass*v+dt*4*(before[k,i,j]-q))/(mass+2*dt+4*dt*dt);q+=v*dt
        assert out.states[-1].q_sdk17[side][j]==pytest.approx(q,abs=2e-13)
    assert out.receipt['current_C_G_and_explicit_input_automatically_applied'] is False
    assert out.receipt['calibrated_error_tube'] is None
