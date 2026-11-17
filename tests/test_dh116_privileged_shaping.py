"""All shaping fixtures are measured-input CPU arithmetic, not native evidence."""
from copy import deepcopy
from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


def module():
    name='go2_pvcnn.control.dh116.tabletop_privileged_shaping'
    assert importlib.util.find_spec(name),'isolated R1/R2 arithmetic module missing'
    return importlib.import_module(name)


def measurement(value,unit,*,valid=None,role='measured'):
    m=module();value=torch.as_tensor(value)
    if unit!='bool':value=value.float()
    shape=value.shape[:-1] if role.endswith('_xyz') else value.shape
    if valid is None:valid=torch.ones(shape,dtype=torch.bool)
    return m.MeasuredTensor(value,valid,unit,{'scope':'CPU_fixture_not_qualified','channel':role},role)


def context(n=1):
    return module().ShapingContext(torch.ones(n,dtype=torch.bool),torch.ones(n,dtype=torch.bool),
        torch.ones(n,dtype=torch.bool),'CPU_arithmetic_phase')


def config():
    m=module()
    return m.ShapingConfig(enabled=True,pad_distance_scale_m=.02,coordination_scale_m=.02,arc_scale_m=.02,
        coefficients={name:(1. if name.startswith('geometry_') or name=='contact_recovery' else -1.) for name in m.TERM_NAMES})


def geometry(offset=.02,area=.2,n=1):
    m=module();target=torch.zeros(n,2,5,3);pad=target.clone();pad[...,0]=offset
    grasp=torch.zeros(n,2,3);grasp[:,1,0]=.1+offset
    target_grasp=torch.zeros_like(grasp);target_grasp[:,1,0]=.1
    target_arc=torch.tensor([[.2,.4]]).expand(n,-1).clone()
    return m.GeometryMeasurements(measurement(pad,'m',role='pad_xyz'),measurement(target,'m',role='target_pad_xyz'),
        measurement(grasp,'m',role='grasp_xyz'),measurement(target_grasp,'m',role='target_grasp_xyz'),
        measurement(target_arc+offset,'m'),measurement(target_arc,'m'),
        measurement(torch.full((n,2,5),area*1e-4),'m^2'),measurement(torch.full((n,2,5),1e-4),'m^2'))


def support(*,after=False,n=1):
    m=module();up=torch.empty(n,2,5);up[:,0]=.8;up[:,1]=.4
    normal=torch.ones(n,2,5);normal[:,:,0]=2.
    contacts=torch.zeros(n,2,5,dtype=torch.bool);contacts[:,:,0]=True
    if after:contacts[:,:,:2]=True
    return m.SupportMeasurements(measurement(up,'N'),measurement(normal,'N'),measurement(torch.full((n,2,5),.2),'N'),
        measurement(contacts,'bool'),measurement(torch.full((n,2,5),1.2 if after else 1.),'N',role='executed_force_target'))


def support_context(n=1):
    m=module();required=torch.zeros(n,2,5,dtype=torch.bool);required[:,:,:2]=True
    return m.SupportContext(context(n),measurement(torch.full((n,),10.),'N'),required,
        torch.tensor([[[0,1],[0,1]]]).expand(n,-1,-1).clone(),
        torch.ones(n,dtype=torch.bool),torch.zeros(n,dtype=torch.long),torch.zeros(n,dtype=torch.long),
        torch.ones(n,2,5,dtype=torch.bool),measurement(torch.full((n,2,5),1.5),'N'),
        measurement(torch.full((n,2,5),.1),'N'),measurement(torch.full((n,2,5),2.),'N'),
        expected_hand_support_share=measurement(torch.full((n,2),.5),'fraction',role='CPU_explicit_equal_share_fixture'))


def test_r1_all_four_potentials_use_actual_geometry_and_progress_only():
    m=module();before,after=geometry(),geometry(.01,.5)
    terms=m.geometry_terms(before,after,context(),config())
    assert set(terms)==set(m.GEOMETRY_TERMS)
    expected={'geometry_pad_distance':.5,'geometry_coordination':.5,'geometry_arc_position':.5,'geometry_contact_area':.3}
    for name,value in expected.items():
        torch.testing.assert_close(terms[name]['per_env_contribution'],torch.tensor([value]),rtol=1e-5,atol=1e-6)
        assert terms[name]['dt_scaling']=='potential_difference_no_dt'
        assert all(key in terms[name] for key in ('raw','normalized','coefficient','valid_mask','source','total_contribution','phase'))
    unchanged=m.geometry_terms(after,after,context(),config())
    assert all(torch.equal(row['per_env_contribution'],torch.zeros(1)) for row in unchanged.values())


def test_r2_balance_recovery_overload_and_executed_force_delta_are_distinct():
    m=module();terms=m.support_terms(support(),support(after=True),support_context(),config(),dt=1/60)
    assert set(terms)==set(m.SUPPORT_TERMS)
    expected={'hand_support_balance':-.2/60,'opposing_pad_balance':-(1/3)/60,
        'contact_recovery':.5,'normal_overload':-(.5/1.5/5)/60,'tangential_overload':-1/60,'force_target_delta':-.1}
    for name,value in expected.items():
        torch.testing.assert_close(terms[name]['per_env_contribution'],torch.tensor([value]),rtol=1e-5,atol=1e-6)
    doubled=m.support_terms(support(),support(after=True),support_context(),config(),dt=2/60)
    for name in m.SUPPORT_TERMS:
        factor=1. if name in ('contact_recovery','force_target_delta') else 2.
        torch.testing.assert_close(doubled[name]['per_env_contribution'],terms[name]['per_env_contribution']*factor)
    assert terms['force_target_delta']['source']['after']['role']=='executed_force_target'


def test_masked_na_and_same_episode_phase_masks_prevent_phantom_or_terminal_rewards():
    m=module();before,after=geometry(n=2),geometry(.01,.5,n=2)
    for state in (before,after):
        for value in vars(state).values():
            value.valid[1]=False;value.value[1]=float('nan')
    terms=m.geometry_terms(before,after,context(2),config())
    assert all(not row['valid_mask'][1] and row['per_env_contribution'][1]==0 for row in terms.values())
    for ctx in (replace(context(),same_episode=torch.tensor([False])),replace(context(),phase_active=torch.tensor([False]))):
        assert all(row['per_env_contribution'].eq(0).all() for row in m.geometry_terms(geometry(),geometry(.01,.5),ctx,config()).values())
    recovery=support_context();recovery=replace(recovery,recovery_epoch_after=torch.tensor([1]))
    assert not m.support_terms(support(),support(after=True),recovery,config(),dt=1/60)['contact_recovery']['valid_mask'].any()


def test_unknown_force_is_not_zero_and_one_hand_cannot_bless_bilateral_terms():
    m=module();before,after=support(),support(after=True)
    for value in vars(after).values():
        value.valid[:,1]=False
        if value.value.dtype!=torch.bool:value.value[:,1]=float('nan')
    terms=m.support_terms(before,after,support_context(),config(),dt=1/60)
    assert all(not row['valid_mask'].any() and row['per_env_contribution'].eq(0).all() for row in terms.values())


@pytest.mark.parametrize('fault',['whole_hand_broadcast','unit','valid_na','requested_target','bad_pair','dt','disabled','duplicate'])
def test_shape_unit_source_owner_and_optin_contracts_fail_closed(fault):
    m=module();before,after=support(),support(after=True);ctx=support_context();cfg=config();dt=1/60;owners=()
    if fault=='whole_hand_broadcast':after=replace(after,upward_support_n=measurement(torch.ones(1,2),'N'))
    elif fault=='unit':after=replace(after,normal_load_n=replace(after.normal_load_n,unit='kg'))
    elif fault=='valid_na':after.normal_load_n.value[0,0,0]=float('nan')
    elif fault=='requested_target':after=replace(after,force_target_n=replace(after.force_target_n,role='requested_force_target'))
    elif fault=='bad_pair':ctx=replace(ctx,opposing_pairs=torch.tensor([[[0,0],[0,1]]]))
    elif fault=='dt':dt=0.
    elif fault=='disabled':cfg=replace(cfg,enabled=False)
    elif fault=='duplicate':owners=['normal_overload']
    with pytest.raises(ValueError):m.support_terms(before,after,ctx,cfg,dt=dt,existing_owners=owners)


def test_candidate_config_is_explicit_disabled_and_contains_all_coefficients():
    path=Path(__file__).resolve().parents[1]/'configs/dh116_training/privileged_teacher_shaping_candidates_v1.json'
    assert path.is_file(),'independent candidate coefficients missing'
    value=json.loads(path.read_text())
    assert value['enabled'] is False and set(value['coefficients'])==set(module().TERM_NAMES)
    assert value['native_measurement_factory']=='pending' and value['native_qualified'] is False
    loaded=module().load_shaping_config(path)
    assert loaded.enabled is False and loaded.coefficients==value['coefficients']


def test_unnecessary_single_hand_target_change_is_not_hidden_by_other_hand_necessity():
    m=module();ctx=support_context();unnecessary=ctx.unnecessary_target_change.clone();unnecessary[:,1]=False
    ctx=replace(ctx,unnecessary_target_change=unnecessary)
    row=m.support_terms(support(),support(after=True),ctx,config(),dt=1/60)['force_target_delta']
    assert row['valid_mask'].all()
    torch.testing.assert_close(row['per_env_contribution'],torch.tensor([-.05]))
    assert row['source']['after']['valid_mask'].shape==(1,2,5)


def test_finite_inputs_overflow_and_mutated_coefficient_contract_are_rejected():
    m=module();state=geometry();state.pad_position_m.value.fill_(1e30)
    with pytest.raises(ValueError,match='finite|overflow'):
        m.geometry_terms(state,geometry(.01,.5),context(),config())
    cfg=config();cfg.coefficients['normal_overload']=float('nan')
    with pytest.raises(ValueError,match='coefficient'):
        m.support_terms(support(),support(after=True),support_context(),cfg,dt=1/60)


def test_typed_composition_closes_without_recounting_generic_potential_or_support(monkeypatch):
    import go2_pvcnn.control.dh116.tabletop_privileged_reward as reward_module
    m=module();native=torch.tensor([1.]);cfg=config();ctx=support_context()
    bundle=m.ShapingMeasurements(geometry(),geometry(.01,.5),context(),support(),support(after=True),ctx)
    # This isolates arithmetic composition only. No test claims native provenance.
    called=[]
    monkeypatch.setattr(reward_module,'_qualification',lambda provenance,key,identity:called.append(key))
    provenance={'geometry':{'scope':'CPU_arithmetic_mock_only'},'support':{'scope':'CPU_arithmetic_mock_only'}}
    actual,trace=reward_module.compose_native_reward(native,{'native':native.clone()},profile='teacher_r2_support_recovery',
        shaping_measurements=bundle,shaping_config=cfg,dt=1/60,provenance=provenance)
    assert called==['geometry','support']
    assert len(trace['terms'])==1+4+6+1
    assert 'teacher_geometry_progress' not in trace['terms']
    summed=sum(row['per_env_contribution'] for row in trace['terms'].values())
    torch.testing.assert_close(actual,summed,rtol=0,atol=2e-7)
    deficit=trace['terms']['teacher_support_deficit']['per_env_contribution']
    torch.testing.assert_close(deficit,torch.tensor([-1.5*.4/60]))
    with pytest.raises(ValueError,match='owner|combined'):
        reward_module.compose_native_reward(native,{'native':native.clone()},profile='teacher_r1_geometry',
            shaping={},shaping_measurements=bundle,shaping_config=cfg)
    with pytest.raises(ValueError,match='overlap|owner'):
        reward_module.compose_native_reward(native,{'support_deficit':native.clone()},profile='teacher_r2_support_recovery',
            shaping_measurements=bundle,shaping_config=cfg,dt=1/60)


def test_composer_rejects_native_batch_broadcast_and_distinguishes_geometry_from_load_balance(monkeypatch):
    import go2_pvcnn.control.dh116.tabletop_privileged_reward as reward_module
    m=module();bundle=m.ShapingMeasurements(geometry(),geometry(.01,.5),context())
    monkeypatch.setattr(reward_module,'_qualification',lambda *args:None)
    with pytest.raises(ValueError,match='shape|match'):
        reward_module.compose_native_reward(torch.tensor([1.,7.]),{'native':torch.tensor([1.,7.])},
            profile='teacher_r1_geometry',shaping_measurements=bundle,shaping_config=config())
    # Registered native dex_balance has a load semantic, not grasp-position error.
    actual,trace=reward_module.compose_native_reward(torch.ones(1),{'native':torch.ones(1),'dex_balance':torch.zeros(1)},
        profile='teacher_r1_geometry',shaping_measurements=bundle,shaping_config=config())
    assert actual.shape==(1,) and 'teacher_geometry_coordination' in trace['terms']


def test_invalid_arc_coordinate_and_finite_term_sum_overflow_fail_closed(monkeypatch):
    m=module();old=geometry();old.arc_position_m.value.fill_(-.1)
    with pytest.raises(ValueError,match='negative|arc'):
        m.geometry_terms(old,geometry(.01,.5),context(),config())
    import go2_pvcnn.control.dh116.tabletop_privileged_reward as reward_module
    monkeypatch.setattr(reward_module,'_qualification',lambda *args:None)
    coefficients={name:0. for name in m.TERM_NAMES};coefficients['geometry_pad_distance']=1e20
    cfg=m.ShapingConfig(enabled=True,pad_distance_scale_m=1.,coefficients=coefficients)
    bundle=m.ShapingMeasurements(geometry(1e18),geometry(0.),context())
    with pytest.raises(ValueError,match='finite|overflow'):
        reward_module.compose_native_reward(torch.tensor([3e38]),{'native':torch.tensor([3e38])},
            profile='teacher_r1_geometry',shaping_measurements=bundle,shaping_config=cfg)


def test_geometric_eighty_twenty_support_share_is_derived_and_not_penalized():
    m=module()
    grasp=measurement(torch.tensor([[[0.,0.,0.],[1.,0.,0.]]]),'m',role='grasp_xyz')
    com=measurement(torch.tensor([[.2,0.,.3]]),'m',valid=torch.ones(1,dtype=torch.bool),role='COM_xyz')
    gravity=measurement(torch.tensor([[0.,0.,-9.81]]),'m/s^2',valid=torch.ones(1,dtype=torch.bool),role='gravity_xyz')
    share=m.expected_support_share_from_geometry(grasp,com,gravity)
    torch.testing.assert_close(share.value,torch.tensor([[.8,.2]]))
    assert share.valid.all() and share.source['native_qualified'] is False
    after=support(after=True);values=after.upward_support_n.value.clone();values[:,0]=1.6;values[:,1]=.4
    after=replace(after,upward_support_n=replace(after.upward_support_n,value=values))
    ctx=replace(support_context(),expected_hand_support_share=share)
    row=m.support_terms(support(),after,ctx,config(),dt=1/60)['hand_support_balance']
    torch.testing.assert_close(row['per_env_contribution'],torch.zeros(1),atol=2e-8,rtol=0)
    assert row['source']['expected_share']['unit']=='fraction'


@pytest.mark.parametrize('fault',['missing','bad_sum','bad_shape','bad_unit','NA'])
def test_expected_share_is_explicit_validated_and_never_defaults_equal(fault):
    m=module();ctx=support_context()
    if fault=='missing':
        ctx=replace(ctx,expected_hand_support_share=None)
    else:
        share=ctx.expected_hand_support_share
        if fault=='bad_sum':share=replace(share,value=torch.tensor([[.8,.8]]))
        elif fault=='bad_shape':share=replace(share,value=torch.tensor([.5,.5]),valid=torch.ones(2,dtype=torch.bool))
        elif fault=='bad_unit':share=replace(share,unit='N')
        elif fault=='NA':share=replace(share,value=torch.full((1,2),float('nan')),valid=torch.zeros(1,2,dtype=torch.bool))
        ctx=replace(ctx,expected_hand_support_share=share)
    if fault in ('missing','NA'):
        row=m.support_terms(support(),support(after=True),ctx,config(),dt=1/60)['hand_support_balance']
        assert not row['valid_mask'].any() and row['per_env_contribution'].eq(0).all()
        assert torch.isnan(row['raw']).all() and torch.isnan(row['normalized']).all()
    else:
        with pytest.raises(ValueError,match='share|shape|fraction|sum'):
            m.support_terms(support(),support(after=True),ctx,config(),dt=1/60)


def test_inapplicable_grasp_com_geometry_stays_na_instead_of_clamping_share():
    m=module();grasp=measurement(torch.tensor([[[0.,0.,0.],[1.,0.,0.]]]),'m',role='grasp_xyz')
    com=measurement(torch.tensor([[2.,0.,.3]]),'m',valid=torch.ones(1,dtype=torch.bool),role='COM_xyz')
    gravity=measurement(torch.tensor([[0.,0.,-9.81]]),'m/s^2',valid=torch.ones(1,dtype=torch.bool),role='gravity_xyz')
    share=m.expected_support_share_from_geometry(grasp,com,gravity)
    assert not share.valid.any() and torch.isnan(share.value).all()


def test_composer_has_explicit_typed_extension_and_rejects_unqualified_inputs():
    from go2_pvcnn.control.dh116.tabletop_privileged_reward import compose_native_reward
    m=module();reward=torch.ones(1);terms={'native':reward.clone()}
    with pytest.raises(ValueError,match='qualified|provenance'):
        compose_native_reward(reward,terms,profile='teacher_r1_geometry',
            shaping_measurements=m.ShapingMeasurements(geometry(),geometry(.01,.5),context()),
            shaping_config=config())
    with pytest.raises(ValueError,match='R0|opt|profile'):
        compose_native_reward(reward,terms,shaping_measurements=m.ShapingMeasurements(geometry(),geometry(.01,.5),context()),
            shaping_config=config())
