"""Synthetic CPU passive-pair sidecar, never a Native/support certificate."""
from copy import deepcopy
from dataclasses import asdict,replace
import importlib
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_bimanual_mpc_scene_measurements import Scene,Rigid,Simulation,clock
from go2_pvcnn.control.dh116 import tabletop_bimanual_mpc_scene_measurements as S


def api():
    path=Path(__file__).parents[1]/'src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_passive_pair_values.py'
    assert path.is_file(),'independent passive-pair sidecar is missing'
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_mpc_passive_pair_values')


def fixture():
    m=api();scene=Scene();rigid=Rigid();sim=Simulation()
    bundle=S.create_passive_contact_views(scene=scene,simulation_view=sim,rigid_body_view=rigid,enabled=True,capacity_per_wire=8)
    clocks=[clock(env_id=e,episode_uid=f'ep{e}') for e in range(2)]
    source=dict(stage_identity='fixture-stage',simulation_view_python_identity=id(sim),
        rigid_view_python_identity=id(rigid),contact_view_python_identities={s:id(v) for s,v in bundle.views.items()},
        wire_body_paths=list(rigid.prim_paths),api_sources={s:dict(path='/CPU/api.py',sha256='a'*64,python_type='CPU.Contact') for s in S.SURFACES})
    units=dict(stage_identity='fixture-stage',meters_per_unit=1.,kilograms_per_unit=1.,source='CPU actual unit readback')
    observer=m.PassivePairObserver(bundle=bundle,clock_reader=lambda:list(clocks),source_reader=lambda:deepcopy(source),
        units_reader=lambda:deepcopy(units),enabled=True)
    return NS(m=m,scene=scene,rigid=rigid,sim=sim,bundle=bundle,clocks=clocks,source=source,units=units,observer=observer)


def feed(f,mask=(True,True),*,normal_fault=None,friction_fault=None,after=None):
    p=f.observer;p.begin(clocks=f.clocks,active_mask=mask)
    for c,live in zip(f.clocks,mask):
        if live and not c.reset:p.observe_scene(c.env_id,f.scene.snapshot(f'/World/envs/env_{c.env_id}/Wire'))
    for surface,view in f.bundle.views.items():
        metadata=lambda:dict(sensor_count=view.sensor_count,filter_count=view.filter_count,
            max_contact_data_count=view.max_contact_data_count,sensor_paths=view.sensor_paths,filter_paths=view.filter_paths)
        p.read(surface,'metadata',view,metadata);p.read(surface,'check',view,view.check)
        for name,function in (('matrix',lambda:view.get_contact_force_matrix(dt=1/480)),
            ('normal',lambda:view.get_contact_data(dt=1/480)),('friction',lambda:view.get_friction_data(dt=1/480))):
            try:
                if (name=='normal' and normal_fault) or (name=='friction' and friction_fault):
                    p.read(surface,name,view,lambda:(_ for _ in ()).throw(RuntimeError('unavailable '+name)))
                else:p.read(surface,name,view,function)
            except RuntimeError:pass
        if after:after(f,surface,view)
        p.read(surface,'metadata_after',view,metadata)
    return p.finish()


def pair(result,env=0,surface='table'):
    return next(p for p in result.pairs if p.clock.env_id==env and p.surface==surface)


def advance(f,**changes):
    for e,c in enumerate(f.clocks):
        f.clocks[e]=replace(c,world_step=c.world_step+1,sdk_step=c.sdk_step+1,
            world_time_s=c.world_time_s+1/480,sdk_time_s=c.sdk_time_s+1/480,**changes)


def test_disabled_does_not_touch_any_input_or_reader():
    class Bomb:
        def __getattribute__(self,name):raise AssertionError(name)
    p=api().PassivePairObserver(bundle=Bomb(),clock_reader=Bomb(),source_reader=Bomb(),units_reader=Bomb())
    assert p.begin(clocks=Bomb(),active_mask=Bomb()) is False
    r=p.finish();assert not r.capture_valid and not r.pairs and not any(r.qualification.values())


def test_actual_three_views_signed_independent_channels_and_no_numeric_ids_or_qualification():
    f=fixture();r=feed(f);assert r.capture_valid,r.errors
    assert len(r.pairs)==6 and len(f.sim.created)==3 and len(r.raw['views'])==3
    assert pair(r).normal_sensor_side_world.value==(0.,0.,-2.)
    assert pair(r).friction_sensor_side_world.value==(.5,0.,-.1)
    assert pair(r).resultant_sensor_side_world.value==(.5,0.,-2.1)
    assert pair(r).sensor_path=='/World/envs/env_0/Wire'
    assert pair(r).filter_path=='/World/envs/env_0/Table/geometry/mesh'
    assert pair(r).sensor_row==0 and pair(r).filter_column==0
    assert all(p.actor_id is None and p.partner_body_id is None for p in r.pairs)
    assert all(p.normal_sensor_side_world.force_side=='NA' for p in r.pairs)
    assert not any(r.qualification.values())
    assert all(v.calls==[('matrix',1/480),('normal',1/480),('friction',1/480)] for v in f.bundle.views.values())


def test_owns_normal_before_friction_alias_and_stores_original_layout_and_all_raw_slots():
    f=fixture();r=feed(f);entry=r.raw['views']['table']
    assert entry['getters']['normal']['value'][4][0]==(1,)
    assert entry['getters']['normal']['value'][5][0]==(0,)
    assert entry['getters']['friction']['value'][3][0]==(3,)
    assert entry['getter_layouts']['normal'][4]['dtype']=='uint32'
    assert entry['getter_layouts']['normal'][0]['source_device']=='cpu'
    assert np.isnan(entry['getters']['normal']['value'][0][-1][0])
    f.bundle.views['table'].scalars[:]=99
    assert entry['getters']['normal']['value'][0][0]==(-2.,)
    with pytest.raises(TypeError):entry['getters']['normal']['value'][0][0][0]=99
    for receipt in r.raw['read_brackets']:
        assert tuple(receipt['before'])==tuple(asdict(c) for c in f.clocks)
        assert tuple(receipt['after'])==tuple(asdict(c) for c in f.clocks)


@pytest.mark.parametrize('channel',['normal','friction'])
def test_one_channel_unavailable_does_not_zero_or_hide_other_channel(channel):
    f=fixture();r=feed(f,normal_fault=channel=='normal',friction_fault=channel=='friction')
    assert r.capture_valid,r.errors
    p=pair(r);bad=getattr(p,channel+'_sensor_side_world');good=getattr(p,('friction' if channel=='normal' else 'normal')+'_sensor_side_world')
    assert not bad.valid and bad.value is None and bad.missing_reason
    assert good.valid and p.resultant_sensor_side_world.value is None


@pytest.mark.parametrize('fault',['regex','cross_env','duplicate','capacity','counts_overlap','nonfinite_owned','wrong_matrix'])
def test_metadata_or_owned_failures_remain_explicit_without_support_credit(fault):
    f=fixture();v=f.bundle.views['table']
    if fault=='regex':v.filter_paths=[['/World/envs/env_.*/Table/geometry/mesh']]*2
    elif fault=='cross_env':v.filter_paths.reverse()
    elif fault=='duplicate':v.sensor_paths=[v.sensor_paths[0]]*2
    elif fault=='capacity':v.normal_count[0]=v.max_contact_data_count
    elif fault=='counts_overlap':v.normal_start[1]=v.normal_start[0]
    elif fault=='nonfinite_owned':v.scalars[0]=np.nan
    else:v.matrix[0,0,2]+=1
    r=feed(f);p=pair(r)
    if fault in ('regex','cross_env','duplicate'):assert not r.capture_valid
    assert not p.normal_sensor_side_world.valid and p.normal_sensor_side_world.value is None
    assert not any(r.qualification.values()) and r.raw['views']['table']


def test_metadata_after_source_unit_and_clock_drift_are_not_retimestamped():
    for fault in ('metadata','source','unit','clock'):
        f=fixture()
        def after(f,s,v):
            if s!='table':return
            if fault=='metadata':v.filter_paths[0][0]='/World/envs/env_0/Other'
            elif fault=='source':f.source['stage_identity']='changed'
            elif fault=='unit':f.units['meters_per_unit']=.01
            else:f.clocks[0]=replace(f.clocks[0],sdk_step=f.clocks[0].sdk_step+1)
        try:r=feed(f,after=after)
        except ValueError:r=f.observer.finish()
        assert not r.capture_valid and r.errors and not any(r.qualification.values())


def test_failed_routing_does_not_label_factory_expected_filter_as_actual_resolved_pair():
    f=fixture();f.bundle.views['table'].filter_paths=[['CONFIGURED_REGEX_TRAP']]*2
    r=feed(f);p=pair(r)
    assert p.filter_path is None and p.pair_basis=='unavailable'
    assert r.raw['creation_expected_filter_paths']['table'][0].endswith('/Table/geometry/mesh')


def test_friction_cannot_run_before_normal_has_been_owned():
    f=fixture();p=f.observer;p.begin(clocks=f.clocks,active_mask=(True,True));v=f.bundle.views['table']
    with pytest.raises(ValueError,match='order'):
        p.read('table','friction',v,lambda:v.get_friction_data(dt=1/480))
    assert not v.calls


def test_actual_composed_units_must_agree_with_independently_read_stage_units():
    f=fixture();f.scene.records[0]['meters_per_unit']=.01;r=feed(f)
    assert not r.capture_valid and not pair(r).normal_sensor_side_world.valid


@pytest.mark.parametrize('reader',['source_reader','units_reader'])
def test_source_or_unit_reader_clock_change_is_bracketed_and_stops_before_any_numeric_getter(reader):
    f=fixture();p=f.observer;original=getattr(p,reader);before=tuple(asdict(c) for c in f.clocks)
    def drift():
        value=original();f.clocks[0]=replace(f.clocks[0],sdk_step=f.clocks[0].sdk_step+1);return value
    setattr(p,reader,drift)
    with pytest.raises(ValueError,match='clock'):p.begin(clocks=f.clocks,active_mask=(True,True))
    assert not any(v.calls for v in f.bundle.views.values())
    assert p.last_partial['clocks']==before and p.last_partial['complete_measurement'] is False
    assert p.last_partial['raw']['read_brackets'][-1]['after_error']


@pytest.mark.parametrize('channel',['normal','friction'])
def test_non_float_sdk_force_layout_is_preserved_but_cannot_become_a_valid_force(channel):
    f=fixture();v=f.bundle.views['table']
    if channel=='normal':
        v.scalars=v.scalars.astype(bool);v.matrix=v.scalars[:2,None,:]*v.normals[:2,None,:]
    else:v.friction=v.friction.astype(bool)
    r=feed(f);field=getattr(pair(r),channel+'_sensor_side_world')
    assert not field.valid and 'dtype' in field.missing_reason
    assert r.raw['views']['table']['getter_layouts'][channel][0]['dtype']=='bool'


def test_finite_raw_channels_cannot_produce_a_valid_nonfinite_resultant():
    f=fixture();v=f.bundle.views['table'];v.scalars[0]=1.e308;v.matrix[0,0]=[0.,0.,1.e308];v.friction[3]=[0.,0.,1.e308]
    with np.errstate(over='ignore'):r=feed(f)
    p=pair(r);assert p.normal_sensor_side_world.valid and p.friction_sensor_side_world.valid
    assert not p.resultant_sensor_side_world.valid and 'nonfinite' in p.resultant_sensor_side_world.missing_reason


def test_non_si_is_not_normalized_and_inactive_env_is_na_without_losing_owned_raw():
    f=fixture();f.units['meters_per_unit']=.01
    for record in f.scene.records:record['meters_per_unit']=.01
    r=feed(f)
    assert r.capture_valid and all(p.normal_sensor_side_world.value is None for p in r.pairs)
    assert 'SI' in pair(r).normal_sensor_side_world.missing_reason
    f=fixture();r=feed(f,(True,False));assert pair(r,env=0).normal_sensor_side_world.valid
    assert not pair(r,env=1).normal_sensor_side_world.valid
    assert r.raw['views']['table']['getters']['normal']['value'][0][1]==(3.,)


@pytest.mark.parametrize('fault',['scalar','matrix','normal_vector','friction'])
def test_inactive_payload_nan_cannot_poison_active_rows_and_raw_is_not_zero_filled(fault):
    f=fixture();v=f.bundle.views['table']
    if fault=='scalar':v.scalars[0]=np.nan
    elif fault=='matrix':v.matrix[0]=np.nan
    elif fault=='normal_vector':v.normals[0]=np.nan
    else:v.friction[3]=np.nan
    r=feed(f,(False,True));active=pair(r,env=1)
    assert r.capture_valid and active.normal_sensor_side_world.valid and active.friction_sensor_side_world.valid
    assert not pair(r).normal_sensor_side_world.valid
    raw=r.raw['views']['table']['getters']
    if fault=='matrix':assert np.isnan(raw['matrix']['value'][0][0][0])
    elif fault=='friction':assert np.isnan(raw['friction']['value'][0][3][0])
    else:assert np.isnan(raw['normal']['value'][0 if fault=='scalar' else 2][0][0])


def test_payload_mask_is_bound_to_actual_sensor_rows_not_clock_cohort_order():
    f=fixture();f.clocks.reverse();v=f.bundle.views['table'];v.scalars[0]=np.nan;v.matrix[0]=np.nan
    r=feed(f,(True,False));active=pair(r,env=1)
    assert active.clock==f.clocks[0] and active.sensor_row==1
    assert active.normal_sensor_side_world.value==(0.,0.,3.)
    assert not pair(r,env=0).normal_sensor_side_world.valid


def test_reset_row_payload_is_uncredited_and_cannot_poison_other_live_environment():
    f=fixture();f.clocks[0]=replace(f.clocks[0],reset=True);v=f.bundle.views['table']
    v.scalars[0]=np.nan;v.matrix[0]=np.nan
    r=feed(f);assert r.capture_valid and pair(r,env=1).normal_sensor_side_world.valid
    assert not pair(r).normal_sensor_side_world.valid and 'reset' in pair(r).normal_sensor_side_world.missing_reason


@pytest.mark.parametrize('fault',['active_nan','inactive_overlap','inactive_saturated'])
def test_active_payload_and_all_batch_structure_guards_still_refuse_without_filling(fault):
    f=fixture();v=f.bundle.views['table']
    if fault=='active_nan':v.scalars[1]=np.nan
    elif fault=='inactive_overlap':v.normal_start[0]=v.normal_start[1]
    else:v.normal_count[0]=v.max_contact_data_count
    r=feed(f,(False,True));field=pair(r,env=1).normal_sensor_side_world
    assert not field.valid and field.value is None and field.missing_reason


def test_first_terminal_pinned_noop_and_real_reset_generation_do_not_reuse_episode():
    f=fixture();f.clocks[0]=replace(f.clocks[0],terminal=True);first=feed(f)
    assert f.observer.first_terminals[(0,'ep0')]==tuple(p for p in first.pairs if p.clock.env_id==0)
    advance(f);second=feed(f,(False,True));assert not pair(second).normal_sensor_side_world.valid
    assert f.observer.first_terminals[(0,'ep0')][0].clock==first.clocks[0]
    advance(f);f.clocks[0]=replace(f.clocks[0],episode_uid='new0',reset_generation=1,reset=True,terminal=False)
    reset=feed(f);assert not pair(reset).normal_sensor_side_world.valid
    advance(f,reset=False);next_=feed(f);assert pair(next_).normal_sensor_side_world.valid
    assert pair(next_).clock.episode_uid=='new0'


def test_retired_inactive_actual_generation_may_change_without_granting_new_episode_credit():
    f=fixture();f.clocks[0]=replace(f.clocks[0],terminal=True);first=feed(f)
    advance(f);f.clocks[0]=replace(f.clocks[0],episode_uid='retired-reset0',reset_generation=1,terminal=False)
    r=feed(f,(False,True));assert r.capture_valid and not pair(r).normal_sensor_side_world.valid
    assert r.raw['retired_inactive_generation_changes'][0]['before']['episode_uid']=='ep0'
    assert f.observer.first_terminals[(0,'ep0')][0].clock==first.clocks[0]
    advance(f)
    with pytest.raises(ValueError,match='retired'):
        f.observer.begin(clocks=f.clocks,active_mask=(True,True))


def test_real_terminal_is_retired_even_when_numeric_metadata_is_invalid():
    f=fixture();f.clocks[0]=replace(f.clocks[0],terminal=True)
    filters=deepcopy(f.bundle.views['table'].filter_paths);f.bundle.views['table'].filter_paths=[['BAD']]*2
    r=feed(f);assert not r.capture_valid
    assert (0,'ep0') in f.observer.first_terminals
    f.bundle.views['table'].filter_paths=filters;advance(f,terminal=False)
    with pytest.raises(ValueError,match='retired'):
        f.observer.begin(clocks=f.clocks,active_mask=(True,True))


@pytest.mark.parametrize('fault',['duplicate_env','wrong_handle','stale','false_reset','policy_backwards'])
def test_source_cohort_and_episode_lifecycle_refuse_cross_generation_data(fault):
    f=fixture()
    if fault=='duplicate_env':f.clocks[1]=f.clocks[0]
    elif fault=='wrong_handle':f.source['contact_view_python_identities']['table']+=1
    else:
        feed(f)
        if fault!='stale':advance(f)
        if fault=='false_reset':f.clocks[0]=replace(f.clocks[0],episode_uid='bad',reset_generation=1)
        elif fault=='policy_backwards':f.clocks[0]=replace(f.clocks[0],policy_step=0)
    with pytest.raises(ValueError):f.observer.begin(clocks=f.clocks,active_mask=(True,True))
