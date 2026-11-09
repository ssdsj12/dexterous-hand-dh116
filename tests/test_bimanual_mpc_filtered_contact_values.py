"""CPU-only F1 fixtures; no SDK, native force or planning qualification."""
from copy import deepcopy
from dataclasses import replace
import importlib
from pathlib import Path
from types import SimpleNamespace as NS
import math
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
BASE='go2_pvcnn.control.dh116.'

def api():
    assert (ROOT/'src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_filtered_contact_values.py').is_file(),'independent F1 values provider is missing'
    return importlib.import_module(BASE+'tabletop_bimanual_mpc_filtered_contact_values')

class View:
    def __init__(self,side,body):
        self.sensor_paths=[f'/World/envs/env_{e}/{side}Hand/link_{body}' for e in (1,0)]
        self.filter_paths=[[f'/World/envs/env_{e}/Wire'] for e in (1,0)]
        self.sensor_count=2;self.filter_count=1;self.max_contact_data_count=8
        self.count=np.zeros((2,1),dtype=np.uint32);self.start=self.count.copy()
        self.normal_count=np.ones((2,1),dtype=np.uint32);self.normal_start=np.array([[0],[1]],dtype=np.uint32)
        self.friction_count=self.normal_count.copy();self.friction_start=np.array([[3],[4]],dtype=np.uint32)
        self.scalar=np.full((8,1),np.nan);self.point=np.full((8,3),np.nan);self.normal=np.full((8,3),np.nan);self.separation=self.scalar.copy()
        self.force=np.full((8,3),np.nan);self.anchor=np.full((8,3),np.nan)
        self.scalar[:2,0]=[-2.,3.];self.point[:2]=[[1.,2.,3.],[4.,5.,6.]];self.normal[:2]=[0.,0.,1.];self.separation[:2]=-.001
        self.force[3:5]=[[.5,0.,-.1],[.25,0.,.2]];self.anchor[3:5]=[[.1,.2,.3],[.4,.5,.6]]
        self.matrix=self.scalar[:2,None,:]*self.normal[:2,None,:];self.calls=[];self.hook=None;self.fail=None;self.ok=True
    def read(self,name,value):
        self.calls.append(name)
        if self.hook:self.hook(name)
        if self.fail==name:raise RuntimeError('unsupported '+name)
        return value
    def check(self):return self.ok
    def get_contact_force_matrix(self,dt):assert dt==1/480;return self.read('matrix',self.matrix)
    def get_contact_data(self,dt):
        assert dt==1/480;self.count[:]=self.normal_count;self.start[:]=self.normal_start
        return self.read('normal',(self.scalar,self.point,self.normal,self.separation,self.count,self.start))
    def get_friction_data(self,dt):
        assert dt==1/480;self.count[:]=self.friction_count;self.start[:]=self.friction_start
        return self.read('friction',(self.force,self.anchor,self.count,self.start))

def fixture(**options):
    state=importlib.import_module(BASE+'tabletop_bimanual_mpc_state')
    metadata=importlib.import_module(BASE+'tabletop_bimanual_mpc_filtered_contact_measurements')
    clocks=[state.SampleClock('CPU-F1',e,f'CPU-F1:{e}:0',10,10/480,1010,5+10/480,1,0) for e in range(2)]
    views=[View(side,i) for side in ('left','right') for i in range(17)]
    sensors=[NS(contact_physx_view=v,body_physx_view=NS(prim_paths=list(v.sensor_paths)),cfg=NS(filter_prim_paths_expr=['CONFIG_TRAP'])) for v in views]
    source=dict(stage_identity='CPU-only-stage',contact_view_runtime_ids=[id(v) for v in views],contact_view_api=dict(
        path='/CPU/api.py',sha256='a'*64,python_type=type(views[0]).__module__+'.'+type(views[0]).__qualname__))
    units=dict(stage_identity=source['stage_identity'],meters_per_unit=1.,kilograms_per_unit=1.)
    metadata_provider=metadata.FilteredPairMetadataProvider(sensors=sensors,wire_root_view=NS(prim_paths=[f'/World/envs/env_{e}/Wire' for e in range(2)]),
        clock_reader=lambda:list(clocks),source_reader=lambda:deepcopy(source),enabled=True)
    snapshot=metadata_provider.capture(clocks);assert snapshot.metadata_binding_valid,snapshot.errors
    provider=api().FilteredPairValueProvider(sensors=sensors,clock_reader=lambda:list(clocks),source_reader=lambda:deepcopy(source),
        units_reader=lambda:deepcopy(units),enabled=True,**options)
    return NS(provider=provider,snapshot=snapshot,clocks=clocks,views=views,sensors=sensors,source=source,units=units)

def capture(f,mask=(True,True)):return f.provider.capture(f.snapshot,active_mask=mask)
def pair(result,index=0,env=0):return next(v for v in result.pairs if v.metadata_pair.sensor_index==index and v.metadata_pair.clock.env_id==env)

def test_disabled_is_zero_read_and_does_not_inspect_snapshot_callbacks_or_views():
    class Bomb:
        def __getattribute__(self,name):raise AssertionError(name)
    p=api().FilteredPairValueProvider(sensors=Bomb(),clock_reader=Bomb(),source_reader=Bomb(),units_reader=Bomb())
    r=p.capture(Bomb(),active_mask=Bomb());assert not r.capture_valid and not r.pairs
    assert not any(r.qualification.values())

def test_signed_normal_friction_and_actual_row_binding_reuse_existing_scene_helper(monkeypatch):
    f=fixture();scene=importlib.import_module(BASE+'tabletop_bimanual_mpc_scene_measurements');original=scene._contact_observation;calls=[]
    def observed(*args,**kwargs):calls.append(True);return original(*args,**kwargs)
    monkeypatch.setattr(scene,'_contact_observation',observed)
    r=capture(f);assert r.capture_valid and len(r.pairs)==68 and len(r.raw['views'])==34 and len(calls)==34
    assert list(pair(r,env=0).normal_sensor_side_world.value)==[0.,0.,3.]
    assert list(pair(r,env=1).normal_sensor_side_world.value)==[0.,0.,-2.]
    assert list(pair(r,env=0).friction_sensor_side_world.value)==[.25,0.,.2]
    assert all(v.calls==['matrix','normal','friction'] for v in f.views)
    assert not any(r.qualification.values())
    assert all(v.metadata_pair.actor_id is None and v.metadata_pair.partner_body_id is None for v in r.pairs)

def test_normal_is_owned_before_friction_reuses_count_start_and_result_is_immutable():
    f=fixture();r=capture(f);normal=r.raw['views']['0']['getters']['normal']['value']
    assert list(normal[4][0])==[1] and list(normal[5][0])==[0]
    assert list(r.raw['views']['0']['getters']['friction']['value'][3][0])==[3]
    f.views[0].scalar[:]=999;f.views[0].normal_count[:]=7
    assert normal[0][0][0]==-2. and pair(r,env=0).normal_sensor_side_world.value[2]==3.
    with pytest.raises(TypeError):r.raw['views']['0']['new']='forged'

def test_matrix_mismatch_invalidates_normal_but_not_independent_friction_and_keeps_raw():
    f=fixture();f.views[0].matrix[:]=17.;r=capture(f)
    assert not pair(r).normal_sensor_side_world.valid and 'matrix' in pair(r).normal_sensor_side_world.missing_reason
    assert pair(r).friction_sensor_side_world.valid
    assert r.raw['views']['0']['getters']['matrix']['value'][0][0][0]==17.

@pytest.mark.parametrize('fault',['saturated','overlap','missing_normal','missing_friction','owned_nan'])
def test_bad_tuple_channel_is_na_and_raw_is_not_zero_filled(fault):
    f=fixture();v=f.views[0]
    if fault=='saturated':v.normal_count[:]=4;v.normal_start[:]=[[0],[4]];v.scalar[:]=1.;v.point[:]=0.;v.normal[:]=[0.,0.,1.];v.separation[:]=0.
    elif fault=='overlap':v.normal_start[:]=0
    elif fault=='missing_normal':v.fail='normal'
    elif fault=='missing_friction':v.fail='friction'
    else:v.scalar[0,0]=np.nan
    r=capture(f);field=pair(r).friction_sensor_side_world if fault=='missing_friction' else pair(r).normal_sensor_side_world
    assert not field.valid and field.value is None and field.missing_reason
    if fault=='owned_nan':assert math.isnan(r.raw['views']['0']['getters']['normal']['value'][0][0][0])
    if fault=='missing_normal':assert pair(r).friction_sensor_side_world.valid

@pytest.mark.parametrize('fault',['clock','source','units','view','path'])
def test_mid_getter_lifecycle_changes_invalidate_entire_capture_but_keep_acquired_raw(fault):
    f=fixture()
    def hook(name):
        if name!='normal':return
        if fault=='clock':f.clocks[0]=replace(f.clocks[0],sdk_step=1011)
        elif fault=='source':f.source['contact_view_api']['sha256']='b'*64
        elif fault=='units':f.units['meters_per_unit']=.01
        elif fault=='view':f.sensors[0].contact_physx_view=View('left',0)
        else:f.views[0].filter_paths[0]=['/World/envs/env_0/Wire']
    f.views[0].hook=hook;r=capture(f)
    assert not r.capture_valid and r.errors
    assert all(not v.normal_sensor_side_world.valid and not v.friction_sensor_side_world.valid for v in r.pairs)
    assert 'normal' in r.raw['views']['0']['getters']
    assert r.raw['views']['0']['getters']['normal']['value'][0][0][0]==-2.

def test_inactive_pairs_are_na_even_when_the_shared_full_view_contains_values():
    f=fixture();r=capture(f,(False,True))
    assert not pair(r,env=0).normal_sensor_side_world.valid and 'inactive' in pair(r,env=0).normal_sensor_side_world.missing_reason
    assert pair(r,env=1).normal_sensor_side_world.valid
    assert len(r.raw['views'])==34

def test_non_si_units_keep_raw_and_no_force_value_credit():
    f=fixture();f.units['meters_per_unit']=.01;r=capture(f)
    assert not pair(r).normal_sensor_side_world.valid and not pair(r).friction_sensor_side_world.valid
    assert 'normal' in r.raw['views']['0']['getters'] and not any(r.qualification.values())

def test_stale_or_forged_f0_snapshot_is_not_repaired_with_current_view_paths():
    f=fixture();f.snapshot=replace(f.snapshot,pairs=(replace(f.snapshot.pairs[0],sensor_row=1),)+f.snapshot.pairs[1:]);r=capture(f)
    assert not r.capture_valid and not any(v.calls for v in f.views)

def test_all_inactive_is_zero_numeric_reads_and_bad_masks_never_grant_credit():
    f=fixture();r=capture(f,(False,False));assert not any(v.calls for v in f.views)
    assert all(not p.normal_sensor_side_world.valid for p in r.pairs)
    with pytest.raises(ValueError):capture(f,(1,True))

@pytest.mark.parametrize('change',[dict(metadata_binding_valid=1),dict(contact_values_audited=0),dict(version='unreviewed-F0-version')])
def test_f0_schema_version_and_boolean_flags_are_exact_not_coerced(change):
    f=fixture();f.snapshot=replace(f.snapshot,**change);r=capture(f)
    assert not r.capture_valid and not any(v.calls for v in f.views)

def test_units_getter_partial_owned_value_survives_post_getter_clock_failure():
    f=fixture()
    def units_reader():
        value=deepcopy(f.units);f.clocks[0]=replace(f.clocks[0],sdk_step=1011)
        return value
    f.provider.units_reader=units_reader;r=capture(f)
    assert not r.capture_valid and not any(v.calls for v in f.views)
    receipt=next(v for v in r.raw['read_brackets'] if v['name']=='units.before')
    assert 'owned_value' in receipt
    assert receipt['owned_value']['meters_per_unit']==1.
    assert 'after_error' in receipt

def test_finite_friction_rows_whose_sum_overflows_remain_raw_but_channel_is_na():
    f=fixture();v=f.views[0];maximum=np.finfo(float).max
    v.friction_count[:]=[[2],[1]];v.friction_start[:]=[[3],[5]]
    v.force[3:5]=[maximum,0.,0.];v.anchor[3:5]=0.;v.force[5]=[1.,0.,0.];v.anchor[5]=0.
    with np.errstate(over='ignore'):r=capture(f)
    field=pair(r,env=1).friction_sensor_side_world
    assert not field.valid and field.value is None and 'nonfinite' in field.missing_reason
    assert pair(r,env=1).normal_sensor_side_world.valid
    assert r.raw['views']['0']['getters']['friction']['value'][0][3][0]==maximum

@pytest.mark.parametrize('name,value',[('filter_count',True),('filter_count',1.0),('sensor_count',2.0)])
def test_actual_count_metadata_type_drift_cannot_relax_strict_f0_binding(name,value):
    f=fixture();setattr(f.views[0],name,value);r=capture(f)
    assert not r.capture_valid and not any(v.calls for v in f.views)
    assert all(not v.normal_sensor_side_world.valid and not v.friction_sensor_side_world.valid for v in r.pairs)

def test_after_read_count_type_drift_invalidates_all_values_with_partial_raw_retained():
    f=fixture()
    def drift(name):
        if name=='normal':f.views[0].filter_count=True
    f.views[0].hook=drift;r=capture(f)
    assert not r.capture_valid and all(not v.normal_sensor_side_world.valid for v in r.pairs)
    assert r.raw['views']['0']['getters']['normal']['value'][0][0][0]==-2.
