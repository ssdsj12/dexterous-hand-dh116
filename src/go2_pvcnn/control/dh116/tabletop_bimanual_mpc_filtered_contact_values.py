"""Independent F1 current filtered-cell values; no control or qualification.

Reuse the existing single-filter Scene acquisition/ownership/reduction helpers.
The caller supplies one fresh valid F0 snapshot, the existing 34 sensors and
actual clock/source/stage-unit readers. No World/view/sensor creation, step,
refresh, settings, history, implicit rebuild, Actor input or motor estimate.
Each full-view raw tuple is archived once; pairs only select its F0 row/column.
Raw NaN/Inf remain recoverable: the caller owns strict JSON null/path/bit encoding.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType

from .tabletop_bimanual_mpc_state import SampleClock, StateField
from .tabletop_bimanual_mpc_filtered_contact_provenance import (
    FilteredPairSnapshot, VERSION as F0_VERSION, _clocks, _freeze, _plain, resolve_filtered_sensor_pairs)

VERSION='dh116-filtered-pair-values-f1-v1'
_QUALIFICATION=MappingProxyType(dict(native_attested=False,resolved_pair_qualified=False,
    contact_values_qualified=False,force_sign_frame_qualified=False,on_wire_force_qualified=False,
    per_contact_partner_id_qualified=False,upstream_overflow_absence_attested=False,
    support_qualified=False,planning_eligible=False,new_expert_qualification_issued=False,
    native_success_qualification_issued=False))

@dataclass(frozen=True,slots=True)
class FilteredPairValue:
    metadata_pair: object
    normal_sensor_side_world: StateField
    friction_sensor_side_world: StateField
    raw_view_key: str

@dataclass(frozen=True,slots=True)
class FilteredPairValuesSnapshot:
    capture_valid: bool
    errors: tuple
    clocks: tuple
    pairs: tuple
    raw: object
    metadata_binding_fingerprint: str | None
    version: str=VERSION
    def __post_init__(self):
        for key in ('errors','clocks','pairs'):object.__setattr__(self,key,tuple(getattr(self,key)))
        object.__setattr__(self,'raw',_freeze(self.raw))
    @property
    def qualification(self):return _QUALIFICATION
    def to_dict(self):return dict(_plain(self),qualification=dict(self.qualification))

def _field(pair,channel,value=None,reason=None):
    return StateField(value=value,source='existing hand contact_physx_view '+channel+'; sensor-side raw observation',
        unit='N',frame='world',timestamp_s=pair.clock.world_time_s if value is not None else None,
        valid=value is not None,env_id=pair.clock.env_id,episode_uid=pair.clock.episode_uid,clock=pair.clock,
        kind='measured' if value is not None else 'unavailable',missing_reason=reason,
        force_side='NA',origin='sensor_side_unqualified')

def _metadata(snapshot):
    if (type(snapshot) is not FilteredPairSnapshot or snapshot.version!=F0_VERSION
            or snapshot.metadata_binding_valid is not True or snapshot.errors):
        raise ValueError('one valid actual F0 snapshot required')
    if snapshot.contact_values_audited is not False or any(v is not False for v in snapshot.qualification.values()):
        raise ValueError('F0 metadata cannot supply contact or physical qualification')
    raw=snapshot.raw
    replay=resolve_filtered_sensor_pairs(raw['sensor_sources'],wire_body_paths=raw['wire_body_paths'],
        clocks_before=tuple(SampleClock(**_plain(c)) for c in raw['clocks_before']),
        clocks_after=tuple(SampleClock(**_plain(c)) for c in raw['clocks_after']),source_identity=raw['source_identity'])
    if (not replay.metadata_binding_valid or replay.pairs!=snapshot.pairs or replay.clocks!=snapshot.clocks
            or replay.binding_fingerprint!=snapshot.binding_fingerprint):
        raise ValueError('F0 DTO/fingerprint differs from fresh owned raw reconstruction')
    return replay

def _layout(value):
    if isinstance(value,(tuple,list)):return [_layout(v) for v in value]
    return dict(shape=list(getattr(value,'shape',())),dtype=str(getattr(value,'dtype','unknown')),
        source_device=str(getattr(value,'device','cpu' if type(value).__module__.startswith('numpy') else 'unknown')))

class FilteredPairValueProvider:
    def __init__(self,*,sensors,clock_reader=None,source_reader=None,units_reader=None,enabled=False,physics_dt=1/480):
        if type(enabled) is not bool:raise ValueError('enabled must be an actual boolean')
        self.enabled=enabled;self.sensors=sensors;self.clock_reader=clock_reader
        self.source_reader=source_reader;self.units_reader=units_reader;self.physics_dt=physics_dt;self._capturing=False
        if enabled and (isinstance(physics_dt,bool) or not isinstance(physics_dt,Real)
                or not math.isfinite(physics_dt) or not math.isclose(physics_dt,1/480,rel_tol=0.,abs_tol=1e-12)):
            raise ValueError('original 480 Hz physics dt required')

    def capture(self,metadata_snapshot,*,active_mask,clock_reader=None,source_reader=None,units_reader=None):
        raw=dict(enabled=self.enabled,views={},read_brackets=[],source_before=None,source_after=None,units_before=None,units_after=None,
            scope='one owned full-view record per existing sensor; pairs select F0 row/column, not numeric IDs or ON_Wire forces')
        if not self.enabled:
            return FilteredPairValuesSnapshot(False,('F1 values provider disabled',),(),(),raw,None)
        if self._capturing:raise ValueError('nested/concurrent F1 capture unsupported')
        if type(metadata_snapshot) is not FilteredPairSnapshot:
            return FilteredPairValuesSnapshot(False,('actual F0 snapshot type required',),(),(),raw,None)
        clocks=metadata_snapshot.clocks;mask=tuple(active_mask)
        if len(mask)!=len(clocks) or any(type(v) is not bool for v in mask):raise ValueError('one actual active boolean per environment required')
        if {c.env_id for c in clocks}!=set(range(len(clocks))):raise ValueError('explicit complete F0 environment cohort required')
        reader=self.clock_reader if clock_reader is None else clock_reader
        source=self.source_reader if source_reader is None else source_reader
        units=self.units_reader if units_reader is None else units_reader
        errors=[];replay=None;reductions={};valid=False;held=();sensor_objects=()
        self._capturing=True
        try:
            replay=_metadata(metadata_snapshot)
            if not any(mask):raise ValueError('inactive cohort; no F1 acquisition or contact credit')
            # Enabled-only dependency: import never constructs USD/SDK objects.
            from . import tabletop_bimanual_mpc_scene_measurements as scene
            if not isinstance(self.sensors,(tuple,list)) or len(self.sensors)!=34:raise ValueError('existing original 34 sensors required')
            sensor_objects=tuple(self.sensors);held=tuple(s.contact_physx_view for s in sensor_objects)
            rows={row['sensor_index']:row for row in replay.raw['sensor_sources']['sensors']}
            for i,view in enumerate(held):
                identity=rows[i]['contact_view_identity']
                if id(view)!=identity['python_runtime_id'] or type(view).__module__+'.'+type(view).__qualname__!=identity['python_type']:
                    raise ValueError('existing F0 contact view identity/type differs')
            acquisition=metadata_snapshot.raw.get('acquisition')
            if acquisition is not None and tuple(acquisition['sensor_object_identities_before'])!=tuple(id(v) for v in sensor_objects):
                raise ValueError('existing sensor object collection differs from F0 acquisition')

            def current():
                try:
                    actual,by_env=_clocks(reader(),reader())
                    if by_env!={c.env_id:c for c in clocks}:raise ValueError('actual SDK/World/episode clock differs from F0 snapshot')
                    if (len(self.sensors)!=len(sensor_objects) or any(a is not b for a,b in zip(self.sensors,sensor_objects))
                            or any(s.contact_physx_view is not v for s,v in zip(sensor_objects,held))):
                        raise ValueError('existing sensor/view lifecycle changed during F1 capture')
                    return actual
                except Exception as exc:raise scene._ClockChanged(str(exc)) from exc

            def read(name,function,*,view_key=None,getter_key=None):
                receipt=dict(name=name,before=current(),after=None,error=None)
                raw['read_brackets'].append(receipt)
                target=None
                if getter_key is not None:
                    target=raw['views'][view_key]['getters'].setdefault(getter_key,dict(value=None,error=None))
                try:
                    value=function()
                    owned=_freeze(scene._plain(value))
                    if target is not None:target['value']=owned
                    else:receipt['owned_value']=owned
                    return value
                except Exception as exc:
                    receipt['error']=type(exc).__name__+': '+str(exc)
                    if target is not None:target['error']=receipt['error']
                    raise
                finally:
                    try:receipt['after']=current()
                    except Exception as exc:
                        receipt['after_error']=type(exc).__name__+': '+str(exc)
                        raise

            raw['source_before']=_freeze(read('source.before',source))
            expected_source=replay.raw['source_identity']['before']
            if raw['source_before']!=expected_source:raise ValueError('actual source identity differs from F0 snapshot')
            raw['units_before']=_freeze(read('units.before',units))
            before_units=raw['units_before']
            if (before_units.get('stage_identity')!=expected_source['stage_identity'] or any(isinstance(before_units.get(k),bool)
                    or not isinstance(before_units.get(k),Real) or not math.isfinite(before_units[k]) or before_units[k]<=0
                    for k in ('meters_per_unit','kilograms_per_unit'))):
                raise ValueError('actual same-stage finite positive distance/mass units required')
            si=before_units['meters_per_unit']==before_units['kilograms_per_unit']==1.

            for index,view in enumerate(held):
                key=str(index);raw['views'][key]=dict(contact_view_identity=rows[index]['contact_view_identity'],getters={},getter_layouts={})
                entry=raw['views'][key]
                class ExistingView:
                    def __getattr__(proxy,name):
                        if name not in ('sensor_count','filter_count','max_contact_data_count','sensor_paths','filter_paths'):
                            raise AttributeError(name)
                        value=read(f'sensor[{index}].{name}',lambda:_freeze(getattr(view,name)))
                        if name in ('sensor_count','filter_count'):
                            expected=len(clocks) if name=='sensor_count' else 1
                            if type(value) is not int or value!=expected:
                                raise scene._ClockChanged('actual sensor/filter count type or value differs from strict F0 binding')
                        return value
                    def check(proxy):return read(f'sensor[{index}].check.actual',view.check)
                    def numeric(proxy,name,dt):
                        value=getattr(view,name)(dt=dt);entry['getter_layouts'][name]=_layout(value)
                        return value
                    def get_contact_force_matrix(proxy,dt):return proxy.numeric('get_contact_force_matrix',dt)
                    def get_contact_data(proxy,dt):return proxy.numeric('get_contact_data',dt)
                    def get_friction_data(proxy,dt):return proxy.numeric('get_friction_data',dt)
                def observe_read(name,function):
                    return read(f'sensor[{index}].{name}',function,view_key=key,getter_key=name)
                paths=list(rows[index]['sensor_paths']);filters=[row[0] for row in rows[index]['filter_paths']['value']]
                result=scene._contact_observation(ExistingView(),wires=paths,filters=filters,dt=self.physics_dt,read=observe_read,si_units=si)
                reductions[index]=result
                entry['reduction']={name:value for name,value in result.items() if name not in ('matrix','normal_tuple','friction_tuple','metadata')}

            # Metadata may change after the helper's initial metadata read.
            # Fresh after-reads never retimestamp the earlier owned numeric raw.
            for index,view in enumerate(held):
                for name,expected in (('sensor_paths',rows[index]['sensor_paths']),('filter_paths',rows[index]['filter_paths']['value']),
                        ('sensor_count',len(clocks)),('filter_count',1)):
                    actual=_freeze(read(f'sensor[{index}].{name}.after',lambda view=view,name=name:getattr(view,name)))
                    if (name in ('sensor_count','filter_count') and type(actual) is not int) or actual!=_freeze(expected):
                        raise ValueError('actual sensor/filter metadata type or value changed during F1 capture')
            raw['source_after']=_freeze(read('source.after',source));raw['units_after']=_freeze(read('units.after',units))
            if raw['source_after']!=raw['source_before']:raise ValueError('actual source identity changed during F1 capture')
            if raw['units_after']!=raw['units_before']:raise ValueError('actual stage units changed during F1 capture')
            current();valid=True
        except Exception as exc:errors.append(type(exc).__name__+': '+str(exc))
        finally:self._capturing=False

        pairs=[]
        for pair in replay.pairs if replay is not None else ():
            reason='; '.join(errors) if not valid else 'inactive slot' if not mask[pair.clock.env_id] else None
            observation=reductions.get(pair.sensor_index,{})
            fields=[]
            for channel in ('normal','friction'):
                value=None;missing=reason
                if missing is None:
                    if observation.get(channel+'_available') is True:
                        reduced=observation.get(channel+'_sum_n')
                        candidate=reduced[pair.sensor_row] if isinstance(reduced,(tuple,list)) and pair.sensor_row<len(reduced) else None
                        if (isinstance(candidate,(tuple,list)) and len(candidate)==3 and all(isinstance(v,Real)
                                and not isinstance(v,bool) and math.isfinite(v) for v in candidate)):
                            value=candidate
                        else:missing='actual '+channel+' reduction has nonfinite values or invalid shape'
                    else:missing=observation.get('missing_reason') or 'actual '+channel+' channel unavailable'
                fields.append(_field(pair,channel,value,missing))
            pairs.append(FilteredPairValue(pair,fields[0],fields[1],str(pair.sensor_index)))
        return FilteredPairValuesSnapshot(valid,tuple(errors),clocks,tuple(pairs),raw,
            None if replay is None else replay.binding_fingerprint)
