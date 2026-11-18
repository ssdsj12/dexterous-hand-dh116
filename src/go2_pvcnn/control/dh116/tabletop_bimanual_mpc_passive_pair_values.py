"""Default-OFF taps of the existing three passive views, never support proof.

No view/World is created here. A tap owns existing getter returns immediately;
reported Wire-sensor forces retain an explicitly unqualified force side.
"""
from dataclasses import asdict,dataclass
import hashlib
import json
import math
from numbers import Real
from types import MappingProxyType

import numpy as np

from .tabletop_bimanual_mpc_state import SampleClock,StateField
from .tabletop_bimanual_mpc_filtered_contact_provenance import _freeze,_plain

VERSION='dh116-passive-pair-values-v1'
SURFACES=('table','negative_y','positive_y')
_QUALIFICATION=MappingProxyType(dict(native_attested=False,force_sign_frame_qualified=False,
    on_wire_force_qualified=False,support_qualified=False,airborne_qualified=False,
    upstream_overflow_absence_attested=False,planning_eligible=False,new_expert_qualification_issued=False,
    native_success_qualification_issued=False,hardware_measurement_qualified=False))


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _layout(value):
    if isinstance(value,(tuple,list)):return tuple(_layout(v) for v in value)
    return dict(shape=list(getattr(value,'shape',())),dtype=str(getattr(value,'dtype','unknown')),
        source_device=str(getattr(value,'device','cpu' if type(value).__module__.startswith('numpy') else 'unknown')))


def _own(value):
    from .tabletop_bimanual_mpc_scene_measurements import _array
    if isinstance(value,(tuple,list)):return tuple(_own(v) for v in value)
    return _array(value)


def _checked_layout(value,layout,*,tuple_data=False):
    arrays=value if tuple_data else (value,);layouts=layout if tuple_data else (layout,)
    if not isinstance(layouts,(tuple,list)) or len(arrays)!=len(layouts):raise ValueError('raw getter layout unavailable')
    for i,(array,record) in enumerate(zip(arrays,layouts)):
        try:
            dtype=np.dtype(record['dtype'].removeprefix('torch.'));shape=record['shape']
            integral=tuple_data and i>=len(arrays)-2
            valid=(isinstance(shape,(tuple,list)) and all(type(n) is int and n>=0 for n in shape)
                and tuple(shape)==array.shape and dtype==array.dtype
                and (dtype.kind in 'iu' if integral else dtype.kind=='f' and dtype.itemsize in (4,8))
                and isinstance(record['source_device'],str) and record['source_device'] not in ('','unknown'))
        except (KeyError,TypeError,ValueError,AttributeError):valid=False
        if not valid:raise ValueError('original SDK array dtype/shape/device unavailable or unsupported')


@dataclass(frozen=True,slots=True)
class PassivePairValue:
    clock:SampleClock
    surface:str
    sensor_row:int
    filter_column:int
    sensor_path:str
    filter_path:str|None
    normal_sensor_side_world:StateField
    friction_sensor_side_world:StateField
    resultant_sensor_side_world:StateField
    pair_basis:str='SDK_resolved_sensor_filter_paths_only'
    actor_id:object=None
    partner_body_id:object=None


@dataclass(frozen=True,slots=True)
class PassivePairSnapshot:
    capture_valid:bool
    errors:tuple
    clocks:tuple
    pairs:tuple
    raw:object
    source_epoch:str|None
    version:str=VERSION
    @property
    def qualification(self):return _QUALIFICATION
    def to_dict(self):return dict(_plain(self),qualification=dict(self.qualification))


def _field(clock,surface,channel,value=None,reason=None):
    dependencies=(f'raw.views.{surface}.getters.normal',f'raw.views.{surface}.getters.friction') if channel=='resultant' else (f'raw.views.{surface}.getters.{channel}',)
    return StateField(value=value,source=f'existing passive contact view[{surface}] {channel}; sensor-side raw observation',
        unit='N',frame='world',timestamp_s=clock.world_time_s if value is not None else None,
        valid=value is not None,env_id=clock.env_id,episode_uid=clock.episode_uid,clock=clock,
        kind='derived' if value is not None else 'unavailable',dependencies=dependencies if value is not None else (),
        missing_reason=reason,force_side='NA',origin='sensor_side_unqualified')


def make_passive_source_reader(*,stage,bundle,stage_reader):
    """Lazy reads of caller-owned handles/files only, no SDK import or setter."""
    def source():
        import inspect
        from pathlib import Path
        if stage_reader() is not stage:raise ValueError('actual active stage changed')
        identity=str(stage.GetRootLayer().identifier)+';session='+str(stage.GetSessionLayer().identifier)+';handle='+hex(id(stage))
        apis={}
        for s in SURFACES:
            typ=type(bundle.views[s]);path=Path(inspect.getfile(typ)).resolve(strict=True)
            apis[s]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),python_type=typ.__module__+'.'+typ.__qualname__)
        return dict(stage_identity=identity,simulation_view_python_identity=id(bundle.simulation_view),
            rigid_view_python_identity=id(bundle.rigid_body_view),contact_view_python_identities={s:id(bundle.views[s]) for s in SURFACES},
            wire_body_paths=list(bundle.rigid_body_view.prim_paths),api_sources=apis)
    return source


class PassivePairObserver:
    def __init__(self,*,bundle,clock_reader,source_reader,units_reader,enabled=False,physics_dt=1/480):
        if type(enabled) is not bool:raise ValueError('explicit enabled bool required')
        if enabled and (isinstance(physics_dt,bool) or not isinstance(physics_dt,Real)
                or not math.isclose(physics_dt,1/480,rel_tol=0,abs_tol=1e-12)):
            raise ValueError('original 480Hz contact dt required')
        self.enabled=enabled;self.bundle=bundle;self.clock_reader=clock_reader
        self.source_reader=source_reader;self.units_reader=units_reader;self.physics_dt=physics_dt
        self._open=False;self._previous={};self._seen={};self._epoch=None;self._terminals={};self._retired=set();self.last_snapshot=None;self.last_partial=None

    @property
    def first_terminals(self):return MappingProxyType(dict(self._terminals))

    def _current(self):
        actual=tuple(self.clock_reader())
        self._last_observed=_freeze([asdict(c) if isinstance(c,SampleClock) else dict(invalid_type=type(c).__name__,raw_repr=repr(c)) for c in actual])
        if actual!=self.clocks or not all(isinstance(c,SampleClock) for c in actual):
            self._errors.append('actual physical clock/episode changed during passive capture')
            raise ValueError(self._errors[-1])
        return [asdict(c) for c in actual]

    def _bracket(self,name,function):
        receipt=dict(name=name,before=self._current(),after=None)
        self._raw['read_brackets'].append(receipt)
        try:
            value=function();receipt['value']=_freeze(value);return value
        except Exception as exc:receipt['error']=type(exc).__name__+': '+str(exc);raise
        finally:
            try:receipt['after']=self._current()
            except Exception as exc:receipt.update(after_error=str(exc),after_observed=self._last_observed);raise

    def _source(self):
        source=_freeze(self.source_reader());b=self.bundle
        if (not b.stage_identities or any(source.get('stage_identity')!=v for v in b.stage_identities)
                or type(source.get('simulation_view_python_identity')) is not int
                or source['simulation_view_python_identity']!=id(b.simulation_view)
                or type(source.get('rigid_view_python_identity')) is not int
                or source['rigid_view_python_identity']!=id(b.rigid_body_view)
                or tuple(source.get('wire_body_paths',()))!=b.wire_paths):
            raise ValueError('actual passive stage/simulation/Wire source identity differs')
        for s in SURFACES:
            identity=source.get('contact_view_python_identities',{}).get(s)
            api=source.get('api_sources',{}).get(s,{})
            if (type(identity) is not int or identity!=id(b.views[s])
                    or not isinstance(api.get('path'),str) or not api['path']
                    or not isinstance(api.get('python_type'),str) or not api['python_type']
                    or not isinstance(api.get('sha256'),str) or len(api['sha256'])!=64
                    or any(c not in '0123456789abcdef' for c in api['sha256'])):
                raise ValueError('actual passive view/API identity unavailable')
        return source

    def _units(self,stage):
        units=_freeze(self.units_reader())
        if units.get('stage_identity')!=stage or not isinstance(units.get('source'),str) or not units['source']:
            raise ValueError('actual same-stage unit source unavailable')
        for k in ('meters_per_unit','kilograms_per_unit'):
            v=units.get(k)
            if isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) or v<=0:
                raise ValueError('actual typed finite positive stage units required')
        return units

    def begin(self,*,clocks,active_mask):
        if not self.enabled:return False
        if self._open:raise ValueError('nested passive transaction unsupported')
        self.last_snapshot=None;self.last_partial=None
        clocks=tuple(clocks);mask=tuple(active_mask)
        if (not clocks or not all(isinstance(c,SampleClock) for c in clocks)
                or len({c.env_id for c in clocks})!=len(clocks) or len(mask)!=len(clocks)
                or any(type(v) is not bool for v in mask)
                or len({(c.run_uuid,c.world_step,c.sdk_step,c.world_time_s,c.sdk_time_s) for c in clocks})!=1):
            raise ValueError('complete unique same-frame physical cohort and active bools required')
        self.clocks=clocks;self.mask=mask;self._errors=[]
        self._raw=dict(version=VERSION,physics_dt_s=self.physics_dt,source_before=None,source_after=None,
            units_before=None,units_after=None,read_brackets=[],views={},scene_records={},active_mask=mask,
            creation_expected_filter_paths=_freeze(self.bundle.filter_paths),retired_inactive_generation_changes=[],
            source_scope='simulation privileged reported contacts, not hardware or support proof')
        try:
            source=self._bracket('source.before',self._source);self._raw['source_before']=source
            units=self._bracket('units.before',lambda:self._units(source['stage_identity']));self._raw['units_before']=units
        except Exception as exc:
            self.last_partial=_freeze(dict(clocks=[asdict(c) for c in clocks],raw=self._raw,error=repr(exc),complete_measurement=False))
            raise
        expected={f'/World/envs/env_{c.env_id}/Wire' for c in clocks}
        if set(source['wire_body_paths'])!=expected or len(source['wire_body_paths'])!=len(expected):
            raise ValueError('actual Wire cohort differs from physical clocks')
        epoch=_digest(source)
        if self._epoch is not None and epoch!=self._epoch:raise ValueError('passive source epoch changed; rebuild observer')
        if not set(self._previous)<={c.env_id for c in clocks}:raise ValueError('passive cohort changed')
        retired=[]
        for c,live in zip(clocks,mask):
            old=self._previous.get(c.env_id)
            if live and (c.env_id,c.episode_uid) in self._retired:raise ValueError('retired inactive generation cannot resume')
            if old:
                steps=c.world_step-old.world_step
                if (c.run_uuid!=old.run_uuid or steps<=0 or c.sdk_step-old.sdk_step!=steps
                        or c.policy_step<old.policy_step or any(not math.isclose(v,steps*self.physics_dt,rel_tol=0,abs_tol=1e-7)
                            for v in (c.world_time_s-old.world_time_s,c.sdk_time_s-old.sdk_time_s))):
                    raise ValueError('stale/discontinuous passive physical clock')
                if c.episode_uid==old.episode_uid:
                    if c.reset_generation!=old.reset_generation or (old.terminal and live):raise ValueError('retired episode cannot resume')
                else:
                    if c.reset_generation<=old.reset_generation or c.episode_uid in self._seen.get(c.env_id,set()):raise ValueError('actual new reset generation required')
                    if not live and (old.terminal or (old.env_id,old.episode_uid) in self._retired):
                        retired.append(dict(before=asdict(old),after=asdict(c),reason='actual generation changed only in retired inactive slot; no new credit'))
                    elif not c.reset:raise ValueError('actual new reset generation required')
        self._raw['retired_inactive_generation_changes']=retired
        self._epoch=epoch;self._values={};self._open=True
        return True

    def observe_scene(self,env_id,record):
        if not self._open:raise ValueError('no passive transaction')
        self._raw['scene_records'][str(env_id)]=_freeze(record)

    def read(self,surface,name,view,function):
        if not self._open or surface not in SURFACES or view is not self.bundle.views[surface]:
            raise ValueError('current existing passive view transaction required')
        entry=self._raw['views'].setdefault(surface,dict(contact_view_python_identity=id(view),getters={},getter_layouts={}))
        if name in entry['getters']:raise ValueError('duplicate passive getter; no hidden resampling')
        order=('metadata','check','matrix','normal','friction','metadata_after')
        if len(entry['getters'])>=len(order) or name!=order[len(entry['getters'])]:raise ValueError('passive getter order must own normal before friction')
        receipt=dict(surface=surface,name=name,before=self._current(),after=None)
        self._raw['read_brackets'].append(receipt)
        try:
            value=function()
            if name in ('matrix','normal','friction'):
                entry['getter_layouts'][name]=_freeze(_layout(value));value=_own(value)
                self._values[(surface,name)]=value
                from .tabletop_bimanual_mpc_scene_measurements import _plain as plain
                raw_value=_freeze(plain(value))
            else:raw_value=_freeze(value)
            entry['getters'][name]=dict(available=True,value=raw_value,source='existing passive view.'+name)
            return value
        except Exception as exc:
            entry['getters'][name]=dict(available=False,value=None,missing_reason=type(exc).__name__+': '+str(exc))
            raise
        finally:
            try:receipt['after']=self._current()
            except Exception as exc:receipt.update(after_error=str(exc),after_observed=self._last_observed);raise

    def finish(self):
        if not self.enabled:return PassivePairSnapshot(False,('observer disabled',),(),(),_freeze({}),None)
        if not self._open:
            if self.last_snapshot is not None:return self.last_snapshot
            raise ValueError('no passive transaction')
        from . import tabletop_bimanual_mpc_scene_measurements as scene
        errors=list(self._errors);routing={};reductions={}
        try:
            after=self._bracket('source.after',self._source);self._raw['source_after']=after
            units=self._bracket('units.after',lambda:self._units(after['stage_identity']));self._raw['units_after']=units
            if _digest(after)!=_digest(self._raw['source_before']) or _digest(units)!=_digest(self._raw['units_before']):
                raise ValueError('actual passive source/units changed during transaction')
            for c,live in zip(self.clocks,self.mask):
                if live and not c.reset:
                    rec=self._raw['scene_records'].get(str(c.env_id))
                    if rec is None or rec.get('stage_identity')!=after['stage_identity']:raise ValueError('actual composed passive scene receipt unavailable')
                    if any(isinstance(rec.get(k),bool) or rec.get(k)!=units[k] for k in ('meters_per_unit','kilograms_per_unit')):
                        raise ValueError('actual composed scene/stage units disagree')
                    expected=scene._validate_scene(rec,f'/World/envs/env_{c.env_id}/Wire')
                    row=list(after['wire_body_paths']).index(f'/World/envs/env_{c.env_id}/Wire')
                    if any(expected[s]!=self.bundle.filter_paths[s][row] for s in SURFACES):raise ValueError('actual composed passive collider mapping changed')
            for s in SURFACES:
                getters=self._raw['views'].get(s,{}).get('getters',{})
                meta=getters.get('metadata',{}).get('value');tail=getters.get('metadata_after',{}).get('value')
                if meta is None or _digest(meta)!=_digest(tail) or getters.get('check',{}).get('value') is not True:
                    raise ValueError('actual passive metadata/check before-after unavailable or changed')
                n=len(self.clocks);filters=meta.get('filter_paths');expected=[[p] for p in self.bundle.filter_paths[s]]
                if filters==tuple(self.bundle.filter_paths[s]):filters=tuple((p,) for p in filters)
                if (type(meta.get('sensor_count')) is not int or meta['sensor_count']!=n
                        or type(meta.get('filter_count')) is not int or meta['filter_count']!=1
                        or tuple(meta.get('sensor_paths',()))!=tuple(after['wire_body_paths'])
                        or _plain(filters)!=expected or type(meta.get('max_contact_data_count')) is not int or meta['max_contact_data_count']<1):
                    raise ValueError('actual passive sensor/filter/capacity routing unavailable')
                routing[s]=meta
        except Exception as exc:errors.append(type(exc).__name__+': '+str(exc))
        si=all(self._raw['units_before'][k]==1. for k in ('meters_per_unit','kilograms_per_unit'))
        active_by_path={f'/World/envs/env_{c.env_id}/Wire':live and not c.reset for c,live in zip(self.clocks,self.mask)}
        row_mask=tuple(active_by_path[p] for p in self._raw['source_before']['wire_body_paths'])
        selected=np.asarray(row_mask,dtype=bool)
        self._raw['active_sensor_row_mask']=row_mask
        self._raw['payload_selection_basis']='actual Wire source path matched to active nonreset physical env; inactive reduction placeholders are not measurements'
        for s in SURFACES:
            for channel in ('normal','friction'):
                try:
                    if s not in routing:raise ValueError('actual passive pair routing invalid')
                    if not si:raise ValueError('actual SI stage-unit normalization unproven')
                    value=self._values.get((s,channel))
                    if value is None:raise ValueError(self._raw['views'][s]['getters'].get(channel,{}).get('missing_reason','getter unavailable'))
                    _checked_layout(value,self._raw['views'][s]['getter_layouts'][channel],tuple_data=True)
                    with np.errstate(over='ignore',invalid='ignore'):
                        total=scene._owned(value,normal=channel=='normal',n=len(self.clocks),capacity=routing[s]['max_contact_data_count'],active_mask=row_mask)
                    if not np.isfinite(total[selected]).all():raise ValueError('nonfinite reduced contact vector')
                    if channel=='normal':
                        matrix=self._values.get((s,'matrix'))
                        _checked_layout(matrix,self._raw['views'][s]['getter_layouts'].get('matrix'))
                        if matrix.shape!=(len(self.clocks),1,3):raise ValueError('exact full-batch matrix layout required')
                        scene._finite(matrix[selected],(int(selected.sum()),1,3))
                        if not np.allclose(total[selected],matrix[selected,0],rtol=1e-4,atol=1e-5):raise ValueError('normal differs from same-pair matrix')
                    reductions[(s,channel)]=(total,None)
                except Exception as exc:reductions[(s,channel)]=(None,type(exc).__name__+': '+str(exc))
        pairs=[]
        for c,live in zip(self.clocks,self.mask):
            row=list(self._raw['source_before']['wire_body_paths']).index(f'/World/envs/env_{c.env_id}/Wire')
            for s in SURFACES:
                fields=[]
                for channel in ('normal','friction'):
                    values,reason=reductions[(s,channel)]
                    if errors:reason='; '.join(errors)
                    if not live or c.reset:reason='inactive slot' if not live else 'reset snapshot'
                    fields.append(_field(c,s,channel,None if reason else values[row].tolist(),reason))
                reason='normal or friction unavailable';result=None
                if all(f.valid for f in fields):
                    with np.errstate(over='ignore',invalid='ignore'):combined=np.asarray(fields[0].value)+np.asarray(fields[1].value)
                    if np.isfinite(combined).all():result=combined.tolist();reason=None
                    else:reason='nonfinite normal plus friction resultant'
                path=self.bundle.filter_paths[s][row] if s in routing else None
                pairs.append(PassivePairValue(c,s,row,0,f'/World/envs/env_{c.env_id}/Wire',path,
                    *fields,_field(c,s,'resultant',result,reason),
                    pair_basis='SDK_resolved_sensor_filter_paths_only' if path is not None else 'unavailable'))
        valid=not errors;epoch=_digest(self._raw['source_before'])
        snapshot=PassivePairSnapshot(valid,tuple(errors),self.clocks,tuple(pairs),_freeze(self._raw),epoch)
        # Physical terminal retirement is independent of numeric validity.
        for c,live in zip(self.clocks,self.mask):
            if live and c.terminal:
                self._terminals.setdefault((c.env_id,c.episode_uid),tuple(p for p in pairs if p.clock.env_id==c.env_id))
                self._retired.add((c.env_id,c.episode_uid));self._previous[c.env_id]=c
                self._seen.setdefault(c.env_id,set()).add(c.episode_uid)
        if valid:
            self._epoch=epoch
            self._retired.update((r['after']['env_id'],r['after']['episode_uid']) for r in self._raw['retired_inactive_generation_changes'])
            for c,live in zip(self.clocks,self.mask):
                self._previous[c.env_id]=c;self._seen.setdefault(c.env_id,set()).add(c.episode_uid)
        self._values.clear();self._open=False;self.last_snapshot=snapshot
        return snapshot
