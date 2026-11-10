"""Pure F0 sensor/filter metadata binding; never a contact or Native verifier.

The caller supplies FRESH owned metadata, actual getter-clock brackets and
active stage/API identities. This module does not read a World or trust a
configuration expression as a resolved path. Old unclocked capture manifests
cannot be given a fresh physical timestamp by this function.

Only the current 34 one-body, one-Wire-filter hand views are supported. Actual
actor/collider IDs, count/start ownership, force side, support, overflow absence
and Native qualification remain unavailable. No existing capture/control or
Student input path imports or enables this module by default.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import re
from types import MappingProxyType

from .tabletop_bimanual_mpc_state import SampleClock


VERSION = 'dh116-filtered-pair-metadata-f0-v1'
PAIR_BASIS = 'SDK_resolved_sensor_filter_paths_only'
_FILTER_SOURCE = 'actual contact_physx_view.filter_paths getter'
_HAND_PATH = re.compile(r'^/World/envs/env_(0|[1-9][0-9]*)/(left|right)Hand/([A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)*)$')
_WIRE_PATH = re.compile(r'^/World/envs/env_(0|[1-9][0-9]*)/Wire$')
_SHA = re.compile(r'^[0-9a-fA-F]{64}$')
_QUALIFICATION = MappingProxyType(dict(native_attested=False,resolved_pair_qualified=False,
    per_contact_partner_id_qualified=False,contact_values_qualified=False,
    force_sign_frame_qualified=False,upstream_overflow_absence_attested=False,
    support_qualified=False,planning_eligible=False,new_expert_qualification_issued=False,
    native_success_qualification_issued=False))


def _freeze(value):
    if isinstance(value,SampleClock):
        return _freeze(asdict(value))
    if value is None or isinstance(value,(str,bool)):
        return value
    if isinstance(value,Integral):
        return int(value)
    if isinstance(value,Real):
        return float(value)  # invalid/nonfinite raw metadata is preserved, never zeroed
    if isinstance(value,Mapping):
        if any(not isinstance(key,str) for key in value):
            raise ValueError('raw metadata mapping keys must be strings')
        return MappingProxyType({key:_freeze(item) for key,item in value.items()})
    if isinstance(value,(list,tuple)):
        return tuple(_freeze(item) for item in value)
    raise ValueError('F0 requires already owned JSON-like metadata, not SDK/custom objects')


def _plain(value):
    if is_dataclass(value):
        return {field.name:_plain(getattr(value,field.name)) for field in fields(value)}
    if isinstance(value,Mapping):
        return {key:_plain(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True,slots=True)
class FilteredSensorPair:
    clock: SampleClock
    sensor_index: int
    sensor_row: int
    filter_column: int
    hand: str
    body_name: str
    body_relative_path: str
    sensor_path: str
    filter_path: str
    contact_view_identity: Mapping
    pair_basis: str = PAIR_BASIS
    partner_body_id: None = None
    actor_id: None = None
    collider_id: None = None

    def __post_init__(self):
        object.__setattr__(self,'contact_view_identity',_freeze(self.contact_view_identity))
        if not isinstance(self.clock,SampleClock):
            raise ValueError('pair requires actual SampleClock')
        if any(value is not None for value in (self.partner_body_id,self.actor_id,self.collider_id)):
            raise ValueError('F0 paths are not numeric actor/collider/partner IDs')


@dataclass(frozen=True,slots=True)
class FilteredPairSnapshot:
    metadata_binding_valid: bool
    errors: tuple[str,...]
    pairs: tuple[FilteredSensorPair,...]
    clocks: tuple[SampleClock,...]
    raw: Mapping
    binding_fingerprint: str | None
    version: str = VERSION
    contact_values_audited: bool = False
    contact_values_missing_reason: str = 'F0 metadata only; contact tuple count/start ownership, capacity, values and force semantics not audited'

    def __post_init__(self):
        object.__setattr__(self,'raw',_freeze(self.raw))
        for name in ('errors','pairs','clocks'):
            object.__setattr__(self,name,tuple(getattr(self,name)))

    @property
    def qualification(self):
        return _QUALIFICATION

    def to_dict(self):
        return dict(_plain(self),qualification=dict(self.qualification))


def _sequence(value,label):
    if not isinstance(value,(tuple,list)):
        raise ValueError(label+' must be an explicit sequence')
    return tuple(value)


def _clocks(before,after):
    before,after=_sequence(before,'clocks_before'),_sequence(after,'clocks_after')
    if not before or any(not isinstance(c,SampleClock) for c in before+after):
        raise ValueError('actual SampleClock objects required')
    by_env={c.env_id:c for c in before}
    if len(by_env)!=len(before) or len(after)!=len(before) or {c.env_id:c for c in after}!=by_env:
        raise ValueError('getter clock bracket must preserve unique env/episode/frame identities')
    keys={(c.run_uuid,c.world_step,c.world_time_s,c.sdk_step,c.sdk_time_s,c.policy_step) for c in before}
    if len(keys)!=1 or any(c.reset for c in before):
        raise ValueError('one actual physical frame required; reset metadata needs an independent rebuild')
    return before,by_env


def _source(identity):
    if not isinstance(identity,Mapping) or not isinstance(identity.get('before'),Mapping):
        raise ValueError('actual source identity before/after bracket required')
    before=identity['before']
    if before!=identity.get('after'):
        raise ValueError('actual active stage/API changed or missing across metadata acquisition')
    stage,api=before.get('stage_identity'),before.get('contact_view_api')
    if not isinstance(stage,str) or not stage.strip() or not isinstance(api,Mapping):
        raise ValueError('actual stage identity and contact view API source required')
    path,sha,kind=api.get('path'),api.get('sha256'),api.get('python_type')
    if (not isinstance(path,str) or not path.startswith('/') or not isinstance(sha,str)
            or _SHA.fullmatch(sha) is None or not isinstance(kind,str) or not kind.strip()):
        raise ValueError('actual active API path/SHA256/Python type identity required')
    return before,kind


def _previous(previous,clocks,fingerprint):
    if previous is None:
        return
    if not isinstance(previous,FilteredPairSnapshot) or previous.metadata_binding_valid is not True:
        raise ValueError('previous must be a valid F0 metadata snapshot')
    if previous.binding_fingerprint!=fingerprint:
        raise ValueError('actual stage/API/view/path binding changed; rebuild independently')
    prior={c.env_id:c for c in previous.clocks}
    if set(prior)!={c.env_id for c in clocks}:
        raise ValueError('actual environment cohort changed; rebuild independently')
    for current in clocks:
        old=prior[current.env_id]
        if (old.terminal or old.reset or any(getattr(old,name)!=getattr(current,name)
                for name in ('run_uuid','env_id','episode_uid','reset_generation'))):
            raise ValueError('cannot retimestamp old metadata across first terminal/reset/episode')
        delta=current.world_step-old.world_step
        if (delta<=0 or current.sdk_step-old.sdk_step!=delta or current.policy_step<old.policy_step
                or not math.isclose(current.world_time_s-old.world_time_s,delta/480,rel_tol=0.,abs_tol=1e-7)
                or not math.isclose(current.sdk_time_s-old.sdk_time_s,delta/480,rel_tol=0.,abs_tol=1e-7)):
            raise ValueError('stale/discontinuous World/SDK/policy clock metadata')


def resolve_filtered_sensor_pairs(sensor_sources,*,wire_body_paths,clocks_before,clocks_after,
                                  source_identity,previous=None):
    """Audit fresh source metadata without a backend read, hook or qualification.

    Accepts the existing sensor_sources shape with fresh acquisition additions:
    each sensor needs ``contact_view_identity={python_runtime_id,python_type}``;
    source_identity needs equal ``before/after`` records containing
    ``stage_identity`` and ``contact_view_api={path,sha256,python_type}``.
    Clock brackets must surround the ACTUAL acquisition, not a manifest reread.
    Pure structural validation cannot independently attest these caller sources.

    Invalid JSON-like evidence returns no pair credit and retains owned raw and
    errors. Custom/borrowed objects are refused. Numeric contact values and their
    count/start intervals are intentionally not inputs to this F0-only function.
    """
    raw=_freeze(dict(sensor_sources=sensor_sources,wire_body_paths=wire_body_paths,
        clocks_before=clocks_before,clocks_after=clocks_after,source_identity=source_identity))
    try:
        clocks,by_env=_clocks(clocks_before,clocks_after)
        source,view_type=_source(raw['source_identity'])
        paths=_sequence(raw['wire_body_paths'],'actual Wire root view paths')
        wires={}
        for path in paths:
            match=_WIRE_PATH.fullmatch(path) if isinstance(path,str) else None
            if match is None or int(match[1]) in wires:
                raise ValueError('actual exact unique Wire root view paths required')
            wires[int(match[1])]=path
        if set(wires)!=set(by_env):
            raise ValueError('actual Wire root view does not match clock environment cohort')
        metadata=raw['sensor_sources']
        if not isinstance(metadata,Mapping) or metadata.get('available') is not True:
            raise ValueError('actual sensor_sources metadata unavailable')
        rows=_sequence(metadata.get('sensors'),'actual sensor rows')
        if len(rows)!=34:
            raise ValueError('current F0 contract requires the original 34 hand sensor views')
        indices=set();view_ids=set();body_sets={'left':set(),'right':set()};pairs=[];signature=[]
        for record in rows:
            if not isinstance(record,Mapping):
                raise ValueError('actual sensor row mapping required')
            index=record.get('sensor_index')
            if type(index) is not int or index not in range(34) or index in indices:
                raise ValueError('34 unique actual sensor indices required')
            indices.add(index)
            view=record.get('contact_view_identity')
            if (not isinstance(view,Mapping) or type(view.get('python_runtime_id')) is not int
                    or view['python_runtime_id']<=0 or view['python_runtime_id'] in view_ids
                    or view.get('python_type')!=view_type):
                raise ValueError('actual distinct contact view lifecycle/type identity required')
            view_ids.add(view['python_runtime_id'])
            sensor_paths=_sequence(record.get('sensor_paths'),'actual sensor_paths')
            bodies=_sequence(record.get('body_prim_paths'),'actual body_prim_paths')
            if (sensor_paths!=bodies or len(sensor_paths)!=len(by_env)
                    or type(record.get('filter_count')) is not int or record['filter_count']!=1
                    or ('sensor_count' in record and (type(record['sensor_count']) is not int
                        or record['sensor_count']!=len(sensor_paths)))):
                raise ValueError('actual sensor/body paths or sensor/filter counts disagree')
            filters=record.get('filter_paths')
            if (not isinstance(filters,Mapping) or filters.get('available') is not True
                    or filters.get('source')!=_FILTER_SOURCE):
                raise ValueError('actual filter_paths getter evidence required; configured expressions are not fallback')
            resolved=_sequence(filters.get('value'),'actual resolved filter_paths')
            if len(resolved)!=len(sensor_paths):
                raise ValueError('actual filter rows do not match sensor rows')
            envs=set();sensor_body=None
            for row,(sensor_path,filter_row) in enumerate(zip(sensor_paths,resolved)):
                match=_HAND_PATH.fullmatch(sensor_path) if isinstance(sensor_path,str) else None
                if match is None:
                    raise ValueError('actual exact per-environment hand rigid-body path required')
                env,side,body=int(match[1]),match[2],match[3]
                if env not in by_env or env in envs or (sensor_body is not None and sensor_body!=(side,body)):
                    raise ValueError('sensor rows must bind one same hand/body to every distinct environment')
                sensor_body=(side,body);envs.add(env)
                if not isinstance(filter_row,(list,tuple)) or tuple(filter_row)!=(wires[env],):
                    raise ValueError('actual singleton Wire filter must match that same sensor environment')
                pairs.append(FilteredSensorPair(by_env[env],index,row,0,side,body.rsplit('/',1)[-1],body,
                    sensor_path,wires[env],view))
            if envs!=set(by_env) or sensor_body in body_sets[sensor_body[0]]:
                raise ValueError('missing environment or duplicate actual hand sensor body')
            body_sets[sensor_body[0]].add(sensor_body)
            signature.append((index,view,sensor_paths,bodies,resolved))
        if any(len(values)!=17 or len({body.rsplit('/',1)[-1] for _,body in values})!=17
               for values in body_sets.values()):
            raise ValueError('exactly 17 distinct sensor paths and unambiguous body names for each hand required')
        fingerprint=hashlib.sha256(json.dumps(_plain(dict(source=source,wires=sorted(wires.items()),
            sensors=sorted(signature,key=lambda row:row[0]))),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        _previous(previous,clocks,fingerprint)
        return FilteredPairSnapshot(True,(),tuple(pairs),clocks,raw,fingerprint)
    except (ValueError,TypeError,KeyError,AttributeError) as error:
        return FilteredPairSnapshot(False,(type(error).__name__+': '+str(error),),(),(),raw,None)
