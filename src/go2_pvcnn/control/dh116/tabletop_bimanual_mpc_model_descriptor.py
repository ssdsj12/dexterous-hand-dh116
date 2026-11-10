"""Opt-in read-only composed mounted-tree descriptor, not FK/RNEA or control.

The caller supplies the EXISTING stage, hand views and scene view.  An owned
ArticulationMeasurementSample at the exact same physical clock supplies q/qd,
binding and inertial readback; this module never replays it as a fresh getter.
Only gravity/disable-gravity/link-transform/Jacobian getters are added.  No
World/view creation, advance, refresh, setter or Actor input path exists here.

The USD facade lazily imports pxr only when explicitly used.  CPU fixtures
inject a facade; their structure/collection validity never attests Native.
Raw unknowns and authored sentinels are retained.  All dynamics, candidate,
motor, planning, Native success and hardware qualifications remain false.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from types import MappingProxyType

import numpy as np

from .tabletop_bimanual_mpc_state import SampleClock
from .tabletop_bimanual_mpc_articulation_measurements import (
    ArticulationMeasurementSample, _binding, _metatype, _owned, _plain,
    _source_record)


VERSION = 'dh116-mounted-model-descriptor-v1'
_SIDES = ('left','right')
_READS = ('get_disable_gravities','get_link_transforms','get_jacobians')
_FIELDS = ('q','qd','link_mass_raw','link_com_pose_xyzw_raw','link_inertia_9_raw','armature')
_FIELD_GETTERS = dict(q=('get_dof_positions',(17,)),qd=('get_dof_velocities',(17,)),
    link_mass_raw=('get_masses',(23,)),link_com_pose_xyzw_raw=('get_coms',(23,7)),
    link_inertia_9_raw=('get_inertias',(23,9)),armature=('get_dof_armatures',(17,)))
_QUALIFICATION = dict(dynamics_qualified=False, candidate_future_dynamics_qualified=False,
    planning_eligible=False, hardware_measurement_qualified=False, native_attested=False,
    native_success_qualification_issued=False, solver_motor_torque_qualified=False)
_SEMANTICS = dict(inertia_frame_qualified=False,armature_inclusion_qualified=False,
    compensation_sign_qualified=False,mimic_dynamics_qualified=False,
    motor_only_torque_qualified=False,contact_model_qualified=False,
    jacobian_frame_and_root_row_qualified=False,link_transform_quaternion_order_qualified=False)


def _thaw(v):
    if isinstance(v,Mapping): return {str(k):_thaw(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [_thaw(x) for x in v]
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    return deepcopy(v)


def _freeze(v):
    if isinstance(v,Mapping): return MappingProxyType({str(k):_freeze(x) for k,x in v.items()})
    if isinstance(v,(list,tuple)): return tuple(_freeze(x) for x in v)
    if isinstance(v,np.generic): return v.item()
    if v is None or isinstance(v,(str,bool,int,float)): return v
    raise TypeError('descriptor contains non-owned value: '+type(v).__name__)


def _digest(v):
    return hashlib.sha256(json.dumps(_thaw(v),sort_keys=True,separators=(',',':'),allow_nan=True).encode()).hexdigest()


def _record_error(exc):
    return dict(value=None,available=False,missing_reason=type(exc).__name__+': '+str(exc),
                raw_shape=None,raw_dtype=None,source_device=None)


def _partial_owned(value):
    """A bad diagnostic value never replaces the original acquisition error."""
    try:return _freeze(_thaw(value))
    except Exception as exc:return _freeze(_record_error(exc))


def _attr(row,name):
    r=row.get('attributes',{}).get(name)
    if not isinstance(r,dict) or r.get('available') is not True:
        raise ValueError('missing composed attribute '+name)
    return r.get('value')


def _rigid(m):
    a=np.asarray(m,dtype=float)
    return (a.shape==(4,4) and np.isfinite(a).all()
        and np.allclose(a[3],[0,0,0,1],atol=1e-8,rtol=0)
        and np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=1e-6,rtol=0)
        and math.isclose(float(np.linalg.det(a[:3,:3])),1.,abs_tol=1e-6,rel_tol=0))


def assemble_mounted_graph(payload,binding):
    """Validate a full actual row binding; retain raw separately on failure.

    Attribute provenance/sentinels remain data, not inferred inertia or forces.
    This verifies graph structure only, never kinematic reference/sign semantics.
    """
    p=_thaw(payload); meta=binding['metatype']
    if (binding.get('fixed_base') is not True or binding.get('dof_count')!=17
            or binding.get('link_count')!=23 or meta.get('joint_count')!=22):
        raise ValueError('only actual fixed-base mounted 17D/23L/22J descriptor supported')
    units=p.get('units',{})
    if (any(isinstance(units.get(k),bool) or not isinstance(units.get(k),Real) or units[k]!=1.
            for k in ('meters_per_unit','kilograms_per_unit')) or units.get('up_axis') not in ('X','Y','Z')):
        raise ValueError('actual SI units/up axis required; no normalization')
    links=p.get('links',[]); paths=[r['path'] for r in links]
    if len(paths)!=23 or len(set(paths))!=23 or set(paths)!=set(binding['link_paths']):
        raise ValueError('composed links do not exactly cover actual backend link paths')
    bypath={r['path']:r for r in links}
    links=[bypath[path] for path in binding['link_paths']]
    for r in links:
        if not _rigid(r.get('xform',{}).get('matrix')):
            raise ValueError('link transform is not proper rigid; scale/shear/reflection unsupported')
    js=p.get('joints',[])
    if len(js)!=22 or len({r['path'] for r in js})!=22:
        raise ValueError('complete unique internal composed joint paths required')
    names=meta['joint_names']; byname={}
    for r in js:
        n=r['path'].rsplit('/',1)[-1]
        if n in byname: raise ValueError('ambiguous composed joint basename')
        byname[n]=r
    if set(byname)!=set(names): raise ValueError('actual joint rows and composed joint names disagree')
    js=[byname[n] for n in names]; child_to_parent={}; dof_joint_rows=[None]*17
    for row,r in enumerate(js):
        if r.get('active') is not True or r.get('defined') is not True:
            raise ValueError('inactive/undefined joint cannot be a backend internal edge')
        if _attr(r,'physics:jointEnabled') is not True or _attr(r,'physics:excludeFromArticulation') is not False:
            raise ValueError('disabled/excluded actual internal joint')
        a,b=r.get('body0'),r.get('body1')
        if (not isinstance(a,list) or not isinstance(b,list) or len(a)!=1 or len(b)!=1
                or a[0] not in bypath or b[0] not in bypath or a[0]==b[0] or b[0] in child_to_parent):
            raise ValueError('joint graph has missing/multiple body or duplicate child')
        child_to_parent[b[0]]=a[0]
        for key,size in (('localPos0',3),('localPos1',3),('localRot0',4),('localRot1',4)):
            value=np.asarray(_attr(r,'physics:'+key),dtype=float)
            if value.shape!=(size,) or not np.isfinite(value).all(): raise ValueError('invalid '+key)
            if size==4 and not math.isclose(float(np.linalg.norm(value)),1.,abs_tol=1e-5,rel_tol=0):
                raise ValueError('unit USD wxyz quaternion required')
        count,offset=meta['joint_dof_counts'][row],meta['joint_dof_offsets'][row]
        typ=r.get('type_name')
        if count==0:
            if typ!='PhysicsFixedJoint': raise ValueError('zero DOF row must be fixed joint')
        elif count==1:
            if typ not in ('PhysicsRevoluteJoint','PhysicsPrismaticJoint') or _attr(r,'physics:axis') not in ('X','Y','Z'):
                raise ValueError('unknown moving joint type/axis')
            if not 0<=offset<17 or dof_joint_rows[offset] is not None:
                raise ValueError('joint DOF coverage is not unique')
            expected=0 if typ=='PhysicsRevoluteJoint' else 1
            if binding['dof_types'][offset]!=expected or binding['dof_paths'][offset]!=r['path']:
                raise ValueError('moving composed joint type/path does not match actual DOF row')
            dof_joint_rows[offset]=row
        else: raise ValueError('multi-axis joint unsupported')
        # Only validate relationship binding; no perfect-mimic or compliance claim.
        for mimic in r.get('mimic_instances',[]):
            ref=mimic.get('reference_joint')
            if not isinstance(ref,list) or len(ref)!=1 or ref[0] not in {j['path'] for j in js}:
                raise ValueError('mimic reference is not uniquely bound inside articulation')
    roots=set(paths)-set(child_to_parent)
    if len(roots)!=1 or any(x is None for x in dof_joint_rows): raise ValueError('joint tree root/DOF coverage incomplete')
    root=next(iter(roots))
    for path in paths:
        seen=set(); here=path
        while here in child_to_parent:
            if here in seen: raise ValueError('cyclic composed joint graph')
            seen.add(here); here=child_to_parent[here]
        if here!=root: raise ValueError('disconnected composed joint graph')
    attachments=p.get('world_attachments',[])
    valid=[r for r in attachments if r.get('type_name')=='PhysicsFixedJoint'
        and r.get('active') is True and r.get('defined') is True
        and _attr(r,'physics:jointEnabled') is True
        and (r.get('body0')==[] and r.get('body1')==[root] or r.get('body1')==[] and r.get('body0')==[root])]
    if len(valid)!=1 or len(attachments)!=1: raise ValueError('one actual world-fixed root attachment required')
    return dict(p,links=links,joints=js,root_link_row=binding['link_paths'].index(root),dof_joint_rows=dof_joint_rows,
        joint_row_mapping='actual metatype names and DOF offsets, exact composed paths; not traversal order',
        kinematic_reference_and_sign_qualified=False)


@dataclass(frozen=True)
class MountedModelDescriptorSample:
    clock: SampleClock
    descriptor: object
    raw: object
    structure_valid: bool
    collection_valid: bool
    errors: tuple
    qualification: object


class MountedModelDescriptorProvider:
    """Existing handles only; per-read all-live clocks bracket one transaction."""

    def __init__(self,*,stage,hands,scene_view,clock_reader,source_reader,stage_reader=None,
                 stage_facade=None,enabled=False):
        if type(enabled) is not bool: raise ValueError('explicit bool enabled required')
        self.stage,self.stage_reader,self.hands=stage,stage_reader,hands
        self.scene_view,self.clock_reader,self.source_reader=scene_view,clock_reader,source_reader
        self.stage_facade=stage_facade
        self.enabled=enabled; self._previous={}; self._identities={}; self._capturing=False
        self._episode_generations={}; self._scene_identity=None
        self.last_samples=None; self.last_partial=None

    def _transition(self,c):
        known=self._episode_generations.get((c.env_id,c.episode_uid))
        if known is not None and known!=c.reset_generation:
            raise ValueError('episode UID reused across reset generations')
        old=self._previous.get(c.env_id)
        if old is None: return
        if c.run_uuid!=old.run_uuid: raise ValueError('run identity changed; rebuild descriptor provider')
        delta=c.world_step-old.world_step
        if delta<=0 or c.sdk_step-old.sdk_step!=delta or any(not math.isclose(
                getattr(c,k)-getattr(old,k),delta/480,abs_tol=1e-7,rel_tol=0)
                for k in ('world_time_s','sdk_time_s')):
            raise ValueError('stale/repeated/misaligned physical clock')
        if old.terminal and c.episode_uid==old.episode_uid: raise ValueError('terminal episode cannot resume')
        if c.episode_uid!=old.episode_uid:
            if c.reset_generation<=old.reset_generation: raise ValueError('episode identity changed without reset generation')
        elif c.reset_generation!=old.reset_generation: raise ValueError('reset generation changed within episode')
        if c.policy_step<old.policy_step: raise ValueError('policy clock moved backwards')

    def capture_batch(self,*,clocks,articulation_samples,active_mask=None):
        self.last_samples=None; self.last_partial=None
        clocks=tuple(clocks); samples=tuple(articulation_samples)
        if not clocks or any(not isinstance(c,SampleClock) for c in clocks) or len(clocks)!=len(samples):
            raise ValueError('one actual articulation sample per SampleClock required')
        if len({c.env_id for c in clocks})!=len(clocks): raise ValueError('duplicate environment')
        if len({(c.run_uuid,c.world_step,c.sdk_step,c.world_time_s,c.sdk_time_s) for c in clocks})!=1:
            raise ValueError('one physical clock frame required per batch')
        active_mask=(True,)*len(clocks) if active_mask is None else tuple(active_mask)
        if len(active_mask)!=len(clocks) or any(type(a) is not bool for a in active_mask):
            raise ValueError('explicit boolean active mask required')
        if self._capturing: raise ValueError('nested/concurrent descriptor capture unsupported')
        live=[(c,s) for c,s,a in zip(clocks,samples,active_mask) if self.enabled and a and not c.reset]
        for c,s in live:
            self._transition(c)
            if not isinstance(s,ArticulationMeasurementSample) or s.clock!=c:
                raise ValueError('articulation sample clock/episode identity does not match live frame')
        scene_handle=self.scene_view if live else None
        if live and self._scene_identity is not None and self._scene_identity!=id(scene_handle):
            raise RuntimeError('existing scene view identity changed; rebuild provider')
        brackets=[]; handles={}; source_before={}; graph_before={}; bindings={}; raw_reads={}; owned_reads={}
        gravity=None; after={}; source_after={}
        def current():
            for c,_ in live:
                if self.clock_reader(c.env_id)!=c: raise RuntimeError('actual clock/episode changed during descriptor capture')
            if live:
                if self.stage_reader is None or self.stage_reader() is not self.stage:
                    raise RuntimeError('actual active stage identity changed or unavailable')
                if self.scene_view is not scene_handle:
                    raise RuntimeError('actual scene view identity changed during capture')
                for side,view in handles.items():
                    if self.hands[side].root_physx_view is not view: raise RuntimeError('actual hand view identity changed')
        def read(name,fn):
            current(); receipt=dict(name=name,before=[asdict(c) for c,_ in live],after=None)
            brackets.append(receipt)
            try:
                value=fn()
                # Never deepcopy a live SDK handle. All remaining reads already
                # return primitive metadata or the existing owned-array record.
                owned_reads[name]=_partial_owned(dict(python_runtime_id=id(value),python_type=type(value).__module__+'.'+type(value).__qualname__)
                    if name.endswith('.existing_view') else value)
                return value
            except Exception as exc:
                receipt['error']=type(exc).__name__+': '+str(exc)
                owned_reads.setdefault(name,_record_error(exc))
                raise
            finally:
                try:
                    current();receipt['after']=[asdict(c) for c,_ in live]
                except Exception as exc:
                    receipt['after_error']=type(exc).__name__+': '+str(exc)
                    raise
        def source(side,path):
            v=read(side+'.source_identity',lambda:_thaw(self.source_reader(side,path)))
            if not isinstance(v,dict) or v.get('root_view_python_identity')!=id(handles[side]):
                raise RuntimeError('actual source view identity is unavailable or disagrees')
            if v.get('articulation_path')!=path or not v.get('stage_identity'):
                raise RuntimeError('actual source path/stage identity unavailable')
            sdk=v.get('sdk_sources')
            if not isinstance(sdk,list) or not sdk or any(not isinstance(r,dict) or not r.get('path')
                    or not isinstance(r.get('sha256'),str) or len(r['sha256'])!=64
                    or any(c not in '0123456789abcdef' for c in r['sha256']) for r in sdk):
                raise RuntimeError('actual SDK source identity/SHA unavailable')
            return v
        def layout(side,v):
            metadata={}
            for key in ('count','max_dofs','max_links','is_homogeneous','prim_paths','dof_paths','link_paths'):
                metadata[key]=read(side+'.'+key,lambda key=key:_plain(getattr(v,key)))
            metadata['shared_metatype']=read(side+'.shared_metatype',lambda:_metatype(v.shared_metatype))
            metadata['metatypes']=[read(side+f'.get_metatype({i})',lambda i=i:_metatype(v.get_metatype(i))) for i in range(metadata['count'])]
            hand_names=read(side+'.hand_names',lambda:dict(joint_names=list(self.hands[side].joint_names),body_names=list(self.hands[side].body_names)))
            return metadata,hand_names
        facade=self.stage_facade
        pending=dict(self._identities)
        self._capturing=True
        try:
            current()
            if live and facade is None: facade=ComposedUSDStageFacade()
            for side in _SIDES if live else ():
                handles[side]=read(side+'.existing_view',lambda:self.hands[side].root_physx_view)
                v=handles[side]
                metadata,hand_names=layout(side,v)
                for c,s in live:
                    key=(side,c.env_id); old=s.raw.get('hands',{}).get(side,{})
                    b=_binding(metadata,side=side,clock=c,hand_names=hand_names)
                    if b!=old.get('binding'): raise RuntimeError('live metadata binding differs from articulation sample identity')
                    bindings[key]=b
                    src=source(side,b['articulation_path'])
                    if src!=old.get('source_identity'): raise RuntimeError('live source identity differs from articulation sample')
                    sb=old.get('source_brackets',{})
                    if any(not sb.get(k,{}).get('available') or sb[k].get('value')!=src for k in ('before','after')):
                        raise RuntimeError('articulation sample source brackets unavailable or inconsistent')
                    identity=(id(self.stage),id(v),_source_record(src)['fingerprint'],_digest(b))
                    if key in self._identities and identity!=self._identities[key]:
                        raise RuntimeError('source/view/layout identity changed; rebuild descriptor provider')
                    pending[key]=identity; source_before[key]=src
                    try:
                        g=read(side+'.composed_before',lambda:_thaw(facade.capture(self.stage,b)))
                        graph_before[key]=dict(value=g,available=True,missing_reason=None,payload_sha256=_digest(g))
                    except Exception as exc:
                        current(); graph_before[key]=_record_error(exc)
                raw_reads[side]={}
                for name in _READS:
                    try: r=read(side+'.'+name,lambda name=name:_owned(getattr(v,name)()))
                    except RuntimeError as exc:
                        # Getter failure is NA; a clock/source race is never swallowed.
                        current(); r=_record_error(exc)
                    except Exception as exc: r=_record_error(exc)
                    raw_reads[side][name]=dict(r,source=f'hand[{side}].root_physx_view.{name}()')
            gravity=None
            if live:
                try: gravity=read('scene.get_gravity',lambda:_owned(scene_handle.get_gravity()))
                except Exception as exc:
                    current(); gravity=_record_error(exc)
                gravity['source']='existing_scene_view.get_gravity()'
            after={}; source_after={}
            for c,_ in live:
                for side in _SIDES:
                    key=(side,c.env_id); b=bindings[key]
                    src=source(side,b['articulation_path'])
                    if src!=source_before[key]: raise RuntimeError('actual source identity changed during capture')
                    source_after[key]=src
                    try:
                        g=read(side+'.composed_after',lambda:_thaw(facade.capture(self.stage,b)))
                        rec=dict(value=g,available=True,missing_reason=None,payload_sha256=_digest(g))
                    except Exception as exc:
                        current(); rec=_record_error(exc)
                    if (rec['available']!=graph_before[key]['available'] or
                            rec.get('payload_sha256')!=graph_before[key].get('payload_sha256')):
                        raise RuntimeError('composed payload/source changed during descriptor capture')
                    after[key]=rec
            for side in _SIDES if live else ():
                metadata,hand_names=layout(side,handles[side])
                for c,_ in live:
                    try: actual=_binding(metadata,side=side,clock=c,hand_names=hand_names)
                    except Exception as exc: raise RuntimeError('actual metadata/layout binding changed during capture') from exc
                    if actual!=bindings[(side,c.env_id)]: raise RuntimeError('actual layout binding changed during capture')
            current(); result=[]
            for c,s,a in zip(clocks,samples,active_mask):
                reason='provider disabled' if not self.enabled else 'inactive slot' if not a else 'reset snapshot' if c.reset else None
                raw=dict(version=VERSION,read_brackets=deepcopy(brackets),hands={},gravity=deepcopy(gravity))
                raw['handle_identity']=dict(stage_python_identity=id(self.stage) if live else None,
                    scene_view_python_identity=id(scene_handle) if live else None,
                    scene_view_python_type=type(scene_handle).__module__+'.'+type(scene_handle).__qualname__ if live else None,
                    hand_view_python_identities={side:id(view) for side,view in handles.items()},
                    identity_is_not_native_attestation=True)
                desc=dict(version=VERSION,clock=asdict(c),hands={},source_scope='simulation privileged read-only descriptor; not hardware')
                errors=[]; structure=True; collection=True
                if reason: errors.append(reason); structure=collection=False
                else:
                    if not _valid_array(gravity,(3,)): errors.append('scene gravity unavailable/nonfinite'); collection=False
                    desc['gravity_readback']=deepcopy(gravity)
                    for side in _SIDES:
                        key=(side,c.env_id); b=bindings[key]; n=s.raw['hands'][side]['metadata']['count']; index=b['selected_view_index']
                        rd=deepcopy(raw_reads[side]); measurements={}; receipts={}; local=[]
                        for name in _FIELDS:
                            field=s.fields.get(side+'.'+name)
                            if field is None or not field.valid or field.clock!=c:
                                local.append('articulation field unavailable: '+name); measurements[name]=None
                            else:
                                getter,shape=_FIELD_GETTERS[name]
                                raw_field=s.raw['hands'][side]['getters'].get(getter)
                                values=np.asarray(_thaw(field.value))
                                if (not _valid_array(raw_field,(n,)+shape,selected_index=index)
                                        or values.shape!=shape or not np.array_equal(values,np.asarray(raw_field['value'])[index])):
                                    local.append('articulation raw/field mismatch: '+name); measurements[name]=None
                                else: measurements[name]=values.tolist()
                                receipts[name]=dict(source=field.source,unit=field.unit,frame=field.frame,
                                    kind=field.kind,timestamp_s=field.timestamp_s,clock=asdict(field.clock),
                                    original_field_valid=field.valid,raw_getter=getter)
                        dims={'get_disable_gravities':(n,23),'get_link_transforms':(n,23,7),'get_jacobians':(n,22,6,17)}
                        for name,r in rd.items():
                            valid=_valid_array(r,dims[name],selected_index=index,allow_bool=name=='get_disable_gravities')
                            arr=np.asarray(r['value']) if valid else None
                            if valid and name=='get_disable_gravities': valid=bool(np.isin(arr[index],(0,1)).all())
                            if valid and name=='get_link_transforms': valid=bool(np.allclose(np.linalg.norm(arr[index,:,3:],axis=-1),1.,atol=1e-5,rtol=0))
                            r['selected']=arr[index].tolist() if valid else None
                            r['selected_valid']=valid
                            if not valid: local.append(name+' unavailable/shape/finite/enum validation failed')
                        gb=graph_before[key]; graph=None
                        try:
                            if not gb['available']: raise ValueError(gb['missing_reason'])
                            graph=assemble_mounted_graph(gb['value'],b)
                            if any(gb['value']['units'].get(k)!=source_before[key].get(k)
                                    for k in ('meters_per_unit','kilograms_per_unit')):
                                raise ValueError('composed and live source units disagree')
                        except (ValueError,TypeError,KeyError) as exc:
                            local.append('composed structure: '+str(exc)); structure=False
                        if local: collection=False; errors.extend(side+': '+e for e in local)
                        raw['hands'][side]=dict(composed_before=deepcopy(gb),composed_after=deepcopy(after[key]),
                            source_before=deepcopy(source_before[key]),source_after=deepcopy(source_after[key]),
                            readbacks=deepcopy(rd),articulation_input=_thaw(s.raw['hands'][side]))
                        desc['hands'][side]=dict(binding=deepcopy(b),graph=graph,measurements=measurements,measurement_receipts=receipts,
                            readbacks=rd,semantics=deepcopy(_SEMANTICS),source_identity=deepcopy(source_before[key]))
                result.append(MountedModelDescriptorSample(c,_freeze(desc),_freeze(raw),structure,
                    bool(collection and structure),tuple(errors),MappingProxyType(dict(_QUALIFICATION))))
            for c,_ in live:
                self._previous[c.env_id]=c
                self._episode_generations[(c.env_id,c.episode_uid)]=c.reset_generation
            if live: self._scene_identity=id(scene_handle)
            self._identities=pending; self.last_samples=tuple(result)
            return self.last_samples
        except BaseException as exc:
            self.last_partial=_freeze(dict(version=VERSION,complete_measurement=False,
                clocks=[asdict(c) for c in clocks],active_mask=list(active_mask),
                errors=[type(exc).__name__+': '+str(exc)],qualification=dict(_QUALIFICATION),
                raw=dict(read_brackets=brackets,owned_reads=owned_reads,gravity=_partial_owned(gravity),
                    hands={side:dict(readbacks={name:_partial_owned(record) for name,record in raw_reads.get(side,{}).items()},
                        sources_before={str(c.env_id):_partial_owned(source_before.get((side,c.env_id))) for c,_ in live},
                        sources_after={str(c.env_id):_partial_owned(source_after.get((side,c.env_id))) for c,_ in live},
                        composed_before={str(c.env_id):_partial_owned(graph_before.get((side,c.env_id))) for c,_ in live},
                        composed_after={str(c.env_id):_partial_owned(after.get((side,c.env_id))) for c,_ in live}) for side in _SIDES},
                    scope='owned evidence from this failed transaction only; never complete or Native-qualified')))
            raise
        finally: self._capturing=False


def _valid_array(r,shape,selected_index=None,allow_bool=False):
    if not isinstance(r,dict) or r.get('available') is not True: return False
    a=np.asarray(r.get('value'))
    return (a.shape==shape and (np.issubdtype(a.dtype,np.number) or allow_bool and np.issubdtype(a.dtype,np.bool_)) and not np.issubdtype(a.dtype,np.complexfloating)
        and bool(np.isfinite(a if selected_index is None else a[selected_index]).all()))


class ComposedUSDStageFacade:
    """Read ONLY the injected existing composed stage, no Stage.Open/Apply/Set.

    All attribute records retain default/authored opinions and property stack.
    No cooking inference, schema-default-to-physical-value substitution or
    quaternion/unit conversion occurs.  Unknown schema fields remain raw/NA.
    """

    def __init__(self,*,usd_geom=None,usd_physics=None):
        self.usd_geom,self.usd_physics=usd_geom,usd_physics

    def capture(self,stage,binding):
        geom,physics=self.usd_geom,self.usd_physics
        if geom is None or physics is None:
            from pxr import UsdGeom, UsdPhysics
            geom,physics=UsdGeom,UsdPhysics
        layers={}
        def layer_record(layer):
            key=layer.identifier
            if key not in layers:
                text=layer.ExportToString()
                layers[key]=dict(identifier=key,real_path=layer.realPath,
                    sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
            return deepcopy(layers[key])
        def value(v):
            if v is None or isinstance(v,(str,bool,int,float)): return v
            if hasattr(v,'GetReal') and hasattr(v,'GetImaginary'): return [float(v.GetReal()),*[float(x) for x in v.GetImaginary()]]
            try: return [value(x) for x in v]
            except TypeError: return str(v)
        def attribute(a):
            try:
                if not a: raise ValueError('attribute is not present in actual schema/composition')
                v=a.Get()
                if v is None: raise ValueError('attribute has no resolved value')
                stack=[dict(layer=layer_record(s.layer),path=str(s.path)) for s in a.GetPropertyStack()]
                return dict(value=value(v),available=True,authored=bool(a.HasAuthoredValueOpinion()),
                    property_stack=stack,resolve_source=str(a.GetResolveInfo().GetSource()),error=None)
            except Exception as exc:
                return dict(value=None,available=False,authored=None,property_stack=[],error=type(exc).__name__+': '+str(exc))
        def relationship(rel):
            try:
                return dict(targets=[str(p) for p in rel.GetTargets()],available=True,
                    authored=bool(rel.HasAuthoredTargets()),
                    property_stack=[dict(layer=layer_record(s.layer),path=str(s.path)) for s in rel.GetPropertyStack()],error=None)
            except Exception as exc:
                return dict(targets=None,available=False,authored=None,property_stack=[],error=type(exc).__name__+': '+str(exc))
        def row(prim):
            r=dict(path=str(prim.GetPath()),type_name=prim.GetTypeName(),active=bool(prim.IsActive()),
                defined=bool(prim.IsDefined()),instance_proxy=bool(prim.IsInstanceProxy()),
                applied_schemas=list(prim.GetAppliedSchemas()),attributes={})
            # Enumerating resolved attributes also retains custom compliance and
            # drive fields unknown to this installed Physx schema.
            r['attributes']={a.GetName():attribute(a) for a in prim.GetAttributes()
                if a.GetName().startswith(('physics:','physx','drive:'))}
            r['relationships']={rel.GetName():relationship(rel) for rel in prim.GetRelationships()
                if rel.GetName().startswith(('physics:','physx'))}
            return r
        cache=geom.XformCache(); links=[]
        for path in binding['link_paths']:
            prim=stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid(): raise ValueError('bound actual link missing in composed stage: '+path)
            r=row(prim)
            local,reset=geom.Xformable(prim).GetLocalTransformation(),geom.Xformable(prim).GetResetXformStack()
            # Gf matrices use row-vector convention; descriptor explicitly uses
            # column-vector matrices, while source attribute values remain raw.
            r['xform']=dict(matrix=np.asarray(cache.GetLocalToWorldTransform(prim),dtype=float).T.tolist(),
                local_matrix=np.asarray(local,dtype=float).T.tolist(),reset_stack=bool(reset),
                matrix_convention='column_vector_from_Gf_transpose')
            r['ancestor_xform_attributes']=[]; r['ancestor_physics_attributes']=[]; parent=prim
            while parent and str(parent.GetPath())!='/':
                attrs={a.GetName():attribute(a) for a in parent.GetAttributes()
                    if a.GetName().startswith('xformOp:') or a.GetName()=='xformOpOrder'}
                r['ancestor_xform_attributes'].append(dict(path=str(parent.GetPath()),attributes=attrs))
                r['ancestor_physics_attributes'].append(dict(path=str(parent.GetPath()),
                    attributes={a.GetName():attribute(a) for a in parent.GetAttributes()
                        if a.GetName().startswith(('physics:','physx'))}))
                parent=parent.GetParent()
            links.append(r)
        paths=set(binding['link_paths']); joints=[]; world=[]; inactive=[]
        # TraverseAll is needed to inventory disabled/deactivated old root_joint.
        for prim in stage.TraverseAll():
            if not prim.IsA(physics.Joint): continue
            j=physics.Joint(prim); a=[str(p) for p in j.GetBody0Rel().GetTargets()]; b=[str(p) for p in j.GetBody1Rel().GetTargets()]
            if not (set(a+b)&paths): continue
            r=row(prim); r.update(body0=a,body1=b,mimic_instances=[])
            for schema in r['applied_schemas']:
                if not schema.startswith('PhysxMimicJointAPI:'): continue
                instance=schema.split(':',1)[1]; prefix='physxMimicJoint:'+instance+':'
                rel=prim.GetRelationship(prefix+'referenceJoint')
                r['mimic_instances'].append(dict(instance=instance,
                    reference_joint=[str(p) for p in rel.GetTargets()] if rel else None,
                    attributes={k:v for k,v in r['attributes'].items() if k.startswith(prefix)},
                    compliance_backend_qualified=False))
            if not r['active'] or not r['defined'] or r['attributes'].get('physics:jointEnabled',{}).get('value') is not True:
                inactive.append(r)
            elif not a or not b: world.append(r)
            else: joints.append(r)
        # Scene attributes and mass hierarchy are raw composed intent, not new
        # runtime getter claims; the actual scene gravity is captured separately.
        scenes=[row(p) for p in stage.TraverseAll() if p.IsA(physics.Scene)]
        for layer in stage.GetLayerStack(includeSessionLayers=True): layer_record(layer)
        return dict(units=dict(meters_per_unit=geom.GetStageMetersPerUnit(stage),
            kilograms_per_unit=physics.GetStageKilogramsPerUnit(stage),up_axis=str(geom.GetStageUpAxis(stage))),
            links=links,joints=joints,world_attachments=world,inactive_joints=inactive,
            scenes=scenes,stage_layers=[layers[k] for k in sorted(layers)])
