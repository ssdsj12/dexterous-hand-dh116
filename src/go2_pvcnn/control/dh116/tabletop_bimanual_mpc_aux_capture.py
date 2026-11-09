"""Independent read-only auxiliary evidence; no MPC control or Actor inputs.

The whole ORIGINAL TensorWorld.advance is bracketed: it increments the local
World counter before calling sim.step. Poll only in the pre-reset observer,
after task.consume has identified the real terminal. Never add a physics step.
Global contact reports retain a representative clock AND all environment
clocks; their actor pairs are not silently assigned to that representative env.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import gzip
import inspect
import json
import math
from numbers import Integral, Real
from pathlib import Path

from .tabletop_bimanual_mpc_state import SampleClock
from .tabletop_bimanual_mpc_capture import (
    _clock, _identity, _integer, _mask, _native_attestation, _number,
    _plain, _raw_json_payload)


VERSION='dh116-mpc-aux-observation-v1'
DELIVERY_VERSION='dh116-mpc-contact-delivery-observation-v1'


@dataclass(frozen=True,slots=True)
class AuxCaptureComponents:
    """Explicit optional observer ownership; legacy CPU three-tuples still work."""
    scene: object
    pairs: object
    metadata: object
    delivery_probe: object = None
    filtered_metadata_provider: object = None


def _require_clock_delta(after,before,steps):
    if any(after[key]-before[key]!=steps for key in ('world_step','sdk_step')):
        raise ValueError('missing continuous auxiliary physical clock sample')
    if any(not math.isclose(after[key]-before[key],steps/480,rel_tol=1e-7,abs_tol=1e-7)
           for key in ('world_time_s','sdk_time_s')):
        raise ValueError('auxiliary SDK/World time increment differs from original physics period')


def _validated_advance_chain(env,world,*,world_type,task_type,sequence_type):
    """Accept only the original method or the exact existing readback sidecar.

    Sequence must first validate its original command contract. Aux then wraps
    that known sidecar, whose saved advance/control must still be the original
    class methods. Arbitrary or nested control replacements remain rejected.
    This source-chain check alone grants no native or measurement qualification.
    """
    task=getattr(env,'task',None)
    if type(world) is not world_type or type(task) is not task_type:
        raise ValueError('original physical World/task source types required')
    advance=world.advance
    if (getattr(advance,'__func__',None) is world_type.advance
            and getattr(advance,'__self__',None) is world):
        if (getattr(task.control,'__func__',None) is not task_type.control
                or getattr(task.control,'__self__',None) is not task):
            raise ValueError('original task control source required')
        return 'original'
    sequence=getattr(advance,'__self__',None)
    if (type(sequence) is not sequence_type
            or getattr(advance,'__func__',None) is not sequence_type._after_advance
            or sequence.world is not world or sequence.task is not task
            or sequence.closed is not False or sequence.require_native_readback is not True
            or getattr(sequence._advance,'__func__',None) is not world_type.advance
            or getattr(sequence._advance,'__self__',None) is not world
            or getattr(sequence._control,'__func__',None) is not task_type.control
            or getattr(sequence._control,'__self__',None) is not task
            or getattr(task.control,'__func__',None) is not sequence_type._before_control
            or getattr(task.control,'__self__',None) is not sequence):
        raise ValueError('original advance or exact source-bound Sequence readback/control chain required')
    return 'SequenceContextCapture->original'


def _serial(value):
    if value is None or isinstance(value,(str,bool)):
        return value
    if isinstance(value,Integral):return int(value)
    if isinstance(value,Real):return float(value)
    if is_dataclass(value):
        return {f.name:_serial(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,Mapping):return {key:_serial(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_serial(item) for item in value]
    if hasattr(value,'tolist'):return _serial(value.tolist())
    # Test facade results only; genuine providers return dataclasses/mappings.
    if hasattr(value,'__dict__'):return _serial(vars(value))
    raise ValueError('auxiliary provider returned unsupported raw object: '+type(value).__name__)


def _native_components(env,world,*,clock_reader,capacity_per_wire,
                       contact_delivery_probe_mode=None,contact_delivery_probe_order=0,
                       delivery_clock_reader=None,capture_filtered_contact_metadata=False):
    """Use existing stage, SimulationView and reporters; no Apply/Set/step."""
    import carb
    import omni.usd
    from .tabletop_bimanual_mpc_scene_measurements import (
        ComposedUSDScene,create_passive_contact_views,SceneMeasurementProvider)
    from .tabletop_bimanual_mpc_contact_provenance import (
        ContactPairProvenanceCollector,open_installed_contact_source)
    stage=omni.usd.get_context().get_stage()
    if stage is None:raise ValueError('actual existing composed stage required')
    sensors=world.w.sensors
    if not sensors or not type(sensors[0]).__module__.startswith('isaaclab.sensors.'):
        raise ValueError('actual existing IsaacLab contact sensor required')
    simulation_view=sensors[0]._physics_sim_view
    scene=ComposedUSDScene(stage=stage,settings=carb.settings.get_settings())
    views=create_passive_contact_views(scene=scene,simulation_view=simulation_view,
        rigid_body_view=world.wire.root_physx_view,enabled=True,
        capacity_per_wire=capacity_per_wire)
    provider=SceneMeasurementProvider(scene=scene,rigid_body_view=world.wire.root_physx_view,
        clock_reader=clock_reader,contact_views=views,enabled=True,physics_dt=world.dt)
    source=open_installed_contact_source(enabled=True,stage=stage)
    pairs=ContactPairProvenanceCollector(enabled=True,source=source)
    metadata=dict(scene_provider=_identity(Path(inspect.getfile(type(provider)))),
        contact_provider=_identity(Path(inspect.getfile(type(pairs)))),
        static_view_policy='three low-level views; existing reporter/settings unchanged',
        capacity_per_wire=capacity_per_wire,
        qualification='observational prototype; static-filter/sign/frame/completeness unqualified')
    if capture_filtered_contact_metadata:
        from .tabletop_bimanual_mpc_filtered_contact_measurements import (
            FilteredPairMetadataProvider,make_source_reader)
        filtered=FilteredPairMetadataProvider(sensors=sensors,
            wire_root_view=world.wire.root_physx_view,
            clock_reader=lambda:tuple(clock_reader(i) for i in range(env.n)),
            source_reader=make_source_reader(stage=stage,sensors=sensors,
                stage_reader=lambda:omni.usd.get_context().get_stage()),enabled=True)
        metadata['filtered_metadata_provider']=_identity(Path(inspect.getfile(type(filtered))))
        return AuxCaptureComponents(provider,pairs,metadata,filtered_metadata_provider=filtered)
    if contact_delivery_probe_mode is None:
        return provider,pairs,metadata
    from .tabletop_bimanual_mpc_contact_delivery_probe import open_installed_delivery_probe
    probe=open_installed_delivery_probe(enabled=True,source=source,
        clock_reader=delivery_clock_reader or (lambda:clock_reader(0)),mode=contact_delivery_probe_mode,
        order=contact_delivery_probe_order)
    try:
        metadata['delivery_probe_implementation']=_identity(Path(inspect.getfile(type(probe))))
    except BaseException as error:
        try:probe.close()
        except BaseException as cleanup_error:raise error from cleanup_error
        raise
    return AuxCaptureComponents(provider,pairs,metadata,probe)


class _AuxCapture:
    def __init__(self,env,world,output,*,run_uuid,native_runtime_verified=False,
                 component_factory=None,capacity_per_wire=64,
                 contact_delivery_probe_mode=None,contact_delivery_probe_order=0,
                 capture_filtered_contact_metadata=False):
        if type(capture_filtered_contact_metadata) is not bool:
            raise ValueError('filtered metadata capture requires an explicit bool')
        if capture_filtered_contact_metadata and contact_delivery_probe_mode is not None:
            raise ValueError('filtered metadata and delivery require independent observer arms')
        if contact_delivery_probe_mode not in (None,'post_step_poll','full_report_callback'):
            raise ValueError('contact delivery probe mode must select one independent observer or None')
        if type(contact_delivery_probe_order) is not int or contact_delivery_probe_order<0:
            raise ValueError('contact delivery probe order must be a nonnegative integer')
        if type(native_runtime_verified) is not bool:
            raise ValueError('native flag must be an actual boolean')
        self.advance_chain='CPU fixture; no native source-chain attestation'
        if native_runtime_verified:
            if component_factory is not None:
                raise ValueError('native attestation excludes injected fixture factories')
            _native_attestation(env,world)
            from .tabletop_tensor_world import TensorWorld
            from .tabletop_tensor_task import TensorTask
            from .tabletop_vision_sequence_capture import SequenceContextCapture
            self.advance_chain=_validated_advance_chain(env,world,world_type=TensorWorld,
                task_type=TensorTask,sequence_type=SequenceContextCapture)
        if not isinstance(run_uuid,str) or not run_uuid:
            raise ValueError('actual shared capture run UUID required')
        if not native_runtime_verified and component_factory is None:
            raise ValueError('CPU fixture requires an explicit component factory; genuine attestation required for SDK acquisition')
        if type(capacity_per_wire) is not int or capacity_per_wire<1:
            raise ValueError('positive explicit buffer capacity required')
        self.env=env;self.world=world;self.run_uuid=run_uuid;self.n=env.n
        self.initial_ids=self._ids()
        if len(self.initial_ids)!=self.n:raise ValueError('physical episode cohort size mismatch')
        self.closed=[False]*self.n;self.counts=[0]*self.n;self.terminals={}
        self._policy=None;self._action=None;self._before=None;self._pending=None;self._last_clock=None
        self._finished=False;self._summary=None;self.frames=0;self.provider_errors=[]
        self.raw_write_errors=[];self.terminal_archive_only=set()
        self.contact_delivery_probe_mode=contact_delivery_probe_mode
        self.contact_delivery_probe_order=contact_delivery_probe_order
        self.delivery_probe=None;self._delivery_summary=None;self.delivery_errors=[]
        self.capture_filtered_contact_metadata=capture_filtered_contact_metadata
        self.filtered_metadata_provider=None;self._filtered_previous=None
        self._filtered_counts=dict(snapshots=0,valid_snapshots=0,invalid_snapshots=0)
        self._filtered_archive_errors=[]
        self.native_runtime_verified=native_runtime_verified
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=False)
        self._stream=gzip.open(self.output/'aux_480hz.jsonl.gz','xt',encoding='utf-8',compresslevel=1)
        self._missing=object()
        self._old_advance=vars(world).get('advance',self._missing)
        self._original=world.advance
        self._prior=getattr(env,'joint_replay_observer',self._missing)
        self._advance_hook=self.advance;self._observer_hook=self.observe
        try:
            if component_factory is None:
                options=dict(clock_reader=self.clock_for_env,capacity_per_wire=capacity_per_wire)
                if contact_delivery_probe_mode is not None:
                    options.update(contact_delivery_probe_mode=contact_delivery_probe_mode,
                        contact_delivery_probe_order=contact_delivery_probe_order,
                        delivery_clock_reader=self._delivery_clock)
                if capture_filtered_contact_metadata:
                    options['capture_filtered_contact_metadata']=True
                components=_native_components(env,world,**options)
            else:
                # Existing CPU factories keep their exact one-keyword contract.
                components=component_factory(clock_reader=self.clock_for_env)
            if isinstance(components,AuxCaptureComponents):
                self.scene,self.pairs,self.metadata=components.scene,components.pairs,components.metadata
                self.delivery_probe=components.delivery_probe
                self.filtered_metadata_provider=components.filtered_metadata_provider
            elif isinstance(components,(tuple,list)) and len(components) in (3,4):
                self.scene,self.pairs,self.metadata=components[:3]
                if len(components)==4:self.delivery_probe=components[3]
            else:raise ValueError('three legacy components or typed optional delivery probe required')
            if contact_delivery_probe_mode is None:
                if self.delivery_probe is not None:
                    raise ValueError('delivery probe requires an explicit enabled mode')
            elif (self.delivery_probe is None or self.delivery_probe.enabled is not True
                    or self.delivery_probe.mode!=contact_delivery_probe_mode):
                raise ValueError('supplied delivery probe must match the explicit mode')
            if capture_filtered_contact_metadata:
                if self.filtered_metadata_provider is None or self.filtered_metadata_provider.enabled is not True:
                    raise ValueError('filtered metadata capture requires an enabled provider')
            elif self.filtered_metadata_provider is not None:
                raise ValueError('filtered metadata provider requires an explicit enabled capture')
            world.advance=self._advance_hook
            env.joint_replay_observer=self._observer_hook
        except Exception as error:
            try:
                self.finish(error='initialization: '+repr(error))
            except Exception as cleanup_error:
                raise error from cleanup_error
            raise

    def _ids(self):
        return tuple(_integer(v,'actual diagnostic episode ID') for v in _plain(self.env.diagnostic_episode_ids))

    def clock_for_env(self,env_id):
        c=_clock(self.world);ids=self._ids()
        done=_plain(getattr(self.env.task,'done',None))
        terminal=False if done is None else bool(done[env_id])
        return SampleClock(run_uuid=self.run_uuid,env_id=env_id,
            episode_uid=f'{self.run_uuid}:{env_id}:{ids[env_id]}',
            world_step=c['world_step'],world_time_s=c['world_time_s'],
            sdk_step=c['sdk_step'],sdk_time_s=c['sdk_time_s'],
            policy_step=0 if self._policy is None else self._policy,
            reset_generation=ids[env_id],terminal=terminal,reset=False)

    def _delivery_clock(self):
        # Read an actual candidate receipt for the current representative. This
        # selects a clock source only; no global actor pair is assigned to it.
        index=self._pending
        if index is None:index=next((i for i in range(self.n) if not self.closed[i]),0)
        return self.clock_for_env(index)

    def submit(self,issued_action24,*,policy_step):
        if self._finished:raise ValueError('auxiliary capture finished')
        step=_integer(policy_step,'policy_step')
        if self._policy is not None and step!=self._policy+1:
            raise ValueError('consecutive policy submissions required')
        if self._pending is not None:raise ValueError('physical bracket remains pending')
        rows=tuple(tuple(_number(v,'issued action') for v in row) for row in _plain(issued_action24))
        if len(rows)!=self.n or any(len(row)!=24 or any(abs(v)>1 for v in row) for row in rows):
            raise ValueError('bounded actual canonical24 action required per environment')
        before=_clock(self.world)
        if self._before is not None:
            _require_clock_delta(before,self._before,8)
            if not all(self.closed) and before!=self._last_clock:
                raise ValueError('missing final physical observer before next policy action')
        else:self._last_clock=before
        self._policy=step;self._action=rows;self._before=before

    def advance(self,*args,**kwargs):
        if self._policy is None:raise ValueError('submit policy action before auxiliary physics capture')
        if self._pending is not None:raise ValueError('previous physical bracket still pending')
        candidates=[i for i in range(self.n) if not self.closed[i]]
        try:
            if candidates:
                i=candidates[0]
                if self._ids()[i]!=self.initial_ids[i]:
                    raise ValueError('open first episode changed before terminal')
                self._pending=i
                before=self.clock_for_env(i)
                self.pairs.begin_step(before,dt_s=self.world.dt)
                if self.delivery_probe is not None:
                    self.delivery_probe.begin_step(before,dt_s=self.world.dt)
            # Exact original call: no extra step, refresh, targets or settings.
            return self._original(*args,**kwargs)
        except BaseException as error:
            try:
                if self._pending is not None:self.pairs.abort_step('original advance: '+repr(error))
            except Exception as cleanup_error:
                if self.delivery_probe is None:raise
                self.delivery_errors.append('pair abort: '+repr(cleanup_error))
            self._pending=None
            self._finish_delivery_failure(error,'original advance')
            raise

    def observe(self,env,measurement,native_live,newly_terminal):
        try:return self._observe(env,measurement,native_live,newly_terminal)
        except BaseException as error:
            self._finish_delivery_failure(error,'observer')
            raise

    def _observe(self,env,measurement,native_live,newly_terminal):
        if self._prior is not self._missing and self._prior is not None:
            self._prior(env,measurement,native_live,newly_terminal)
        actual_live=_mask(_plain(native_live),self.n,'actual_live')
        actual_terminal=_mask(_plain(newly_terminal),self.n,'actual_terminal')
        active=tuple(actual_live[i] and not self.closed[i] for i in range(self.n))
        terminals=tuple(actual_terminal[i] and active[i] for i in range(self.n))
        if not any(active):
            if self._pending is not None:
                self.pairs.abort_step('no active first-cohort physical sample')
                if self.delivery_probe is not None:
                    self.delivery_probe.end_step(self.clock_for_env(self._pending),
                        step_completed=False,poll_report=None)
            self._pending=None
            return
        if self._pending is None:raise ValueError('missing or duplicate physical bracket')
        ids=self._ids()
        if any(ids[i]!=self.initial_ids[i] for i in range(self.n) if active[i]):
            raise ValueError('first physical episode changed without terminal')
        clocks=tuple(self.clock_for_env(i) for i in range(self.n))
        if any(clocks[i].terminal!=terminals[i] for i in range(self.n) if active[i]):
            raise ValueError('actual task terminal differs from pre-reset callback')
        c=_clock(self.world)
        _require_clock_delta(c,self._last_clock,1)
        for clock in clocks:
            if any(getattr(clock,key)!=c[key] for key in c):
                raise ValueError('environment clock changed while reading auxiliary snapshot')
        delta=c['world_step']-self._before['world_step']
        if not 1<=delta<=8:
            raise ValueError('auxiliary physical clock is outside its submitted action')
        _require_clock_delta(c,self._before,delta)
        index=self._pending
        packet=dict(version=VERSION,run_uuid=self.run_uuid,clock=c,clocks=_serial(clocks),
            policy_step=self._policy,control_substep=delta,
            actual_live=list(actual_live),actual_terminal=list(actual_terminal),
            cohort_live=list(active),first_terminal=list(terminals),
            issued_action24=_serial(self._action),policy_clock_before=dict(self._before),
            global_contact_report_scope='all reported actors; representative clock does not assign pairs to its env',
            representative_contact_clock_env_id=index,
            native_success_qualification_issued=False,new_expert_qualification_issued=False,
            native_runtime_attestation_scope='runner only; not component or support qualification',
            source_role='legacy Teacher auxiliary observation; not Student input or control')
        pair=None;samples=None;delivery=None;filtered=None;filtered_type=None
        try:
            pair=self.pairs.end_step(clocks[index],poll=True,
                read_clock=lambda:self.clock_for_env(index),step_completed=True)
            self._pending=None
            if self.delivery_probe is not None:
                delivery=self.delivery_probe.end_step(clocks[index],step_completed=True,poll_report=pair)
            if self.filtered_metadata_provider is not None:
                from .tabletop_bimanual_mpc_filtered_contact_provenance import FilteredPairSnapshot
                filtered_type=FilteredPairSnapshot
                filtered=self.filtered_metadata_provider.capture(clocks=clocks,previous=self._filtered_previous)
                if (type(filtered) is not FilteredPairSnapshot or filtered.clocks!=clocks
                        or any(value is not False for value in filtered.qualification.values())):
                    raise ValueError('filtered metadata provider returned a different actual clock or qualification')
                if filtered.metadata_binding_valid:
                    if len(filtered.pairs)!=34*self.n:
                        raise ValueError('filtered metadata pair count is incomplete')
                    self._filtered_previous=filtered
            samples=self.scene.capture_batch(clocks=clocks,active=active)
            if len(samples)!=self.n or any(sample.clock!=clock for sample,clock in zip(samples,clocks)):
                raise ValueError('scene provider returned a different actual episode/clock')
            if _clock(self.world)!=c:
                raise ValueError('physical clock changed during auxiliary provider readback')
        except Exception as error:
            # A copied contact receipt and real terminal must survive a later
            # provider failure. This partial frame is explicitly unusable for
            # planning and the original error still aborts the diagnostic run.
            message=repr(error);self.provider_errors.append(message)
            packet.update(capture_status='provider_error',provider_error=message,
                scene_samples=None,provider_partial_scene_raw=_serial(samples),
                contact_report=_serial(pair),complete_measurement=False)
            if self.delivery_probe is not None:packet['contact_delivery_step']=_serial(delivery)
            if self.capture_filtered_contact_metadata:
                packet.update(filtered_pair_metadata=None,
                    provider_partial_filtered_metadata=filtered.to_dict() if filtered_type is not None and type(filtered) is filtered_type else None)
            try:self._record(packet,active,terminals,c)
            except Exception as write_error:
                if self.delivery_probe is None and not self.capture_filtered_contact_metadata:raise
                if self.delivery_probe is not None:
                    self.delivery_errors.append('partial frame write: '+repr(write_error))
                if self.capture_filtered_contact_metadata:
                    self._filtered_archive_errors.append('partial frame write: '+repr(write_error))
            self._finish_delivery_failure(error,'provider')
            raise
        packet.update(capture_status='complete_observation_only',provider_error=None,
            scene_samples=_serial(samples),contact_report=_serial(pair),
            complete_measurement=True)
        if self.delivery_probe is not None:packet['contact_delivery_step']=_serial(delivery)
        if self.capture_filtered_contact_metadata:packet['filtered_pair_metadata']=filtered.to_dict()
        try:self._record(packet,active,terminals,c)
        except BaseException as error:
            self._finish_delivery_failure(error,'raw write')
            raise

    def _finish_delivery_failure(self,error,scope):
        """Close owned observers on failure, without replacing the causal error."""
        if self.delivery_probe is None:return
        try:self.finish(error=scope+': '+repr(error))
        except BaseException as cleanup_error:
            self.delivery_errors.append('failure cleanup: '+repr(cleanup_error))

    def _record(self,packet,active,terminals,clock):
        packet=_raw_json_payload(packet)
        encoded=json.dumps(packet,allow_nan=False,separators=(',',':'))
        # First terminal has its own archive even when the streaming sink fails.
        # A pinned receipt is not evidence that the raw frame was written.
        terminal_ids=[i for i in range(self.n) if terminals[i]]
        for i in terminal_ids:
            self.terminals[str(i)]=packet;self.closed[i]=True
        try:self._stream.write(encoded+'\n')
        except Exception as error:
            self.raw_write_errors.append(repr(error))
            self.terminal_archive_only.update(terminal_ids)
            raise
        self.frames+=1;self._last_clock=clock
        if self.capture_filtered_contact_metadata and packet.get('filtered_pair_metadata') is not None:
            self._filtered_counts['snapshots']+=1
            key='valid_snapshots' if packet['filtered_pair_metadata']['metadata_binding_valid'] else 'invalid_snapshots'
            self._filtered_counts[key]+=1
        for i in range(self.n):
            if active[i]:self.counts[i]+=1

    def _restore(self):
        if getattr(self.world,'advance',None) is self._advance_hook:
            if self._old_advance is self._missing:delattr(self.world,'advance')
            else:self.world.advance=self._old_advance
        if getattr(self.env,'joint_replay_observer',None) is self._observer_hook:
            if self._prior is self._missing:delattr(self.env,'joint_replay_observer')
            else:self.env.joint_replay_observer=self._prior

    def _write_summary(self,error):
        with (self.output/'terminal_snapshots.json').open('x',encoding='utf-8') as stream:
            json.dump(self.terminals,stream,allow_nan=False)
        summary=dict(version=VERSION,run_uuid=self.run_uuid,physical_frames=self.frames,
            records_per_env=list(self.counts),terminal_count=len(self.terminals),
            native_runtime_verified=self.native_runtime_verified,error=error or
                ('; '.join(self.provider_errors+self.raw_write_errors) or None),
            provider_error_frames=len(self.provider_errors),provider_errors=list(self.provider_errors),
            raw_write_errors=list(self.raw_write_errors),
            terminal_archive_only_env_ids=sorted(self.terminal_archive_only),
            scene_qualification=False,contact_pair_qualification=False,support_qualification=False,
            native_success_qualification_issued=False,new_expert_qualification_issued=False,
            Student_success_count=0,ppo_updates=0,raw=_identity(self.output/'aux_480hz.jsonl.gz'),
            reset_generation_source='actual diagnostic_episode_ids alias; not an independent counter',
            world_time_source='World step times original dt; derived local elapsed time',
            sdk_time_source='actual SimulationContext.current_time with distinct origin',
            terminals=_identity(self.output/'terminal_snapshots.json'),
            components=getattr(self,'metadata',None),implementation=_identity(Path(__file__)))
        summary['advance_source_chain']=self.advance_chain
        if self.contact_delivery_probe_mode is not None or self.delivery_probe is not None:
            summary['contact_delivery_probe']=self._delivery_summary
        if self.capture_filtered_contact_metadata:
            summary['filtered_contact_metadata']=dict(self._filtered_counts,
                archive_errors=list(self._filtered_archive_errors),
                scope='fresh resolved sensor/filter metadata only; no numeric contacts or force qualification',
                native_success_qualification_issued=False,new_expert_qualification_issued=False,
                planning_eligible=False,support_qualified=False,contact_values_qualified=False)
        with (self.output/'aux_capture_summary.json').open('x',encoding='utf-8') as stream:
            json.dump(summary,stream,indent=2,allow_nan=False)
        self._summary=summary
        return summary

    def _flush_delivery(self):
        """Called only after probe close. Independent records include late/orphans.

        These archives have global scope. Representative clocks remain candidate
        receipts and never assign a global report to an environment or frame.
        Each sink is attempted independently so a failed callback archive cannot
        skip step diagnostics, terminal snapshots, raw close, or error summary.
        """
        probe=self.delivery_probe;errors=[]
        summary=dict(version=DELIVERY_VERSION,run_uuid=self.run_uuid,
            mode=self.contact_delivery_probe_mode,order=self.contact_delivery_probe_order,
            callback_record_count=None,step_diagnostic_count=None,
            callback_records_written=0,step_diagnostics_written=0,
            callback_raw=None,step_raw=None,errors=list(self.delivery_errors),
            global_scope='global reported actors; representative clocks are candidates, not env/frame joins',
            clock_association=None,pair_env_association=None,native_attested=False,
            planning_eligible=False,native_force_qualified=False,support_qualified=False,
            expert_qualified=False,native_success_qualification_issued=False,
            new_expert_qualification_issued=False)
        for attribute,filename,count_key,written_key,identity_key,record_key in (
                ('records','contact_delivery_callbacks.jsonl.gz','callback_record_count',
                 'callback_records_written','callback_raw','receipt'),
                ('step_diagnostics','contact_delivery_steps.jsonl.gz','step_diagnostic_count',
                 'step_diagnostics_written','step_raw','diagnostic')):
            path=self.output/filename
            try:
                records=() if probe is None else tuple(getattr(probe,attribute))
                summary[count_key]=len(records)
                with gzip.open(path,'xt',encoding='utf-8',compresslevel=1) as stream:
                    for record in records:
                        packet=dict(version=DELIVERY_VERSION,run_uuid=self.run_uuid,
                            global_scope=summary['global_scope'],
                            native_success_qualification_issued=False,
                            new_expert_qualification_issued=False)
                        packet[record_key]=_serial(record)
                        stream.write(json.dumps(_raw_json_payload(packet),allow_nan=False,
                            separators=(',',':'))+'\n')
                        summary[written_key]+=1
            except Exception as exc:
                errors.append(exc);summary['errors'].append(filename+': '+repr(exc))
            try:
                if path.is_file():summary[identity_key]=_identity(path)
            except Exception as exc:
                errors.append(exc);summary['errors'].append(filename+' identity: '+repr(exc))
        summary['subscription_active_after_close']=False if probe is None else probe.subscription_active
        if probe is not None:
            summary['probe_cleanup_errors']=list(probe.cleanup_errors)
            summary['probe_installation_errors']=list(probe.installation_errors)
        summary['archive_complete']=not errors and not summary['errors'] and not summary['subscription_active_after_close']
        self._delivery_summary=summary
        path=self.output/'contact_delivery_summary.json'
        try:
            with path.open('x',encoding='utf-8') as stream:
                json.dump(summary,stream,indent=2,allow_nan=False)
            # The main summary carries this identity; no recursive self hash.
            self._delivery_summary=dict(summary,summary=_identity(path))
        except Exception as exc:
            errors.append(exc);summary['errors'].append('delivery summary: '+repr(exc))
        return errors

    def finish(self,*,error=None):
        if self._finished:
            # Fail-fast finalization may run under an outer Core/Model hook.
            # Retry only ownership-checked restoration after that owner unwinds;
            # the already-closed resources and exclusive archives stay intact.
            self._restore()
            return self._summary
        errors=[]
        try:self._restore()
        except Exception as exc:errors.append(exc)
        if self._pending is not None:
            try:self.pairs.abort_step('capture finish with incomplete physics bracket')
            except Exception as exc:errors.append(exc)
            finally:self._pending=None
            error=error or 'incomplete final physical bracket'
        # Stop SDK callbacks before snapshotting any independent callback list
        # or opening its archive. close itself releases even if archiving its
        # unfinished bracket fails; later cleanup must still run.
        if self.delivery_probe is not None:
            try:self.delivery_probe.close()
            except Exception as exc:
                errors.append(exc);self.delivery_errors.append('probe close: '+repr(exc))
        if self.contact_delivery_probe_mode is not None or self.delivery_probe is not None:
            try:errors.extend(self._flush_delivery())
            except Exception as exc:
                errors.append(exc);self.delivery_errors.append('delivery flush: '+repr(exc))
        try:self._stream.close()
        except Exception as exc:errors.append(exc)
        messages=([str(error)] if error is not None else [])+[repr(exc) for exc in errors]
        # Exclusive archives are attempted only once even when finalization
        # fails. Cleanup errors invalidate evidence and never skip raw close.
        self._finished=True
        try:result=self._write_summary('; '.join(messages) if messages else None)
        except Exception as summary_error:
            if errors:raise errors[0] from summary_error
            raise
        if errors:raise errors[0]
        return result


def install_mpc_aux_capture(env,world,output,*,enabled=False,**options):
    if type(enabled) is not bool:raise ValueError('explicit bool enabled required')
    if not enabled:return None
    return _AuxCapture(env,world,output,**options)
