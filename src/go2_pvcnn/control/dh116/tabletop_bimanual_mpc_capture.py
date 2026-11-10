"""Exclusive Teacher-state archives; no control, simulation stepping or Actor input.

An archive verifies evidence identities, not the truth of a sensor or task
success. Native attestation belongs to the opt-in runner. Each live physical
sample and its actual submitted canonical24 action are retained before reset.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import inspect
import copy
import struct
from numbers import Real
from pathlib import Path
import uuid

from .tabletop_bimanual_mpc_state import MPCState, validate_transition


VERSION = 'dh116-bimanual-mpc-state-capture-v1'


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name+' must be a nonnegative integer')
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(name+' must be finite numeric data')
    return float(value)


def _mask(values, n, name):
    result = tuple(values)
    if len(result) != n or any(type(v) is not bool for v in result):
        raise ValueError(name+' requires one actual boolean per environment')
    return result


def _identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return dict(path=str(path.resolve()),sha256=digest.hexdigest())


class MPCStateArchive:
    """Archive a single cohort's first physical episodes without reset reuse."""
    def __init__(self, output, *, n, run_uuid, episode_uids, physics_hz=480,
                 policy_hz=60, native_runtime_verified=False,
                 action_owner='legacy_teacher_mpc'):
        if type(n) is not int or n < 1:
            raise ValueError('positive environment count required')
        if (physics_hz,policy_hz)!=(480,60):
            raise ValueError('original480Hz/60Hz/eight-substep contract required')
        if not isinstance(run_uuid,str) or not run_uuid:
            raise ValueError('actual capture run_uuid required')
        uids=tuple(episode_uids)
        if (len(uids)!=n or len(set(uids))!=n
                or any(not isinstance(v,str) or not v for v in uids)):
            raise ValueError('one unique actual episode identity per environment required')
        if type(native_runtime_verified) is not bool:
            raise ValueError('runner native flag must be boolean')
        if action_owner not in ('legacy_teacher_mpc','legacy_teacher_tape'):
            raise ValueError('this privileged archive cannot count a Student controller')
        self.output=Path(output).resolve()
        self.output.mkdir(parents=True,exist_ok=False)
        self.n=n;self.run_uuid=run_uuid;self.episode_uids=uids
        self.native_runtime_verified=native_runtime_verified;self.action_owner=action_owner
        self._stream=gzip.open(self.output/'states_480hz.jsonl.gz','xt',encoding='utf-8',compresslevel=1)
        self._closed=[False]*n;self._last=[None]*n;self._terminals={}
        self._records=0;self._counts=[0]*n;self._unavailable={}
        self._policy=None;self._before=None;self._action=None;self._next_world=None
        self._finished=False;self._report=None

    @property
    def closed(self):
        return tuple(self._closed)

    def submit(self, issued_action24, *, policy_step, clock_before):
        if self._finished:
            raise ValueError('archive is finished')
        step=_integer(policy_step,'policy_step')
        if self._policy is not None and step!=self._policy+1:
            raise ValueError('actual policy submissions must be consecutive')
        if not isinstance(clock_before,dict):
            raise ValueError('actual pre-submit clock dictionary required')
        before={name:_integer(clock_before[name],name) for name in ('world_step','sdk_step')}
        before.update({name:_number(clock_before[name],name) for name in ('world_time_s','sdk_time_s')})
        if any(value<0 for value in before.values()):
            raise ValueError('pre-submit clocks must be nonnegative')
        if self._policy is not None:
            if (before['world_step']-self._before['world_step']!=8
                    or before['sdk_step']-self._before['sdk_step']!=8
                    or self._next_world!=before['world_step']+1):
                raise ValueError('each60Hz policy interval requires all eight480Hz physical steps')
            for name in ('world_time_s','sdk_time_s'):
                if not math.isclose(before[name]-self._before[name],8/480,abs_tol=1e-7,rel_tol=1e-7):
                    raise ValueError('eight physical steps must retain the original60Hz policy period')
        rows=tuple(tuple(_number(v,'issued action') for v in row) for row in issued_action24)
        if len(rows)!=self.n or any(len(row)!=24 for row in rows):
            raise ValueError('actual submitted action must be [N,24]')
        if any(abs(v)>1 for row in rows for v in row):
            raise ValueError('actual action must retain bounded canonical24')
        if self._next_world is not None and before['world_step']+1!=self._next_world:
            raise ValueError('policy boundary must follow the recorded physical steps')
        self._policy=step;self._before=before;self._action=rows
        self._next_world=before['world_step']+1

    def record_batch(self, states, *, live, terminal):
        if self._finished:
            raise ValueError('archive is finished')
        if self._policy is None:
            raise ValueError('submit actual policy action before physical samples')
        states=tuple(states);live=_mask(live,self.n,'live');terminal=_mask(terminal,self.n,'terminal')
        if len(states)!=self.n:
            raise ValueError('one state slot per environment required')
        chosen=[i for i in range(self.n) if live[i] and not self._closed[i]]
        if any(terminal[i] and not live[i] and not self._closed[i] for i in range(self.n)):
            raise ValueError('new terminal must be an actual live physical sample')
        if not chosen:
            return 0
        packets=[];frame=None
        for i in chosen:
            state=states[i]
            if not isinstance(state,MPCState):
                raise ValueError('actual immutable MPCState required')
            c=state.clock
            if c.run_uuid!=self.run_uuid or c.env_id!=i or c.episode_uid!=self.episode_uids[i]:
                raise ValueError('actual capture environment/episode identity differs')
            if c.reset:
                raise ValueError('reset boundary cannot be merged into the first physical episode')
            if c.policy_step!=self._policy:
                raise ValueError('state policy_step differs from actual submitted action')
            if c.terminal!=terminal[i]:
                raise ValueError('state terminal flag differs from actual pre-reset terminal')
            delta=c.world_step-self._before['world_step']
            if (c.world_step!=self._next_world or not 1<=delta<=8
                    or c.sdk_step-self._before['sdk_step']!=delta
                    or c.world_time_s<=self._before['world_time_s']
                    or c.sdk_time_s<=self._before['sdk_time_s']):
                raise ValueError('consecutive World/SDK physical step clocks required')
            for current,name in ((c.world_time_s,'world_time_s'),(c.sdk_time_s,'sdk_time_s')):
                if not math.isclose(current-self._before[name],delta/480,abs_tol=1e-7,rel_tol=1e-7):
                    raise ValueError('physical timestamps must match the original480Hz substep duration')
            shared=(c.world_step,c.world_time_s,c.sdk_step,c.sdk_time_s,c.policy_step)
            if frame is not None and frame!=shared:
                raise ValueError('one batch must describe one actual World/SDK clock')
            frame=shared
            if self._last[i] is not None:
                validate_transition(self._last[i],state)
                if (c.world_step!=self._last[i].clock.world_step+1
                        or c.sdk_step!=self._last[i].clock.sdk_step+1):
                    raise ValueError('missing or duplicated live physical steps')
            packet=dict(version=VERSION,run_uuid=self.run_uuid,env_id=i,
                episode_uid=c.episode_uid,state=state.to_dict(),
                action_owner=self.action_owner,issued_action24=list(self._action[i]),
                policy_clock_before=dict(self._before),
                counts_as_Student_success=False)
            # Serialize and validate the whole batch before writing any row.
            encoded=json.dumps(packet,allow_nan=False,separators=(',',':'))
            packets.append((i,state,packet,encoded))
        for i,state,packet,encoded in packets:
            self._stream.write(encoded+'\n')
            self._last[i]=state;self._records+=1;self._counts[i]+=1
            for name,field in state.fields.items():
                if not field.valid:self._unavailable[name]=self._unavailable.get(name,0)+1
            if terminal[i]:
                self._terminals[str(i)]=packet;self._closed[i]=True
        self._next_world=frame[0]+1
        return len(packets)

    def finish(self, *, error=None):
        if self._finished:
            return self._report
        self._stream.close()
        terminal_path=self.output/'terminal_snapshots.json'
        with terminal_path.open('x',encoding='utf-8') as stream:
            json.dump(self._terminals,stream,indent=2,allow_nan=False);stream.write('\n')
        self._report=dict(version=VERSION,run_uuid=self.run_uuid,record_count=self._records,
            records_per_env=self._counts,terminal_count=len(self._terminals),
            closed_first_episodes=list(self._closed),unavailable_field_samples=self._unavailable,
            physics_hz=480,policy_hz=60,action_size=24,action_owner=self.action_owner,
            native_runtime_verified=self.native_runtime_verified,
            native_attestation_source='caller_runner;not_proved_by_archive_format',
            native_success_qualification_issued=False,new_expert_qualification_issued=False,
            Student_success_count=0,ppo_updates=0,error=error,
            states=_identity(self.output/'states_480hz.jsonl.gz'),terminals=_identity(terminal_path))
        with (self.output/'capture_summary.json').open('x',encoding='utf-8') as stream:
            json.dump(self._report,stream,indent=2,allow_nan=False);stream.write('\n')
        self._finished=True
        return self._report


def _plain(value):
    if hasattr(value,'detach'):
        value=value.detach()
    if hasattr(value,'cpu'):
        value=value.cpu()
    if hasattr(value,'tolist'):
        value=value.tolist()
    return value


def _raw_json_payload(raw):
    """Keep invalid raw numbers recoverable without relaxing the strict DTO.

    Null placeholders have exact tree paths and captured Python-float64 bit
    representations. They never become zero or valid sensor measurements.
    """
    replacements=[]
    def visit(value,path):
        if isinstance(value,Real) and not isinstance(value,bool) and not math.isfinite(value):
            number=float(value)
            kind='nan' if math.isnan(number) else ('positive_infinity' if number>0 else 'negative_infinity')
            replacements.append(dict(path=list(path),kind=kind,
                float64_be_hex=struct.pack('>d',number).hex()))
            return None
        if isinstance(value,dict):
            return {key:visit(item,(*path,key)) for key,item in value.items()}
        if isinstance(value,(list,tuple)):
            return [visit(item,(*path,i)) for i,item in enumerate(value)]
        return value
    result=visit(raw,())
    result['raw_number_encoding']=dict(version='dh116-raw-nonfinite-path-and-float64-bits-v1',
        nonfinite_float_values=replacements,
        scope='captured Python numeric representation;not an independent original sensor-buffer bit dump',
        missing_or_nonfinite_values_are_valid_measurements=False)
    return result


def _clock(world):
    step=_integer(int(world._physics_steps),'world_step')
    dt=_number(world.dt,'original physics dt')
    if dt!=1/480:
        raise ValueError('original480Hz physics dt required for capture')
    sim=world.w.sim
    return dict(world_step=step,world_time_s=step*dt,
                sdk_step=_integer(int(sim.current_time_step_index),'sdk_step'),
                sdk_time_s=_number(float(sim.current_time),'sdk_time_s'))


def _sensor_sources(world):
    """Record getter results, without claiming they are resolved partner IDs."""
    sensors=getattr(world.w,'sensors',None)
    if sensors is None:
        return dict(available=False,reason='sensor collection unavailable',sensors=[],
                    per_contact_partner_id_recorded=False)
    rows=[]
    for index,sensor in enumerate(sensors):
        row=dict(sensor_index=index,source=type(sensor).__module__+'.'+type(sensor).__name__)
        for name,view_name,attribute in (
                ('sensor_paths','contact_physx_view','sensor_paths'),
                ('body_prim_paths','body_physx_view','prim_paths'),
                ('filter_count','contact_physx_view','filter_count'),
                ('configured_filter_expressions','cfg','filter_prim_paths_expr')):
            try:
                obj=getattr(sensor,view_name,None)
                value=_plain(getattr(obj,attribute))
                json.dumps(value,allow_nan=False)
                row[name]=value
            except (AttributeError,TypeError,ValueError,RuntimeError) as error:
                row[name]=None;row[name+'_error']=type(error).__name__+': '+str(error)
        try:
            value=_plain(sensor.contact_physx_view.filter_paths)
            json.dumps(value,allow_nan=False)
            row['filter_paths']=dict(available=value is not None,value=value,
                source='actual contact_physx_view.filter_paths getter',resolved_semantics='unknown')
        except (AttributeError,TypeError,ValueError,RuntimeError) as error:
            row['filter_paths']=dict(available=False,value=None,error=type(error).__name__+': '+str(error),
                source='contact_physx_view.filter_paths unavailable',resolved_semantics='unknown')
        rows.append(row)
    return dict(available=True,sensors=rows,per_contact_partner_id_recorded=False,
                scope='configured and actual view path provenance;not per-contact partner-body readback')


def _native_attestation(env,world):
    # Fast rejection also keeps CPU fixtures from importing the Isaac runtime.
    if (type(world).__name__!='TensorWorld' or type(env).__name__!='TensorEnv'
            or not type(world).__module__.endswith('.tabletop_tensor_world')
            or not type(env).__module__.endswith('.tabletop_tensor_env')):
        raise ValueError('native fixture cannot assert genuine Isaac attestation')
    from .tabletop_tensor_world import TensorWorld
    from .tabletop_tensor_env import TensorEnv
    if (type(world) is not TensorWorld or type(env) is not TensorEnv
            or not type(world.w).__module__.endswith('tabletop_gpu_isaac')
            or not type(world.w.sim).__module__.startswith('isaaclab.')):
        raise ValueError('genuine Isaac TensorWorld/TensorEnv attestation required')
    if getattr(world.refresh,'__func__',None) is not TensorWorld.refresh:
        raise ValueError('native readback receipt requires the original TensorWorld.refresh method')


class _RefreshReadback:
    """Observe the return of each original refresh; never request a refresh.

    Reset may refresh every SDK DATA cache without advancing physics. Recording
    the actual method call avoids equating DATA's generation with World time.
    A receipt is provenance, not a caller-supplied assertion of freshness.
    """
    def __init__(self,world):
        self.world=world;self._original=world.refresh
        self._missing=object();self._old_attribute=vars(world).get('refresh',self._missing)
        self._receipt=None;self._sequence=0;self._wrapper=self.observe
        try:
            self.method_identity=_identity(Path(inspect.getfile(self._original)))
            self.method_sha256=hashlib.sha256(inspect.getsource(self._original).encode('utf-8')).hexdigest()
        except (TypeError,OSError):
            self.method_identity=None
            self.method_sha256=None
        world.refresh=self._wrapper

    def _data_generations(self):
        items=[(f'hands[{i}]',hand.data) for i,hand in enumerate(self.world.hands)]
        items.append(('wire',self.world.wire.data))
        return {name:_plain(getattr(data,'_sim_timestamp',None)) for name,data in items}

    def observe(self,*args,**kwargs):
        self._receipt=None
        before=_clock(self.world);generations_before=self._data_generations()
        result=self._original(*args,**kwargs)
        after=_clock(self.world);generations_after=self._data_generations()
        self._sequence+=1
        self._receipt=dict(sequence=self._sequence,
            world_sdk_clock_before=[before[k] for k in ('world_step','sdk_step','sdk_time_s')],
            world_sdk_clock_after=[after[k] for k in ('world_step','sdk_step','sdk_time_s')],
            data_generation_before=generations_before,data_generation_after=generations_after,
            dt=float(self.world.dt),original_method_sha256=self.method_sha256,
            original_method_file=self.method_identity,
            source_scope='actual original TensorWorld.refresh returned;SDK_DATA generation only',
            invocation_args=_plain(args),invocation_kwargs=_plain(kwargs))
        return result

    def receipt(self):
        return copy.deepcopy(self._receipt)

    def close(self):
        if getattr(self.world,'refresh',None) is not self._wrapper:
            return
        if self._old_attribute is self._missing:
            delattr(self.world,'refresh')
        else:
            self.world.refresh=self._old_attribute


class _ObserverCapture:
    """Chain the existing pre-reset observer; never step or control the scene."""
    def __init__(self,env,world,output,*,run_uuid=None,native_runtime_verified=False,
                 bridge_factory=None,source_models=None):
        if native_runtime_verified:
            _native_attestation(env,world)
        if bridge_factory is None:
            from .tabletop_bimanual_mpc_measurements import BimanualMPCMeasurementBridge
            bridge_factory=BimanualMPCMeasurementBridge
        elif native_runtime_verified:
            from .tabletop_bimanual_mpc_measurements import BimanualMPCMeasurementBridge
            if bridge_factory is not BimanualMPCMeasurementBridge:
                raise ValueError('native attestation excludes an injected fixture producer')
        self.env=env;self.world=world;self.run_uuid=run_uuid or str(uuid.uuid4())
        self.ids=tuple(_integer(v,'actual episode ID') for v in _plain(env.diagnostic_episode_ids))
        self.n=env.n
        if len(self.ids)!=self.n:
            raise ValueError('actual episode cohort size mismatch')
        self.archive=MPCStateArchive(output,n=self.n,run_uuid=self.run_uuid,
            episode_uids=tuple(f'{self.run_uuid}:{i}:{ep}' for i,ep in enumerate(self.ids)),
            native_runtime_verified=native_runtime_verified)
        self._missing=object()
        self._prior=getattr(env,'joint_replay_observer',self._missing)
        self._old_raw=getattr(world,'capture_tripod_contact_evidence',self._missing)
        self._observer=self.observe;self._finished=False;self._report=None
        self._raw=None;self._refresh=None;self.source_models=source_models
        try:
            self._raw=gzip.open(self.archive.output/'sensor_evidence_480hz.jsonl.gz','xt',encoding='utf-8',compresslevel=1)
            self.sources=_sensor_sources(world)
            world.capture_tripod_contact_evidence=True
            options=dict(run_uuid=self.run_uuid)
            if source_models is not None:options['source_models']=source_models
            if native_runtime_verified:
                self._refresh=_RefreshReadback(world)
                options['refresh_receipt_provider']=self._refresh.receipt
            self.bridge=bridge_factory(world,**options)
            env.joint_replay_observer=self._observer
        except Exception as error:
            self._restore()
            try:
                if self._raw is not None:self._raw.close()
            finally:
                self.archive.finish(error='initialization: '+repr(error))
            raise

    def _restore(self):
        if self._refresh is not None:self._refresh.close()
        if self._prior is self._missing:
            if getattr(self.env,'joint_replay_observer',None) is self._observer:
                delattr(self.env,'joint_replay_observer')
        else:
            current=getattr(self.env,'joint_replay_observer',self._prior)
            if current is self._observer:self.env.joint_replay_observer=self._prior
        if self._old_raw is self._missing:
            if hasattr(self.world,'capture_tripod_contact_evidence'):
                delattr(self.world,'capture_tripod_contact_evidence')
        else:self.world.capture_tripod_contact_evidence=self._old_raw

    def submit(self,issued_action24,*,policy_step):
        self.archive.submit(_plain(issued_action24),policy_step=policy_step,clock_before=_clock(self.world))

    def observe(self,env,measurement,native_live,newly_terminal):
        if self._prior is not self._missing and self._prior is not None:
            self._prior(env,measurement,native_live,newly_terminal)
        if self._finished:
            return
        actual_live=_mask(_plain(native_live),self.n,'live')
        actual_terminal=_mask(_plain(newly_terminal),self.n,'terminal')
        live=tuple(actual_live[i] and not self.archive.closed[i] for i in range(self.n))
        terminal=tuple(actual_terminal[i] and live[i] for i in range(self.n))
        if not any(live):
            return
        if self.archive._policy is None:
            raise ValueError('submit the actual Teacher action before state capture')
        ids=tuple(_integer(v,'actual episode ID') for v in _plain(env.diagnostic_episode_ids))
        uids=tuple(f'{self.run_uuid}:{i}:{ep}' for i,ep in enumerate(ids))
        clock=_clock(self.world)
        task=getattr(env,'task',None)
        failure=_plain(getattr(task,'failure',None));success=_plain(getattr(task,'success',None))
        states=self.bridge.capture(episode_uids=uids,episode_ids=ids,reset_generations=ids,
            policy_step=self.archive._policy,
            control_substep=clock['world_step']-self.archive._before['world_step'],
            live=live,terminal=terminal,
            fatal=None if failure is None else [bool(v) for v in failure],
            native_success=None if success is None else [bool(v) for v in success])
        raw=dict(version=VERSION,run_uuid=self.run_uuid,clock=clock,policy_step=self.archive._policy,
            episode_ids=list(ids),episode_uids=list(uids),live=list(live),terminal=list(terminal),
            actual_live=list(actual_live),actual_terminal=list(actual_terminal),
            reset_generation_source='actual TensorEnv.diagnostic_episode_ids;not an independent counter',
            world_time_source='TensorWorld physical step times original dt;derived local elapsed time',
            sdk_time_source='actual SimulationContext.current_time',
            producer_evidence=self.bridge.last_evidence)
        encoded=json.dumps(_raw_json_payload(raw),allow_nan=False,separators=(',',':'))
        self.archive.record_batch(states,live=live,terminal=terminal)
        self._raw.write(encoded+'\n')

    def finish(self,*,error=None):
        if self._finished:
            return self._report
        self._restore()
        raw_error=None
        try:
            self._raw.close()
        except Exception as exc:
            raw_error=exc
        core=self.archive.finish(error=error if raw_error is None else repr(raw_error))
        if raw_error is not None:
            raise raw_error
        self._report=dict(core,raw_evidence=_identity(self.archive.output/'sensor_evidence_480hz.jsonl.gz'),
            archive_summary=_identity(self.archive.output/'capture_summary.json'),sensor_sources=self.sources,
            source_models=self.source_models,
            capture_implementation=_identity(Path(__file__)),
            producer_implementation=_identity(Path(inspect.getfile(type(self.bridge)))),
            no_Physics_step_or_control_write_from_observer=True,
            reset_generation_scope='actual per-env diagnostic episode ID alias, not an independent hardware counter')
        with (self.archive.output/'native_state_capture_report.json').open('x',encoding='utf-8') as stream:
            json.dump(self._report,stream,indent=2,allow_nan=False);stream.write('\n')
        self._finished=True
        return self._report


def install_mpc_state_capture(env,world,output,*,enabled=False,**options):
    """The default path must not inspect the environment or touch the backend."""
    if not enabled:
        return None
    return _ObserverCapture(env,world,output,**options)


def validate_mpc_state_capture_options(args):
    """Require the independently frozen legacy Teacher measurement route.

    This gate imports no simulator and grants no native-success qualification.
    Existing Teacher collection validation still checks budgets and exclusions.
    """
    if not getattr(args,'vision_mpc_state_capture',False):
        return
    required=('vision_teacher_source','vision_teacher_mpc','vision_sequence_context_sidecar',
              'vision_student','native_sensor_batch','surface_force_sharing')
    if any(not getattr(args,name,None) for name in required):
        raise ValueError('MPC state capture requires explicit legacy Teacher/MPC, sidecar and native contact sensors')
    expected=dict(vision_teacher_mpc_profile='pressure_geometry_constrained',
        vision_camera_mode='fixed',vision_width=224,vision_height=96,
        vision_action_distribution='tanh_gaussian',physics_hz=480,object_kind='solid_cylinder')
    if any(getattr(args,name,None)!=value for name,value in expected.items()):
        raise ValueError('MPC state capture requires the original constrained Teacher, fixed224x96, canonical tanh action and480Hz rigid-cylinder route')
    if getattr(args,'vision_mirror_preprocessing','m0_raw')!='m0_raw':
        raise ValueError('MPC state capture baseline must preserve M0')
    wrappers=('bimanual_teacher_physics_audit','bimanual_tripod_teacher_coordination_proof',
              'bimanual_common_pause','bimanual_explicit_sdk_adapter','vision_stage6_native_config',
              'vision_tracking_experiment_config','vision_evaluation_checkpoint','resume')
    if any(getattr(args,name,None) for name in wrappers):
        raise ValueError('MPC state capture baseline excludes alternate control wrappers, evaluation and PPO resume')
