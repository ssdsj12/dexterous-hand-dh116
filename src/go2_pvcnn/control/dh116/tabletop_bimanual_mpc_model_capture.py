"""Opt-in pre-reset full-articulation observation archive, never MPC control.

Only the existing joint_replay_observer is wrapped. Callback identity and actual
clocks/terminals are validated and terminal metadata pinned before calling the
prior observer. The prior still runs before any model getter at that same
original 480 Hz clock. No advance/control/refresh/target/settings method is
replaced or called. Native runner identity never qualifies dynamics or motors.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import inspect
import json
from pathlib import Path

from .tabletop_bimanual_mpc_state import SampleClock
from .tabletop_bimanual_mpc_capture import (
    _clock, _identity, _integer, _mask, _native_attestation, _number,
    _plain, _raw_json_payload)
from .tabletop_bimanual_mpc_aux_capture import _require_clock_delta, _serial


VERSION = 'dh116-mpc-model-observation-v1'
_QUALIFICATION = dict(dynamics_qualified=False, solver_motor_torque_qualified=False,
    candidate_future_dynamics_qualified=False, hardware_measurement_qualified=False,
    planning_eligible=False, native_success_qualification_issued=False,
    new_expert_qualification_issued=False)


def _make_source_reader(*, stage, hands, usd_geom, usd_physics, stage_reader=None):
    """Read actual existing stage/active Python wrapper identity on each bracket.

    Static file hashes are cached only while actual stat identity is unchanged.
    No authored unit is invented; a failed unit/source read makes source NA in
    the provider. These file identities never attest loaded binary semantics.
    """
    files = {}
    def source_reader(side, actual_articulation_path):
        if stage_reader is not None and stage_reader() is not stage:
            raise ValueError('actual active stage changed; rebuild provider')
        if side not in ('left', 'right') or not isinstance(actual_articulation_path,str) or not actual_articulation_path:
            raise ValueError('actual hand side and articulation path required')
        prim = stage.GetPrimAtPath(actual_articulation_path)
        if not prim or not prim.IsValid():
            raise ValueError('actual articulation prim is unavailable in existing stage')
        root, session = stage.GetRootLayer(), stage.GetSessionLayer()
        layers = dict(root=dict(identifier=root.identifier, real_path=root.realPath),
                      session=dict(identifier=session.identifier, real_path=session.realPath))
        if not layers['root']['identifier'] or not layers['session']['identifier']:
            raise ValueError('actual root/session layer identity required')
        view = hands[side].root_physx_view
        path = Path(inspect.getfile(type(view))).resolve()
        stat = path.stat()
        key = (str(path),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if str(path) not in files or files[str(path)][0] != key:
            files[str(path)] = (key,_identity(path))
        return dict(stage_identity=json.dumps(layers,sort_keys=True,separators=(',',':')),
            stage_layers=layers, meters_per_unit=usd_geom.GetStageMetersPerUnit(stage),
            kilograms_per_unit=usd_physics.GetStageKilogramsPerUnit(stage),
            articulation_path=actual_articulation_path, prim_type_name=prim.GetTypeName(),
            root_view_python_identity=id(view),
            sdk_sources=[dict(files[str(path)][1],active_python_type=type(view).__module__+'.'+type(view).__qualname__,
                scope='actual existing root_physx_view Python wrapper source; binary semantics unqualified')])
    return source_reader


def _native_provider(env, world, *, clock_reader):
    """Lazy Native-only imports; reuse the existing stage and both hand views."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics
    from .tabletop_bimanual_mpc_articulation_measurements import ArticulationMeasurementProvider
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise ValueError('actual existing composed stage required')
    if len(world.hands) != 2:
        raise ValueError('actual existing left/right hand pair required')
    hands = dict(zip(('left','right'),world.hands))
    reader = _make_source_reader(stage=stage,hands=hands,usd_geom=UsdGeom,usd_physics=UsdPhysics,
        stage_reader=lambda:omni.usd.get_context().get_stage())
    provider = ArticulationMeasurementProvider(hands=hands,clock_reader=clock_reader,
        source_reader=reader,enabled=True,physics_dt=world.dt)
    return provider, dict(provider_source=_identity(Path(inspect.getfile(type(provider)))),
        source_scope='simulation privileged current-state full-articulation getters; not hardware',
        source_identity_policy='actual stage root/session layers, stage units, actual articulation path, active view wrapper file SHA256')


def _file_identity(path):
    try:
        return _identity(path)
    except OSError as error:
        return dict(path=str(path.resolve()),sha256=None,error=repr(error))


class _ModelCapture:
    def __init__(self,env,world,output,*,run_uuid,native_runtime_verified=False,provider_factory=None):
        if type(native_runtime_verified) is not bool:
            raise ValueError('native flag must be an actual boolean')
        if native_runtime_verified:
            _native_attestation(env,world)
            if provider_factory is not None:
                raise ValueError('native attestation excludes injected fixture factories')
        elif provider_factory is None:
            raise ValueError('CPU fixture requires explicit provider factory; genuine attestation required for SDK acquisition')
        if not isinstance(run_uuid,str) or not run_uuid:
            raise ValueError('actual shared capture run UUID required')
        if type(env.n) is not int or env.n < 1:
            raise ValueError('actual positive environment count required')
        self.env=env; self.world=world; self.run_uuid=run_uuid; self.n=env.n
        self.initial_ids=self._ids()
        if len(self.initial_ids)!=self.n:
            raise ValueError('actual episode cohort size mismatch')
        _clock(world)
        self.closed=[False]*self.n; self.counts=[0]*self.n; self.terminals={}
        self.frames=0; self.raw_records=0; self.provider_errors=[]; self.raw_write_errors=[]
        self.prior_observer_errors=[]
        self.terminal_archive_only=set(); self._policy=None; self._action=None
        self._before=None; self._last_clock=None; self._finished=False; self._summary=None
        self.native_runtime_verified=native_runtime_verified
        self._missing=object(); self._prior=getattr(env,'joint_replay_observer',self._missing)
        self._observer_hook=self.observe; self._stream=None
        self.output=Path(output).resolve(); self.output.mkdir(parents=True,exist_ok=False)
        try:
            self._stream=gzip.open(self.output/'model_480hz.jsonl.gz','xt',encoding='utf-8',compresslevel=1)
            if provider_factory is None:
                self.provider,self.metadata=_native_provider(env,world,clock_reader=self.clock_for_env)
            else:
                self.provider,self.metadata=provider_factory(clock_reader=self.clock_for_env)
            env.joint_replay_observer=self._observer_hook
        except BaseException as error:
            try:
                self.finish(error='initialization: '+repr(error))
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def _ids(self):
        return tuple(_integer(v,'actual diagnostic episode ID') for v in _plain(self.env.diagnostic_episode_ids))

    def clock_for_env(self,env_id):
        c=_clock(self.world); ids=self._ids()
        done=_mask(_plain(self.env.task.done),self.n,'actual task terminal')
        return SampleClock(run_uuid=self.run_uuid,env_id=env_id,
            episode_uid=f'{self.run_uuid}:{env_id}:{ids[env_id]}',
            policy_step=0 if self._policy is None else self._policy,
            reset_generation=ids[env_id],terminal=done[env_id],reset=False,**c)

    def submit(self,issued_action24,*,policy_step):
        if self._finished:
            raise ValueError('model capture finished')
        step=_integer(policy_step,'policy_step')
        if self._policy is not None and step!=self._policy+1:
            raise ValueError('consecutive policy submissions required')
        rows=tuple(tuple(_number(v,'issued action') for v in row) for row in _plain(issued_action24))
        if len(rows)!=self.n or any(len(row)!=24 or any(abs(v)>1 for v in row) for row in rows):
            raise ValueError('bounded actual canonical24 action required per environment')
        before=_clock(self.world)
        if self._before is not None:
            _require_clock_delta(before,self._before,8)
            if not all(self.closed) and before!=self._last_clock:
                raise ValueError('missing final physical observer before next policy action')
        else:
            self._last_clock=before
        self._policy=step; self._action=rows; self._before=before

    def observe(self,env,measurement,native_live,newly_terminal):
        if self._finished:
            return
        if env is not self.env:
            raise ValueError('actual observer environment owner required')
        actual_live=_mask(_plain(native_live),self.n,'actual_live')
        actual_terminal=_mask(_plain(newly_terminal),self.n,'actual_terminal')
        active=tuple(actual_live[i] and not self.closed[i] for i in range(self.n))
        terminals=tuple(actual_terminal[i] and active[i] for i in range(self.n))
        if any(actual_terminal[i] and not actual_live[i] and not self.closed[i] for i in range(self.n)):
            raise ValueError('first terminal requires an actual live pre-reset sample')
        if not any(active):
            return
        if self._policy is None:
            raise ValueError('submit actual policy action before model capture')
        ids=self._ids()
        if any(ids[i]!=self.initial_ids[i] for i in range(self.n) if active[i]):
            raise ValueError('first physical episode changed without terminal')
        clocks=tuple(self.clock_for_env(i) for i in range(self.n))
        if any(clocks[i].terminal!=terminals[i] for i in range(self.n) if active[i]):
            raise ValueError('actual task terminal differs from pre-reset callback')
        c=_clock(self.world)
        _require_clock_delta(c,self._last_clock,1)
        if any(any(getattr(clock,key)!=c[key] for key in c) for clock in clocks):
            raise ValueError('actual clock changed while reading model snapshot')
        delta=c['world_step']-self._before['world_step']
        if not 1<=delta<=8:
            raise ValueError('model physical clock outside submitted policy action')
        _require_clock_delta(c,self._before,delta)
        packet=dict(version=VERSION,run_uuid=self.run_uuid,clock=c,clocks=_serial(clocks),
            policy_step=self._policy,control_substep=delta,actual_live=list(actual_live),
            actual_terminal=list(actual_terminal),cohort_live=list(active),first_terminal=list(terminals),
            issued_action24=_serial(self._action),policy_clock_before=dict(self._before),
            action_owner='legacy_teacher',source_role='legacy Teacher model observation; not Student input or control',
            native_runtime_attestation_scope='runner only; not dynamics, support, motor or hardware qualification',
            capture_status='capture_pending',complete_measurement=False,terminal_measurement_valid=False,
            **_QUALIFICATION)
        # Only validated actual terminals may be pinned. An upstream observer
        # failure must not erase these receipts before Model acquires anything.
        terminal_ids=[i for i,t in enumerate(terminals) if t]
        for i in terminal_ids:
            self.terminals[str(i)]=deepcopy(packet); self.closed[i]=True
        try:
            if self._prior is not self._missing and self._prior is not None:
                self._prior(env,measurement,native_live,newly_terminal)
            if any(self.clock_for_env(i)!=clocks[i] for i,a in enumerate(active) if a):
                raise ValueError('actual model source clock/episode changed during prior observer')
        except BaseException as error:
            message=repr(error); self.prior_observer_errors.append(message)
            packet.update(capture_status='prior_observer_error',prior_observer_error=message,
                          provider_error=None,model_samples=None,provider_partial_model_raw=None,
                          provider_partial_model_raw_encoding_error=None,
                          complete_measurement=False,terminal_measurement_valid=False)
            self._record_invalid(packet,active,terminal_ids,c,error)
            raise
        samples=None
        try:
            samples=tuple(self.provider.capture_batch(clocks=clocks,active=active))
            if len(samples)!=self.n or any(s.clock!=clock for s,clock in zip(samples,clocks)):
                raise ValueError('model provider returned different actual episode/clock')
            if any(self.clock_for_env(i)!=clocks[i] for i,a in enumerate(active) if a):
                raise ValueError('actual model source clock/episode changed during readback')
            if any(any(value is not False for value in s.qualification.values()) for s in samples):
                raise ValueError('model observation cannot issue any qualification')
            serialized=_serial(samples)
        except Exception as error:
            message=repr(error); self.provider_errors.append(message)
            partial=None; encoding_error=None
            if samples is not None:
                try:
                    partial=_serial(samples)
                    # Validate encodability without replacing nonfinite values;
                    # the single archive encoder retains their full root paths.
                    json.dumps(partial,allow_nan=True)
                except Exception as exc:
                    partial=None; encoding_error=repr(exc)
            packet.update(capture_status='provider_error',provider_error=message,model_samples=None,
                          provider_partial_model_raw=partial,
                          provider_partial_model_raw_encoding_error=encoding_error,
                          complete_measurement=False,terminal_measurement_valid=False)
            self._record_invalid(packet,active,terminal_ids,c,error)
            raise
        packet.update(capture_status='complete_observation_only',provider_error=None,model_samples=serialized,
                      complete_measurement=True,terminal_measurement_valid=True)
        self._record(packet,active,terminal_ids,c,complete=True)

    def _record_invalid(self,packet,active,terminal_ids,clock,original_error):
        try:
            self._record(packet,active,terminal_ids,clock,complete=False)
        except BaseException as archive_error:
            # A failed diagnostic sink must not replace the acquisition error.
            raise original_error from archive_error

    def _record(self,packet,active,terminal_ids,clock,*,complete):
        try:
            packet=_raw_json_payload(packet)
            encoded=json.dumps(packet,allow_nan=False,separators=(',',':'))
            for i in terminal_ids:
                self.terminals[str(i)]=deepcopy(packet)
            self._stream.write(encoded+'\n')
        except Exception as error:
            message=repr(error); self.raw_write_errors.append(message)
            self.terminal_archive_only.update(terminal_ids)
            for i in terminal_ids:
                self.terminals[str(i)].update(terminal_measurement_valid=False,archive_error=message)
            raise
        self.raw_records+=1; self._last_clock=clock
        if complete:
            self.frames+=1
            for i,a in enumerate(active):
                if a:self.counts[i]+=1

    def _restore(self):
        if getattr(self.env,'joint_replay_observer',None) is self._observer_hook:
            if self._prior is self._missing:
                delattr(self.env,'joint_replay_observer')
            else:
                self.env.joint_replay_observer=self._prior

    def _write_summary(self,error):
        terminal_path=self.output/'terminal_snapshots.json'
        with terminal_path.open('x',encoding='utf-8') as stream:
            json.dump(self.terminals,stream,allow_nan=False)
        summary=dict(version=VERSION,run_uuid=self.run_uuid,physical_frames=self.frames,
            raw_records=self.raw_records,records_per_env=list(self.counts),terminal_count=len(self.terminals),
            physical_frames_semantics='successful complete observation transactions written; not complete dynamics, capability or state qualification',
            native_runtime_verified=self.native_runtime_verified,capture_valid=error is None,error=error,
            provider_error_frames=len(self.provider_errors),provider_errors=list(self.provider_errors),
            prior_observer_error_frames=len(self.prior_observer_errors),
            prior_observer_errors=list(self.prior_observer_errors),
            raw_write_errors=list(self.raw_write_errors),terminal_archive_only_env_ids=sorted(self.terminal_archive_only),
            Student_success_count=0,ppo_updates=0,physics_hz=480,policy_hz=60,action_size=24,
            reset_generation_source='actual diagnostic_episode_ids alias; not an independent counter',
            world_time_source='World step times original dt; derived local elapsed time',
            sdk_time_source='actual SimulationContext.current_time with distinct origin',
            raw=_file_identity(self.output/'model_480hz.jsonl.gz'),terminals=_file_identity(terminal_path),
            components=_serial(getattr(self,'metadata',None)),implementation=_identity(Path(__file__)),
            **_QUALIFICATION)
        with (self.output/'model_capture_summary.json').open('x',encoding='utf-8') as stream:
            json.dump(summary,stream,indent=2,allow_nan=False)
        self._summary=summary
        return summary

    def finish(self,*,error=None):
        if self._finished:
            return self._summary
        errors=[]
        try:
            self._restore()
        except BaseException as exc:
            errors.append(exc)
        try:
            if self._stream is not None:self._stream.close()
        except BaseException as exc:
            errors.append(exc)
        messages=([str(error)] if error is not None else [])+self.prior_observer_errors+self.provider_errors+self.raw_write_errors+[repr(e) for e in errors]
        failure='; '.join(messages) if messages else None
        if failure:
            for packet in self.terminals.values():
                packet.update(terminal_measurement_valid=False,archive_error=failure)
        self._finished=True
        result=self._write_summary(failure)
        if errors:
            raise errors[0]
        return result


def install_mpc_model_capture(env,world,output,*,enabled=False,**options):
    if type(enabled) is not bool:
        raise ValueError('explicit bool enabled required')
    if not enabled:
        return None
    return _ModelCapture(env,world,output,**options)
