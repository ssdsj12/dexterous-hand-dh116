"""Actual original-control/PPO collector, with durable poison-on-source-failure.

Exactly one episode.control per decision; no sim.step or optimizer call here.
Source adapters supply typed measured endpoints, never qualification booleans.
"""
from contextlib import contextmanager, nullcontext, suppress
from copy import deepcopy
from dataclasses import asdict,is_dataclass
import hashlib
import math
import os
import json
from pathlib import Path
from time import perf_counter_ns

import torch

from .tabletop_deformable_r0_replay import decode_owned
from .tabletop_fast_owned_codec import encode_owned
from .tabletop_deformable_privileged_ppo import DeformablePPOTrainer
from .tabletop_deformable_rollout_admission import gae_boundary


def _copy(value):
    return decode_owned(encode_owned(value))


def _digest(value):
    return hashlib.sha256(json.dumps(encode_owned(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class NativeJournal:
    def __init__(self,path,*,scope="owner_source_unqualified"):
        self.path=Path(path);self.scope=scope;self._seq=0
        self._stream=self.path.open("x",encoding="utf-8")

    def emit(self,operation,payload):
        def owned(v):
            if is_dataclass(v): return {k:owned(getattr(v,k)) for k in v.__dataclass_fields__}
            if isinstance(v,dict) or hasattr(v,"items"): return {k:owned(x) for k,x in v.items()}
            if isinstance(v,(tuple,list)): return [owned(x) for x in v]
            return encode_owned(v)
        row=dict(sequence=self._seq,scope=self.scope,operation=operation,payload=owned(payload))
        self._stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
        self._stream.flush();os.fsync(self._stream.fileno());self._seq+=1
        return row["sequence"]

    def close(self): self._stream.close()


class _WindowWallTime:
    PHASES = ("source_binding", "pre_action", "physical_control", "post_action")

    def __init__(self):
        self.phase_ns = dict.fromkeys(self.PHASES, 0)
        self.phase = self.PHASES[0]
        self.started_ns = self.boundary_ns = perf_counter_ns()
        self.physical_observer_ns = 0

    def advance(self, next_phase):
        boundary = perf_counter_ns()
        self.phase_ns[self.phase] = boundary - self.boundary_ns
        self.phase = next_phase
        self.boundary_ns = boundary

    @contextmanager
    def observe(self):
        started_ns = perf_counter_ns()
        try:
            yield
        finally:
            self.physical_observer_ns += perf_counter_ns() - started_ns


class _TerminalObserver:
    """Original recorder runs first; propagate trusted reward terminal immediately."""
    def __init__(self,inner,reason,journal,on_advanced,wall_time):
        self.inner,self.reason,self.journal,self.on_advanced=inner,reason,journal,on_advanced
        self.wall_time = wall_time
    def control(self,action,*,observe):
        def delivered(row):
            # Raw advanced exchange survives even if original boundary conversion
            # raises before the episode's physical recorder is reached.
            timing = self.wall_time()
            with timing.observe() if timing is not None else nullcontext():
                self.on_advanced(row)
                self.journal.emit("delivered_exchange",row)
                observe(row)
                return self.reason()
        return self.inner.control(action,observe=delivered)


class NativeTeacherCollector:
    def __init__(self,owner,trainer,reward,live,*,previous=None):
        from .tabletop_training_reward import TrainingReward
        from .tabletop_native_teacher_model import NativeContextTeacher
        lift_diagnostic = False
        if type(reward) is not TrainingReward:
            from .tabletop_teacher_lift_reward import SupportedLiftDiagnosticReward
            lift_diagnostic = type(reward) is SupportedLiftDiagnosticReward
        if type(trainer) is not DeformablePPOTrainer or not (type(reward) is TrainingReward or lift_diagnostic):
            raise ValueError("original PPO trainer and loaded TrainingReward required")
        if not isinstance(trainer.model,NativeContextTeacher) or trainer.num_envs != 1:
            raise ValueError("independent single-environment NativeContextTeacher required")
        if previous is not None:
            if (type(previous) is not NativeTeacherCollector or previous.trainer is not trainer
                    or previous.quarantined or not previous.episode_done or trainer._pending is not None
                    or previous._epoch!=previous._fingerprint()):
                raise ValueError("completed prior episode and same clean pending policy required")
        elif trainer._pending is not None or len(trainer.rollout):
            raise ValueError("clean initial trainer boundary required")
        self.owner,self.trainer,self.reward,self.live=owner,trainer,reward,live
        self.history,self.context,self.sources=owner.history,owner.context,owner.sources
        self._source_profile = None
        if self.sources.scope!="cpu_fixture":
            from .tabletop_native_teacher_session import NativeEpisodeOwner,LiveContract
            from .tabletop_native_teacher_sources import NativePinchSources,NativeTripodSources,NativeContactTripodSources
            from .tabletop_native_teacher_model import model_contract,verify_native_teacher_identity
            style=owner.contract.config.preflight.get("style") if type(owner) is NativeEpisodeOwner else None
            mode=owner.contract.config.preflight.get("profile") if type(owner) is NativeEpisodeOwner else None
            source_type=(NativeContactTripodSources if mode == "approach_contact_prelift_v1" and style == "tripod"
                         else {"pinch":NativePinchSources,"tripod":NativeTripodSources}.get(style))
            if (type(owner) is not NativeEpisodeOwner or type(live) is not LiveContract
                    or source_type is None or type(self.sources) is not source_type
                    or live.static is not owner.contract or trainer.model.model_identity["grasp"]!=style
                    or trainer.model._backend_profile is not owner.backend_profile
                    or trainer.config!=live.static.ppo_config
                    or trainer.rollout_steps!=live.static.config.rollout_steps):
                raise ValueError("same verified owner/model/profile/approved PPO config required")
            verify_native_teacher_identity(trainer.model.model_identity,model_contract(
                live.model_contract.config,owner.backend_profile,live.model_contract.context_mode))
            from .tabletop_native_teacher_lift_trial import read_lift_trial
            trial = read_lift_trial(live.reward_contract)
            if (lift_diagnostic != (trial is not None) or lift_diagnostic and (
                    reward.fixed_reference is not getattr(owner, "lift_reference", None)
                    or reward._contact_owner() is not owner.context
                    or reward._diagnostic_sha != trial.config_sha256)):
                raise ValueError("same live native Lift reward and pinned trial required")
            self._source_profile = owner.backend_profile
        self.journal=owner.ledger
        self._student_stereo = None
        self._wall_trace_sink = self._window_wall = None
        self.quarantined=self._closed=self._active=self.episode_done=False
        self.consumed_sdk_steps=0 if previous is None else previous.consumed_sdk_steps
        self.transitions=0 if previous is None else previous.transitions
        self._terminal=None;self._guard_results=[]
        self._used_tokens=set() if previous is None else set(previous._used_tokens);self._epoch=self._fingerprint()
        self._updates=trainer.counters["updates"]
        self.sources.reset_reward(reward)
        ep=owner.episode;original_record=ep._record;original_exchange=ep._exchange
        self._original_record,self._original_exchange=original_record,original_exchange
        def record(row):
            # Owner recorder durably fsyncs the COMPLETE physical event first.
            # Physical history is then validated before reward can consume it.
            try:
                original_record(row)
                self.journal.emit("budget_consumed",dict(sdk_steps=self.consumed_sdk_steps,
                    solver_calls=self.consumed_sdk_steps,internal_solver_substeps=self.consumed_sdk_steps*self.sources.solver_substeps))
                if row.get("kind")!="physical" or not self._active:
                    raise RuntimeError("physical source outside pending window")
                self._guard_results.append(row["exchange"]["guard_failure"])
                endpoint=self.sources.observe(row)
                if lift_diagnostic:
                    self.reward.observe_recorded_event(row, endpoint)
                else:
                    self.reward.observe(endpoint)
                if self.reward._terminal:
                    self._terminal="reward:"+"|".join(self.reward._reasons)
            except BaseException:
                self.quarantined=True
                raise
        ep._record=record
        def advanced(row):
            self.consumed_sdk_steps+=1
            runtime=getattr(getattr(owner,"contract",None),"runtime",None)
            if runtime is not None: runtime.charge(row,self.sources.solver_substeps)
        ep._exchange=_TerminalObserver(original_exchange,lambda:self._terminal,self.journal,advanced,
                                       lambda: self._window_wall)
        owner._restores.append(lambda:setattr(ep,"_record",original_record))
        owner._restores.append(lambda:setattr(ep,"_exchange",original_exchange))
        self.journal.emit("collector_origin",dict(policy=self._epoch,
            native_qualification=False,teacher_admission=False,scope=self.sources.scope))

    def _fingerprint(self):
        tr=self.trainer
        return dict(session_id=tr.session_id,policy_version=tr.policy_version,
            weights=_digest(tr.model.state_dict()),model=_digest(tr.model.model_identity))

    def _check_policy(self):
        if self._fingerprint()!=self._epoch: raise ValueError("policy epoch/model weights changed within collection")

    def accept_updated_policy(self):
        if self._active or self.quarantined or self.trainer._pending is not None or len(self.trainer.rollout):
            raise RuntimeError("clean completed PPO update boundary required")
        old=self._epoch;new=self._fingerprint()
        if (new["session_id"]!=old["session_id"] or new["policy_version"]!=old["policy_version"]+1
                or new["model"]!=old["model"] or self.trainer.counters["updates"]!=self._updates+1):
            raise ValueError("original trainer's next policy epoch required")
        self._epoch=new;self._updates=self.trainer.counters["updates"]
        self.journal.emit("policy_epoch",new)

    def update(self):
        if self.quarantined or self._active or self._closed:
            raise RuntimeError("quarantined/active/closed rollout cannot update")
        if len(self.trainer.rollout)!=self.trainer.rollout_steps:
            raise RuntimeError("complete original rollout required; discarded steps cannot refill")
        result=self.trainer.update()
        self.accept_updated_policy()
        return result

    def attach_student_stereo(self, camera, output, *, camera_profile="front"):
        """Transfer a fresh camera on this simulation to this opt-in collector."""
        from .tabletop_native_teacher_stereo import capture_identity
        from .tabletop_teacher_stereo_alignment import student_camera_calibration
        from .tabletop_tiled_stereo_capture import TiledStereoCapture

        if (type(camera) is not TiledStereoCapture or camera.sim is not self.owner.sim
                or camera.last_capture_id != 0 or self._student_stereo is not None
                or self._closed or self.quarantined or self._active or self.transitions != 0):
            raise ValueError("fresh fixed Student stereo and the same unadvanced Teacher owner required")
        project = Path(self.owner.contract.config.project).resolve()
        expected = student_camera_calibration(project, camera_profile)
        output = Path(output).resolve()
        if camera.calibration.sha256 != expected.sha256 or output.is_relative_to(project):
            raise ValueError("fixed Student calibration and independent evidence directory required")
        output.mkdir(exist_ok=False)
        camera.snapshot = lambda: capture_identity(
            self.owner.pre_action_state(), expected.sha256, sdk_timestamp_s=float(camera.sim.current_time))
        self._student_stereo = camera, output, camera_profile

    def _pack_observation(self, pre, context, *, purpose):
        recorded = getattr(self.sources, "pack_recorded", None)
        if recorded is None:
            return self.sources.pack(self.trainer.model, pre, context), {}
        observation, sequence = recorded(self.trainer.model, pre, context, purpose=purpose)
        return observation, dict(source_pack_sequence=sequence)

    @contextmanager
    def trace_wall_windows(self, sink):
        """Use a separate diagnostic sink for this synchronous collection scope."""
        if not callable(sink) or self._active or self._closed or self._wall_trace_sink is not None:
            raise ValueError("an idle collector and an independent timing sink are required")
        self._wall_trace_sink = sink
        try:
            yield
        finally:
            self._wall_trace_sink = None

    def collect(self):
        from .tabletop_native_teacher_profile import control_source_scope
        if self.quarantined or self._closed or self._active or self.episode_done:
            raise RuntimeError("closed/quarantined/terminal collector; new verified episode required")
        tr=self.trainer
        remaining=self.sources.remaining_substeps()
        if remaining<8:
            self.journal.emit("window_not_started",dict(remaining_substeps=remaining))
            return None
        if len(tr.rollout)>=tr.rollout_steps: raise RuntimeError("rollout full; complete original PPO update first")
        self._active=True;self._terminal=None;self._guard_results=[]
        sink = self._wall_trace_sink
        timing = self._window_wall = _WindowWallTime() if sink is not None else None
        sdk_steps_before, transitions_before = self.consumed_sdk_steps, self.transitions
        packet = None
        try:
            scope = (nullcontext() if self._source_profile is None
                     else control_source_scope(self._source_profile))
            with scope as verify_source:
                if timing is not None:
                    timing.advance("pre_action")
                packet = self._collect_window(remaining, verify_source)
                return packet
        except BaseException as exc:
            self.quarantined=True
            try:
                self.journal.emit("rollout_quarantined",dict(type=type(exc).__name__,message=str(exc),
                    consumed_sdk_steps=self.consumed_sdk_steps,recorded_transitions=len(tr.rollout),
                    pending_preserved=tr._pending is not None,recovery="last_clean_checkpoint_only"))
            except BaseException: pass
            raise
        finally:
            self._active = False
            self._window_wall = None
            if timing is not None:
                timing.advance(None)
                terminal_reason = (self._guard_results[-1]
                                   if packet is not None and packet.terminated else None)
                # A trace write failure cannot replace the original physical/source failure.
                with suppress(BaseException) if self.quarantined else nullcontext():
                    sink("control_window_wall", version="dh116-control-window-wall-v1",
                         scope=self.sources.scope, control_index=transitions_before,
                         elapsed_ns=timing.boundary_ns - timing.started_ns, phase_ns=timing.phase_ns,
                         physical_observer_ns=timing.physical_observer_ns,
                         sdk_steps_delivered=self.consumed_sdk_steps - sdk_steps_before,
                         transition_recorded=self.transitions > transitions_before,
                         terminated=bool(packet and packet.terminated), truncated=bool(packet and packet.truncated),
                         terminal_reason=terminal_reason, quarantined=self.quarantined,
                         native_qualification=False, teacher_admission=False)

    def _collect_window(self, remaining, verify_source):
        from .tabletop_training_reward import SourceFailure
        from .tabletop_native_teacher_context import ContextIdentity

        tr, owner = self.trainer, self.owner
        self._check_policy()
        pre=owner.pre_action_state();prehash=_digest(pre)
        captured=self.history.capture_pre_action(pre)
        context=self.context.freeze_pre_action(ContextIdentity.from_clock(pre["common"]["identity"]),history_owner=self.history)
        obs, decision_source = self._pack_observation(pre, context, purpose="decision")
        if _digest(owner.pre_action_state())!=prehash: raise SourceFailure("owner advanced/changed during pre-action packing")
        stereo = {}
        if self._student_stereo is not None:
            from .tabletop_native_teacher_stereo import save_aligned_rgb_pair
            camera, output, camera_profile = self._student_stereo
            label = save_aligned_rgb_pair(
                pre, camera.capture(), camera.calibration, output, origins=camera.origins, camera_profile=camera_profile)
            if label["state_sha256"] != prehash or _digest(owner.pre_action_state()) != prehash:
                raise SourceFailure("Teacher state changed during Student RGB capture")
            stereo = dict(student_stereo=label)
            remaining = self.sources.remaining_substeps()
            if remaining < 8:
                self.journal.emit("window_not_started", dict(
                    remaining_substeps=remaining, reason="after_student_stereo", **decision_source, **stereo))
                return None
        sample=tr.collect_step(obs,obs);token=sample["collection_token"]
        if token in self._used_tokens: raise SourceFailure("replayed policy collection token")
        decision=self.history.begin(pre,tr)
        self.reward.begin(self.sources.submitted_request(sample,decision),remaining_substeps=remaining)
        self._used_tokens.add(token)
        self.journal.emit("decision",dict(pre_action_sha256=prehash,observation=obs,
            collection_token=token,policy=self._epoch,sampled_action24=sample["actions"],history=decision,
            **decision_source, **stereo))
        timing = self._window_wall
        if timing is not None:
            timing.advance("physical_control")
        result=owner.control(sample["actions"])
        if timing is not None:
            timing.advance("post_action")
        if result["physical_samples"]!=len(self._guard_results) or not 1<=len(self._guard_results)<=8:
            raise SourceFailure("actual original control window differs from durable rows")
        # History's existing sequence validates the ORIGINAL physical guard.
        # Additional trusted reward guard is retained in actual result/packet.
        historical=deepcopy(result);historical["termination_reason"]=self._guard_results[-1]
        if len(self._guard_results)<8 and historical["termination_reason"] is None:
            # Sequence's terminal rule requires a declared physical stop;
            # actual observer result is the trusted independent health guard.
            raise SourceFailure("trusted observer terminal lacks persisted physical guard")
        settlement=self.history.end(historical)
        terminated=self.reward._terminal
        truncated=not terminated and self.sources.external_truncated()
        packet=self.reward.settle_once(token,truncated=truncated,
            buffer_full=not terminated and not truncated and len(tr.rollout)+1==tr.rollout_steps)
        post_state=self.sources.context_state(packet)
        post_context=self.context.advance_physical_endpoint(post_state,history_owner=self.history,collection_token=token)
        issued=self.history.issued_for_pending(tr,settlement)
        endpoint=deepcopy(settlement["endpoint"])
        endpoint={k:endpoint[k] for k in ("env_id","episode_uid","reset_id","rod_uid","physics_step","rod_tick","timestamp_s")}
        endpoint["phase"]="post_action_pre_reset"
        next_endpoint=None;value=torch.zeros(1)
        bootstrap_source = None
        if not terminated:
            next_pre=owner.pre_action_state();next_hash=_digest(next_pre)
            if ContextIdentity.from_clock(next_pre["common"]["identity"])!=post_context.metadata.identities[0]:
                raise SourceFailure("next value endpoint differs from actual pre-reset endpoint")
            next_obs, bootstrap_binding = self._pack_observation(next_pre, post_context, purpose="bootstrap")
            # Preserve the critic input before evaluation, while it still belongs
            # to the physical endpoint of this action and before any Reset.
            observation = _copy(next_obs)
            with torch.no_grad(): value=tr.model.value(next_obs).detach().cpu().clone()
            if _digest(owner.pre_action_state())!=next_hash: raise SourceFailure("bootstrap source advanced before reset")
            next_endpoint=deepcopy(endpoint)
            bootstrap_source = dict(collection_token=token, identity=next_endpoint,
                                    pre_action_sha256=next_hash, observation=observation,
                                    observation_sha256=_digest(observation), **bootstrap_binding)
        boundary=gae_boundary(terminated=bool(terminated),truncated=bool(truncated),
            endpoint_identity=endpoint,next_value_identity=next_endpoint)
        self._check_policy()
        if verify_source is not None:
            verify_source()
        if stereo:
            label = stereo["student_stereo"]
            if label["student_sync_usable"] and label["student_hand_order_usable"]:
                self.journal.emit("student_stereo_action",dict(collection_token=token,
                                  pre_action_sha256=prehash, issued_actions=issued, **stereo))
            else:
                self.journal.emit("student_stereo_rejected",dict(collection_token=token,
                                  pre_action_sha256=prehash, **stereo))
        self.journal.emit("settlement",dict(reward=packet,post_context=post_context,
            history=settlement,result=result,issued_actions=issued,next_value=value,
            gae_boundary=boundary,bootstrap_source=bootstrap_source))
        tr.record_step(torch.tensor([packet.reward]),torch.tensor([bool(terminated or truncated)]),value,
            truncated=torch.tensor([bool(truncated)]),bootstrap_mask=torch.tensor([not terminated]),
            executed_actions=issued,collection_token=token)
        self.journal.emit("transition_recorded",dict(collection_token=token,rollout_length=len(tr.rollout)))
        self.context.commit_post_context(post_context)
        self.transitions+=1;self.episode_done=bool(terminated or truncated)
        return packet

    def restart_episode(self,factories):
        from .tabletop_native_teacher_session import verify_reset_and_sources,_new_reward
        if self.quarantined or not self.episode_done or self._active:
            raise RuntimeError("only a completed terminal/external truncation can reset")
        if self._student_stereo is not None:
            raise RuntimeError("Student stereo recording requires an independently attached episode owner")
        static=self.live.static
        self.close()
        owner=factories.episode(static)
        try:
            reset=owner.reset(static.reset_seed,episode_uid=owner.episode_uid,owner_resets=owner.owner_resets)
            live=verify_reset_and_sources(owner,reset,static)
            reward=_new_reward(factories,live,owner)
            return NativeTeacherCollector(owner,self.trainer,reward,live,previous=self)
        except BaseException:
            try:owner.close()
            except BaseException as cleanup:
                try:owner.record_cleanup_failure(cleanup)
                except BaseException:pass
            raise

    def close(self):
        if self._closed:return
        self._closed=True
        stereo, self._student_stereo = self._student_stereo, None
        try:
            try:
                if stereo is not None:
                    stereo[0].close()
            finally:
                self.owner.close()
        except BaseException as cleanup:
            self.owner.record_cleanup_failure(cleanup)
            raise
        finally:
            self._original_record=self._original_exchange=None
