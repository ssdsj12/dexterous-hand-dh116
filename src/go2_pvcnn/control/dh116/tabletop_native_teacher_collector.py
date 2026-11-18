"""Actual original-control/PPO collector, with durable poison-on-source-failure.

Exactly one episode.control per decision; no sim.step or optimizer call here.
Source adapters supply typed measured endpoints, never qualification booleans.
"""
from copy import deepcopy
from dataclasses import asdict,is_dataclass
import math
import os
import json
from pathlib import Path

import torch

from .tabletop_deformable_r0_replay import _copy,_digest,encode_owned
from .tabletop_deformable_privileged_ppo import DeformablePPOTrainer
from .tabletop_deformable_rollout_admission import gae_boundary


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

    def close(self): self._stream.close()


class _TerminalObserver:
    """Original recorder runs first; propagate trusted reward terminal immediately."""
    def __init__(self,inner,reason,journal,on_advanced):
        self.inner,self.reason,self.journal,self.on_advanced=inner,reason,journal,on_advanced
    def control(self,action,*,observe):
        def delivered(row):
            # Raw advanced exchange survives even if original boundary conversion
            # raises before the episode's physical recorder is reached.
            self.on_advanced(row)
            self.journal.emit("delivered_exchange",row)
            observe(row)
            return self.reason()
        return self.inner.control(action,observe=delivered)


class NativeTeacherCollector:
    def __init__(self,owner,trainer,reward,live,*,previous=None):
        from .tabletop_training_reward import TrainingReward
        from .tabletop_native_teacher_model import NativeContextTeacher
        if type(trainer) is not DeformablePPOTrainer or type(reward) is not TrainingReward:
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
        if self.sources.scope!="cpu_fixture":
            from .tabletop_native_teacher_session import NativeEpisodeOwner,LiveContract
            from .tabletop_native_teacher_sources import NativePinchSources
            from .tabletop_native_teacher_model import model_contract,verify_native_teacher_identity
            if (type(owner) is not NativeEpisodeOwner or type(live) is not LiveContract
                    or type(self.sources) is not NativePinchSources or live.static is not owner.contract
                    or trainer.model._backend_profile is not owner.backend_profile
                    or trainer.config!=live.static.ppo_config
                    or trainer.rollout_steps!=live.static.config.rollout_steps):
                raise ValueError("same verified owner/model/profile/approved PPO config required")
            verify_native_teacher_identity(trainer.model.model_identity,model_contract(
                live.model_contract.config,owner.backend_profile,live.model_contract.context_mode))
        self.journal=owner.ledger
        self._student_stereo = None
        self.quarantined=self._closed=self._active=self.episode_done=False
        self.consumed_sdk_steps=0 if previous is None else previous.consumed_sdk_steps
        self.transitions=0 if previous is None else previous.transitions
        self._terminal=None;self._rows=[]
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
                self._rows.append(_copy(row))
                endpoint=self.sources.observe(row)
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
        ep._exchange=_TerminalObserver(original_exchange,lambda:self._terminal,self.journal,advanced)
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

    def attach_student_stereo(self, camera, output):
        """Transfer a fresh camera on this simulation to this opt-in collector."""
        from .tabletop_native_teacher_stereo import capture_identity
        from .tabletop_stereo_calibration import StereoCalibration
        from .tabletop_teacher_stereo_alignment import STUDENT_CAMERA
        from .tabletop_tiled_stereo_capture import TiledStereoCapture

        if (type(camera) is not TiledStereoCapture or camera.sim is not self.owner.sim
                or camera.last_capture_id != 0 or self._student_stereo is not None
                or self._closed or self.quarantined or self._active or self.transitions != 0):
            raise ValueError("fresh fixed Student stereo and the same unadvanced Teacher owner required")
        project = Path(self.owner.contract.config.project).resolve()
        expected = StereoCalibration.from_json(project / STUDENT_CAMERA)
        output = Path(output).resolve()
        if camera.calibration.sha256 != expected.sha256 or output.is_relative_to(project):
            raise ValueError("fixed Student calibration and independent evidence directory required")
        output.mkdir(exist_ok=False)
        camera.snapshot = lambda: capture_identity(
            self.owner.pre_action_state(), expected.sha256, sdk_timestamp_s=float(camera.sim.current_time))
        self._student_stereo = camera, output

    def collect(self):
        from .tabletop_training_reward import SourceFailure
        from .tabletop_native_teacher_context import ContextIdentity
        if self.quarantined or self._closed or self._active or self.episode_done:
            raise RuntimeError("closed/quarantined/terminal collector; new verified episode required")
        tr,owner=self.trainer,self.owner
        remaining=self.sources.remaining_substeps()
        if remaining<8:
            self.journal.emit("window_not_started",dict(remaining_substeps=remaining))
            return None
        if len(tr.rollout)>=tr.rollout_steps: raise RuntimeError("rollout full; complete original PPO update first")
        self._active=True;self._terminal=None;self._rows=[]
        try:
            self._check_policy()
            pre=owner.pre_action_state();prehash=_digest(pre)
            captured=self.history.capture_pre_action(pre)
            context=self.context.freeze_pre_action(ContextIdentity.from_clock(pre["common"]["identity"]),history_owner=self.history)
            obs=self.sources.pack(tr.model,pre,context)
            if _digest(owner.pre_action_state())!=prehash: raise SourceFailure("owner advanced/changed during pre-action packing")
            stereo = {}
            if self._student_stereo is not None:
                from .tabletop_native_teacher_stereo import save_aligned_rgb_pair
                camera, output = self._student_stereo
                label = save_aligned_rgb_pair(
                    pre, camera.capture(), camera.calibration, output, origins=camera.origins)
                if label["state_sha256"] != prehash or _digest(owner.pre_action_state()) != prehash:
                    raise SourceFailure("Teacher state changed during Student RGB capture")
                stereo = dict(student_stereo=label)
                remaining = self.sources.remaining_substeps()
                if remaining < 8:
                    self.journal.emit("window_not_started", dict(
                        remaining_substeps=remaining, reason="after_student_stereo", **stereo))
                    return None
            sample=tr.collect_step(obs,obs);token=sample["collection_token"]
            if token in self._used_tokens: raise SourceFailure("replayed policy collection token")
            decision=self.history.begin(pre,tr)
            self.reward.begin(self.sources.submitted_request(sample,decision),remaining_substeps=remaining)
            self._used_tokens.add(token)
            self.journal.emit("decision",dict(pre_action_sha256=prehash,observation=obs,
                collection_token=token,policy=self._epoch,sampled_action24=sample["actions"],history=decision,**stereo))
            result=owner.control(sample["actions"])
            if result["physical_samples"]!=len(self._rows) or not 1<=len(self._rows)<=8:
                raise SourceFailure("actual original control window differs from durable rows")
            # History's existing sequence validates the ORIGINAL physical guard.
            # Additional trusted reward guard is retained in actual result/packet.
            historical=deepcopy(result);historical["termination_reason"]=self._rows[-1]["exchange"]["guard_failure"]
            if len(self._rows)<8 and historical["termination_reason"] is None:
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
            if not terminated:
                next_pre=owner.pre_action_state();next_hash=_digest(next_pre)
                if ContextIdentity.from_clock(next_pre["common"]["identity"])!=post_context.metadata.identities[0]:
                    raise SourceFailure("next value endpoint differs from actual pre-reset endpoint")
                next_obs=self.sources.pack(tr.model,next_pre,post_context)
                with torch.no_grad(): value=tr.model.value(next_obs).detach().cpu().clone()
                if _digest(owner.pre_action_state())!=next_hash: raise SourceFailure("bootstrap source advanced before reset")
                next_endpoint=deepcopy(endpoint)
            boundary=gae_boundary(terminated=bool(terminated),truncated=bool(truncated),
                endpoint_identity=endpoint,next_value_identity=next_endpoint)
            self._check_policy()
            if stereo:
                self.journal.emit("student_stereo_action",dict(collection_token=token,
                                  pre_action_sha256=prehash, issued_actions=issued, **stereo))
            self.journal.emit("settlement",dict(reward=packet,post_context=post_context,
                history=settlement,result=result,issued_actions=issued,next_value=value,gae_boundary=boundary))
            tr.record_step(torch.tensor([packet.reward]),torch.tensor([bool(terminated or truncated)]),value,
                truncated=torch.tensor([bool(truncated)]),bootstrap_mask=torch.tensor([not terminated]),
                executed_actions=issued,collection_token=token)
            self.journal.emit("transition_recorded",dict(collection_token=token,rollout_length=len(tr.rollout)))
            self.context.commit_post_context(post_context)
            self.transitions+=1;self.episode_done=bool(terminated or truncated)
            return packet
        except BaseException as exc:
            self.quarantined=True
            try:
                self.journal.emit("rollout_quarantined",dict(type=type(exc).__name__,message=str(exc),
                    consumed_sdk_steps=self.consumed_sdk_steps,recorded_transitions=len(tr.rollout),
                    pending_preserved=tr._pending is not None,recovery="last_clean_checkpoint_only"))
            except BaseException: pass
            raise
        finally: self._active=False

    def restart_episode(self,factories):
        from .tabletop_native_teacher_session import verify_reset_and_sources
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
            reward=factories.reward(live.reward_contract)
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
