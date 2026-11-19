"""Two-stage native Teacher construction over one borrowed physical scene.

No App is created or closed here. R06 outer lifecycle owns native teardown after
text-only failure capture/traceback frame clearing. CPU tests grant no native scope.
"""
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
import json

from .tabletop_native_teacher_preflight import (candidate_config, static_preflight,
    load_reset_manifest, validate_budget, live_preflight, _binding)
from .tabletop_native_teacher_profile import BackendProfile, ROOT
from .tabletop_native_teacher_collector import NativeTeacherCollector


@dataclass(frozen=True)
class NativeFactories:
    episode: object
    policy: object
    trainer: object
    reward: object

    def __post_init__(self):
        if not all(callable(x) for x in (self.episode,self.policy,self.trainer,self.reward)):
            raise TypeError("four explicit callable factories required")


@dataclass(frozen=True)
class NativeSessionConfig:
    preflight: dict
    model_config: dict
    journal_path: object
    episode_cap_s: float
    rollout_steps: int
    project: object = ROOT


@dataclass(frozen=True)
class StaticContract:
    config: NativeSessionConfig
    backend_profile: BackendProfile
    reset_manifest: object
    reset_seed: dict
    ppo_config: object
    reward_config: object
    budget: dict
    runtime: object
    config_digest: str


@dataclass(frozen=True)
class ModelContract:
    config: dict
    backend_profile: BackendProfile
    context_mode: str


@dataclass(frozen=True)
class LiveContract:
    static: StaticContract
    model_contract: ModelContract
    reward_contract: object
    verification: dict


def native_reset_source_limits(*,long_clock=False):
    if long_clock:
        return dict(schema="dh116-mesh-reset-clock-v2-limits",max_reset_sdk_step=100000,
            long_term_reset_supported=True,policy_interval_sdk_cap=None,
            reader="tabletop_long_clock_v2._clock_long",
            blocked_reason=None,release_condition="long_clock_v2_enabled")
    return dict(schema="dh116-mesh-reset-clock-v1-limits",max_reset_sdk_step=240,
        long_term_reset_supported=False,policy_interval_sdk_cap=None,
        reader="tabletop_mesh_reset_order._clock",
        blocked_reason="bounded_mesh_reset_clock_v2_source_missing_for_reset_above_sdk240",
        release_condition="independent budget/profile/actual-clock Reset reader v2; frozen v1 remains unchanged")


def reset_clock_preflight(owner,*,long_clock=False):
    limits=native_reset_source_limits(long_clock=long_clock)
    step=owner.sim.current_time_step_index
    if type(step) is not int or step<0:raise ValueError("actual absolute SDK Reset clock required")
    if step>limits["max_reset_sdk_step"]:
        raise ValueError(limits["blocked_reason"])
    return dict(limits,actual_reset_sdk_step=step,reset_clock_admitted=True)


def verify_static_contract(config):
    if type(config) is not NativeSessionConfig:
        raise ValueError("explicit NativeSessionConfig and independently pinned sources required")
    import math
    if (type(config.rollout_steps) is not int or config.rollout_steps < 1
            or type(config.episode_cap_s) not in (int,float)
            or not math.isfinite(config.episode_cap_s) or config.episode_cap_s <= 0):
        raise ValueError("finite episode cap and rollout length required")
    profile=BackendProfile.load(config.project)
    gate=static_preflight(config.preflight,project=config.project,backend_profile=profile)
    if not gate["static_ready"]:
        raise ValueError("native static contract denied: "+", ".join(gate["blocked_reasons"]))
    # C04's implemented real reader is Pinch smoke. Full requires a separately
    # implemented common evaluator/source adapter; JSON declarations cannot supply it.
    if config.preflight["style"] != "pinch":
        raise ValueError(config.preflight["style"]+"_distance_reader_implementation_pending")
    if config.preflight["profile"] != "approach_wiring_smoke_v1":
        raise ValueError("full_shared_contact_slip_hold_source_adapter_missing")
    if config.model_config.get("grasp")!=config.preflight["style"]:
        raise ValueError("model/source grasp identity differs")
    from .tabletop_training_reward import load_config
    from .tabletop_privileged_ppo import PPOConfig
    from .tabletop_native_teacher_model import model_contract
    manifest=load_reset_manifest(config.preflight,project=config.project)
    budget=validate_budget(config.preflight,project=config.project)
    path,_=_binding(config.preflight["reward_config"],config.project)
    reward=load_config(path,ppo_gamma=config.preflight["ppo"]["gamma"])
    mapping={"learning_rate":"lr","gamma":"gamma","gae_lambda":"lam","clip":"clip",
        "value":"value_coef","entropy":"entropy_coef","max_grad_norm":"max_grad_norm",
        "epochs":"epochs","minibatches":"minibatches","target_kl":"target_kl",
        "kl_backtrack_steps":"kl_backtrack_steps"}
    values={mapping[k]:v for k,v in reward.ppo.items() if k in mapping}
    for key,value in config.preflight["ppo"].items():
        target=mapping.get(key,key)
        if target not in values or values[target]!=value:
            raise ValueError("preflight/PPO/reward numeric config differs: "+key)
    if config.rollout_steps!=reward.ppo["rollout_steps"]:
        raise ValueError("rollout length differs from approved reward/PPO config")
    ppo=PPOConfig(**values)
    model_contract(config.model_config,profile,config.preflight["profile"])
    from .tabletop_deformable_r0_replay import _digest
    binding=_digest(dict(preflight=config.preflight,model_config=config.model_config,
        episode_cap_s=config.episode_cap_s,rollout_steps=config.rollout_steps))
    return StaticContract(config,profile,manifest,manifest.data()["reset_seed"],ppo,reward,budget,SessionBudget(budget),binding)


def verify_reset_and_sources(owner,reset,contract):
    if type(owner) is not NativeEpisodeOwner or owner.contract is not contract:
        raise ValueError("sole same-contract NativeEpisodeOwner required")
    from .tabletop_deformable_r0_replay import _digest
    if _digest(dict(preflight=contract.config.preflight,model_config=contract.config.model_config,
            episode_cap_s=contract.config.episode_cap_s,rollout_steps=contract.config.rollout_steps))!=contract.config_digest:
        raise ValueError("static model/context/reward/budget configuration changed during Reset")
    verification=live_preflight(owner,owner.live_reset_contract(reset),
        backend_profile=contract.backend_profile,manifest=contract.reset_manifest)
    owner.initialize_context_and_reward_source(reset)
    return LiveContract(contract,ModelContract(contract.config.model_config,
        contract.backend_profile,contract.config.preflight["profile"]),contract.reward_config,verification)


def build_fresh_pinch_owner(contract, *, sim, owners, prepared, ownership, ledger,
                            run_uuid, episode_uid, owner_resets, mesh_plan):
    """Use the original Pinch builder inside the existing owner and journal."""
    import sys
    from contextlib import nullcontext
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    from . import tabletop_dual_pinch_builder as builder
    from .tabletop_native_teacher_collector import NativeJournal
    from .tabletop_native_teacher_profile import profile_reader_scope

    fresh = None
    try:
        if (type(contract) is not StaticContract or type(ledger) is not NativeJournal
                or contract.config.preflight["style"] != "pinch"
                or contract.config.preflight["profile"] != "approach_wiring_smoke_v1"
                or Path(contract.config.journal_path) != ledger.path):
            raise ValueError("verified Pinch smoke contract and its own journal required")
        def record_setup(row):
            if row["kind"] == "setup":
                # Failed scene construction still has external ownership
                # evidence. NativeEpisodeOwner writes setup once to its journal.
                ownership.event("pinch_scene_setup", source_row=row)

        profile = contract.backend_profile
        with profile_reader_scope(profile):
            scene_scope = nullcontext()
            if mesh_plan is None:
                from .tabletop_dual_hand_scene_preparation import prepare_scene
                from .tabletop_mesh_reset_order import prepare_mesh_reset_plan

                if prepared is None:
                    prepared = prepare_scene(contract.config.project)
                scene_scope = bindings((builder, "BACKEND", Path(profile.binary_path)),
                                       (builder, "BACKEND_SHA", profile.binary_sha256))
            with scene_scope:
                fresh = builder.build_fresh_owned(
                    sim, owners, project=contract.config.project, prepared=prepared, ownership=ownership,
                    record=record_setup, run_uuid=run_uuid, episode_uid=episode_uid,
                    physics_budget_end=contract.budget["max_sdk_steps"], expected_reset_seed=contract.reset_seed)
            if mesh_plan is None:
                sdf = sys.modules.get("ropesim.sdf")
                if sdf is None:
                    raise ValueError("actual loaded SDF module required to derive the Reset mesh plan")
                mesh_plan = prepare_mesh_reset_plan(contract.config.project, prepared, fresh.episode._rod,
                                                    sdf_module=sdf)
            return NativeEpisodeOwner(
                fresh.episode, contract, episode_uid=episode_uid, owner_resets=owner_resets,
                mesh_plan=mesh_plan, ledger=ledger, setup_events=fresh.setup_events,
                initial_geometry=fresh.initial_geometry)
    finally:
        sim = owners = prepared = ownership = ledger = owner_resets = fresh = None


def create_native_collector(config,factories):
    contract=verify_static_contract(config)
    owner=factories.episode(contract)
    try:
        reset=owner.reset(contract.reset_seed,episode_uid=owner.episode_uid,owner_resets=owner.owner_resets)
        live=verify_reset_and_sources(owner,reset,contract)
        model=factories.policy(live.model_contract)
        trainer=factories.trainer(model,contract.ppo_config)
        reward=factories.reward(live.reward_contract)
        return NativeTeacherCollector(owner,trainer,reward,live)
    except BaseException:
        try:
            owner.close()
        except BaseException as cleanup:
            try: owner.record_cleanup_failure(cleanup)
            except BaseException:
                # A broken diagnostic sink is never allowed to replace the
                # original source/model/reset exception on Python 3.10.
                pass
        raise


class SessionBudget:
    """One monotonic external budget clock for all episode owners in this session."""
    def __init__(self,limits):
        import time
        self.limits=dict(limits);self.started=time.monotonic()
        self.sdk_steps=self.solver_calls=self.internal_solver_substeps=self.resets=self.episodes=0
        self.scene=None;self.last_clock=None;self.setup_sdk_steps=0;self._setup_recorded=False
        self._parent_deadline_s = None

    def bind_parent_deadline(self, deadline_s):
        import math
        import time

        if type(deadline_s) not in (int, float) or not math.isfinite(deadline_s) or deadline_s <= time.monotonic():
            raise ValueError("fresh parent launch deadline required before episode construction")
        if self._parent_deadline_s is not None and self._parent_deadline_s != deadline_s:
            raise ValueError("parent launch deadline changed within the owned session")
        self._parent_deadline_s = deadline_s

    def _remaining_at(self, now):
        remaining = self.limits["wall_timeout_s"] - (now - self.started)
        if self._parent_deadline_s is not None:
            remaining = min(remaining, self._parent_deadline_s - now)
        return max(0., remaining)

    def remaining_s(self):
        import time

        return self._remaining_at(time.monotonic())

    def wall_readback(self):
        import time

        now = time.monotonic()
        return dict(started_s=self.started, observed_s=now, local_cap_s=self.limits["wall_timeout_s"],
                    parent_deadline_s=self._parent_deadline_s, remaining_s=self._remaining_at(now))

    def record_setup(self,owner,events):
        events=tuple(events)
        if self._setup_recorded:
            if events: raise ValueError("scene setup cannot be replayed on a later episode")
            return
        if not events: raise ValueError("first scene actual setup cost/readback required")
        from .tabletop_deformable_dual_evidence import _sdk_boundary
        first,last=events[0],events[-1]
        for event in events:_sdk_boundary(dict(sdk_step=event["sdk_step"],sdk_time_s=event["sdk_time_s"]),event["sdk_step"])
        if (first["sdk_step"]!=0 or last["sdk_step"]!=owner.sim.current_time_step_index
                or last["sdk_time_s"]!=owner.sim.current_time):
            raise ValueError("actual original scene setup absolute SDK boundaries required")
        self.setup_sdk_steps=last["sdk_step"]-first["sdk_step"]
        self.sdk_steps+=self.setup_sdk_steps;self._setup_recorded=True
        if self.sdk_steps>self.limits["max_sdk_steps"]:raise RuntimeError("scene setup exceeds SDK budget")

    def reserve_reset(self,owner):
        identity=(id(owner.sim),id(owner.rod),id(owner.rod.rope))
        if self.scene is not None and identity!=self.scene: raise ValueError("same borrowed scene/rod required")
        if (self.resets>=self.limits["max_resets"] or self.episodes>=self.limits["max_episode_count"]
                or self.remaining_s()<=0): raise RuntimeError("session Reset/episode/wall budget exhausted")
        self.scene=identity;self.resets+=1;self.episodes+=1

    def charge(self,row,solver_substeps):
        # Called for an already advanced raw exchange; preserve consumed cost
        # even if any following persistence/source validation fails.
        self.sdk_steps+=1;self.solver_calls+=1;self.internal_solver_substeps+=solver_substeps
        self.last_clock=deepcopy(row["rod"])

    def remaining_steps(self):
        return max(0,min(self.limits["max_sdk_steps"]-self.sdk_steps,
            self.limits["max_solver_calls"]-self.solver_calls,8 if self.remaining_s()>=8/480 else 0))


class NativeEpisodeOwner:
    """Session lease for the existing episode, never another scene/SDK owner.

    The original episode and all SDK/rod/App handles remain borrowed. close only
    disarms this lease and restores callbacks; outer R06 lifecycle releases native
    resources after clearing exception frames. Returning from close proves no exit.
    """
    def __init__(self,episode,contract,*,episode_uid,owner_resets,mesh_plan,ledger,setup_events,initial_geometry,previous_reset=None):
        from .tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode
        from .tabletop_native_teacher_action_history import NativeAction24HistoryOwner
        from .tabletop_native_teacher_sources import NativePinchSources
        if type(episode) is not DualHandPhysicalEpisode or episode._state != "new":
            raise ValueError("one fresh original physical episode over existing scene required")
        self.episode,self.contract=episode,contract
        self.sim,self.rod,self.bindings=episode._sim,episode._rod,episode._bindings
        self.backend_profile=contract.backend_profile
        self.episode_uid,self.mesh_plan,self.ledger=episode_uid,mesh_plan,ledger
        self._closed=False;self._restores=[]
        self._initial_geometry=initial_geometry
        self.owner_resets={}
        # The factory cannot return this lease if initialization fails. Protect
        # its first callback mutation and every later setup/source operation here,
        # rather than relying on create_native_collector to receive the owner.
        try:
            self.history=NativeAction24HistoryOwner(initial_action24=__import__("torch").zeros(1,24),
                seed_source="new original controller reset zero request",physics_budget_end=episode._end,
                solver_substeps=self.rod._base.cfg["rod"]["solver_substeps"],previous_reset=previous_reset)
            self.owner_resets=dict(owner_resets)
            self.owner_resets["controller"]=self.history.wrap_controller_reset(owner_resets["controller"])
            original=episode._record
            def record(row):
                ledger.emit("physical" if row["kind"]=="physical" else "owner_event",row)
                original(deepcopy(row));self.history.record(row)
            self._restores.append(lambda:setattr(episode,"_record",original))
            episode._record=record
            setup_events=tuple(setup_events)
            contract.runtime.record_setup(self,setup_events)
            for event in setup_events:
                if event.get("kind")!="setup": raise ValueError("original scene setup evidence required")
                ledger.emit("setup",event)
            self.sources=NativePinchSources(self,contract)
        except BaseException:
            try:
                self.close()
            except BaseException as cleanup:
                try:self.record_cleanup_failure(cleanup)
                except BaseException:pass
            raise

    def reset(self,seed,*,episode_uid,owner_resets):
        from . import tabletop_deformable_dual_hand_episode as module
        from .tabletop_mesh_reset_readback import MeasuredMeshReset
        from .tabletop_mesh_reset_numeric_state import numeric_reset_state_scope
        from .tabletop_deformable_clearance_guard import ClearanceGuardedExchange
        from .tabletop_native_teacher_profile import profile_reader_scope
        if self._closed: raise RuntimeError("closed owner lease")
        # Preserve the original reset API/physical owner; the scoped existing
        # successors add numeric/mesh readback and unchanged clearance guards.
        self.ledger.emit("reset_clock_preflight",reset_clock_preflight(self))
        self.contract.runtime.reserve_reset(self)
        old_reset,old_exchange=module.DualHandResetTransaction,module.FlexiblePhysicalExchange
        def reset_factory(seed,*,episode_uid):
            return MeasuredMeshReset(seed,episode_uid=episode_uid,mesh_plan=self.mesh_plan)
        with profile_reader_scope(self.backend_profile):
            try:
                module.DualHandResetTransaction=reset_factory
                module.FlexiblePhysicalExchange=ClearanceGuardedExchange
                with numeric_reset_state_scope(sim=self.sim,rod=self.rod,
                        record=lambda row:self.ledger.emit("numeric_reset",row)) as audit:
                    result=self.episode.reset(seed,episode_uid=episode_uid,owner_resets=owner_resets)
                self.numeric_audit=deepcopy(audit)
            finally:
                module.DualHandResetTransaction,module.FlexiblePhysicalExchange=old_reset,old_exchange
            for event in self._initial_geometry():
                if event.get("kind")!="initial_geometry": raise ValueError("original same-clock geometry evidence required")
                self.ledger.emit("initial_geometry",event);self.history.record(event)
            self.sources.after_reset(result)
        return result

    def live_reset_contract(self,reset):
        from .tabletop_native_teacher_preflight import LiveResetContract
        return LiveResetContract(self,self.sim,self.rod,self.bindings,reset["reset"],self.numeric_audit,
            self.episode.pre_action_state()["common"],self.sources.parameters,self.sources.reset_distance,self.mesh_plan)

    def initialize_context_and_reward_source(self,reset):
        self.sources.initialize_context(reset)

    def pre_action_state(self): return self.episode.pre_action_state()
    def control(self,action): return self.episode.control(action)

    def close(self):
        if self._closed: return
        self._closed=True
        errors=[]
        for restore in reversed(self._restores):
            try: restore()
            except BaseException as exc: errors.append(exc)
        self._restores=[]
        # These callbacks borrow original SDK/controller handles. Drop our copy
        # (including the wrapped controller closure), never mutate caller owners.
        self.owner_resets={}
        sources=getattr(self,"sources",None)
        if sources is not None:
            sources._error=None
            sources.owner=None
        self._initial_geometry=None
        # Drop borrowed references, but never call SDK STOP, rope destructor,
        # or App.close while the original exception may still own its frames.
        self.episode=self.sim=self.rod=self.bindings=None
        self.ledger.emit("owner_lease_closed",dict(native_resources_released=False,
            borrowed_references_dropped=True,app_closed=False))
        if errors: raise errors[0]

    def record_cleanup_failure(self,error):
        # Text only: do not retain another exception/traceback owning PhysX views.
        payload=dict(type=type(error).__name__,message=str(error))
        try:
            self.ledger.emit("cleanup_failure",payload)
        except BaseException:
            # Separate append-only fallback survives a closed/broken journal.
            # Failure of both sinks cannot replace the primary operation error.
            try:
                import os
                path=self.ledger.path.with_suffix(".cleanup.jsonl")
                with path.open("a",encoding="utf-8") as stream:
                    stream.write(json.dumps(payload,allow_nan=False)+"\n")
                    stream.flush();os.fsync(stream.fileno())
            except BaseException:
                pass
