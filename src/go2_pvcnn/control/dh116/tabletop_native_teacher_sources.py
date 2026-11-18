"""Pinch smoke adapter using the existing SDK, SDF, parameter and health readers.

It never imports a native extension, fabricates full contact/slip sources, or
changes the grasp string to borrow another style's geometry qualification.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch

from .tabletop_native_teacher_profile import profile_reader_scope
from .tabletop_training_reward import RewardEndpoint,SourceStamp,Measurement,SubmittedRequest,SourceFailure


@contextmanager
def _canonical_pinch_history():
    """Route the one locked Pinch receipt through the existing reader's canonical path."""
    from scripts import prepare_dh116_pinch_margin_contact as legacy
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

    original = legacy.read_locked
    def read(path, expected):
        path = Path(path)
        if path == legacy.OLD_PAIR and expected == legacy.OLD_PAIR_SHA:
            path = path.resolve(strict=True)
        return original(path, expected)

    with bindings((legacy, "read_locked", read)):
        yield


class NativePinchSources:
    scope="owner_source_unqualified"
    reward_scope="unqualified_source"

    def __init__(self,owner,contract,*,scope="owner_source_unqualified"):
        from .tabletop_bilateral_r0_approach import BilateralMeasuredPinchApproach
        from .tabletop_native_parameter_endpoints import NativeParameterEndpoints
        from .tabletop_native_teacher_policy_pinch_distance import PolicyOwnedPinchApproach
        if contract.config.preflight["style"]!="pinch":
            raise ValueError("independent style-specific distance source missing")
        if contract.config.preflight["profile"]!="approach_wiring_smoke_v1":
            raise ValueError("full_shared_contact_slip_hold_source_adapter_missing")
        if scope not in ("cpu_fixture","owner_source_unqualified"):raise ValueError("explicit unqualified source scope required")
        self.scope=scope;self.reward_scope="cpu_fixture" if scope=="cpu_fixture" else "unqualified_source"
        self.owner,self.contract=owner,contract
        # C01 uses the legacy reader only at the actual Reset (no schedule).
        # Policy intervals use the independent owned-history reader.
        with _canonical_pinch_history():
            self.reset_distance=BilateralMeasuredPinchApproach(contract.config.project)
        self.distance=PolicyOwnedPinchApproach(contract.config.project,scope=scope)
        # This reader is always instantiated under the reviewed current profile;
        # never borrow the historical 2fea packet/default sources.
        with profile_reader_scope(contract.backend_profile):
            self.reader=NativeParameterEndpoints(np.asarray(contract.reset_manifest.data()["rod_state"]["positions"],np.float32)[0],device=str(owner.bindings[0].hand.data.joint_pos.device))
        self.solver_substeps=owner.rod._base.cfg["rod"]["solver_substeps"]
        self._sdk_start=owner.sim.current_time_step_index
        self._error=None;self.parameters=None;self._last=None

    def _stamp(self,clock,source):
        return SourceStamp(clock["episode_uid"],clock["physics_step"],clock["solver_time_s"],
            self.reward_scope,source,env_id=clock["env_id"],reset_uid=str(clock["reset_id"]),rod_uid=clock["rod_uid"])

    def _endpoint(self,clock,distance,health,guard):
        values=distance["auxiliary"]["samples"]["approach_distance_m"]["value"]
        identity=distance["auxiliary"]["identity"]
        if values is None or any(identity.get(k)!=clock[k] for k in identity):
            raise SourceFailure("same-clock bilateral Pinch distance unavailable")
        if len(values)!=2 or not np.isfinite(values).all():
            raise SourceFailure("finite actual bilateral distance required")
        from .tabletop_r0_health_separation import _verified
        health=_verified(health)
        if (health.get("source_scope")!=("cpu_fixture" if self.scope=="cpu_fixture" else "native_getter") or health.get("source_instance_id")!=id(self.owner.rod.rope)
                or health.get("backend_sha256")!=self.contract.backend_profile.binary_sha256
                or health.get("getter")!="RopeBatch.last_error"):
            raise SourceFailure("actual same-owner native health getter required")
        if health.get("identity")!= {k:clock[k] for k in health["identity"]}:
            raise SourceFailure("same-clock actual native health source required")
        code=health["last_error"]
        if type(code) is not int: raise SourceFailure("actual native health getter unavailable")
        stamp=self._stamp(clock,"policy_owned_pinch_endpoint_v1")
        def measurement(value,source,unit="dimensionless"):
            return Measurement(value,unit,self._stamp(clock,source))
        fields=dict(distance=measurement(float(np.mean(values)),"PolicyOwnedPinchApproach.fixed_material41","m"),
            numerical_health=measurement(code==0,"RopeBatch.last_error.same_owner"),
            constraint_failure=measurement(guard is not None,"original_joint_force_clearance_and_policy_source_geometry_guards"),
            original_guard_joint_force_clearance_geometry=measurement(guard is not None,"original_joint_force_clearance_and_policy_source_geometry_guards"))
        # IDs are fixed material identities, not freshly selected nearest nodes.
        return RewardEndpoint(stamp,tuple(str(i) for i in range(41)),fields)

    def _health(self,clock):
        from .tabletop_r0_health_separation import capture_solver_health
        return capture_solver_health(self.owner.rod,backend=sys.modules["ropesim._ropesim"],
            sim=self.owner.sim,expected_clock=clock)

    def after_reset(self,result):
        owner=self.owner;receipt=result["reset"];clock=owner.rod.clock()
        self._raw=deepcopy(receipt["rod_reset"])
        self.parameters=self.reader.capture_reset(receipt["rod_reset"],material_reference_uid=self.owner.episode_uid+":material41",
            rope=owner.rod.rope,backend=sys.modules["ropesim._ropesim"],sim=owner.sim,read_clock=owner.rod.clock)
        d=self.reset_distance.capture_reset(receipt,self.parameters,grasp="pinch")
        h=self._health(clock)
        self._reset_timestamp_s=clock["timestamp_s"]
        self._reset_endpoint=self._last=self._endpoint(clock,d,h,None)
        self._reset_distance=self._last.fields["distance"].raw
        self._reset_nodes=torch.as_tensor(receipt["rod_reset"]["node_positions"],dtype=torch.float32).clone()
        self._rest=float(np.asarray(receipt["rod_reset"]["rest_lengths_m"]).sum())
        self._reference=self.owner.episode_uid+":fixed_material_nodes41"
        owner.ledger.emit("reset_sources",dict(parameters=self.parameters,distance=d,health=h,endpoint=self._last))
        self._install_exchange()

    def _install_exchange(self):
        owner=self.owner
        exchange=owner.episode._exchange;step=exchange._step
        def enriched(action,substep):
            row=step(action,substep)
            self._raw=deepcopy(row["rod"])
            # Failure is attached and delivered before being raised by observe.
            # Original physical state is already advanced; never lose that row.
            try:
                with profile_reader_scope(self.contract.backend_profile):
                    clock=owner.rod.clock()
                    self.parameters=self.reader.capture(row["rod"],rope=owner.rod.rope,
                        backend=sys.modules["ropesim._ropesim"],sim=owner.sim,read_clock=owner.rod.clock)
                    row["material_director"]=deepcopy(self.parameters)
                    row["original_guard_failure"]=row["guard_failure"]
                    row["closure_geometry"]=self.distance.geometry.capture(row,sim=owner.sim,
                        bindings=owner.bindings,read_clock=owner.rod.clock)
                    row["native_teacher_health"]=self._health(clock)
                    geometry=row["closure_geometry"]
                    if geometry["source_complete"] and geometry["failure_reasons"] and row["guard_failure"] is None:
                        row["guard_failure"]=geometry["terminal_reason"]
                    if row["native_teacher_health"]["last_error"] not in (0,None) and row["guard_failure"] is None:
                        row["guard_failure"]="native_numerical_health"
            except BaseException as exc:
                self._error=exc
                row["native_teacher_source_error"]=dict(type=type(exc).__name__,message=str(exc))
            return row
        exchange._step=enriched
        owner._restores.append(lambda:setattr(exchange,"_step",step))

    def initialize_context(self,result):
        from .tabletop_native_teacher_context import TaskContextOwner
        owner=self.owner;c=self.contract
        owner.context=TaskContextOwner(backend_profile=c.backend_profile,mode=c.reward_config.profile,
            scope=self.scope,episode_cap_s=c.config.episode_cap_s,external_cap_s=c.budget["wall_timeout_s"],grasp="pinch")
        owner.context.reset_from_verified_state(self.context_state(None),history_owner=owner.history)

    def reset_reward(self,reward):
        p=SubmittedRequest(self.owner.episode_uid+":reset", "controller_reset_zero",self._last.stamp,
            (0.,)*24,"normalized_submitted_command24_preprojection")
        reward.reset_from_verified_state(self._last,previous_request=p)

    def submitted_request(self,sample,decision):
        return SubmittedRequest(sample["collection_token"],decision["policy"]["model_weights_sha256"],
            self._last.stamp,tuple(float(v) for v in sample["actions"][0]),"normalized_submitted_command24_preprojection")

    def observe(self,event):
        if self._error is not None: raise self._error
        row=event["exchange"]
        with profile_reader_scope(self.contract.backend_profile):
            from .tabletop_deformable_r0_replay import _digest
            director=row["material_director"]
            if (director.get("source_instance_id")!=id(self.owner.rod.rope)
                    or _digest(director)!=_digest(self.parameters)):
                raise SourceFailure("current same-owner parameter getter packet differs")
            # This runs AFTER durable physical recording and history.record.
            # The independent reader binds the complete row to the actual issued
            # request/SDK targets, collection token and current history window.
            d=self.distance.capture(event,history_owner=self.owner.history,grasp="pinch")
            from .tabletop_deformable_clearance_guard import validate_post_step_clearance
            raw=dict(row,guard_failure=row["original_guard_failure"])
            checked=validate_post_step_clearance(raw)
            geometry=row["closure_geometry"]
            if not geometry["source_complete"]:
                raise SourceFailure("required actual geometry/distance source unavailable")
            guard=checked["terminal_reason"] or geometry.get("terminal_reason")
            self.owner.ledger.emit("policy_pinch_distance",d)
            self._last=self._endpoint(row["rod"],d,row["native_teacher_health"],guard)
        return self._last

    def _remaining_wall(self):
        return self.contract.runtime.remaining_s()

    def remaining_substeps(self):
        return min(self.contract.runtime.remaining_steps(),
            max(0,self.owner.episode._end-self.owner.sim.current_time_step_index))

    def external_truncated(self):
        now=self.owner.rod.clock()
        return (now["timestamp_s"]-self._reset_timestamp_s >= self.contract.config.episode_cap_s
                or self.remaining_substeps()<8)

    def context_state(self,packet):
        from .tabletop_native_teacher_context import ContextIdentity
        identity=ContextIdentity.from_clock(dict(self.owner.rod.clock(),policy_step=self.owner.episode._control_index,substep=0))
        return dict(identity=identity,source_scope=self.scope,backend_sha256=self.contract.backend_profile.binary_sha256,
            node_positions=self._reset_nodes.clone(),distance_mean_m=self._reset_distance,
            rest_length_m=self._rest,target_reference_uid=self._reference,
            external_budget=dict(source="external_budget_clock",identity=identity,
                cap_s=self.contract.budget["wall_timeout_s"],remaining_s=self._remaining_wall()),
            event_history=dict(source="task_event_history",identity=identity,valid=True,
                success_emitted=False,trusted_failure_emitted=bool(packet and packet.event<0)))

    def pack(self, model, pre, context):
        return self._pack(model, pre, context)[0]

    def pack_recorded(self, model, pre, context, *, purpose):
        """Bind the original pack call to an unqualified, durable source row."""
        if purpose not in ("decision", "bootstrap"):
            raise ValueError("decision or bootstrap pack purpose required")
        if self.owner.ledger.scope != self.scope or context.metadata.scope != self.scope:
            raise ValueError("source pack scope must match the owning journal and context")
        return self._pack(model, pre, context, purpose=purpose)

    def _pack(self, model, pre, context, *, purpose=None):
        from .tabletop_deformable_r0_replay import _copy, _digest

        profile = self.contract.backend_profile
        raw = deepcopy(self._raw)
        parameters = self.parameters
        if purpose is not None:
            buffers = _copy(dict(pre_action=pre, raw_rod=raw, parameters=parameters, context=asdict(context)))
        with profile_reader_scope(profile):
            observation = model.pack_native_inputs(pre, raw, parameters, context, backend_profile=profile)
        if purpose is None:
            return observation, None

        buffers["observation"] = _copy(observation)
        sequence = self.owner.ledger.emit("teacher_input_source", dict(
            purpose=purpose, clock=asdict(context.metadata.identities[0]),
            context_owner_uid=context.metadata.owner_uid, backend_profile_sha256=profile.digest,
            native_qualification=False, buffers=buffers,
            sha256={name: _digest(buffer) for name, buffer in buffers.items()}))
        return observation, sequence
