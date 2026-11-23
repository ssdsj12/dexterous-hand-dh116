"""Independent Pinch/Tripod smoke sources over the original SDK and physical readers.

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
from .tabletop_teacher_pose_goal import ERROR_FIELDS, measured_active_angles


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


class _NativeGeometrySources:
    STYLE = None
    PROFILE = "approach_wiring_smoke_v1"
    scope="owner_source_unqualified"
    reward_scope="unqualified_source"

    def __init__(self,owner,contract,*,scope="owner_source_unqualified"):
        from .tabletop_native_parameter_endpoints import NativeParameterEndpoints
        if self.STYLE not in ("pinch","tripod","power") or contract.config.preflight["style"]!=self.STYLE:
            raise ValueError("independent style-specific distance source missing")
        if contract.config.preflight["profile"]!=self.PROFILE:
            raise ValueError("full_shared_contact_slip_hold_source_adapter_missing")
        if scope not in ("cpu_fixture","owner_source_unqualified"):raise ValueError("explicit unqualified source scope required")
        self.scope=scope;self.reward_scope="cpu_fixture" if scope=="cpu_fixture" else "unqualified_source"
        self.owner,self.contract=owner,contract
        # Each concrete adapter supplies its own verified Reset and policy metric.
        # Parameter/history/health/packing code is shared; grasp qualification is not.
        self._configure_distances(owner,contract,scope)
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

    def _endpoint(self,clock,distance,health,guard,*,hands=None):
        values=distance["auxiliary"]["samples"]["approach_distance_m"]["value"]
        identity=distance["auxiliary"]["identity"]
        if values is None or any(identity.get(k)!=clock[k] for k in identity):
            raise SourceFailure("same-clock bilateral "+self.STYLE.title()+" distance unavailable")
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
        stamp=self._stamp(clock,"policy_owned_"+self.STYLE+"_endpoint_v1")
        def measurement(value,source,unit="dimensionless"):
            return Measurement(value,unit,self._stamp(clock,source))
        distance_source = ("NativePowerPoseApproach.fixed_material_station" if self.STYLE == "power"
                           else "PolicyOwned"+self.STYLE.title()+"Approach.fixed_material41")
        fields=dict(distance=measurement(float(np.mean(values)),distance_source,"m"),
            numerical_health=measurement(code==0,"RopeBatch.last_error.same_owner"),
            constraint_failure=measurement(guard is not None,"original_joint_force_clearance_and_policy_source_geometry_guards"),
            original_guard_joint_force_clearance_geometry=measurement(guard is not None,"original_joint_force_clearance_and_policy_source_geometry_guards"))
        goal = self.contract.reward_config.finger_goal
        if goal is not None:
            if goal.grasp != self.STYLE:
                raise SourceFailure("finger goal grasp differs from the live owner")
            try:
                errors = goal.max_errors_rad(measured_active_angles(hands))
            except (KeyError, TypeError, ValueError) as error:
                raise SourceFailure("actual same-step finger goal readback required") from error
            for name, value in zip(ERROR_FIELDS, errors):
                fields[name] = measurement(value, "sdk_named_active_q", "rad")
        # IDs are fixed material identities, not freshly selected nearest nodes.
        return RewardEndpoint(stamp,tuple(str(i) for i in range(41)),fields)

    def _health(self,clock):
        from .tabletop_native_teacher_solver_health import capture_teacher_solver_health
        return capture_teacher_solver_health(self.owner.rod,backend=sys.modules["ropesim._ropesim"],
            sim=self.owner.sim,expected_clock=clock,grasp=self.STYLE)

    def after_reset(self,result):
        owner=self.owner;receipt=result["reset"];clock=owner.rod.clock()
        self._raw=deepcopy(receipt["rod_reset"])
        self.parameters=self.reader.capture_reset(receipt["rod_reset"],material_reference_uid=self.owner.episode_uid+":material41",
            rope=owner.rod.rope,backend=sys.modules["ropesim._ropesim"],sim=owner.sim,read_clock=owner.rod.clock)
        d=self.reset_distance.capture_reset(receipt,self.parameters,grasp=self.STYLE)
        h=self._health(clock)
        self._reset_timestamp_s=clock["timestamp_s"]
        self._reset_endpoint=self._last=self._endpoint(clock,d,h,None,hands=receipt["hands"] if self.contract.reward_config.finger_goal is not None else None)
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
                    row["closure_geometry"]=self._capture_geometry(row,owner)
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

    def _capture_geometry(self,row,owner):
        return self.distance.geometry.capture(row,sim=owner.sim,
            bindings=owner.bindings,read_clock=owner.rod.clock)

    def initialize_context(self,result):
        from .tabletop_native_teacher_context import TaskContextOwner
        owner=self.owner;c=self.contract
        owner.context=TaskContextOwner(backend_profile=c.backend_profile,mode=c.reward_config.profile,
            scope=self.scope,episode_cap_s=c.config.episode_cap_s,external_cap_s=c.budget["wall_timeout_s"],grasp=self.STYLE)
        owner.context.reset_from_verified_state(self.context_state(None),history_owner=owner.history)

    def reset_reward(self,reward):
        p=SubmittedRequest(self.owner.episode_uid+":reset", "controller_reset_zero",self._last.stamp,
            (0.,)*24,"normalized_submitted_command24_preprojection")
        reward.reset_from_verified_state(self._last,previous_request=p)

    def submitted_request(self,sample,decision):
        return SubmittedRequest(sample["collection_token"],decision["policy"]["model_weights_sha256"],
            self._last.stamp,tuple(float(v) for v in sample["actions"][0]),"normalized_submitted_command24_preprojection")

    def _validate_parameter_event(self, event):
        if self._error is not None:
            raise self._error
        from .tabletop_deformable_r0_replay import _digest
        director = event["exchange"]["material_director"]
        if (director.get("source_instance_id") != id(self.owner.rod.rope)
                or _digest(director) != _digest(self.parameters)):
            raise SourceFailure("current same-owner parameter getter packet differs")

    def observe(self,event):
        self._validate_parameter_event(event)
        row=event["exchange"]
        with profile_reader_scope(self.contract.backend_profile):
            # This runs AFTER durable physical recording and history.record.
            # The independent reader binds the complete row to the actual issued
            # request/SDK targets, collection token and current history window.
            d=self.distance.capture(event,history_owner=self.owner.history,grasp=self.STYLE)
            from .tabletop_deformable_clearance_guard import validate_post_step_clearance
            raw=dict(row,guard_failure=row["original_guard_failure"])
            checked=validate_post_step_clearance(raw)
            geometry=row["closure_geometry"]
            if not geometry["source_complete"]:
                raise SourceFailure("required actual geometry/distance source unavailable")
            guard=checked["terminal_reason"] or geometry.get("terminal_reason")
            self.owner.ledger.emit("policy_"+self.STYLE+"_distance",d)
            hands = None
            if self.contract.reward_config.finger_goal is not None:
                post = row["action_readback"]["post"]
                clock = row["rod"]
                keys = ("episode_uid", "env_id", "physics_step", "reset_id", "rod_tick", "rod_uid")
                if len(post["identity"]) != 1 or any(post["identity"][0].get(key) != clock[key] for key in keys):
                    raise SourceFailure("finger goal SDK/rod clock differs")
                hands = post["hands"]
            self._last=self._endpoint(row["rod"],d,row["native_teacher_health"],guard,hands=hands)
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


class NativePinchSources(_NativeGeometrySources):
    """Original bilateral two-pad source and policy-owned history contract."""
    STYLE = "pinch"

    def _configure_distances(self,owner,contract,scope):
        from .tabletop_bilateral_r0_approach import BilateralMeasuredPinchApproach
        from .tabletop_native_teacher_policy_pinch_distance import PolicyOwnedPinchApproach
        with _canonical_pinch_history():
            self.reset_distance=BilateralMeasuredPinchApproach(contract.config.project)
        self.distance=PolicyOwnedPinchApproach(contract.config.project,scope=scope)


class NativeTripodSources(_NativeGeometrySources):
    """Independent six-pad metric over the authentic17-slot supported task."""
    STYLE = "tripod"

    def _configure_distances(self,owner,contract,scope):
        from .tabletop_tripod_reset_manifest import TripodResetManifest, SCOPE
        from .tabletop_tripod_reset_distance import MeasuredTripodResetApproach
        from .tabletop_native_teacher_policy_tripod_distance import PolicyOwnedTripodApproach
        from .tabletop_power_shared_rod import PowerSharedRodRuntime, PROXIMAL_POWER_MESH_NAMES
        from .tabletop_deformable_r0_replay import encode_owned
        from .tabletop_native_teacher_profile import digest
        manifest=contract.reset_manifest
        if type(manifest) is not TripodResetManifest:
            raise ValueError("source-derived independent Tripod Reset manifest required")
        data=manifest.data()
        if (data["style"]!="tripod" or data["scope"]!=SCOPE
                or tuple(owner.rod.mesh_names)!=PROXIMAL_POWER_MESH_NAMES
                or digest(encode_owned(owner.rod.bindings))!=data["mesh_bindings_sha256"]):
            raise ValueError("actual Tripod17 task and same-side mesh sources differ")
        if scope=="owner_source_unqualified":
            if type(owner.rod) is not PowerSharedRodRuntime:
                raise ValueError("actual single supported17 runtime required for native Tripod")
            manifest.verify(project=contract.config.project)
        task=data["task_config"]
        self.reset_distance=MeasuredTripodResetApproach(contract.config.project,scope=scope)
        self.distance=PolicyOwnedTripodApproach(contract.config.project,scope=scope,
            task_config=Path(task["path"]),task_config_sha256=task["sha256"],
            mesh_bindings=owner.rod.bindings)


class NativePowerSources(_NativeGeometrySources):
    """Power-only pose diagnostic over the actual supported scene."""
    STYLE = "power"

    def _configure_distances(self,owner,contract,scope):
        from .tabletop_native_teacher_power_distance import NativePowerPoseApproach
        from .tabletop_power_shared_rod import PowerSharedRodRuntime, PROXIMAL_POWER_MESH_NAMES

        goal = contract.reward_config.finger_goal
        if goal is None or goal.grasp != self.STYLE:
            raise ValueError("Power finger goal required for the pose diagnostic")
        if (tuple(owner.rod.mesh_names) != PROXIMAL_POWER_MESH_NAMES
                or scope == "owner_source_unqualified" and type(owner.rod) is not PowerSharedRodRuntime):
            raise ValueError("actual supported17 Power runtime required")
        self.reset_distance = self.distance = NativePowerPoseApproach(
            contract.config.project,contract.reset_manifest,scope=scope)

    def _capture_geometry(self,row,owner):
        return self.distance.capture_geometry(row,sim=owner.sim,
            bindings=owner.bindings,read_clock=owner.rod.clock)


class NativeContactTripodSources(NativeTripodSources):
    """Actual supported17 contact course; old smoke and other grasps stay independent."""
    PROFILE = "approach_contact_prelift_v1"

    def _install_exchange(self):
        from .tabletop_native_teacher_contact_measurement import capture_supported_contact
        super()._install_exchange()
        exchange = self.owner.episode._exchange
        step = exchange._step
        def measured(action, substep):
            row = step(action, substep)
            try:
                contact = capture_supported_contact(row["rod"], grasp=self.STYLE)
                row["teacher_contact_guard"] = dict(
                    identity=contact["identity"], raw_rod_sha256=contact["raw_rod_sha256"],
                    table_contact_failure=contact["table_contact_failure"],
                    support_load_n=contact["support_load_n"])
                if contact["table_contact_failure"] and row["guard_failure"] is None:
                    row["guard_failure"] = "native_teacher_table_contact"
            except BaseException as error:
                self._error = error
                row["native_teacher_source_error"] = dict(type=type(error).__name__, message=str(error))
            return row
        exchange._step = measured
        self.owner._restores.append(lambda: setattr(exchange,"_step",step))

    def _endpoint(self, clock, distance, health, guard):
        from dataclasses import replace
        endpoint = super()._endpoint(clock, distance, health, guard)
        if not getattr(self,"_contact_ready",False):
            return endpoint  # Real Reset2 remains unprimed; never invent contact.
        contact = self.owner.context.contact_readback
        identity = self.owner.context._contact_identity
        if any(clock[k] != getattr(identity,k) for k in identity.__dataclass_fields__):
            raise SourceFailure("same-clock original contact Context endpoint required")
        fields = dict(endpoint.fields)
        for name in ("coverage_left","coverage_right","opposed_left","opposed_right",
                     "bilateral_support","table_contact_failure"):
            fields[name] = Measurement(contact[name],"dimensionless",self._stamp(clock,contact["source"]))
        pads = tuple((side,tuple(contact["evidence"]["per_hand"][side]["required_pads"]))
                     for side in ("left","right"))
        return replace(endpoint,fields=fields,required_pad_ids=pads)

    def initialize_context(self, result):
        from .tabletop_native_teacher_contact_context import PrimedContactContextOwner
        from .tabletop_native_teacher_source_priming import prime_supported_sources
        from .tabletop_native_teacher_profile import profile_reader_scope
        owner, contract = self.owner, self.contract
        owner.context = PrimedContactContextOwner(backend_profile=contract.backend_profile,
            scope=self.scope,episode_cap_s=contract.config.episode_cap_s,
            external_cap_s=contract.budget["wall_timeout_s"],grasp=self.STYLE)
        owner.context.reset_pending_from_verified_state(self.context_state(None),history_owner=owner.history)
        try:
            with profile_reader_scope(contract.backend_profile):
                receipt = prime_supported_sources(owner.episode,history=owner.history,
                    runtime=contract.runtime,journal=owner.ledger)
                if self._error is not None:
                    raise self._error
                event = owner.history.priming_event
                self._validate_parameter_event(event)
                distance = self.distance.capture_priming(event,history_owner=owner.history,grasp=self.STYLE)
                owner.context.commit_primed_state(self.context_state(None),history_owner=owner.history,priming_event=event)
                self._contact_ready = True
                self._last = self._endpoint(event["exchange"]["rod"],distance,event["exchange"]["native_teacher_health"],None)
                owner.ledger.emit("primed_sources",dict(receipt=receipt,parameters=self.parameters,distance=distance,
                    contact=owner.context.contact_readback,endpoint=self._last,
                    reset_distance_ref_m=owner.context._fixed[2]))
        except BaseException:
            owner.history._failed = True
            owner.context._contact_failed = True
            contract.runtime.source_priming_failed = True
            raise

    def reset_reward(self, reward):
        receipt = self.owner.history.priming
        request = SubmittedRequest(receipt["priming_row_sha256"],"source_priming_no_policy",
            self._last.stamp,(0.,)*24,"normalized_submitted_command24_preprojection")
        reward.reset_from_verified_state(self._last,previous_request=request)

    def observe(self, event):
        self._validate_parameter_event(event)
        self.owner.context.observe_physical_event(event,history_owner=self.owner.history)
        return super().observe(event)
