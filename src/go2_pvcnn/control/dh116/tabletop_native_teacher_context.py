"""C02 context50 and copy-on-write task memory; no native qualification.

Source labels describe required provenance. They never certify a live owner.
C04 must connect and verify the actual physical owner before native admission.
"""
from dataclasses import dataclass, asdict
from copy import deepcopy
import weakref
import math
from types import MappingProxyType
import uuid

import numpy as np
import torch

from .tabletop_native_teacher_profile import CONTEXT_SCHEMA_VERSION, digest
from .tabletop_deformable_r0_replay import _digest
from .tabletop_teacher_mounted_wrist_source import _clock

CONTACT_PRELIFT = "approach_contact_prelift_v1"
MODES = ("approach_wiring_smoke_v1", "flexible_train_r1_full", CONTACT_PRELIFT)
CONTACT_PRELIFT_EXCLUDED = (7, 47)
SCOPES = ("cpu_fixture", "owner_source_unqualified")
SMOKE_EXCLUDED = (4, 5, 6, 7, 47)


def _schema():
    rows = []
    def add(name, unit, scale, source, applicability="required"):
        rows.append(MappingProxyType(dict(index=len(rows), name=name, unit=unit, scale=scale,
            source=source, validity="finite when valid; unknown NaN/false", applicability=applicability)))
    for phase in ("Approach", "Close", "Lift", "Hold"):
        add("phase_one_hot."+phase, "bool", 1., "physical_phase_state_machine")
    for side in ("L","R"):
        add("required_contact_fraction."+side, "1", 1., "qualified_dual_hand_contact", "full_only")
    add("bilateral_contact_duration", "s", .25, "qualified_contact_history", "full_only")
    add("continuous_hold_duration", "s", 5., "qualified_task_history", "full_only")
    for side in ("L","R"):
        add("fixed_material_anchor_station."+side, "m", "rest_length_m",
            "fixed_material_anchor_history", "anchored_only")
    for side in ("L","R"):
        add("fixed_anchor_valid."+side, "bool", 1., "fixed_material_anchor_history")
    for i in range(24):
        add("previous_normalized_request24."+str(i), "1", 1., "Action24HistoryOwner")
    for name in ("reset_median_node_z","reset_min_node_z"):
        add(name,"m",1.,"verified_reset_41_nodes")
    add("distance_ref","m",1.,"max_actual_reset_distance_0.005m")
    add("actual_reset_distance_mean","m",1.,"verified_reset_distance")
    for name,unit in (("lift_target","m"),("hold_target","s"),("slip_limit","m")):
        add(name,unit,1.,"success_contract_30mm_5s_0.5mm")
    add("episode_elapsed","s","explicit_episode_cap_s","physical_clock")
    add("external_time_remaining","s","explicit_external_cap_s","external_budget_clock")
    add("finite_task_deadline_enabled","bool",1.,"task_contract_no_finite_deadline")
    add("finite_task_time_remaining","s",1.,"task_contract_no_finite_deadline","not_applicable")
    add("prohibited_support_latched","bool",1.,"qualified_support_history","full_only")
    add("success_event_emitted","bool",1.,"task_event_history")
    add("trusted_failure_event_emitted","bool",1.,"task_event_history")
    return tuple(rows)


CONTEXT_FIELDS = _schema()
CONTEXT_SCHEMA_SHA256 = digest(dict(version=CONTEXT_SCHEMA_VERSION,
    fields=[dict(field) for field in CONTEXT_FIELDS]))


@dataclass(frozen=True, slots=True)
class ContextIdentity:
    run_uuid: str
    env_id: int
    episode_uid: str
    reset_id: int
    rod_uid: str
    physics_step: int
    rod_tick: int
    timestamp_s: float
    solver_time_s: float

    def __post_init__(self):
        for key in ("run_uuid","episode_uid","rod_uid"):
            if type(getattr(self,key)) is not str or not getattr(self,key).strip():
                raise ValueError("immutable nonempty context identity required")
        for key in ("env_id","reset_id","physics_step","rod_tick"):
            if type(getattr(self,key)) is not int or getattr(self,key) < 0:
                raise ValueError("explicit integer context identity required")
        for key in ("timestamp_s","solver_time_s"):
            value=getattr(self,key)
            if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                raise ValueError("finite context clock required")

    @classmethod
    def from_clock(cls, value):
        if not isinstance(value,dict):
            raise ValueError("complete physical or pre-action clock required")
        if "phase" in value:
            checked=_clock(value)
        else:
            from .tabletop_deformable_dual_evidence import _clock_row
            checked=_clock_row(value)
        return cls(**{name:checked[name] for name in cls.__dataclass_fields__})

    @property
    def epoch(self):
        return self.run_uuid,self.env_id,self.episode_uid,self.reset_id,self.rod_uid


@dataclass(frozen=True, slots=True)
class ContextMetadata:
    owner_uid: str
    identities: tuple
    backend_profile_sha256: str
    scope: str
    mode: str
    sources: tuple
    applicable: tuple
    reasons: tuple
    reference_uids: tuple
    previous_request_token: str
    grasp: str = "pinch"

    def __post_init__(self):
        if (type(self.identities) is not tuple or not self.identities
                or any(type(x) is not ContextIdentity for x in self.identities)
                or len({x.env_id for x in self.identities}) != len(self.identities)
                or self.scope not in SCOPES or self.mode not in MODES
                or self.grasp not in ("pinch","tripod","power")):
            raise ValueError("immutable complete context identities/scope required")
        if any(type(x) is not str or not x.strip() for x in
                (self.owner_uid,self.previous_request_token,self.backend_profile_sha256)):
            raise ValueError("context owner/profile/previous request token required")
        for value in (self.sources,self.applicable,self.reasons,self.reference_uids):
            if type(value) is not tuple:
                raise ValueError("immutable context metadata tuples required")
        if (len(self.sources) != 50 or len(self.applicable) != 50 or len(self.reasons) != 50
                or any(type(x) is not bool for x in self.applicable)
                or any(type(x) is not str or not x for x in self.sources)
                or any(x is not None and (type(x) is not str or not x.strip()) for x in self.reasons)
                or not self.reference_uids
                or any(type(x) is not tuple or len(x) != 2
                    or any(type(y) is not str or not y for y in x) for x in self.reference_uids)
                or len({x[0] for x in self.reference_uids}) != len(self.reference_uids)):
            raise ValueError("complete immutable context source/applicability/reference metadata required")


@dataclass(frozen=True, slots=True)
class TaskContextPacket:
    values: torch.Tensor
    valid: torch.Tensor
    metadata: ContextMetadata

    def owned(self):
        return TaskContextPacket(self.values.clone(), self.valid.clone(), self.metadata)

    @property
    def native_qualified(self):
        return False


def encode_context(values, valid):
    """Generic last-dimension mask representation; never modifies the raw values."""
    if (not isinstance(values,torch.Tensor) or not values.is_floating_point() or values.ndim < 1
            or not isinstance(valid,torch.Tensor) or valid.dtype != torch.bool
            or valid.shape != values.shape or valid.device != values.device
            or not bool(torch.isfinite(values[valid]).all())
            or not bool(torch.isnan(values[~valid]).all())):
        raise ValueError("context values/validity require finite valid and raw NaN unknown")
    return torch.cat((torch.where(valid,values,0.),valid.to(values.dtype)),dim=-1)


def validate_context_tensors(values, valid, *, mode=None):
    if (not isinstance(values,torch.Tensor) or values.dtype != torch.float32
            or values.ndim != 2 or values.shape[1] != 50 or not len(values)):
        raise ValueError("context requires float32[N,50]")
    encode_context(values,valid)
    # Required in either profile. Anchors are optional until fixed; full-only
    # sources remain explicit NA in smoke and are required by full.
    required=[i for i in range(50) if i not in (*SMOKE_EXCLUDED,8,9,46)]
    if not bool(valid[:,required].all()):
        raise ValueError("required applicable context source is unavailable")
    if bool(valid[:,46].any()) or not bool((values[:,45] == 0).all()):
        raise ValueError("finite task deadline is disabled by this task contract")
    for i in (0,1,2,3,10,11,45,48,49):
        if not bool(((values[:,i] == 0) | (values[:,i] == 1)).all()):
            raise ValueError("boolean context field differs")
    if not bool((values[:,:4].sum(-1) == 1).all()):
        raise ValueError("one actual physical phase required")
    if not bool((abs(values[:,12:36]) <= 1).all()):
        raise ValueError("previous normalized request outside action24")
    for j in (0,1):
        anchored=values[:,10+j] == 1
        if not torch.equal(valid[:,8+j],anchored):
            raise ValueError("anchor station validity differs from actual anchor state")
        if not bool(((values[anchored,8+j] >= 0) & (values[anchored,8+j] <= 1)).all()):
            raise ValueError("anchor material station outside rest geometry")
    expected=values.new_tensor([.03,5.,.0005])
    if (not torch.equal(values[:,40:43],expected.expand(len(values),3))
            or not bool((values[:,39] >= 0).all())
            or not torch.allclose(values[:,38],values[:,39].clamp_min(.005),rtol=0,atol=1e-8)
            or not bool((values[:,36] >= values[:,37]).all())
            or not bool((values[:,43:45] >= 0).all())
            or not bool((values[:,44] <= 1).all())):
        raise ValueError("fixed reset reference, success contract or explicit budget context differs")
    if mode == MODES[0]:
        if bool(valid[:,SMOKE_EXCLUDED].any()) or not bool((values[:,0] == 1).all()):
            raise ValueError("smoke context excludes contact/support and remains physical Approach")
    elif mode == MODES[1]:
        if not bool(valid[:,SMOKE_EXCLUDED].all()):
            raise ValueError("full context requires qualified contact/support/history sources")
    elif mode == CONTACT_PRELIFT:
        if (not bool(valid[:,[4,5,6]].all()) or bool(valid[:,CONTACT_PRELIFT_EXCLUDED].any())
                or not bool((values[:,1] == 1).all()) or bool(valid[:,8:10].any())
                or not bool((values[:,10:12] == 0).all()) or not bool((values[:,48] == 0).all())):
            raise ValueError("contact-only Close requires measured contact, no anchors/hold/support latch/success")
    for i in (4,5):
        if not bool(((values[valid[:,i],i] >= 0) & (values[valid[:,i],i] <= 1)).all()):
            raise ValueError("contact fraction outside [0,1]")
    if any(not bool((values[valid[:,i],i] >= 0).all()) for i in (6,7)):
        raise ValueError("continuous context durations must be nonnegative")
    if not bool(((values[valid[:,47],47] == 0) | (values[valid[:,47],47] == 1)).all()):
        raise ValueError("support latch must be boolean")


def validate_context(packet, *, backend_profile, identities=None):
    if type(packet) is not TaskContextPacket or type(packet.metadata) is not ContextMetadata:
        raise ValueError("typed context packet and immutable sidecar required")
    meta=packet.metadata
    if meta.backend_profile_sha256 != backend_profile.digest:
        raise ValueError("context backend profile differs")
    validate_context_tensors(packet.values,packet.valid,mode=meta.mode)
    if len(meta.identities) != len(packet.values):
        raise ValueError("context identity batch differs")
    if identities is not None and meta.identities != tuple(identities):
        raise ValueError("context time/episode identity differs")
    if meta.sources != tuple(x["source"] for x in CONTEXT_FIELDS):
        raise ValueError("required context source schema differs")
    applicable=list(meta.applicable)
    wanted=[True]*50
    wanted[46]=False
    for j in (0,1):
        # A batch with different anchor applicability must be packed separately.
        anchors=packet.valid[:,8+j]
        if not bool((anchors == anchors[0]).all()):
            raise ValueError("different anchor applicability needs separate context packet")
        wanted[8+j]=bool(anchors[0])
    if meta.mode == MODES[0]:
        for i in SMOKE_EXCLUDED: wanted[i]=False
    elif meta.mode == CONTACT_PRELIFT:
        for i in CONTACT_PRELIFT_EXCLUDED: wanted[i]=False
    if applicable != wanted:
        raise ValueError("context applicability differs from task/source profile")
    for i,applies in enumerate(applicable):
        if (applies and not bool(packet.valid[:,i].all())) or (not applies and (
                bool(packet.valid[:,i].any()) or not meta.reasons[i])):
            raise ValueError("required applicable unknown or missing explicit NA reason")
    return packet


def _packet_hash(packet):
    return _digest(dict(values=packet.values,valid=packet.valid,metadata=asdict(packet.metadata)))


from .tabletop_deformable_grasp_task import FlexibleGraspEpisode as _PhaseTask
_ORIGINAL_TASK_STEP = _PhaseTask.step
_ORIGINAL_TASK_RESET = _PhaseTask.reset


class _PhaseStepObserver:
    """Exclusive instance lease; observe one original step without changing its result."""
    def __init__(self, context, source):
        self.context=weakref.ref(context)
        self.source=weakref.ref(source)
        self.original_step=source.step

    def __call__(self,*args,**kwargs):
        original=self.original_step
        context=self.context()
        if original is None:
            raise ValueError("closed phase observer cannot advance physics")
        if context is None:
            self.close()
            return original(*args,**kwargs)
        try:
            result=original(*args,**kwargs)
            context._capture_phase_receipt(result)
            return result
        except BaseException:
            # Keep no exception/traceback or borrowed native owner on failure.
            context._phase_failed=True
            self.close()
            raise

    def close(self):
        original=self.original_step
        source=self.source()
        context=self.context()
        if source is not None and source.step is self:
            source.step=original
        self.original_step=None
        if context is not None:
            context._phase_source=None
        self.source=lambda:None


class TaskContextOwner:
    """One episode, with pre/post snapshots and an explicit bootstrap-before-commit boundary."""

    def __init__(self, *, backend_profile, mode, scope, episode_cap_s, external_cap_s, grasp="pinch"):
        if mode == CONTACT_PRELIFT:
            from .tabletop_native_teacher_contact_context import PrimedContactContextOwner
            if type(self) is not PrimedContactContextOwner:
                raise ValueError("event-bound PrimedContactContextOwner required for contact mode")
        if (grasp not in ("pinch","tripod","power") or mode not in MODES or scope not in SCOPES or any(type(v) not in (int,float)
                or not math.isfinite(v) or v <= 0 for v in (episode_cap_s,external_cap_s))):
            raise ValueError("explicit task profile/scope and finite episode/external caps required")
        self.profile,self.mode,self.scope=backend_profile,mode,scope
        self.grasp=grasp
        self.episode_cap_s,self.external_cap_s=float(episode_cap_s),float(external_cap_s)
        self.owner_uid=uuid.uuid4().hex
        self._current=self._pending=self._pre=None
        self._used_tokens=set()
        self._closed=self._phase_failed=False
        self._phase_lease=self._phase_source=self._phase_receipt=None

    @property
    def current(self):
        if self._current is None: raise ValueError("verified reset required")
        return self._current.owned()

    @property
    def pending_post(self):
        if self._pending is None: raise ValueError("physical post context required")
        return self._pending.owned()

    def close(self):
        """Release the full-phase lease and restore the exact original bound step."""
        self._closed=True
        lease=self._phase_lease
        if lease is not None:
            lease.close()
        self._phase_source=None
        self._phase_receipt=None

    def _phase_snapshot(self,source,identity):
        if (not isinstance(source,_PhaseTask)
                or getattr(source.reset,"__func__",None) is not _ORIGINAL_TASK_RESET
                or source.env_id != identity.env_id or source.grasp != self.grasp
                or getattr(source,"identity",None) != (identity.episode_uid,identity.reset_id,identity.rod_uid)
                or getattr(source,"clock",None) != (identity.physics_step,identity.rod_tick,identity.timestamp_s)
                or source.active is not True
                or self.scope != "cpu_fixture" and source.backend_sha256 != self.profile.binary_sha256):
            raise ValueError("original same-clock phase owner/episode/grasp required")
        return deepcopy(dict(identity=source.identity,clock=source.clock,grasp=source.grasp,
            phase=source.phase,hold_start=source.hold_start,hold_steps=source.hold_steps,hold_s=source.hold_s,
            previous_bilateral=source.previous_bilateral,material_anchor=source.material_anchor,
            slip_references=source.slip_references,prohibited_support_latched=source.prohibited_support_latched))

    def _reset_phase_receipt(self,source,identity):
        if not isinstance(source,_PhaseTask):
            raise ValueError("full context requires original phase owner")
        if getattr(source.step,"__func__",None) is not _ORIGINAL_TASK_STEP:
            raise ValueError("original phase step required; phase owner lease already bound or replaced")
        snapshot=self._phase_snapshot(source,identity)
        if (snapshot["phase"] != "approach" or snapshot["material_anchor"] is not None
                or snapshot["slip_references"] != {"left":{},"right":{}}
                or snapshot["hold_start"] is not None or snapshot["hold_steps"] != 0 or snapshot["hold_s"] != 0
                or snapshot["previous_bilateral"] or snapshot["prohibited_support_latched"]):
            raise ValueError("fresh Reset phase owner must be unanchored Approach with cleared history")
        return dict(kind="reset",state=snapshot)

    def _capture_phase_receipt(self,result):
        source=self._phase_source
        if source is None or self._phase_failed or self._closed:
            raise ValueError("active original phase owner lease required")
        identity=self._reset_identity
        stamp=ContextIdentity(identity.run_uuid,identity.env_id,identity.episode_uid,identity.reset_id,
            identity.rod_uid,result["physics_step"],result["rod_tick"],result["timestamp_s"],
            identity.solver_time_s)
        snapshot=self._phase_snapshot(source,stamp)
        if (source.step is not self._phase_lease or self._phase_lease.original_step.__func__ is not _ORIGINAL_TASK_STEP
                or result["env_id"] != identity.env_id or result["episode_uid"] != identity.episode_uid
                or result["reset_id"] != identity.reset_id or result["rod_uid"] != identity.rod_uid
                or result["grasp"] != self.grasp or result["phase"] != snapshot["phase"]
                or result["hold_s"] != snapshot["hold_s"] or result["hold_steps"] != snapshot["hold_steps"]):
            raise ValueError("original phase step return/owner receipt differs")
        self._phase_receipt=dict(kind="step",state=snapshot,
            result={key:deepcopy(result[key]) for key in ("env_id","episode_uid","reset_id","rod_uid",
                "physics_step","rod_tick","timestamp_s","grasp","phase","hold_s","hold_steps")})

    def _physical_phase_receipt(self,source,identity):
        if (source is None or source is not self._phase_source or self._phase_failed or self._closed
                or self._phase_lease is None or source.step is not self._phase_lease
                or self._phase_receipt is None or self._phase_receipt["kind"] != "step"):
            raise ValueError("owned original phase step receipt required")
        if self._phase_snapshot(source,identity) != self._phase_receipt["state"]:
            raise ValueError("phase owner changed after its actual physical receipt")
        return deepcopy(self._phase_receipt)

    def _history(self, owner):
        if self._closed:
            raise ValueError("closed task context owner cannot continue")
        from .tabletop_action24_history_owner import Action24HistoryOwner
        allowed=(Action24HistoryOwner,)
        try:
            from .tabletop_native_teacher_action_history import NativeAction24HistoryOwner
        except ModuleNotFoundError as error:
            if error.name != __package__+".tabletop_native_teacher_action_history":
                raise
        else:
            allowed+=(NativeAction24HistoryOwner,)
        if type(owner) not in allowed:
            raise ValueError("actual Action24HistoryOwner required; no guessed previous request")
        owner._check()
        if (self.scope != "cpu_fixture" and owner._backend != self.profile.binary_sha256):
            raise ValueError("live history/backend profile differs; native owner connection pending")
        return owner

    def _state(self,state):
        if (not isinstance(state,dict) or type(state.get("identity")) is not ContextIdentity
                or state.get("source_scope") != self.scope
                or state.get("backend_sha256") != self.profile.binary_sha256):
            raise ValueError("same-clock immutable source/profile state required")
        return state["identity"]

    def _build(self,identity,previous,token,state,*,anchors=None,references=None,phase_receipt=None):
        anchors=self._anchors if anchors is None else anchors
        references=self._references if references is None else references
        values=torch.full((1,50),torch.nan,dtype=torch.float32)
        valid=torch.zeros((1,50),dtype=torch.bool)
        def setv(indices,data):
            values[:,indices]=torch.as_tensor(data,dtype=torch.float32)
            valid[:,indices]=True
        phase="Close" if self.mode == CONTACT_PRELIFT else "Approach"
        if self.mode == MODES[1]:
            if phase_receipt is None:
                raise ValueError("owned original phase receipt required for full context")
            mapping={"approach":"Approach","grasp":"Close","lift":"Lift","hold":"Hold"}
            phase=mapping.get(phase_receipt["state"]["phase"])
            if (phase is None or "phase" in state and state["phase"] != phase
                    or "phase_source" in state and state["phase_source"] != "physical_phase_state_machine"):
                raise ValueError("caller phase differs from original physical phase receipt")
        setv(slice(0,4),[float(p == phase) for p in ("Approach","Close","Lift","Hold")])
        setv(slice(10,12),[float(x is not None) for x in anchors])
        for j,station in enumerate(anchors):
            if station is not None: setv(8+j,station/self._rest)
        setv(slice(12,36),previous)
        setv(slice(36,43),self._fixed)
        elapsed=identity.timestamp_s-self._reset_identity.timestamp_s
        setv(43,elapsed/self.episode_cap_s)
        external=state.get("external_budget")
        if (not isinstance(external,dict) or external.get("source") != "external_budget_clock"
                or external.get("identity") != identity or external.get("cap_s") != self.external_cap_s
                or type(external.get("remaining_s")) not in (int,float)
                or not math.isfinite(external["remaining_s"])
                or not 0 <= external["remaining_s"] <= self.external_cap_s):
            raise ValueError("explicit same-clock external budget source/cap required")
        if self._current is not None and external["remaining_s"]/self.external_cap_s > float(
                self._current.values[0,44])+1e-7:
            raise ValueError("external remaining budget cannot increase within an episode")
        setv(44,external["remaining_s"]/self.external_cap_s)
        setv(45,0.)
        if "success_event_emitted" in state or "trusted_failure_event_emitted" in state:
            raise ValueError("event booleans require same-clock task event history source")
        events=state.get("event_history")
        success,failure=self._success,self._failure
        if events is not None:
            if (not isinstance(events,dict) or events.get("source") != "task_event_history"
                    or events.get("identity") != identity or events.get("valid") is not True
                    or type(events.get("success_emitted")) is not bool
                    or type(events.get("trusted_failure_emitted")) is not bool):
                raise ValueError("same-clock task event history required")
            if (self._current is None and (events["success_emitted"] or events["trusted_failure_emitted"])
                    or success and not events["success_emitted"]
                    or failure and not events["trusted_failure_emitted"]):
                raise ValueError("task event history cannot leak across reset or unlatch")
            success,failure=events["success_emitted"],events["trusted_failure_emitted"]
        setv(48,float(success))
        setv(49,float(failure))
        if self.mode in (MODES[1], CONTACT_PRELIFT):
            fields=state.get("physical_fields",{})
            for i in (SMOKE_EXCLUDED if self.mode == MODES[1] else (4,5,6)):
                item=fields.get(i)
                if (not isinstance(item,dict) or item.get("source") != CONTEXT_FIELDS[i]["source"]
                        or item.get("identity") != identity or item.get("valid") is not True):
                    raise ValueError("full applicable contact/support source missing")
                raw=item.get("value")
                if type(raw) not in (int,float) or not math.isfinite(raw):
                    raise ValueError("finite physical source value required")
                if i == 7:
                    # The original task makes the first GOOD PHYSICAL endpoint zero.
                    # A later control endpoint may already contain complete hold intervals.
                    if raw != phase_receipt["state"]["hold_s"]:
                        raise ValueError("hold duration differs from original phase owner history")
                setv(i,raw/CONTEXT_FIELDS[i]["scale"])
        applicable=tuple(bool(valid[0,i]) for i in range(50))
        reasons=tuple(None if x else ("unanchored" if i in (8,9)
            else "finite task deadline disabled" if i == 46
            else "prelift_contact_only_excluded" if self.mode == CONTACT_PRELIFT else "smoke excluded")
            for i,x in enumerate(applicable))
        meta=ContextMetadata(self.owner_uid,(identity,),self.profile.digest,self.scope,self.mode,
            tuple(x["source"] for x in CONTEXT_FIELDS),applicable,reasons,references,token,self.grasp)
        return validate_context(TaskContextPacket(values,valid,meta),backend_profile=self.profile)

    def _freeze_reset_references(self,state,*,history_owner,phase_owner=None):
        history=self._history(history_owner)
        if self._current is not None or history._reset is None:
            raise ValueError("fresh context owner and actual controller reset receipt required")
        identity=self._state(state)
        if ContextIdentity.from_clock(history._reset["identity"]) != identity:
            raise ValueError("context reset and Action24HistoryOwner clock differ")
        nodes=state.get("node_positions")
        distance=state.get("distance_mean_m")
        rest=state.get("rest_length_m")
        reference=state.get("target_reference_uid")
        if (not isinstance(nodes,torch.Tensor) or nodes.shape != (1,41,3)
                or not bool(torch.isfinite(nodes).all()) or type(distance) not in (int,float)
                or not math.isfinite(distance) or distance < 0 or type(rest) not in (int,float)
                or not math.isfinite(rest) or rest <= 0 or type(reference) is not str or not reference):
            raise ValueError("verified 41 reset nodes, actual distance/rest and fixed reference required")
        anchors=state.get("anchor_stations_m",(None,None))
        anchor_uids=state.get("anchor_uids",(None,None))
        if (type(anchors) is not tuple or type(anchor_uids) is not tuple
                or len(anchors) != 2 or len(anchor_uids) != 2):
            raise ValueError("immutable fixed material anchor stations/UIDs required")
        if anchors != (None,None) or anchor_uids != (None,None):
            raise ValueError("fresh Reset must be unanchored; caller anchors are not a source")
        phase_receipt=self._reset_phase_receipt(phase_owner,identity) if self.mode == MODES[1] else None
        self._history_uid=history._owner_uid
        self._anchors,self._anchor_uids,self._rest=anchors,anchor_uids,rest
        self._references=(("target_material",reference),)+tuple(
            ("anchor."+str(i),uid) for i,uid in enumerate(anchor_uids) if uid is not None)
        z=nodes[0,:,2]
        self._fixed=[float(z.median()),float(z.min()),max(distance,.005),distance,.03,5.,.0005]
        self._reset_identity=identity
        self._success=self._failure=False
        return history,identity,phase_receipt

    def reset_from_verified_state(self,state,*,history_owner,phase_owner=None):
        history,identity,phase_receipt=self._freeze_reset_references(
            state,history_owner=history_owner,phase_owner=phase_owner)
        packet=self._build(identity,history._previous,_digest(history._reset),state,phase_receipt=phase_receipt)
        if self.mode == MODES[1]:
            self._phase_source=phase_owner
            self._phase_receipt=phase_receipt
            self._phase_lease=_PhaseStepObserver(self,phase_owner)
            phase_owner.step=self._phase_lease
        self._current=packet
        return self.current

    def freeze_pre_action(self,identity,*,history_owner):
        history=self._history(history_owner)
        if (type(identity) is not ContextIdentity or self._current is None or self._pre is not None
                or self._pending is not None or history._owner_uid != self._history_uid
                or history._captured is None
                or ContextIdentity.from_clock(history._captured["identity"]) != identity
                or self._current.metadata.identities != (identity,)
                or not np.array_equal(history._previous,self._current.values[:,12:36].numpy())):
            raise ValueError("same owned settled pre-action history/identity required")
        self._pre=self._current.owned()
        self._pre_previous_sha=history._previous_sha
        return self._pre.owned()

    def _anchor_state(self,source,identity):
        # Reuse the ORIGINAL physical task's first qualified bilateral _slip
        # history. A SourceStamp or dictionary boolean cannot establish it.
        from .tabletop_deformable_grasp_task import FlexibleGraspEpisode
        if (not isinstance(source,FlexibleGraspEpisode)
                or type(source)._slip is not FlexibleGraspEpisode._slip
                or source.env_id != identity.env_id or source.grasp != self.grasp
                or source.identity != (identity.episode_uid,identity.reset_id,identity.rod_uid)
                or source.clock != (identity.physics_step,identity.rod_tick,identity.timestamp_s)
                or source.material_anchor is None
                or self.scope != "cpu_fixture" and source.backend_sha256 != self.profile.binary_sha256
                or not math.isclose(float(source.rest_lengths.sum()),self._rest,abs_tol=1e-8,rel_tol=0)):
            raise ValueError("original same-clock material/tangential anchor owner required")
        if hasattr(self,"_anchor_source") and source is not self._anchor_source:
            raise ValueError("fixed anchor owner changed within the episode")
        stations,uncertainty=source.material_anchor
        if (len(stations) != 2 or len(uncertainty) != 2
                or any(type(v) not in (int,float,np.float64,np.float32)
                    or not math.isfinite(v) or not 0 <= v <= self._rest for v in stations)
                or any(not math.isfinite(u) or not 0 <= u <= .00025 for u in uncertainty)
                or abs(stations[0]-stations[1]) <= sum(uncertainty)):
            raise ValueError("first qualified original material anchor geometry required")
        refs=[x for x in self._references if not x[0].startswith(("anchor.","tangent."))]
        uids=[]
        for i,side in enumerate(("left","right")):
            owned=source.slip_references.get(side,{})
            for kind,label in (("material","anchor"),("tangential","tangent")):
                value=owned.get(kind)
                if (not isinstance(value,tuple) or len(value) != 2
                        or any(type(x) is not str or not x for x in value)):
                    raise ValueError("original fixed material/tangential reference sources required")
                refs.append((label+"."+str(i),value[0]))
                if kind == "material": uids.append(value[0])
        if uids[0] != uids[1]:
            raise ValueError("both material anchors must share one rest-arclength reference")
        return tuple(float(x) for x in stations),tuple(uids),tuple(refs)

    def advance_physical_endpoint(self,state,*,history_owner,collection_token,anchor_owner=None,phase_owner=None):
        history=self._history(history_owner)
        identity=self._state(state)
        settled=history._settled
        if (self._pre is None or self._pending is not None or history._owner_uid != self._history_uid
                or not isinstance(settled,dict) or collection_token in self._used_tokens
                or settled["policy"]["collection_token"] != collection_token
                or settled["previous_action_sha256"] != self._pre_previous_sha
                or ContextIdentity.from_clock(settled["endpoint"]) != identity):
            raise ValueError("actual accepted issued request and unused collection token required")
        prior=self._pre.metadata.identities[0]
        if (identity.epoch != prior.epoch or identity.physics_step <= prior.physics_step
                or identity.rod_tick <= prior.rod_tick or identity.timestamp_s <= prior.timestamp_s
                or identity.solver_time_s <= prior.solver_time_s):
            raise ValueError("physical endpoint must advance within the same episode")
        if (state.get("target_reference_uid") != dict(self._references)["target_material"]
                or state.get("rest_length_m") != self._rest
                or state.get("distance_mean_m") != self._fixed[3]):
            raise ValueError("episode fixed anchor/reference/dref changed")
        if not np.array_equal(settled["issued_action24"],history._previous):
            raise ValueError("previous issued request differs from history readback")
        phase_receipt=self._physical_phase_receipt(phase_owner,identity) if self.mode == MODES[1] else None
        if self.mode == MODES[1]:
            if anchor_owner is not None and anchor_owner is not phase_owner:
                raise ValueError("full phase and anchor must share the original physical task owner")
            if phase_owner.material_anchor is not None:
                anchor_owner=phase_owner
        anchors,uids,references=self._anchors,self._anchor_uids,self._references
        if anchor_owner is not None:
            proposed,proposed_uids,proposed_refs=self._anchor_state(anchor_owner,identity)
            for old,old_uid,new,new_uid in zip(anchors,uids,proposed,proposed_uids):
                if old is not None and (new != old or new_uid != old_uid):
                    raise ValueError("fixed material anchor cannot be reanchored after contact loss")
            anchors,uids,references=proposed,proposed_uids,proposed_refs
        if (state.get("anchor_stations_m",anchors) != anchors
                or state.get("anchor_uids",uids) != uids):
            raise ValueError("new anchor needs its actual owner; fixed anchors cannot change")
        self._pending=self._build(identity,history._previous,collection_token,state,
            anchors=anchors,references=references,phase_receipt=phase_receipt)
        self._pending_anchor_state=anchors,uids,references,anchor_owner
        self._pending_hash=_packet_hash(self._pending)
        self._used_tokens.add(collection_token)
        return self.pending_post

    def commit_post_context(self,packet):
        if self._pending is None or _packet_hash(packet) != self._pending_hash:
            raise ValueError("one unchanged owned post context required after next-value evaluation")
        self._current=self._pending.owned()
        self._anchors,self._anchor_uids,self._references,anchor_source=self._pending_anchor_state
        if anchor_source is not None: self._anchor_source=anchor_source
        self._success=bool(self._current.values[0,48])
        self._failure=bool(self._current.values[0,49])
        self._pending=self._pre=None
        return self.current
