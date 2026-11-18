"""Independent fixed-S0 context51 and owned CPU task memory.

No live W4 SourceAdapterV2 exists in this version. Production construction is
rejected; fixture phase/anchor inputs never establish physical qualification.
The old context50, physical task and checkpoint namespaces are unchanged.
"""
from dataclasses import dataclass, asdict
from copy import deepcopy
import math
from types import MappingProxyType
import uuid
import hashlib
from pathlib import Path

import numpy as np
import torch

from .tabletop_native_teacher_profile import digest
from .tabletop_native_teacher_context import ContextIdentity, CONTEXT_FIELDS
from .tabletop_native_phase_contract_v2 import PhaseOwnerV2, PhaseReceiptV2, PHASES_V2
from . import tabletop_native_phase_contract_v2 as _phase_module
from .tabletop_deformable_r0_replay import _digest

MODES_V2 = ("fixed_s0_v2",)
SCOPES = ("cpu_fixture", "owner_source_unqualified")
PHYSICAL_FIELDS_V2 = (4, 5, 6, 7, 47)
CONTEXT_SCHEMA_VERSION_V2 = "dh116.fixed_s0.context51.v2"
PHASE_CONTRACT_SHA256_V2 = hashlib.sha256(Path(_phase_module.__file__).read_bytes()).hexdigest()


def _schema_v2():
    rows = [dict(x) for x in CONTEXT_FIELDS]
    for i, phase in enumerate(PHASES_V2):
        rows[i].update(name="execution_phase_one_hot."+phase,
                       source="execution_phase_owner_v2")
    rows.append(dict(index=50, name="prelift_stable_duration", unit="s",
        scale="explicit_prelift_scale_s", source="execution_phase_owner_v2",
        validity="finite when valid; unknown NaN/false", applicability="required"))
    return tuple(MappingProxyType(row) for row in rows)


CONTEXT_FIELDS_V2 = _schema_v2()
CONTEXT_SCHEMA_SHA256_V2 = digest(dict(version=CONTEXT_SCHEMA_VERSION_V2,
    phase_contract_sha256=PHASE_CONTRACT_SHA256_V2,
    fields=[dict(field) for field in CONTEXT_FIELDS_V2]))


def _sha(value):
    return type(value) is str and len(value) == 64 and all(x in "0123456789abcdef" for x in value)


@dataclass(frozen=True, slots=True)
class ContextMetadataV2:
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
    phase_contract_sha256: str
    fixed_s0_manifest_sha256: str
    prelift_scale_s: float
    grasp: str = "pinch"

    def __post_init__(self):
        if (self.phase_contract_sha256 != PHASE_CONTRACT_SHA256_V2
                or not _sha(self.fixed_s0_manifest_sha256)
                or type(self.prelift_scale_s) not in (int, float)
                or not math.isfinite(self.prelift_scale_s) or self.prelift_scale_s <= 0):
            raise ValueError("context v2 phase/S0/positive prelift scale identity required")
        if (type(self.identities) is not tuple or not self.identities
                or any(type(x) is not ContextIdentity for x in self.identities)
                or len({x.env_id for x in self.identities}) != len(self.identities)
                or self.scope not in SCOPES or self.mode not in MODES_V2
                or self.grasp not in ("pinch","tripod","power")):
            raise ValueError("immutable complete context identities/scope required")
        if any(type(x) is not str or not x.strip() for x in
                (self.owner_uid,self.previous_request_token,self.backend_profile_sha256)):
            raise ValueError("context owner/profile/previous request token required")
        for value in (self.sources,self.applicable,self.reasons,self.reference_uids):
            if type(value) is not tuple:
                raise ValueError("immutable context metadata tuples required")
        if (len(self.sources) != 51 or len(self.applicable) != 51 or len(self.reasons) != 51
                or any(type(x) is not bool for x in self.applicable)
                or any(type(x) is not str or not x for x in self.sources)
                or any(x is not None and (type(x) is not str or not x.strip()) for x in self.reasons)
                or not self.reference_uids
                or any(type(x) is not tuple or len(x) != 2
                    or any(type(y) is not str or not y for y in x) for x in self.reference_uids)
                or len({x[0] for x in self.reference_uids}) != len(self.reference_uids)):
            raise ValueError("complete immutable context source/applicability/reference metadata required")


@dataclass(frozen=True, slots=True)
class TaskContextPacketV2:
    values: torch.Tensor
    valid: torch.Tensor
    metadata: ContextMetadataV2

    def owned(self):
        return TaskContextPacketV2(self.values.clone(), self.valid.clone(), deepcopy(self.metadata))

    @property
    def native_qualified(self):
        return False


def encode_context_v2(values, valid):
    """Encode exactly context51 as 51 masked values plus 51 validity bits."""
    if (not isinstance(values, torch.Tensor) or values.dtype != torch.float32
            or values.ndim != 2 or values.shape[1] != 51 or not len(values)):
        raise ValueError("context v2 encoder requires float32[N,51]")
    if (not isinstance(values,torch.Tensor) or not values.is_floating_point() or values.ndim < 1
            or not isinstance(valid,torch.Tensor) or valid.dtype != torch.bool
            or valid.shape != values.shape or valid.device != values.device
            or not bool(torch.isfinite(values[valid]).all())
            or not bool(torch.isnan(values[~valid]).all())):
        raise ValueError("context values/validity require finite valid and raw NaN unknown")
    return torch.cat((torch.where(valid,values,0.),valid.to(values.dtype)),dim=-1)


def validate_context_tensors_v2(values, valid, *, mode=None):
    if (not isinstance(values,torch.Tensor) or values.dtype != torch.float32
            or values.ndim != 2 or values.shape[1] != 51 or not len(values)):
        raise ValueError("context requires float32[N,51]")
    encode_context_v2(values,valid)
    # Independent v2 always requires measured contact/history fields; only
    # unestablished material anchors and the disabled deadline are NA.
    required=[i for i in range(51) if i not in (8,9,46)]
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
    if mode not in (None, *MODES_V2):
        raise ValueError("independent fixed-S0 v2 context mode required")
    for i in (4,5):
        if not bool(((values[valid[:,i],i] >= 0) & (values[valid[:,i],i] <= 1)).all()):
            raise ValueError("contact fraction outside [0,1]")
    if any(not bool((values[valid[:,i],i] >= 0).all()) for i in (6,7,50)):
        raise ValueError("continuous context durations must be nonnegative")
    if not bool(((values[valid[:,47],47] == 0) | (values[valid[:,47],47] == 1)).all()):
        raise ValueError("support latch must be boolean")


def validate_context_v2(packet, *, backend_profile, identities=None):
    if type(packet) is not TaskContextPacketV2 or type(packet.metadata) is not ContextMetadataV2:
        raise ValueError("typed context packet and immutable sidecar required")
    meta=packet.metadata
    meta.__post_init__()
    if meta.scope != "cpu_fixture":
        raise ValueError("full_v2_phase_source_adapter_missing")
    if meta.backend_profile_sha256 != backend_profile.digest:
        raise ValueError("context backend profile differs")
    validate_context_tensors_v2(packet.values,packet.valid,mode=meta.mode)
    if len(meta.identities) != len(packet.values):
        raise ValueError("context identity batch differs")
    if identities is not None and meta.identities != tuple(identities):
        raise ValueError("context time/episode identity differs")
    if meta.sources != tuple(x["source"] for x in CONTEXT_FIELDS_V2):
        raise ValueError("required context source schema differs")
    applicable=list(meta.applicable)
    wanted=[True]*51
    wanted[46]=False
    for j in (0,1):
        # A batch with different anchor applicability must be packed separately.
        anchors=packet.valid[:,8+j]
        if not bool((anchors == anchors[0]).all()):
            raise ValueError("different anchor applicability needs separate context packet")
        wanted[8+j]=bool(anchors[0])
    if applicable != wanted:
        raise ValueError("context applicability differs from task/source profile")
    for i,applies in enumerate(applicable):
        if (applies and not bool(packet.valid[:,i].all())) or (not applies and (
                bool(packet.valid[:,i].any()) or not meta.reasons[i])):
            raise ValueError("required applicable unknown or missing explicit NA reason")
    return packet


def _packet_hash(packet):
    return _digest(dict(values=packet.values,valid=packet.valid,metadata=asdict(packet.metadata)))


class TaskContextOwnerV2:
    """One episode, with pre/post snapshots and an explicit bootstrap-before-commit boundary."""

    def __init__(self, *, backend_profile, mode, scope, episode_cap_s, external_cap_s,
                 prelift_scale_s, grasp="pinch", source_adapter=None):
        if (grasp not in ("pinch","tripod","power") or mode not in MODES_V2 or scope not in SCOPES or any(type(v) not in (int,float)
                or not math.isfinite(v) or v <= 0 for v in (episode_cap_s,external_cap_s))):
            raise ValueError("explicit task profile/scope and finite episode/external caps required")
        if (type(prelift_scale_s) not in (int, float)
                or not math.isfinite(prelift_scale_s) or prelift_scale_s <= 0):
            raise ValueError("positive finite prelift_scale_s required")
        if scope != "cpu_fixture":
            # W4 has not supplied an actual source adapter, its source SHA, nor
            # its same-owner snapshot witness. A same-named caller object is not it.
            raise ValueError("full_v2_phase_source_adapter_missing")
        if source_adapter is not None:
            raise ValueError("CPU fixture cannot bind or impersonate native SourceAdapterV2")
        self.prelift_scale_s = float(prelift_scale_s)
        self.profile,self.mode,self.scope=backend_profile,mode,scope
        self.grasp=grasp
        self.episode_cap_s,self.external_cap_s=float(episode_cap_s),float(external_cap_s)
        self.owner_uid=uuid.uuid4().hex
        self._current=self._pending=self._pre=None
        self._used_tokens=set()
        self._closed=False
        self._phase_source=self._phase_receipt=None
        self._phase_seen_ref=self._phase_seen_value=None

    @property
    def current(self):
        if self._current is None: raise ValueError("verified reset required")
        return self._current.owned()

    @property
    def pending_post(self):
        if self._pending is None: raise ValueError("physical post context required")
        return self._pending.owned()

    def close(self):
        """Drop borrowed CPU sources; no task.step monkeypatch or native release."""
        self._closed=True
        self._phase_source=self._phase_receipt=None
        self._phase_seen_ref=self._phase_seen_value=None
        self._anchor_source=None
        self._pending_anchor_state=None
        self._pending=self._pre=None

    def _phase_snapshot(self, source, identity, *, reset=False):
        if (type(source) is not PhaseOwnerV2 or self._closed
                or not reset and source is not self._phase_source):
            raise ValueError("same exact v2 phase owner required")
        if self._phase_seen_ref is not None and asdict(self._phase_seen_ref) != self._phase_seen_value:
            raise ValueError("previously owned phase receipt mutated")
        receipt=source.current_receipt
        if (type(receipt) is not PhaseReceiptV2
                or receipt.episode_uid != identity.episode_uid
                or receipt.physics_tick != identity.physics_step
                or receipt.phase not in PHASES_V2
                or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0
                    for v in (receipt.prelift_stable_s,receipt.final_hold_s))
                or type(receipt.success_candidate) is not bool):
            raise ValueError("same episode/tick and finite v2 phase receipt required")
        if reset and (receipt.phase != "Close" or receipt.prelift_stable_s != 0
                or receipt.final_hold_s != 0 or receipt.success_candidate):
            raise ValueError("fresh Reset phase must be Close with both clocks clear")
        if receipt.phase != "LiftHold" and (receipt.final_hold_s != 0 or receipt.success_candidate):
            raise ValueError("final hold clock only belongs to LiftHold")
        if receipt.phase == "Close" and receipt.prelift_stable_s != 0:
            raise ValueError("Close must clear prelift clock")
        if receipt.success_candidate != (receipt.phase == "LiftHold" and receipt.final_hold_s >= 5):
            raise ValueError("phase success candidate clock differs")
        # CPU receipt identity is not physical provenance. W4 must witness the
        # original producer in production; the constructor currently rejects it.
        return deepcopy(receipt)

    def _remember_phase(self, source, receipt):
        self._phase_source=source
        self._phase_receipt=receipt
        self._phase_seen_ref=source.current_receipt
        self._phase_seen_value=asdict(source.current_receipt)

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
        values=torch.full((1,51),torch.nan,dtype=torch.float32)
        valid=torch.zeros((1,51),dtype=torch.bool)
        def setv(indices,data):
            values[:,indices]=torch.as_tensor(data,dtype=torch.float32)
            valid[:,indices]=True
        if phase_receipt is None:
            raise ValueError("owned independent v2 phase receipt required")
        phase=phase_receipt.phase
        if ("phase" in state and state["phase"] != phase
                or "phase_source" in state and state["phase_source"] != "execution_phase_owner_v2"
                or "phase_receipt" in state):
            raise ValueError("caller phase/receipt differs from owned execution phase")
        setv(slice(0,4),[float(p == phase) for p in PHASES_V2])
        setv(50,phase_receipt.prelift_stable_s/self.prelift_scale_s)
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
        if self.mode == MODES_V2[0]:
            fields=state.get("physical_fields",{})
            for i in PHYSICAL_FIELDS_V2:
                item=fields.get(i)
                if (not isinstance(item,dict) or item.get("source") != CONTEXT_FIELDS_V2[i]["source"]
                        or item.get("identity") != identity or item.get("valid") is not True):
                    raise ValueError("full applicable contact/support source missing")
                raw=item.get("value")
                if type(raw) not in (int,float) or not math.isfinite(raw):
                    raise ValueError("finite physical source value required")
                if i == 7:
                    # New execution phase owns the final clock independently
                    # of PreliftHold; this field must match its observed endpoint.
                    if raw != phase_receipt.final_hold_s:
                        raise ValueError("hold duration differs from v2 phase owner history")
                setv(i,raw/CONTEXT_FIELDS_V2[i]["scale"])
        applicable=tuple(bool(valid[0,i]) for i in range(51))
        reasons=tuple(None if x else ("unanchored" if i in (8,9)
            else "finite task deadline disabled" if i == 46 else "smoke excluded")
            for i,x in enumerate(applicable))
        meta=ContextMetadataV2(self.owner_uid,(identity,),self.profile.digest,self.scope,self.mode,
            tuple(x["source"] for x in CONTEXT_FIELDS_V2),applicable,reasons,references,token,PHASE_CONTRACT_SHA256_V2,
            self._fixed_s0_sha,self.prelift_scale_s,self.grasp)
        return validate_context_v2(TaskContextPacketV2(values,valid,meta),backend_profile=self.profile)

    def reset_from_verified_state(self,state,*,history_owner,phase_owner=None):
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
        phase_receipt=self._phase_snapshot(phase_owner,identity,reset=True)
        fixed_s0_sha=state.get("fixed_s0_manifest_sha256")
        if not _sha(fixed_s0_sha):
            raise ValueError("fixed S0 manifest SHA required; CPU string grants no native qualification")
        self._fixed_s0_sha=fixed_s0_sha
        self._history_uid=history._owner_uid
        self._anchors,self._anchor_uids,self._rest=anchors,anchor_uids,rest
        self._references=(("target_material",reference),)+tuple(
            ("anchor."+str(i),uid) for i,uid in enumerate(anchor_uids) if uid is not None)
        z=nodes[0,:,2]
        self._fixed=[float(z.median()),float(z.min()),max(distance,.005),distance,.03,5.,.0005]
        self._reset_identity=identity
        self._success=self._failure=False
        packet=self._build(identity,history._previous,_digest(history._reset),state,phase_receipt=phase_receipt)
        self._remember_phase(phase_owner,phase_receipt)
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
        self._phase_snapshot(self._phase_source,identity)
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

    def advance_physical_endpoint(self,state,*,history_owner,collection_token,phase_owner):
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
        if state.get("fixed_s0_manifest_sha256") != self._fixed_s0_sha:
            raise ValueError("fixed S0 manifest cannot change within an episode")
        phase_receipt=self._phase_snapshot(phase_owner,identity)
        anchor_owner=state.get("anchor_owner")
        anchors,uids,references=self._anchors,self._anchor_uids,self._references
        if anchor_owner is not None:
            proposed,proposed_uids,proposed_refs=self._anchor_state(anchor_owner,identity)
            for old,old_uid,new,new_uid in zip(anchors,uids,proposed,proposed_uids):
                if old is not None and (new != old or new_uid != old_uid):
                    raise ValueError("fixed material anchor cannot be reanchored after contact loss")
            proposed_by_name=dict(proposed_refs)
            if any(proposed_by_name.get(name) != uid for name,uid in references):
                raise ValueError("fixed material/tangential reference UID cannot change within an episode")
            anchors,uids,references=proposed,proposed_uids,proposed_refs
        if (state.get("anchor_stations_m",anchors) != anchors
                or state.get("anchor_uids",uids) != uids):
            raise ValueError("new anchor needs its actual owner; fixed anchors cannot change")
        self._pending=self._build(identity,history._previous,collection_token,state,
            anchors=anchors,references=references,phase_receipt=phase_receipt)
        self._pending_anchor_state=anchors,uids,references,anchor_owner
        self._remember_phase(phase_owner,phase_receipt)
        self._pending_hash=_packet_hash(self._pending)
        self._used_tokens.add(collection_token)
        return self.pending_post

    def commit_post_context(self,packet):
        if self._closed:
            raise ValueError("closed task context owner cannot commit")
        if type(packet) is not TaskContextPacketV2:
            raise ValueError("exact v2 context packet required")
        if self._pending is None or _packet_hash(packet) != self._pending_hash:
            raise ValueError("one unchanged owned post context required after next-value evaluation")
        self._current=self._pending.owned()
        self._anchors,self._anchor_uids,self._references,anchor_source=self._pending_anchor_state
        if anchor_source is not None: self._anchor_source=anchor_source
        self._success=bool(self._current.values[0,48])
        self._failure=bool(self._current.values[0,49])
        self._pending=self._pre=None
        return self.current
