"""Tensor settlement of the frozen TrainingReward equations (no native admission).

Homogeneous batches use one file-backed RewardConfig and accept actual device
tensors. Python schema/identity work happens at reset/begin, never per-substep
per-lane endpoints. Each observe combines all numerical checks into one batch
guard; terminal lanes freeze and their subsequent frame entries are ignored.
The driver must stop/disable physical stepping of those lanes itself.

Float inputs are promoted to binary64, matching Python scalar arithmetic. This
does not promise bitwise CPU/CUDA transcendental or reduction identity. CONTACT
requires the original live PrimedContactContextOwner and is explicitly refused.
This API is a numerical consumer, not an endpoint attestation or PPO adapter.
"""
from dataclasses import dataclass
from types import MappingProxyType
import math
import struct

import torch

from .tabletop_training_reward import (
    TrainingReward, SourceFailure, SMOKE, FULL, CURRICULUM, CONTACT,
    SUCCESS, SCHEMA_SHA256, PHYSICS_HZ, WINDOW_SUBSTEPS, _BASE_UNITS,
    _FULL_UNITS, _BOOL_FIELDS, ERROR_FIELDS,
)

_STEP32 = struct.unpack(">f", struct.pack(">f", 1. / PHYSICS_HZ))[0]


@dataclass(frozen=True)
class TensorLaneSchema:
    """Actual owner's immutable identity and reader declarations for one lane.

    field_specs contains (name, unit, reader_source), including proven-NA slip
    in FULL. anchor_specs is ordered left/material, left/tangential,
    right/material, right/tangential and contains (reference_uid, drift_reader,
    uncertainty_reader). It may first become available at a window boundary;
    once declared it cannot change. Numeric anchor reference coordinates/ticks
    are consumed from the frame and latched at their first actual observation.
    """
    episode_uid: str
    env_id: int
    reset_uid: str
    rod_uid: str
    scope: str
    source: str
    target_ids: tuple
    field_specs: tuple
    required_pad_ids: tuple = ()
    anchor_specs: tuple = ()
    frame: str = "world"


@dataclass(frozen=True)
class TensorRewardFrame:
    """One actual 480 Hz observation; tensors have leading dimension N.

    fields supplies each declared finite scalar/boolean field except FULL's
    separately represented slip. FULL requires explicit anchor_present[N] and
    drift/uncertainty[N,4], reference[N,4,K], reference_tick[N,4]. Absent anchor
    entries may contain NaN; absence is proven only before any latched anchor
    and before bilateral contact, never substituted with numeric zero.
    """
    physics_tick: torch.Tensor
    solver_time_s: torch.Tensor
    fields: dict
    anchor_present: torch.Tensor | None = None
    anchor_drift: torch.Tensor | None = None
    anchor_uncertainty: torch.Tensor | None = None
    anchor_reference: torch.Tensor | None = None
    anchor_reference_tick: torch.Tensor | None = None


@dataclass(frozen=True)
class TensorRewardPacket:
    tokens: tuple
    policy_uids: tuple
    config_sha256: str
    schema_sha256: str
    profile: str
    profile_version: int
    gamma: float
    reward: torch.Tensor
    duration_s: torch.Tensor
    phi_pre: torch.Tensor
    phi_post_raw: torch.Tensor
    endpoint_phi: torch.Tensor
    cost_per_second: torch.Tensor
    integrated_cost: torch.Tensor
    event: torch.Tensor
    reason_masks: object
    substeps: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    buffer_full: torch.Tensor
    done: torch.Tensor
    post_context: object
    terms: object
    pre_context: object
    pre_terms: object
    substep_terms: object
    event_terms: object
    schemas: tuple

    @property
    def native_ready(self): return False
    @property
    def teacher_admission(self): return False
    @property
    def native_qualification(self): return False

    def bootstrap(self, shaped_value):
        if (not isinstance(shaped_value, torch.Tensor) or shaped_value.shape != self.reward.shape
                or shaped_value.device != self.reward.device or not shaped_value.is_floating_point()
                or bool((~torch.isfinite(shaped_value)).any())):
            raise ValueError("finite shaped value from actual post_context required")
        return torch.where(self.terminated, 0., self.gamma * shaped_value)

    def event_reasons(self, lane):
        """Explicit cold-path materialization, never called by settlement."""
        return tuple(name for name, mask in self.reason_masks.items() if bool(mask[lane]))


def measured_finger_errors(actual_active_q, finger_goal):
    """Named active SDK q [N,2,6] -> max participating errors [N,2].

    The owner must gather the six ACTIVE joints in their original named order.
    Validation of finite/nonnegative outputs is fused into observe/reset guard.
    """
    if (not isinstance(actual_active_q, torch.Tensor) or actual_active_q.ndim != 3
            or tuple(actual_active_q.shape[1:]) != (2, 6)
            or not actual_active_q.is_floating_point()):
        raise ValueError("actual named active q [N,2,6] required")
    q = actual_active_q.to(dtype=torch.float64)
    target = torch.tensor(finger_goal.target_rad, device=q.device, dtype=q.dtype)
    error = (q - target).abs()[:, :, finger_goal.participating].amax(dim=2)
    # A corrupt nonparticipating joint is invalid in measured_active_angles too.
    return torch.where(torch.isfinite(q).all(dim=(1, 2))[:, None], error, torch.nan)


def _text(value):
    return type(value) is str and bool(value.strip())


class TensorTrainingReward:
    """Exactly once batch windows, with reset-selected lane state on device.

    reset_selected(ids, schemas, frame, previous_request=..., tokens=...,
                   policy_uids=...) consumes only K selected reset observations.
    begin(schemas, request=[N,24], tokens=..., policy_uids=...,
          physics_tick=[N], solver_time_s=[N], remaining_substeps=...) reserves
    all N lanes against the actual PRE clock. observe(frame) consumes one real
    interval for each nonterminal lane and returns a device readiness mask.
    settle_once(tokens, truncated=[N], buffer_full=[N]) returns an owned packet.

    remaining_substeps is a Python int >= 8 or actual int64 tensor [N]. Token
    and episode identity sets persist across selected resets. A corrupt frame
    quarantines the batch; selected reset can recover only the reset lanes.
    """
    def __init__(self, config, *, num_envs, device="cpu"):
        self.config = TrainingReward(config).config  # Original file/SHA validation.
        if self.config.profile == CONTACT:
            raise SourceFailure("CONTACT requires original live contact owner; tensor attestation unavailable")
        if dict(self.config.success) != dict(SUCCESS):
            raise ValueError("original success thresholds required; no relaxed tensor success")
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("positive batch size required")
        self.n, self.device = num_envs, torch.device(device)
        self.full = self.config.profile in (FULL, CURRICULUM)
        self.schemas = [None] * num_envs
        self._tokens = [set() for _ in range(num_envs)]
        self._episodes = [set() for _ in range(num_envs)]
        self._pending = None
        self._poisoned_lanes = set()
        self._policies = None
        self._f = lambda value=0.: torch.full((num_envs,), value, device=self.device, dtype=torch.float64)
        self._b = lambda value=False: torch.full((num_envs,), value, device=self.device, dtype=torch.bool)
        self._i = lambda value=0: torch.full((num_envs,), value, device=self.device, dtype=torch.int64)
        self._tick, self._time = self._i(), self._f()
        self._reset_tick = self._i()
        self._dref, self._median, self._min = self._f(math.nan), self._f(math.nan), self._f(math.nan)
        self._contact_start, self._hold_start = self._i(-1), self._i(-1)
        self._contact_s, self._hold_s = self._f(), self._f()
        self._anchor_latched, self._ever_bilateral = self._b(), self._b()
        self._anchor_ref = None
        self._anchor_tick = torch.full((num_envs, 4), -1, dtype=torch.int64, device=self.device)
        self._anchor_drift = torch.full((num_envs,4),math.nan,device=self.device,dtype=torch.float64)
        self._anchor_uncertainty = self._anchor_drift.clone()
        self._slip, self._slip_status = self._f(math.nan), self._i(-1)
        self._dwell, self._arrived = self._i(), self._b()
        self._previous = torch.full((num_envs, 24), math.nan, device=self.device, dtype=torch.float64)
        self._previous_tokens, self._previous_policies = [None]*num_envs, [None]*num_envs
        self._phi, self._closed, self._quarantined = self._f(math.nan), self._b(True), self._b()
        self._terms = {}
        self._last_values = {}

    @property
    def native_ready(self): return False
    @property
    def teacher_admission(self): return False
    @property
    def native_qualification(self): return False
    @property
    def quarantined(self): return self._quarantined.clone()
    @property
    def post_context(self): return self._context()

    def _tensor(self, value, shape, *, boolean=False, integer=False):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.device != self.device or value.requires_grad
                or (boolean and value.dtype != torch.bool)
                or (integer and value.dtype != torch.int64)
                or (not boolean and not integer and not value.is_floating_point())):
            raise SourceFailure("actual device tensor shape/dtype/ownership mismatch")
        return value if boolean or integer else value.to(torch.float64)

    def _guard(self, bad, message):
        # The sole batch -> host numerical guard at each source boundary.
        if bool(bad.any()):
            self._quarantined[:] = True
            self._poisoned_lanes = set(range(self.n))
            raise SourceFailure(message)

    def _schema(self, schemas, ids, *, reset):
        if type(schemas) is not tuple or len(schemas) != len(ids):
            raise SourceFailure("one immutable actual lane schema per selected lane required")
        previous_names = None
        for lane, s in zip(ids, schemas):
            if (type(s) is not TensorLaneSchema
                    or any(not _text(v) for v in (s.episode_uid,s.reset_uid,s.rod_uid,s.scope,s.source))
                    or type(s.env_id) is not int or s.env_id < 0
                    or type(s.frame) is not str or s.frame!="world"
                    or s.scope not in ("cpu_fixture","unqualified_source")
                    or type(s.target_ids) is not tuple or not s.target_ids
                    or any(not _text(v) for v in s.target_ids) or len(set(s.target_ids)) != len(s.target_ids)
                    or type(s.field_specs) is not tuple or any(type(v) is not tuple or len(v)!=3
                        or any(not _text(x) for x in v) for v in s.field_specs)):
                raise SourceFailure("immutable original identity/source/material schema required")
            names=tuple(name for name,unit,source in s.field_specs)
            if len(set(names)) != len(names) or previous_names is not None and names != previous_names:
                raise SourceFailure("homogeneous field schema required")
            previous_names=names
            units=dict(_BASE_UNITS)
            if self.full: units.update(_FULL_UNITS)
            units.update((name,"dimensionless") for name in names if name.startswith("original_guard_"))
            if self.config.finger_goal is not None: units.update((name,"rad") for name in ERROR_FIELDS)
            if any(name not in names for name in units):
                raise SourceFailure("missing applicable original reward field")
            if any(name in units and unit != units[name] for name,unit,source in s.field_specs):
                raise SourceFailure("original source units mismatch")
            if self.full:
                pads=s.required_pad_ids
                if (type(pads) is not tuple or len(pads)!=2 or {x[0] for x in pads}!={"left","right"}
                        or any(type(p) is not tuple or not p or any(not _text(v) for v in p)
                            or len(set(p))!=len(p) for side,p in pads)
                        or "slip" not in names or dict((n,u) for n,u,_ in s.field_specs)["slip"]!="m"):
                    raise SourceFailure("fixed actual pad denominator and slip history schema required")
            if (type(s.anchor_specs) is not tuple or len(s.anchor_specs) not in (0,4)
                    or any(type(v) is not tuple or len(v)!=3 or any(not _text(x) for x in v) for v in s.anchor_specs)):
                raise SourceFailure("fixed four anchor reader/UID schema required")
            if not reset:
                prior=self.schemas[lane]
                if prior is None or any(getattr(prior,k)!=getattr(s,k) for k in (
                        "episode_uid","env_id","reset_uid","rod_uid","scope","source","frame","target_ids","field_specs","required_pad_ids")):
                    raise SourceFailure("episode identity/source/schema changed before window")
                if prior.anchor_specs and prior.anchor_specs!=s.anchor_specs:
                    raise SourceFailure("fixed anchor UID/source changed; reanchoring forbidden")

    def _frame(self, frame, schemas=None, *, field_names=None, anchor_available=None):
        k=self.n if schemas is None else len(schemas)
        if type(frame) is not TensorRewardFrame or type(frame.fields) is not dict:
            raise SourceFailure("actual batched frame required")
        tick=self._tensor(frame.physics_tick,(k,),integer=True)
        time=self._tensor(frame.solver_time_s,(k,))
        names=field_names if field_names is not None else {n for n,_,_ in schemas[0].field_specs} - ({"slip"} if self.full else set())
        if set(frame.fields)!=names:
            raise SourceFailure("actual value bundle differs from prevalidated schema")
        values={n:self._tensor(v,(k,),boolean=n in _BOOL_FIELDS or n.startswith("original_guard_"))
                for n,v in frame.fields.items()}
        bad=(tick<0)|~torch.isfinite(time)
        for value in values.values():
            if value.dtype!=torch.bool: bad |= ~torch.isfinite(value)
        bad |= values["distance"]<0
        if self.config.finger_goal is not None:
            for n in ERROR_FIELDS: bad |= values[n]<0
        anchors=None
        if self.full:
            present=self._tensor(frame.anchor_present,(k,),boolean=True)
            drift=self._tensor(frame.anchor_drift,(k,4))
            uncertainty=self._tensor(frame.anchor_uncertainty,(k,4))
            if frame.anchor_reference is None or frame.anchor_reference.ndim!=3 or frame.anchor_reference.shape[2]<1:
                raise SourceFailure("actual fixed anchor reference coordinates required")
            ref=self._tensor(frame.anchor_reference,(k,4,frame.anchor_reference.shape[2]))
            rt=self._tensor(frame.anchor_reference_tick,(k,4),integer=True)
            available=(torch.tensor([bool(s.anchor_specs) for s in schemas],device=self.device)
                       if anchor_available is None else anchor_available)
            bad |= present & (~available | (~torch.isfinite(drift)|~torch.isfinite(uncertainty)
                     |(drift<0)|(uncertainty<0)|~torch.isfinite(drift+uncertainty)
                     |~torch.isfinite(drift-uncertainty)).any(dim=1)
                     |~torch.isfinite(ref).all(dim=(1,2)))
            for n in ("coverage_left","coverage_right"):
                bad |= (values[n]<0)|(values[n]>1)
            anchors=(present,drift,uncertainty,ref,rt)
        return tick,time,values,anchors,bad

    def _potential(self, ids, tick, v, anchors, *, reset=False):
        # Returns candidate state, committed only after the one numerical guard.
        dref=self._dref[ids]
        distance_ratio=v["distance"]/dref
        pd=2.*(1.-distance_ratio.clamp(0.,1.))
        terms={"distance":pd}
        state={"_derived_bad":~torch.isfinite(distance_ratio)}
        phi=pd
        if self.config.finger_goal is not None:
            goal=self.config.finger_goal
            errors=torch.stack([v[n] for n in ERROR_FIELDS],dim=1)
            pg=torch.exp(-(errors-goal.tolerance_rad).clamp_min(0.)/goal.scale_rad).sum(dim=1)/2.
            terms["finger_goal"]=pg;phi=phi+pg
        if self.full:
            present,d,u,ref,rt=anchors
            contact=(v["coverage_left"]==1)&(v["coverage_right"]==1)&v["opposed_left"]&v["opposed_right"]
            cs=torch.where(contact,torch.where(self._contact_start[ids]<0,tick,self._contact_start[ids]),-1)
            contact_s=torch.where(cs<0,0.,(tick-cs).to(torch.float64)/PHYSICS_HZ)
            slip=torch.where(present,(d+u).amax(dim=1),torch.nan)
            slip_false=((d-u)>.0005).any(dim=1)
            slip_true=((d+u)<=.0005).all(dim=1)
            status=torch.where(present,torch.where(slip_false,0,torch.where(slip_true,1,-1)),-1)
            md=v["median_z"]-self._median[ids];mn=v["min_z"]-self._min[ids]
            h=torch.minimum(md,mn)
            genuine=contact&v["bilateral_support"]&v["off_table"]&v["no_external_support"]&~v["prohibited_support"]
            success=self.config.success
            if self.config.profile==CURRICULUM:
                lift_h=.5*(md+mn)
                lift_ok=v["bilateral_support"]&v["no_external_support"]&~v["prohibited_support"]
                pl=torch.where(lift_ok,(lift_h/success["net_lift_m"]).clamp(0.,1.),0.)
                hold=lift_ok&(lift_h>=success["net_lift_m"])&v["hold_motion"]&(status!=0)&v["numerical_health"]&~v["constraint_failure"]
                hs=self._hold_start[ids]
                hold_s=torch.where(hold,self._hold_s[ids]+1./PHYSICS_HZ,(self._hold_s[ids]-2./PHYSICS_HZ).clamp_min(0.))
            else:
                pl=torch.where(genuine,(h/success["net_lift_m"]).clamp(0.,1.),0.)
                hold=genuine&(h>=success["net_lift_m"])&v["hold_motion"]&(status==1)&v["numerical_health"]&~v["constraint_failure"]
                hs=torch.where(hold,torch.where(self._hold_start[ids]<0,tick,self._hold_start[ids]),-1)
                hold_s=torch.where(hs<0,0.,(tick-hs).to(torch.float64)/PHYSICS_HZ)
            pc=(v["coverage_left"]+v["coverage_right"])/2.
            geom=(v["opposed_left"].to(torch.float64)+v["opposed_right"].to(torch.float64))/2.
            terms.update(contact=2.*pc,persistence=.5*(contact_s/.25).clamp(0.,1.),
                geometry=geom,lift=4.*pl,hold=4.*(hold_s/success["hold_s"]).clamp(0.,1.))
            phi=phi+terms["contact"]+terms["persistence"]+terms["geometry"]+terms["lift"]+terms["hold"]
            state.update(_contact_start=cs,_contact_s=contact_s,_hold_start=hs,_hold_s=hold_s,
                _slip=slip,_slip_status=status,_ever_bilateral=self._ever_bilateral[ids]|contact,
                _anchor_latched=self._anchor_latched[ids]|present)
            state["_derived_bad"] |= ~torch.isfinite(md)|~torch.isfinite(mn)|~torch.isfinite(hold_s)|~torch.isfinite(contact_s)
            state["_derived_bad"] |= present & (~torch.isfinite(slip/.0005)|~torch.isfinite(slip/success["max_slip_m"]))
            if self.config.profile==CURRICULUM:
                state["_derived_bad"] |= lift_ok & ~torch.isfinite(lift_h/success["net_lift_m"])
            else:
                state["_derived_bad"] |= genuine & ~torch.isfinite(h/success["net_lift_m"])
            # NaN denotes unknown, and cannot clear history after a first anchor.
            state["_derived_bad"] |= ~present & (self._anchor_latched[ids]|self._ever_bilateral[ids]|contact)
            state["_derived_bad"] |= present & ((rt<self._reset_tick[ids,None])|(rt>tick[:,None])).any(dim=1)
            if self._anchor_ref is not None:
                if self._anchor_ref.shape[2]!=ref.shape[2]:
                    raise SourceFailure("fixed reference vector schema changed")
                state["_derived_bad"] |= present & self._anchor_latched[ids] & (
                    (ref!=self._anchor_ref[ids]).any(dim=(1,2)) | (rt!=self._anchor_tick[ids]).any(dim=1))
        return phi,terms,state

    def _commit(self, ids, active, phi, terms, state, tick, time, values, anchors):
        def put(target,value): target[ids]=torch.where(active,value,target[ids])
        for name,value in state.items():
            if name!="_derived_bad": put(getattr(self,name),value)
        if anchors is not None:
            present,d,u,ref,rt=anchors
            if self._anchor_ref is None:
                self._anchor_ref=torch.full((self.n,4,ref.shape[2]),math.nan,device=self.device,dtype=torch.float64)
            mask=(active&present)[:,None,None]
            self._anchor_ref[ids]=torch.where(mask,ref,self._anchor_ref[ids])
            self._anchor_tick[ids]=torch.where((active&present)[:,None],rt,self._anchor_tick[ids])
            self._anchor_drift[ids]=torch.where((active&present)[:,None],d,self._anchor_drift[ids])
            self._anchor_uncertainty[ids]=torch.where((active&present)[:,None],u,self._anchor_uncertainty[ids])
        put(self._phi,phi);put(self._tick,tick);put(self._time,time)
        for name,value in terms.items():
            if name not in self._terms: self._terms[name]=self._f(math.nan)
            put(self._terms[name],value)
        for name,value in values.items():
            if name not in self._last_values:
                self._last_values[name]=self._b() if value.dtype==torch.bool else self._f(math.nan)
            put(self._last_values[name],value)

    def reset_selected(self, lane_ids, schemas, frame, *, previous_request,tokens,policy_uids):
        if self._pending is not None and not self._poisoned_lanes:
            raise RuntimeError("cannot reset unfinished window")
        if self._poisoned_lanes:
            # Every lane of a failed batch is quarantined. Its reserved tokens
            # stay consumed, but no unfinished failed window blocks recovery.
            self._pending=None;self._policies=None
        ids=self._ids(lane_ids)
        self._schema(schemas,lane_ids,reset=True)
        self._identities(tokens,policy_uids,len(lane_ids))
        for lane,s,token in zip(lane_ids,schemas,tokens):
            if s.episode_uid in self._episodes[lane]: raise RuntimeError("episode reset UID must be new")
        tick,time,values,anchors,bad=self._frame(frame,schemas)
        previous=self._tensor(previous_request,(len(lane_ids),24))
        bad |= (~torch.isfinite(previous)|(previous.abs()>1)).any(dim=1)
        bad |= ~values["numerical_health"]|values["constraint_failure"]
        for n,v in values.items():
            if n.startswith("original_guard_"): bad |= v
        # Validate before any reset mutation; a bad reset cannot clear quarantine.
        self._guard(bad,"reset numerical/health/source validation failed")
        self._reset_tick[ids]=tick;self._dref[ids]=values["distance"].clamp_min(.005)
        for name in ("_contact_start","_hold_start"): getattr(self,name)[ids]=-1
        for name in ("_contact_s","_hold_s","_dwell"): getattr(self,name)[ids]=0
        for name in ("_anchor_latched","_ever_bilateral","_arrived"): getattr(self,name)[ids]=False
        self._slip[ids]=math.nan;self._slip_status[ids]=-1
        self._anchor_tick[ids]=-1
        self._anchor_drift[ids]=math.nan;self._anchor_uncertainty[ids]=math.nan
        if self._anchor_ref is not None: self._anchor_ref[ids]=math.nan
        if self.full:
            self._median[ids]=values["median_z"];self._min[ids]=values["min_z"]
        phi,terms,state=self._potential(ids,tick,values,anchors,reset=True)
        self._guard(~torch.isfinite(phi)|state.get("_derived_bad",torch.zeros_like(bad)),"reset finite derived source required")
        active=torch.ones_like(bad)
        self._commit(ids,active,phi,terms,state,tick,time,values,anchors)
        self._previous[ids]=previous;self._closed[ids]=False;self._quarantined[ids]=False
        self._poisoned_lanes.difference_update(lane_ids)
        self._previous_tokens=list(self._previous_tokens);self._previous_policies=list(self._previous_policies)
        for lane,s,token,policy in zip(lane_ids,schemas,tokens,policy_uids):
            self.schemas[lane]=s;self._episodes[lane].add(s.episode_uid);self._tokens[lane].add(token)
            self._previous_tokens[lane]=token;self._previous_policies[lane]=policy

    def _ids(self, ids):
        if (type(ids) is not tuple or not ids or any(type(i) is not int or not 0<=i<self.n for i in ids)
                or len(set(ids))!=len(ids)):
            raise ValueError("unique selected lane IDs required")
        return torch.tensor(ids,device=self.device,dtype=torch.int64)

    @staticmethod
    def _identities(tokens,policies,n):
        if (type(tokens) is not tuple or type(policies) is not tuple or len(tokens)!=n or len(policies)!=n
                or any(not _text(v) for v in tokens+policies)):
            raise SourceFailure("explicit immutable token and policy identities required")

    def begin(self, schemas, *, request,tokens,policy_uids,physics_tick,solver_time_s,remaining_substeps):
        if self._poisoned_lanes: raise SourceFailure("quarantined rollout requires verified reset")
        if self._pending is not None: raise RuntimeError("prior token has unfinished window")
        if any(s is None for s in self.schemas): raise RuntimeError("active verified reset required")
        try:
            self._schema(schemas,tuple(range(self.n)),reset=False)
            self._identities(tokens,policy_uids,self.n)
            action=self._tensor(request,(self.n,24))
            tick=self._tensor(physics_tick,(self.n,),integer=True)
            time=self._tensor(solver_time_s,(self.n,))
        except SourceFailure:
            self._quarantined[:]=True;self._poisoned_lanes=set(range(self.n));raise
        for lane,token in enumerate(tokens):
            if token in self._tokens[lane]: raise RuntimeError("request token already reserved/settled")
        if type(remaining_substeps) is int:
            if remaining_substeps<WINDOW_SUBSTEPS: raise ValueError("eight (8) budgeted substeps required")
            budget_bad=self._b()
        else:
            budget=self._tensor(remaining_substeps,(self.n,),integer=True)
            budget_bad=budget<WINDOW_SUBSTEPS
        bad=(~torch.isfinite(action)|(action.abs()>1)).any(dim=1)|~torch.isfinite(time)|(tick!=self._tick)|(time!=self._time)
        bad |= self._quarantined|self._closed|budget_bad
        self._guard(bad,"quarantined/closed, budget or actual PRE clock/request validation failed")
        self.schemas=list(schemas)
        self._window_fields={n for n,_,_ in schemas[0].field_specs} - ({"slip"} if self.full else set())
        self._window_anchors=torch.tensor([bool(s.anchor_specs) for s in schemas],device=self.device)
        self._native_float32_clock=torch.tensor([s.scope=="unqualified_source" and s.source in (
            "policy_owned_pinch_endpoint_v1","policy_owned_tripod_endpoint_v1","policy_owned_power_endpoint_v1") for s in schemas],device=self.device)
        self._pending,self._policies=tokens,policy_uids
        for lane,token in enumerate(tokens): self._tokens[lane].add(token)
        self._request=action.clone()
        self._qcmd=(((action-self._previous)/2.)**2).sum(dim=1)/24.
        self._phi_pre=self._phi.clone();self._pre_context=self._context();self._pre_terms=self._term_payload()
        self._count,self._terminal,self._event=self._i(),self._b(),self._f()
        self._arrival_pending=self._b()
        self._costs=torch.full((self.n,8),math.nan,device=self.device,dtype=torch.float64)
        self._slip_history=torch.full_like(self._costs,math.nan)
        self._reasons={name:self._b() for name in ("numerical_health","constraint_failure")}
        self._reasons.update((n,self._b()) for n,_,_ in schemas[0].field_specs if n.startswith("original_guard_"))
        if self.full: self._reasons.update(prohibited_support=self._b(),success=self._b())
        self._observe_calls=0

    def _interval(self, time):
        interval=time-self._time
        match=(interval-1./PHYSICS_HZ).abs()<=1e-9
        if self.config.profile==SMOKE:
            native=self._native_float32_clock
            both=torch.stack((self._time,time),dim=1)
            f32=both.to(torch.float32).to(torch.float64)
            # Match Python frexp-based ULP including zero and subnormal clamp.
            _,exp=torch.frexp(both)
            ulp=torch.maximum(torch.full_like(both,2.**-149),torch.ldexp(torch.ones_like(both),exp-24))
            ulp=torch.where(both==0,2.**-149,ulp)
            bound=(.5*ulp.sum(dim=1)+abs(_STEP32-1./PHYSICS_HZ)).clamp_max(1e-6)
            match |= native & ((both>=0)&(both<=float.fromhex("0x1.fffffep+127"))&(both==f32)).all(dim=1)&((interval-1./PHYSICS_HZ).abs()<=bound)
        return match

    def observe(self, frame):
        if self._poisoned_lanes: raise SourceFailure("quarantined rollout has no usable PPO packet")
        if self._pending is None or self._observe_calls>=8:
            raise RuntimeError("no open physical interval; terminal/full window cannot advance")
        try:
            tick,time,v,anchors,bad=self._frame(frame,field_names=self._window_fields,anchor_available=self._window_anchors)
            active=~self._terminal&(self._count<8)
            bad |= (tick!=self._tick+1)|~self._interval(time)
            ids=torch.arange(self.n,device=self.device)
            phi,terms,state=self._potential(ids,tick,v,anchors)
            bad |= ~torch.isfinite(phi)|state.get("_derived_bad",self._b())
            cost=.05*self._qcmd+.10
            if self.full:
                slip=state["_slip"]
                cost=cost+torch.where(torch.isnan(slip),0.,self.config.costs["slip"]*(slip/self.config.success["max_slip_m"]).clamp(0.,2.))
            bad |= ~torch.isfinite(cost)
            self._guard(self._quarantined|(bad&active),"nonfinite/numerical source, fixed anchor or consecutive actual 480Hz clock failure")
            failures={"numerical_health":~v["numerical_health"],"constraint_failure":v["constraint_failure"]}
            failures.update((n,value) for n,value in v.items() if n.startswith("original_guard_"))
            if self.full: failures["prohibited_support"]=v["prohibited_support"]
            failure=self._b()
            for mask in failures.values(): failure |= mask
            if self.config.finger_goal is not None:
                goal=self.config.finger_goal
                tol=~failure
                for name in ERROR_FIELDS: tol &= v[name]<=goal.tolerance_rad
                dwell=torch.where(tol,self._dwell+1,0)
                change=active&~self._arrived
                self._dwell=torch.where(change,dwell,self._dwell)
                self._arrival_pending |= change&(dwell>=goal.arrival_steps)
            success=~failure&(state["_hold_s"]>=self.config.success["hold_s"]) if self.full else self._b()
            event=torch.where(failure,-40.,torch.where(success,100.,0.)).to(torch.float64)
            if self.full: event=torch.where(v["prohibited_support"],-50.,event)
            for name,mask in failures.items(): self._reasons[name] |= active&mask
            if self.full: self._reasons["success"] |= active&success
            self._event=torch.where(active,event,self._event)
            self._costs[:,self._observe_calls]=torch.where(active,cost,torch.nan)
            if self.full: self._slip_history[:,self._observe_calls]=torch.where(active,state["_slip"],torch.nan)
            self._count += active.to(torch.int64)
            self._terminal |= active&(failure|success)
            self._commit(ids,active,phi,terms,state,tick,time,v,anchors)
            self._previous=torch.where(active[:,None],self._request,self._previous)
            # Shared tokens/policies are established once; no lane data readback.
            self._previous_tokens=self._pending;self._previous_policies=self._policies
            self._observe_calls+=1
            return (self._count==8)|self._terminal
        except SourceFailure:
            self._quarantined[:]=True;self._poisoned_lanes=set(range(self.n));raise

    def settle_once(self, tokens, *, truncated=None,buffer_full=None):
        if self._poisoned_lanes: raise SourceFailure("quarantined rollout has no usable PPO packet")
        if self._pending is None or tokens!=self._pending:
            raise RuntimeError("token absent or already settled")
        if self._observe_calls==0:
            raise RuntimeError("incomplete window: eight samples unless trusted terminal")
        tr=self._b() if truncated is None else self._tensor(truncated,(self.n,),boolean=True)
        bf=self._b() if buffer_full is None else self._tensor(buffer_full,(self.n,),boolean=True)
        incomplete=(self._count==0)|((self._count<8)&~self._terminal)
        boundary=(tr&bf)|(self._terminal&tr)
        event=self._event.clone()
        arrived=self._arrival_pending&~self._terminal
        if self.config.finger_goal is not None:
            event += arrived.to(torch.float64)*self.config.finger_goal.arrival_reward
        # Only actual intervals enter the integral; unused NaN slots remain NA.
        intervals=torch.arange(8,device=self.device)[None,:]<self._count[:,None]
        integrated=torch.where(intervals,self._costs,0.).sum(dim=1)/PHYSICS_HZ
        endpoint_phi=torch.where(self._terminal,0.,self._phi)
        reward=self.config.gamma*endpoint_phi-self._phi_pre-integrated+event
        numerical=~torch.isfinite(reward)|~torch.isfinite(integrated)
        # Success path performs one batch guard for boundary + reward arithmetic.
        if bool((incomplete|boundary|numerical).any()):
            if bool(incomplete.any()): raise RuntimeError("incomplete window: eight samples unless trusted terminal")
            if bool(boundary.any()): raise ValueError("ambiguous terminal/truncation/buffer boundary")
            self._quarantined[:]=True;self._poisoned_lanes=set(range(self.n))
            raise SourceFailure("finite integrated reward required")
        if self.config.finger_goal is not None: self._arrived |= arrived
        context=self._context()
        term_payload=self._term_payload()
        event_terms={
            "trusted_constraint_failure":self._term(self._event<0,"dimensionless","derived:original_failure_predicates",
                torch.where((self._event<0)&(self._event!=-50.),self._event,0.))}
        if self.full:
            event_terms.update(prohibited_support=self._term(self._last_values["prohibited_support"],"dimensionless",
                self._sources("prohibited_support"),torch.where(self._event==-50.,self._event,0.)),
                success=self._term(self._event==100.,"dimensionless","derived:independent_reward_hold_predicate",
                torch.where(self._event>0,self._event,0.)))
        else:
            event_terms.update(prohibited_support=self._excluded(),success=self._excluded())
        if self.config.finger_goal is not None:
            event_terms["finger_goal_arrived_once"]=self._term(arrived,"dimensionless",
                "derived:measured_bilateral_active_joint_goal",arrived.to(torch.float64)*self.config.finger_goal.arrival_reward)
        contribution=lambda value:torch.where(intervals,value,torch.nan)
        substep_terms={
            "command_rate":self._term(contribution(self._qcmd[:,None]),"dimensionless","submitted_request",
                contribution(.05*self._qcmd[:,None]/PHYSICS_HZ),applicable=intervals),
            "time":self._term(contribution(torch.full_like(self._costs,1./PHYSICS_HZ)),"s","actual_endpoint_clock",
                contribution(torch.full_like(self._costs,.10/PHYSICS_HZ)),applicable=intervals)}
        substep_terms["slip"]=(self._term(self._slip_history,"m","derived:conservative_fixed_anchors",
            contribution((self._costs-.05*self._qcmd[:,None]-.10)/PHYSICS_HZ),
            applicable=intervals&~torch.isnan(self._slip_history),history_reason="no_bilateral_contact_since_reset")
            if self.full else self._excluded(shape=self._costs.shape))
        packet=TensorRewardPacket(self._pending,self._policies,self.config.config_sha256,SCHEMA_SHA256,
            self.config.profile,self.config.profile_version,self.config.gamma,reward.clone(),
            self._count.to(torch.float64)/PHYSICS_HZ,self._phi_pre.clone(),self._phi.clone(),endpoint_phi.clone(),
            self._costs.clone(),integrated.clone(),event.clone(),self._copy(self._reasons),self._count.clone(),
            self._terminal.clone(),tr.clone(),bf.clone(),(self._terminal|tr).clone(),context,term_payload,
            self._pre_context,self._pre_terms,MappingProxyType(substep_terms),MappingProxyType(event_terms),tuple(self.schemas))
        self._closed=self._terminal|tr;self._pending=None;self._policies=None
        return packet

    @staticmethod
    def _copy(values):
        return MappingProxyType({name:value.clone() for name,value in values.items()})

    def _term(self, raw, unit, source, contribution=None, *, applicable=None,history_reason=None):
        shape=(self.n,) if contribution is None else contribution.shape
        return MappingProxyType(dict(raw=raw.clone() if isinstance(raw,torch.Tensor) else raw,
            unit=unit,source=source,valid=torch.ones(shape,dtype=torch.bool,device=self.device),
            applicable=(torch.ones(shape,dtype=torch.bool,device=self.device) if applicable is None else applicable.clone()),
            contribution=(torch.zeros(shape,dtype=torch.float64,device=self.device) if contribution is None else contribution.clone()),
            history_reason=history_reason))

    def _excluded(self,shape=None):
        shape=(self.n,) if shape is None else shape
        return self._term(None,"dimensionless","profile_excluded",torch.zeros(shape,device=self.device,dtype=torch.float64),
            applicable=torch.zeros(shape,device=self.device,dtype=torch.bool),history_reason="profile_excluded")

    def _sources(self,*names):
        return tuple(";".join(dict((n,s) for n,u,s in schema.field_specs)[name] for name in names)
                     for schema in self.schemas)

    def _term_payload(self):
        terms={name:self._excluded() for name in ("distance","contact","persistence","geometry","lift","hold",
            "slip","velocity","prohibited_support","task_deadline")}
        terms["distance"]=self._term(self._last_values["distance"],"m",self._sources("distance"),self._terms["distance"])
        if self.config.finger_goal is not None:
            errors=torch.stack([self._last_values[n] for n in ERROR_FIELDS],dim=1)
            terms["finger_goal"]=self._term(errors,"rad",self._sources(ERROR_FIELDS[0]),self._terms["finger_goal"])
        if self.full:
            v=self._last_values
            terms.update(
                contact=self._term((v["coverage_left"]+v["coverage_right"])/2.,"coverage_fraction",
                    self._sources("coverage_left","coverage_right"),self._terms["contact"]),
                persistence=self._term(self._contact_s,"s","derived:continuous_required_bilateral_original_opposed",self._terms["persistence"]),
                geometry=self._term((v["opposed_left"].double()+v["opposed_right"].double())/2.,"dimensionless",
                    self._sources("opposed_left","opposed_right"),self._terms["geometry"]),
                lift=self._term(torch.minimum(v["median_z"]-self._median,v["min_z"]-self._min),"m",
                    self._sources("median_z","min_z"),self._terms["lift"]),
                hold=self._term(self._hold_s,"s","derived:continuous_instantaneous_hold",self._terms["hold"]),
                prohibited_support=self._term(v["prohibited_support"],"dimensionless",self._sources("prohibited_support")),
                slip=self._term(self._slip,"m",tuple(";".join(a[1] for a in s.anchor_specs) for s in self.schemas),
                    torch.where(torch.isnan(self._slip),0.,(self._slip/.0005).clamp(0.,2.)),
                    applicable=~torch.isnan(self._slip),history_reason="no_bilateral_contact_since_reset"))
        for name,unit,source in self.schemas[0].field_specs:
            raw=self._slip if self.full and name=="slip" else self._last_values[name]
            # The scalar endpoint's explicit history-only slip field remains NA
            # even while drift/uncertainty are supplied by applicable anchors.
            if self.full and name=="slip":
                terms["raw:slip"]=self._term(None,unit,self._sources(name),applicable=self._b(),
                    history_reason="no_bilateral_contact_since_reset")
            else: terms["raw:"+name]=self._term(raw,unit,self._sources(name))
        if self.full:
            for j,(side,kind) in enumerate(( (s,k) for s in ("left","right") for k in ("material","tangential"))):
                for label,raw,source_index in (("drift",self._anchor_drift,1),("uncertainty",self._anchor_uncertainty,2)):
                    terms["raw:anchor:"+side+":"+kind+":"+label]=self._term(raw[:,j],"m",
                        tuple(s.anchor_specs[j][source_index] if s.anchor_specs else None for s in self.schemas),
                        applicable=self._anchor_latched)
        return MappingProxyType(terms)

    def _context(self):
        values=dict(physics_tick=self._tick,solver_time_s=self._time,distance_ref_m=self._dref,
            reset_median_z_m=self._median,reset_min_z_m=self._min,contact_s=self._contact_s,hold_s=self._hold_s,
            slip_status=self._slip_status,slip_m=self._slip,previous_request=self._previous,
            anchor_reference_tick=self._anchor_tick)
        if self._anchor_ref is not None: values["anchor_reference"]=self._anchor_ref
        result={name:value.clone() for name,value in values.items()}
        result.update(previous_request_token=tuple(self._previous_tokens),previous_policy_uid=tuple(self._previous_policies),
            schemas=tuple(self.schemas),finite_task_remaining_s=None,config_sha256=self.config.config_sha256,
            schema_sha256=SCHEMA_SHA256)
        return MappingProxyType(result)
