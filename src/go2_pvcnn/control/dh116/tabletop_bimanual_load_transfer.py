"""Default-OFF experimental phase/request machine, never an action controller.

Each instance belongs to one actual env/episode and consumes 480Hz SampleClock
and sourced StateField observations. External original predicates supply any
following/tilt/tracking/support tolerance: this module invents none. Requests
do not submit targets, cancel queued drives, create contact or qualify success.
Original measured-hold/pressure/unload, physics guards and SurfaceLiftSuccess
(30mm / 5s / fixed-anchor 0.5mm) remain authoritative and unchanged.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from types import MappingProxyType

from .tabletop_bimanual_mpc_state import SampleClock,StateField

VERSION='dh116-experimental-bimanual-load-transfer-v1'
PHYSICS_DT_S=1/480
STABILIZE_INTERVALS=96
TRANSFER_RATE_M_S=.0005
TRANSFER_REQUEST_CAP_M=.002
MINIMUM_NET_LIFT_M=.03
ORIGINAL_HOLD_S=5.
FIXED_ANCHOR_LIMIT_M=.0005
ORIGINAL_TRANSLATION_MAX_M_S=.01
STYLES=('pinch_thumb_index','tripod_thumb_index_middle')
QUALIFICATION=MappingProxyType(dict(control_admissible=False,geometry_qualified=False,contact_model_qualified=False,
    support_qualified=False,planning_eligible=False,native_attested=False,native_success_qualification_issued=False,
    new_expert_qualification_issued=False,hardware_measurement_qualified=False))
_BOOLS=('approach_ready','tracking_safe','object_following','tilt_safe','slip_safe','support_safe','airborne',
    'transfer_complete','recovery_ready','mandatory_unload','fatal_failure','original_hold_qualified','terminal_success')
_NUMERIC=dict(hand_upward_support=(2,'N'),table_reaction=(3,'N'),end_support_reaction=(3,'N'),
    net_lift_m=(None,'m'),fixed_anchor_drift_m=(None,'m'),original_hold_duration_s=(None,'s'))
_NAMES=frozenset((*_BOOLS,'bilateral_opposed',*_NUMERIC))


class Phase(str,Enum):
    APPROACH='APPROACH'
    CLOSE='CLOSE'
    STABILIZE='STABILIZE'
    LOAD_TRANSFER='LOAD_TRANSFER'
    LIFT='LIFT'
    AIRBORNE_HOLD='AIRBORNE_HOLD'
    RECOVERY='RECOVERY'
    FAILED='FAILED'


@dataclass(frozen=True)
class LoadTransferConfig:
    lift_rate_m_s: float=.001
    lift_speed_evidence_source: str | None=None
    lift_speed_evidence_sha256: str | None=None
    def __post_init__(self):
        rate=self.lift_rate_m_s
        if isinstance(rate,bool) or not isinstance(rate,Real) or not math.isfinite(rate) or not 0<rate<=.003:
            raise ValueError('bounded experimental lift speed in (0,3mm/s] required; original 10mm/s limit unchanged')
        if rate>.001 and (not isinstance(self.lift_speed_evidence_source,str) or not self.lift_speed_evidence_source.strip()
                or not isinstance(self.lift_speed_evidence_sha256,str) or len(self.lift_speed_evidence_sha256)!=64
                or any(c not in '0123456789abcdef' for c in self.lift_speed_evidence_sha256)):
            raise ValueError('above-default lift speed requires explicit independent experimental evidence identity')


@dataclass(frozen=True)
class LoadTransferFeedback:
    clock: SampleClock
    grasp_style: str
    fields: object
    def __post_init__(self):
        if not isinstance(self.clock,SampleClock) or self.grasp_style not in STYLES:
            raise ValueError('actual clock and explicit supported grasp style required')
        if not isinstance(self.fields,Mapping) or set(self.fields)!=_NAMES or any(not isinstance(v,StateField) for v in self.fields.values()):
            raise ValueError('complete sourced StateField observations required, declaring missing values explicitly')
        object.__setattr__(self,'fields',MappingProxyType(dict(self.fields)))


@dataclass(frozen=True)
class LoadTransferRequest:
    clock: SampleClock
    phase: Phase
    enabled: bool
    reason: str
    common_ascent_request_m_s: float
    stable_contact_duration_s: float
    transfer_requested_exposure_m: float
    request_measured_hold: bool
    request_pressure_recovery: bool
    finished: bool
    original_observer_reported_success: bool | None
    field_evidence: object
    config: LoadTransferConfig
    version: str=VERSION
    request_interval_s: float=PHYSICS_DT_S
    request_source: str='Stage7 section7 experimental state-machine suggestion; original execution guards required'
    transfer_is_measured_displacement: bool=False
    hold_issued: bool=False
    sdk_targets_issued: bool=False
    actor_observation_changed: bool=False
    reward_changed: bool=False
    preserve_original_pressure_and_unload: bool=True
    counts_as_student_success: bool=False
    def __post_init__(self):object.__setattr__(self,'field_evidence',MappingProxyType(dict(self.field_evidence)))
    @property
    def qualification(self):return QUALIFICATION


def _values(feedback):
    c=feedback.clock;values={}
    for name,f in feedback.fields.items():
        length,unit=_NUMERIC.get(name,(None,'dimensionless'))
        frame='world' if unit in ('N','m') else 'not_applicable'
        if (f.clock!=c or f.env_id!=c.env_id or f.episode_uid!=c.episode_uid or f.unit!=unit or f.frame!=frame
                or f.force_side!=('ON_OBJECT' if unit=='N' else 'NA')):
            raise ValueError('same-clock/env/episode/unit/frame/source-side observation required: '+name)
        if name in ('bilateral_opposed','hand_upward_support','support_safe') and f.contact_basis!='actual_contact':
            raise ValueError('actual-contact predicate required; geometry prediction is not contact: '+name)
        if not f.valid:
            values[name]=None;continue
        if f.timestamp_s!=c.world_time_s or f.kind not in ('measured','derived'):
            raise ValueError('fresh dynamic physical observation required: '+name)
        if name in ('table_reaction','end_support_reaction') and f.kind!='measured':
            raise ValueError('actual reaction measurement required; inferred mass*g residual is not a reaction')
        value=f.value
        if name in _BOOLS:
            if type(value) is not bool:raise ValueError('actual boolean predicate required: '+name)
        elif name=='bilateral_opposed':
            if not isinstance(value,tuple) or len(value)!=2 or any(type(v) is not bool for v in value):
                raise ValueError('separate left/right actual opposition booleans required')
        else:
            numbers=(value,) if length is None else value
            if length is not None and (not isinstance(value,tuple) or len(value)!=length):
                raise ValueError('complete per-hand/reaction observation shape required: '+name)
            if any(isinstance(v,bool) or not isinstance(v,Real) for v in numbers):
                raise ValueError('numeric physical observation required: '+name)
            if any(not math.isfinite(v) for v in numbers):
                values[name]=None;continue
            if name in ('fixed_anchor_drift_m','original_hold_duration_s') and value<0:
                raise ValueError('nonnegative original observer data required: '+name)
        values[name]=value
    return values


class BimanualLoadTransfer:
    """One env/episode, no implicit reset, actuator write or automatic 3mm/s ramp.

    object_following must be a valid external actual-state predicate even on
    entry. Unknown is never assumed stationary/following. Requested exposure
    accounts suggested positive transfer intervals, NOT native execution or
    actual palm displacement; recovery never replenishes this request ledger.
    """
    def __init__(self,*,enabled=False,grasp_style='pinch_thumb_index',config=None):
        if type(enabled) is not bool or grasp_style not in STYLES:raise ValueError('explicit enable/style required')
        config=LoadTransferConfig() if config is None else config
        if not isinstance(config,LoadTransferConfig):raise ValueError('explicit LoadTransferConfig required')
        self.enabled=enabled;self.grasp_style=grasp_style;self.config=config
        self._phase=Phase.APPROACH;self._clock=None;self._stable_intervals=0;self._exposure=0.
        self._closed=None;self._reset_pending=False;self._seen_episode_uids=set()
    @property
    def phase(self):return self._phase
    @property
    def transfer_requested_exposure_m(self):return self._exposure

    def reset(self,clock,*,grasp_style):
        if not isinstance(clock,SampleClock) or not clock.reset or clock.terminal or grasp_style not in STYLES:
            raise ValueError('explicit actual new-episode reset clock/style required')
        if self._clock is not None and (clock.run_uuid!=self._clock.run_uuid or clock.env_id!=self._clock.env_id
                or clock.episode_uid in self._seen_episode_uids or clock.reset_generation<=self._clock.reset_generation
                or clock.world_step<self._clock.world_step or clock.sdk_step<self._clock.sdk_step):
            raise ValueError('reset must identify a fresh generation of the same actual env/run')
        if self._clock is not None:
            steps=clock.world_step-self._clock.world_step;expected=steps*PHYSICS_DT_S
            if (clock.sdk_step-self._clock.sdk_step!=steps or clock.policy_step<self._clock.policy_step
                    or clock.world_time_s<self._clock.world_time_s or clock.sdk_time_s<self._clock.sdk_time_s
                    or not math.isclose(clock.world_time_s-self._clock.world_time_s,expected,rel_tol=0,abs_tol=1e-9)
                    or not math.isclose(clock.sdk_time_s-self._clock.sdk_time_s,expected,rel_tol=0,abs_tol=1e-9)):
                raise ValueError('same-World reset must preserve actual World/SDK step-time and monotonic policy clocks')
        self.grasp_style=grasp_style;self._clock=clock;self._phase=Phase.APPROACH
        self._stable_intervals=0;self._exposure=0.;self._closed=None;self._reset_pending=True
        self._seen_episode_uids.add(clock.episode_uid)

    def _continuity(self,c):
        old=self._clock
        if old is None:return
        if (c.run_uuid,c.env_id,c.episode_uid,c.reset_generation)!=(old.run_uuid,old.env_id,old.episode_uid,old.reset_generation):
            raise ValueError('explicit reset required before another env/episode/generation')
        if self._closed is not None:return
        if (c.reset or c.world_step!=old.world_step+1 or c.sdk_step!=old.sdk_step+1
                or not math.isclose(c.world_time_s-old.world_time_s,PHYSICS_DT_S,rel_tol=0,abs_tol=1e-9)
                or not math.isclose(c.sdk_time_s-old.sdk_time_s,PHYSICS_DT_S,rel_tol=0,abs_tol=1e-9)
                or c.policy_step not in (old.policy_step,old.policy_step+1)):
            raise ValueError('contiguous original 480Hz World/SDK physical observations required')

    def _request(self,f,reason,*,rate=0.,hold=False,recovery=False,finished=False,reported=None):
        result=LoadTransferRequest(f.clock,self._phase,self.enabled,reason,rate,self._stable_intervals*PHYSICS_DT_S,
            self._exposure,hold,recovery,finished,reported,f.fields,self.config)
        if finished:self._closed=result
        return result

    @staticmethod
    def _safe(v):
        return all(v[n] is True for n in ('tracking_safe','object_following','tilt_safe','slip_safe','support_safe'))

    @classmethod
    def _original_standard(cls,v):
        names=('bilateral_opposed','tracking_safe','object_following','tilt_safe','slip_safe','support_safe','airborne',
            'mandatory_unload','fatal_failure','hand_upward_support','table_reaction','end_support_reaction',
            'net_lift_m','fixed_anchor_drift_m','original_hold_duration_s','original_hold_qualified')
        return (all(v[n] is not None for n in names) and cls._safe(v) and all(v['bilateral_opposed'])
            and v['airborne'] is True and v['mandatory_unload'] is False and v['fatal_failure'] is False
            and v['net_lift_m']>=MINIMUM_NET_LIFT_M and v['fixed_anchor_drift_m']<=FIXED_ANCHOR_LIMIT_M
            and v['original_hold_duration_s']>=ORIGINAL_HOLD_S and v['original_hold_qualified'] is True)

    def step(self,feedback):
        if not isinstance(feedback,LoadTransferFeedback) or feedback.grasp_style!=self.grasp_style:
            raise ValueError('matching owned feedback/style required')
        if not self.enabled:return self._request(feedback,'disabled')
        c=feedback.clock;self._continuity(c)
        if self._closed is not None:return self._closed  # First terminal/fatal evidence is never a later no-op.
        v=_values(feedback);first=self._clock is None;self._clock=c;self._seen_episode_uids.add(c.episode_uid)
        if v['fatal_failure'] is True:
            self._phase=Phase.FAILED;self._stable_intervals=0
            return self._request(feedback,'original_fatal_failure',finished=True,reported=v['terminal_success'] if c.terminal else None)
        if c.terminal:
            reported=v['terminal_success']
            self._phase=(Phase.FAILED if reported is False else Phase.AIRBORNE_HOLD
                if reported is True and self._original_standard(v) else Phase.RECOVERY)
            return self._request(feedback,'original_terminal_observer_retained' if self._phase==Phase.AIRBORNE_HOLD else
                'original_failed_terminal' if reported is False else 'terminal_outcome_unknown_or_incomplete',finished=True,reported=reported)
        initial=first or self._reset_pending;self._reset_pending=False
        missing=[n for n,value in v.items() if value is None]
        contact_lost=self._phase in (Phase.STABILIZE,Phase.LOAD_TRANSFER,Phase.LIFT,Phase.AIRBORNE_HOLD) and not (
            v['bilateral_opposed'] is not None and all(v['bilateral_opposed']))
        if missing or v['mandatory_unload'] is True or not self._safe(v) or contact_lost:
            self._phase=Phase.RECOVERY;self._stable_intervals=0
            return self._request(feedback,'missing:'+','.join(missing) if missing else 'original_unload_or_actual_motion_gate_lost',hold=True,recovery=True)
        if initial:return self._request(feedback,'initial_actual_sample_no_elapsed_interval')
        if self._phase==Phase.APPROACH:
            if v['approach_ready']:self._phase=Phase.CLOSE
        elif self._phase==Phase.CLOSE:
            if all(v['bilateral_opposed']):self._phase=Phase.STABILIZE;self._stable_intervals=0
        elif self._phase==Phase.STABILIZE:
            self._stable_intervals+=1
            if self._stable_intervals>=STABILIZE_INTERVALS:self._phase=Phase.LOAD_TRANSFER
        elif self._phase==Phase.RECOVERY:
            if v['recovery_ready'] and all(v['bilateral_opposed']):self._phase=Phase.STABILIZE;self._stable_intervals=0
            else:return self._request(feedback,'await_original_pressure_recovery',hold=True,recovery=True)
        elif self._phase==Phase.LOAD_TRANSFER and v['transfer_complete'] and self._exposure>0:
            self._phase=Phase.LIFT
        if self._phase in (Phase.LIFT,Phase.AIRBORNE_HOLD):
            if v['net_lift_m']>=MINIMUM_NET_LIFT_M and v['airborne']:
                self._phase=Phase.AIRBORNE_HOLD
                if v['fixed_anchor_drift_m']<=FIXED_ANCHOR_LIMIT_M:
                    return self._request(feedback,'original_airborne_hold_observer_required',hold=True)
            if v['net_lift_m']>=MINIMUM_NET_LIFT_M or self._phase==Phase.AIRBORNE_HOLD:
                self._phase=Phase.RECOVERY;self._stable_intervals=0
                return self._request(feedback,'hold_height_airborne_or_fixed_anchor_gate_lost',hold=True,recovery=True)
            return self._request(feedback,'bounded_lift_request',rate=float(self.config.lift_rate_m_s))
        if self._phase==Phase.LOAD_TRANSFER:
            remaining=TRANSFER_REQUEST_CAP_M-self._exposure
            if remaining<=0:return self._request(feedback,'transfer_request_exposure_cap_wait_actual_complete',hold=True)
            distance=min(remaining,TRANSFER_RATE_M_S*PHYSICS_DT_S)
            self._exposure=TRANSFER_REQUEST_CAP_M if distance==remaining else self._exposure+distance
            return self._request(feedback,'bounded_load_transfer_request_not_execution',rate=distance/PHYSICS_DT_S)
        return self._request(feedback,'phase_observation_only')
