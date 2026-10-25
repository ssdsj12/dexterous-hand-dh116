"""Bounded pressure feedback from measured pad/object loads, never commanded effort."""
import numpy as np
from .tabletop_load_parameters import LEGACY_LOAD_PARAMETERS

MIN_FORCE = .1
MAX_FORCE = 1.
OFFSET_LIMIT = .08
OFFSET_RATE = .03
FORCE_RATE = .5
FILTER_TAU = .05
DEADBAND = .02


def _array(value, shape, name):
    a=np.asarray(value,dtype=float)
    if a.shape!=shape or not np.isfinite(a).all():
        raise ValueError(f'finite {name} with shape {shape} required')
    return a


def closure_direction(value):
    direction=_array(value,(2,6),'closure direction')
    if not np.allclose(np.linalg.norm(direction,axis=1),1.,atol=1e-6,rtol=0):
        raise ValueError('each calibrated closure direction must have unit norm')
    return direction.copy()


class ReactionForceController:
    def __init__(self, parameters=None, *, full_hand_contact=False):
        self.full_hand_contact = bool(full_hand_contact)
        self.parameters = parameters or LEGACY_LOAD_PARAMETERS
        self.reset()

    def reset(self):
        self.offset=np.zeros(2)
        self.filtered=np.zeros(2)
        self.target=np.full(2,MIN_FORCE)
        self.requested=np.full(2,MIN_FORCE)
        self.boost=np.zeros(2)
        self.last_slip=0.
        self.blocked=np.zeros(2,dtype=bool)

    def update(self, *, action, forces, blocked, slip_m, phase, dt):
        action=_array(action,(2,),'force action')
        forces=_array(forces,(2,) if self.full_hand_contact else (4,),'measured object loads')
        blocked=_array(blocked,(2,),'coupling blocked')
        if (np.any(abs(action)>1) or np.any(forces<0) or np.any((blocked!=0)&(blocked!=1))
                or not np.isscalar(dt) or not np.isfinite(dt) or not 0<dt<=1/60
                or not np.isscalar(slip_m) or not np.isfinite(slip_m) or slip_m<0
                or phase not in ('close','hold','lift')):
            raise ValueError('invalid force-feedback action, measurements or clock')
        loads=forces if self.full_hand_contact else forces.reshape(2,2).sum(axis=1)
        self.filtered += (1.-np.exp(-dt/FILTER_TAU))*(loads-self.filtered)
        maximum=self.parameters.max_force_n
        deadband=self.parameters.deadband_n
        nominal=MIN_FORCE+(action+1.)*.5*(maximum-MIN_FORCE)
        # Constant displacement is not continuing slip. No contact means there
        # is no supported object whose slip could justify a pressure boost.
        if phase=='lift' and slip_m>.002 and (slip_m-self.last_slip)/dt>.002:
            self.boost += np.where(loads>.02,.25*dt,0.)
        self.last_slip=float(slip_m)
        self.boost=np.clip(self.boost,0.,maximum-nominal)
        self.requested=np.clip(nominal+self.boost,MIN_FORCE,maximum)
        rate=self.parameters.force_rate_n_s
        self.target += np.clip(self.requested-self.target,-rate*dt,rate*dt)
        error=self.target-self.filtered
        # Respond immediately to excessive live load rather than waiting for
        # the low-pass state; deadband otherwise holds the achieved pressure.
        error=np.where(loads>self.target+deadband,self.target-loads,error)
        error=np.where(abs(error)<=deadband,0.,error)
        movement=np.clip(.1*error,-OFFSET_RATE,OFFSET_RATE)*dt
        self.blocked=blocked.astype(bool)
        movement=np.where(self.blocked & (movement>0.),0.,movement)
        self.offset=np.clip(self.offset+movement,-OFFSET_LIMIT,OFFSET_LIMIT)

    def ready(self, forces, *, speed, slip):
        loads=_array(forces,(2,) if self.full_hand_contact else (4,),'measured object loads')
        if not self.full_hand_contact: loads=loads.reshape(2,2).sum(axis=1)
        # A regulation warning is handled by command recovery and palm gating.
        # It does not erase independently measured stable load; hard failures
        # preempt the task update in TabletopEnv/TabletopTask.
        return bool(np.isfinite([speed,slip]).all() and 0<=speed<.02 and 0<=slip<=.003
            and np.all(loads>=.8*self.requested)
            and np.all(loads<=self.requested*1.25+self.parameters.deadband_n))

    def lift_allowed(self, forces, *, slip_m, dt):
        loads=_array(forces,(2,) if self.full_hand_contact else (4,),'measured object loads')
        if not self.full_hand_contact: loads=loads.reshape(2,2).sum(axis=1)
        slipping=slip_m>.002 and (slip_m-self.last_slip)/dt>.002
        return bool(np.all(loads>=.8*self.requested) and not slipping)

    def observation(self):
        maximum=self.parameters.max_force_n
        return np.r_[self.offset/OFFSET_LIMIT,self.filtered/maximum,
            self.target/maximum,self.boost/maximum,self.requested/maximum,
            self.last_slip/.02].astype(np.float32)

    def diagnostics(self):
        return dict(target_force_n=self.target.tolist(),requested_force_n=self.requested.tolist(),
            filtered_hand_load_n=self.filtered.tolist(),closure_offset_rad=self.offset.tolist(),
            slip_boost_n=self.boost.tolist(),force_control_blocked=self.blocked.tolist())


def force_feedback_contract(parameters=None):
    contract = dict(observation_version='dh116-reaction-force-v1',observation_size=136,
        action='20D_joint_palm_reaction_force_v1',grip_control='measured_reaction_force_v1',
        force_feedback=dict(target_hand_load_range_n=[MIN_FORCE,MAX_FORCE],
            measurement='sum_of_two_pad_object_force_magnitudes_per_hand',
            target_rate_n_s=FORCE_RATE,filter_tau_s=FILTER_TAU,deadband_n=DEADBAND,
            closure_offset_limit_rad=OFFSET_LIMIT,closure_rate_rad_s=OFFSET_RATE,
            feedback_gain_rad_per_n_s=.1,slip_boost_n_s=.25,
            combined_joint_radius_rad=.15,drive_stiffness_target=20.,
            readiness_fraction=.8,readiness_upper_factor=1.25,readiness_slip_m=.003,
            readiness_speed_m_s=.02,lift_pause_load_fraction=.8,
            lift_pause_slip_m=.002,lift_pause_slip_speed_m_s=.002,close_dwell_s=.2,hold_dwell_s=.3,acquisition_timeout_s=6.),
        lift_attempt_after_s=None,lift_gate='measured_hand_load_stability_v3',
        admission='safe_repeatable_close_exploration_v1',initial_force_action_mean=-.95)
    if parameters is not None:
        contract.update(observation_version='dh116-reaction-force-load-v2',
            load_parameters=parameters.as_dict(), active_stiffness_range=[10.,parameters.stiffness_max],
            active_stiffness_rate_per_s=parameters.stiffness_rate,
            active_effort_limit_nm=parameters.effort_limit_nm)
        contract['force_feedback'].update(target_hand_load_range_n=[MIN_FORCE,parameters.max_force_n],
            target_rate_n_s=parameters.force_rate_n_s,deadband_n=parameters.deadband_n,
            drive_stiffness_target=parameters.stiffness_max)
    return contract


def initialize_force_policy(policy):
    """Only fresh policies: start at action mean -.95 near the load lower bound.

    This requests .1225 N with the legacy 1 N range and .1975 N with 4 N.

    All weights, other action biases, exploration variance and optimizer state
    remain intact. Resume loads its learned weights and never calls this.
    """
    import torch
    layer=policy.action_net[0]
    if layer.out_features != 20:
        raise ValueError('reaction-force policy requires twenty actions')
    with torch.no_grad():
        layer.bias[-2:].fill_(float(np.arctanh(-.95)))
