"""Gym task orchestration over a measured, single-scene physics backend.

Backend contract: ``reset(offset)``, ``advance(offset, command, phase=...)``,
``measure()``; dt, decimation, and calibrated 12D grasp attributes. Only the
backend may touch physics. This class never assigns object state or contact.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .tabletop_control import (OBSERVATION_SIZE, bounded_residual_target,
    build_observation, curriculum_complete, progress_reward, prepared_approach_reference)
from .tabletop_task import (TabletopTask,
                           advance_root_offset, guarded_descent)
from .tabletop_training import PHYSICAL_FAILURES, STAGES


class TabletopEnv(gym.Env):
    action_space = gym.spaces.Box(-1., 1., (12,), dtype=np.float32)
    observation_space = gym.spaces.Box(-np.inf, np.inf, (OBSERVATION_SIZE,), dtype=np.float32)

    def __init__(self, backend, *, stage='approach', initial_offset=.04,
                 reset_jitter=.001, episode_sink=None, residual_scale=.02,
                 reset_phase='approach', control_mode='residual', target_radius=.08,
                 grasp_mode='four_pad', force_direction=None, trace_steps=True,
                 force_palm_compensation=None, load_parameters=None):
        self.backend = backend
        self.trace_steps = bool(trace_steps)
        if reset_phase not in ('approach', 'close') or control_mode not in ('residual', 'joint_target', 'adaptive_grasp'):
            raise ValueError('unknown reset phase/control mode')
        self.direct_joint_control = control_mode in ('joint_target', 'adaptive_grasp')
        if self.direct_joint_control and reset_phase != 'close':
            raise ValueError('direct finger training requires close reset')
        self.reset_phase, self.control_mode = reset_phase, control_mode
        if grasp_mode not in ('four_pad', 'object_stability'):
            raise ValueError('unknown grasp mode')
        if control_mode == 'adaptive_grasp' and grasp_mode != 'object_stability':
            raise ValueError('adaptive grasp requires object stability success criteria')
        self.grasp_mode = grasp_mode
        self.target_radius = float(target_radius)
        if self.direct_joint_control:
            from .tabletop_direct import joint_target
            joint_target(backend.grasp, np.zeros(12), self.target_radius)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (108,), dtype=np.float32)
        if control_mode == 'adaptive_grasp':
            from .tabletop_adaptive import ACTION_SIZE, OBSERVATION_SIZE as ADAPTIVE_OBSERVATION_SIZE
            self.action_space = gym.spaces.Box(-1., 1., (ACTION_SIZE,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (ADAPTIVE_OBSERVATION_SIZE,), dtype=np.float32)
        self.force_controller = None
        if force_direction is not None:
            if control_mode != 'adaptive_grasp':
                raise ValueError('force feedback requires adaptive grasp')
            from .tabletop_force_feedback import ReactionForceController, closure_direction
            self.force_direction = closure_direction(force_direction)
            self.force_controller = ReactionForceController(load_parameters)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (136,), dtype=np.float32)
        self.force_palm_compensation = None
        if load_parameters is not None and self.force_controller is None:
            raise ValueError('load parameters require force feedback')
        if force_palm_compensation is not None:
            from .tabletop_pressure_geometry import compensated_palm_action
            if self.force_controller is None:
                raise ValueError('pressure geometry requires force feedback')
            compensated_palm_action(np.zeros((2, 3)), np.zeros(2), force_palm_compensation)
            self.force_palm_compensation = np.asarray(force_palm_compensation, dtype=float).copy()
        self.initial_offset = float(initial_offset)
        self.reset_jitter = float(reset_jitter)
        self.residual_scale = float(residual_scale)
        if not np.isfinite(residual_scale) or not 0 < residual_scale <= .1:
            raise ValueError('residual radius must be in (0,0.1] rad')
        if not 0 <= reset_jitter <= .002 or not np.isfinite(initial_offset) or initial_offset <= reset_jitter:
            raise ValueError('positive offset and reset height jitter in [0,0.002] required')
        self.set_stage(stage)
        self.mode = 'evaluation'
        self.episode_sink = episode_sink
        self.log = []
        self.closed = True

    def set_stage(self, stage):
        if stage not in STAGES:
            raise ValueError('unknown curriculum stage')
        if self.reset_phase == 'close' and stage == 'approach':
            raise ValueError('close reset cannot qualify approach')
        if getattr(self, 'stage', None) != stage:
            self.flush_episode()
        self.stage = stage

    def set_mode(self, mode):
        if self.mode != mode:
            self.flush_episode()
        self.mode = mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.flush_episode()
        self.task = TabletopTask(phase=self.reset_phase, grasp_mode=self.grasp_mode,
                                force_feedback=self.force_controller is not None)
        if self.force_controller is not None:
            self.force_controller.reset()
        self.pressure_palm_target = np.zeros((2, 3))
        self.pressure_palm_clipped = np.zeros((2, 3), dtype=bool)
        self.offset = self.initial_offset + self.np_random.uniform(-self.reset_jitter, self.reset_jitter)
        self.close_progress = 0.
        self.descent_stop = None
        self.reference = np.concatenate([prepared_approach_reference(g)
                                         for g in self.backend.grasp.reshape(2, 6)])
        self.command = self.reference.copy()
        if self.direct_joint_control:
            self.reference = self.backend.grasp.copy()
            self.command = np.asarray(self.backend.initial_command).copy()
        self.filtered_residual = np.zeros(12)
        self.previous = self.command.copy()
        self.grip_relative = None
        self.grip_regions = None
        self.reached_phases = [self.reset_phase]
        self.log = []
        self.return_ = 0.
        self.closed = False
        self.backend.reset(self.offset)
        self._measure()
        self.last_observation = self.observe()
        if self.measurement['joint_failure']:
            raise ValueError(f"unsafe reset: {self.measurement['joint_failure']}")
        if (self.metrics['clearance_m'] < -.001 or self.metrics['max_force_n'] > 30
                or self.metrics['object_bottom_m'] < .29):
            raise ValueError('unsafe reset: physical geometry/load')
        return self.last_observation.copy(), self._info()

    def _measure(self):
        m = self.backend.measure()
        slip = 0. if self.grip_relative is None else float(np.linalg.norm(
            np.asarray(m['object_relative']) - self.grip_relative))
        for key, shape in (('material_relative', (2, 3)), ('material_speed', (2,))):
            if key in m:
                value = np.asarray(m[key], dtype=float)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f'finite {key} with shape {shape} required')
                if key == 'material_speed' and np.any(value < 0.):
                    raise ValueError('nonnegative material_speed required')
                m[key] = value
        if self.grip_regions is not None and 'material_relative' in m:
            slip = float(np.max(np.linalg.norm(m['material_relative'] - self.grip_regions, axis=1)))
        speed = (float(np.max(m['material_speed'])) if 'material_speed' in m
                 else float(np.linalg.norm(m['object_velocity'][:3])))
        metrics = dict(approach_error_m=m['approach_error'], pad_forces_n=m['forces'],
            clearance_m=m['clearance'], object_bottom_m=m['bottom'],
            object_speed_m_s=speed,
            slip_m=slip, max_force_n=float(np.max(m['forces'])))
        if not np.isfinite(np.r_[metrics['approach_error_m'], metrics['pad_forces_n'],
                metrics['clearance_m'], metrics['object_bottom_m'],
                metrics['object_speed_m_s'], slip, metrics['max_force_n']]).all():
            raise ValueError('nonfinite physical measurement')
        if self.control_mode == 'adaptive_grasp':
            from .tabletop_adaptive import measured_state
            measured_state(m)
        self.measurement, self.metrics = m, metrics

    def observe(self):
        m = self.measurement
        observation = build_observation(q=m['q'], qd=m['qd'], previous=self.previous,
            reference=self.reference, forces=m['forces'], task=self.task, offset=self.offset,
            close_progress=self.close_progress, descent_stop=self.descent_stop,
            clearance=m['clearance'], object_relative=m['object_relative'],
            object_quat=m['object_quat'], object_velocity=m['object_velocity'],
            grip_relative=self.grip_relative, filtered_residual=self.filtered_residual,
            mimic_errors=m.get('mimic_errors', np.zeros(10)))
        if self.control_mode == 'joint_target':
            adjustment = np.asarray(m.get('palm_adjustment', np.zeros(4)), dtype=np.float32)
            if adjustment.shape != (4,) or not np.isfinite(adjustment).all():
                raise ValueError('finite palm adjustment required')
            observation = np.r_[observation, adjustment/.003].astype(np.float32)
        elif self.control_mode == 'adaptive_grasp':
            from .tabletop_adaptive import observation_tail
            observation = np.r_[observation, observation_tail(m)].astype(np.float32)
        if self.direct_joint_control:
            # Resolve the 0.1 N grasp threshold without using the 30 N hard
            # guard as the learning signal's normalization scale.
            observation[48:52] = np.log1p(np.maximum(m['forces'], 0.)/.1)
        if self.force_controller is not None:
            observation = np.r_[observation, self.force_controller.observation()].astype(np.float32)
        if self.control_mode == 'adaptive_grasp' and not np.isfinite(observation).all():
            raise ValueError('adaptive observation must be finite')
        return observation

    def _info(self):
        info = dict(is_success=self.task.success, failure=self.task.failure, phase=self.task.phase,
            stage_success=curriculum_complete(self.stage, self.task),
            reached_phases=list(self.reached_phases),
            safety_failure=self.task.failure if self.task.failure in PHYSICAL_FAILURES else None)
        if self.force_controller is not None:
            info['force_feedback'] = self.force_controller.diagnostics()
            info['force_feedback']['measured_hand_load_n'] = np.asarray(self.metrics['pad_forces_n']).reshape(2,2).sum(axis=1).tolist()
        if self.force_palm_compensation is not None:
            info['pressure_geometry'] = dict(
                requested_compensation_m=(self.force_controller.offset[:, None]*self.force_palm_compensation).tolist(),
                combined_palm_target_m=self.pressure_palm_target.tolist(),
                palm_target_error_m=(self.pressure_palm_target-self.measurement['adaptive_palm_offset']).tolist(),
                target_clipped=self.pressure_palm_clipped.tolist())
        return info

    def flush_episode(self):
        if not getattr(self, 'log', []):
            return
        info = self._info()
        summary = dict(info, mode=self.mode, stage=self.stage, complete=self.closed,
            phases=list(self.reached_phases), return_=self.return_,
            final_bottom_m=self.metrics['object_bottom_m'],
            min_clearance_m=min(row['clearance_m'] for row in self.log))
        if self.episode_sink:
            self.episode_sink(summary, self.log)
        self.log = []

    def step(self, residual):
        """Run one control tick against the original standalone backend."""
        iterator = self.step_iter(residual)
        while True:
            try:
                offset, command, kwargs = next(iterator)
            except StopIteration as ended:
                return ended.value
            self.backend.advance(offset, command, **kwargs)

    def step_iter(self, residual):
        """Yield each command before physics, then consume fresh measurements.

        A shared-world caller must apply all environments' yielded requests,
        advance physics once, refresh backend data, and only then resume these
        generators. Return value is the usual Gym transition, including early
        termination at the measured substep that caused it.
        """
        if self.closed:
            raise RuntimeError('episode ended; reset before stepping')
        dt = self.backend.dt
        reward = 0.
        terms = {}
        used_time = 0.
        nonfinite = False
        coupling_pause = False
        previous_command = self.command.copy()
        try:
            finger_action = residual
            if self.control_mode == 'adaptive_grasp':
                from .tabletop_adaptive import split_action
                finger_action, palm_action, grip_action = split_action(residual)
            bounded_residual_target(self.reference, finger_action, self.residual_scale)
        except ValueError:
            self.task.failure = 'invalid_action'
        for _ in range(self.backend.decimation if not self.task.failure else 0):
            phase = self.task.phase
            before = self.metrics
            warning = .004 if self.direct_joint_control else .008
            coupling_pause = bool(np.max(np.abs(self.measurement.get('mimic_errors', np.zeros(10)))) >= warning)
            tracking_ready = bool(np.max(np.abs(self.command-self.measurement['q'])) < .03
                                  and (self.direct_joint_control
                                       or np.max(np.abs(self.reference-self.command)) < .03))
            if phase == 'approach':
                contact = np.any(self.measurement['forces'] >= .1)
                if contact:
                    self.descent_stop = 'pad_contact'
                elif self.measurement['clearance'] <= .002:
                    self.descent_stop = 'table_margin'
                if self.descent_stop is None and tracking_ready and not coupling_pause:
                    self.offset = guarded_descent(self.offset, dt/2.,
                        self.measurement['clearance']-.0001, self.measurement['forces'])
                self.reference = np.concatenate([prepared_approach_reference(g)
                    for g in self.backend.grasp.reshape(2, 6)])
            elif phase == 'close' and not coupling_pause and self.control_mode == 'residual':
                # Follow the actual calibrated pose, independently of PPO.
                # Generic joint-range percentages do not preserve this pose.
                if tracking_ready and not np.any(self.measurement['forces'] >= .1):
                    self.descent_stop = None
                    self.offset = guarded_descent(self.offset, dt/4.,
                        self.measurement['clearance']-.0001, self.measurement['forces'])
                self.close_progress = min(1., self.close_progress + dt/4.)
                self.reference += np.clip(self.backend.grasp-self.reference, -.15*dt, .15*dt)
            elif phase == 'lift' and not coupling_pause and tracking_ready:
                if (self.force_controller is None or self.force_controller.lift_allowed(
                        self.metrics['pad_forces_n'],slip_m=self.metrics['slip_m'],dt=dt)):
                    self.offset = advance_root_offset(self.offset, phase, dt)
            # Limit residual bandwidth separately from nominal reference
            # motion. Expose this actuator state to the policy (v3 observation).
            if not coupling_pause or self.direct_joint_control:
                if self.direct_joint_control:
                    from .tabletop_direct import joint_target, coupling_recovery_target
                    target = joint_target(self.reference, finger_action, self.target_radius)
                    if self.force_controller is not None:
                        errors = np.asarray(self.measurement['mimic_errors']).reshape(2, 5)
                        self.force_controller.update(action=grip_action,
                            forces=self.metrics['pad_forces_n'],
                            blocked=np.max(np.abs(errors), axis=1) >= warning,
                            slip_m=self.metrics['slip_m'], phase=phase, dt=dt)
                        adjusted = target + (self.force_controller.offset[:, None]*self.force_direction).ravel()
                        target = joint_target(self.reference,
                            np.clip((adjusted-self.reference)/.15, -1., 1.), .15)
                    if coupling_pause:
                        errors = np.asarray(self.measurement.get('mimic_errors', np.zeros(10)))
                        # Move the loaded master toward the angle consistent
                        # with its measured follower, removing coupling strain.
                        # Other fingers remain controllable; palms still wait.
                        target = coupling_recovery_target(target, self.measurement['q'], errors, warning)
                    self.command += np.clip(target-self.command, -.05*dt, .05*dt)
                    self.filtered_residual = self.command-self.reference
                else:
                    self.filtered_residual += np.clip(self.residual_scale*np.asarray(residual)
                        - self.filtered_residual, -.05*dt, .05*dt)
                    self.filtered_residual = np.clip(self.filtered_residual,
                                                     -self.residual_scale, self.residual_scale)
                    target = bounded_residual_target(self.reference,
                        self.filtered_residual/self.residual_scale, self.residual_scale)
                    self.command += np.clip(target-self.command, -.3*dt, .3*dt)
            request = dict(phase=phase, palm_motion_allowed=tracking_ready and not coupling_pause)
            if self.control_mode == 'adaptive_grasp':
                requested_palm = palm_action.copy()
                if self.force_palm_compensation is not None:
                    from .tabletop_pressure_geometry import compensated_palm_action
                    from .tabletop_adaptive_backend import PALM_LIMIT_M
                    requested_palm, self.pressure_palm_clipped = compensated_palm_action(
                        palm_action, self.force_controller.offset, self.force_palm_compensation)
                    self.pressure_palm_target = requested_palm*PALM_LIMIT_M
                request.update(palm_action=requested_palm,
                    grip_action=np.ones(2) if self.force_controller is not None else grip_action.copy())
            yield (self.offset, self.command.copy(), request)
            used_time += dt
            try:
                self._measure()
                self.observe()  # Reject invalid velocities/quaternions too.
            except ValueError:
                self.task.failure = 'nonfinite_state'
                self.metrics = before
                nonfinite = True
                break
            if self.measurement['joint_failure']:
                self.task.failure = self.measurement['joint_failure']
            contact = bool(np.any(self.measurement['forces'] >= .1))
            blocked = self.measurement['clearance'] <= .002 and not (
                contact and self.measurement['clearance'] >= .002)
            # Handoff must use the resulting physical state, not the state
            # that permitted commands before this simulation step.
            handoff_ready = bool(
                np.max(np.abs(self.measurement.get('mimic_errors', np.zeros(10)))) < .008
                and np.max(np.abs(self.command-self.measurement['q'])) < .03
                and np.max(np.abs(self.reference-self.command)) < .03)
            grasp_ready = False
            if self.force_controller is not None:
                grasp_ready = self.force_controller.ready(self.metrics['pad_forces_n'],
                    speed=self.metrics['object_speed_m_s'], slip=self.metrics['slip_m'])
            event = self.task.update(dt, **self.metrics, grasp_ready=grasp_ready,
                approach_contact_ready=contact, approach_blocked=blocked,
                approach_pregrasp_ready=handoff_ready)
            if phase == 'approach' and self.task.phase == 'close':
                self.descent_stop = 'pad_contact' if contact else None
            if self.task.phase != phase:
                if self.task.phase not in self.reached_phases:
                    self.reached_phases.append(self.task.phase)
                if self.task.phase == 'hold':
                    self.grip_relative = self.measurement['object_relative'].copy()
                    regions = self.measurement.get('material_relative')
                    self.grip_regions = None if regions is None else regions.copy()
                elif self.task.phase == 'lift' and self.grasp_mode == 'object_stability':
                    self.grip_relative = self.measurement['object_relative'].copy()
                    regions = self.measurement.get('material_relative')
                    self.grip_regions = None if regions is None else regions.copy()
                elif self.task.phase == 'close':
                    self.grip_relative = None
                    self.grip_regions = None
            stage_done = curriculum_complete(self.stage, self.task)
            value, pieces = progress_reward(phase=phase, previous_metrics=before,
                metrics=self.metrics, action=self.command, reference=self.reference,
                previous=previous_command, dt=dt, success_event=event,
                failed=bool(self.task.failure),
                stage_event=stage_done and self.stage in ('approach', 'close'))
            if self.direct_joint_control:
                from .tabletop_direct import contact_quality
                # Targets are the policy's decisions, not errors against a
                # scripted closure. Reward bilateral balance and dwell time.
                value -= pieces['contact'] + pieces['mpc_tracking']
                pieces['mpc_tracking'] = 0.
                pieces['contact'] = (2.*(contact_quality(self.metrics['pad_forces_n'])
                                        - contact_quality(before['pad_forces_n']))
                                     if phase in ('close', 'hold') else 0.)
                pieces['contact_dwell'] = (dt if phase in ('close', 'hold')
                    and np.all(self.metrics['pad_forces_n'] >= .1)
                    and np.all(before['pad_forces_n'] >= .1) else 0.)
                value += pieces['contact'] + pieces['contact_dwell']
                if self.grasp_mode == 'object_stability':
                    value -= pieces['contact'] + pieces['contact_dwell'] + pieces['lift']
                    # Relative motion during an unsuccessful pickup is not
                    # slip of an already supported object. Do not swamp the
                    # first-contact signal with post-grasp slip penalties.
                    value -= pieces['slip']
                    pieces['slip'] = (-min((max(0.,self.metrics['slip_m']-.02)/.02)**2,25.)*dt
                        if self.metrics['object_bottom_m'] > .305 else 0.)
                    value += pieces['slip']
                    # Optional contact guidance has no pad-count requirement.
                    from .tabletop_contact_reward import contact_potential
                    support = lambda m: float(contact_potential(m['pad_forces_n']))
                    pieces['contact'] = support(self.metrics)-support(before) if phase in ('close','hold') else 0.
                    pieces['contact_dwell'] = 0.
                    height = lambda m: float(np.clip((m['object_bottom_m']-.30)/.03,0.,1.))
                    pieces['lift'] = 5.*(height(self.metrics)-height(before)) if phase == 'lift' else 0.
                    pieces['lift_dwell'] = (dt if phase == 'lift'
                        and self.metrics['object_bottom_m'] >= .31
                        and self.metrics['object_speed_m_s'] < .03
                        and self.metrics['slip_m'] <= .02 else 0.)
                    value += pieces['contact'] + pieces['lift'] + pieces['lift_dwell']
            reward += value
            for key, value in pieces.items():
                terms[key] = terms.get(key, 0.) + value
            if self.task.failure or self.task.success or stage_done:
                break
        if self.task.failure in ('nonfinite_state', 'invalid_action'):
            # These guards can fire before reward calculation. Preserve a
            # finite terminal transition instead of aborting SB3 on NaNs.
            reward -= 25.
            terms['failure'] = -25.
        return self.finish_transition(residual,reward,terms,used_time,coupling_pause,nonfinite)

    def finish_transition(self,residual,reward,terms,used_time,coupling_pause,nonfinite):
        """Record a completed control tick; also used by the batched task."""
        self.return_ += reward
        self.previous = self.command.copy()
        info = self._info()
        self.closed = bool(self.task.failure or self.task.success or info['stage_success'])
        try:
            observation = self.observe() if not nonfinite else self.last_observation.copy()
        except ValueError:
            observation = self.last_observation.copy()
        self.last_observation = observation.copy()
        if not self.trace_steps:
            minimum = min(self.metrics['clearance_m'],
                          self.log[0]['clearance_m'] if self.log else float('inf'))
            self.log[:] = [dict(clearance_m=minimum)]
        else:
            row = dict(t=self.task.elapsed, phase=self.task.phase, reward=reward,
                residual_action=(np.asarray(residual).tolist() if self.task.failure != 'invalid_action' else None),
                filtered_residual_rad=self.filtered_residual.tolist(),
                terms=terms, physical_dt=used_time, motion_paused=coupling_pause,
                root_offset_m=self.offset, clearance_m=self.metrics['clearance_m'],
                object_bottom_m=self.metrics['object_bottom_m'],
                object_relative=self.measurement['object_relative'].tolist(),
                object_quat=self.measurement['object_quat'].tolist(),
                object_velocity=self.measurement['object_velocity'].tolist(),
                pad_forces_n=self.metrics['pad_forces_n'].tolist(), slip_m=self.metrics['slip_m'],
                command_rad=self.command.tolist(), reference_rad=self.reference.tolist(),
                approach_error_m=self.metrics['approach_error_m'], descent_stop=self.descent_stop,
                nonfinite_measurement=nonfinite, lowest_link=self.measurement['lowest_link'],
                hands=[] if nonfinite else self.measurement['diagnostics'])
            if self.control_mode == 'adaptive_grasp':
                row.update({name: None if nonfinite else np.asarray(self.measurement[name]).tolist()
                            for name in ('adaptive_palm_offset', 'grip_stiffness', 'pad_relative')})
            if self.force_controller is not None:
                row['force_feedback'] = self.force_controller.diagnostics()
                row['force_feedback']['measured_hand_load_n'] = np.asarray(self.metrics['pad_forces_n']).reshape(2,2).sum(axis=1).tolist()
            if 'pressure_geometry' in info:
                row['pressure_geometry'] = info['pressure_geometry']
            self.log.append(row)
        if self.closed:
            self.flush_episode()
        return observation, float(reward), self.closed, False, info
