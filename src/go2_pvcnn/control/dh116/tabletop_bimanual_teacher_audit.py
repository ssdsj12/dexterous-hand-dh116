"""Opt-in read-only 480 Hz Teacher/MPC first-episode evidence.

The source action is a Teacher/MPC issued physical request, never Actor output
or a latent. Samples are captured after native consume/guards and before reset.
No-op samples remain separately labelled; first terminal rows are frozen.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .linear_coupling import ACTIVE
from .tabletop_surface_force_sharing import surface_targets
from .tabletop_evaluation_telemetry import EvaluationTelemetry, SLIP_REASON_CODES
from .tabletop_joint_target_readback import read_joint_target_evidence, BACKEND_SOURCE, CACHE_SOURCE

_MISSING = object()
VERSION = 'dh116-teacher-480hz-first-episode-audit-v2'


FIELDS = (
    'env_id', 'episode_id', 'policy_step', 'physics_step', 'timestamp_s', 'teacher_control_step',
    'native_live', 'live', 'newly_terminal', 'native_success', 'failure_code',
    'object_bottom_m', 'native_reference_bottom_m', 'object_net_lift_m',
    'measured_reset_bottom_m', 'object_reset_relative_lift_m', 'native_hold_s',
    'native_anchor_drift_m', 'native_anchor_drift_valid',
    'fresh_contact_anchor_slip_m', 'fresh_contact_anchor_slip_valid',
    'fresh_contact_anchor_slip_applicable', 'fresh_contact_anchor_slip_reason',
    'left_palm_x_w_m', 'left_palm_y_w_m', 'left_palm_z_w_m',
    'right_palm_x_w_m', 'right_palm_y_w_m', 'right_palm_z_w_m',
    'left_palm_dx_m', 'left_palm_dy_m', 'left_palm_dz_m',
    'right_palm_dx_m', 'right_palm_dy_m', 'right_palm_dz_m',
    'left_load_n', 'right_load_n', 'left_contact_valid', 'right_contact_valid',
    'left_contact_measurement_valid', 'right_contact_measurement_valid',
    'left_sdk_target_native_readback', 'right_sdk_target_native_readback',
    'left_sdk_target_finite', 'right_sdk_target_finite',
    'left_sdk_command_cache_readback', 'right_sdk_command_cache_readback',
    'left_sdk_command_cache_finite', 'right_sdk_command_cache_finite',
)
FIELDS += tuple(f'teacher_issued_action_{i}' for i in range(24))
FIELDS += tuple(f'sdk_target_{side}_{name}_rad' for side in ('left', 'right') for name in ACTIVE)
FIELDS += tuple(f'command_cache_target_{side}_{name}_rad' for side in ('left', 'right') for name in ACTIVE)
FIELDS += tuple(f'reconstructed_target_{side}_{name}_rad' for side in ('left', 'right') for name in ACTIVE)
FIELDS += tuple(f'actual_q_{side}_{name}_rad' for side in ('left', 'right') for name in ACTIVE)
FIELDS += tuple(f'actual_qd_{side}_{name}_rad_s' for side in ('left', 'right') for name in ACTIVE)


class TeacherPhysicsAudit:
    def __init__(self, env, world, output, *, native_runtime_verified=False):
        self.env, self.world, self.output = env, world, Path(output)
        self.n = world.n
        self.native_runtime_verified = native_runtime_verified is True
        self.paths = [self.output / name for name in (
            'teacher_physics_audit.npz', 'teacher_physics_audit_fields.json',
            'teacher_physics_audit_summary.json')]
        if any(path.exists() or path.is_symlink() for path in self.paths):
            raise FileExistsError('Teacher physical audit output must be independent')
        for hi, hand in enumerate(world.hands):
            if [hand.joint_names[index] for index in world.w.ids[hi]] != list(ACTIVE):
                raise ValueError('Teacher physical audit requires canonical named SDK active order')
        self.initial_ids = env.diagnostic_episode_ids.detach().clone()
        self.baseline = world.roots[:, :, :3].detach().clone()
        self.reset_bottom = env.task.m['bottom'].detach().clone()
        self.reference = torch.as_tensor(env.task.surface_reference_bottom_m, device=world.device,
                                          dtype=torch.float64).expand(self.n).clone()
        self.closed = torch.zeros(self.n, dtype=torch.bool, device=world.device)
        self.samples = []
        self.joint_names = [tuple(hand.joint_names) for hand in world.hands]
        self.target_samples, self.terminal_target_samples = {}, {}
        for side, names in zip(('left', 'right'), self.joint_names):
            for prefix in ('backend_full_target', 'command_cache_full_target', 'dof_types'):
                name = f'{prefix}_{side}'
                self.target_samples[name] = []
                self.terminal_target_samples[name] = np.full((self.n, len(names)), np.nan)
        self.target_readback_statuses = []
        self._target_status_keys = set()
        self.terminal_samples = np.full((self.n, len(FIELDS)), np.nan)
        self.terminal_valid = np.zeros(self.n, dtype=bool)
        self.issued = None
        self.teacher_control_step = None
        self.initial_policy_step = int(env._diagnostic_policy_step)
        self.finished = False
        self.prior = getattr(env, 'joint_replay_observer', None)
        self.observer = self.observe
        self.fresh_packet = None
        self.prior_telemetry = getattr(env, 'evaluation_telemetry', _MISSING)
        if self.prior_telemetry is _MISSING or self.prior_telemetry is None:
            self.prior_callback = None
            self.telemetry = EvaluationTelemetry(self.n, world.device, self.capture_packet)
            # Read the existing reset measurement. Never reset or step physics.
            self.telemetry.reset(torch.arange(self.n, device=world.device), env.task.m)
            env.evaluation_telemetry = self.telemetry
        else:
            self.telemetry = self.prior_telemetry
            self.prior_callback = self.telemetry.callback
            self.telemetry.callback = self.capture_packet
        env.joint_replay_observer = self.observer

    @torch.no_grad()
    def capture_packet(self, packet):
        if self.prior_callback is not None:
            self.prior_callback(packet)
        names = ('slip_m', 'slip_valid', 'slip_applicable', 'slip_reason_code')
        values = {}
        for name in names:
            value = packet[name]
            if not isinstance(value, torch.Tensor) or value.shape != (self.n,):
                raise ValueError('fresh native contact slip requires one value per environment: ' + name)
            values[name] = value.detach().clone()
        self.fresh_packet = dict(episode_ids=self.env.diagnostic_episode_ids.detach().clone(),
            physics_step=int(self.world._physics_steps), timestamp_s=float(self.world.w.sim.current_time),
            **values)

    def submit(self, issued, *, teacher_control_step):
        if self.finished or issued.shape != (self.n, 24) or not bool(torch.isfinite(issued).all()):
            raise ValueError('current finite Teacher/MPC issued physical [N,24] action required')
        if type(teacher_control_step) is not int or teacher_control_step < 0:
            raise ValueError('actual nonnegative Teacher collection tick required')
        self.issued = issued.detach().clone().double()
        self.teacher_control_step = teacher_control_step

    def _target_evidence(self, hi, hand):
        """Keep the complete named backend/cache arrays separate, including failures."""
        evidence = read_joint_target_evidence(hand)
        if tuple(evidence['joint_names']) != self.joint_names[hi]:
            raise RuntimeError('Teacher target readback joint identity changed')
        side = ('left', 'right')[hi]
        arrays, available, finite, status = {}, {}, {}, {}
        for source, prefix in (('backend', 'backend_full_target'),
                               ('command_cache', 'command_cache_full_target'),
                               ('dof_types', 'dof_types')):
            record = evidence[source]
            values = record['values']
            present = bool(record['available'] and values.shape == (self.n, len(self.joint_names[hi])))
            error = record['error']
            if record['available'] and not present:
                error = 'Teacher target readback requires the complete current environment cohort'
            value = np.array(values, copy=True) if present else np.full((self.n, len(self.joint_names[hi])), np.nan)
            arrays[f'{prefix}_{side}'] = value
            available[source] = torch.full((self.n,), present, dtype=torch.bool, device=self.world.device)
            finite[source] = torch.as_tensor(np.isfinite(value).all(1) & present, device=self.world.device)
            status[source] = dict(available=present, source=record['source'], dtype=record['dtype'],
                                  shape=record['shape'], error=error)
        status.update(side=side, joint_names=evidence['joint_names'],
            units_by_name=evidence['units_by_name'], unit_source=evidence['unit_source'],
            expected_units_by_name=evidence['expected_units_by_name'],
            expected_unit_source=evidence['expected_unit_source'],
            wrist_joint_indices=evidence['wrist_joint_indices'], wrist_target_frame=evidence['wrist_target_frame'])
        key = json.dumps(status, sort_keys=True)
        if key not in self._target_status_keys:
            self._target_status_keys.add(key)
            self.target_readback_statuses.append(dict(first_physics_step=int(self.world._physics_steps), **status))
        return arrays, available, finite

    @torch.no_grad()
    def observe(self, env, measurement, native_live, newly_terminal):
        if self.prior is not None:
            self.prior(env, measurement, native_live, newly_terminal)
        if self.issued is None or self.finished:
            raise RuntimeError('Teacher physics audit requires submit before a physical substep')
        world = self.world
        ids = env.diagnostic_episode_ids
        packet = self.fresh_packet
        if (packet is None or packet['physics_step'] != int(world._physics_steps)
                or packet['timestamp_s'] != float(world.w.sim.current_time)
                or not torch.equal(packet['episode_ids'], ids)):
            raise RuntimeError('fresh contact slip packet episode/physics/time mismatch')
        live = native_live & ~self.closed & (ids == self.initial_ids)
        terminal = newly_terminal & live
        loads = measurement['effective_pad_loads_n'].double()
        vectors = measurement['effective_pad_vectors_w'].double()
        valid_pad = measurement['effective_pad_valid']
        contact_measurement = (torch.isfinite(loads).all(-1) & (loads >= 0).all(-1)
                               & torch.isfinite(vectors).all((-1, -2)))
        bad = getattr(world, 'pad_measurements', {}).get('effective_pad_bad_env')
        if bad is not None:
            contact_measurement &= ~bad[:, None]
        clean_loads = torch.where(contact_measurement[:, :, None], loads, 0.)
        clean_vectors = torch.where(contact_measurement[:, :, None, None], vectors, 0.)
        contact = surface_targets(clean_loads, clean_vectors, valid=valid_pad)['paired'] & contact_measurement
        actual_q, actual_qd, targets, target_valid = [], [], [], []
        target_finite, cache_targets, cache_valid, cache_finite, reconstructed = [], [], [], [], []
        full_targets = {}
        for hi, hand in enumerate(world.hands):
            joint_ids = world.w.ids[hi]
            actual_q.append(world.q[:, hi, joint_ids].double())
            actual_qd.append(world.qd[:, hi, joint_ids].double())
            arrays, available, finite = self._target_evidence(hi, hand)
            full_targets.update(arrays)
            side = ('left', 'right')[hi]
            targets.append(torch.as_tensor(arrays[f'backend_full_target_{side}'][:, joint_ids],
                                           dtype=torch.float64, device=world.device))
            cache_targets.append(torch.as_tensor(arrays[f'command_cache_full_target_{side}'][:, joint_ids],
                                                 dtype=torch.float64, device=world.device))
            # Availability describes an actual complete getter result. NaNs
            # remain raw evidence, with numeric validity recorded separately.
            target_valid.append(available['backend'])
            target_finite.append(finite['backend'])
            cache_valid.append(available['command_cache'])
            cache_finite.append(finite['command_cache'])
            reconstructed.append(world.commands[:, hi].float().double())
        bottom = measurement['bottom'].double()
        hold = getattr(getattr(env.task, 'surface_lift_success', None), 'duration',
                       torch.full_like(bottom, float('nan')))
        tracker = getattr(env.task, 'surface_lift_success', None)
        drift = getattr(tracker, 'drift_m', torch.full_like(bottom, float('nan')))
        drift_valid = (getattr(tracker, '_latched', torch.zeros_like(live))
                       & getattr(tracker, '_physical_window_valid', torch.zeros_like(live))
                       & torch.isfinite(drift))
        drift = torch.where(drift_valid, drift, float('nan'))
        palm = world.roots[:, :, :3].double()
        scalar = [torch.arange(self.n, device=world.device), ids,
            torch.full_like(ids, int(env._diagnostic_policy_step)),
            torch.full_like(ids, int(world._physics_steps)),
            torch.full_like(bottom, float(world.w.sim.current_time)),
            torch.full_like(ids, self.teacher_control_step), native_live, live, terminal,
            env.task.success, env.task.failure, bottom, self.reference, bottom-self.reference,
            self.reset_bottom, bottom-self.reset_bottom, hold, drift, drift_valid,
            packet['slip_m'], packet['slip_valid'], packet['slip_applicable'], packet['slip_reason_code']]
        columns = [value.double() for value in scalar]
        columns.extend(palm.flatten(1).unbind(-1))
        columns.extend((palm-self.baseline).flatten(1).unbind(-1))
        columns.extend(measurement['hand_object_loads'].double().unbind(-1))
        columns.extend(contact.unbind(-1))
        columns.extend(contact_measurement.unbind(-1))
        columns.extend(target_valid)
        columns.extend(target_finite)
        columns.extend(cache_valid)
        columns.extend(cache_finite)
        columns.extend(self.issued.unbind(-1))
        columns.extend(torch.cat(targets, -1).unbind(-1))
        columns.extend(torch.cat(cache_targets, -1).unbind(-1))
        columns.extend(torch.cat(reconstructed, -1).unbind(-1))
        columns.extend(torch.cat(actual_q, -1).unbind(-1))
        columns.extend(torch.cat(actual_qd, -1).unbind(-1))
        sample = torch.stack([value.double() for value in columns], -1).cpu().numpy()
        if sample.shape != (self.n, len(FIELDS)):
            raise RuntimeError('Teacher physical audit schema mismatch')
        chosen = terminal.cpu().numpy()
        self.terminal_samples[chosen] = sample[chosen]
        self.terminal_valid[chosen] = True
        for name, value in full_targets.items():
            self.target_samples[name].append(value)
            self.terminal_target_samples[name][chosen] = value[chosen]
        self.samples.append(sample)
        self.closed |= terminal | (ids != self.initial_ids)

    def finish(self, *, error=None):
        if self.finished:
            return
        self.finished = True
        self.env.joint_replay_observer = self.prior
        if self.prior_telemetry is _MISSING:
            if hasattr(self.env, 'evaluation_telemetry'): delattr(self.env, 'evaluation_telemetry')
        elif self.prior_telemetry is None:
            self.env.evaluation_telemetry = None
        else:
            self.telemetry.callback = self.prior_callback
            self.env.evaluation_telemetry = self.prior_telemetry
        samples = np.stack(self.samples) if self.samples else np.empty((0, self.n, len(FIELDS)))
        summary = dict(version=VERSION,
            status='error' if error else 'recorded', error=error, physics_batches=len(samples),
            native_runtime_verified=self.native_runtime_verified,
            environments=self.n, first_terminal_count=int(self.terminal_valid.sum()),
            source='Teacher_or_MPC_issued_physical24;not_Actor_or_latent',
            target_source='sdk_target_columns_are_backend_only;cache_and_reconstruction_are_separate',
            target_sources=dict(backend=BACKEND_SOURCE, command_cache=CACHE_SOURCE,
                reconstructed='float32_submission_reconstructed_from_integrated_world.commands;active6_only'),
            target_availability='complete_named_backend_or_cache_getter;numeric_finiteness_separate;missing_values_NaN;no_cache_fallback',
            full_target_frame='actual_named_DOF_targets;carriage_translation_is_anchor_local_not_measured_world_palm',
            target_readback_statuses=self.target_readback_statuses,
            legacy_v1_target_scope='IsaacLab_command_cache_or_reconstruction;not_PhysX_backend_readback',
            net_lift_reference='unchanged_native_task_surface_reference_bottom_m',
            native_anchor_drift_source='SurfaceLiftSuccess.drift_m_when_latched_and_physical_window_valid;last_completed_60Hz_window',
            fresh_contact_anchor_slip_source='fresh_EvaluationTelemetry_packet;pre_consume_contact_anchor_surface_slip_tracking_preferred',
            fresh_contact_anchor_slip_scope='contact_anchor_diagnostic;distinct_from_native_5s_Hold_window',
            fresh_contact_anchor_slip_reason_codes=SLIP_REASON_CODES,
            palm_reference='actual_DH116_base_link', sample_timing='after_consume_before_physical_reset',
            noop_policy='stored_live_false;first_terminal_samples_frozen',
            initial_policy_step=self.initial_policy_step,
            episodes=[dict(env_id=i, episode_id=int(self.initial_ids[i]), supervision_episode_id=i,
                           episode_uid=f'{self.output.name}:{i}:{int(self.initial_ids[i])}',
                           terminal_recorded=bool(self.terminal_valid[i])) for i in range(self.n)],
            counts_as_student_success=False)
        full = {name: np.stack(values) if values else np.empty((0, *self.terminal_target_samples[name].shape))
                for name, values in self.target_samples.items()}
        full.update({f'terminal_{name}': value for name, value in self.terminal_target_samples.items()})
        full.update({f'joint_names_{side}': np.array(names)
                     for side, names in zip(('left', 'right'), self.joint_names)})
        with self.paths[0].open('xb') as stream:
            np.savez_compressed(stream, samples=samples, fields=np.array(FIELDS),
                                terminal_samples=self.terminal_samples, terminal_valid=self.terminal_valid, **full)
        with self.paths[1].open('x', encoding='utf-8') as stream:
            json.dump(dict(version=VERSION, fields=list(FIELDS), target_sources=summary['target_sources'],
                           full_target_frame=summary['full_target_frame']), stream, indent=2)
        with self.paths[2].open('x', encoding='utf-8') as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
        return summary


def install_teacher_physics_audit(env, world, output, *, enabled=False, native_runtime_verified=False):
    """Disabled path does not inspect native state or install any callback."""
    return TeacherPhysicsAudit(env, world, output, native_runtime_verified=native_runtime_verified) if enabled else None
