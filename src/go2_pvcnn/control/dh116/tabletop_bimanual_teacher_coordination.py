"""Proof-gated tripod Teacher control candidate, excluded from BC labels.

Pinch rows never receive a target write from this module. Tripod rows reuse
the existing privileged Z3 and coupled tracking projection as independent
diagnostics. Their 480 Hz requests require a future expert-label contract.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from .tabletop_bimanual_coordination import BimanualCoordinator, coordination_state_from_native
from .tabletop_bimanual_tracking_guard import BimanualTrackingGuard, tracking_state_from_native
from .tabletop_evaluation_telemetry import EvaluationTelemetry
from .tabletop_free_wrist_grasp import CONTRACT as WRIST
from .tabletop_policy_control import CONTRACT as POLICY
from .tabletop_surface_contract import ACTION_CONTRACT


VERSION = 'dh116-tripod-teacher-coordination-candidate-v1'
_MISSING = object()
FIELDS = ('env_id', 'episode_id', 'policy_step', 'physics_step', 'timestamp_s',
          'tripod_candidate', 'live', 'newly_terminal', 'resume_ready',
          'cancel_upward_requested_left', 'cancel_upward_requested_right',
          'cancel_upward_applied_left', 'cancel_upward_applied_right',
          'guard_pause_left', 'guard_pause_right',
          'guard_cancel_translation_left', 'guard_cancel_translation_right',
          'guard_cancel_rotation_left', 'guard_cancel_rotation_right',
          'source_downward_admitted_left', 'source_downward_admitted_right',
          'J_lift', 'J_contact', 'J_slip', 'J_joint', 'J_sync', 'J_total')


def read_tripod_coordination_proof(path, *, expected_sha256=None):
    """Reuse the T5 auditor's exact gates and immutable-input SHA checks."""
    path = Path(path).absolute()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('T5 proof changed after pre-Kit validation')
    tool = Path(__file__).resolve().parents[4] / 'tools/audit_stage5_teacher_regression.py'
    spec = importlib.util.spec_from_file_location('_dh116_t5_proof_validator', tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('T5 proof must be a completed regression JSON object')
    verified = module.validate_teacher_regression_proof(value)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('T5 proof changed during validation')
    return dict(path=str(path), sha256=digest, validation=verified,
                teacher_seed=value['initial_state'].get('seed'),
                validator_source_sha256=hashlib.sha256(tool.read_bytes()).hexdigest())


def validate_tripod_coordination_options(args):
    path = getattr(args, 'bimanual_tripod_teacher_coordination_proof', None)
    if path is None:
        return None
    if (not getattr(args, 'vision_teacher_source', None)
            or getattr(args, 'bimanual_explicit_sdk_adapter', False)
            or getattr(args, 'vision_evaluation_checkpoint', None)
            or getattr(args, 'vision_m2_replay', None)):
        raise ValueError('tripod coordination requires independent Teacher mode and excludes explicit SDK adapter')
    proof = read_tripod_coordination_proof(path,
        expected_sha256=getattr(args, '_bimanual_tripod_proof_sha256', None))
    source = Path(args.vision_teacher_source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != proof['validation']['teacher_source_sha256']:
        raise ValueError('tripod candidate Teacher source differs from verified T5 source')
    if getattr(args, 'seed', proof['teacher_seed']) != proof['teacher_seed']:
        raise ValueError('tripod candidate seed differs from verified T5 seed')
    args._bimanual_tripod_proof_sha256 = proof['sha256']
    return proof


def supervision_eligibility(grasp_styles):
    """Only unchanged pinch requests retain their existing label semantics."""
    if any(style not in ('pinch_thumb_index', 'tripod_thumb_index_middle') for style in grasp_styles):
        raise ValueError('known actual grasp styles required')
    return [style == 'pinch_thumb_index' for style in grasp_styles]


class TripodTeacherCoordination:
    def __init__(self, env, world, output, grasp_styles, proof):
        if getattr(env, 'bimanual_action_adapter', None) is not None:
            raise ValueError('tripod candidate requires an independent action path')
        if len(grasp_styles) != world.n:
            raise ValueError('one actual reference grasp style per environment required')
        self.supervision_eligible = supervision_eligibility(grasp_styles)
        self.mask = torch.tensor([not eligible for eligible in self.supervision_eligible],
                                 device=world.device, dtype=torch.bool)
        if not bool(self.mask.any()):
            raise ValueError('tripod candidate requires at least one actual tripod environment')
        self.env, self.world, self.output, self.proof = env, world, Path(output), proof
        self.paths = [self.output / name for name in (
            'tripod_teacher_coordination.npz', 'tripod_teacher_coordination_contract.json')]
        if any(path.exists() or path.is_symlink() for path in self.paths):
            raise FileExistsError('independent tripod candidate evidence paths required')
        self.coordinator = BimanualCoordinator(world.n, mode='Z3')
        self.guard = BimanualTrackingGuard(world.n, enabled=True, coupled_pause=True)
        self.prior = {name: getattr(env, name, _MISSING) for name in (
            'evaluation_telemetry', 'cache_bimanual_slip', '_bimanual_slip_packet',
            'bimanual_action_adapter', 'joint_replay_observer')}
        prior_telemetry = self.prior['evaluation_telemetry']
        prior_callback = None if prior_telemetry in (_MISSING, None) else prior_telemetry.callback
        telemetry = EvaluationTelemetry(world.n, world.device,
            prior_callback if prior_callback is not None else lambda packet: None)
        # Read-only initialization from the already completed physical reset.
        telemetry.reset(torch.arange(world.n, device=world.device), env.task.m)
        self.initial_ids = env.diagnostic_episode_ids.detach().clone()
        self.closed = torch.zeros_like(self.mask)
        self.pending = None
        self.rows, self.sources, self.executed = [], [], []
        self.finished = False
        env.evaluation_telemetry = telemetry
        env.cache_bimanual_slip = True
        env._bimanual_slip_packet = None
        env.bimanual_action_adapter = self.project
        env.joint_replay_observer = self.observe

    @torch.no_grad()
    def project(self, source):
        w, t = self.world, self.env.task
        state = coordination_state_from_native(w, t, self.env)
        coordinated, info = self.coordinator.project(source, state, dt_s=w.dt)
        selected_live = (self.mask & state.live & ~self.closed
                         & (self.env.diagnostic_episode_ids == self.initial_ids))
        source_z = source[:, [14, 17]]
        source_downward = (selected_live & ~info['resume_ready'] & info['measurement_valid']
                           & state.contact_valid.all(-1))[:, None] & (source_z < 0)
        clearance_lower = ((POLICY['palm_descent_clearance_m'] - state.clearance_m)
                           / (ACTION_CONTRACT['translation_rate_m_s'] * w.dt)).clamp(-1, 0)
        bounded_source_z = torch.maximum(source_z, clearance_lower)
        coordinated[:, [14, 17]] = torch.where(source_downward, bounded_source_z,
                                               coordinated[:, [14, 17]])
        # The existing native XYZ vector-norm limit executes later. X/Y are
        # unchanged and |bounded_source_z| <= |source_z|, so acquisition never
        # increases the source's physical downward speed. Tracking wins last.
        track_state = tracking_state_from_native(w, t)
        guarded, guard = self.guard.project(coordinated, track_state)
        issued = torch.where(selected_live[:, None], guarded, source).clone()
        downward_admitted = source_downward & (issued[:, [14, 17]] < 0)
        if not torch.equal(issued[:, :12], source[:, :12]):
            raise RuntimeError('tripod candidate may not alter finger requests')
        ids = selected_live.nonzero().flatten()
        applied_upward = torch.zeros_like(info['cancel_upward_target'])
        if ids.numel():
            # All mutations are indexed exclusively by actual live tripod rows.
            palms = w.palms[ids].clone()
            palm_targets = t.palm_targets[ids].clone()
            angles, rotation_targets = w.wrist_angles[ids].clone(), t.rotation_targets[ids].clone()
            measured = track_state.measured_palm_offset_m
            low, high = source.new_tensor(WRIST['palm_lower_m']), source.new_tensor(WRIST['palm_upper_m'])
            hold_z_valid = (torch.isfinite(measured[:, :, 2])
                            & (measured[:, :, 2] >= low[2]) & (measured[:, :, 2] <= high[2]))
            cancel_z = info['cancel_upward_target'][ids] & hold_z_valid[ids]
            applied_upward[ids] = cancel_z
            palms[:, :, 2] = torch.where(cancel_z, torch.minimum(palms[:, :, 2], measured[ids, :, 2]), palms[:, :, 2])
            palm_targets[:, :, 2] = torch.where(cancel_z,
                torch.minimum(palm_targets[:, :, 2], measured[ids, :, 2]), palm_targets[:, :, 2])
            cancel_translation = guard['cancel_translation'][ids, :, None]
            palms = torch.where(cancel_translation, guard['hold_offset_m'][ids], palms)
            palm_targets = torch.where(cancel_translation, guard['hold_offset_m'][ids], palm_targets)
            cancel_rotation = guard['cancel_rotation'][ids, :, None]
            angles = torch.where(cancel_rotation, guard['hold_intrinsic_xyz_rad'][ids], angles)
            rotation_targets = torch.where(cancel_rotation, guard['hold_intrinsic_xyz_rad'][ids], rotation_targets)
            w.palms[ids], t.palm_targets[ids] = palms, palm_targets
            w.wrist_angles[ids], t.rotation_targets[ids] = angles, rotation_targets
        self.pending = (source.detach().clone(), issued.detach().clone(), info, guard,
                        applied_upward, selected_live, downward_admitted)
        return issued

    @torch.no_grad()
    def observe(self, env, measurement, native_live, newly_terminal):
        prior = self.prior['joint_replay_observer']
        if prior not in (_MISSING, None):
            prior(env, measurement, native_live, newly_terminal)
        if self.pending is None:
            raise RuntimeError('tripod candidate physical observation missing its issued request')
        source, issued, info, guard, applied_upward, selected_live, downward_admitted = self.pending
        ids = env.diagnostic_episode_ids
        live = native_live & ~self.closed & (ids == self.initial_ids)
        terminal = newly_terminal & live
        scalar = [torch.arange(self.world.n, device=self.world.device), ids,
            torch.full_like(ids, int(env._diagnostic_policy_step)),
            torch.full_like(ids, int(self.world._physics_steps)),
            torch.full_like(source[:, 0], float(self.world.w.sim.current_time)),
            self.mask, live, terminal, info['resume_ready'] & selected_live]
        for value in (info['cancel_upward_target'] & selected_live[:, None], applied_upward,
                      guard['pause_hand'] & selected_live[:, None],
                      guard['cancel_translation'] & selected_live[:, None],
                      guard['cancel_rotation'] & selected_live[:, None], downward_admitted):
            scalar.extend(value.unbind(-1))
        scalar.extend(torch.where(selected_live, info['objective'][name], float('nan'))
                      for name in ('lift', 'contact', 'slip', 'joint', 'sync', 'total'))
        self.rows.append(torch.stack([value.double() for value in scalar], -1).cpu().numpy())
        self.sources.append(source.cpu().numpy())
        self.executed.append(issued.cpu().numpy())
        self.closed |= terminal | (ids != self.initial_ids)
        self.pending = None

    def close(self, *, error=None):
        if self.finished:
            return
        self.finished = True
        for name, value in self.prior.items():
            if value is _MISSING:
                if hasattr(self.env, name): delattr(self.env, name)
            else:
                setattr(self.env, name, value)
        n = self.world.n
        with self.paths[0].open('xb') as stream:
            np.savez_compressed(stream,
                samples=np.stack(self.rows) if self.rows else np.empty((0, n, len(FIELDS))),
                fields=np.array(FIELDS),
                source_requests=np.stack(self.sources) if self.sources else np.empty((0, n, 24)),
                executed_requests=np.stack(self.executed) if self.executed else np.empty((0, n, 24)),
                tripod_mask=self.mask.cpu().numpy())
        with self.paths[1].open('x', encoding='utf-8') as stream:
            json.dump(dict(version=VERSION, qualification='physical_diagnostic_candidate_only', error=error,
                proof=self.proof, source='Teacher_or_MPC_source_request',
                execution='480Hz_Z3_plus_coupled_tracking_projection;then_existing_native_guards',
                objective_phase='pre_control_Z3_candidate_before_optional_source_descent', record_phase='after_consume_before_reset',
                unready_acquisition='retain_nonpositive_source_Z_with_unchanged_clearance_floor_and_XYZ_speed_norm;coupled_tracking_guard_last',
                source_downward_admission_phase='post_coupled_tracking_guard_request;before_original_native_constraints',
                acquisition_clearance_floor_m=POLICY['palm_descent_clearance_m'],
                acquisition_translation_rate_m_s=ACTION_CONTRACT['translation_rate_m_s'],
                objective_applicability='tripod_candidate_rows_only;pinch_NA',
                tripod_supervision_qualified=False, counts_as_student_success=False,
                supervision_eligible_env_ids=[i for i, valid in enumerate(self.supervision_eligible) if valid],
                candidate_env_ids=self.mask.nonzero().flatten().tolist(),
                coupled_tracking_guard=self.guard.contract()), stream, indent=2, allow_nan=False)


def install_tripod_teacher_coordination(env, world, output, grasp_styles, proof_path, *, expected_sha256=None):
    proof = read_tripod_coordination_proof(proof_path, expected_sha256=expected_sha256)
    return TripodTeacherCoordination(env, world, output, grasp_styles, proof)
