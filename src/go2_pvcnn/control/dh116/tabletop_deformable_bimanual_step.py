"""One 480 Hz bimanual action boundary; no simulation or solver ownership.

Reuse the existing canonical 24D kernels, MountView and PhysX target reader.
Callers retain SDK/solver stepping, measured-state refresh and original table
projection. The requested palm offsets are NOT safe world poses: submit needs
the caller's separately projected world root targets and their source label.
This bridge grants neither collision qualification nor Teacher admission.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import torch

from .linear_coupling import ACTIVE, JOINTS, MIMIC, active_limits
from .tabletop_command_kernels import finger_pre_guard, surface_final_guard, wrist_requested_targets
from .tabletop_joint_target_readback import read_joint_target_evidence, WRIST_LINEAR, WRIST_ANGULAR
from .tabletop_palm_mount import MountView, local_translation, CONTRACT as MOUNT_CONTRACT
from .tabletop_surface_contract import ACTION_CONTRACT, validate_actions

VERSION = 'dh116-deformable-bimanual-step-v1'
DT = 1 / 480
SIDES = ('left', 'right')
CLOCK_KEYS = ('run_uuid', 'env_id', 'episode_uid', 'reset_id', 'rod_uid',
              'physics_step', 'rod_tick', 'policy_step', 'substep',
              'timestamp_s', 'solver_time_s', 'clock_valid')
GROUPS = dict(left_fingers=(0, 6), right_fingers=(6, 12),
              left_world_translation=(12, 15), right_world_translation=(15, 18),
              left_intrinsic_rotation=(18, 21), right_intrinsic_rotation=(21, 24))


@dataclass(frozen=True)
class HandBinding:
    """Caller-owned handles; env_paths maps external env IDs to exact SDK paths."""
    side: str
    hand: object
    env_paths: dict
    mount: object = None


def _owned(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def _tensor(value, shape):
    value = torch.as_tensor(_owned(value), dtype=torch.float64)
    if tuple(value.shape) != tuple(shape) or not bool(torch.isfinite(value).all()):
        raise ValueError(f'finite owned array {shape} required')
    return value


def _clock(read_clock, env_ids):
    rows = deepcopy(read_clock())
    if not isinstance(rows, (list, tuple)) or len(rows) != len(env_ids):
        raise ValueError('one explicit shared-clock row per environment required')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or any(k not in row for k in CLOCK_KEYS):
            raise ValueError('complete SDK/rod/reset clock identity required')
        row = {k: row[k] for k in CLOCK_KEYS}
        if row['clock_valid'] is not True:
            raise ValueError('runtime must report an explicitly valid clock')
        for key in ('run_uuid', 'episode_uid', 'rod_uid'):
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError('nonempty clock identity required')
        for key in ('env_id', 'reset_id', 'physics_step', 'rod_tick', 'policy_step', 'substep'):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError('nonnegative integer clock identity required')
        for key in ('timestamp_s', 'solver_time_s'):
            if type(row[key]) not in (float, int) or not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError('finite nonnegative actual clock required')
        if not math.isclose(row['timestamp_s'], row['physics_step']*DT, rel_tol=0, abs_tol=1e-9):
            raise ValueError('480Hz SDK timestamp differs from physics_step')
        if row['env_id'] in result:
            raise ValueError('duplicate environment clock')
        result[row['env_id']] = row
    if set(result) != set(env_ids):
        raise ValueError('clock environment mapping differs')
    if len({(row['run_uuid'], row['physics_step'], row['timestamp_s']) for row in result.values()}) != 1:
        raise ValueError('one shared SDK clock/run across all environment rows required')
    return [result[env] for env in env_ids]


def _bindings(bindings, env_ids):
    if len(bindings) != 2 or tuple(b.side for b in bindings) != SIDES:
        raise ValueError('explicit left/right bindings in canonical order required')
    if (bindings[0].hand is bindings[1].hand
            or bindings[0].hand.root_physx_view is bindings[1].hand.root_physx_view):
        raise ValueError('two distinct actual articulation instances/views required')
    for binding in bindings:
        if set(binding.env_paths) != set(env_ids):
            raise ValueError('complete external env mapping required')
        paths = list(binding.hand.root_physx_view.prim_paths)
        wanted = list(binding.env_paths.values())
        if (len(paths) != len(env_ids) or len(set(paths)) != len(paths)
                or set(paths) != set(wanted) or len(set(wanted)) != len(wanted)
                or any(not isinstance(p, str) or not p.startswith('/')
                       or f'{binding.side}Hand' not in p.split('/') for p in paths)):
            raise ValueError('actual articulation prim_paths differ from explicit side/env mapping')
        if binding.mount is not None and (not isinstance(binding.mount, MountView)
                or binding.mount.hand is not binding.hand):
            raise ValueError('original mount (MountView) must own this exact hand')


def _capture(bindings, env_ids, read_clock):
    _bindings(bindings, env_ids)
    before = _clock(read_clock, env_ids)
    result = {}
    for binding in bindings:
        hand = binding.hand
        paths = list(hand.root_physx_view.prim_paths)
        rows = [paths.index(binding.env_paths[env]) for env in env_ids]
        evidence = read_joint_target_evidence(hand)
        names = evidence['joint_names']
        allowed = (set(JOINTS), set(JOINTS + WRIST_LINEAR + WRIST_ANGULAR))
        if not evidence['joint_names_valid'] or set(names) not in allowed:
            raise ValueError('actual complete named 11DOF hand or 17DOF carriage required')
        for key in ('backend', 'dof_types'):
            item = evidence[key]
            if not item['available'] or not bool(np.all(item['finite_env'])):
                raise ValueError('actual backend target and DOF types required; no cache fallback')
        try:
            q = _owned(hand.root_physx_view.get_dof_positions())
            qd = _owned(hand.root_physx_view.get_dof_velocities())
        except Exception as error:
            raise ValueError('actual SDK joint position/velocity readback required') from error
        shape = (len(env_ids), len(names))
        if any(a.shape != shape or a.dtype.kind != 'f' or not np.isfinite(a).all()
               for a in (q, qd, evidence['backend']['values'])):
            raise ValueError('matching finite named measured q/qd and backend target buffers required')
        types = evidence['dof_types']['values']
        if types.shape != shape or any(not np.all(types[:, j] == (1 if name in WRIST_LINEAR else 0))
                                       for j, name in enumerate(names)):
            raise ValueError('actual native DOF units differ from original joint types')
        active = [names.index(name) for name in ACTIVE]
        mounted = binding.mount is not None
        if mounted and (len(names) != 17
                or binding.mount.ids != [names.index(n) for n in WRIST_LINEAR]
                or binding.mount.angular_ids != [names.index(n) for n in WRIST_ANGULAR]):
            raise ValueError('complete original six-axis mount indices required')
        root = _owned(hand.data.root_pose_w)[rows] if mounted else None
        if mounted and (root.shape != (len(rows), 7) or not np.isfinite(root).all()
                        or not np.allclose(np.linalg.norm(root[:, 3:], axis=1), 1., atol=1e-5, rtol=0)):
            raise ValueError('measured mount anchor pose required')
        native_root = None
        if mounted:
            try:
                native_root = _owned(hand.root_physx_view.get_root_transforms())
            except Exception as error:
                raise ValueError('actual mount anchor transform readback required') from error
            if native_root.shape != (len(rows), 7) or not np.isfinite(native_root).all():
                raise ValueError('complete finite native mount anchor transforms required')
            # Same SDK xyzw -> IsaacLab wxyz permutation as ArticulationData.root_link_pose_w.
            native_root = native_root[rows][:, [0, 1, 2, 6, 3, 4, 5]]
            if not np.allclose(native_root, root, rtol=0, atol=1e-6):
                raise ValueError('mount anchor cache differs from actual SDK root transform')
        result[binding.side] = dict(
            joint_names=list(names), prim_paths=[paths[r] for r in rows], row_indices=rows,
            instance_object_id=id(hand), view_object_id=id(hand.root_physx_view),
            q=q[rows].copy(), qd=qd[rows].copy(), dof_types=types[rows].copy(),
            measured_source='physx_backend_get_dof_positions/get_dof_velocities',
            backend_targets=evidence['backend']['values'][rows].copy(),
            backend_source=evidence['backend']['source'], active_indices=active,
            command_cache_targets=(evidence['command_cache']['values'][rows].copy()
                if evidence['command_cache']['available'] else None),
            command_cache_source=evidence['command_cache']['source'],
            units_by_name=deepcopy(evidence['units_by_name']), mount_anchor_pose_w=root,
            native_anchor_pose_w=native_root, anchor_source='physx_backend_get_root_transforms_xyzw_to_wxyz',
            mounted=mounted)
        if names != list(hand.joint_names) or paths != list(hand.root_physx_view.prim_paths):
            raise ValueError('native names/instance paths changed during capture')
    if _clock(read_clock, env_ids) != before:
        raise ValueError('shared clock changed during bimanual measured-state capture')
    return dict(identity=before, hands=result)


class BimanualStepConnection:
    """A single-use transaction containing only owned arrays/scalars, no handles.

    begin -> submit -> caller SDK/rod step and SDK refresh -> finish.
    No implicit step/reset, native qualification, action-to-force shortcut or
    clock counter lives here. A write failure poisons this transaction because
    an SDK write cannot be rolled back safely. Construct a new transaction per
    physical substep; the existing control-step owner enforces eight substeps.
    """
    def __init__(self, *, env_ids, solver_substeps):
        self.env_ids = tuple(env_ids)
        if (not self.env_ids or len(set(self.env_ids)) != len(self.env_ids)
                or any(type(i) is not int or i < 0 for i in self.env_ids)
                or type(solver_substeps) is not int or solver_substeps < 1):
            raise ValueError('distinct environment IDs and explicit positive solver substeps required')
        self.solver_substeps = solver_substeps
        self._state = 'new'
        self._pre = self._request = self._submission = None

    def begin(self, bindings, action24, *, read_clock, palms, angles):
        if self._state != 'new':
            raise ValueError('new transaction required')
        validate_actions(action24)
        n = len(self.env_ids)
        if action24.shape[0] != n:
            raise ValueError('action batch must follow explicit external env order')
        action = action24.detach().cpu().double().clone()
        pre = _capture(bindings, self.env_ids, read_clock)
        for hi, side in enumerate(SIDES):
            if not pre['hands'][side]['mounted'] and bool(
                    torch.cat((action[:, 12+hi*3:15+hi*3], action[:, 18+hi*3:21+hi*3]), 1).any()):
                raise ValueError('nonzero wrist action requires complete original six-axis mount')
        command = torch.cat([torch.from_numpy(pre['hands'][s]['backend_targets'][
            :, pre['hands'][s]['active_indices']]).double() for s in SIDES], 1)
        q = torch.cat([torch.from_numpy(pre['hands'][s]['q'][
            :, pre['hands'][s]['active_indices']]).double() for s in SIDES], 1)
        residuals = []
        for side in SIDES:
            row = pre['hands'][side]
            names, measured = row['joint_names'], row['q']
            residuals.extend([measured[:, names.index(f)] - ratio*measured[:, names.index(m)] - offset
                              for f, m, ratio, offset in MIMIC])
        mimic = torch.from_numpy(np.stack(residuals, axis=1))
        low, high = (torch.from_numpy(np.tile(a, 2)) for a in active_limits())
        rates = dict(joint_rate_rad_s=ACTION_CONTRACT['joint_rate_rad_s'],
                     wrist_rate_m_s=ACTION_CONTRACT['translation_rate_m_s'],
                     rotation_rate_rad_s=ACTION_CONTRACT['rotation_rate_rad_s'])
        target = finger_pre_guard(command, action, q, mimic, low, high, DT, rates,
                                  policy_owned=True)
        _, target = surface_final_guard(command, target, low, high, DT)
        palm, angle = wrist_requested_targets(_tensor(palms, (n, 2, 3)),
                        _tensor(angles, (n, 2, 3)), action, rates, DT)
        request = dict(action24=action, action_contract=deepcopy(ACTION_CONTRACT),
            action_groups={k: action[:, lo:hi].numpy().copy() for k, (lo, hi) in GROUPS.items()},
            requested_finger_targets_rad=target.clone(), requested_palm_offsets_m=palm.clone(),
            requested_angles_rad=angle.clone(),
            command_baseline='actual PhysX backend target, copied before this action',
            controller_route='original policy_owned surface kernels; no force redistribution',
            wrist_projection_status='required_from_existing_caller; requested_offsets_are_not_world_poses')
        self._pre, self._request, self._state = pre, request, 'prepared'
        return deepcopy(request)

    def submit(self, bindings, *, read_clock, projected_root_poses=None,
               projected_angles=None, projection_source=None):
        if self._state != 'prepared':
            raise ValueError('prepared transaction required')
        current = _capture(bindings, self.env_ids, read_clock)
        if current['identity'] != self._pre['identity']:
            raise ValueError('pre/action clock changed before submission')
        for side in SIDES:
            old, now = self._pre['hands'][side], current['hands'][side]
            for field in ('joint_names', 'prim_paths', 'row_indices', 'mounted', 'instance_object_id', 'view_object_id'):
                if old[field] != now[field]:
                    raise ValueError('pre/action instance mapping changed')
            for field in ('q', 'qd', 'backend_targets'):
                if not np.array_equal(old[field], now[field]):
                    raise ValueError('measured state or backend targets changed before submission')
            if old['mounted'] and not np.array_equal(old['mount_anchor_pose_w'], now['mount_anchor_pose_w']):
                raise ValueError('mount anchor changed before submission')
        n = len(self.env_ids)
        mounted = any(row['mounted'] for row in current['hands'].values())
        roots = angles = None
        if mounted:
            if not isinstance(projection_source, str) or not projection_source.strip():
                raise ValueError('explicit caller projection source required; no table safety inferred')
            roots = _tensor(projected_root_poses, (n, 2, 7))
            angles = _tensor(projected_angles, (n, 2, 3))
            if (not torch.allclose(torch.linalg.vector_norm(roots[..., 3:], dim=-1),
                                   torch.ones((n, 2), dtype=torch.float64), atol=1e-5, rtol=0)
                    or bool((angles.abs() > math.pi/2).any())):
                raise ValueError('normalized poses and original wrist angle limits required')
        elif any(v is not None for v in (projected_root_poses, projected_angles, projection_source)):
            raise ValueError('unmounted hands have no wrist target submission')
        expected = {}
        prepared = []
        for hi, binding in enumerate(bindings):
            row = current['hands'][binding.side]
            # Invert external env order to the actual SDK buffer row order.
            permutation = np.argsort(row['row_indices']).tolist()
            target = self._request['requested_finger_targets_rad'][:, hi*6:hi*6+6].clone()
            indices = list(row['active_indices'])
            values = target.clone()
            if row['mounted']:
                local = local_translation(roots[:, hi], torch.from_numpy(row['mount_anchor_pose_w']).double())
                if bool((local.abs() > MOUNT_CONTRACT['limit_m']).any()):
                    raise ValueError('projected wrist translation exceeds original carriage limits')
                indices += binding.mount.ids + binding.mount.angular_ids
                values = torch.cat((values, local, angles[:, hi]), 1)
            expected[binding.side] = dict(joint_indices=indices, values=values.float().numpy().copy())
            prepared.append((permutation, target))
        if _clock(read_clock, self.env_ids) != current['identity']:
            raise ValueError('clock changed before first SDK write')
        # All pure validation happens before the first mutation; no retry after a write failure.
        self._state = 'failed'
        for hi, binding in enumerate(bindings):
            permutation, target = prepared[hi]
            device = binding.hand.data.joint_pos.device
            if current['hands'][binding.side]['mounted']:
                binding.mount.move(roots[permutation, hi].to(device),
                                   angles=angles[permutation, hi].to(device))
            binding.hand.set_joint_position_target(target[permutation].float().to(device),
                joint_ids=current['hands'][binding.side]['active_indices'])
            binding.hand.write_data_to_sim()
        if _clock(read_clock, self.env_ids) != current['identity']:
            raise ValueError('SDK/rod step occurred inside target submission')
        self._submission = dict(expected_backend_targets=expected,
            projected_root_poses=None if roots is None else roots.clone(),
            projected_angles=None if angles is None else angles.clone(),
            projection_source=projection_source, table_safety_verified_by_this_adapter=False)
        self._state = 'submitted'
        return deepcopy(self._submission)

    def finish(self, bindings, *, read_clock):
        if self._state != 'submitted':
            raise ValueError('submitted transaction required')
        self._state = 'failed'
        post = _capture(bindings, self.env_ids, read_clock)
        for pre, after in zip(self._pre['identity'], post['identity']):
            fixed = ('run_uuid', 'env_id', 'episode_uid', 'reset_id', 'rod_uid', 'policy_step', 'substep')
            if any(pre[key] != after[key] for key in fixed):
                raise ValueError('episode/reset/policy identity changed across physical step')
            if (after['physics_step'] != pre['physics_step'] + 1
                    or after['rod_tick'] != pre['rod_tick'] + self.solver_substeps
                    or not math.isclose(after['solver_time_s'] - pre['solver_time_s'],
                                        DT, abs_tol=1e-8, rel_tol=0)):
                raise ValueError('SDK/rod tick or actual solver clock skipped/repeated')
        for side in SIDES:
            pre, after = self._pre['hands'][side], post['hands'][side]
            if any(pre[k] != after[k] for k in ('joint_names', 'prim_paths', 'row_indices', 'mounted', 'instance_object_id', 'view_object_id')):
                raise ValueError('post-step instance mapping changed')
            expected = self._submission['expected_backend_targets'][side]
            actual = after['backend_targets'][:, expected['joint_indices']]
            if not np.allclose(actual, expected['values'], rtol=0, atol=1e-7):
                raise ValueError('actual backend target differs from submitted float32 target')
        self._state = 'finished'
        return deepcopy(dict(version=VERSION, env_ids=self.env_ids, pre=self._pre,
            request=self._request, submission=self._submission, post=post,
            observed_sdk_step_delta=1, observed_rod_tick_delta=self.solver_substeps,
            new_physics_steps_executed_by_adapter=0, optimizer_updates=0,
            qualified_for_teacher_ppo=False, native_qualified=False,
            target_readback_is_not_measured_joint_tracking=True))
