"""Differentiable commands conditional on recorded feedback, never dynamics.

Teacher and Student each retain their own cumulative command state. The
measured context is held fixed; this does not predict Student contact or pose.
Only the native surface/free-wrist path without stateful callbacks is admitted.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from .tabletop_command_kernels import (
    finger_pre_guard, surface_final_guard, wrist_requested_targets,
    palm_transport, per_hand_safe_z,
)
from .tabletop_free_wrist_grasp import CONTRACT, compose_rotation
from .tabletop_surface_contract import ACTION_CONTRACT


STATE_KEYS = ('joint_command_rad', 'palm_command_offset_m', 'wrist_command_angle_rad')
UNSUPPORTED_CALLBACKS = ('tripod_recovery_runtime', 'manual_pinch_home',
    'policy_lift_hold', 'palm_reposition', 'close_grasp_hold', 'contact_effort')


def action_contract_sha256():
    return hashlib.sha256(json.dumps(ACTION_CONTRACT, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def controller_source_sha256():
    digest = hashlib.sha256()
    for name in ('tabletop_command_kernels.py', 'tabletop_free_wrist_grasp.py',
                 'tabletop_tensor_world.py', 'tabletop_vision_command_shadow.py'):
        digest.update(name.encode()); digest.update((Path(__file__).parent/name).read_bytes())
    return digest.hexdigest()


def controller_contract(task, world):
    if not all(getattr(task, key, False) for key in
               ('policy_control', 'free_wrist_grasp', 'surface_force_sharing')):
        raise ValueError('only native 24D surface free-wrist policy control is supported')
    if getattr(task, 'surface_redistribution', None) is not False:
        raise ValueError('unsupported stateful surface_redistribution callback')
    for name in UNSUPPORTED_CALLBACKS:
        if getattr(task, name, None) is not None:
            raise ValueError('unsupported controller callback: '+name)
    if 'control' in vars(task) or 'advance' in vars(world):
        raise ValueError('unsupported replaced control/advance callback')
    result = dict(version='dh116-conditional-command-v1', profile='surface_free_wrist_v1',
        action_contract=deepcopy(ACTION_CONTRACT), free_wrist_contract=deepcopy(CONTRACT),
        surface_policy_owned_targets=bool(getattr(task, 'surface_policy_owned_targets', False)),
        pinch_force_hold=bool(getattr(task, 'pinch_force_hold', False)),
        redistribution=False, callbacks=[], physics_substeps=8,
        action_contract_sha256=action_contract_sha256(),
        controller_source_sha256=controller_source_sha256())
    return result


def _validate_contract(contract):
    if (contract.get('profile') != 'surface_free_wrist_v1'
            or contract.get('action_contract') != ACTION_CONTRACT
            or contract.get('free_wrist_contract') != CONTRACT
            or contract.get('redistribution') is not False or contract.get('callbacks') != []):
        raise ValueError('unsupported controller contract/profile/callbacks')
    if contract.get('controller_source_sha256') != controller_source_sha256():
        raise ValueError('controller source SHA256 mismatch')


def command_state(task, world):
    return {key: value.detach().clone() for key, value in zip(STATE_KEYS,
        (task.command, world.palms, world.wrist_angles))}


def measured_context(task, world, live):
    """Snapshot raw inputs before control; no consume(), SDK write, or sim step."""
    n = task.command.shape[0]
    def own(value):
        return value.detach().clone()
    floor = getattr(task, 'diagnostic_palm_descent_clearance_m',
        getattr(task, 'palm_descent_clearance_m', .0021))
    if getattr(task, 'pinch_force_hold', False):
        floor = .0003
    context = dict(dt_s=task.command.new_full((n,), world.dt),
        measured_q_rad=own(task.m['q']), measured_qd_rad_s=own(task.m['qd']),
        measured_root_pose=own(world.roots), root_seed_pose=own(world.root_seed),
        prior_root_target_pose=own(world.root_targets),
        clearance_m=own(world.clearances), task_offset_m=own(task.offset),
        clearance_floor_m=task.command.new_full((n,), floor),
        live=own(live), allowed=own(live), joint_lower=own(task.low.expand(n, -1)),
        joint_upper=own(task.high.expand(n, -1)), mimic_errors=own(task.m['mimic_errors']),
        teacher_force_controlled_digits=own(getattr(task, 'teacher_force_controlled_digits',
            torch.zeros(n, 2, 5, device=task.command.device, dtype=torch.bool))),
        active_controller_flags=torch.tensor([
            bool(getattr(task, 'surface_policy_owned_targets', False)),
            bool(getattr(task, 'pinch_force_hold', False))], device=task.command.device,
            dtype=torch.bool).expand(n, -1).clone())
    valid = torch.ones(n, device=live.device, dtype=torch.bool)
    for value in context.values():
        if value.dtype.is_floating_point:
            valid &= torch.isfinite(value).reshape(n, -1).all(-1)
    context['source_valid'] = valid
    return context


def shadow_step(command_state, action24, measured_context, controller_contract):
    _validate_contract(controller_contract)
    state, ctx, contract = command_state, measured_context, controller_contract
    action = action24.to(state['joint_command_rad'])
    if action.ndim != 2 or action.shape[-1] != 24 or not torch.isfinite(action).all():
        raise ValueError('finite [B,24] action required')
    source_valid = ctx['source_valid']
    if not bool(source_valid.all()):
        if bool((~source_valid & ctx['live']).any()):
            raise ValueError('source_valid=false on a live substep: incomplete feedback cannot unfold')
        # Missing proven-inactive suffixes are skipped, never filled with a
        # synthetic measurement. index_copy preserves gradients of live rows.
        index = source_valid.nonzero().flatten()
        new = {key: value for key, value in state.items()}
        diagnostics = dict(finger=state[STATE_KEYS[0]], palm=state[STATE_KEYS[1]],
            wrist_angle=state[STATE_KEYS[2]], physics_valid=source_valid & ctx['live'])
        if len(index):
            subset = lambda tree: {key: value[index] for key, value in tree.items()}
            partial, output = shadow_step(subset(state), action[index], subset(ctx), contract)
            new = {key: value.index_copy(0, index, partial[key]) for key, value in state.items()}
            for key in ('finger', 'palm', 'wrist_angle'):
                diagnostics[key] = diagnostics[key].index_copy(0, index, output[key])
        diagnostics['sdk_float32'] = {key: diagnostics[key].float() for key in ('finger', 'palm', 'wrist_angle')}
        return new, diagnostics
    dt_tensor = torch.as_tensor(ctx['dt_s'])
    if (not torch.isfinite(dt_tensor).all() or not (dt_tensor > 0).all()
            or not (dt_tensor == dt_tensor.flatten()[0]).all()):
        raise ValueError('matching positive native physics dt required')
    dt = float(dt_tensor.flatten()[0])
    rates = dict(joint_rate_rad_s=ACTION_CONTRACT['joint_rate_rad_s'],
        wrist_rate_m_s=ACTION_CONTRACT['translation_rate_m_s'],
        rotation_rate_rad_s=ACTION_CONTRACT['rotation_rate_rad_s'])
    command, palms, angles = (state[k] for k in STATE_KEYS)
    low, high, live, allowed = (ctx[k] for k in ('joint_lower', 'joint_upper', 'live', 'allowed'))
    flags = ctx['active_controller_flags']
    if (not (flags[:, 0] == contract['surface_policy_owned_targets']).all()
            or not (flags[:, 1] == contract['pinch_force_hold']).all()):
        raise ValueError('active controller flags changed inside recorded window')
    target = finger_pre_guard(command, action, ctx['measured_q_rad'], ctx['mimic_errors'],
        low, high, dt, rates, policy_owned=contract['surface_policy_owned_targets'],
        pinch_force_hold=contract['pinch_force_hold'],
        teacher_force_controlled_digits=ctx['teacher_force_controlled_digits'])
    requested, target = surface_final_guard(command, target, low, high, dt)
    command = torch.where(live[:, None], target, command)
    palm_target, angle_target = wrist_requested_targets(palms, angles, action, rates, dt)
    floors = ctx['clearance_floor_m']
    if not (floors == floors.flatten()[0]).all():
        raise ValueError('matching native clearance floor required')
    floor = float(floors.flatten()[0])
    palms, preprojection, reasons = palm_transport(palms, palm_target, ctx['clearance_m'],
        floor, CONTRACT['wrist_rate_m_s'], dt, action.new_tensor(CONTRACT['palm_lower_m']),
        action.new_tensor(CONTRACT['palm_upper_m']), allowed, live, vector_norm=True)
    per_hand, safe_z = per_hand_safe_z(palms, ctx['measured_root_pose'], ctx['root_seed_pose'],
        ctx['task_offset_m'], ctx['clearance_m'], floor)
    reasons = reasons | ((per_hand-palms[:, :, 2]).abs() > 1e-12).long()*2
    z = torch.where(live[:, None], per_hand, palms[:, :, 2])
    palms = torch.cat((palms[:, :, :2], z[:, :, None]), -1)
    angles = torch.where((allowed & live)[:, None, None], angle_target, angles)
    seed = ctx['root_seed_pose']
    position = seed[:, :, :3] + palms
    position = torch.cat((position[:, :, :2],
        (position[:, :, 2] + ctx['task_offset_m'][:, None])[:, :, None]), -1)
    root_pose = torch.cat((position, compose_rotation(seed[:, :, 3:], angles)), -1)
    root_pose = torch.where(live[:, None, None], root_pose, ctx['prior_root_target_pose'])
    return dict(zip(STATE_KEYS, (command, palms, angles))), dict(
        finger=command, palm=palms, wrist_angle=angles, root_pose=root_pose,
        requested_finger=requested, safe_z_m=safe_z, pre_projection_palm=preprojection,
        projection_reasons=reasons, physics_valid=source_valid & live, sdk_float32=dict(finger=command.float(),
            palm=palms.float(), wrist_angle=angles.float()),
        interpretation='conditional_recorded_feedback_unfolding_not_world_dynamics')


def _slice_context(tree, policy, substep):
    if isinstance(tree, dict):
        return {key: _slice_context(value, policy, substep) for key, value in tree.items()}
    return tree[:, policy, substep]


def shadow_sequence(initial_command_state, actions, contexts, controller_contract, *, physics_valid=None):
    if actions.ndim != 3 or actions.shape[-1] != 24:
        raise ValueError('actions must be [B,L,24]')
    if contexts['dt_s'].shape != (*actions.shape[:2], 8):
        raise ValueError('contexts must contain eight actual physics substeps per action')
    if physics_valid is not None:
        if physics_valid.shape != contexts['dt_s'].shape:
            raise ValueError('physics_valid must be [B,L,8]')
        if bool((~physics_valid & contexts['live']).any()):
            raise ValueError('cannot skip an expected live physics substep')
    state = initial_command_state
    targets = {key: [] for key in ('finger', 'palm', 'wrist_angle')}
    for policy in range(actions.shape[1]):
        tick = {key: [] for key in targets}
        for substep in range(8):
            state, out = shadow_step(state, actions[:, policy],
                _slice_context(contexts, policy, substep), controller_contract)
            for key in tick:
                tick[key].append(out[key])
        for key in targets:
            targets[key].append(torch.stack(tick[key], 1))
    return {key: torch.stack(value, 1) for key, value in targets.items()}
