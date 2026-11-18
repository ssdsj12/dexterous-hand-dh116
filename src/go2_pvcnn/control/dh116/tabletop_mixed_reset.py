"""Fixed two-style native resets for one surface PPO batch.

Inputs are the unshifted adaptive_initialization layout/seed. Static supports,
object geometry and clone origins are shared. Only hand pose/joint reset state
differs; no simulator step, policy initialization or episode counter is changed.
"""
from copy import deepcopy

import numpy as np
import torch

from .linear_coupling import ACTIVE, JOINTS, expand
from .tabletop_end_supported_scene import translate_end_support_seed


def _array(value, shape, name, device):
    result = torch.as_tensor(np.asarray(value), device=device, dtype=torch.float64).clone()
    if tuple(result.shape) != shape or not bool(torch.isfinite(result).all()):
        raise ValueError(f'finite {name} with shape {shape} required')
    return result


def _equal(actual, expected, name, tolerance=1e-7):
    if actual.shape != expected.shape or not torch.allclose(actual, expected, atol=tolerance, rtol=0):
        raise ValueError(f'mixed reset requires matching {name}')


def install_mixed_reset(world, task, secondary_layout, secondary_seed):
    """Install a permanent half/half allocation after host/tensor preflight.

    Restricted to the 24D surface/free-wrist path: force closure direction and
    legacy pressure compensation are not used by this controller.
    """
    if world.n != task.n or world.n < 2 or world.n % 2:
        raise ValueError('mixed reset requires an even matching environment count')
    if not (getattr(task, 'surface_force_sharing', False)
            and getattr(task, 'free_wrist_grasp', False)
            and getattr(world, 'continuous_motion', False)
            and getattr(task, 'surface_redistribution', None) is False):
        raise ValueError('mixed reset requires policy-owned surface free-wrist control')
    if hasattr(task, '_per_env_reset_seed'):
        raise ValueError('mixed reset is already installed')
    layout = deepcopy(secondary_layout)
    scene = getattr(world.w, 'end_support_scene', None)
    seed = translate_end_support_seed(secondary_seed, scene)
    # Match GpuWorld: layout and seed each receive the common support shift once.
    if scene is not None:
        for side in ('left', 'right'):
            pose = np.asarray(layout['hand_roots'][side], dtype=float).copy()
            pose[2] += scene['dz_m']
            layout['hand_roots'][side] = pose
        center = np.asarray(layout['wire_center'], dtype=float).copy()
        center[2] += scene['dz_m']
        layout['wire_center'] = center
    device = world.root_seed.device
    array = lambda value, shape, name: _array(value, shape, name, device)
    offset = array(seed['root_offset'], (), 'root offset')
    _equal(offset, array(world.initial_offset, (), 'primary offset'), 'root offset')
    for name, shape in [('wire_pose', (7,)), ('wire_velocity', (6,))]:
        _equal(array(seed[name], shape, name),
               array(world.w.seed[name], shape, name), name)
    for name in ('wire_center', 'table_center', 'table_size'):
        _equal(array(layout[name], (3,), name),
               array(world.w.layout[name], (3,), name), name)
    for name in ('wire_length_m', 'wire_radius_m', 'wire_axis'):
        if layout.get(name) != world.w.layout.get(name):
            raise ValueError('mixed reset requires matching wire geometry: '+name)
    roots = array([layout['hand_roots'][side] for side in ('left', 'right')], (2, 7), 'roots')
    if not torch.allclose(roots[:, 3:].norm(dim=1), torch.ones(2, device=device, dtype=roots.dtype),
                          atol=1e-6, rtol=0):
        raise ValueError('mixed root quaternions must be unit length')
    xy = array(seed['root_xy'], (2,), 'root_xy')
    q = array(seed['joint_positions'], (2, 11), 'joint positions')
    qd = array(seed['joint_velocities'], (2, 11), 'joint velocities')
    initial = array(seed['initial_command'], (12,), 'initial command')
    grasp = array(seed['grasp_command'], (12,), 'grasp command')
    active = [JOINTS.index(name) for name in ACTIVE]
    _equal(q[:, active].reshape(12), initial, 'joint/initial command')
    for hi in range(2):
        coupled = array(expand(initial[hi*6:hi*6+6].cpu().tolist()), (11,), 'coupled joints')
        _equal(q[hi], coupled, 'joint coupling')
    if bool(((initial < task.low) | (initial > task.high)).any()):
        raise ValueError('mixed initial command outside physical joint limits')
    if bool(((grasp < task.low) | (grasp > task.high)).any()):
        raise ValueError('mixed grasp command outside physical joint limits')
    if 'active_joints' in layout:
        _equal(array([layout['active_joints'][s] for s in ('left', 'right')], (2, 6), 'grasp'),
               grasp.reshape(2, 6), 'layout grasp command')
    # Prepare all replacements before mutating the live reset state.
    half = world.n // 2
    expected_roots = world.root_seed.clone()
    expected_goals = world.goal_roots.clone()
    expected_roots[half:] = roots
    expected_roots[half:, :, :3] += world.origins[half:, None]
    expected_roots[half:, :, :2] += xy
    expected_goals[half:] = roots[:, :3] + world.origins[half:, None]
    seed_q, seed_qd = [value.clone() for value in world.seed_q], [value.clone() for value in world.seed_qd]
    for hi in range(2):
        seed_q[hi][half:, world.w.named_ids[hi]] = q[hi].to(seed_q[hi])
        seed_qd[hi][half:, world.w.named_ids[hi]] = qd[hi].to(seed_qd[hi])
    replacements = {}
    for name in ('reference', 'command', 'previous', 'old_command',
                 'joint_requested_target', 'joint_governed_target', 'filtered_residual'):
        current = getattr(task, name)
        primary = task._seed.get(name, task._seed['command'])
        value = primary.expand_as(current).clone()
        value[half:] = (grasp if name == 'reference' else
                       torch.zeros_like(initial) if name == 'filtered_residual' else initial)
        replacements[name] = value
    world.root_seed.copy_(expected_roots)
    world.goal_roots.copy_(expected_goals)
    for hi in range(2):
        world.seed_q[hi].copy_(seed_q[hi])
        world.seed_qd[hi].copy_(seed_qd[hi])
    task._per_env_reset_seed = replacements
    world.mixed_reset_style_ids = torch.arange(world.n, device=device) // half
    return validate_mixed_reset(world, task)


def validate_mixed_reset(world, task):
    """Read back full and selective real tensor resets without advancing physics."""
    if not hasattr(task, '_per_env_reset_seed'):
        raise ValueError('mixed reset is not installed')
    ids = torch.arange(world.n, device=world.root_seed.device)
    half = world.n // 2
    expected_pose = world.root_seed.clone()
    expected_pose[:, :, 2] += world.initial_offset
    expected_q = torch.cat([world.seed_q[h][:, world.w.ids[h]].double() for h in range(2)], dim=1)
    expected_all_q = torch.stack(world.seed_q, dim=1).double()

    def check(actual, expected, name, tolerance=2e-5):
        if actual.shape != expected.shape or not bool(torch.isfinite(actual).all()):
            raise RuntimeError('invalid mixed reset '+name)
        error = (actual.double() - expected.double()).abs().max().item()
        if error > tolerance:
            raise RuntimeError(f'mixed reset {name} mismatch: max error {error}')
        return error

    def reset(selected):
        measurement = world.reset(selected)
        task.reset(selected, measurement)
        return measurement

    reset(ids)
    pose_error = check(world.roots[:, :, :3], expected_pose[:, :, :3], 'root pose')
    # Quaternion signs are equivalent; native readback can normalize/flip them.
    actual_quat = world.roots[:, :, 3:]
    expected_quat = expected_pose[:, :, 3:]
    signs = torch.where((actual_quat * expected_quat).sum(-1, keepdim=True) < 0, -1., 1.)
    quaternion_error = check(actual_quat * signs, expected_quat, 'root quaternion')
    joint_error = check(world.q, expected_all_q, 'joint position')
    check(task.m['q'], expected_q, 'measured active joint')
    check(torch.cat((world.wp, world.wq), dim=1), world.seed_wire, 'wire pose')
    check(world.wv, world.seed_velocity, 'wire velocity')
    for name, expected in task._per_env_reset_seed.items():
        check(getattr(task, name), expected, 'task '+name)
    check(task.observe(), task.last_observation, 'observation cache')
    # Keep marker values solely in task bookkeeping, never in physical state.
    # A reset in either style must not erase a nonselected row's live episode.
    isolation = []
    try:
        for selected in (ids[:1], ids[half:half+1]):
            untouched = ids[~torch.isin(ids, selected)]
            task.elapsed.fill_(1.)
            before_pose, before_q = world.roots[untouched].clone(), world.q[untouched].clone()
            before_command = task.command[untouched].clone()
            reset(selected)
            check(world.roots[untouched], before_pose, 'unselected root isolation', 1e-7)
            check(world.q[untouched], before_q, 'unselected joint isolation', 1e-7)
            check(task.command[untouched], before_command, 'unselected task isolation', 1e-7)
            check(task.elapsed[untouched], torch.ones_like(task.elapsed[untouched]), 'episode isolation', 0.)
            check(task.elapsed[selected], torch.zeros_like(task.elapsed[selected]), 'selected reset clock', 0.)
            for name, expected in task._per_env_reset_seed.items():
                check(getattr(task, name)[selected], expected[selected], 'selected '+name)
            isolation.append(int(selected[0]))
    finally:
        reset(ids)
    rows = []
    for index, label in ((0, 'pinch_thumb_index'), (half, 'tripod_thumb_index_middle')):
        pose = world.roots[index].clone()
        pose[:, :3] -= world.origins[index]
        rows.append(dict(style=label, env_id=index, actual_local_root_pose=pose.cpu().tolist(),
                         actual_active_q=task.m['q'][index].cpu().tolist()))
    return dict(status='passed', counts=[half, half], allocation='fixed_first_half_pinch_second_half_tripod',
                root_position_max_error_m=pose_error, root_quaternion_max_error=quaternion_error,
                joint_position_max_error_rad=joint_error, selective_reset_env_ids=isolation,
                unselected_rows_unchanged=True, physics_steps_advanced=0, representatives=rows)
