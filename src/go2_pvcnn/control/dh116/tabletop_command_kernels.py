"""Pure command arithmetic shared by native control and conditional unfolding.

No SDK, simulation, learned dynamics, or controller-state mutation lives here.
The order of operations follows the original free-wrist controller exactly.
"""
import math

import torch

from .linear_coupling import ACTIVE, MIMIC
from .tabletop_velocity_limits import ACTIVE_VELOCITY_LIMIT_RAD_S


def finger_pre_guard(command, action, q, mimic_errors, low, high, dt, rates,
                     *, policy_owned=False, pinch_force_hold=False,
                     teacher_force_controlled_digits=None):
    from .tabletop_free_wrist_grasp import CONTRACT
    target = command + action[:, :12] * rates['joint_rate_rad_s'] * dt
    for j, (_, master, ratio, _) in enumerate(MIMIC):
        if policy_owned or (pinch_force_hold and j < 2):
            continue
        for h in range(2):
            k = 6*h + ACTIVE.index(master)
            error = mimic_errors[:, 5*h+j]
            correction = (5.*error/ratio*dt).clamp(
                -ACTIVE_VELOCITY_LIMIT_RAD_S*dt, ACTIVE_VELOCITY_LIMIT_RAD_S*dt)
            correct = error.abs() >= .012
            if teacher_force_controlled_digits is not None:
                correct &= ~teacher_force_controlled_digits[:, h, j]
            target[:, k] += torch.where(correct, correction, 0.)
    lead = CONTRACT['joint_target_lead_rad']
    target = target.maximum(q-lead).minimum(q+lead).maximum(low).minimum(high)
    return command + (target-command).clamp(
        -ACTIVE_VELOCITY_LIMIT_RAD_S*dt, ACTIVE_VELOCITY_LIMIT_RAD_S*dt)


def surface_final_guard(command, target, low, high, dt):
    target = command + (target-command).clamp(
        -ACTIVE_VELOCITY_LIMIT_RAD_S*dt, ACTIVE_VELOCITY_LIMIT_RAD_S*dt)
    return target, target.maximum(low).minimum(high)


def limited_step(action, rate, dt):
    norm = action.norm(dim=-1, keepdim=True).clamp_min(1.)
    return action/norm*(rate*dt)


def wrist_requested_targets(palms, angles, actions, rates, dt):
    from .tabletop_free_wrist_grasp import CONTRACT
    n = actions.shape[0]
    lo = actions.new_tensor(CONTRACT['palm_lower_m'])
    hi = actions.new_tensor(CONTRACT['palm_upper_m'])
    palms = (palms + limited_step(actions[:, 12:18].reshape(n, 2, 3),
        rates['wrist_rate_m_s'], dt)).maximum(lo).minimum(hi)
    angles = (angles + limited_step(actions[:, 18:24].reshape(n, 2, 3),
        rates['rotation_rate_rad_s'], dt)).clamp(-math.pi/2, math.pi/2)
    return palms, angles


def palm_transport(palms, target, clearances, clearance_floor, rate, dt,
                   lower, upper, allowed, live, *, vector_norm):
    from .tabletop_tensor_world import constrain_palm_z_movement
    movement = (target-palms).clamp(-rate*dt, rate*dt)
    if vector_norm:
        norm = movement.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        movement = movement*((rate*dt)/norm).clamp_max(1.)
    movement = constrain_palm_z_movement(movement, clearances, clearance_floor)
    pre_projection = palms + movement
    reasons = (movement.abs().amax(-1) >= (rate*dt)-1e-12).long()
    palms = torch.where((allowed & live)[:, None, None],
        torch.maximum(torch.minimum(palms+movement, upper), lower), palms)
    return palms, pre_projection, reasons


def per_hand_safe_z(palms, roots, root_seed, offset, clearances, clearance_floor):
    measured_z = roots[:, :, 2] - root_seed[:, :, 2] - offset[:, None]
    safe_z = measured_z - (clearances-clearance_floor)
    return torch.maximum(palms[:, :, 2], safe_z), safe_z
