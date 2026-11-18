"""Masked sensor-actor losses for recorded-feedback conditional command unfolds.

Commands must come from the faithful command shadow, not unbounded velocity
integration. This objective does not predict contact or establish native success.
"""
import math

import torch
import torch.nn.functional as F


def sequence_bc_loss(pred_action, expert_action, valid, *, pred_commands=None,
                     expert_commands=None, physics_valid=None,
                     command_weight=.1, delta_weight=0.):
    """Return ``(total, metrics)``; all Teacher labels are detached.

    Padding must be finite and is excluded from every reduction. The command
    groups receive equal weight after their respective physical normalization.
    The delta term fits Teacher changes rather than suppressing action changes.
    """
    for name, weight in (('command_weight', command_weight), ('delta_weight', delta_weight)):
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not isinstance(pred_action, torch.Tensor) or pred_action.ndim != 3 or pred_action.shape[-1] != 24:
        raise ValueError('actions must be [B,L,24]')
    if not isinstance(expert_action, torch.Tensor) or expert_action.shape != pred_action.shape:
        raise ValueError('expert action shape mismatch')
    if not isinstance(valid, torch.Tensor) or valid.shape != pred_action.shape[:2]:
        raise ValueError('valid mask shape mismatch')
    if valid.dtype != torch.bool or not bool(valid.any()):
        raise ValueError('valid must be a nonempty boolean mask')
    if pred_action.device != expert_action.device or pred_action.device != valid.device:
        raise ValueError('actions and valid must share a device')
    if (not pred_action.is_floating_point() or not expert_action.is_floating_point()
            or not bool(torch.isfinite(pred_action).all()) or not bool(torch.isfinite(expert_action).all())):
        raise ValueError('nonfinite or nonfloating action')
    expert_action = expert_action.detach()
    error = pred_action - expert_action
    action_loss = error.square().mean(-1)[valid].mean()
    delta_loss = pred_action.sum() * 0.
    pair_valid = valid[:, 1:] & valid[:, :-1]
    if bool(pair_valid.any()):
        delta_error = error[:, 1:] - error[:, :-1]
        delta_loss = delta_error.square().mean(-1)[pair_valid].mean()
    command_loss = pred_action.sum() * 0.
    if command_weight > 0:
        if not isinstance(pred_commands, dict) or not isinstance(expert_commands, dict) or physics_valid is None:
            raise ValueError('command loss requires shadow commands and masks')
        if (not isinstance(physics_valid, torch.Tensor) or physics_valid.ndim != 3
                or physics_valid.shape[:2] != valid.shape or physics_valid.dtype != torch.bool
                or physics_valid.device != valid.device):
            raise ValueError('physics_valid must be boolean [B,L,K] on the action device')
        mask = physics_valid & valid[..., None]
        if not bool(mask.any()):
            raise ValueError('no valid conditional command samples')
        terms = []
        for name, scale, tail in (('finger', .01, (12,)), ('palm', .0005, (2, 3)),
                                  ('wrist_angle', .01, (2, 3))):
            predicted, target = pred_commands.get(name), expert_commands.get(name)
            shape = (*mask.shape, *tail)
            if (not isinstance(predicted, torch.Tensor) or not isinstance(target, torch.Tensor)
                    or predicted.shape != shape or target.shape != shape
                    or predicted.device != valid.device or target.device != valid.device):
                raise ValueError(f'invalid command shape or device: {name}')
            if (not predicted.is_floating_point() or not target.is_floating_point()
                    or not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(target).all())):
                raise ValueError('nonfinite or nonfloating command')
            error_command = (predicted - target.detach()) / scale
            term = F.smooth_l1_loss(error_command, torch.zeros_like(error_command), beta=1., reduction='none')
            terms.append(term.flatten(3).mean(-1)[mask].mean())
        command_loss = torch.stack(terms).mean()
    total = action_loss + command_weight * command_loss + delta_weight * delta_loss
    metrics = dict(action=action_loss, command=command_loss, delta=delta_loss)
    for name, start, end in (('fingers', 0, 12), ('left_fingers', 0, 6), ('right_fingers', 6, 12),
                             ('world_xyz', 12, 18), ('intrinsic_xyz_rotation', 18, 24),
                             ('left_palm_z', 14, 15), ('right_palm_z', 17, 18)):
        metrics[f'{name}_mse'] = error[..., start:end].square().mean(-1)[valid].mean()
    return total, metrics
