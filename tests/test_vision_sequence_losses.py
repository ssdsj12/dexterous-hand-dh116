"""CPU loss/gradient checks; these do not establish native grasp success."""
import importlib

import pytest
import torch


def loss(*args, **kwargs):
    module = importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_sequence_losses')
    return module.sequence_bc_loss(*args, **kwargs)


def commands(value, *, requires_grad=False):
    return {name: torch.full(shape, value * scale, requires_grad=requires_grad)
            for name, shape, scale in (
                ('finger', (1, 3, 8, 12), .01),
                ('palm', (1, 3, 8, 2, 3), .0005),
                ('wrist_angle', (1, 3, 8, 2, 3), .01))}


def test_masked_action_and_adjacent_delta_fit_changes_instead_of_smoothing():
    expert = torch.zeros(1, 3, 24, requires_grad=True)
    pred = torch.tensor([0., 1., 10.]).view(1, 3, 1).expand(1, 3, 24).clone().requires_grad_()
    valid = torch.tensor([[True, True, False]])
    total, metrics = loss(pred, expert, valid, command_weight=0., delta_weight=.02)
    assert total.item() == pytest.approx(.52)
    assert metrics['action'].item() == pytest.approx(.5)
    assert metrics['delta'].item() == pytest.approx(1.)
    total.backward()
    assert expert.grad is None
    assert torch.count_nonzero(pred.grad[:, 2]) == 0
    pred = torch.tensor([0., 1., 0.]).view(1, 3, 1).expand(1, 3, 24).requires_grad_()
    _, metrics = loss(pred, pred.detach(), torch.ones(1, 3, dtype=torch.bool), command_weight=0.)
    assert metrics['delta'].item() == 0.


def test_commands_normalized_per_group_and_teacher_detached():
    pred = torch.zeros(1, 3, 24, requires_grad=True)
    teacher = torch.zeros_like(pred, requires_grad=True)
    student_commands, teacher_commands = commands(1., requires_grad=True), commands(0., requires_grad=True)
    valid = torch.tensor([[True, False, True]])
    physics = torch.ones(1, 3, 8, dtype=torch.bool)
    physics[:, 2, 1:] = False
    total, metrics = loss(pred, teacher, valid, pred_commands=student_commands,
                         expert_commands=teacher_commands, physics_valid=physics)
    assert metrics['command'].item() == pytest.approx(.5)
    assert total.item() == pytest.approx(.05)
    total.backward()
    assert teacher.grad is None
    for name in student_commands:
        grad = student_commands[name].grad
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad[:, 0]) > 0
        assert torch.count_nonzero(grad[:, 1]) == 0
        assert torch.count_nonzero(grad[:, 2, 1:]) == 0
        assert teacher_commands[name].grad is None


def test_nonadjacent_valid_samples_do_not_receive_delta_loss():
    pred = torch.ones(1, 3, 24, requires_grad=True)
    _, metrics = loss(pred, torch.zeros_like(pred), torch.tensor([[True, False, True]]), command_weight=0.)
    assert metrics['delta'].item() == 0.


@pytest.mark.parametrize('bad', ['shape', 'teacher_shape', 'valid_shape', 'valid_dtype', 'empty', 'nan', 'integer'])
def test_invalid_actions_or_masks_fail_closed(bad):
    pred, teacher, valid = torch.zeros(1, 3, 24), torch.zeros(1, 3, 24), torch.ones(1, 3, dtype=torch.bool)
    if bad == 'shape': pred = pred.squeeze(0)
    elif bad == 'teacher_shape': teacher = teacher[:, :-1]
    elif bad == 'valid_shape': valid = valid[:, :-1]
    elif bad == 'valid_dtype': valid = valid.float()
    elif bad == 'empty': valid[:] = False
    elif bad == 'nan': pred[0, 2, 0] = float('nan')
    elif bad == 'integer': pred = pred.long()
    with pytest.raises(ValueError): loss(pred, teacher, valid, command_weight=0.)


@pytest.mark.parametrize('bad', ['missing', 'missing_key', 'mask_dtype', 'mask_shape', 'empty', 'shape', 'nan', 'weight'])
def test_command_loss_requires_real_finite_shadow_targets_and_masks(bad):
    pred, teacher = torch.zeros(1, 3, 24), torch.zeros(1, 3, 24)
    valid, physics = torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 3, 8, dtype=torch.bool)
    pc, tc = commands(0.), commands(0.)
    weight = .1
    if bad == 'missing': pc = None
    elif bad == 'missing_key': pc.pop('finger')
    elif bad == 'mask_dtype': physics = physics.float()
    elif bad == 'mask_shape': physics = physics[:, :, 0]
    elif bad == 'empty': physics[:] = False
    elif bad == 'shape': pc['palm'] = pc['palm'][:, :, :, 0]
    elif bad == 'nan': pc['palm'][0, 0, 0, 0, 0] = float('nan')
    elif bad == 'weight': weight = float('nan')
    with pytest.raises(ValueError):
        loss(pred, teacher, valid, pred_commands=pc, expert_commands=tc,
             physics_valid=physics, command_weight=weight)


def test_group_statistics_preserve_left_right_z_semantics():
    pred, teacher = torch.zeros(1, 1, 24), torch.zeros(1, 1, 24)
    pred[0, 0, 14], pred[0, 0, 17] = .2, -.4
    _, metrics = loss(pred, teacher, torch.ones(1, 1, dtype=torch.bool), command_weight=0.)
    assert metrics['left_palm_z_mse'].item() == pytest.approx(.04)
    assert metrics['right_palm_z_mse'].item() == pytest.approx(.16)
    assert metrics['fingers_mse'].item() == 0.
