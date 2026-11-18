"""Versioned opt-in PointNet activations must not change historical actors."""
from copy import deepcopy
import importlib

import pytest
import torch
from torch import nn

from go2_pvcnn.control.dh116.tabletop_vision_student import VisionStudentActor, POLICY_DIM
from go2_pvcnn.control.dh116.tabletop_vision_squashed_policy import UnboundedVisionStudentActor


def test_explicit_relu_matches_legacy_state_and_output_bitwise():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        original = VisionStudentActor()
        state = deepcopy(original.state_dict())
        explicit = VisionStudentActor(point_activation='relu')
        explicit.load_state_dict(state)
        observation = torch.randn(2, POLICY_DIM)
        assert torch.equal(original(observation), explicit(observation))
        assert [type(explicit.point_net[i]) for i in (1, 3, 5)] == [nn.ReLU] * 3


def test_leaky_activation_retains_shape_and_negative_gradient():
    actor = UnboundedVisionStudentActor(point_activation='leaky_relu_001')
    assert actor.action_net[-1].__class__ is nn.Identity
    assert [type(actor.point_net[i]) for i in (1, 3, 5)] == [nn.LeakyReLU] * 3
    for i in (1, 3, 5):
        x = torch.tensor([-2., 1.], requires_grad=True)
        actor.point_net[i](x).sum().backward()
        assert torch.allclose(x.grad, torch.tensor([.01, 1.]))
    assert actor(torch.zeros(2, POLICY_DIM)).shape == (2, 24)


def test_activation_contract_is_explicit_and_rejects_unknown_or_malformed():
    module = importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_point_activation')
    assert module.point_activation_from_contract({}) == 'relu'
    contract = {module.ACTIVATION_KEY: module.point_activation_contract('leaky_relu_001')}
    assert module.point_activation_from_contract(contract) == 'leaky_relu_001'
    broken = deepcopy(contract)
    broken[module.ACTIVATION_KEY]['negative_slope'] = .1
    with pytest.raises(ValueError):
        module.point_activation_from_contract(broken)
    with pytest.raises(ValueError):
        VisionStudentActor(point_activation='auto_reset')


def test_new_tanh_checkpoint_preserves_declared_activation_in_inference(tmp_path):
    from test_vision_supervised_bc import source_v3, sensor_rows
    from go2_pvcnn.control.dh116.tabletop_vision_point_activation import (
        ACTIVATION_KEY, point_activation_contract,
    )
    from go2_pvcnn.control.dh116.tabletop_vision_inference import load_vision_inference
    _, saved = source_v3(tmp_path)
    saved['contract'][ACTIVATION_KEY] = point_activation_contract('leaky_relu_001')
    path = tmp_path/'explicit_leaky.pt'
    torch.save(saved, path)
    loaded = load_vision_inference(path, saved['contract'])
    assert loaded.actor.point_activation == 'leaky_relu_001'
    assert isinstance(loaded.actor.point_net[1], nn.LeakyReLU)
    assert loaded.act(sensor_rows()).shape == (8, 24)


def test_malformed_activation_rejected_by_checkpoint_validator(tmp_path):
    from test_vision_supervised_bc import source_v3
    from go2_pvcnn.control.dh116.tabletop_vision_action_contract import validate_squashed_checkpoint
    _, saved = source_v3(tmp_path)
    saved['contract']['actor_pointnet_activation'] = {'mode': 'relu'}
    with pytest.raises(ValueError, match='activation'):
        validate_squashed_checkpoint(saved)
