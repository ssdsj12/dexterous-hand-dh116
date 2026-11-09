"""Explicit PointNet-only activation contract for newly derived models."""
from torch import nn

ACTIVATION_KEY = 'actor_pointnet_activation'
VERSION = 'dh116-vision-pointnet-activation-v1'


def make_point_activation(mode):
    if mode == 'relu':
        return nn.ReLU()
    if mode == 'leaky_relu_001':
        return nn.LeakyReLU(negative_slope=.01, inplace=False)
    raise ValueError('unsupported PointNet activation')


def point_activation_contract(mode):
    make_point_activation(mode)
    return dict(version=VERSION, mode=mode, dimensions=[3, 32, 64, 64],
                activation_indices=[1, 3, 5],
                negative_slope=.01 if mode == 'leaky_relu_001' else 0.)


def point_activation_from_contract(contract):
    if not isinstance(contract, dict):
        raise ValueError('PointNet activation requires a model contract')
    if ACTIVATION_KEY not in contract:
        return 'relu'  # All historical contracts used ReLU.
    value = contract[ACTIVATION_KEY]
    if not isinstance(value, dict):
        raise ValueError('versioned PointNet activation contract required')
    mode = value.get('mode')
    expected = point_activation_contract(mode)
    if value != expected:
        raise ValueError('PointNet activation contract differs from supported mode')
    return mode
