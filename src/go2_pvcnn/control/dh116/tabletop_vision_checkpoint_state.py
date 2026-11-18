"""Shared, side-effect-free validation of complete visual PPO checkpoint state.

Version and contract provenance are validated by the caller. This module has
no dependency on action-contract or training code, and creates no policy or
optimizer. The topology and Adam parameter IDs describe the existing 3844D
visual actor, 371D privileged critic and unchanged 24D command.
"""
import math
from numbers import Real

import torch


VISUAL_POLICY_PARAMETERS = (
    ('log_std', (24,)),
    ('actor.point_net.0.weight', (32, 3)), ('actor.point_net.0.bias', (32,)),
    ('actor.point_net.2.weight', (64, 32)), ('actor.point_net.2.bias', (64,)),
    ('actor.point_net.4.weight', (64, 64)), ('actor.point_net.4.bias', (64,)),
    ('actor.action_net.0.weight', (128, 772)), ('actor.action_net.0.bias', (128,)),
    ('actor.action_net.2.weight', (128, 128)), ('actor.action_net.2.bias', (128,)),
    ('actor.action_net.4.weight', (24, 128)), ('actor.action_net.4.bias', (24,)),
    ('critic.0.weight', (128, 371)), ('critic.0.bias', (128,)),
    ('critic.2.weight', (128, 128)), ('critic.2.bias', (128,)),
    ('critic.4.weight', (1, 128)), ('critic.4.bias', (1,)),
)


def _float32_tensor(value, shape, name):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
            or value.dtype != torch.float32 or value.layout != torch.strided
            or value.is_quantized or not bool(torch.isfinite(value).all())):
        raise ValueError('visual checkpoint requires finite float32 '+name+' with shape '+str(shape))


def _validate_step(value, name):
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.dtype == torch.bool or value.is_complex():
            raise ValueError('visual Adam step must be a finite nonnegative scalar: '+name)
        value = value.item()
    try:
        valid = (not isinstance(value, bool) and isinstance(value, Real)
                 and math.isfinite(value) and value >= 0)
    except (ValueError, TypeError, OverflowError):
        valid = False
    if not valid:
        raise ValueError('visual Adam step must be a finite nonnegative scalar: '+name)


def _finite_real(value):
    try:
        return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def _validate_adam_group(group, learning_rate):
    for name in ('lr', 'betas', 'eps', 'weight_decay'):
        if name not in group:
            raise ValueError('visual Adam group is missing required hyperparameter: '+name)
    for name in ('lr', 'eps', 'weight_decay'):
        value = group[name]
        if not _finite_real(value) or value < 0:
            raise ValueError('visual Adam '+name+' must be a finite nonnegative scalar')
    if (not _finite_real(learning_rate) or learning_rate <= 0
            or group['lr'] != learning_rate):
        raise ValueError('visual Adam lr must match the saved finite positive learning_rate')
    betas = group['betas']
    if (not isinstance(betas, (tuple, list)) or len(betas) != 2
            or any(not isinstance(beta, float) or not math.isfinite(beta)
                   or not 0 <= beta < 1 for beta in betas)):
        raise ValueError('visual Adam betas must be two finite floats in [0, 1)')
    for name in ('amsgrad', 'maximize', 'capturable', 'differentiable', 'decoupled_weight_decay'):
        if name in group and type(group[name]) is not bool:
            raise ValueError('visual Adam '+name+' setting must be boolean')
    if group.get('maximize', False):
        raise ValueError('visual PPO minimizes its loss; Adam maximize=True is unsupported')
    for name in ('foreach', 'fused'):
        if name in group and group[name] is not None and type(group[name]) is not bool:
            raise ValueError('visual Adam '+name+' setting must be boolean or None')
    if group.get('fused', False) and (group.get('foreach', False)
                                     or group.get('differentiable', False)):
        raise ValueError('visual Adam fused cannot be combined with foreach or differentiable')


def validate_visual_checkpoint_state(saved):
    """Reject incomplete policy/Adam/noise/RNG state without modifying it.

    At iteration zero Adam may have no moments yet. A completed PPO iteration
    requires state for every actor, critic and noise parameter. The CPU RNG
    bytes are tested on a private generator; the process RNG is never changed.
    Saved CUDA RNG remains opaque here and is preserved by the caller.
    """
    if not isinstance(saved, dict):
        raise ValueError('visual checkpoint must be a dictionary')
    iteration = saved.get('iteration')
    if type(iteration) is not int or iteration < 0:
        raise ValueError('visual checkpoint iteration must be a nonnegative integer')
    policy = saved.get('policy')
    names = [name for name, shape in VISUAL_POLICY_PARAMETERS]
    if not isinstance(policy, dict) or policy.keys() != set(names):
        raise ValueError('visual checkpoint requires the complete visual policy topology')
    for name, shape in VISUAL_POLICY_PARAMETERS:
        _float32_tensor(policy[name], shape, 'policy.'+name)
    std = policy['log_std'].exp()
    tolerance = 4*torch.finfo(torch.float32).eps
    if not bool(((std >= .001*(1-tolerance)) & (std <= .3*(1+tolerance))).all()):
        raise ValueError('visual checkpoint latent std must be within .001 and .3')

    optimizer = saved.get('optimizer')
    groups = optimizer.get('param_groups') if isinstance(optimizer, dict) else None
    state = optimizer.get('state') if isinstance(optimizer, dict) else None
    identifiers = list(range(len(VISUAL_POLICY_PARAMETERS)))
    if (not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], dict)
            or not isinstance(state, dict)):
        raise ValueError('visual checkpoint requires one canonical Adam parameter group')
    group = groups[0]
    parameters = group.get('params')
    if (not isinstance(parameters, list) or parameters != identifiers
            or any(type(value) is not int for value in parameters)
            or any(type(key) is not int or key not in identifiers for key in state)
            or ('param_names' in group and group['param_names'] != names)):
        raise ValueError('visual checkpoint has an incompatible Adam parameter mapping')
    _validate_adam_group(group, saved.get('learning_rate'))
    if iteration > 0 and state.keys() != set(identifiers):
        raise ValueError('completed visual PPO iteration requires complete Adam parameter state')
    for identifier, (name, shape) in enumerate(VISUAL_POLICY_PARAMETERS):
        entry = state.get(identifier, {})
        if not isinstance(entry, dict):
            raise ValueError('visual Adam parameter state must be a dictionary: '+name)
        if not entry and iteration == 0:
            continue
        required = ('step', 'exp_avg', 'exp_avg_sq')
        if group.get('amsgrad', False): required += ('max_exp_avg_sq',)
        if any(field not in entry for field in required):
            raise ValueError('visual checkpoint has incomplete Adam parameter state: '+name)
        _validate_step(entry['step'], name)
        for field in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if field not in entry: continue
            value = entry[field]
            _float32_tensor(value, shape, 'Adam.'+name+'.'+field)
            if field != 'exp_avg' and bool((value < 0).any()):
                raise ValueError('visual Adam second moments must be nonnegative: '+name+'.'+field)

    rng = saved.get('torch_rng')
    if (not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8
            or rng.ndim != 1 or rng.layout != torch.strided):
        raise ValueError('visual checkpoint requires valid Torch CPU RNG bytes')
    try:
        torch.Generator(device='cpu').set_state(rng.detach().cpu())
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError('visual checkpoint has an invalid Torch CPU RNG state') from error
