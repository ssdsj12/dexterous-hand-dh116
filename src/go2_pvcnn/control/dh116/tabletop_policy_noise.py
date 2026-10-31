"""Projected Gaussian exploration for normalized, actuator-clipped actions."""
import math
import torch

CONTRACT = dict(version='bounded_policy_noise_v1',min_std=.05,max_std=1.,
    parameter='log_std',projection='after_each_optimizer_step_and_resume',
    projected_adam_moments='zero_only_changed_coordinates',
    action_log_probability='unclipped_gaussian_sample')


def _set_policy_log_std(algorithm, bounded):
    parameter=algorithm.policy.log_std
    changed=parameter!=bounded
    parameter.copy_(bounded)
    state=algorithm.optimizer.state.get(parameter,{})
    for name in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
        if name in state:state[name].masked_fill_(changed,0.)
    return changed


@torch.no_grad()
def configure_surface_noise(algorithm, initial_std=.005, min_std=.001, max_std=.3):
    """Opt a 24-action surface policy into lower Gaussian exploration noise.

    Validate before changing the policy or optimizer, then initialize the actual
    log_std and clear only its changed Adam coordinates. Return the independent
    contract also stored on algorithm.policy_noise_contract. Existing projection
    hooks read this contract on each step; sampling and likelihoods are unchanged.
    """
    message='surface noise requires finite 0 < min_std <= initial_std <= max_std'
    try:
        initial_std, min_std, max_std = map(float, (initial_std, min_std, max_std))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if (not all(math.isfinite(value) for value in (initial_std, min_std, max_std))
            or not 0 < min_std <= initial_std <= max_std):
        raise ValueError(message)
    parameter=algorithm.policy.log_std
    if parameter.ndim != 1 or parameter.numel() != 24:
        raise ValueError('surface policy noise requires exactly 24 actions')
    if not bool(torch.isfinite(parameter).all()):
        raise RuntimeError('nonfinite policy log_std before noise projection')
    contract=dict(CONTRACT, version='bounded_surface_policy_noise_v2',
        initial_std=initial_std, min_std=min_std, max_std=max_std)
    _set_policy_log_std(algorithm, torch.full_like(parameter, math.log(initial_std)))
    algorithm.policy_noise_contract=contract
    return contract


@torch.no_grad()
def project_policy_noise(algorithm):
    """Bound finite noise without changing actor/critic or unrelated Adam state.

    Keep the actual Gaussian distribution, stored raw actions and PPO likelihood
    consistent. Projection constrains its trainable scale; actuator clipping is
    still part of the environment and is separately measured.
    """
    parameter=algorithm.policy.log_std
    if not bool(torch.isfinite(parameter).all()):
        raise RuntimeError('nonfinite policy log_std before noise projection')
    contract=getattr(algorithm,'policy_noise_contract',None)
    if contract is None:contract=CONTRACT
    bounded=parameter.clamp(math.log(contract['min_std']),math.log(contract['max_std']))
    return _set_policy_log_std(algorithm, bounded)


def install_noise_projection(algorithm):
    # Runs after every Adam minibatch, not just after the entire PPO update.
    algorithm.optimizer.register_step_post_hook(lambda *_: project_policy_noise_hook(algorithm))


def project_policy_noise_hook(algorithm):
    project_policy_noise(algorithm)
