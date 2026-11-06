"""Tanh-transformed Gaussian evaluated at saved, pre-tanh latent samples.

This is a separate policy family from the legacy Gaussian plus actuator clamp.
Both have latent-policy interpretations. Never invert rounded bounded actions:
finite FP32 latents can legitimately produce commands equal to -1 or +1.
"""
import math

import torch
from torch.distributions import Normal
from torch.nn import functional as F


DISTRIBUTION_CONTRACT = dict(
    version='dh116-tanh-latent-gaussian-v1',
    latent_distribution='diagonal_normal', action_transform='tanh',
    action_log_probability='tanh_transformed_gaussian_at_saved_latent',
    deterministic_action='tanh_latent_mean',
    entropy='current_reparameterized_mc_tanh_entropy',
    kl='exact_latent_normal_old_to_new',
)


def require_finite(name, value):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite')


def validate_parameters(mu, std):
    require_finite('latent mean', mu)
    require_finite('latent std', std)
    if not bool((std > 0).all()):
        raise ValueError('latent std must be positive and finite')


def tanh_log_det_jacobian(z):
    """Stable elementwise log(1-tanh(z)^2), including rounded endpoints."""
    require_finite('latent actions', z)
    return 2.*(math.log(2.) - z - F.softplus(-2.*z))


def squashed_log_prob(z, mu, std):
    validate_parameters(mu, std)
    require_finite('latent actions', z)
    result = (Normal(mu, std).log_prob(z)-tanh_log_det_jacobian(z)).sum(-1)
    require_finite('transformed log probability', result)
    return result


def squashed_entropy(mu, std, epsilon):
    """One reparameterized current-policy sample per row (not rollout z)."""
    validate_parameters(mu, std)
    require_finite('entropy epsilon', epsilon)
    if epsilon.shape != mu.shape:
        raise ValueError('entropy epsilon must match latent mean shape')
    result = (Normal(mu, std).entropy()+tanh_log_det_jacobian(mu+std*epsilon)).sum(-1)
    require_finite('bounded policy entropy', result)
    return result


def latent_gaussian_kl(old_mu, old_std, new_mu, new_std):
    """Exact KL(old||new); invariant under a common bijective tanh map."""
    validate_parameters(old_mu, old_std)
    validate_parameters(new_mu, new_std)
    result = (torch.log(new_std/old_std)
              +(old_std.square()+(old_mu-new_mu).square())/(2.*new_std.square())-.5).sum(-1)
    require_finite('analytic KL', result)
    return result
