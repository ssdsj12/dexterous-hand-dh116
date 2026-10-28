"""Projected Gaussian exploration for normalized, actuator-clipped actions."""
import math
import torch

CONTRACT = dict(version='bounded_policy_noise_v1',min_std=.05,max_std=1.,
    parameter='log_std',projection='after_each_optimizer_step_and_resume',
    projected_adam_moments='zero_only_changed_coordinates',
    action_log_probability='unclipped_gaussian_sample')


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
    bounded=parameter.clamp(math.log(CONTRACT['min_std']),math.log(CONTRACT['max_std']))
    changed=parameter!=bounded
    parameter.copy_(bounded)
    state=algorithm.optimizer.state.get(parameter,{})
    for name in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
        if name in state:state[name].masked_fill_(changed,0.)
    return changed


def install_noise_projection(algorithm):
    # Runs after every Adam minibatch, not just after the entire PPO update.
    algorithm.optimizer.register_step_post_hook(lambda *_: project_policy_noise_hook(algorithm))


def project_policy_noise_hook(algorithm):
    project_policy_noise(algorithm)
