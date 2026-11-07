"""Sensor-only point cloud actor with a separate privileged PPO critic."""

import math

import torch
from torch import nn
from tensordict import TensorDict
from .tabletop_vision_ppo import VisionPPO
from rsl_rl.modules import ActorCritic

from .tabletop_policy_noise import configure_surface_noise, install_noise_projection


POINTS = 128
HISTORY = 4
PROPRIO_DIM = 62
FRAME_DIM = 2 * POINTS * 3 + POINTS + PROPRIO_DIM + 3
POLICY_DIM = HISTORY * FRAME_DIM
OBSERVATION_VERSION = 'dh116-vision-sensor-history-v1'


class SensorHistory:
    """Keep four sensor frames, oldest first, clearing selected rows on reset.

    Coordinates are metres in the actual left/right wrist link frames.
    q and qd are measured active encoder radians and radians/second.
    wrist_poses are measured/FK xyz+wxyz, after removing calibrated env origins.
    previous_action is the last issued normalized 24D actuator command.
    No simulator or task object is accepted by this interface.
    """

    def __init__(self, num_envs, device='cpu'):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError('num_envs must be a positive integer')
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.frames = torch.zeros(num_envs, HISTORY, FRAME_DIM, device=self.device)

    def _float(self, value, shape, name):
        value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if value.shape != shape:
            raise ValueError(f'{name} must have shape {shape}')
        return value

    def _mask(self, value, shape, name):
        value = torch.as_tensor(value, device=self.device)
        if value.dtype != torch.bool or value.shape != shape:
            raise ValueError(f'{name} must be boolean with shape {shape}')
        return value

    @torch.no_grad()
    def update(self, *, points_in_hands, point_valid, q, qd, wrist_poses,
               previous_action, dt, image_valid, object_valid, reset_mask=None):
        n = self.num_envs
        points = self._float(points_in_hands, (n, 2, POINTS, 3), 'points_in_hands')
        valid = self._mask(point_valid, (n, POINTS), 'point_valid')
        image_ok = self._mask(image_valid, (n,), 'image_valid')
        object_ok = self._mask(object_valid, (n,), 'object_valid')
        joints = self._float(q, (n, 12), 'q')
        speeds = self._float(qd, (n, 12), 'qd')
        wrists = self._float(wrist_poses, (n, 2, 7), 'wrist_poses')
        actions = self._float(previous_action, (n, 24), 'previous_action')
        delta = torch.as_tensor(dt, dtype=torch.float32, device=self.device)
        if delta.ndim == 0:
            delta = delta.expand(n)
        if delta.shape != (n,):
            raise ValueError('dt must be a scalar or [N] seconds')
        valid = valid & image_ok[:, None] & object_ok[:, None]
        points = torch.where(valid[:, None, :, None], points, 0.)
        proprio = torch.cat((joints, speeds, wrists.flatten(1), actions), dim=-1)
        if (not bool(torch.isfinite(points).all() & torch.isfinite(proprio).all()
                     & torch.isfinite(delta).all() & (delta > 0.).all())
                or not bool((wrists[..., 3:].abs().amax(-1) > 0.).all())
                or not bool((actions.abs() <= 1.).all())):
            raise ValueError('sensor values must be finite with valid wrist poses, dt and actions')
        flags = torch.stack((delta, image_ok.float(), object_ok.float()), dim=-1)
        frame = torch.cat((points.flatten(1), valid.float(), proprio, flags), dim=-1)
        if reset_mask is not None:
            reset = self._mask(reset_mask, (n,), 'reset_mask')
            self.frames[reset] = 0.
        self.frames[:, :-1] = self.frames[:, 1:].clone()
        self.frames[:, -1] = frame
        return self.frames.flatten(1).clone()


class VisionStudentActor(nn.Module):
    """Shared masked PointNet per hand, followed by a four-frame action MLP."""

    def __init__(self, action_size=24, *, wrist_quaternion_preprocessing='raw'):
        super().__init__()
        from .tabletop_vision_quaternion import validate_wrist_quaternion_preprocessing
        self.wrist_quaternion_preprocessing = validate_wrist_quaternion_preprocessing(wrist_quaternion_preprocessing)
        if action_size != 24:
            raise ValueError('vision student requires the existing 24D motion action')
        self.point_net = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.action_net = nn.Sequential(
            nn.Linear(HISTORY * (2 * 64 + PROPRIO_DIM + 3), 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, action_size), nn.Tanh(),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(2.))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.action_net[-2].weight, .01)

    def forward(self, observation):
        if observation.ndim != 2 or observation.shape[-1] != POLICY_DIM:
            raise ValueError(f'vision policy observations must have shape [N,{POLICY_DIM}]')
        if self.wrist_quaternion_preprocessing != 'raw':
            from .tabletop_vision_quaternion import canonicalize_wrist_quaternion_history
            observation = canonicalize_wrist_quaternion_history(observation)
        frames = observation.reshape(-1, HISTORY, FRAME_DIM)
        point_end = 2 * POINTS * 3
        mask_end = point_end + POINTS
        mask = frames[..., point_end:mask_end] > .5
        points = frames[..., :point_end].reshape(-1, HISTORY, 2, POINTS, 3)
        points = torch.where(mask[:, :, None, :, None], points, 0.)
        features = self.point_net(points)
        features = features.masked_fill(~mask[:, :, None, :, None], -torch.inf)
        pooled = features.amax(dim=-2)
        pooled = torch.where(mask.any(dim=-1)[:, :, None, None], pooled, 0.)
        frame_features = torch.cat((pooled.flatten(2), frames[..., mask_end:]), dim=-1)
        return self.action_net(frame_features.flatten(1))


def make_vision_algorithm(obs: TensorDict, steps, device='cpu', *,
                          microbatch_size=128, inference_batch_size=128):
    """Build asymmetric RSL PPO with no critic input path into the visual actor."""
    if (type(steps) is not int or steps < 1 or len(obs.batch_size) != 1
            or obs.batch_size[0] < 1 or obs['policy'].shape != (obs.batch_size[0], POLICY_DIM)
            or obs['critic'].shape != (obs.batch_size[0], 371)):
        raise ValueError('vision PPO requires [N,3844] policy and [N,371] critic observations')
    obs = obs.to(device)
    policy = ActorCritic(
        obs, {'policy': ['policy'], 'critic': ['critic']}, 24,
        actor_hidden_dims=[128, 128], critic_hidden_dims=[128, 128], activation='tanh',
        init_noise_std=.1, noise_std_type='log',
    )
    policy.actor = VisionStudentActor()
    policy = policy.to(device)
    for layer in policy.critic:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, math.sqrt(2.))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(policy.critic[-1].weight, 1.)
    algorithm = VisionPPO(
        policy, microbatch_size=microbatch_size, inference_batch_size=inference_batch_size, num_learning_epochs=5, num_mini_batches=min(8, len(obs)),
        learning_rate=3e-4, gamma=.995, lam=.95, clip_param=.2, entropy_coef=.005,
        value_loss_coef=.5, max_grad_norm=.5, schedule='adaptive', desired_kl=.02,
        device=str(device),
    )
    algorithm.init_storage('rl', len(obs), steps, obs, [24])
    configure_surface_noise(algorithm, initial_std=.1, min_std=.001, max_std=.3)
    install_noise_projection(algorithm)
    return algorithm
