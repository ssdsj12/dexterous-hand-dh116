"""Stereo-only actor and independent privileged critic, sharing audited PPO math.

PPO imports are lazy so deployment actor loading has no training dependency.
"""
from copy import deepcopy
from dataclasses import fields

import torch
from torch import nn

from .tabletop_stereo_observation import StereoHistoryBatch
from .tabletop_stereo_state_estimator import StereoStateEstimator, StereoStudentActor
from .tabletop_stereo_contract import action_sha256, require_sha

VERSION = 'dh116-stereo-asymmetric-student-ppo-v1'
INPUT_KEYS = tuple(field.name for field in fields(StereoHistoryBatch))


def normalize_network_config(config=None):
    value = deepcopy(config or {})
    if not isinstance(value, dict):
        raise ValueError('network config dictionary required')
    for name, default in dict(embed_dim=128, history=4, num_heads=4, num_layers=2, dropout=0.).items():
        value.setdefault(name, default)
    value.setdefault('image_processing_batch_size', None)
    value.setdefault('actor_input_mode','latent')
    value.setdefault('point_coordinate_mode','world')
    value.setdefault('point_source_encoding',False)
    if type(value['point_source_encoding']) is not bool:raise ValueError('point_source_encoding must be bool')
    value.setdefault('task_state_config',{})
    if value['actor_input_mode'] not in ('latent','estimated_task_state','hybrid'):
        raise ValueError('invalid Actor input mode')
    if value['point_coordinate_mode'] not in ('world','palms') or not isinstance(value['task_state_config'],dict):
        raise ValueError('explicit point coordinate mode and task-state config required')
    batch_size = value['image_processing_batch_size']
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError('image_processing_batch_size must be a positive integer or None')
    for name in ('embed_dim', 'history', 'num_heads', 'num_layers'):
        if type(value[name]) is not int or value[name] < 1:
            raise ValueError('positive network dimension required: ' + name)
    if value['embed_dim'] % value['num_heads'] or value['dropout'] != 0.:
        raise ValueError('divisible attention dimensions and dropout=0 required for PPO likelihood consistency')
    if value.get('action_size', 24) != 24 or value.get('proprio', 62) != 62 or value.get('points', 128) != 128:
        raise ValueError('independent stereo input requires 24 actions,62 proprio,128 points')
    if value.get('action_contract_sha256', action_sha256()) != action_sha256():
        raise ValueError('canonical action SHA mismatch')
    if 'image_resolution_wh' in value:
        resolution = value['image_resolution_wh']
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2 or any(type(v) is not int or v < 16 for v in resolution):
            raise ValueError('RGB resolution must be integer [width,height], each >=16')
    if 'backbone' in value:
        backbone = value['backbone']
        expected = dict(type='GN_ResNet18', block='BasicBlock', stages=[2,2,2,2],
            channels=[64,128,256,512], normalization='GroupNorm', weights='scratch',
            pretrained=False, shared_left_right=True)
        if not isinstance(backbone, dict) or any(key not in expected or item != expected[key] for key, item in backbone.items()):
            raise ValueError('only the actual shared GroupNorm ResNet18 backbone is supported')
    return value


def make_stereo_actor(network_config=None):
    config = normalize_network_config(network_config)
    estimator = StereoStateEstimator(**{name: config[name] for name in
        ('embed_dim', 'history', 'num_heads', 'num_layers', 'image_processing_batch_size','point_coordinate_mode','point_source_encoding')})
    if 'image_resolution_wh' in config:
        width, height = config['image_resolution_wh']
        def check_resolution(module, arguments):
            if tuple(arguments[0].left_rgb.shape[-2:]) != (height, width):
                raise ValueError('RGB history resolution differs from network/calibration contract')
        estimator.register_forward_pre_hook(check_resolution)
    return StereoStudentActor(estimator, action_contract_sha256=action_sha256(),
        input_mode=config['actor_input_mode'],task_state_config=config['task_state_config'])


def history_to_tree(history, *, owned=True):
    """Own/neutralize legacy arrays, or expose validated input for pool capture.

    ``owned=False`` avoids cloning a full visual history; the indexed storage
    capture hook immediately takes ownership of each new exact sensor frame.
    """
    if isinstance(history, dict):
        history = StereoHistoryBatch.from_mapping(history)
    if type(history) is not StereoHistoryBatch:
        raise TypeError('only exact StereoHistoryBatch sensor input is accepted')
    history.validate()
    if not owned: return history.as_dict()
    result = {key: value.detach().clone() for key, value in history.as_dict().items()}
    active = result['history_valid']
    depth = result['depth_valid'] & active[:, :, None, None]
    point = result['point_valid'] & active[:, :, None]
    for key in ('depth_m', 'depth_confidence'):
        result[key] = torch.where(depth, result[key], 0.)
    result['points_world'] = torch.where(point[:, :, :, None], result['points_world'], 0.)
    for hand in ('left','right'):
        mask=result['point_valid_'+hand] & point
        result['points_'+hand+'_palm']=torch.where(mask[...,None],result['points_'+hand+'_palm'],0.)
    result['point_confidence']=torch.where(point,result['point_confidence'],0.)
    return result


def _history(value):
    if type(value) is StereoHistoryBatch:
        return value.validate()
    if isinstance(value, dict):
        return StereoHistoryBatch.from_mapping(value)
    raise TypeError('StereoHistoryBatch or exact sensor dictionary required; privileged input forbidden')


class StereoAsymmetricPolicy(nn.Module):
    def __init__(self, network_config=None):
        super().__init__()
        self.network_config = normalize_network_config(network_config)
        self.actor = make_stereo_actor(self.network_config)
        self.critic = nn.Sequential(nn.Linear(371, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 1))

    @property
    def log_std(self):
        return self.actor.log_std

    def latent_mean(self, actor_tree):
        return self.actor.latent_mean(_history(actor_tree))

    def deterministic(self, history_or_tree):
        return self.actor.deterministic(_history(history_or_tree))

    def value(self, critic_obs):
        if (not isinstance(critic_obs, torch.Tensor) or critic_obs.ndim != 2 or critic_obs.shape[1] != 371
                or not critic_obs.is_floating_point() or not bool(torch.isfinite(critic_obs).all())):
            raise ValueError('independent finite privileged critic [N,371] required')
        return self.critic(critic_obs).squeeze(-1)


def make_student_trainer(config, num_envs, device='cpu', rollout_steps=None):
    from .tabletop_privileged_ppo import PPOConfig, PPOTrainer
    if not isinstance(config, dict) or not isinstance(config.get('network', {}), dict):
        raise ValueError('explicit independent Student configuration required')
    if config.get('critic_observation_size', 371) != 371 or config.get('action_size', 24) != 24:
        raise ValueError('371D critic and canonical24 actions required')
    if config.get('actor_input', 'StereoHistoryBatch_only') != 'StereoHistoryBatch_only':
        raise ValueError('Student actor input must be stereo sensor history only')
    model = StereoAsymmetricPolicy(config.get('network'))
    steps = config.get('rollout_steps', 32) if rollout_steps is None else rollout_steps
    storage_mode=config.get('rollout_storage','indexed_frames')
    if storage_mode not in ('indexed_frames','legacy'):
        raise ValueError('rollout_storage must be indexed_frames or legacy')
    max_visual_batch=config.get('max_visual_batch',1)
    if type(max_visual_batch) is not int or max_visual_batch<1:
        raise ValueError('max_visual_batch must be a positive integer')
    actor_storage=None
    if storage_mode=='indexed_frames':
        from .tabletop_stereo_rollout_storage import IndexedStereoRolloutStorage
        actor_storage=IndexedStereoRolloutStorage(max_visual_batch=max_visual_batch)
    trainer = PPOTrainer(model, PPOConfig(**config.get('ppo', {})), rollout_steps=steps, num_envs=num_envs,
        device=device,actor_storage=actor_storage,
        max_visual_batch=max_visual_batch if actor_storage is not None else None)
    trainer.rollout_storage_mode=storage_mode
    trainer.student_config = deepcopy(config)
    trainer.auxiliary_bc = None
    # Config opt-in alone cannot attach an unqualified dataset or alter storage.
    if config.get('auxiliary_bc', {}).get('enabled', False):
        raise ValueError('auxiliary BC requires explicit qualified independent attachment')
    return trainer


class _IndependentAuxiliaryBC:
    def __init__(self, trainer, bc, metadata):
        self.trainer, self.bc, self.metadata = trainer, bc, deepcopy(metadata)

    def step(self, *args, **kwargs):
        if not self.trainer.clean_boundary:
            raise RuntimeError('auxiliary BC requires clean PPO boundary with no pending rollout')
        result = self.bc.step(*args, **kwargs)
        self.trainer.policy_version += 1
        self.trainer._new_rollout()
        return result


def attach_auxiliary_bc(trainer, bc_trainer, *, metadata):
    if not trainer.clean_boundary:
        raise RuntimeError('auxiliary BC cannot attach to dirty pending/nonempty PPO boundary')
    if (not isinstance(metadata, dict) or metadata.get('qualified') is not True
            or metadata.get('source_scope') != 'native' or metadata.get('schema_only') is not False
            or metadata.get('ppo_storage_insertion') is not False):
        raise ValueError('qualified independent BC dataset metadata required')
    require_sha(metadata.get('dataset_sha256'), 'qualified BC dataset SHA')
    qualification = metadata.get('qualification_evidence')
    if not isinstance(qualification, (str, dict)):
        raise ValueError('raw native qualification evidence required for auxiliary BC')
    from .tabletop_stereo_promotion import verify_native_episode
    verify_native_episode(qualification, expected_identity=metadata.get('model_identity'))
    optimizer = getattr(bc_trainer, 'optimizer', None)
    if optimizer is None or optimizer is trainer.optimizer or not callable(getattr(bc_trainer, 'step', None)):
        raise ValueError('independent BC optimizer and step interface required')
    critic_ids = set(map(id, trainer.model.critic.parameters()))
    if any(id(parameter) in critic_ids for group in optimizer.param_groups for parameter in group['params']):
        raise ValueError('auxiliary BC optimizer cannot update privileged critic')
    trainer.auxiliary_bc = _IndependentAuxiliaryBC(trainer, bc_trainer, metadata)
    return trainer.auxiliary_bc
