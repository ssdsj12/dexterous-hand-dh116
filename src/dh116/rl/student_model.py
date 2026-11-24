"""Strict sensor history, stereo encoder/temporal estimator/state adapter, three actor modes, independent critic and visual rollout storage."""
import math as stere_encod__math
from dataclasses import dataclass
from dataclasses import fields as stere_rollo_stora__fields
import math as stere_state_estim__math
from copy import deepcopy
import copy
import torch as stere_tempo__torch
import torch as stere_encod__torch
from dataclasses import fields as stere_obser__fields
import hashlib as stere_rollo_stora__hashlib
import torch as stere_state_estim__torch
from dataclasses import fields as stere_stude_ppo__fields
import hashlib as stere_task_state_adapt__hashlib
from torch import nn as stere_tempo__nn
from torch import nn as stere_encod__nn
from dataclasses import field
import itertools
from torch import nn as stere_state_estim__nn
import torch as stere_stude_ppo__torch
import json

def position_encoding(position, width):
    """Continuous explicit sinusoidal coordinates; same device/dtype as input."""
    frequency = stere_encod__torch.exp(stere_encod__torch.arange((width + 1) // 2, device=position.device, dtype=position.dtype) * (-stere_encod__math.log(10000.0) / max(1, (width + 1) // 2)))
    phase = position[..., None] * frequency
    return stere_encod__torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)[..., :width]
import torch as stere_obser__torch
import torch as stere_rollo_stora__torch
from torch.nn import functional as F
from torch import nn as stere_stude_ppo__nn
import math as stere_task_state_adapt__math

def _norm(channels):
    return stere_encod__nn.GroupNorm(min(32, channels // 2), channels)
SENSOR_FIELDS = ('left_rgb', 'right_rgb', 'depth_m', 'depth_valid', 'depth_confidence', 'points_world', 'point_valid', 'proprio', 'dt', 'image_age', 'image_valid', 'points_left_palm', 'points_right_palm', 'point_valid_left', 'point_valid_right', 'point_confidence', 'point_provenance')
from dh116.env.contracts import EagerRuntimePolicy as stere_stude_ppo__EagerRuntimePolicy
from pathlib import Path

class PalmPointEncoder(stere_encod__nn.Module):
    """Shared observed-point weights, separate hand IDs, no world-origin input."""

    def __init__(self, embed_dim=128, point_source_encoding=False):
        super().__init__()
        self.project = stere_encod__nn.Linear(3, embed_dim)
        self.hand_embedding = stere_encod__nn.Parameter(stere_encod__torch.empty(2, embed_dim))
        stere_encod__nn.init.normal_(self.hand_embedding, std=0.02)
        self.point_source_encoding = point_source_encoding
        if point_source_encoding:
            from dh116.physics.wire import WireSourceEncoder
            self.source_encoder = WireSourceEncoder(embed_dim)

    def forward(self, left, right, left_valid, right_valid, confidence, provenance=None):
        if left.shape != right.shape or left.ndim != 3 or left.shape[1:] != (128, 3) or (left_valid.shape != left.shape[:2]) or (right_valid.shape != left_valid.shape) or (left_valid.dtype != stere_encod__torch.bool) or (right_valid.dtype != stere_encod__torch.bool) or (confidence.shape != left_valid.shape):
            raise ValueError('two128-point palm views, boolean masks and confidence required')
        tokens = []
        masks = []
        for (index, (points, valid)) in enumerate(((left, left_valid), (right, right_valid))):
            valid = valid & stere_encod__torch.isfinite(points).all(-1) & stere_encod__torch.isfinite(confidence) & (confidence > 0)
            p = stere_encod__torch.where(valid[..., None], points, 0.0)
            if self.point_source_encoding:
                if provenance is None:
                    raise ValueError('explicit point provenance required by source-aware Actor')
                encoded = self.source_encoder(p, valid, provenance)
                valid = encoded['centerline_valid'] | encoded['observed_surface_valid']
                t = encoded['tokens'] + self.hand_embedding[index]
            else:
                t = self.project(p) + self.hand_embedding[index]
            t = t + position_encoding(stere_encod__torch.linspace(0, 1, 128, device=p.device), t.shape[-1])[None]
            tokens.append(stere_encod__torch.where(valid[..., None], t, 0.0))
            masks.append(valid)
        return (stere_encod__torch.cat(tokens, 1), stere_encod__torch.cat(masks, 1))

def _populate_palms(batch):
    """Compatibility inputs derive local geometry only from the same measured FK.

    Default confidence is an availability indicator, not a calibrated matching
    probability. Native/inference pipelines supply actual RGB-derived scores.
    """
    if any((getattr(batch, key) is None for key in ('points_left_palm', 'points_right_palm', 'point_valid_left', 'point_valid_right'))):
        from dh116.physics.vision import world_to_palms
        prefix = batch.points_world.shape[:-2]
        pose = batch.proprio[..., 24:38].reshape(-1, 2, 7).clone()
        history = getattr(batch, 'history_valid', None)
        if history is not None:
            inactive = ~history.flatten()
            pose[inactive] = 0.0
            pose[inactive, :, 3] = 1.0
        local = world_to_palms(batch.points_world.reshape(-1, 128, 3), pose).reshape(*prefix, 2, 128, 3)
        for (index, hand) in enumerate(('left', 'right')):
            mask = batch.point_valid & stere_obser__torch.isfinite(local[..., index, :, :]).all(-1)
            if getattr(batch, 'point_valid_' + hand) is None:
                setattr(batch, 'point_valid_' + hand, mask)
            if getattr(batch, 'points_' + hand + '_palm') is None:
                setattr(batch, 'points_' + hand + '_palm', stere_obser__torch.where(mask[..., None], local[..., index, :, :], float('nan')))
    if batch.point_confidence is None:
        batch.point_confidence = batch.point_valid.float()
    if batch.point_provenance is None:
        batch.point_provenance = stere_obser__torch.zeros_like(batch.point_valid, dtype=stere_obser__torch.uint8)
from dh116.env.contracts import validate_runtime_execution
import torch as stere_task_state_adapt__torch

class BasicBlock(stere_encod__nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = stere_encod__nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.norm1 = _norm(channels)
        self.conv2 = stere_encod__nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.norm2 = _norm(channels)
        self.downsample = stere_encod__nn.Identity() if in_channels == channels and stride == 1 else stere_encod__nn.Sequential(stere_encod__nn.Conv2d(in_channels, channels, 1, stride, bias=False), _norm(channels))

    def forward(self, value):
        residual = self.downsample(value)
        value = stere_encod__torch.relu(self.norm1(self.conv1(value)))
        return stere_encod__torch.relu(self.norm2(self.conv2(value)) + residual)

def _validate(values, *, history_valid=None):
    for name in SENSOR_FIELDS:
        if not isinstance(values[name], stere_obser__torch.Tensor):
            raise TypeError(f'{name} must be a tensor')
    left = values['left_rgb']
    prefix = left.shape[:2] if history_valid is not None else left.shape[:1]
    ndim = 5 if history_valid is not None else 4
    if left.ndim != ndim or left.shape[-3] != 3 or min(left.shape[-2:]) < 16 or (not prefix[0]):
        raise ValueError('RGB shape must be [B,3,H,W] or [B,T,3,H,W], H/W>=16')
    (h, w) = left.shape[-2:]
    shapes = dict(left_rgb=(*prefix, 3, h, w), right_rgb=(*prefix, 3, h, w), depth_m=(*prefix, h, w), depth_valid=(*prefix, h, w), depth_confidence=(*prefix, h, w), points_world=(*prefix, 128, 3), point_valid=(*prefix, 128), proprio=(*prefix, 62), points_left_palm=(*prefix, 128, 3), points_right_palm=(*prefix, 128, 3), point_valid_left=(*prefix, 128), point_valid_right=(*prefix, 128), point_confidence=(*prefix, 128), point_provenance=(*prefix, 128), dt=prefix, image_age=prefix, image_valid=(*prefix, 2))
    for (name, shape) in shapes.items():
        value = values[name]
        dtype = stere_obser__torch.bool if name in ('depth_valid', 'point_valid', 'image_valid', 'point_valid_left', 'point_valid_right') else stere_obser__torch.uint8 if name == 'point_provenance' else stere_obser__torch.float32
        if tuple(value.shape) != tuple(shape) or value.dtype != dtype or value.device != left.device:
            raise ValueError(f'{name} must have shape {shape}, dtype {dtype}, device {left.device}')
    if bool((values['point_provenance'] > 2).any()):
        raise ValueError('unknown point provenance code')
    active = stere_obser__torch.ones(prefix, dtype=stere_obser__torch.bool, device=left.device)
    if history_valid is not None:
        if history_valid.shape != prefix or history_valid.dtype != stere_obser__torch.bool or history_valid.device != left.device:
            raise ValueError('history_valid must be boolean [B,T] on the observation device')
        active = history_valid
    for name in ('left_rgb', 'right_rgb', 'proprio', 'dt', 'image_age'):
        if not bool(stere_obser__torch.isfinite(values[name]).all()):
            raise ValueError(f'{name} must be finite')
    for name in ('left_rgb', 'right_rgb'):
        if bool(((values[name] < 0) | (values[name] > 1)).any()):
            raise ValueError('RGB must be float32 in [0,1]')
    if bool(((values['dt'] <= 0) & active).any()) or bool((values['image_age'] < 0).any()):
        raise ValueError('active dt must be positive and image_age nonnegative')
    proprio = values['proprio']
    if bool(((proprio[..., 38:].abs() > 1) & active[..., None]).any()):
        raise ValueError('previous actually-issued action must be bounded normalized24')
    for lo in (27, 34):
        if bool(((proprio[..., lo:lo + 4].norm(dim=-1) <= 1e-08) & active).any()):
            raise ValueError('active measured palm wxyz quaternion must be nonzero')
    depth_valid = values['depth_valid'] & active[..., None, None]
    depth = values['depth_m']
    confidence = values['depth_confidence']
    if bool((depth_valid & (~stere_obser__torch.isfinite(depth) | (depth <= 0))).any()) or bool((depth_valid & (~stere_obser__torch.isfinite(confidence) | (confidence < 0) | (confidence > 1))).any()):
        raise ValueError('valid depth/confidence must be finite, depth>0 and confidence in [0,1]')
    point_valid = values['point_valid'] & active[..., None]
    if bool((point_valid & ~stere_obser__torch.isfinite(values['points_world']).all(-1)).any()):
        raise ValueError('valid world points must be finite')
    for hand in ('left', 'right'):
        valid = values['point_valid_' + hand] & point_valid
        if bool((valid & ~stere_obser__torch.isfinite(values['points_' + hand + '_palm']).all(-1)).any()):
            raise ValueError('valid palm-local points require finite observed world geometry')
    confidence = values['point_confidence']
    if bool((point_valid & (~stere_obser__torch.isfinite(confidence) | (confidence < 0) | (confidence > 1))).any()):
        raise ValueError('valid point confidence must be finite in [0,1]')
from torch import nn as stere_task_state_adapt__nn

class ResNet18(stere_encod__nn.Module):
    """Eight BasicBlocks, standard 64/128/256/512 widths, no pretrained weights."""

    def __init__(self):
        super().__init__()
        self.stem = stere_encod__nn.Sequential(stere_encod__nn.Conv2d(3, 64, 7, 2, 3, bias=False), _norm(64), stere_encod__nn.ReLU(), stere_encod__nn.MaxPool2d(3, 2, 1))
        self.layer1 = stere_encod__nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = stere_encod__nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = stere_encod__nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = stere_encod__nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))

    def forward(self, value):
        return self.layer4(self.layer3(self.layer2(self.layer1(self.stem(value)))))

class _TensorBatch:
    __slots__ = ()

    @classmethod
    def from_mapping(cls, value):
        expected = {f.name for f in stere_obser__fields(cls)}
        if isinstance(value, dict) and 'point_provenance' not in value:
            value = dict(value, point_provenance=None)
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError('unknown or missing sensor fields; privileged inputs are forbidden')
        return cls(**value)

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in stere_obser__fields(self)}

    def _map(self, operation):
        return type(self)(**{name: operation(value) for (name, value) in self.as_dict().items()})

    def to(self, device):
        return self._map(lambda value: value.to(device))

    def clone(self):
        return self._map(lambda value: value.clone())

    def detach(self):
        return self._map(lambda value: value.detach())

    def index_select(self, indices):
        indices = stere_obser__torch.as_tensor(indices, dtype=stere_obser__torch.long, device=self.left_rgb.device)
        if indices.ndim != 1:
            raise ValueError('batch indices must be one dimensional')
        return self._map(lambda value: value.index_select(0, indices))

    def __len__(self):
        return self.left_rgb.shape[0]

class StereoEncoder(stere_encod__nn.Module):

    def __init__(self, embed_dim=128, num_heads=4, image_processing_batch_size=None):
        super().__init__()
        if embed_dim < 8 or embed_dim % num_heads:
            raise ValueError('embedding must divide attention heads')
        if image_processing_batch_size is not None and (type(image_processing_batch_size) is not int or image_processing_batch_size < 1):
            raise ValueError('image processing batch size must be positive or None')
        self.embed_dim = embed_dim
        self.image_processing_batch_size = image_processing_batch_size
        self.backbone = ResNet18()
        self.project = stere_encod__nn.Conv2d(512, embed_dim, 1)
        self.view_embedding = stere_encod__nn.Parameter(stere_encod__torch.empty(2, embed_dim))
        stere_encod__nn.init.normal_(self.view_embedding, std=0.02)
        self.cross_attention = stere_encod__nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True)
        self.norm = stere_encod__nn.LayerNorm(embed_dim)

    def _cross(self, query, key, query_valid, key_valid):
        mask = ~key_valid[:, None].expand(-1, key.shape[1]).clone()
        mask[~key_valid, 0] = False
        key = stere_encod__torch.where(key_valid[:, None, None], key, 0.0)
        context = self.cross_attention(query, key, key, key_padding_mask=mask, need_weights=False)[0]
        context = stere_encod__torch.where(key_valid[:, None, None], context, 0.0)
        return stere_encod__torch.where(query_valid[:, None, None], self.norm(query + context), 0.0)

    def forward(self, left_rgb, right_rgb, image_valid):
        if left_rgb.ndim != 4 or right_rgb.shape != left_rgb.shape or left_rgb.shape[1] != 3 or (image_valid.shape != (len(left_rgb), 2)) or (image_valid.dtype != stere_encod__torch.bool):
            raise ValueError('matching RGB [B,3,H,W] and boolean view validity [B,2] required')
        if left_rgb.dtype != stere_encod__torch.float32 or right_rgb.dtype != stere_encod__torch.float32 or right_rgb.device != left_rgb.device or (image_valid.device != left_rgb.device) or (min(left_rgb.shape[-2:]) < 16):
            raise ValueError('RGB float32 and view masks must share a device; H/W>=16')
        if not bool(stere_encod__torch.isfinite(left_rgb).all() & stere_encod__torch.isfinite(right_rgb).all()) or bool(((left_rgb < 0) | (left_rgb > 1)).any() | ((right_rgb < 0) | (right_rgb > 1)).any()):
            raise ValueError('RGB must be finite in [0,1]')
        left = stere_encod__torch.where(image_valid[:, 0, None, None, None], left_rgb, 0.0)
        right = stere_encod__torch.where(image_valid[:, 1, None, None, None], right_rgb, 0.0)
        images = stere_encod__torch.cat((left, right), 0)
        batch = self.image_processing_batch_size
        if batch is None:
            features = self.project(self.backbone(images))
        else:
            chunks = []
            for start in range(0, len(images), batch):
                image = images[start:start + batch]
                if self.training and stere_encod__torch.is_grad_enabled():
                    from torch.utils.checkpoint import checkpoint
                    output = checkpoint(lambda value: self.project(self.backbone(value)), image, use_reentrant=False)
                else:
                    output = self.project(self.backbone(image))
                chunks.append(output)
            features = stere_encod__torch.cat(chunks, 0)
        n = left.shape[0]
        (h, w) = features.shape[-2:]
        (yy, xx) = stere_encod__torch.meshgrid(stere_encod__torch.linspace(-1, 1, h, device=left.device), stere_encod__torch.linspace(-1, 1, w, device=left.device), indexing='ij')
        position = position_encoding(xx.flatten(), self.embed_dim // 2)
        position = stere_encod__torch.cat((position, position_encoding(yy.flatten(), self.embed_dim - self.embed_dim // 2)), -1)
        tokens = features.flatten(2).transpose(1, 2) + position[None]
        (l, r) = (tokens[:n] + self.view_embedding[0], tokens[n:] + self.view_embedding[1])
        lout = self._cross(l, r, image_valid[:, 0], image_valid[:, 1])
        rout = self._cross(r, l, image_valid[:, 1], image_valid[:, 0])
        valid = image_valid[:, :, None].expand(n, 2, h * w).flatten(1)
        return (stere_encod__torch.cat((lout, rout), 1), valid)

@dataclass(slots=True)
class SensorBatch(_TensorBatch):
    left_rgb: stere_obser__torch.Tensor
    right_rgb: stere_obser__torch.Tensor
    depth_m: stere_obser__torch.Tensor
    depth_valid: stere_obser__torch.Tensor
    depth_confidence: stere_obser__torch.Tensor
    points_world: stere_obser__torch.Tensor
    point_valid: stere_obser__torch.Tensor
    proprio: stere_obser__torch.Tensor
    dt: stere_obser__torch.Tensor
    image_age: stere_obser__torch.Tensor
    image_valid: stere_obser__torch.Tensor
    points_left_palm: stere_obser__torch.Tensor = field(default=None, kw_only=True)
    points_right_palm: stere_obser__torch.Tensor = field(default=None, kw_only=True)
    point_valid_left: stere_obser__torch.Tensor = field(default=None, kw_only=True)
    point_valid_right: stere_obser__torch.Tensor = field(default=None, kw_only=True)
    point_confidence: stere_obser__torch.Tensor = field(default=None, kw_only=True)
    point_provenance: stere_obser__torch.Tensor = field(default=None, kw_only=True)

    def __post_init__(self):
        try:
            _populate_palms(self)
        except (RuntimeError, IndexError, AttributeError) as error:
            raise ValueError('invalid sensor shape while deriving palm coordinates') from error
        self.validate()

    def validate(self):
        _validate(self.as_dict())
        return self

@dataclass(slots=True)
class StereoHistoryBatch(SensorBatch):
    history_valid: stere_obser__torch.Tensor

    def validate(self):
        _validate(self.as_dict(), history_valid=self.history_valid)
        return self

    @classmethod
    def cat(cls, batches):
        batches = list(batches)
        if not batches or any((type(batch) is not cls for batch in batches)):
            raise TypeError('nonempty StereoHistoryBatch list required')
        return cls(**{f.name: stere_obser__torch.cat([getattr(b, f.name) for b in batches], 0) for f in stere_obser__fields(cls)})
stere_state_estim__StereoEncoder = StereoEncoder
stere_state_estim__PalmPointEncoder = PalmPointEncoder
stere_state_estim__position_encoding = position_encoding
stere_tempo__position_encoding = position_encoding

class StereoHistory:

    def __init__(self, num_envs, history=4, device='cpu'):
        if type(num_envs) is not int or num_envs < 1 or type(history) is not int or (history < 1):
            raise ValueError('positive integer batch and history required')
        (self.num_envs, self.history, self.device) = (num_envs, history, stere_obser__torch.device(device))
        self._values = None
        self._valid = stere_obser__torch.zeros(num_envs, history, device=self.device, dtype=stere_obser__torch.bool)

    def reset(self, mask):
        if not isinstance(mask, stere_obser__torch.Tensor) or mask.shape != (self.num_envs,) or mask.dtype != stere_obser__torch.bool:
            raise ValueError('reset mask must be boolean [B]')
        mask = mask.to(self.device)
        self._valid[mask] = False
        if self._values is not None:
            for value in self._values.values():
                value[mask] = 0

    @stere_obser__torch.no_grad()
    def update(self, sensor, reset_mask=None):
        if type(sensor) is not SensorBatch:
            raise TypeError('only SensorBatch is accepted; no privileged/task fields')
        sensor.validate()
        if len(sensor) != self.num_envs or sensor.left_rgb.device != self.device:
            raise ValueError('sensor batch/device differs from history')
        if self._values is None:
            self._values = {name: stere_obser__torch.zeros(self.num_envs, self.history, *value.shape[1:], dtype=value.dtype, device=self.device) for (name, value) in sensor.as_dict().items()}
        if reset_mask is not None:
            self.reset(reset_mask)
        for (name, value) in sensor.as_dict().items():
            buffer = self._values[name]
            if buffer.shape[2:] != value.shape[1:]:
                raise ValueError('sensor shape changed within history')
            buffer[:, :-1] = buffer[:, 1:].clone()
            buffer[:, -1] = value
        self._valid[:, :-1] = self._valid[:, 1:].clone()
        self._valid[:, -1] = True
        return StereoHistoryBatch(**{name: value.clone() for (name, value) in self._values.items()}, history_valid=self._valid.clone())

class StereoTemporal(stere_tempo__nn.Module):

    def __init__(self, embed_dim=128, history=4, num_heads=4, num_layers=2):
        super().__init__()
        if history < 1 or num_layers < 1 or embed_dim % num_heads:
            raise ValueError('invalid temporal dimensions')
        (self.embed_dim, self.history) = (embed_dim, history)
        layer = stere_tempo__nn.TransformerEncoderLayer(embed_dim, num_heads, 4 * embed_dim, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = stere_tempo__nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.time_project = stere_tempo__nn.Linear(2, embed_dim)
        self.norm = stere_tempo__nn.LayerNorm(embed_dim)

    def forward(self, frames, history_valid, dt=None, image_age=None):
        if frames.ndim != 3 or frames.shape[1:] != (self.history, self.embed_dim) or history_valid.shape != frames.shape[:2] or (history_valid.dtype != stere_tempo__torch.bool) or (frames.dtype != stere_tempo__torch.float32) or (history_valid.device != frames.device):
            raise ValueError('temporal frames [B,T,D] and boolean history mask required')
        if not bool(stere_tempo__torch.isfinite(frames).all()):
            raise ValueError('finite frame embeddings required')
        frames = stere_tempo__torch.where(history_valid[:, :, None], frames, 0.0)
        position = stere_tempo__position_encoding(stere_tempo__torch.arange(self.history, device=frames.device, dtype=frames.dtype), self.embed_dim)
        explicit_dt = dt is not None
        if dt is None:
            dt = stere_tempo__torch.zeros_like(history_valid, dtype=frames.dtype)
        if image_age is None:
            image_age = stere_tempo__torch.zeros_like(dt)
        if dt.shape != history_valid.shape or image_age.shape != dt.shape or dt.device != frames.device or (image_age.device != frames.device) or (dt.dtype != frames.dtype) or (image_age.dtype != frames.dtype):
            raise ValueError('time arrays must have shape [B,T]')
        if not bool(stere_tempo__torch.isfinite(dt).all() & stere_tempo__torch.isfinite(image_age).all()) or bool((image_age < 0).any()) or (explicit_dt and bool(((dt <= 0) & history_valid).any())):
            raise ValueError('finite positive active dt and nonnegative image age required')
        time = stere_tempo__torch.stack((stere_tempo__torch.where(history_valid, dt, 0.0).cumsum(1), stere_tempo__torch.where(history_valid, image_age, 0.0)), -1)
        frames = frames + position[None] + self.time_project(time)
        safe = history_valid.clone()
        empty = ~safe.any(1)
        safe[empty, 0] = True
        encoded = self.transformer(frames, src_key_padding_mask=~safe)
        index = (history_valid.long() * stere_tempo__torch.arange(1, self.history + 1, device=frames.device)).amax(1) - 1
        selected = encoded[stere_tempo__torch.arange(len(frames), device=frames.device), index.clamp_min(0)]
        return stere_tempo__torch.where(empty[:, None], stere_tempo__torch.zeros_like(selected), self.norm(selected))
stere_rollo_stora__StereoHistoryBatch = StereoHistoryBatch
stere_state_estim__StereoHistoryBatch = StereoHistoryBatch
stere_state_estim__StereoTemporal = StereoTemporal
stere_stude_ppo__StereoHistoryBatch = StereoHistoryBatch
stere_task_state_adapt__StereoHistoryBatch = StereoHistoryBatch
stere_rollo_stora__INPUT_KEYS = tuple((field.name for field in stere_rollo_stora__fields(stere_rollo_stora__StereoHistoryBatch)))
from dh116.env.contracts import action_sha256 as canonical_action_sha256
EXTRA_SHAPES = {'geometry': (128, 3), 'velocity': (128, 3), 'deformation': (128,), 'contact_logits': (2, 3), 'slip_logits': (2,), 'slip_velocity': (2,), 'force_proxy': (2, 3)}
_RGB = ('left_rgb', 'right_rgb')
HEAD_SHAPES = dict(privileged_state=(371,), geometry=(128, 3), velocity=(128, 3), deformation=(128,), contact_logits=(2, 3), force_proxy=(2, 3), slip_logits=(2,), slip_velocity=(2,))
_VISUAL_HEADS = {'geometry_head', 'velocity_head', 'contact_head', 'slip_head'}
_GENERATION_STRIDE = 1 << 32

class StereoStateEstimator(stere_state_estim__nn.Module):

    def __init__(self, embed_dim=128, history=4, num_heads=4, num_layers=2, image_processing_batch_size=None, point_coordinate_mode='world', task_state_metadata=False, point_source_encoding=False):
        super().__init__()
        if point_coordinate_mode not in ('world', 'palms'):
            raise ValueError('world or palms point coordinates required')
        self.point_coordinate_mode = point_coordinate_mode
        if point_source_encoding and point_coordinate_mode != 'palms':
            raise ValueError('source encoding requires palm-local point coordinates')
        self.point_source_encoding = bool(point_source_encoding)
        self.task_state_metadata = bool(task_state_metadata or point_coordinate_mode == 'palms')
        (self.embed_dim, self.history) = (embed_dim, history)
        self.encoder = stere_state_estim__StereoEncoder(embed_dim, num_heads, image_processing_batch_size)
        self.depth_project = stere_state_estim__nn.Linear(5, embed_dim)
        self.point_project = stere_state_estim__nn.Linear(3, embed_dim)
        self.proprio_project = stere_state_estim__nn.Sequential(stere_state_estim__nn.Linear(64, embed_dim), stere_state_estim__nn.GELU(), stere_state_estim__nn.Linear(embed_dim, embed_dim))
        self.frame_embedding = stere_state_estim__nn.Parameter(stere_state_estim__torch.empty(3, embed_dim))
        stere_state_estim__nn.init.normal_(self.frame_embedding, std=0.02)
        self.null_token = stere_state_estim__nn.Parameter(stere_state_estim__torch.zeros(1, 1, embed_dim))
        self.spatial_attention = stere_state_estim__nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True)
        self.gate = stere_state_estim__nn.Linear(2 * embed_dim, embed_dim)
        self.spatial_norm = stere_state_estim__nn.LayerNorm(embed_dim)
        self.temporal = stere_state_estim__StereoTemporal(embed_dim, history, num_heads, num_layers)
        self.heads = stere_state_estim__nn.ModuleDict({name: stere_state_estim__nn.Linear(embed_dim, stere_state_estim__math.prod(shape)) for (name, shape) in HEAD_SHAPES.items()})
        if point_coordinate_mode == 'palms':
            self.palm_point_encoder = stere_state_estim__PalmPointEncoder(embed_dim, point_source_encoding=point_source_encoding)

    def _spatial(self, h):
        (b, t) = h.history_valid.shape
        n = b * t
        d = self.embed_dim
        active = h.history_valid.flatten()
        views = (h.image_valid & h.history_valid[:, :, None]).flatten(0, 1)
        left = stere_state_estim__torch.where(h.history_valid[:, :, None, None, None], h.left_rgb, 0.0).flatten(0, 1)
        right = stere_state_estim__torch.where(h.history_valid[:, :, None, None, None], h.right_rgb, 0.0).flatten(0, 1)
        (stereo, stereo_valid) = self.encoder(left, right, views)
        pair = views.all(1)
        depth_valid = (h.depth_valid & h.history_valid[:, :, None, None]).flatten(0, 1) & pair[:, None, None]
        depth = stere_state_estim__torch.where(depth_valid, h.depth_m.flatten(0, 1), 0.0)
        confidence = stere_state_estim__torch.where(depth_valid, h.depth_confidence.flatten(0, 1), 0.0)
        weight = F.adaptive_avg_pool2d(depth_valid.float()[:, None], (3, 3))
        pooled = F.adaptive_avg_pool2d(depth[:, None], (3, 3)) / weight.clamp_min(1e-08)
        conf = F.adaptive_avg_pool2d(confidence[:, None], (3, 3)) / weight.clamp_min(1e-08)
        (yy, xx) = stere_state_estim__torch.meshgrid(stere_state_estim__torch.linspace(-1, 1, 3, device=left.device), stere_state_estim__torch.linspace(-1, 1, 3, device=left.device), indexing='ij')
        coordinates = stere_state_estim__torch.stack((xx.flatten(), yy.flatten()), -1)[None].expand(n, -1, -1)
        depth_input = stere_state_estim__torch.cat((pooled.flatten(2).transpose(1, 2), conf.flatten(2).transpose(1, 2), weight.flatten(2).transpose(1, 2), coordinates), -1)
        depth_tokens = self.depth_project(depth_input) + self.frame_embedding[0]
        depth_mask = weight.flatten(1) > 0
        point_mask = (h.point_valid & h.history_valid[:, :, None]).flatten(0, 1) & pair[:, None]
        points = stere_state_estim__torch.where(point_mask[:, :, None], h.points_world.flatten(0, 1), 0.0)
        point_tokens = self.point_project(points) + self.frame_embedding[1]
        point_tokens = point_tokens + stere_state_estim__position_encoding(stere_state_estim__torch.linspace(0, 1, 128, device=left.device), d)[None]
        if self.point_coordinate_mode == 'palms':
            (point_tokens, point_mask) = self.palm_point_encoder(h.points_left_palm.flatten(0, 1), h.points_right_palm.flatten(0, 1), (h.point_valid_left & h.point_valid & h.history_valid[:, :, None]).flatten(0, 1) & pair[:, None], (h.point_valid_right & h.point_valid & h.history_valid[:, :, None]).flatten(0, 1) & pair[:, None], h.point_confidence.flatten(0, 1), h.point_provenance.flatten(0, 1))
        proprio = stere_state_estim__torch.where(active[:, None], h.proprio.flatten(0, 1), 0.0)
        if self.point_coordinate_mode == 'palms':
            proprio = proprio.clone()
            origin = 0.5 * (proprio[:, 24:27] + proprio[:, 31:34])
            proprio[:, 24:27] -= origin
            proprio[:, 31:34] -= origin
        timing = stere_state_estim__torch.stack((stere_state_estim__torch.where(active, h.dt.flatten(), 0.0), stere_state_estim__torch.where(active, h.image_age.flatten(), 0.0)), -1)
        query = self.proprio_project(stere_state_estim__torch.cat((proprio, timing), -1)) + self.frame_embedding[2]
        tokens = stere_state_estim__torch.cat((stereo, depth_tokens, point_tokens, self.null_token.expand(n, -1, -1)), 1)
        valid = stere_state_estim__torch.cat((stereo_valid, depth_mask, point_mask, stere_state_estim__torch.ones(n, 1, device=left.device, dtype=stere_state_estim__torch.bool)), 1)
        tokens = stere_state_estim__torch.where(valid[:, :, None], tokens, 0.0)
        visual = self.spatial_attention(query[:, None], tokens, tokens, key_padding_mask=~valid, need_weights=False)[0][:, 0]
        gated = stere_state_estim__torch.sigmoid(self.gate(stere_state_estim__torch.cat((query, visual), -1))) * visual
        return stere_state_estim__torch.where(active[:, None], self.spatial_norm(query + gated), 0.0).reshape(b, t, d)

    def forward(self, history):
        if type(history) is not stere_state_estim__StereoHistoryBatch:
            raise TypeError('Actor estimator accepts only StereoHistoryBatch; privileged mappings forbidden')
        history.validate()
        if history.history_valid.shape[1] != self.history:
            raise ValueError('history length mismatch')
        frames = self._spatial(history)
        latent = self.temporal(frames, history.history_valid, history.dt, history.image_age)
        result = {'latent': latent}
        result.update({name: head(latent).reshape(len(history), *HEAD_SHAPES[name]) for (name, head) in self.heads.items()})
        if self.task_state_metadata:
            valid = history.history_valid[:, -1] & history.image_valid[:, -1].all(-1) & history.point_valid[:, -1].any(-1)
            result['state_valid'] = valid[:, None].expand(-1, 371)
        if self.point_coordinate_mode == 'palms':
            mask = history.point_valid[:, -1]
            p = stere_state_estim__torch.where(mask[..., None], history.points_world[:, -1], 0.0)
            result['world_auxiliary'] = (self.point_project(p) * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return result
_KNOWN_KEYS = {'joint_lower', 'joint_upper', 'reset_joint_reference', 'grip_drive_stiffness', 'wrist_reset_xyz'}
_GENERATIONS = itertools.count(1)
from dh116.env.contracts import EagerRuntimePolicy as stere_state_estim__EagerRuntimePolicy

def field_history_frames(row):
    """Number of valid frames required by a canonical field's runtime gate."""
    return 2 if row['estimator_head'] in ('velocity_head', 'slip_head') or row['block'].startswith('interval_') else 1

def _identity(tensor):
    value = tensor.detach().contiguous()
    return (str(value.dtype), tuple(value.shape), stere_rollo_stora__hashlib.sha256(value.numpy()).digest())

class StereoStudentActor(stere_state_estim__EagerRuntimePolicy, stere_state_estim__nn.Module):
    """Unbounded mu24 and latent std; use existing saved-latent Gaussian math."""

    def __init__(self, estimator=None, *, embed_dim=128, action_contract_sha256=None, image_processing_batch_size=None, input_mode='latent', task_state_config=None):
        super().__init__()
        if input_mode not in ('latent', 'estimated_task_state', 'hybrid'):
            raise ValueError('invalid Actor input mode')
        self.input_mode = input_mode
        from copy import deepcopy
        self.task_state_config = deepcopy(task_state_config or {})
        expected = canonical_action_sha256()
        if action_contract_sha256 is not None and action_contract_sha256 != expected:
            raise ValueError('canonical action contract SHA256 mismatch')
        self.action_contract_sha256 = expected
        self.estimator = StereoStateEstimator(embed_dim=embed_dim, image_processing_batch_size=image_processing_batch_size) if estimator is None else estimator
        if not isinstance(self.estimator, StereoStateEstimator):
            raise TypeError('StereoStateEstimator required')
        size = self.estimator.embed_dim
        if input_mode != 'latent':
            _stere_state_estim_import__TaskStateAdapter = TaskStateAdapter
            self.task_state_adapter = _stere_state_estim_import__TaskStateAdapter(**task_state_config or {})
            self.estimator.task_state_metadata = True
            size = self.task_state_adapter.feature_size + (size if input_mode == 'hybrid' else 0)
        self.action_net = stere_state_estim__nn.Sequential(stere_state_estim__nn.Linear(size, 128), stere_state_estim__nn.Tanh(), stere_state_estim__nn.Linear(128, 24))
        stere_state_estim__nn.init.orthogonal_(self.action_net[-1].weight, 0.01)
        stere_state_estim__nn.init.zeros_(self.action_net[-1].bias)
        self.log_std = stere_state_estim__nn.Parameter(stere_state_estim__torch.full((24,), stere_state_estim__math.log(0.1)))

    def features_from_estimates(self, estimates, history):
        features = estimates['latent']
        if self.input_mode != 'latent':
            selected = {key: value for (key, value) in estimates.items() if key != 'world_auxiliary'}
            task = self.task_state_adapter(selected, history)
            features = task['actor_features']
            if self.input_mode == 'hybrid':
                features = stere_state_estim__torch.cat((estimates['latent'], features), -1)
        return features

    def latent_mean(self, history):
        estimates = self.estimator(history)
        return self.action_net(self.features_from_estimates(estimates, history))

    def forward(self, history):
        return self.latent_mean(history)

    def deterministic(self, history):
        return self.latent_mean(history).tanh()

class TaskStateAdapter(stere_task_state_adapt__nn.Module):
    """Select declared fields in schema index order and preserve autograd.

    ``allowlist`` contains canonical indices, semantic names, or block names.
    ``known_config`` contains explicitly declared raw SI configuration values;
    joint limits are training normalization constants, not hardware approval.
    ``source_qualification`` maps schema validity keys to booleans or boolean
    [B]/[B,371] eligibility masks. Missing keys fail closed. Optional estimator
    sidebands ``state_valid`` and ``state_uncertainty`` are [B,371]; uncertainty
    is nonnegative in each normalized field's units. Absent uncertainty uses 1
    as an explicit unknown sentinel, never a zero error claim.

    Extra visual heads are opt-in and append to the selected 371D fields.
    Their units/coordinates are explicitly unbound until ``extra_head_metadata``
    declares them; no extra node is relabelled as a canonical 371D quantity.
    """

    def __init__(self, *, allowlist=None, known_config=None, source_qualification=None, extra_heads=(), extra_head_metadata=None):
        super().__init__()
        path = Path(__file__).resolve().parents[3] / 'configs/contracts/privileged_state.json'
        raw = path.read_bytes()
        schema = json.loads(raw.decode('utf-8-sig'))
        self.state_contract_sha256 = stere_task_state_adapt__hashlib.sha256(raw).hexdigest()
        rows = schema['dimensions']
        if schema.get('size') != 371 or [r['index'] for r in rows] != list(range(371)):
            raise ValueError('complete canonical 371D schema required')
        self._rows = copy.deepcopy(rows)
        self.known_config = {} if known_config is None else copy.deepcopy(dict(known_config))
        unknown = set(self.known_config) - _KNOWN_KEYS
        if unknown:
            raise ValueError('unknown known_config fields: ' + ','.join(sorted(unknown)))
        self._validate_known_config()
        self.source_qualification = {} if source_qualification is None else dict(source_qualification)
        source_keys = {key for row in rows for key in row['required_validity_keys']}
        unknown = set(self.source_qualification) - (source_keys | {'rotation_symmetry_valid', 'force_proxy'})
        if unknown:
            raise ValueError('unknown source qualification keys: ' + ','.join(sorted(unknown)))
        for (key, value) in self.source_qualification.items():
            if key == 'force_proxy':
                if not isinstance(value, dict):
                    raise TypeError('force_proxy raw qualification mapping required')
            elif type(value) is not bool and (not (isinstance(value, stere_task_state_adapt__torch.Tensor) and value.dtype == stere_task_state_adapt__torch.bool)):
                raise ValueError('source eligibility must be bool or boolean tensor: ' + key)
        if allowlist is None:
            indices = [r['index'] for r in rows if self._deployable(r) and (r['estimator_head'] in _VISUAL_HEADS or r['block'] in ('joint_position', 'joint_velocity', 'last_submitted_action24'))]
        else:
            if isinstance(allowlist, (str, bytes)):
                raise TypeError('allowlist must be a sequence of fields')
            indices = []
            for field in allowlist:
                if type(field) is int:
                    matches = [field] if 0 <= field < 371 else []
                elif isinstance(field, str):
                    matches = [r['index'] for r in rows if field in (r['semantic_name'], r['block'])]
                else:
                    raise TypeError('allowlist fields must be canonical indices or names')
                if not matches:
                    raise ValueError('unknown schema allowlist field: ' + str(field))
                if any((not self._deployable(rows[i]) for i in matches)):
                    raise ValueError('NA/forbidden schema field is not deployable: ' + str(field))
                indices.extend(matches)
        self.field_indices = tuple(sorted(set(indices)))
        self.field_order = tuple((rows[i]['semantic_name'] for i in self.field_indices))
        if isinstance(extra_heads, str):
            raise TypeError('extra_heads must be a sequence')
        self.extra_heads = tuple(extra_heads)
        if len(set(self.extra_heads)) != len(self.extra_heads) or any((k not in EXTRA_SHAPES for k in self.extra_heads)):
            raise ValueError('unknown or duplicate extra estimator head')
        self.extra_head_metadata = {} if extra_head_metadata is None else copy.deepcopy(dict(extra_head_metadata))
        if set(self.extra_head_metadata) - set(self.extra_heads):
            raise ValueError('extra_head_metadata requires a selected extra head')
        self.value_size = len(self.field_indices) + sum((stere_task_state_adapt__math.prod(EXTRA_SHAPES[k]) for k in self.extra_heads))
        if not self.value_size:
            raise ValueError('at least one task state field required')
        self.input_size = self.feature_size = 3 * self.value_size

    @staticmethod
    def _deployable(row):
        return row['deployment_allowed'] is True and row['supervision_status'] != 'NA' and (row['student_source'] != 'unobservable') and ('supervision_forbidden' not in row['required_validity_keys'])

    def _validate_known_config(self):
        shapes = {'joint_lower': ((6,), (12,)), 'joint_upper': ((6,), (12,)), 'reset_joint_reference': ((12,),), 'grip_drive_stiffness': ((2,),), 'wrist_reset_xyz': ((2, 3),)}
        for (key, value) in self.known_config.items():
            value = stere_task_state_adapt__torch.as_tensor(value, dtype=stere_task_state_adapt__torch.float32)
            if tuple(value.shape) not in shapes[key] or not bool(stere_task_state_adapt__torch.isfinite(value).all()):
                raise ValueError('known_config requires finite raw SI values and declared shape: ' + key)
            self.known_config[key] = value.tolist()
        if ('joint_lower' in self.known_config) != ('joint_upper' in self.known_config):
            raise ValueError('joint_lower and joint_upper must both be explicitly supplied')
        if 'joint_lower' in self.known_config:
            (low, high) = (stere_task_state_adapt__torch.tensor(self.known_config[k]).repeat(2 if len(self.known_config[k]) == 6 else 1) for k in ('joint_lower', 'joint_upper'))
            if not bool((high > low).all()):
                raise ValueError('joint normalization upper must exceed lower')

    @staticmethod
    def _prediction(value, shape, reference, name):
        if not isinstance(value, stere_task_state_adapt__torch.Tensor) or tuple(value.shape) != tuple(shape) or (not value.is_floating_point()) or (value.device != reference.device):
            raise ValueError(name + ' must be floating prediction with shape ' + str(tuple(shape)) + ' on history device')
        return value

    def _eligibility(self, key, reference, index=None):
        value = self.source_qualification.get(key, False)
        if type(value) is bool:
            return stere_task_state_adapt__torch.full((reference.shape[0],), value, dtype=stere_task_state_adapt__torch.bool, device=reference.device)
        value = value.to(reference.device)
        if value.shape == reference.shape[:1]:
            return value
        if value.shape == (reference.shape[0], 371) and index is not None:
            return value[:, index]
        raise ValueError('source eligibility mask shape must be [B] or [B,371]: ' + key)

    def _known(self, key, reference):
        return stere_task_state_adapt__torch.as_tensor(self.known_config[key], device=reference.device, dtype=reference.dtype)

    def _joint_normalization(self, value, reference):
        if 'joint_lower' not in self.known_config:
            return None
        (low, high) = (self._known(key, reference) for key in ('joint_lower', 'joint_upper'))
        if low.numel() == 6:
            low = low.repeat(2)
        if high.numel() == 6:
            high = high.repeat(2)
        return 2 * (value - low) / (high - low) - 1

    def _passthrough(self, row, history, prediction):
        """Return a field only where the 62D sensor contract supplies its source."""
        p = history.proprio[:, -1]
        (block, col) = (row['block'], row['block_column'])
        if block == 'joint_position':
            q = self._joint_normalization(p[:, :12], prediction)
            return (None if q is None else q[:, col], 'history.proprio.q; explicit normalization limits')
        if block == 'joint_velocity':
            return (p[:, 12 + col] / 0.3, 'history.proprio.qd')
        if block == 'last_submitted_action24':
            return (p[:, 38 + col], 'history.proprio.previous_actually_issued_action24')
        if block == 'reset_joint_reference' and block in self.known_config:
            value = self._joint_normalization(self._known(block, prediction), prediction)
            if value is not None:
                return (value[col].expand(len(p)), 'declared known_config.reset_joint_reference')
        if block == 'grip_drive_stiffness' and block in self.known_config:
            return (((self._known(block, prediction)[col] - 10) / 10).expand(len(p)), 'declared known_config.grip_drive_stiffness')
        if block == 'wrist_pose_from_reset':
            return (None, 'sensor palm_link FK is not canonical wrist root; same-frame palm-to-root calibration unavailable')
        return (None, 'unavailable in declared 62D sensor history/configuration')

    def _canonical_fields(self, estimates, history, prediction):
        b = len(history)
        supplied = estimates.get('state_valid')
        if supplied is None:
            supplied = stere_task_state_adapt__torch.ones_like(prediction, dtype=stere_task_state_adapt__torch.bool)
        if not isinstance(supplied, stere_task_state_adapt__torch.Tensor) or supplied.dtype != stere_task_state_adapt__torch.bool or supplied.shape != prediction.shape or (supplied.device != prediction.device):
            raise ValueError('state_valid mask must be boolean [B,371] on prediction device')
        uncertainty = estimates.get('state_uncertainty')
        if uncertainty is not None:
            self._prediction(uncertainty, prediction.shape, prediction, 'state_uncertainty')
            if bool((~stere_task_state_adapt__torch.isfinite(uncertainty) | (uncertainty < 0)).any()):
                raise ValueError('state_uncertainty must be finite and nonnegative')
        current = history.history_valid[:, -1]
        pair = history.image_valid[:, -1].all(-1)
        observed = history.point_valid[:, -1].any(-1) | history.depth_valid[:, -1].flatten(1).any(-1)
        (values, masks, uncertainties, metadata) = ([], [], [], [])
        blocked = {}
        for index in self.field_indices:
            row = self._rows[index]
            visual = row['estimator_head'] in _VISUAL_HEADS
            mask = current.clone()
            for key in row['required_validity_keys']:
                mask = mask & self._eligibility(key, prediction, index)
            if visual:
                value = prediction[:, index]
                mask = mask & pair & observed & supplied[:, index]
                if field_history_frames(row) == 2:
                    mask = mask & ((history.history_valid & history.image_valid.all(-1)).sum(-1) >= 2)
                if row['block'] == 'object_quaternion':
                    mask = mask & (prediction[:, 64:68].norm(dim=-1) > 1e-08)
                    mask = mask & self._eligibility('rotation_symmetry_valid', prediction, index)
                mask = mask & stere_task_state_adapt__torch.isfinite(value)
                value = stere_task_state_adapt__torch.where(mask, value, 0.0)
                if row['loss_type'] == 'classification':
                    value = value.sigmoid()
                source = 'estimated371.' + row['estimator_head']
                uncert = stere_task_state_adapt__torch.ones(b, device=prediction.device, dtype=prediction.dtype) if uncertainty is None else uncertainty[:, index]
                uncertainty_source = 'unknown sentinel 1; not calibrated' if uncertainty is None else 'estimated normalized-field uncertainty sideband; calibration unverified'
            else:
                (value, source) = self._passthrough(row, history, prediction)
                if value is None:
                    value = stere_task_state_adapt__torch.zeros(b, device=prediction.device, dtype=prediction.dtype)
                    mask = mask & False
                    blocked[row['semantic_name']] = source
                mask = mask & stere_task_state_adapt__torch.isfinite(value)
                uncert = stere_task_state_adapt__torch.ones_like(value)
                uncertainty_source = 'unknown sensor/configuration uncertainty sentinel 1'
            values.append(stere_task_state_adapt__torch.where(mask, value, 0.0))
            masks.append(mask)
            uncertainties.append(stere_task_state_adapt__torch.where(mask, uncert, stere_task_state_adapt__torch.ones_like(uncert)))
            metadata.append(dict(index=index, semantic_name=row['semantic_name'], unit=row['unit'], coordinate_frame=row['coordinate_frame'], normalization=row['normalization'], source=source, validity=list(row['required_validity_keys']), validity_description=row['validity_mask'], uncertainty=uncertainty_source, role='estimated_contact_probability_proxy' if visual and row['loss_type'] == 'classification' else 'estimated_task_field' if visual else 'declared_sensor_or_configuration_passthrough'))
        return (values, masks, uncertainties, metadata, blocked)

    def _extra_fields(self, estimates, history, prediction):
        (values, masks, uncertainties, metadata, blocked) = ([], [], [], [], {})
        b = len(history)
        for name in self.extra_heads:
            shape = (b, *EXTRA_SHAPES[name])
            width = stere_task_state_adapt__math.prod(EXTRA_SHAPES[name])
            if name not in estimates:
                raise ValueError('missing selected extra prediction: ' + name)
            raw = self._prediction(estimates[name], shape, prediction, name)
            gate = history.history_valid[:, -1] & history.image_valid[:, -1].all(-1)
            gate = gate & history.point_valid[:, -1].any(-1)
            declared = self.extra_head_metadata.get(name, {})
            if not isinstance(declared, dict):
                raise ValueError('extra head metadata must be a mapping')
            if name in ('geometry', 'velocity', 'deformation', 'slip_velocity') and (not all((isinstance(declared.get(k), str) and declared[k].strip() for k in ('unit', 'coordinate_frame')))):
                gate = gate & False
                blocked[name] = 'extra visual units and coordinate_frame are not declared'
            if name in ('velocity', 'slip_velocity', 'slip_logits'):
                gate = gate & ((history.history_valid & history.image_valid.all(-1)).sum(-1) >= 2)
            for key in ('source_valid', 'identity_aligned', 'timestamp_valid', 'visibility_valid', 'correspondence_valid'):
                gate = gate & self._eligibility(key, prediction)
            valid = gate.reshape(b, *[1] * (raw.ndim - 1)).expand_as(raw) & stere_task_state_adapt__torch.isfinite(raw)
            if name == 'force_proxy':
                from dh116.rl.student_data import StereoEstimationLoss as StereoEstimationLoss
                qualification = self.source_qualification.get('force_proxy')
                (qualified, reason) = StereoEstimationLoss._force_mask(qualification, raw)
                valid = valid & qualified
                if reason:
                    blocked[name] = reason
            supplied = estimates.get(name + '_valid')
            if supplied is not None:
                if not isinstance(supplied, stere_task_state_adapt__torch.Tensor) or supplied.dtype != stere_task_state_adapt__torch.bool or supplied.shape != raw.shape or (supplied.device != raw.device):
                    raise ValueError('extra validity mask shape must match prediction: ' + name)
                valid = valid & supplied
            value = stere_task_state_adapt__torch.where(valid, raw, 0.0)
            if name in ('contact_logits', 'slip_logits'):
                value = value.sigmoid()
            value = stere_task_state_adapt__torch.where(valid, value, 0.0)
            uncertainty = estimates.get(name + '_uncertainty')
            if uncertainty is None:
                uncertainty = stere_task_state_adapt__torch.ones_like(raw)
            else:
                self._prediction(uncertainty, shape, prediction, name + '_uncertainty')
                if bool((~stere_task_state_adapt__torch.isfinite(uncertainty) | (uncertainty < 0)).any()):
                    raise ValueError('extra uncertainty must be finite nonnegative: ' + name)
            values.extend(value.reshape(b, width).unbind(-1))
            masks.extend(valid.reshape(b, width).unbind(-1))
            uncertainties.extend(stere_task_state_adapt__torch.where(valid, uncertainty, stere_task_state_adapt__torch.ones_like(uncertainty)).reshape(b, width).unbind(-1))
            declared = self.extra_head_metadata.get(name, {})
            if not isinstance(declared, dict):
                raise ValueError('extra head metadata must be a mapping')
            unit = 'N' if name == 'force_proxy' else 'probability' if name.endswith('logits') else declared.get('unit', 'unbound estimator units')
            frame = 'accepted_pad_normal' if name == 'force_proxy' else declared.get('coordinate_frame', 'unbound estimator head coordinates')
            for col in range(width):
                metadata.append(dict(index=None, semantic_name=f'{name}[{col}]', unit=unit, coordinate_frame=frame, normalization=declared.get('normalization', 'no additional normalization'), source='estimated_extra_head.' + name, validity='sensor/eligibility and explicit per-head mask; raw proof additionally required for force', uncertainty='unknown sentinel 1 unless explicit per-head uncertainty supplied; calibration unverified', role='additional_estimated_feature; not canonical371 field'))
        return (values, masks, uncertainties, metadata, blocked)

    def forward(self, estimates, history):
        if type(history) is not stere_task_state_adapt__StereoHistoryBatch:
            raise TypeError('TaskStateAdapter requires StereoHistoryBatch; privileged mappings forbidden')
        history.validate()
        if not isinstance(estimates, dict):
            raise TypeError('named estimator outputs required')
        allowed = {'latent', 'privileged_state', 'state_valid', 'state_uncertainty'} | set(EXTRA_SHAPES)
        allowed |= {key + suffix for key in EXTRA_SHAPES for suffix in ('_valid', '_uncertainty')}
        if set(estimates) - allowed:
            raise ValueError('unknown/privileged estimator inputs forbidden')
        prediction = self._prediction(estimates.get('privileged_state'), (len(history), 371), history.proprio, 'estimated371')
        (values, masks, uncertainties, fields, blocked) = self._canonical_fields(estimates, history, prediction)
        extra = self._extra_fields(estimates, history, prediction)
        values.extend(extra[0])
        masks.extend(extra[1])
        uncertainties.extend(extra[2])
        fields.extend(extra[3])
        blocked.update(extra[4])
        value = stere_task_state_adapt__torch.stack(values, -1)
        valid = stere_task_state_adapt__torch.stack(masks, -1)
        uncertainty = stere_task_state_adapt__torch.stack(uncertainties, -1)
        coverage = []
        selected = set(self.field_indices)
        for row in self._rows:
            status = 'NA' if not self._deployable(row) else 'selected' if row['index'] in selected else 'excluded'
            coverage.append(dict(index=row['index'], semantic_name=row['semantic_name'], status=status, reason='schema forbids deployment/supervision' if status == 'NA' else 'declared allowlist; per-sample validity required' if status == 'selected' else 'outside configured task-state profile'))
        metadata = dict(fields=fields, schema_coverage=coverage, blocked_reasons=blocked, state_contract_file_sha256=self.state_contract_sha256, field_order=tuple((f['semantic_name'] for f in fields)), feature_layout=('values', 'valid', 'uncertainty'), evidence_scope='runtime_configuration_not_native_qualification', uncertainty_policy='sentinel 1 is unknown; no calibrated uncertainty supplied by the estimator', missing_value_policy='NA in values; neutral finite encoding only in masked actor_features')
        return dict(values=stere_task_state_adapt__torch.where(valid, value, float('nan')), valid=valid, uncertainty=uncertainty, metadata=metadata, actor_features=stere_task_state_adapt__torch.cat((value, valid.to(value.dtype), uncertainty), -1))

class IndexedStereoRolloutStorage:
    """Storage hooks consumed by PPOTrainer, independent of its PPO equations.

    ``capture`` owns exact CPU sensor frames and returns int64[B] history IDs.
    ``resolve`` materializes only explicitly requested IDs on the target device.
    Explicit diagnostics may request a large CPU batch; PPOTrainer bounds all
    collection, frozen checks, and gradient forwards by ``max_visual_batch``.
    """

    def __init__(self, *, max_visual_batch=1):
        if type(max_visual_batch) is not int or max_visual_batch < 1:
            raise ValueError('max_visual_batch must be a positive integer')
        self.max_visual_batch = max_visual_batch
        self._generation = 0
        self.reset()

    def reset(self):
        self._generation = next(_GENERATIONS)
        self._images = []
        self._image_ids = {}
        self._frames = []
        self._frame_ids = {}
        self._samples = []
        self._contract = None
        self._rgb_uint8_count = self._rgb_float32_count = 0
        self._max_resolved_batch = 0

    def capture(self, value):
        if isinstance(value, dict):
            value = stere_rollo_stora__StereoHistoryBatch.from_mapping(value)
        if type(value) is not stere_rollo_stora__StereoHistoryBatch:
            raise TypeError('indexed rollout accepts exact sensor history only; privileged mappings forbidden')
        value.validate()
        contract = tuple(((key, tuple(tensor.shape[1:]), tensor.dtype) for (key, tensor) in value.as_dict().items()))
        if self._contract is not None and contract != self._contract:
            raise ValueError('sensor history shape/dtype changed within on-policy rollout')
        self._contract = contract
        arrays = {key: tensor.detach() for (key, tensor) in value.as_dict().items()}
        (batch, timesteps) = value.history_valid.shape
        refs = []
        for row in range(batch):
            frame_ids = []
            for tick in range(timesteps):
                non_rgb = {key: tensor[row, tick].cpu() for (key, tensor) in arrays.items() if key not in _RGB}
                metadata = tuple((_identity(non_rgb[name]) for name in ('dt', 'image_age', 'image_valid', 'history_valid')))
                image_ids = []
                for key in _RGB:
                    rgb = arrays[key][row, tick].cpu()
                    identity = (key, _identity(rgb), metadata)
                    image_id = self._image_ids.get(identity)
                    if image_id is None:
                        encoded = (rgb * 255).round().to(stere_rollo_stora__torch.uint8)
                        decoded = encoded.to(stere_rollo_stora__torch.float32) / 255
                        lossless = stere_rollo_stora__torch.equal(decoded.view(stere_rollo_stora__torch.uint8), rgb.contiguous().view(stere_rollo_stora__torch.uint8))
                        stored = encoded.clone() if lossless else rgb.clone()
                        image_id = len(self._images)
                        self._images.append(stored)
                        self._image_ids[identity] = image_id
                        if lossless:
                            self._rgb_uint8_count += 1
                        else:
                            self._rgb_float32_count += 1
                    image_ids.append(image_id)
                identity = (tuple(image_ids), tuple(((key, _identity(tensor)) for (key, tensor) in non_rgb.items())))
                frame_id = self._frame_ids.get(identity)
                if frame_id is None:
                    frame_id = len(self._frames)
                    self._frames.append((tuple(image_ids), {key: tensor.clone() for (key, tensor) in non_rgb.items()}))
                    self._frame_ids[identity] = frame_id
                frame_ids.append(frame_id)
            sample_id = len(self._samples)
            self._samples.append(stere_rollo_stora__torch.tensor(frame_ids, dtype=stere_rollo_stora__torch.int64))
            refs.append(self._generation * _GENERATION_STRIDE + sample_id)
        return stere_rollo_stora__torch.tensor(refs, dtype=stere_rollo_stora__torch.int64)

    def resolve(self, references, device='cpu'):
        if not isinstance(references, stere_rollo_stora__torch.Tensor) or references.dtype != stere_rollo_stora__torch.int64 or references.ndim != 1 or (not references.numel()):
            raise ValueError('compact int64[B] history references required')
        ids = references.detach().cpu()
        generation = stere_rollo_stora__torch.div(ids, _GENERATION_STRIDE, rounding_mode='floor')
        indices = ids.remainder(_GENERATION_STRIDE)
        if bool((generation != self._generation).any()) or bool((indices >= len(self._samples)).any()):
            raise ValueError('stale or foreign sensor history reference generation')
        self._max_resolved_batch = max(self._max_resolved_batch, len(ids))
        output = {key: [] for key in stere_rollo_stora__INPUT_KEYS}
        for index in indices.tolist():
            frames = [self._frames[i] for i in self._samples[index].tolist()]
            for key in stere_rollo_stora__INPUT_KEYS:
                if key in _RGB:
                    view = _RGB.index(key)
                    values = []
                    for (image_ids, _) in frames:
                        rgb = self._images[image_ids[view]]
                        values.append(rgb.float() / 255 if rgb.dtype == stere_rollo_stora__torch.uint8 else rgb)
                else:
                    values = [record[key] for (_, record) in frames]
                output[key].append(stere_rollo_stora__torch.stack(values))
        return {key: stere_rollo_stora__torch.stack(values).to(device) for (key, values) in output.items()}

    def stats(self):
        tensors = list(self._images) + self._samples
        tensors.extend((value for (_, record) in self._frames for value in record.values()))
        return dict(image_count=len(self._images), frame_count=len(self._frames), sample_count=len(self._samples), rgb_uint8_count=self._rgb_uint8_count, rgb_float32_count=self._rgb_float32_count, storage_bytes=sum((t.numel() * t.element_size() for t in tensors)), max_resolved_batch=self._max_resolved_batch, device='cpu', identity='SHA256 of exact dtype/shape/raw bytes and available sensor metadata', evidence_scope='software_storage_only_not_native_capture')
stere_stude_ppo__StereoStateEstimator = StereoStateEstimator
stere_stude_ppo__StereoStudentActor = StereoStudentActor
from dh116.env.contracts import action_sha256
from dh116.env.contracts import require_sha
VERSION = 'dh116-stereo-asymmetric-student-ppo-v1'
stere_stude_ppo__INPUT_KEYS = tuple((field.name for field in stere_stude_ppo__fields(stere_stude_ppo__StereoHistoryBatch)))

def normalize_network_config(config=None):
    validate_runtime_execution(config=config)
    value = deepcopy(config or {})
    if not isinstance(value, dict):
        raise ValueError('network config dictionary required')
    for (name, default) in dict(embed_dim=128, history=4, num_heads=4, num_layers=2, dropout=0.0).items():
        value.setdefault(name, default)
    value.setdefault('image_processing_batch_size', None)
    value.setdefault('actor_input_mode', 'latent')
    value.setdefault('point_coordinate_mode', 'world')
    value.setdefault('point_source_encoding', False)
    if type(value['point_source_encoding']) is not bool:
        raise ValueError('point_source_encoding must be bool')
    value.setdefault('task_state_config', {})
    if value['actor_input_mode'] not in ('latent', 'estimated_task_state', 'hybrid'):
        raise ValueError('invalid Actor input mode')
    if value['point_coordinate_mode'] not in ('world', 'palms') or not isinstance(value['task_state_config'], dict):
        raise ValueError('explicit point coordinate mode and task-state config required')
    batch_size = value['image_processing_batch_size']
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError('image_processing_batch_size must be a positive integer or None')
    for name in ('embed_dim', 'history', 'num_heads', 'num_layers'):
        if type(value[name]) is not int or value[name] < 1:
            raise ValueError('positive network dimension required: ' + name)
    if value['embed_dim'] % value['num_heads'] or value['dropout'] != 0.0:
        raise ValueError('divisible attention dimensions and dropout=0 required for PPO likelihood consistency')
    if value.get('action_size', 24) != 24 or value.get('proprio', 62) != 62 or value.get('points', 128) != 128:
        raise ValueError('independent stereo input requires 24 actions,62 proprio,128 points')
    if value.get('action_contract_sha256', action_sha256()) != action_sha256():
        raise ValueError('canonical action SHA mismatch')
    if 'image_resolution_wh' in value:
        resolution = value['image_resolution_wh']
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2 or any((type(v) is not int or v < 16 for v in resolution)):
            raise ValueError('RGB resolution must be integer [width,height], each >=16')
    if 'backbone' in value:
        backbone = value['backbone']
        expected = dict(type='GN_ResNet18', block='BasicBlock', stages=[2, 2, 2, 2], channels=[64, 128, 256, 512], normalization='GroupNorm', weights='scratch', pretrained=False, shared_left_right=True)
        if not isinstance(backbone, dict) or any((key not in expected or item != expected[key] for (key, item) in backbone.items())):
            raise ValueError('only the actual shared GroupNorm ResNet18 backbone is supported')
    return value

def make_stereo_actor(network_config=None):
    config = normalize_network_config(network_config)
    estimator = stere_stude_ppo__StereoStateEstimator(**{name: config[name] for name in ('embed_dim', 'history', 'num_heads', 'num_layers', 'image_processing_batch_size', 'point_coordinate_mode', 'point_source_encoding')})
    if 'image_resolution_wh' in config:
        (width, height) = config['image_resolution_wh']

        def check_resolution(module, arguments):
            if tuple(arguments[0].left_rgb.shape[-2:]) != (height, width):
                raise ValueError('RGB history resolution differs from network/calibration contract')
        estimator.register_forward_pre_hook(check_resolution)
    return stere_stude_ppo__StereoStudentActor(estimator, action_contract_sha256=action_sha256(), input_mode=config['actor_input_mode'], task_state_config=config['task_state_config'])

def history_to_tree(history, *, owned=True):
    """Own/neutralize legacy arrays, or expose validated input for pool capture.

    ``owned=False`` avoids cloning a full visual history; the indexed storage
    capture hook immediately takes ownership of each new exact sensor frame.
    """
    if isinstance(history, dict):
        history = stere_stude_ppo__StereoHistoryBatch.from_mapping(history)
    if type(history) is not stere_stude_ppo__StereoHistoryBatch:
        raise TypeError('only exact StereoHistoryBatch sensor input is accepted')
    history.validate()
    if not owned:
        return history.as_dict()
    result = {key: value.detach().clone() for (key, value) in history.as_dict().items()}
    active = result['history_valid']
    depth = result['depth_valid'] & active[:, :, None, None]
    point = result['point_valid'] & active[:, :, None]
    for key in ('depth_m', 'depth_confidence'):
        result[key] = stere_stude_ppo__torch.where(depth, result[key], 0.0)
    result['points_world'] = stere_stude_ppo__torch.where(point[:, :, :, None], result['points_world'], 0.0)
    for hand in ('left', 'right'):
        mask = result['point_valid_' + hand] & point
        result['points_' + hand + '_palm'] = stere_stude_ppo__torch.where(mask[..., None], result['points_' + hand + '_palm'], 0.0)
    result['point_confidence'] = stere_stude_ppo__torch.where(point, result['point_confidence'], 0.0)
    return result

def _history(value):
    if type(value) is stere_stude_ppo__StereoHistoryBatch:
        return value.validate()
    if isinstance(value, dict):
        return stere_stude_ppo__StereoHistoryBatch.from_mapping(value)
    raise TypeError('StereoHistoryBatch or exact sensor dictionary required; privileged input forbidden')

class StereoAsymmetricPolicy(stere_stude_ppo__EagerRuntimePolicy, stere_stude_ppo__nn.Module):

    def __init__(self, network_config=None):
        super().__init__()
        self.network_config = normalize_network_config(network_config)
        self.actor = make_stereo_actor(self.network_config)
        self.critic = stere_stude_ppo__nn.Sequential(stere_stude_ppo__nn.Linear(371, 256), stere_stude_ppo__nn.ELU(), stere_stude_ppo__nn.Linear(256, 256), stere_stude_ppo__nn.ELU(), stere_stude_ppo__nn.Linear(256, 1))

    @property
    def log_std(self):
        return self.actor.log_std

    def latent_mean(self, actor_tree):
        return self.actor.latent_mean(_history(actor_tree))

    def deterministic(self, history_or_tree):
        return self.actor.deterministic(_history(history_or_tree))

    def value(self, critic_obs):
        if not isinstance(critic_obs, stere_stude_ppo__torch.Tensor) or critic_obs.ndim != 2 or critic_obs.shape[1] != 371 or (not critic_obs.is_floating_point()) or (not bool(stere_stude_ppo__torch.isfinite(critic_obs).all())):
            raise ValueError('independent finite privileged critic [N,371] required')
        return self.critic(critic_obs).squeeze(-1)

def make_student_trainer(config, num_envs, device='cpu', rollout_steps=None):
    from dh116.rl.ppo import PPOConfig
    from dh116.rl.ppo import PPOTrainer
    validate_runtime_execution(config=config)
    if not isinstance(config, dict) or not isinstance(config.get('network', {}), dict):
        raise ValueError('explicit independent Student configuration required')
    if config.get('critic_observation_size', 371) != 371 or config.get('action_size', 24) != 24:
        raise ValueError('371D critic and canonical24 actions required')
    if config.get('actor_input', 'StereoHistoryBatch_only') != 'StereoHistoryBatch_only':
        raise ValueError('Student actor input must be stereo sensor history only')
    model = StereoAsymmetricPolicy(config.get('network'))
    steps = config.get('rollout_steps', 32) if rollout_steps is None else rollout_steps
    storage_mode = config.get('rollout_storage', 'indexed_frames')
    if storage_mode not in ('indexed_frames', 'legacy'):
        raise ValueError('rollout_storage must be indexed_frames or legacy')
    max_visual_batch = config.get('max_visual_batch', 1)
    if type(max_visual_batch) is not int or max_visual_batch < 1:
        raise ValueError('max_visual_batch must be a positive integer')
    actor_storage = None
    if storage_mode == 'indexed_frames':
        _stere_stude_ppo_import__IndexedStereoRolloutStorage = IndexedStereoRolloutStorage
        actor_storage = _stere_stude_ppo_import__IndexedStereoRolloutStorage(max_visual_batch=max_visual_batch)
    trainer = PPOTrainer(model, PPOConfig(**config.get('ppo', {})), rollout_steps=steps, num_envs=num_envs, device=device, actor_storage=actor_storage, max_visual_batch=max_visual_batch if actor_storage is not None else None)
    validate_runtime_execution(model=trainer.model)
    trainer.rollout_storage_mode = storage_mode
    trainer.student_config = deepcopy(config)
    trainer.auxiliary_bc = None
    if config.get('auxiliary_bc', {}).get('enabled', False):
        raise ValueError('auxiliary BC requires explicit qualified independent attachment')
    return trainer

class _IndependentAuxiliaryBC:

    def __init__(self, trainer, bc, metadata):
        (self.trainer, self.bc, self.metadata) = (trainer, bc, deepcopy(metadata))

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
    if not isinstance(metadata, dict) or metadata.get('qualified') is not True or metadata.get('source_scope') != 'native' or (metadata.get('schema_only') is not False) or (metadata.get('ppo_storage_insertion') is not False):
        raise ValueError('qualified independent BC dataset metadata required')
    require_sha(metadata.get('dataset_sha256'), 'qualified BC dataset SHA')
    qualification = metadata.get('qualification_evidence')
    if not isinstance(qualification, (str, dict)):
        raise ValueError('raw native qualification evidence required for auxiliary BC')
    from dh116.rl.qualification import verify_native_episode
    verify_native_episode(qualification, expected_identity=metadata.get('model_identity'))
    optimizer = getattr(bc_trainer, 'optimizer', None)
    if optimizer is None or optimizer is trainer.optimizer or (not callable(getattr(bc_trainer, 'step', None))):
        raise ValueError('independent BC optimizer and step interface required')
    critic_ids = set(map(id, trainer.model.critic.parameters()))
    if any((id(parameter) in critic_ids for group in optimizer.param_groups for parameter in group['params'])):
        raise ValueError('auxiliary BC optimizer cannot update privileged critic')
    trainer.auxiliary_bc = _IndependentAuxiliaryBC(trainer, bc_trainer, metadata)
    return trainer.auxiliary_bc
