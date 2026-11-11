"""Strict deployable stereo observations; no task, Teacher or critic input."""
from dataclasses import dataclass, fields, field

import torch


SENSOR_FIELDS = ('left_rgb', 'right_rgb', 'depth_m', 'depth_valid', 'depth_confidence',
                 'points_world', 'point_valid', 'proprio', 'dt', 'image_age', 'image_valid',
                 'points_left_palm','points_right_palm','point_valid_left','point_valid_right','point_confidence')


def _populate_palms(batch):
    """Compatibility inputs derive local geometry only from the same measured FK.

    Default confidence is an availability indicator, not a calibrated matching
    probability. Native/inference pipelines supply actual RGB-derived scores.
    """
    if any(getattr(batch,key) is None for key in ('points_left_palm','points_right_palm',
            'point_valid_left','point_valid_right')):
        from .tabletop_stereo_geometry import world_to_palms
        prefix=batch.points_world.shape[:-2]
        pose=batch.proprio[...,24:38].reshape(-1,2,7).clone()
        history=getattr(batch,'history_valid',None)
        if history is not None:
            inactive=~history.flatten()
            pose[inactive]=0.;pose[inactive,:,3]=1.
        local=world_to_palms(batch.points_world.reshape(-1,128,3),pose).reshape(*prefix,2,128,3)
        for index,hand in enumerate(('left','right')):
            mask=batch.point_valid & torch.isfinite(local[...,index,:,:]).all(-1)
            if getattr(batch,'point_valid_'+hand) is None:setattr(batch,'point_valid_'+hand,mask)
            if getattr(batch,'points_'+hand+'_palm') is None:
                setattr(batch,'points_'+hand+'_palm',torch.where(mask[...,None],local[...,index,:,:],float('nan')))
    if batch.point_confidence is None:batch.point_confidence=batch.point_valid.float()


def _validate(values, *, history_valid=None):
    for name in SENSOR_FIELDS:
        if not isinstance(values[name], torch.Tensor):
            raise TypeError(f'{name} must be a tensor')
    left = values['left_rgb']
    prefix = left.shape[:2] if history_valid is not None else left.shape[:1]
    ndim = 5 if history_valid is not None else 4
    if left.ndim != ndim or left.shape[-3] != 3 or min(left.shape[-2:]) < 16 or not prefix[0]:
        raise ValueError('RGB shape must be [B,3,H,W] or [B,T,3,H,W], H/W>=16')
    h, w = left.shape[-2:]
    shapes = dict(left_rgb=(*prefix,3,h,w), right_rgb=(*prefix,3,h,w),
        depth_m=(*prefix,h,w), depth_valid=(*prefix,h,w), depth_confidence=(*prefix,h,w),
        points_world=(*prefix,128,3), point_valid=(*prefix,128), proprio=(*prefix,62),
        points_left_palm=(*prefix,128,3),points_right_palm=(*prefix,128,3),
        point_valid_left=(*prefix,128),point_valid_right=(*prefix,128),point_confidence=(*prefix,128),
        dt=prefix, image_age=prefix, image_valid=(*prefix,2))
    for name, shape in shapes.items():
        value = values[name]
        dtype = torch.bool if name in ('depth_valid','point_valid','image_valid','point_valid_left','point_valid_right') else torch.float32
        if tuple(value.shape) != tuple(shape) or value.dtype != dtype or value.device != left.device:
            raise ValueError(f'{name} must have shape {shape}, dtype {dtype}, device {left.device}')
    active = torch.ones(prefix, dtype=torch.bool, device=left.device)
    if history_valid is not None:
        if (history_valid.shape != prefix or history_valid.dtype != torch.bool
                or history_valid.device != left.device):
            raise ValueError('history_valid must be boolean [B,T] on the observation device')
        active = history_valid
    for name in ('left_rgb','right_rgb','proprio','dt','image_age'):
        if not bool(torch.isfinite(values[name]).all()):
            raise ValueError(f'{name} must be finite')
    for name in ('left_rgb','right_rgb'):
        if bool(((values[name]<0)|(values[name]>1)).any()):
            raise ValueError('RGB must be float32 in [0,1]')
    if bool(((values['dt']<=0)&active).any()) or bool((values['image_age']<0).any()):
        raise ValueError('active dt must be positive and image_age nonnegative')
    proprio = values['proprio']
    if bool(((proprio[...,38:].abs()>1)&active[...,None]).any()):
        raise ValueError('previous actually-issued action must be bounded normalized24')
    for lo in (27,34):
        if bool(((proprio[...,lo:lo+4].norm(dim=-1)<=1e-8)&active).any()):
            raise ValueError('active measured palm wxyz quaternion must be nonzero')
    depth_valid = values['depth_valid'] & active[...,None,None]
    depth = values['depth_m']; confidence = values['depth_confidence']
    if (bool((depth_valid & (~torch.isfinite(depth)|(depth<=0))).any())
            or bool((depth_valid & (~torch.isfinite(confidence)|(confidence<0)|(confidence>1))).any())):
        raise ValueError('valid depth/confidence must be finite, depth>0 and confidence in [0,1]')
    point_valid = values['point_valid'] & active[...,None]
    if bool((point_valid & ~torch.isfinite(values['points_world']).all(-1)).any()):
        raise ValueError('valid world points must be finite')
    for hand in ('left','right'):
        valid=values['point_valid_'+hand] & point_valid
        if bool((valid & ~torch.isfinite(values['points_'+hand+'_palm']).all(-1)).any()):
            raise ValueError('valid palm-local points require finite observed world geometry')
    confidence=values['point_confidence']
    if bool((point_valid & (~torch.isfinite(confidence)|(confidence<0)|(confidence>1))).any()):
        raise ValueError('valid point confidence must be finite in [0,1]')


class _TensorBatch:
    __slots__ = ()

    @classmethod
    def from_mapping(cls, value):
        expected = {f.name for f in fields(cls)}
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError('unknown or missing sensor fields; privileged inputs are forbidden')
        return cls(**value)

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def _map(self, operation):
        return type(self)(**{name:operation(value) for name,value in self.as_dict().items()})

    def to(self, device):
        return self._map(lambda value:value.to(device))

    def clone(self):
        return self._map(lambda value:value.clone())

    def detach(self):
        return self._map(lambda value:value.detach())

    def index_select(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.left_rgb.device)
        if indices.ndim != 1:
            raise ValueError('batch indices must be one dimensional')
        return self._map(lambda value:value.index_select(0,indices))

    def __len__(self):
        return self.left_rgb.shape[0]


@dataclass(slots=True)
class SensorBatch(_TensorBatch):
    left_rgb: torch.Tensor
    right_rgb: torch.Tensor
    depth_m: torch.Tensor
    depth_valid: torch.Tensor
    depth_confidence: torch.Tensor
    points_world: torch.Tensor
    point_valid: torch.Tensor
    proprio: torch.Tensor
    dt: torch.Tensor
    image_age: torch.Tensor
    image_valid: torch.Tensor
    points_left_palm: torch.Tensor = field(default=None,kw_only=True)
    points_right_palm: torch.Tensor = field(default=None,kw_only=True)
    point_valid_left: torch.Tensor = field(default=None,kw_only=True)
    point_valid_right: torch.Tensor = field(default=None,kw_only=True)
    point_confidence: torch.Tensor = field(default=None,kw_only=True)

    def __post_init__(self):
        try:_populate_palms(self)
        except (RuntimeError,IndexError,AttributeError) as error:
            raise ValueError('invalid sensor shape while deriving palm coordinates') from error
        self.validate()

    def validate(self):
        _validate(self.as_dict())
        return self


@dataclass(slots=True)
class StereoHistoryBatch(SensorBatch):
    history_valid: torch.Tensor

    def validate(self):
        _validate(self.as_dict(),history_valid=self.history_valid)
        return self

    @classmethod
    def cat(cls, batches):
        batches = list(batches)
        if not batches or any(type(batch) is not cls for batch in batches):
            raise TypeError('nonempty StereoHistoryBatch list required')
        return cls(**{f.name:torch.cat([getattr(b,f.name) for b in batches],0) for f in fields(cls)})


class StereoHistory:
    def __init__(self, num_envs, history=4, device='cpu'):
        if type(num_envs) is not int or num_envs<1 or type(history) is not int or history<1:
            raise ValueError('positive integer batch and history required')
        self.num_envs,self.history,self.device = num_envs,history,torch.device(device)
        self._values = None
        self._valid = torch.zeros(num_envs,history,device=self.device,dtype=torch.bool)

    def reset(self, mask):
        if not isinstance(mask,torch.Tensor) or mask.shape!=(self.num_envs,) or mask.dtype!=torch.bool:
            raise ValueError('reset mask must be boolean [B]')
        mask = mask.to(self.device)
        self._valid[mask] = False
        if self._values is not None:
            for value in self._values.values(): value[mask] = 0

    @torch.no_grad()
    def update(self, sensor, reset_mask=None):
        if type(sensor) is not SensorBatch:
            raise TypeError('only SensorBatch is accepted; no privileged/task fields')
        sensor.validate()
        if len(sensor)!=self.num_envs or sensor.left_rgb.device!=self.device:
            raise ValueError('sensor batch/device differs from history')
        if self._values is None:
            self._values={name:torch.zeros(self.num_envs,self.history,*value.shape[1:],
                dtype=value.dtype,device=self.device) for name,value in sensor.as_dict().items()}
        if reset_mask is not None: self.reset(reset_mask)
        for name,value in sensor.as_dict().items():
            buffer=self._values[name]
            if buffer.shape[2:]!=value.shape[1:]: raise ValueError('sensor shape changed within history')
            buffer[:,:-1] = buffer[:,1:].clone(); buffer[:,-1] = value
        self._valid[:,:-1]=self._valid[:,1:].clone(); self._valid[:,-1]=True
        return StereoHistoryBatch(**{name:value.clone() for name,value in self._values.items()},
            history_valid=self._valid.clone())
