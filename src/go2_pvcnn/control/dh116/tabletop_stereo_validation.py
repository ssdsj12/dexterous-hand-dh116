"""Fail-closed RGB stereo schema, timing and identity validation; no native imports."""
import copy
import math
from numbers import Integral
import torch


def validate_images(left,right,calibration):
    if (not isinstance(left,torch.Tensor) or not isinstance(right,torch.Tensor)
            or left.shape!=right.shape or left.ndim!=4 or left.shape[1:]!=(3,calibration.height,calibration.width)
            or left.shape[0]<1 or not left.is_floating_point() or not right.is_floating_point()
            or left.dtype!=right.dtype or left.device!=right.device):
        raise ValueError('RGB stereo requires matching float [B,3,H,W] tensors')
    for image in (left,right):
        if not torch.isfinite(image).all() or (image<0).any() or (image>1).any():
            raise ValueError('RGB stereo must be finite in [0,1]')


def validate_identity(identity,batch_size,timestamp):
    """Caller supplies pre-action identity; no task or simulator truth is queried."""
    required={'run_uuid','env_id','episode_uid','reset_id','policy_tick','physics_tick_start','physics_tick_end','simulation_timestamp'}
    rows=identity if isinstance(identity,list) else [identity]
    if not rows or len(rows)!=batch_size:raise ValueError('identity batch mismatch; supply per-environment records')
    seen=set()
    for row in rows:
        if not isinstance(row,dict) or not required<=row.keys():raise ValueError('missing stereo identity')
        for key in ('env_id','reset_id','policy_tick','physics_tick_start','physics_tick_end'):
            if isinstance(row[key],bool) or not isinstance(row[key],Integral) or row[key]<0:
                raise ValueError('invalid integer stereo identity: '+key)
        if not isinstance(row['run_uuid'],str) or not row['run_uuid'] or not isinstance(row['episode_uid'],str) or not row['episode_uid']:
            raise ValueError('invalid run/episode stereo identity')
        if row['env_id'] in seen:raise ValueError('duplicate environment stereo identity')
        seen.add(row['env_id'])
        if (not row['run_uuid'] or not math.isfinite(float(row['simulation_timestamp']))
                or abs(float(row['simulation_timestamp'])-timestamp)>1e-6
                or int(row['physics_tick_start'])<0 or int(row['physics_tick_end'])<int(row['physics_tick_start'])
                or int(row['policy_tick'])<0):raise ValueError('stereo identity/time mismatch')
    return copy.deepcopy(identity)


def validate_pair_timing(packet,last_frame=-1):
    l,r=packet['left_frame_id'],packet['right_frame_id']
    a,b=float(packet['left_simulation_time']),float(packet['right_simulation_time'])
    if (type(l) is not int or type(r) is not int or l!=r or l<0
            or not math.isfinite(a) or a<0 or a!=b):raise ValueError('stereo frame/timestamp mismatch')
    if l<=last_frame:raise RuntimeError('stale stereo frame; fresh pair required')
    return l,a


def depth_metrics(estimate,reference,valid,reference_valid):
    """Evaluation only: reference must have separate source qualification/provenance."""
    if estimate.shape!=reference.shape or valid.shape!=estimate.shape or reference_valid.shape!=estimate.shape:
        raise ValueError('depth metric shapes must match')
    mask=valid.bool()&reference_valid.bool()&torch.isfinite(estimate)&torch.isfinite(reference)&(reference>0)
    if not mask.any():return dict(mae_m=None,rmse_m=None,count=0,coverage=0.)
    error=(estimate-reference)[mask]
    return dict(mae_m=float(error.abs().mean()),rmse_m=float(error.square().mean().sqrt()),
                count=int(mask.sum()),coverage=float(mask.float().mean()))
