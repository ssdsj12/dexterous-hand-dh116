"""Opt-in recovery boundaries around original native guards, never physics edits.

The callback can only shorten already-governed finger target increments. Actual
submission/measurement accounting belongs to the runtime, not this function.
"""
from __future__ import annotations

import math

import torch


def apply_post_guard_recovery(task, governed, live, dt_s, requested_action24):
    runtime=getattr(task,'tripod_recovery_runtime',None)
    if runtime is None:
        return governed
    command=task.command
    if (not isinstance(command,torch.Tensor) or command.ndim!=2 or command.shape[1]!=12
            or not command.is_floating_point()):
        raise ValueError('canonical floating command [N,12] required')
    n=len(command)
    def tensor(value,shape,dtype,name):
        if (not isinstance(value,torch.Tensor) or value.shape!=shape
                or value.dtype!=dtype or value.device!=command.device):
            raise ValueError(name+' shape/dtype/device mismatch')
        return value
    tensor(governed,command.shape,command.dtype,'governed target')
    tensor(live,(n,),torch.bool,'live')
    requested_mask=tensor(runtime.finger_mask,command.shape,torch.bool,'finger scope')
    stage=getattr(task,'tripod_recovery_stage',None)
    world=getattr(task,'free_wrist_world',None)
    styles=getattr(world,'mixed_reset_style_ids',None)
    if (stage not in ('A','B','C','D') or not isinstance(styles,torch.Tensor)
            or styles.shape!=(n,) or styles.device!=command.device
            or styles.dtype!=torch.long or bool(((styles!=0)&(styles!=1)).any())):
        raise ValueError('verified native mixed style and recovery stage required for scope')
    authority=torch.zeros_like(requested_mask)
    authority[:,6:10]=True
    if stage!='A':authority[:,:4]=True
    authority &= (styles==1)[:,None]
    if bool((requested_mask&~authority).any()):
        raise ValueError('runtime finger scope exceeds native Tripod/stage authority')
    loads=task.m.get('effective_pad_loads_n')
    tensor(loads,(n,2,5),command.dtype,'native pad loads')
    # Acquisition may not cancel ANY existing bearing-pad pressure command,
    # including an original mandatory overload release. Unknown loads are
    # equally unavailable for new automatic acquisition, rather than empty.
    protected=(~torch.isfinite(loads)) | (loads<0) | (loads>.05)
    protected=protected[:,:,[0,0,1,2,3,4]].reshape(n,12)
    mask=requested_mask & authority & live[:,None] & ~protected
    q=tensor(task.m['q'],command.shape,command.dtype,'measured q')
    if (type(dt_s) not in (int,float) or not math.isfinite(dt_s) or dt_s<=0
            or not isinstance(requested_action24,torch.Tensor)
            or requested_action24.shape!=(n,24) or not requested_action24.is_floating_point()
            or requested_action24.device!=command.device):
        raise ValueError('finite dt and canonical requested action [N,24] required')
    for value in (command,governed,q,requested_action24):
        if not bool(torch.isfinite(value).all()):
            raise ValueError('finite native-boundary inputs required')
    packet=dict(command_rad=command.detach().clone(),measured_q_rad=q.detach().clone(),
        governed_target_rad=governed.detach().clone(),
        lower_rad=task.low.detach().clone(),upper_rad=task.high.detach().clone(),
        requested_action24=requested_action24.detach().clone(),
        live=live.detach().clone(),finger_mask=mask.detach().clone(),
        pressure_protected=protected.detach().clone(),dt_s=float(dt_s))
    result=runtime.filter_post_guard_targets(packet)
    tensor(result,command.shape,command.dtype,'recovery target')
    if not bool(torch.isfinite(result).all()):
        raise ValueError('nonfinite recovery target')
    if not torch.equal(result[~mask],governed[~mask]):
        raise ValueError('recovery target escaped enabled live finger scope')
    # Every component stays on the already-governed old→new segment. This
    # cannot add velocity, reverse pressure relief or enlarge an original step.
    lower=command.minimum(governed);upper=command.maximum(governed)
    if not bool(((result>=lower)&(result<=upper))[mask].all()):
        raise ValueError('recovery target bypassed original governed increment')
    low=task.low.maximum(q-.1);high=task.high.minimum(q+.1)
    if not bool(((result>=low)&(result<=high))[mask].all()):
        raise ValueError('recovery target violates original joint/lead bounds')
    return result


def cancel_queued_common_ascent(world,task,pause):
    """Cancel queued Z drive goals, without changing measured physical state.

    The original world transport/clearance guards still execute afterwards.
    Non-paused rows and horizontal/rotation targets are untouched.
    """
    palms=world.palms;target=task.palm_targets
    if (not isinstance(pause,torch.Tensor) or pause.dtype!=torch.bool
            or pause.shape!=(len(palms),) or pause.device!=palms.device):
        raise ValueError('matching boolean common-ascent pause mask required')
    if not bool(pause.any()):
        return
    measured=task.m.get('adaptive_palm_offset')
    if (not isinstance(measured,torch.Tensor) or measured.shape!=palms.shape
            or measured.dtype!=palms.dtype or measured.device!=palms.device
            or target.shape!=palms.shape or target.dtype!=palms.dtype
            or target.device!=palms.device or not bool(torch.isfinite(measured[pause]).all())):
        raise ValueError('measured palm offsets required to cancel queued ascent')
    palms[pause,:,2]=measured[pause,:,2]
    target[pause,:,2]=measured[pause,:,2]


def invalidate_native_contact_snapshot(world,ids):
    """Expire only reset environments; preserve already-owned pre-reset rows."""
    old=getattr(world,'tripod_contact_snapshot',None)
    if old is None or not ids.numel():
        return
    valid=old.get('valid_envs')
    if valid is None:
        valid=torch.full((world.n,),bool(old.get('complete',False)),
                         device=ids.device,dtype=torch.bool)
    else:
        valid=valid.detach().clone()
    valid[ids]=False
    world.tripod_contact_snapshot=dict(old,complete=False,valid_envs=valid,
        invalidated_by_physical_reset=True,reset_env_ids=ids.detach().clone())
