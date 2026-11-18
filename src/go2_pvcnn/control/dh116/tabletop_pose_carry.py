"""Explicit native teacher pose retention; never moves/attaches the object."""
import torch

def carry_latch(active,net_height,paired,reset):
    return (active|((net_height>=.003)&paired.all(-1)))&~reset


def prelift_retention(active,duration,loads,paired,eligible,reset,dt,*,
                      min_hand_n=2.,min_pad_n=.5,dwell_s=.2):
    """Teacher-only acquisition from measured pads, before moving the wrists.

    This selects a loaded native grasp; it does not establish lift success or
    restrict the PPO policy. Force precision/equal sharing are not admission
    tests, and an interval containing reset never contributes to the dwell.
    """
    finite=torch.isfinite(loads).all((-1,-2))
    selected=torch.where(eligible,loads,0.)
    good=(finite&paired.all(-1)&(selected.sum(-1)>=min_hand_n).all(-1)
          &((loads>=min_pad_n)|~eligible).all((-1,-2))
          &((loads<=.05)|eligible).all((-1,-2))&~reset)
    duration=torch.where(good,duration+dt,0.)
    return (active|(duration>=dwell_s-1e-12))&~reset,duration

def carry_action(action,active,net_height,lower,upper,speed,*,target_height=.0315):
    # Zero velocity retains the actual joint POSITION COMMAND, including its
    # existing motor preload. Native PD, physical mimic and motor caps remain.
    up=torch.where(net_height<target_height,torch.full_like(action[:,0],speed),0.)
    actual=up[:,None].expand(-1,2).maximum(lower[:,:,8]).minimum(upper[:,:,8])
    held=torch.zeros_like(action);held[:,[14,17]]=actual/.01
    return torch.where(active[:,None],held,action),up
