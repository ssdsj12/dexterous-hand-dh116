"""Measured physical lift hold; force tracking error is a diagnostic objective."""
import torch


def hold_window_step(duration,anchor,latched,relative_samples,physical_ok,dt,*,slip_tolerance_m=.0005):
    """Update a continuous stable-lift window from ALL raw substep positions.

    relative_samples: (substeps,environment,hand,rod-point,xyz). Caller checks
    actual height, effective opposed pads, native depth and invalid/reset state.
    Once holding begins its anchor stays fixed. A slip breaks the window and
    uses the last raw pose for a future window; no duration survives that slip.
    Settling during lifting cannot permanently veto later genuine stability.
    """
    start=torch.where(latched[:,None,None,None],anchor,relative_samples[0])
    drift=(relative_samples-start[None]).norm(dim=-1).amax((0,2,3))
    restart=physical_ok&(drift>slip_tolerance_m)
    good=physical_ok&~restart
    duration=torch.where(good,torch.where(latched,duration,0.)+dt,0.)
    next_anchor=torch.where((~physical_ok|restart)[:,None,None,None],relative_samples[-1],start)
    return duration,next_anchor,physical_ok.clone(),dict(good=good,restart=restart,drift_m=drift)
