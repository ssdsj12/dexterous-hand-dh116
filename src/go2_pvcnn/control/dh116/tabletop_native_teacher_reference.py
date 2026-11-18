"""Reference-state helpers for the explicit native teacher, never PPO admission.

Geometric closure is not a force measurement. Native forces, depth, pose and
the qualification trace remain untouched by these reference updates.
"""
import torch


def closure_step(pressure,redistribution,missing,eligible,can_seek,slipping,depth,dt):
    # Increasing the whole grip during lag must not overpower the unloading
    # correction on its already stronger face. Close weak/missing faces first.
    seek=eligible&(missing|(slipping[:,None,None]&(redistribution>=0)))&can_seek[:,None,None]
    inward=redistribution+seek*.0004*dt
    # Retreat the final reference, including its initial preload. Never allow
    # closure or balancing integration to cancel a native-depth retreat.
    return (pressure+torch.where(depth>.0006,-.001*dt,inward)).clamp(-.0001,.0025)


def rise_step(rise,actual_lift,paired,acquired,speed,dt):
    room=(actual_lift.clamp_min(0)+.00075-rise).clamp_min(0)
    delta=torch.minimum(torch.full_like(rise,speed*dt),room)
    delta=torch.minimum(delta,(.031-rise).clamp_min(0))
    delta=torch.where(paired&acquired,delta,0.)
    return rise+delta,delta/dt


def anchor_step(anchor,latched,relative,acquired,good,reset):
    latched=latched&~reset
    new=acquired&good&~reset&~latched
    result=torch.where(new[:,None,None,None],relative,anchor)
    return result,latched|new


def ascent_floor(current_z,grip_z,relative_rise,up,acquired):
    requested=(4*(grip_z+relative_rise[:,None]-current_z)+up[:,None]).clamp(0,.006)
    return torch.where(acquired[:,None],requested,torch.full_like(current_z,-.008))


def force_tracking(loads,desired,eligible,hand_goal,tolerance,*,face_loads=None):
    hand_error=loads.sum(-1)-hand_goal
    digit_error=loads-desired
    valid=(hand_error.abs()<=tolerance+1e-12).all(-1)
    valid &= ((digit_error.abs()<=tolerance+1e-12)|~eligible).all((-1,-2))
    if face_loads is not None:
        valid &= ((face_loads-hand_goal/2).abs()<=tolerance+1e-12).all((-1,-2))
    return valid,hand_error,digit_error


def hybrid_velocity(error,normal,load,goal,depth,up,*,gain=.0004):
    """Native compression correction relative to full planned lift transport.

    Returns a motion request, never a fabricated measured or applied force.
    The caller still applies the original physical actuator limits.
    """
    n=normal/normal.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    tangent=4*(error-(error*n).sum(-1,keepdim=True)*n)
    tangent*=(.003/tangent.norm(dim=-1,keepdim=True).clamp_min(1e-12)).clamp_max(1.)
    vertical=torch.zeros_like(error);vertical[...,2]=up[:,None,None]
    inward=(gain*(goal-load)).clamp(-.001,.001)
    inward=torch.where(load<=.05,torch.full_like(inward,.0004),inward)
    inward=torch.where(depth>.0006,torch.full_like(inward,-.001),inward)
    return vertical+tangent+inward[...,None]*n


def reference_velocity_box(current,low_reference,high_reference,limits,dt):
    return ((low_reference-current)/dt).clamp(-limits,limits), ((high_reference-current)/dt).clamp(-limits,limits)


def scheduled_ascent(rise,acquired,speed,dt):
    delta=torch.minimum(torch.full_like(rise,speed*dt),(.031-rise).clamp_min(0))
    delta=torch.where(acquired,delta,0.)
    return rise+delta,delta/dt


def acquisition_dwell(stable,interval_good,interval_reset,dt):
    """A period spanning reset retains raw measurements but no new readiness."""
    return torch.where(interval_good&~interval_reset,stable+dt,0.)


def lift_motion_permission(force_ready,episode_age,explore_after_s):
    """Optional bounded exploration may move while force correction continues."""
    if explore_after_s is None:return force_ready
    return force_ready|(episode_age>=explore_after_s)


def interval_force_state(physics_loads,physics_vectors,desired,eligible,hand_goal,tolerance):
    """User-selected force accuracy: actual substep mean, without extra EMA."""
    from .tabletop_surface_force_sharing import surface_targets
    loads=physics_loads.mean(0);vectors=physics_vectors.mean(0)
    faces=surface_targets(loads,vectors,valid=loads>0)
    good,hand_error,digit_error=force_tracking(loads,desired,eligible,hand_goal,tolerance,
        face_loads=faces['face_loads'])
    good &= faces['paired'].all(-1)&faces['balanced'].all(-1)
    return dict(mean_loads=loads,mean_vectors=vectors,good=good,hand_error=hand_error,
        digit_error=digit_error,face_error=faces['face_loads']-hand_goal/2,
        paired=faces['paired'],balanced=faces['balanced'],face_loads=faces['face_loads'])
