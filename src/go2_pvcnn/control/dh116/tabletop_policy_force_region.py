"""Optional measured per-hand force servo and below-axis contact shaping.

No simulator imports. Caller owns fixed stiffness, actuator guards, checkpoint
compatibility checks and close/non-held selection. Measurements are never edited.
"""
import math
import torch
from .tabletop_grasp_acquisition import potential as geometry_potential


CONTRACT = dict(version='policy_force_region_v1',
    action='20D_joint12_wrist6_force2',force_range_n=[2.,12.],
    force_units='per_hand_measured_object_contact_load_N; not_world_Fz',
    stiffness=10.,stiffness_units='Nm/rad',
    checkpoint='incompatible_with_20D_stiffness_policy; explicit_contract_validation_required_no_silent_restore',
    contact_target_z_m=0.,contact_region='measured_valid_and_all_below_centerline',
    correction_gain_rad_n_s=.1,correction_rate_rad_s=.03,correction_budget_rad=.08,
    mimic_limit_rad=.012,contact_load_min_n=.05,
    update='physics_dt; only_live_close_nonheld_rows; caller_applies_final_guards',
    reset='task_selected_reset_from_seed; enable_initializes_all_rows',
    observation_size=6,
    observation_fields=('contact_radial_z2_div_.008','contact_valid2','contact_all_below_centerline2'),
    radial_z='world_Z_component_after_removing_projection_on_rotated_object_local_Y_axis',
    quaternion_order='wxyz',invalid_contact_observation=0.,
    nonfinite='raise_ValueError; invalid_contact_centroid_may_be_nan',
    contact_region_weights=dict(below=2.,above=.5,invalid=0.),
    contact_weight_use='per_hand_normal_Fz_positive_reward_only; actual_negative_Fz_penalty_unweighted; force_servo_and_grip_still_require_below',
    potential='zero_load_geometry*mean(exp(-abs(pad_radial_z+.002)/.005))+20*min(clamp(load,0,1)*contact_weights)',
    no_measurement_fabrication=True,no_physics_change=True)


def _finite(value,name):
    if not isinstance(value,torch.Tensor) or not torch.isfinite(value).all():
        raise ValueError(name+' must be finite tensor')
    return value


def _radial_z(points,quat):
    _finite(points,'points'); _finite(quat,'object_quat')
    scale=quat.abs().amax(-1,keepdim=True)
    if (scale==0).any():
        raise ValueError('object_quat must have finite nonzero norm')
    q=quat/scale
    q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True)
    w,x,y,z=q.unbind(-1)
    axis=torch.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),-1)
    along=(points*axis[:,None,:]).sum(-1)
    return points[:,:,2]-along*axis[:,None,2]


def enable(task):
    """Initialize episode state and one-row seeds consumed by task.reset(ids)."""
    task.force_region=True
    task.contact_target_z_m=torch.zeros_like(task.f_target)
    task.f_target.fill_(2.)
    task.f_requested.fill_(2.)
    task.f_offset.zero_()
    for name in ('force_tracking','contact_region'):
        task.terms[name]=task.f_target.new_zeros(task.f_target.shape[0])
    for name in ('contact_target_z_m','f_target','f_requested','f_offset'):
        task._seed[name]=getattr(task,name)[0].clone()


def requested_force(actions):
    """Last two actions are per-hand contact-load requests, not stiffness."""
    _finite(actions,'actions')
    if actions.ndim!=2 or actions.shape[1]!=20:
        raise ValueError('actions must have shape (N,20)')
    return 2.+5.*(actions[:,-2:].clamp(-1.,1.)+1.)


def adjust(task,target,requested,live,dt):
    """Add bounded integral closing-axis offsets to raw (N,12) joint targets."""
    if not math.isfinite(float(dt)) or dt<=0:
        raise ValueError('dt must be finite positive seconds')
    loads=task.m['hand_object_loads']
    for name,value in (('target',target),('requested',requested),('f_offset',task.f_offset),
                       ('loads',loads),('direction',task.direction),
                       ('mimic_errors',task.m['mimic_errors'])):
        _finite(value,name)
    requested=requested.to(task.f_target).clamp(2.,12.)
    if getattr(task,'manual_fast_force',False):
        from .tabletop_manual_fast_force import adjust as manual_adjust
        return manual_adjust(task,target,requested,live,dt)
    valid=task.m['contact_valid'] & (loads>.05)
    below=task.m['contact_all_below_centerline']
    severe=task.m['mimic_errors'].reshape(-1,2,5).abs().amax(-1)>=.012
    change=(.1*(requested-loads)*dt).clamp(-.03*dt,.03*dt)
    change=torch.where((change>0)&~(valid&below&~severe),0.,change)
    offset=torch.where(valid,(task.f_offset+change).clamp(-.08,.08),0.)
    task.f_offset[live]=offset.to(task.f_offset)[live]
    task.f_target[live]=requested[live]
    task.f_requested[live]=requested[live]
    task.f_filtered[live]=loads.to(task.f_filtered)[live]
    out=target.clone()
    adjusted=target+(task.f_offset[:,:,None]*task.direction).flatten(1)
    out[live]=adjusted.to(target)[live]
    return out


def observe(task):
    """Measured radial centroid Z (two), validity (two), below-axis flags (two)."""
    m=task.m
    valid=m['contact_valid']
    centers=torch.where(valid[:,:,None],m['contact_center_relative_m'],0.)
    radial=_radial_z(centers,m['object_quat'])/.008
    return torch.cat((radial,valid.to(radial.dtype),
                      m['contact_all_below_centerline'].to(radial.dtype)),1)


def force_ready(task,loads):
    """Bilateral actual load in the inclusive 2--12 N band; no synthetic loads."""
    return (torch.isfinite(loads)&(loads>=2.)&(loads<=12.)).all(1)


def contact_weights(m):
    """Return (N,2) weights: valid below 2, valid above .5, invalid 0.

    This helper weights reward only; it does not relax contact admission or the
    positive force servo gate. Negative normal-force penalties stay unweighted.
    """
    loads=m['hand_object_loads']
    weights=torch.where(m['contact_all_below_centerline'],
                        torch.full_like(loads,2.),torch.full_like(loads,.5))
    return torch.where(m['contact_valid'],weights,0.)


def potential(m,radius,length):
    """Lower radial geometry shaping plus region-weighted bilateral load."""
    loads=_finite(m['hand_object_loads'],'hand_object_loads')
    zero_load=dict(m,hand_object_loads=torch.zeros_like(loads))
    geometry=geometry_potential(zero_load,radius,length)
    radial=_radial_z(m['pad_relative'],m['object_quat'])
    lower_proximity=torch.exp(-(radial+.002).abs()/.005).mean(1)
    bilateral=(loads.clamp(0.,1.)*contact_weights(m)).amin(1)
    return geometry*lower_proximity+20.*bilateral
