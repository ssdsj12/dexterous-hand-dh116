"""Stateless phase-specific upward force rewards from actual hand contacts.

normal is signed world-up normal-only hand force (no friction). lift_force is
signed world-up normal plus friction. Neither input includes table reactions,
gravity compensation or sensor clipping. Caller supplies these measured forces.
"""
import math
import torch


CONTRACT = dict(
    version='tabletop_phase_force_reward_v1',
    observation_size=0, stateless=True,
    force_units='N; signed_world_up_actual_hand_contact_forces',
    normal_excludes_friction=True,
    lift_force_includes_friction=True,
    excluded_sources=('table_reaction','gravity_compensation','sensor_clipping'),
    weight_units='N; actual_object_weight; positive_finite',
    grasp_phases=(1,2), lift_phases=(3,),
    grasp_rate_points_s=2., lift_rate_points_s=5.,
    grasp_load_band_n='per_hand > .05 and <= 12',
    lift_load_band_n='per_hand >= 2 and <= 12',
    contact_region='below_axis_only_for_these_terms; existing_above_.5_below_2_contact_reward_unchanged',
    force_target='per_hand_2_to_12N_normal_clamping_target_excludes_friction; unchanged',
    close_locked_pi='unchanged; reward_does_not_write_controller_state',
    normalization='clamp_positive_force_divided_by_half_object_weight_to_1; mean_over_both_hands',
    signed_gate='sum_of_both_signed_hand_forces_must_be_nonnegative_before_masking',
    invalid_terminal='valid_false_zero',
    unknown_phase='zero',
    invalid_numeric='ValueError_for_nonfinite_input_or_nonpositive_weight_or_dt',
    no_physics_change=True, no_measurement_fabrication=True)


def reward_terms(phase,normal,lift_force,weight,loads,below,valid,dt,*,phase_free=False,unrestricted_load=False):
    """Return two (N,) float64 rewards on the phase tensor's device.

    A single qualifying hand earns up to half the rate. The signed total gate
    includes both hands even if one hand fails the load or below-axis gate.
    """
    try:
        seconds=float(dt)
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError('dt must be finite positive seconds') from exc
    if not math.isfinite(seconds) or seconds<=0 or not math.isfinite(5.*seconds):
        raise ValueError('dt must be finite positive representable seconds')
    if not isinstance(phase,torch.Tensor) or phase.ndim!=1 or phase.dtype!=torch.long:
        raise ValueError('phase must be an (N,) long tensor')
    n=phase.shape[0]
    for name,value,shape in (
        ('normal',normal,(n,2)),('lift_force',lift_force,(n,2)),
        ('weight',weight,(n,)),('loads',loads,(n,2)),
        ('below',below,(n,2)),('valid',valid,(n,))):
        if not isinstance(value,torch.Tensor) or value.shape!=shape or value.device!=phase.device:
            raise ValueError(name+' has invalid shape or device')
        if name in ('below','valid'):
            if value.dtype!=torch.bool:
                raise ValueError(name+' must be boolean')
        elif value.is_complex() or value.dtype==torch.bool or not torch.isfinite(value).all():
            raise ValueError(name+' must be real finite numeric tensor')
    normal=normal.to(torch.float64)
    lift_force=lift_force.to(torch.float64)
    weight=weight.to(torch.float64)
    loads=loads.to(torch.float64)
    if (weight<=0).any():
        raise ValueError('weight must be positive finite Newtons')
    # Divide before multiplying by two: positive subnormal weights must not
    # underflow to a zero half-weight and produce 0/0 for a zero force.
    normal_support=(normal.clamp_min(0)/weight[:,None]*2.).clamp_max(1.)
    lift_support=(lift_force.clamp_min(0)/weight[:,None]*2.).clamp_max(1.)
    grasp_hands=below & (loads>.05)
    lift_hands=below & (loads>.05)
    if not unrestricted_load:
        grasp_hands &= loads<=12.
        lift_hands &= (loads>=2.) & (loads<=12.)
    grasp_live=valid & (torch.ones_like(valid) if phase_free else ((phase==1)|(phase==2))) & (normal.sum(1)>=0)
    lift_live=valid & (torch.ones_like(valid) if phase_free else (phase==3)) & (lift_force.sum(1)>=0)
    grasp=(normal_support*grasp_hands).mean(1)*(2.*seconds)
    lift=(lift_support*lift_hands).mean(1)*(5.*seconds)
    return dict(grasp_upward_force=torch.where(grasp_live,grasp,0.),
                lift_upward_force=torch.where(lift_live,lift,0.))
