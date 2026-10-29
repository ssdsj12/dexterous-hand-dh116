"""Independent manual thumb/index closing; unequal loads allowed inside a band."""
import torch

CONTRACT=dict(version='manual_digit_force_ratio_v1',max_measured_ratio=1.5,
    target_ratio_limit=1.4,split='preserve_measured_share_projected_inside_ratio_band',
    load='object_filtered_normal_magnitude_per_digit_N; friction_excluded',
    gain_rad_n_s=.1,rate_rad_s=.12,closure_budget_rad=.08,deadband_n=.03,
    mimic_pause_rad=.012,scope='explicit_manual_mode_only',
    mimic_closing_taper_start_rad=.008,mimic_closing_stop_rad=.011,
    mimic_release_start_rad=.0105,mimic_release_gain_s_inv=10.,
    ratio_success='both_digits_at_least_0.05N_and_max<=1.5*min',
    physical_force_or_ratio_guaranteed=False)


def enable(t):
    direction=torch.zeros((*t.direction.shape[:2],2,6),device=t.device,dtype=t.command.dtype)
    direction[:,:,0,:2]=t.direction[:,:,:2]
    direction[:,:,1,2]=t.direction[:,:,2]
    norm=direction.norm(dim=-1,keepdim=True)
    if not torch.isfinite(direction).all() or (norm<=1e-12).any():
        raise ValueError('finite nonzero thumb and index closing axes required')
    # Preserve authored per-joint velocity/budget. Normalizing each finger
    # separately amplifies the right thumb by >2x and changed the grasp pose.
    t.manual_digit_directions=direction
    t.manual_digit_offsets=torch.zeros((*t.f_target.shape,2),device=t.device,dtype=t.command.dtype)
    t.manual_digit_targets=torch.zeros_like(t.manual_digit_offsets)
    t.manual_digit_force_balance=True
    t.manual_smooth_mimic=True
    t.manual_mimic_limited=torch.zeros_like(t.manual_digit_offsets,dtype=torch.bool)
    t.manual_digit_contact_acquired=torch.zeros_like(t.manual_mimic_limited)
    for name in ('manual_digit_offsets','manual_digit_targets','manual_mimic_limited','manual_digit_contact_acquired'):
        t._seed[name]=getattr(t,name)[0].clone()


def adjust(t,target,requested,live,dt):
    grouped=t.m['grouped_object_loads']
    if grouped.shape!=(t.n,2,6) or not torch.isfinite(grouped).all() or (grouped<0).any():
        raise ValueError('finite nonnegative object-filtered per-digit loads required')
    loads=grouped[:,:,:2]
    pair=loads.sum(-1)
    fraction=torch.where(pair>1e-6,loads[:,:,0]/pair.clamp_min(1e-6),.5)
    low=1/(1+CONTRACT['target_ratio_limit'])
    fraction=fraction.clamp(low,1-low)
    available=(requested-grouped[:,:,2:].sum(-1)).clamp_min(0.)
    desired=available[:,:,None]*torch.stack((fraction,1-fraction),-1)
    # A weak digit at its micro-closing budget cannot catch a stronger one.
    # Back off its partner instead of falsely declaring the 1.5 band achieved.
    exhausted=t.manual_digit_offsets>=CONTRACT['closure_budget_rad']-1e-6
    partner_cap=torch.where(exhausted.flip(-1),loads.flip(-1)*CONTRACT['target_ratio_limit'],torch.inf)
    desired=torch.minimum(desired,partner_cap)
    error=desired-loads
    t.manual_digit_contact_acquired[live] |= (loads>=.05)[live]
    gain=getattr(t,'manual_digit_force_gain',CONTRACT['gain_rad_n_s'])
    change=(gain*error*dt).clamp(
        -CONTRACT['rate_rad_s']*dt,CONTRACT['rate_rad_s']*dt)
    # Only the initial no-contact search is fast. A brief unloaded sample after
    # acquisition must not switch back to a high-speed closing pulse.
    change=torch.where(~t.manual_digit_contact_acquired&(error>0),
                       torch.full_like(change,CONTRACT['rate_rad_s']*dt),change)
    change=torch.where(error.abs()<=CONTRACT['deadband_n'],0.,change)
    mimic=t.m['mimic_errors'].reshape(t.n,2,5)[:,:,:2].abs()
    closing=((CONTRACT['mimic_closing_stop_rad']-mimic)/
             (CONTRACT['mimic_closing_stop_rad']-CONTRACT['mimic_closing_taper_start_rad'])).clamp(0.,1.)
    change=torch.where(change>0,change*closing,change)
    # Integrate release in the same closure coordinates as force correction.
    # A thresholded q_master+error/ratio override caused an open/close relay.
    ratios=change.new_tensor([.88178789,.9206])
    master_direction=torch.stack((t.manual_digit_directions[:,:,0,1],t.manual_digit_directions[:,:,1,2]),-1).abs()
    release=-CONTRACT['mimic_release_gain_s_inv']*(mimic-CONTRACT['mimic_release_start_rad']).clamp_min(0.)*dt/(ratios*master_direction).clamp_min(1e-6)
    change=torch.where(mimic>CONTRACT['mimic_release_start_rad'],torch.minimum(change,release),change)
    change=change.clamp(-CONTRACT['rate_rad_s']*dt,CONTRACT['rate_rad_s']*dt)
    t.manual_mimic_limited[live]=(closing<1.)[live]
    offsets=(t.manual_digit_offsets+change).clamp(-CONTRACT['closure_budget_rad'],CONTRACT['closure_budget_rad'])
    t.manual_digit_offsets[live]=offsets[live]
    t.manual_digit_targets[live]=desired[live]
    # Keep legacy hand-level telemetry finite; it is no longer the control state.
    t.f_offset[live]=t.manual_digit_offsets.mean(-1)[live]
    t.f_target[live]=requested[live];t.f_requested[live]=requested[live]
    t.f_filtered[live]=t.m['hand_object_loads'][live]
    delta=(t.manual_digit_offsets[:,:,:,None]*t.manual_digit_directions).sum(2).flatten(1)
    result=target.clone();result[live]=(target+delta)[live]
    return result


def summary(t):
    loads=t.m['grouped_object_loads'][0,:,:2].detach().cpu().tolist()
    ratios=[max(pair)/min(pair) if min(pair)>=.05 else None for pair in loads]
    return dict(max_allowed_ratio=1.5,measured_load_n=loads,
        target_load_n=t.manual_digit_targets[0].detach().cpu().tolist(),
        closure_offset_rad=t.manual_digit_offsets[0].detach().cpu().tolist(),
        mimic_limited=t.manual_mimic_limited[0].detach().cpu().tolist(),
        measured_ratio=ratios,balanced=[r is not None and r<=1.5 for r in ratios])
