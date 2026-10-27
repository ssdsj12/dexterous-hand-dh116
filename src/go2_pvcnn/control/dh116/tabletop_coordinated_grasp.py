"""Measured-motion, bounded loaded-grasp governor; physics guards stay external."""
import math
import torch
from .linear_coupling import ACTIVE, MIMIC
from .tabletop_pressure_relief import pressure_limited_target
from .tabletop_velocity_limits import CLOSE_COMMAND_RATE_RAD_S

CONTRACT=dict(version='measured_motion_grasp_v2',position_lead_rad=.008,
    coupling_budget_rad=.008,severe_coupling_rad=.012,retreat_below_policy_rad=.03,
    command_rate_rad_s=CLOSE_COMMAND_RATE_RAD_S,palm_compensation='measured_pad_midpoint_relative_to_palm',
    pressure_priority=True,physical_guards_changed=False,
    thumb_limit_margin_rad=.005,thumb_limit_response='release_opposing_index_pause_motion',
    retreat_scope='index_only',policy_bounds='request_only_mechanical_governor_has_priority')


def midpoint_coordinates(m):
    palms=m['adaptive_palm_offset']
    return (m['pad_relative'].reshape(-1,2,2,3).mean(2)+m['object_relative'][:,None]
            -palms+palms.mean(1,keepdim=True))


def control(t,actions,live,dt):
    m=t.m;q=m['q'];errors=m['mimic_errors'];loads=m['hand_object_loads']
    lead=CONTRACT['position_lead_rad'];budget=CONTRACT['coupling_budget_rad']
    thumb_ids=[1,7];index_ids=[2,8]
    thumb_limit=(q[:,thumb_ids]<t.low[thumb_ids]+CONTRACT['thumb_limit_margin_rad'])&(loads>.05)
    severe=(errors.abs().amax(1)>=CONTRACT['severe_coupling_rad'])|thumb_limit.any(1)
    coupling_ok=errors.abs().amax(1)<budget
    tracking=(t.command-q).abs().amax(1)<.03
    slipping=(t.slip>.002)&((t.slip-t.f_last_slip)/dt>.002)
    policy_palms=actions[:,12:18].reshape(t.n,2,3)*t.palm_limit
    shift=midpoint_coordinates(m)-t.control_pad_anchor
    raw_palm=policy_palms-shift
    palm=raw_palm.maximum(-t.palm_limit).minimum(t.palm_limit)
    releasing=torch.zeros_like(loads,dtype=torch.bool)
    if hasattr(t,'palm_reposition'):
        if getattr(t,'force_direction_reposition',False):
            palm,releasing=t.palm_reposition.apply(
                palm,m['adaptive_palm_offset'],loads,t.phase,live,dt,
                m['vertical_hand_force_n'],clearance=m['hand_clearance'])
        else:
            palm,releasing=t.palm_reposition.apply(
                palm,m['adaptive_palm_offset'],loads,t.phase,live,dt,
                clearance=m['hand_clearance'])
        palm=palm.maximum(-t.palm_limit).minimum(t.palm_limit)
    t.palm_targets[live]=palm[live];t.palm_actions[live]=(palm/t.palm_limit)[live]
    t.palm_clipped[live]=(raw_palm.abs()>t.palm_limit)[live]
    maximum=t.parameters.max_force_n;deadband=t.parameters.deadband_n
    filtered=t.f_filtered+(1-math.exp(-dt/.05))*(loads-t.f_filtered)
    nominal=.1+(actions[:,18:]+1)*.5*(maximum-.1)
    boost=(t.f_boost+torch.where(((t.phase==3)&slipping)[:,None]&(loads>.02),.25*dt,0.)).clamp_min(0.)
    boost=torch.minimum(boost,maximum-nominal);requested=(nominal+boost).clamp(.1,maximum)
    rate=t.parameters.force_rate_n_s
    target_force=requested if rate is None else t.f_target+(requested-t.f_target).clamp(-rate*dt,rate*dt)
    error=target_force-filtered;error=torch.where(loads>target_force+deadband,target_force-loads,error)
    error=torch.where(error.abs()<=deadband,0.,error)
    blocked=(errors.reshape(t.n,2,5).abs().amax(2)>=budget)|((t.command-q).reshape(t.n,2,6).abs().amax(2)>=lead)
    blocked |= thumb_limit|releasing
    movement=(.1*error).clamp(-.03,.03)*dt;movement=torch.where(blocked&(movement>0),0.,movement)
    for name,value in (('filtered',filtered),('boost',boost),('requested',requested),('target',target_force),
                       ('offset',(t.f_offset+movement).clamp(-.08,.08)),('blocked',blocked)):
        getattr(t,'f_'+name)[live]=value[live]
    t.f_last_slip[live]=t.slip[live]
    if t.envelope is None:
        target=(t.reference+.08*actions[:,:12]).maximum(t.low).minimum(t.high)
        lower=t.low;upper=t.high
    else:
        target=t.center+actions[:,:12]*torch.where(actions[:,:12]>=0,t.upper-t.center,t.center-t.lower)
        lower=t.lower.clone();lower[index_ids]-=CONTRACT['retreat_below_policy_rad']
        lower=torch.maximum(t.low,lower);upper=t.upper
    target=target+(t.f_offset[:,:,None]*t.direction).flatten(1)
    if hasattr(t,'palm_reposition'):
        from .tabletop_loaded_palm_motion import CONTRACT as REPOSITION
        # These are requests only: mimic/pressure governors below retain priority.
        # Stop opening once unloaded. Holding all finger commands avoids
        # sinking the open pads toward the table while waiting/moving.
        target=torch.where(releasing[:,:,None].expand(-1,-1,6).reshape(t.n,12),t.command,target)
        lower=lower.expand(t.n,-1).clone()
        for hand in range(2):
            for joint in (6*hand+1,6*hand+2):
                opening=releasing[:,hand]&(loads[:,hand]>REPOSITION['released_load_n'])
                target[:,joint]=torch.where(opening,t.command[:,joint]-.05*dt,target[:,joint])
                floor=t.low[joint] if t.envelope is None else torch.maximum(t.low[joint],t.lower[joint]-REPOSITION['release_retreat_below_policy_rad'])
                lower[:,joint]=torch.where(releasing[:,hand],torch.minimum(floor,t.command[:,joint]).clamp_min(t.low[joint]),lower[:,joint])
    target=target.maximum(q-lead).minimum(q+lead)
    for j,(_,master,ratio,_) in enumerate(MIMIC):
        for hand in range(2):
            k=6*hand+ACTIVE.index(master);err=errors[:,5*hand+j]
            center=q[:,k]+err/ratio
            permitted=target[:,k].maximum(center-budget/ratio).minimum(center+budget/ratio)
            recovery=torch.where(err.abs()>=CONTRACT['severe_coupling_rad'],center,permitted)
            target[:,k]=recovery.maximum(q[:,k]-lead).minimum(q[:,k]+lead)
    target,over=pressure_limited_target(t.command.reshape(t.n,2,6),target.reshape(t.n,2,6),
        loads,target_force,t.direction,dt,deadband)
    target=target.flatten(1)
    for hand,(thumb,index) in enumerate(zip(thumb_ids,index_ids)):
        target[:,index]=torch.where(thumb_limit[:,hand],torch.minimum(target[:,index],t.command[:,index]-.05*dt),target[:,index])
        # Hold the previous thumb request subject to the measured lead governor
        # below. Reducing an old q+.02 request to q+.008 retains a closing
        # position bias; preserving stale overshoot would defeat that governor.
        # Slew and damping mean this is not a guarantee of actual closing force.
        target[:,thumb]=torch.where(thumb_limit[:,hand],torch.maximum(target[:,thumb],t.command[:,thumb]),target[:,thumb])
    # The policy floor is not a mechanical stop. Measured loaded joints may
    # fall below it; do not override the lead governor by commanding that floor.
    governed_lower=torch.minimum(lower,q+lead).maximum(t.low)
    governed_upper=torch.maximum(upper,q-lead).minimum(t.high)
    target=target.maximum(q-lead).minimum(q+lead)
    target=target.maximum(governed_lower).minimum(governed_upper)
    t.pressure_limited[live]=over[live];t.paused[live]=severe[live]
    lift=(t.phase==3)&coupling_ok&~severe&tracking&t.grip_force_ready(loads,lifting=True)&~slipping&~over.any(1)&live
    if t.shaping is not None:
        lifting=(t.phase==3)&live
        gates=torch.stack((lifting,lift&(t.offset<.05),lifting&(~coupling_ok|severe),lifting&~tracking,
            lifting&~t.grip_force_ready(loads,lifting=True),lifting&slipping,lifting&over.any(1),lifting&(t.offset>=.05)),1)
        t.lift_diagnostic_masks[live]=gates[live]
    t.offset[lift]=(t.offset[lift]+.01*dt).clamp_max(.05)
    t.command[live]+=(target-t.command).clamp(-CLOSE_COMMAND_RATE_RAD_S*dt,
        CLOSE_COMMAND_RATE_RAD_S*dt)[live]
    return tracking&~severe&live&~over.any(1)
