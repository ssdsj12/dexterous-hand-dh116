"""Measured-motion, bounded loaded-grasp governor; physics guards stay external."""
import math
import torch
from .linear_coupling import ACTIVE, MIMIC
from .tabletop_pressure_relief import pressure_limited_target
from .tabletop_velocity_limits import CLOSE_COMMAND_RATE_RAD_S

CONTRACT=dict(version='measured_motion_grip_recovery_v3',position_lead_rad=.008,
    coupling_budget_rad=.008,severe_coupling_rad=.012,retreat_below_policy_rad=.03,
    command_rate_rad_s=CLOSE_COMMAND_RATE_RAD_S,palm_compensation='measured_pad_midpoint_relative_to_palm',
    pressure_priority=True,physical_guards_changed=False,
    thumb_limit_margin_rad=.005,thumb_limit_response='release_opposing_index_pause_motion',
    retreat_scope='index_only',policy_bounds='request_only_mechanical_governor_has_priority',
    synchronized_release_z='highest_per_hand_safe_target_v1',contact_z_correction_units='meters_v1',
    lift_speed_m_s=.05,lift_recovery='accepted_close_grasp_recovery_servo_v5',
    held_pressure_reference='configured_max_force_n_capped_at_12')


def midpoint_coordinates(m):
    palms=m['adaptive_palm_offset']
    return (m['pad_relative'].reshape(-1,2,2,3).mean(2)+m['object_relative'][:,None]
            -palms+palms.mean(1,keepdim=True))


def control(t,actions,live,dt):
    m=t.m;q=m['q'];errors=m['mimic_errors'];loads=m['hand_object_loads']
    hold=t.lift_grasp_hold
    if not hasattr(t,'joint_requested_target'):
        t.joint_requested_target=t.command.clone()
        t.joint_governed_target=t.command.clone()
        t.force_tracking_limited=torch.zeros_like(t.f_blocked)
        t.force_guard_conflict=torch.zeros_like(t.f_blocked)
    # Production captures on the accepted transition. A task initialized
    # directly in lift captures existing references before its first new action.
    hold.capture(t,live&(t.phase==3),t.palm_targets)
    hold.recover(t,live,dt)
    lift_holding=hold.active&(t.phase==3)
    close_hold=t.close_grasp_hold
    close_holding=close_hold.active&(t.phase<3)[:,None]
    holding=lift_holding[:,None]|close_holding
    holding_joint=holding[:,:,None].expand(-1,-1,6).reshape(t.n,12)
    held_command=torch.where(lift_holding[:,None],hold.command,close_hold.command)
    held_force=torch.where(lift_holding[:,None],hold.force_target,close_hold.force_target)
    held_palm=torch.where(lift_holding[:,None,None],hold.palm_offset,close_hold.palm_offset)
    lead=CONTRACT['position_lead_rad'];budget=CONTRACT['coupling_budget_rad']
    thumb_ids=[1,7];index_ids=[2,8]
    thumb_limit=(q[:,thumb_ids]<t.low[thumb_ids]+CONTRACT['thumb_limit_margin_rad'])&(loads>.05)
    severe=(errors.abs().amax(1)>=CONTRACT['severe_coupling_rad'])|thumb_limit.any(1)
    coupling_ok=errors.abs().amax(1)<budget
    tracking=(t.command-q).abs().amax(1)<.03
    slipping=(t.slip>.002)&((t.slip-t.f_last_slip)/dt>.002)
    from .tabletop_tensor_task import constrain_palm_exploration
    policy_actions=constrain_palm_exploration(
        actions[:,12:18].reshape(t.n,2,3),t.phase,t.palm_exploration_axis)
    # Shared search height ends once either side has an accepted anchor.
    independent=close_holding.any(1)
    if t.palm_exploration_axis=='z_down':
        own_z=actions[:,12:18].reshape(t.n,2,3)[:,:,2].clamp(-1.,0.)
        policy_actions[:,:,2]=torch.where(independent[:,None],own_z,policy_actions[:,:,2])
    policy_palms=policy_actions*t.palm_limit
    shift=midpoint_coordinates(m)-t.control_pad_anchor
    raw_palm=policy_palms-shift
    palm=raw_palm.maximum(-t.palm_limit).minimum(t.palm_limit)
    # Initial search shares a bounded Z request. Once one side is accepted,
    # the unheld hand keeps its own exploration request.
    palm[:,:,2]=torch.where(independent[:,None],palm[:,:,2],palm[:,:,2].mean(dim=1,keepdim=True))
    if t.contact_target_z_m is not None:
        contact_valid=m['contact_valid'] & m['contact_all_below_centerline']
        contact_valid &= (t.phase<3)[:,None]
        centers=m['contact_center_relative_m'][:,:,2]
        targets=t.contact_target_z_m
        finite=contact_valid & torch.isfinite(centers) & torch.isfinite(targets)
        count=finite.sum(dim=1,keepdim=True)
        correction=torch.where(finite,targets-centers,torch.zeros_like(centers)).sum(dim=1,keepdim=True)
        correction=correction/count.clamp_min(1)
        correction=torch.where(count>0,correction,torch.zeros_like(correction))
        correction=correction.clamp(-getattr(t,'contact_z_step_m',.0005),
                                    getattr(t,'contact_z_step_m',.0005))
        palm[:,:,2]=(palm[:,:,2]+correction).clamp(
            -float(t.palm_limit[2]),float(t.palm_limit[2]))
        palm[:,:,2]=torch.where(independent[:,None],palm[:,:,2],palm[:,:,2].mean(dim=1,keepdim=True))
    palm=torch.where(holding[:,:,None],held_palm,palm)
    releasing=torch.zeros_like(loads,dtype=torch.bool)
    if hasattr(t,'palm_reposition'):
        if getattr(t,'force_direction_reposition',False):
            palm,releasing=t.palm_reposition.apply(
                palm,m['adaptive_palm_offset'],loads,t.phase,live,dt,
                m['vertical_hand_force_n'],clearance=m['hand_clearance'],held_hands=close_holding)
        else:
            palm,releasing=t.palm_reposition.apply(
                palm,m['adaptive_palm_offset'],loads,t.phase,live,dt,
                clearance=m['hand_clearance'],
                clearance_floor_m=getattr(t,'diagnostic_palm_descent_clearance_m',
                                         getattr(t,'palm_descent_clearance_m',.0021)),held_hands=close_holding)
        palm=palm.maximum(-t.palm_limit).minimum(t.palm_limit)
        # Each hand may still be holding while the other is already unloaded.
        # Averaging after release/clearance projection would lower the waiting
        # hand; the shared request must satisfy both hands' descent limits.
        selected=live&(t.phase==1)&t.palm_reposition.mask&~close_holding.any(1)
        shared_z=palm[:,:,2].amax(dim=1,keepdim=True)
        palm[:,:,2]=torch.where(selected[:,None],shared_z,palm[:,:,2])
        # Once neither hand is waiting for release, both moving goals must
        # describe that same safe height. Otherwise a separately clipped lower
        # goal can never be reached and the released hand times out. Do not
        # replace its pending goal with a temporary release hold.
        state=t.palm_reposition.state
        moving=(selected&~(state==1).any(1))[:,None]&(state==2)
        t.palm_reposition.goal[:,:,2]=torch.where(
            moving,shared_z,t.palm_reposition.goal[:,:,2])
    palm=torch.where(holding[:,:,None],held_palm,palm)
    t.palm_targets[live]=palm[live];t.palm_actions[live]=(palm/t.palm_limit)[live]
    t.palm_clipped[live]=(raw_palm.abs()>t.palm_limit)[live]
    maximum=t.parameters.max_force_n;deadband=t.parameters.deadband_n
    filtered=t.f_filtered+(1-math.exp(-dt/.05))*(loads-t.f_filtered)
    from .tabletop_tensor_task import POLICY_MIN_FORCE_N, POLICY_MAX_FORCE_N
    nominal=POLICY_MIN_FORCE_N+(actions[:,18:]+1)*.5*(POLICY_MAX_FORCE_N-POLICY_MIN_FORCE_N)
    boost=(t.f_boost+torch.where(((t.phase==3)&slipping)[:,None]&(loads>.02),.25*dt,0.)).clamp_min(0.)
    boost=torch.minimum(boost,(maximum-nominal).clamp_min(0.));requested=(nominal+boost).clamp(POLICY_MIN_FORCE_N,maximum).clamp_max(POLICY_MAX_FORCE_N)
    boost=torch.where(holding,t.f_boost,boost)
    requested=torch.where(holding,held_force,requested)
    rate=t.parameters.force_rate_n_s
    target_force=requested if rate is None else t.f_target+(requested-t.f_target).clamp(-rate*dt,rate*dt)
    target_force=torch.where(holding,held_force,target_force)
    error=target_force-filtered;error=torch.where(loads>target_force+deadband,target_force-loads,error)
    error=torch.where(error.abs()<=deadband,0.,error)
    blocked=(errors.reshape(t.n,2,5).abs().amax(2)>=budget)|((t.command-q).reshape(t.n,2,6).abs().amax(2)>=lead)
    blocked |= thumb_limit|releasing
    old_offset=t.f_offset.clone()
    force_movement=(.1*error).clamp(-.03,.03)*dt
    # Accepted contact is a retained grasp, not a bidirectional position servo.
    # Ordinary force overshoot must not open it; overload guards below remain.
    force_movement=torch.where(holding,force_movement.clamp_min(0.),force_movement)
    movement=torch.where(blocked&(force_movement>0),0.,force_movement)
    for name,value in (('filtered',filtered),('boost',boost),('requested',requested),('target',target_force),
                       ('offset',torch.where(holding,t.f_offset,(t.f_offset+movement).clamp(-.08,.08))),('blocked',blocked)):
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
    # Close's lifetime integral may already be saturated when lift is accepted.
    # Regulate from the last applied command; discarded requests never become
    # stored lift displacement. The fresh budget is relative to the accepted
    # applied grasp, and all mechanical governors below still have priority.
    held_request=t.command+(force_movement[:,:,None]*t.direction).flatten(1)
    held_step=t.command+(movement[:,:,None]*t.direction).flatten(1)
    radius=(.08*t.direction.abs()).flatten(1)
    # Recovery may already have moved a joint outside this feedback budget.
    # Limit new feedback motion without restoring an old, unsafe reference.
    feedback_lower=torch.minimum(held_command-radius,t.command)
    feedback_upper=torch.maximum(held_command+radius,t.command)
    held_target=held_step.maximum(feedback_lower).minimum(feedback_upper)
    target=torch.where(holding_joint,held_target,target)
    requested_joint=torch.where(holding_joint,held_request,target).clone()
    if hasattr(t,'palm_reposition'):
        if getattr(t,'force_direction_reposition',False):
            from .tabletop_force_direction_reposition import CONTRACT as REPOSITION
        else:
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
                floor=t.low[joint] if t.envelope is None else torch.maximum(t.low[joint],t.lower[joint]-REPOSITION.get('release_retreat_below_policy_rad',CONTRACT['retreat_below_policy_rad']))
                lower[:,joint]=torch.where(releasing[:,hand],torch.minimum(floor,t.command[:,joint]).clamp_min(t.low[joint]),lower[:,joint])
    target=target.maximum(q-lead).minimum(q+lead)
    for j,(_,master,ratio,_) in enumerate(MIMIC):
        for hand in range(2):
            k=6*hand+ACTIVE.index(master);err=errors[:,5*hand+j]
            center=q[:,k]+err/ratio
            permitted=target[:,k].maximum(center-budget/ratio).minimum(center+budget/ratio)
            recovery=torch.where(err.abs()>=CONTRACT['severe_coupling_rad'],center,permitted)
            target[:,k]=recovery.maximum(q[:,k]-lead).minimum(q[:,k]+lead)
    # A retained target is a force setpoint, not an overload ceiling. Normal
    # overshoot must not dismantle an accepted grasp while it is supporting
    # the object. Preserve the existing release band at the policy ceiling.
    pressure_reference=torch.where(holding,torch.full_like(target_force,min(maximum,POLICY_MAX_FORCE_N)),target_force)
    target,over=pressure_limited_target(t.command.reshape(t.n,2,6),target.reshape(t.n,2,6),
        loads,pressure_reference,t.direction,dt,deadband)
    target=target.flatten(1)
    conflict=torch.zeros_like(blocked)
    conflict_joint=torch.zeros_like(q,dtype=torch.bool)
    for j,(_,master,_,_) in enumerate(MIMIC):
        for hand in range(2):
            k=6*hand+ACTIVE.index(master);err=errors[:,5*hand+j]
            # Overload release and follower recovery can require opposite
            # master motion. A neutral measured-q request adds no position bias
            # in either direction; slew still applies to a stale command.
            incompatible=over[:,hand]&((target[:,k]-t.command[:,k])*err<0)&(err.abs()>=budget)
            target[:,k]=torch.where(incompatible,q[:,k],target[:,k])
            conflict[:,hand] |= incompatible
            conflict_joint[:,k] |= incompatible
    for hand,(thumb,index) in enumerate(zip(thumb_ids,index_ids)):
        target[:,index]=torch.where(thumb_limit[:,hand],torch.minimum(target[:,index],t.command[:,index]-.05*dt),target[:,index])
        # Hold the previous thumb request subject to the measured lead governor
        # below. Reducing an old q+.02 request to q+.008 retains a closing
        # position bias; preserving stale overshoot would defeat that governor.
        # Slew and damping mean this is not a guarantee of actual closing force.
        target[:,thumb]=torch.where(thumb_limit[:,hand],torch.maximum(target[:,thumb],t.command[:,thumb]),target[:,thumb])
    # The policy floor is not a mechanical stop. Measured loaded joints may
    # fall below it; do not override the lead governor by commanding that floor.
    # Safe recovery can also put the accepted grasp outside the policy envelope.
    # Retain that command while mechanical bounds and all governors still apply.
    lower=torch.where(holding_joint,t.low,lower)
    upper=torch.where(holding_joint,t.high,upper)
    governed_lower=torch.minimum(lower,q+lead).maximum(t.low)
    governed_upper=torch.maximum(upper,q-lead).minimum(t.high)
    target=target.maximum(q-lead).minimum(q+lead)
    target=target.maximum(governed_lower).minimum(governed_upper)
    # Neutral conflict resolution outranks stale thumb hold/policy floors.
    target=torch.where(conflict_joint,q.maximum(t.low).minimum(t.high),target)
    # Soft lead/mimic projections constrain NEW tightening. They must not
    # retract an already accepted pose merely because loaded joints lag it.
    # Explicit overload, severe coupling and thumb-limit recovery still win.
    retain=holding&~over&~conflict&~thumb_limit
    retain &= errors.reshape(t.n,2,5).abs().amax(2)<CONTRACT['severe_coupling_rad']
    direction=t.direction.flatten(1)
    closing=(target-t.command)*direction.sign()
    retained=t.command+closing.clamp_min(0.)*direction.sign()
    target=torch.where(retain[:,:,None].expand(-1,-1,6).reshape(t.n,12),retained,target)
    target=target.maximum(t.low).minimum(t.high)
    t.held_safety_correction=holding&~retain&live[:,None]
    applied=t.command+(target-t.command).clamp(-CLOSE_COMMAND_RATE_RAD_S*dt,
                                               CLOSE_COMMAND_RATE_RAD_S*dt)
    missing=((requested_joint-applied).reshape(t.n,2,6)*t.direction*error.sign()[:,:,None])>1e-12
    limited=(missing.any(2)&(error!=0))|blocked|conflict
    # Close's scalar closure integral must not retain signed movement that
    # was discarded by a later joint/mimic/pressure/slew projection. This flag
    # is recomputed each tick, so available authority can be used immediately.
    unwind=live[:,None]&~holding&(movement!=0)&limited
    t.f_offset[unwind]=old_offset[unwind]
    t.f_blocked[live]=limited[live]
    t.force_tracking_limited[live]=limited[live]
    t.force_guard_conflict[live]=conflict[live]
    t.joint_requested_target[live]=requested_joint[live]
    t.joint_governed_target[live]=target[live]
    severe=severe|conflict.any(1)
    t.pressure_limited[live]=over[live];t.paused[live]=(severe|hold.recovering)[live]
    # Admission already checked both current hands. Continue the accepted lift
    # while force feedback operates, rather than repeating close every substep.
    lift=(t.phase==3)&~hold.recovering&~severe&tracking&~over.any(1)&live
    if t.shaping is not None:
        lifting=(t.phase==3)&live
        gates=torch.stack((lifting,lift&(t.offset<.05),lifting&severe,lifting&~tracking,
            torch.zeros_like(lifting),torch.zeros_like(lifting),lifting&over.any(1),lifting&(t.offset>=.05)),1)
        t.lift_diagnostic_masks[live]=gates[live]
    t.offset[lift]=(t.offset[lift]+CONTRACT['lift_speed_m_s']*dt).clamp_max(.05)
    t.command[live]=applied[live]
    return tracking&~severe&live&~over.any(1)
