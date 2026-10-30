"""Policy-owned finger, contact and wrist-pose exploration without a pose latch."""
import math
import torch
from .tabletop_velocity_limits import ACTIVE_VELOCITY_LIMIT_RAD_S

CONTRACT=dict(version='free_wrist_grasp_v1',action_size=26,
    action_order='joint_velocity12_world_translation_velocity6_intrinsic_XYZ_velocity6_normal_force2',
    joint_rate_rad_s=ACTIVE_VELOCITY_LIMIT_RAD_S,joint_target_lead_rad=.1,
    palm_lower_m=[-.03,-.03,-.004],palm_upper_m=[.03,.03,.05],
    wrist_rate_m_s=.01,rotation_rate_rad_s=.5,rotation_limit_rad=math.pi/2,
    joint_bounds='actual_affine_mimic_compatible_limits_not_reset_pose_envelope',
    posture_reference='reset_only;zero_action_holds_current_targets',
    normal_force_range_n=[2.,12.],friction_excluded=True,
    stage_flow=None,pose_latch=False,automatic_ascent=False,
    orientation='seed_world_quaternion_times_intrinsic_X_then_Y_then_Z',
    hardware_calibration_complete=False)
CONTRACT.update(finger_active_joints=12,finger_passive_joints=10,
    external_wrist_drives=12,normal_force_parameters=2,
    hand_body_dof_count=22,policy_command_size=26)
MEASUREMENT_KEYS=('wrist_pose','wrist_velocity','wrist_joint_angles')


def multiply(a,b):
    aw,ax,ay,az=a.unbind(-1);bw,bx,by,bz=b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,
        aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw),-1)


def compose_rotation(seed,angles):
    if seed.shape[:-1]!=angles.shape[:-1] or seed.shape[-1]!=4 or angles.shape[-1]!=3:
        raise ValueError('matching wxyz quaternion and XYZ angle batches required')
    if not torch.isfinite(seed).all() or not torch.isfinite(angles).all():
        raise ValueError('finite wrist rotation required')
    scale=seed.abs().amax(-1,keepdim=True)
    if (scale<=0).any():raise ValueError('nonzero wrist quaternion required')
    q=seed/scale;q=q/q.norm(dim=-1,keepdim=True)
    for axis in range(3):
        step=torch.zeros_like(q);step[...,0]=torch.cos(angles[...,axis]/2)
        step[...,axis+1]=torch.sin(angles[...,axis]/2)
        q=multiply(q,step)
    return q/q.norm(dim=-1,keepdim=True)


def enable(t,w):
    if not (getattr(t,'free_grasp_lift',False) and getattr(w,'continuous_motion',False)
            and getattr(w,'free_wrist_grasp',False)):
        raise ValueError('free wrist requires continuous free grasp/lift')
    if getattr(w.w,'mounts',None) is None or any(len(getattr(m,'angular_ids',()))!=3 for m in w.w.mounts):
        raise ValueError('complete six-axis physical wrist mounts required')
    t.free_wrist_grasp=True;t.continuous_motion=True;t.action_size=26
    t.free_wrist_world=w
    # Initial state is a reset seed, not the center of the allowed grasp range.
    t.lower=t.low.clone();t.upper=t.high.clone();t.center=(t.low+t.high)/2
    t.rotation_targets=torch.zeros((t.n,2,3),device=t.device,dtype=t.command.dtype)
    t.actions=torch.zeros((t.n,26),device=t.device,dtype=t.command.dtype)
    t.previous_actions=torch.zeros_like(t.actions)
    for name in ('rotation_targets','actions','previous_actions'):
        t._seed[name]=getattr(t,name)[0].clone()
    m=w.measure()
    for key in MEASUREMENT_KEYS:t.m[key]=m[key].clone()
    t.keys=tuple(t.keys)+MEASUREMENT_KEYS
    from .tabletop_dexhands_feedback import reward_rates
    for name in reward_rates(t.m,t.f_requested):t.terms[name]=torch.zeros_like(t.reward)
    t.last_observation=t.observe().clone()


def _limited_step(action,rate,dt):
    norm=action.norm(dim=-1,keepdim=True).clamp_min(1.)
    return action/norm*(rate*dt)


def control(t,actions,live,dt):
    if not math.isfinite(dt) or dt<=0:raise ValueError('positive finite physics dt required')
    if actions.shape!=(t.n,26):raise ValueError('expected N x 26 actions')
    actions=actions.to(t.command)
    from .linear_coupling import ACTIVE,MIMIC
    from .tabletop_pressure_relief import pressure_limited_target
    w=t.free_wrist_world
    target=t.command+actions[:,:12]*CONTRACT['joint_rate_rad_s']*dt
    # Bounded follower correction is superimposed on the selected trajectory;
    # a finite error never becomes a wrist-motion admission test.
    for j,(_,master,ratio,_) in enumerate(MIMIC):
        if getattr(t,'pinch_force_hold',False) and j<2:
            # The original passive drives still track their affine targets.
            # During explicit pinch/hold, do not undo the thumb/index force
            # controller's master target to chase finite tracking deflection.
            continue
        for h in range(2):
            k=6*h+ACTIVE.index(master);error=t.m['mimic_errors'][:,5*h+j]
            correction=(5.*error/ratio*dt).clamp(-ACTIVE_VELOCITY_LIMIT_RAD_S*dt,ACTIVE_VELOCITY_LIMIT_RAD_S*dt)
            target[:,k]+=torch.where(error.abs()>=.012,correction,0.)
    q=t.m['q'];lead=CONTRACT['joint_target_lead_rad']
    target=target.maximum(q-lead).minimum(q+lead).maximum(t.low).minimum(t.high)
    target=t.command+(target-t.command).clamp(-ACTIVE_VELOCITY_LIMIT_RAD_S*dt,ACTIVE_VELOCITY_LIMIT_RAD_S*dt)
    requested_force=2.+(actions[:,-2:]+1.)*5.
    ceiling=torch.full_like(requested_force,(12.-t.parameters.deadband_n)/1.25)
    applied,over=pressure_limited_target(t.command.reshape(t.n,2,6),target.reshape(t.n,2,6),
        t.m['hand_object_loads'],ceiling,t.direction,dt,t.parameters.deadband_n)
    applied=applied.flatten(1).maximum(t.low).minimum(t.high)
    t.command[live]=applied[live]
    lo=actions.new_tensor(CONTRACT['palm_lower_m']);hi=actions.new_tensor(CONTRACT['palm_upper_m'])
    palms=(w.palms+_limited_step(actions[:,12:18].reshape(t.n,2,3),.01,dt)).maximum(lo).minimum(hi)
    angles=(w.wrist_angles+_limited_step(actions[:,18:24].reshape(t.n,2,3),.5,dt)).clamp(-math.pi/2,math.pi/2)
    t.palm_targets[live]=palms[live];t.rotation_targets[live]=angles[live]
    t.palm_actions[live]=actions[:,12:18].reshape(t.n,2,3)[live]
    t.policy_stiffness[live]=10.
    t.joint_requested_target[live]=target[live];t.joint_governed_target[live]=applied[live]
    t.f_target[live]=requested_force[live];t.f_requested[live]=requested_force[live]
    t.f_filtered[live]=t.m['hand_object_loads'][live]
    t.f_offset[live]=0.;t.f_boost[live]=0.
    t.pressure_limited[live]=over[live];t.paused[live]=over.any(1)[live]
    t.f_blocked[live]=over.any(1)[live,None];t.palm_clipped[live]=False
    if hasattr(t,'force_guard_conflict'):t.force_guard_conflict[live]=False
    if t.shaping is not None:t.lift_diagnostic_masks=torch.zeros((t.n,8),device=t.device,dtype=torch.bool)
    return live&~over.any(1)


def observe(t):
    from .tabletop_dexhands_feedback import geometry_features
    m=t.m
    pose=m['wrist_pose'].clone()
    pose[:,:,:3]-=t.free_wrist_world.root_seed[:,:,:3]
    pose[:,:,:3]/=.05
    velocities=m['wrist_velocity']/m['q'].new_tensor([.05,.05,.05,1.,1.,1.])
    joints=2*(t.command-t.low)/(t.high-t.low)-1
    return torch.cat((pose.flatten(1),velocities.flatten(1),m['wrist_joint_angles'].flatten(1)/(math.pi/2),
        t.rotation_targets.flatten(1)/(math.pi/2),t.free_wrist_world.wrist_angles.flatten(1)/(math.pi/2),
        t.free_wrist_world.palms.flatten(1)/.05,joints,t.actions,geometry_features(m)),1)


def add_feedback(t,pieces,valid,dt):
    from .tabletop_dexhands_feedback import reward_rates
    pieces['lift']=torch.zeros_like(t.reward)
    for name,rate in reward_rates(t.m,t.f_requested,t.actions,t.previous_actions).items():
        pieces[name]=torch.where(valid&(t.failure==0),rate*dt,0.)


def apply_contract(contract,t):
    from .tabletop_palm_mount import ROTATION_CONTRACT
    from .tabletop_dexhands_feedback import CONTRACT as FEEDBACK
    contract.update(free_wrist_grasp=dict(CONTRACT),wrist_rotation=dict(ROTATION_CONTRACT),
        dexhands_feedback=dict(FEEDBACK),action='26D_free_fingers_worldXYZ_intrinsicXYZ_normal_force_v1',
        observation_size=t.observe().shape[-1],observation_version='dh116-free-wrist-v1',
        joint_envelope='full_actual_mimic_compatible_limits_reset_pose_not_a_constraint',
        palm_motion='policy_integrated_independent_XYZ_and_intrinsic_XYZ_all_episode',
        command_rate_rad_s=CONTRACT['joint_rate_rad_s'],palm_rate_m_s=CONTRACT['wrist_rate_m_s'],
        initial_force_action_mean=-.4,initial_force_target_n=5.,
        policy_control=dict(CONTRACT,automatic_force_correction=False,retained_pose_override=False,
            ordinary_contact_pause=False),
        grasp_controller=dict(CONTRACT),
        force_feedback=dict(enabled=False,target='policy_learned_normal_load_goal',
            measured_load_guaranteed=False,overload_release_retained=True),
        palm_exploration=dict(lower_m=CONTRACT['palm_lower_m'],upper_m=CONTRACT['palm_upper_m']))
    contract['palm_transport']=dict(contract.get('palm_transport',{}),
        motion='three_prismatic_and_three_revolute_drives',rotation=dict(ROTATION_CONTRACT))
    contract['force_region_control']=dict(version='free_wrist_load_features_rewards_v1',
        control=False,action_source='final_two_of_26_policy_commands',
        force_target_range_n=[2.,12.],friction_excluded=True,
        use='load_region_observations_signed_force_rewards_and_overload_release',
        fixed_pose_force_servo=False)
    if 'free_grasp_lift' in contract:
        contract['free_grasp_lift']=dict(contract['free_grasp_lift'],actions=CONTRACT['action_order']+';26D')
    contract.pop('grasp_shaping',None)
    return contract
