"""Policy-owned joint/wrist targets with actuator bounds and hard guards."""
import torch
from .linear_coupling import ACTIVE, MIMIC
from .tabletop_velocity_limits import ACTIVE_VELOCITY_LIMIT_RAD_S
from .tabletop_palm_mount import CONTRACT as WRIST
from .tabletop_pressure_relief import pressure_limited_target

CONTRACT = dict(version='policy_owned_grasp_v2',
    joint_rate_rad_s=ACTIVE_VELOCITY_LIMIT_RAD_S,
    wrist_rate_m_s=WRIST['velocity_limit_m_s'],
    wrist_rate_metric='cartesian_vector_norm',
    palm_lower_m=[-.003, -.003, -.004], palm_upper_m=[.003, .003, .05],
    severe_coupling_rad=.012, stiffness_min=10., palm_descent_clearance_m=.0003,
    automatic_force_correction=False, automatic_ascent=False,
    retained_pose_override=False, ordinary_contact_pause=False,
    hard_guards='mechanical_bounds_severe_mimic_overload_table_clearance_and_runtime_faults',
    qualification='offline_candidate_not_physics_validated')


def validate_configuration(args):
    if (not args.tensor_pipeline or args.palm_transport != 'driven_carriage'
            or not args.pressure_relief or not args.skip_hold
            or args.grasp_readiness != 'measured_load'
            or args.palm_exploration_axis != 'xyz' or args.reset_height_jitter_m != 0):
        raise ValueError('policy controller requires tensor driven carriage, pressure relief, '
                         'measured-load direct lift, xyz actions and zero reset jitter')
    if getattr(args,'teacher_search_cases',None) and not (
            getattr(args,'force_region_control',False) and getattr(args,'policy_lift_hold',False)):
        raise ValueError('policy teacher requires force-region control and latched lift hold')
    for name in ('release_reposition', 'force_direction_reposition', 'contact_effort_feedback',
                 'control_probe_checkpoint', 'play_checkpoint'):
        if getattr(args, name, None):
            raise ValueError('policy controller is incompatible with '+name)


def enable(task):
    if any(hasattr(task, name) for name in ('coordinated_control', 'palm_reposition', 'contact_effort')):
        raise ValueError('policy control cannot be combined with scripted grasp controllers')
    task.policy_control = True
    task.palm_descent_clearance_m = CONTRACT['palm_descent_clearance_m']
    task.policy_stiffness = torch.full_like(task.f_target, CONTRACT['stiffness_min'])
    task._seed['policy_stiffness'] = task.policy_stiffness[0].clone()
    task.joint_requested_target = task.command.clone()
    task.joint_governed_target = task.command.clone()
    if task.shaping is not None:
        task.lift_diagnostic_masks = torch.zeros((task.n, 8), device=task.device, dtype=torch.bool)


def control(t, actions, live, dt):
    if hasattr(t,'manual_joint_contact'):
        return t.manual_joint_contact.control(actions,live,dt)
    continuous = getattr(t,'continuous_motion',False)
    q = t.m['q']
    if t.envelope is None:
        requested = t.reference + .08 * actions[:, :12]
    else:
        requested = t.center + actions[:, :12] * torch.where(
            actions[:, :12] >= 0, t.upper-t.center, t.center-t.lower)
    requested = requested.maximum(t.low).minimum(t.high)
    target = requested.clone()
    palm_action = actions[:, 12:18].reshape(t.n, 2, 3)
    lower = q.new_tensor(CONTRACT['palm_lower_m'])
    upper = q.new_tensor(CONTRACT['palm_upper_m'])
    palms = palm_action * torch.where(palm_action >= 0, upper, -lower)
    stiffness = (CONTRACT['stiffness_min'] + (actions[:, 18:]+1)*.5*
                 (t.parameters.stiffness_max-CONTRACT['stiffness_min']))
    held = torch.zeros_like(live)
    hold_paused = torch.zeros_like(live)
    if hasattr(t, 'policy_lift_hold') and not getattr(t,'manual_digit_force_balance',False):
        hold = t.policy_lift_hold
        # Expose the current wrist request to the hold servo before replacing
        # it with the accepted common-height target. Close remains policy-owned.
        t.palm_targets[live] = palms[live]
        held = live & hold.active & (t.phase == 3)
        held_target, held_palms, held_stiffness, hold_paused = hold.targets(t,live,dt)
        target = torch.where(held[:,None],held_target,target)
        palms = torch.where(held[:,None,None],held_palms,palms)
        stiffness = torch.where(held[:,None],held_stiffness,stiffness)
    force_region=getattr(t,'force_region',False)
    requested_force=None
    if force_region:
        from .tabletop_policy_force_region import requested_force as force_request,adjust
        requested_force=force_request(actions)
        target=adjust(t,target,requested_force,live&~held,dt)
        stiffness=torch.where(held[:,None],stiffness,torch.full_like(stiffness,10.))
    errors = t.m['mimic_errors']
    severe = errors.abs().amax(1) >= CONTRACT['severe_coupling_rad']
    # Correct only severe linkage inconsistency; ordinary loaded lag does not
    # rewrite the policy target or stop exploration.
    for j, (_, master, ratio, _) in enumerate(MIMIC):
        if j<2 and getattr(t,'manual_smooth_mimic',False):
            continue  # Manual digit servo owns continuous follower correction.
        for hand in range(2):
            k = 6*hand + ACTIVE.index(master)
            error = errors[:, 5*hand+j]
            target[:, k] = torch.where(error.abs() >= CONTRACT['severe_coupling_rad'],
                                      q[:, k]+error/ratio, target[:, k])
    ceiling = torch.full_like(t.f_target, min(t.parameters.max_force_n, 12.))
    # The constrained mode releases at the actual 12N cap, rather than the
    # legacy 1.25*12+deadband protection band.
    relief_ceiling=((ceiling-t.parameters.deadband_n)/1.25 if force_region else ceiling)
    target, over = pressure_limited_target(
        t.command.reshape(t.n, 2, 6), target.reshape(t.n, 2, 6),
        t.m['hand_object_loads'], relief_ceiling, t.direction, dt, t.parameters.deadband_n)
    target = target.flatten(1).maximum(t.low).minimum(t.high)
    # Overload release must not worsen a severe follower error. As in the
    # coordinated controller, neutralize competing requests at measured q.
    for j, (_, master, _, _) in enumerate(MIMIC):
        for hand in range(2):
            k = 6*hand + ACTIVE.index(master)
            error = errors[:, 5*hand+j]
            conflict = (over[:, hand] & (error.abs() >= CONTRACT['severe_coupling_rad'])
                        & ((target[:, k]-q[:, k])*error < 0))
            target[:, k] = torch.where(conflict, q[:, k].clamp(t.low[k], t.high[k]), target[:, k])
    applied = t.command + (target-t.command).clamp(
        -CONTRACT['joint_rate_rad_s']*dt, CONTRACT['joint_rate_rad_s']*dt)
    # Finite coupling deviations remain visible and corrected at joint slew;
    # continuous mode does not turn them into a whole-environment wrist veto.
    mimic_blocks = torch.zeros_like(severe) if continuous else severe
    blocked = mimic_blocks | over.any(1) | (held & hold_paused)
    t.command[live] = applied[live]
    if hasattr(t,'manual_pinch_jacobian') and not hasattr(t,'manual_pinch_home'):
        from .tabletop_manual_pinch_geometry import compensation
        correction=compensation(t,applied,requested)
        t.manual_pinch_compensation[live]=correction[live]
        palms=(palms+correction).maximum(lower).minimum(upper)
    if hasattr(t,'policy_lift_hold'):
        t.policy_lift_hold.project_applied(t,held)
    t.joint_requested_target[live] = requested[live]
    t.joint_governed_target[live] = target[live]
    t.palm_targets[live] = palms[live]
    t.palm_actions[live] = palm_action[live]
    t.palm_clipped[live] = False
    t.policy_stiffness[live] = stiffness[live]
    # Legacy force fields remain observable, explicitly as a protection ceiling
    # in this contract, never as a claimed policy force command or achieved load.
    close_force=requested_force if force_region else ceiling
    force_target = (torch.where(held[:,None],t.policy_lift_hold.force_target,close_force)
                    if hasattr(t,'policy_lift_hold') else close_force)
    t.f_target[live] = force_target[live]
    t.f_requested[live] = force_target[live]
    t.f_filtered[live] = t.m['hand_object_loads'][live]
    if not force_region:t.f_offset[live] = 0.
    t.f_boost[live] = 0.
    t.f_blocked[live] = blocked[live, None]
    t.f_last_slip[live] = t.slip[live]
    t.paused[live] = blocked[live]
    t.pressure_limited[live] = over[live]
    if t.shaping is not None:
        lifting = live & (t.phase == 3)
        upward = (palms[:, :, 2] > t.m['adaptive_palm_offset'][:, :, 2]).any(1)
        zero = torch.zeros_like(lifting)
        t.lift_diagnostic_masks = torch.stack((lifting, lifting & upward & ~blocked,
            lifting & severe, zero, zero, zero, lifting & over.any(1), zero), 1)
    allowed=live & ~blocked
    if getattr(t,'manual_stable_grip_ascent',False) and not continuous:
        measured=t.m['adaptive_palm_offset']
        only_up=((palms[:,:,:2]-measured[:,:,:2]).abs().amax((1,2))<=.0001)
        only_up &= (palms[:,:,2]>=measured[:,:,2]-1e-6).all(1)
        only_up &= (palms[:,:,2]>measured[:,:,2]+1e-6).all(1)
        inside_hard=errors.abs().amax(1)<.02
        touching=(t.m['contact_valid']&(t.m['hand_object_loads']>.05)).all(1)
        allowed |= live&only_up&inside_hard&touching&~over.any(1)&~(held&hold_paused)
    return allowed


def apply_contract(contract, task):
    from .tabletop_tensor_world import PALM_TARGET_CONTRACT
    for key in ('force_action_range_n', 'force_action_semantics', 'initial_force_target_n',
                'initial_force_action_mean', 'lift_grasp_hold', 'close_grasp_hold'):
        contract.pop(key, None)
    contract.update(action='20D_joint12_wrist6_stiffness2_v1',
        grasp_controller=dict(CONTRACT), command_rate_rad_s=CONTRACT['joint_rate_rad_s'],
        palm_rate_m_s=CONTRACT['wrist_rate_m_s'],
        palm_descent_clearance_m=task.palm_descent_clearance_m,
        palm_target_safety=dict(PALM_TARGET_CONTRACT, version='policy_per_hand_clearance_v1',
            synchronization='none_per_hand', clearance='per_hand_measured_safe_target',
            scope='live_policy_rows', reposition_goal_feedback='not_applicable'),
        palm_motion='policy_independent_xyz_in_all_phases',
        palm_exploration=dict(lower_m=CONTRACT['palm_lower_m'], upper_m=CONTRACT['palm_upper_m']),
        force_feedback=dict(enabled=False, force_observation_slots='protection_ceiling_not_policy_target'),
        stiffness_action=dict(range=[CONTRACT['stiffness_min'], task.parameters.stiffness_max],
                              units='Nm/rad', initial_action_mean=-.95),
        active_stiffness_rate_per_s=None,
        lift_gate_semantics='policy_upward_request_before_world_clearance_projection',
        policy_control=dict(CONTRACT))
    return contract
