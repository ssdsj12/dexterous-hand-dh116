"""Continuous policy-owned grasp/lift with no stage admission or pose latch."""

CONTRACT=dict(version='continuous_free_grasp_lift_v1',stage_flow='none',
    legacy_internal_slot='constant_3_only_for_nonstaged_reward_compatibility_not_a_stage',
    phase_observation=False,close_timeout=False,close_completion_events=False,
    entry_conditions=None,pose_latch=False,ordinary_contact_ascent_pause=False,
    actions='policy_joint12_wristXYZ6_normal_force2_at_every_physics_update',
    force_reward_timing='both_normal_upward_and_normal_plus_friction_upward_all_episode',
    force_reward_rates_unchanged=True,downward_force_penalty_unchanged=True,
    stage_completion_bonus=False,initial_pad_depth_m=.002,
    success='actual_30mm_lift_and_bilateral_contact_speed_lt_.03_slip_le_.02_support_nonnegative_for_.5s',
    guards='unchanged_mechanical_mimic_overload_clearance_nonfinite_and_episode_reset',
    total_episode_timeout_s=15.,physics_changed=False)


def bilateral_contact(task):
    return (task.m['contact_valid'] & (task.m['hand_object_loads']>.05)).all(1)


def enable(task):
    if not getattr(task,'policy_control',False) or not getattr(task,'force_region',False):
        raise ValueError('free grasp/lift requires policy force-region control')
    if any(hasattr(task,name) for name in ('policy_lift_hold','coordinated_control','palm_reposition')):
        raise ValueError('free grasp/lift cannot use staged or latched controllers')
    task.free_grasp_lift=True
    task.phase.fill_(3)
    task._seed['phase']=task.phase[0].clone()
    task.contact_time.zero_();task._seed['contact_time']=task.contact_time[0].clone()
    task.force_scoring_started.fill_(True)
    task._seed['force_scoring_started']=task.force_scoring_started[0].clone()
    task.last_observation=task.observe().clone()


def apply_contract(contract,task):
    for key in ('grasp_readiness','constant_force_hold','close_completion_force',
                'close_downward_penalty','lift_pressure_timing','lift_gate_semantics',
                'close_completion_force_gate','close_completion_force_telemetry',
                'phase_force_measurements'):
        contract.pop(key,None)
    contract['free_grasp_lift']=dict(CONTRACT)
    contract['observation_size']=task.observe().shape[-1]
    contract['observation_version']='dh116-free-grasp-lift-v1'
    contract['palm_motion']='policy_owned_XYZ_all_episode'
    contract['force_feedback']=dict(contract['force_feedback'],scope='all_episode_no_stage')
    contract['policy_control']=dict(contract['policy_control'],retained_pose_override=False,
        ordinary_contact_pause=False,automatic_force_correction='all_episode_contact_feedback')
    contract['phase_force_reward']=dict(contract['phase_force_reward'],
        timing='both_terms_all_episode_no_stage_mask',grasp_phases=None,lift_phases=None)
    contract['force_action_semantics']=dict(quantity='per_hand_normal_clamping_target',
        units='N',range=[2.,12.],timing='all_episode_policy_request',friction_excluded=True)
    if 'stable_lift_reward' in contract:
        contract['stable_lift_reward']=dict(contract['stable_lift_reward'],
            gate='actual_bilateral_contact_finite_nonfailure_speed_lt_.03_slip_le_.002_support_nonnegative_all_episode')
    if 'grasp_shaping' in contract:
        contract['grasp_shaping']=dict(contract['grasp_shaping'],
            milestone_condition='actual_bilateral_contact_speed_lt_.03_slip_le_.02_all_episode')
