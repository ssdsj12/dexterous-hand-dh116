"""RSL-RL PPO: device-resident observations, rollout storage, GAE and updates."""
import json
import copy
import math
import os
import time
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from .tabletop_training import write_json
from .tabletop_policy_noise import CONTRACT as POLICY_NOISE_CONTRACT, install_noise_projection, project_policy_noise

VERSION = 'dh116-rsl-tensor-ppo-v1'


def publish_resolved_tensor_configuration(output,contract):
    path=Path(output)/'configuration.json'
    if not path.exists():return
    report=json.loads(path.read_text())
    if not isinstance(report,dict):raise ValueError('configuration report must be an object')
    report.update(contract=copy.deepcopy(contract),tensor_setup_resolved=True)
    write_json(path,report)


def apply_continuous_motion_contract(contract, enabled):
    """Share the continuous-mode identity and reset semantics with startup."""
    if not enabled:
        return contract
    from .tabletop_tensor_env import (CONTINUOUS_MOTION_CONTRACT,
        BOUNDED_FAILURE_RESET_CONTRACT, JOINT_LIMIT_RESET_CONTRACT, OBJECT_DROP_RESET_CONTRACT)
    from .tabletop_palm_mount import CONTRACT as TRANSPORT_CONTRACT
    from .tabletop_policy_recovery import CONTRACT as POLICY_RECOVERY_CONTRACT
    contract['continuous_motion'] = dict(CONTINUOUS_MOTION_CONTRACT,
        adapter_fault_bits=dict(object_reference=1, normal=2, friction=4, transport=8),
        transport=dict(tracking_warning_m=TRANSPORT_CONTRACT['tracking_guard_m'],
            linear_speed_warning_m_s=TRANSPORT_CONTRACT['linear_speed_guard_m_s'],
            finite_exceedance='diagnostic_only',
            invalid='nonfinite_actual_target_or_velocity_isolated_reset',
            command_velocity_limit_m_s=TRANSPORT_CONTRACT['velocity_limit_m_s'],
            effort_limit_n=TRANSPORT_CONTRACT['effort_limit_n']))
    contract['bounded_failure_reset'] = dict(BOUNDED_FAILURE_RESET_CONTRACT,
        strict_mimic_rad=None, max_mimic_exclusive_rad=None, max_force_exclusive_n=None,
        finite_mimic='diagnostic_only', invalid_state='validated_isolated_reset',
        requires='invalid_live_row_without_pending_reset_then_validated_isolated_reset',
        hard_fault='abort_only_when_reset_invalid_or_immediate_recurrence')
    contract['joint_limit_reset'] = dict(JOINT_LIMIT_RESET_CONTRACT,
        early_active_joint_margin=None,
        other_faults='all_invalid_physical_rows_receive_one_validated_isolated_reset')
    contract['object_drop_reset'] = dict(OBJECT_DROP_RESET_CONTRACT,
        other_guards='actual_actuator_limits_and_clearance_preserved_finite_mimic_transport_diagnostic_invalid_state_validated_isolated_reset')
    contract['policy_action_recovery'] = dict(POLICY_RECOVERY_CONTRACT)
    contract.pop('training_joint_limit', None)
    return contract


def tensor_contract_metadata(contract, configured_max_force_n):
    """Add the tensor policy contract fields to an existing full-hand contract."""
    from .tabletop_tensor_task import (PALM_XY_SEARCH_LIMIT_M, PALM_Z_SEARCH_LIMIT_M,
        PALM_XY_EXPLORATION_WEIGHT, POLICY_INITIAL_FORCE_ACTION_MEAN,
        POLICY_INITIAL_FORCE_TARGET_N, POLICY_MIN_FORCE_N, POLICY_MAX_FORCE_N)
    from .tabletop_velocity_limits import CLOSE_COMMAND_RATE_RAD_S
    enriched=copy.deepcopy(contract)
    enriched.update(
        force_action_range_n=[POLICY_MIN_FORCE_N, POLICY_MAX_FORCE_N],
        command_rate_rad_s=CLOSE_COMMAND_RATE_RAD_S,
        force_action_semantics=dict(version='normal_load_target_v2',
            quantity='per_hand_summed_normal_contact_load',units='N',
            decode='2+(action+1)*5',measured_force_guaranteed=False,
            lift='accepted_close_target_with_internal_feedback'),
        phase_force_measurements=dict(close='signed_object_filtered_normal_vertical_sum',
            hold='signed_object_filtered_normal_vertical_sum',
            lift='signed_object_filtered_normal_plus_friction_vertical_sum'),
        initial_force_action_mean=POLICY_INITIAL_FORCE_ACTION_MEAN,
        initial_force_target_n=POLICY_INITIAL_FORCE_TARGET_N,
        palm_exploration=dict(version='synchronized_z_v1',
            xy_bounds_m=[-PALM_XY_SEARCH_LIMIT_M, PALM_XY_SEARCH_LIMIT_M],
            z_bounds_m=[-PALM_Z_SEARCH_LIMIT_M, 0.],
            xy_weight=PALM_XY_EXPLORATION_WEIGHT))
    feedback=dict(enriched.get('force_feedback', {}))
    feedback['target_hand_load_range_n']=[POLICY_MIN_FORCE_N, float(configured_max_force_n)]
    enriched['force_feedback']=feedback
    return enriched


def apply_palm_exploration_contract(contract, axis):
    """Add below-axis behavior while retaining synchronized-Z bounds and weights."""
    if axis != 'xyz':
        exploration=dict(contract.get('palm_exploration', {}))
        exploration.update(version='synchronized_z_v1', axis='world_z',
            close_policy='z<=0', close_xy='bounded_weighted_refinement',
            lift_policy='palm_zero_offset;task_offset_lifts_hand',
            palm_descent_clearance_m=.0003)
        exploration['effective_xy_bounds_m']=[-.000125,.000125]
        contract['palm_exploration']=exploration
        contract['palm_descent_clearance_m']=exploration['palm_descent_clearance_m']
        contract['palm_motion']='synchronized_exploration_z_with_bounded_xy'
    return contract


def apply_palm_target_contract(contract, task):
    """Record every controller path that receives the world safety projection."""
    if getattr(task,'policy_control',False) or hasattr(task,'coordinated_control') or task.palm_exploration_axis=='z_down':
        from .tabletop_tensor_world import PALM_TARGET_CONTRACT
        contract['palm_target_safety']=dict(PALM_TARGET_CONTRACT)
    return contract


def apply_close_completion_contract(contract,task):
    from .tabletop_tensor_task import CLOSE_COMPLETION_FORCE_CONTRACT
    from .tabletop_vertical_force import CLOSE_EVENT_CONTRACT
    if task.vertical_force_reward:
        contract['close_completion_force_gate']=dict(CLOSE_COMPLETION_FORCE_CONTRACT)
        if task.skip_hold:
            contract['close_completion_force_telemetry']=dict(CLOSE_EVENT_CONTRACT)
    if hasattr(task,'close_grasp_hold'):
        from .tabletop_close_grasp_hold import CONTRACT
        contract['close_grasp_hold']=dict(CONTRACT)
        contract['close_completion_force_gate']=dict(CLOSE_COMPLETION_FORCE_CONTRACT,
            version='instant_bilateral_close_completion_v1',
            condition='each_hand_normal_z >= 0 and vertical_force_n >= 0',
            timing='current_bilateral_readiness_no_dwell',nonforce_close_dwell_s=0.,
            nonforce_hold_dwell_s=0.,negative_force='blocks_current_transition',
            nonforce_instability='blocks_current_transition',dwell_cap=0.,
            scope='coordinated_rows')
        contract['close_completion_force_telemetry']=dict(CLOSE_EVENT_CONTRACT)
        if 'grasp_readiness' in contract:
            contract['grasp_readiness']=dict(contract['grasp_readiness'],
                version='instant_bilateral_measured_close_v1',close_dwell_s=0.,
                hold_dwell_s=0.,phase_flow='close_to_lift',scope='coordinated_rows')
        if 'palm_exploration' in contract:
            contract['palm_exploration']=dict(contract['palm_exploration'],
                close_hold_policy='hold_accepted_hand;other_hand_explores_independently')
    return contract


def publish_stopped_monitor(path,error,physics_peaks,iteration,timesteps,num_envs,*,
                            close_completion_telemetry=None,velocity_diagnostics=None,
                            policy_action_recovery=None):
    """Keep completed-update counters and include the failing rollout's peaks.

    Telemetry failure must never replace the original training exception.
    """
    from .tabletop_tensor_env import PHYSICS_PEAK_NAMES
    try:
        row=json.loads(Path(path).read_text())
        if not isinstance(row,dict): row={}
    except (OSError,ValueError):
        row={}
    row.update(status='stopped',error=str(error),iteration=iteration,
               timesteps=timesteps,num_envs=num_envs,backend=VERSION)
    if policy_action_recovery is not None:
        row['policy_action_recovery']=dict(policy_action_recovery)
    if close_completion_telemetry is not None:
        row['close_completion_telemetry']=close_completion_telemetry
    if velocity_diagnostics is not None:
        try:
            row['velocity_diagnostics']=velocity_diagnostics()
            row.pop('velocity_diagnostics_readback_error',None)
        except Exception as readback_error:
            row['velocity_diagnostics_readback_error']=str(readback_error)
    try:
        values=physics_peaks.cpu().tolist()
        invalid=[name for name,value in zip(PHYSICS_PEAK_NAMES,values) if not math.isfinite(value)]
        row['physics_peaks_since_start']={name:value if math.isfinite(value) else None
            for name,value in zip(PHYSICS_PEAK_NAMES,values)}
        row['physics_peak_nonfinite_fields']=invalid
        row.pop('physics_peak_readback_error',None)
    except Exception as readback_error:
        row['physics_peak_readback_error']=str(readback_error)
    try:
        write_json(path,row)
    except Exception as publication_error:
        note = 'Could not publish stopped monitor: '+str(publication_error)
        if hasattr(error, 'add_note'):
            error.add_note(note)
        else:
            error.__notes__ = [*getattr(error, '__notes__', []), note]


def apply_lift_grasp_hold_contract(contract,task):
    if not hasattr(task,'lift_grasp_hold'):return
    from .tabletop_lift_grasp_hold import CONTRACT
    contract['lift_grasp_hold']=dict(CONTRACT)
    if 'palm_exploration' in contract:
        contract['palm_exploration']=dict(contract['palm_exploration'],
            lift_policy='hold_applied_palm_offset;task_offset_lifts_hand')


def checkpoint_contract_changes(saved,current):
    """Only explicitly supported migrations; all unrelated fields stay exact."""
    from .tabletop_tensor_env import VELOCITY_GUARD_CONTRACT, TERMINAL_RESET_CONTRACT, LIVE_VELOCITY_GUARD_CONTRACT
    from .tabletop_pressure_relief import CONTRACT
    from .tabletop_velocity_limits import VELOCITY_HEADROOM_CONTRACT
    from .tabletop_contact_window import reward_contract, REWARD_VERSION as WINDOW_REWARD, OBSERVATION_VERSION
    from .tabletop_tensor_world import ROOT_POSE_WRITE_CONTRACT
    from .tabletop_tensor_task import (MEASURED_READINESS_CONTRACT, measured_readiness_contract,
        CLOSE_DOWNWARD_PENALTY_CONTRACT, POLICY_MIN_FORCE_N, POLICY_MAX_FORCE_N)
    candidate=dict(saved);changes=[]
    from .tabletop_tensor_task import CLOSE_COMPLETION_FORCE_CONTRACT
    from .tabletop_vertical_force import CLOSE_EVENT_CONTRACT
    force_contract=current.get('vertical_force_reward',{})
    if ('close_completion_force_gate' not in saved
            and current.get('close_completion_force_gate')==CLOSE_COMPLETION_FORCE_CONTRACT
            and force_contract.get('source')=='object_filtered_normal_on_hand_negated'
            and force_contract.get('friction_excluded') is True):
        candidate['close_completion_force_gate']=dict(CLOSE_COMPLETION_FORCE_CONTRACT)
        changes.append(CLOSE_COMPLETION_FORCE_CONTRACT['version'])
    if ('close_completion_force_telemetry' not in saved
            and current.get('close_completion_force_telemetry')==CLOSE_EVENT_CONTRACT
            and current.get('close_completion_force_gate')==CLOSE_COMPLETION_FORCE_CONTRACT
            and current.get('grasp_readiness',{}).get('phase_flow')=='close_to_lift'):
        candidate['close_completion_force_telemetry']=dict(CLOSE_EVENT_CONTRACT)
        changes.append(CLOSE_EVENT_CONTRACT['version'])
    old_impulse_contract=dict(CLOSE_DOWNWARD_PENALTY_CONTRACT,
        version='close_downward_impulse_reset_v3',impulse_limit_n_s=2.)
    if (saved.get('close_downward_penalty')==old_impulse_contract
            and current.get('close_downward_penalty')==CLOSE_DOWNWARD_PENALTY_CONTRACT
            and CLOSE_DOWNWARD_PENALTY_CONTRACT['impulse_limit_n_s'] is None):
        candidate['close_downward_penalty']=dict(CLOSE_DOWNWARD_PENALTY_CONTRACT)
        changes.append('close_downward_no_impulse_reset_v4')
    if (saved.get('force_action_range_n')==[.1,12.] and
            current.get('force_action_range_n')==[POLICY_MIN_FORCE_N,POLICY_MAX_FORCE_N]):
        changes.append('force_action_range_2_to_12')
        candidate['force_action_range_n'] = current['force_action_range_n']
    from .tabletop_coordinated_grasp import CONTRACT as CONTROL_CONTRACT
    # Grip recovery and 5x faster transport change physical behavior. An old
    # or absent controller contract must never be silently upgraded on resume.
    # Restoring a clearance deficit changes physical control, not metadata.
    # Require an exact safety contract rather than upgrading old checkpoints.
    # The 0.05 m/s carriage contract also requires an exact saved match.
    from .tabletop_loaded_palm_motion import CONTRACT as REPOSITION_CONTRACT
    if ('palm_reposition' not in saved and current.get('palm_reposition')==REPOSITION_CONTRACT
            and current.get('grasp_controller')==CONTROL_CONTRACT):
        candidate['palm_reposition']=dict(REPOSITION_CONTRACT);changes.append(REPOSITION_CONTRACT['version'])

    from .tabletop_vertical_force import vertical_reward_contract, CONTRACT as FORCE_TELEMETRY_CONTRACT
    legacy_force_telemetry=dict(version='hand_object_normal_vertical_force_v2',units='N',positive='world_up_z',
        source='object_filtered_normal_on_hand_negated',friction_excluded=True,
        excludes='table_and_other_hand_contacts',sampling='live_physics_substeps',
        window='completed_rollout_not_training_lifetime',use='telemetry_only_not_reward_or_lift_gate')
    # V2 was labelled normal-only but incorrectly included friction in totals.
    # V3 excludes it in the statistics; reward and physics remain unchanged.
    if (saved.get('vertical_force_telemetry')==legacy_force_telemetry
            and current.get('vertical_force_telemetry')==FORCE_TELEMETRY_CONTRACT):
        candidate['vertical_force_telemetry']=dict(FORCE_TELEMETRY_CONTRACT)
        changes.append('normal_force_telemetry_excludes_friction_v3')
    old_vertical=vertical_reward_contract('continuous')
    new_vertical=vertical_reward_contract('continuous','after_close')
    old_fields=dict(reward=old_vertical['version'],phase_reward_version=old_vertical['version'],
        vertical_force_reward=old_vertical,observation_version='dh116-free-grasp-contact-window-v2',observation_size=150)
    new_fields=dict(reward=new_vertical['version'],phase_reward_version=new_vertical['version'],
        vertical_force_reward=new_vertical,observation_version='dh116-after-close-force-v3',observation_size=151)
    if all(saved.get(k)==v for k,v in old_fields.items()) and all(current.get(k)==v for k,v in new_fields.items()):
        candidate.update(new_fields);changes.append('force_scoring_after_close_v3')
    close_upward=vertical_reward_contract('continuous','downward_after_close')
    close_fields=dict(new_fields,reward=close_upward['version'],phase_reward_version=close_upward['version'],
                      vertical_force_reward=close_upward)
    if all(saved.get(k)==v for k,v in new_fields.items()) and all(current.get(k)==v for k,v in close_fields.items()):
        candidate.update(close_fields);changes.append('close_upward_reward_v4')
    direct_vertical=vertical_reward_contract('continuous','downward_after_close',skip_hold=True)
    direct_old=dict(close_fields,grasp_readiness=measured_readiness_contract(False))
    direct_new=dict(close_fields,vertical_force_reward=direct_vertical,grasp_readiness=measured_readiness_contract(True))
    if all(saved.get(k)==v for k,v in direct_old.items()) and all(current.get(k)==v for k,v in direct_new.items()):
        candidate.update(direct_new);changes.append('close_direct_lift_v1')
    # The close/lift state machine keeps the same observation shape while the
    # signed force source changes from normal+friction to normal-only. Migrate
    # old rigid checkpoints explicitly so the latest teacher checkpoint can be
    # replayed under the new force semantics.
    legacy_force=saved.get('vertical_force_reward')
    current_force=current.get('vertical_force_reward')
    legacy_close=saved.get('close_downward_penalty')
    current_close=current.get('close_downward_penalty')
    normal_only_migration=(isinstance(legacy_force,dict)
        and legacy_force.get('source')=='object_filtered_normal_plus_friction_on_hand_negated'
        and isinstance(current_force,dict)
        and current_force.get('source')=='object_filtered_normal_on_hand_negated'
        and isinstance(legacy_close,dict)
        and legacy_close.get('version') in ('close_downward_impulse_reset_v3','close_downward_event_penalty_v1')
        and current_close==CLOSE_DOWNWARD_PENALTY_CONTRACT)
    if normal_only_migration:
        excluded={'reward','phase_reward_version','vertical_force_reward',
                  'vertical_force_telemetry','close_downward_penalty'}
        # Compose already validated metadata migrations (including the new
        # instantaneous gate) with this existing force-channel migration.
        if all(candidate.get(k)==current.get(k) for k in (candidate.keys()|current.keys())-excluded):
            candidate.update(reward=current.get('reward'),phase_reward_version=current.get('phase_reward_version'),
                vertical_force_reward=current_force,close_downward_penalty=current_close)
            if 'vertical_force_telemetry' in current:
                candidate['vertical_force_telemetry']=current['vertical_force_telemetry']
            changes.append('normal_force_excludes_friction_v2')
    if ('close_downward_penalty' not in saved
            and current.get('close_downward_penalty')==CLOSE_DOWNWARD_PENALTY_CONTRACT
            and all(saved.get(k)==v and current.get(k)==v for k,v in direct_new.items())):
        candidate['close_downward_penalty']=dict(CLOSE_DOWNWARD_PENALTY_CONTRACT)
        changes.append(CLOSE_DOWNWARD_PENALTY_CONTRACT['version'])
    from .tabletop_grasp_shaping import CONTRACT as SHAPING_CONTRACT
    shaping_old=dict(direct_new,close_downward_penalty=CLOSE_DOWNWARD_PENALTY_CONTRACT)
    shaping_new=dict(shaping_old,grasp_shaping=SHAPING_CONTRACT,observation_size=156,
                     observation_version='dh116-force-progress-milestones-v1')
    if ('grasp_shaping' not in saved and all(saved.get(k)==v for k,v in shaping_old.items())
            and all(current.get(k)==v for k,v in shaping_new.items())):
        candidate.update(shaping_new);changes.append('force_progress_lift_milestones_v1')
    if ('policy_noise' not in saved and current.get('policy_noise')==POLICY_NOISE_CONTRACT
            and all(saved.get(k)==v and current.get(k)==v for k,v in dict(
                tensor_training=VERSION,policy='bounded_gaussian_mean_v1',
                action='20D_piecewise_joint_envelope_palm_load_v1').items())):
        candidate['policy_noise']=POLICY_NOISE_CONTRACT
        changes.append('bounded_policy_noise_v1')
    # Historical scenes hard-coded four velocity iterations. The 480 Hz/64
    # position-iteration rigid scene passed the saved fault-action replay at
    # sixteen; no other solver or scene changes are implicitly migrated.
    if ('velocity_iterations' not in saved and current.get('velocity_iterations')==16
            and all(saved.get(k)==v and current.get(k)==v for k,v in
                    dict(physics_hz=480,solver_type=0,position_iterations=64).items())):
        candidate['velocity_iterations']=16
        changes.append('contact_solver_velocity_4_to_16')
    from .tabletop_material import hand_material_contract, material_contract
    def solid_cylinder(c):
        # Historical built-in solid proxy deliberately has no object kind key;
        # hollow/flexible/external assets carry additional object provenance.
        return (c.get('object_kind','solid_cylinder')=='solid_cylinder'
                and not any(k in c for k in ('object','object_asset','object_backend','object_usd','object_mass_kg')))
    if (saved.get('hand_contact_material')==hand_material_contract('grip')['hand_contact_material']
            and current.get('hand_contact_material')==hand_material_contract('grip_compliant')['hand_contact_material']
            and solid_cylinder(saved) and solid_cylinder(current)
            and all(saved.get(k)==v and current.get(k)==v for k,v in dict(
                physics_hz=480,solver_type=0,position_iterations=64,velocity_iterations=16).items())):
        candidate['hand_contact_material']=current['hand_contact_material']
        changes.append('compliant_silicone_pads_v1')
    # Finite mimic compliance passed the saved 272-step contact-impulse replay.
    # This is a one-way candidate migration, not a general physics override;
    # the final comparison below still rejects every unrelated changed field.
    mimic_baseline=dict(physics_hz=480,solver_type=0,position_iterations=64,
        velocity_iterations=16,num_envs=4096,mimic_damping_ratio=1.,
        active_velocity_limit_rad_s=.3,passive_velocity_limit_rad_s=2.97,
        control_hz=60,active_effort_limit_nm=.2,active_stiffness_range=[10.,20.],
        active_stiffness_reset=10.,active_stiffness_rate_per_s=10.,active_damping=.2,
        root_pose_write=ROOT_POSE_WRITE_CONTRACT,full_hand_contact=True,
        **material_contract('extreme_grip'),
        live_velocity_guard=LIVE_VELOCITY_GUARD_CONTRACT,
        **hand_material_contract('grip_compliant'))
    if (saved.get('mimic_natural_frequency_s_inv')==0.
            and current.get('mimic_natural_frequency_s_inv')==400.
            and solid_cylinder(saved) and solid_cylinder(current)
            and all(saved.get(k)==v and current.get(k)==v for k,v in mimic_baseline.items())):
        candidate['mimic_natural_frequency_s_inv']=400.
        changes.append('compliant_mimic_0_to_400')
    old_load=saved.get('load_parameters',{})
    new_load=current.get('load_parameters',{})
    old_feedback=saved.get('force_feedback',{})
    new_feedback=current.get('force_feedback',{})
    if (old_load.get('max_force_n')==4.
            and old_load.get('force_rate_n_s')==.5
            and new_load==dict(old_load,max_force_n=12.,force_rate_n_s=2.)
            and old_feedback.get('target_hand_load_range_n')==[.1,4.]
            and old_feedback.get('target_rate_n_s')==.5
            and new_feedback==dict(old_feedback,target_hand_load_range_n=[.1,12.],target_rate_n_s=2.)):
        candidate.update(load_parameters=new_load,force_feedback=new_feedback)
        changes.append('hand_load_target_4n_to_12n')
    if (old_load.get('max_force_n')==12.
            and old_load.get('force_rate_n_s')==2.
            and new_load==dict(old_load,force_rate_n_s=None)
            and old_feedback.get('target_hand_load_range_n')==[.1,12.]
            and old_feedback.get('target_rate_n_s')==2.
            and new_feedback==dict(old_feedback,target_rate_n_s=None)):
        candidate.update(load_parameters=new_load,force_feedback=new_feedback)
        changes.append('immediate_force_target_v1')
    if (saved.get('passive_velocity_limit_rad_s')==3.
            and current.get('passive_velocity_limit_rad_s')==2.97
            and 'velocity_headroom' not in saved
            and current.get('velocity_headroom')==VELOCITY_HEADROOM_CONTRACT):
        candidate.update(passive_velocity_limit_rad_s=2.97,velocity_headroom=VELOCITY_HEADROOM_CONTRACT)
        changes.append('add_passive_velocity_headroom')
    window=current.get('contact_reward_window',{})
    start=window.get('start') if isinstance(window,dict) else None
    old_reward=dict(phase_reward_version='phase_potential_v1_10n_5',
        reward='full_hand_object_support_progress_v1_10n_5',
        observation_version='dh116-free-grasp-v1',observation_size=148)
    if (start in ('first_contact','close_start') and 'contact_reward_window' not in saved
            and all(saved.get(k)==v for k,v in old_reward.items())):
        replacement=dict(phase_reward_version=WINDOW_REWARD,reward=WINDOW_REWARD,
            observation_version=OBSERVATION_VERSION,observation_size=150,
            contact_reward_window=reward_contract(start))
        if all(current.get(k)==v for k,v in replacement.items()):
            candidate.update(replacement);changes.append('contact_window_reward_v2')
    window_identity=dict(phase_reward_version=WINDOW_REWARD,reward=WINDOW_REWARD,
        observation_version=OBSERVATION_VERSION,observation_size=150)
    if (saved.get('contact_reward_window')==reward_contract('first_contact')
            and window==reward_contract('close_start')
            and all(saved.get(k)==v and current.get(k)==v for k,v in window_identity.items())):
        candidate['contact_reward_window']=window
        changes.append('contact_window_start_at_close')
    for name,expected,change in (
        ('grasp_readiness',MEASURED_READINESS_CONTRACT,'measured_hand_load_readiness_v1'),
        ('root_pose_write',ROOT_POSE_WRITE_CONTRACT,'skip_unchanged_fixed_root_pose'),
        ('pressure_relief',CONTRACT,'add_measured_pressure_relief'),
        ('terminal_reset',TERMINAL_RESET_CONTRACT,'add_immediate_validated_terminal_reset')):
        if name not in saved and current.get(name)==expected:
            candidate[name]=expected;changes.append(change)
    from .tabletop_tensor_env import BOUNDED_FAILURE_RESET_CONTRACT,JOINT_LIMIT_RESET_CONTRACT
    if ('bounded_failure_reset' not in saved
            and saved.get('terminal_reset')==TERMINAL_RESET_CONTRACT
            and current.get('terminal_reset')==TERMINAL_RESET_CONTRACT
            and current.get('bounded_failure_reset')==BOUNDED_FAILURE_RESET_CONTRACT):
        candidate['bounded_failure_reset']=dict(BOUNDED_FAILURE_RESET_CONTRACT)
        changes.append('joint_bounded_failure_reset_v2')
    if ('joint_limit_reset' not in saved
            and saved.get('terminal_reset')==TERMINAL_RESET_CONTRACT
            and current.get('terminal_reset')==TERMINAL_RESET_CONTRACT
            and current.get('joint_limit_reset')==JOINT_LIMIT_RESET_CONTRACT):
        candidate['joint_limit_reset']=dict(JOINT_LIMIT_RESET_CONTRACT)
        changes.append('finite_joint_limit_episode_reset_v1')
    from .tabletop_tensor_env import OBJECT_DROP_RESET_CONTRACT
    if ('object_drop_reset' not in saved
            and current.get('object_drop_reset')==OBJECT_DROP_RESET_CONTRACT
            and saved.get('terminal_reset')==TERMINAL_RESET_CONTRACT):
        candidate['object_drop_reset']=dict(OBJECT_DROP_RESET_CONTRACT)
        changes.append(OBJECT_DROP_RESET_CONTRACT['version'])
    if candidate!=current:
        raise ValueError('tensor checkpoint contract mismatch; legacy SB3 resume is unsupported')
    return changes


def extend_checkpoint_observations(saved,algorithm):
    """Append approved observation fields with zero weights and Adam moments."""
    policy=copy.deepcopy(saved['policy']);optimizer=copy.deepcopy(saved['optimizer'])
    expected=algorithm.policy.state_dict();heads=('actor.0.weight','critic.0.weight')
    old_size=policy['actor.0.weight'].shape[1];new_size=expected['actor.0.weight'].shape[1]
    if (old_size,new_size) not in ((148,150),(150,151),(151,156)):
        raise ValueError('unsupported observation extension')
    if policy.keys()!=expected.keys():raise ValueError('checkpoint policy keys mismatch')
    for name,value in policy.items():
        if name in heads:
            if value.shape!=(128,old_size) or expected[name].shape!=(128,new_size):
                raise ValueError('checkpoint observation shape mismatch')
            expanded=value.new_zeros((128,new_size));expanded[:,:old_size]=value;policy[name]=expanded
        elif value.shape!=expected[name].shape:raise ValueError('checkpoint parameter shape mismatch')
    names={id(p):name for name,p in algorithm.policy.named_parameters()}
    if len(optimizer['param_groups'])!=len(algorithm.optimizer.param_groups):
        raise ValueError('checkpoint optimizer groups mismatch')
    for source_group,target_group in zip(optimizer['param_groups'],algorithm.optimizer.param_groups):
        if len(source_group['params'])!=len(target_group['params']):
            raise ValueError('checkpoint optimizer parameter count mismatch')
        for source_id,p in zip(source_group['params'],target_group['params']):
            if names[id(p)] not in heads:continue
            state=optimizer['state'].get(source_id,{})
            for name in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                if name in state:
                    value=state[name]
                    if value.shape!=(128,old_size):raise ValueError('checkpoint optimizer moment shape mismatch')
                    expanded=value.new_zeros((128,new_size));expanded[:,:old_size]=value;state[name]=expanded
    return policy,optimizer


def validate_world_adapter(legacy, world, task):
    """One-time host reference comparison and asymmetric selected reset audit."""
    legacy.refresh(sensors=False)
    expected=legacy.measure_batch()
    actual=world.measure()
    errors={}
    tensor_only={}
    for key,value in actual.items():
        if key=='joint_failure': continue
        if getattr(world,'free_wrist_grasp',False) and key in ('wrist_pose','wrist_velocity','wrist_joint_angles'):
            shape={'wrist_pose':(world.n,2,7),'wrist_velocity':(world.n,2,6),
                   'wrist_joint_angles':(world.n,2,3)}[key]
            if value.shape!=shape or not bool(torch.isfinite(value).all()):
                raise ValueError('invalid free wrist startup measurement: '+key)
            if key=='wrist_pose' and not bool((value[:,:,3:].norm(dim=-1)>0).all()):
                raise ValueError('invalid free wrist startup quaternion')
            tensor_only[key]=dict(shape=list(value.shape),dtype=str(value.dtype),finite=True)
            errors[key]=0.
            continue
        if getattr(world,'continuous_motion',False) and key in ('adapter_fault_bits','transport_warning'):
            # These diagnostics have no host equivalent. Validate their own
            # startup schema instead of treating them as unchecked telemetry.
            if (not isinstance(value,torch.Tensor) or value.shape!=(world.n,)
                    or not bool(torch.isfinite(value).all())):
                raise ValueError(f'invalid tensor startup measurement: {key}')
            if key=='adapter_fault_bits':
                valid=value==0  # any raw adapter fault must fail startup
            else:
                valid=(value==0)|(value==1)  # finite transport warnings are allowed
            if not bool(valid.all()):
                raise ValueError(f'invalid tensor startup measurement: {key}')
            errors[key]=0.0
            tensor_only[key]=dict(shape=list(value.shape),dtype=str(value.dtype),
                                  finite=True,nonzero_rows=int((value!=0).sum()))
            continue
        if key.startswith('contact_'):
            # PhysX contact positions/normals are tensor-world telemetry and
            # have no host-adapter reference at the no-sensor validation tick.
            errors[key]=0.0
            continue
        if key not in expected:
            if key != 'vertical_hand_force_n':
                raise KeyError(f'host reference missing tensor measurement: {key}')
            # The host adapter does not expose signed contact-force components.
            # At this validation point TensorWorld was refreshed without sensors,
            # so its per-hand vertical force is intentionally the zero reset value.
            reference=torch.zeros_like(value)
        else:
            reference=torch.as_tensor(expected[key],device=world.device,dtype=value.dtype)
        error=float((value-reference).abs().max())
        errors[key]=error
        if error>2e-6: raise ValueError(f'tensor initialization differs from host reference: {key} {error}')
    original=world.wire.data.root_pose_w.clone()
    shifted=original.clone()
    shifted[:,2]+=torch.arange(world.n,device=world.device)*1e-5
    world.wire.write_root_pose_to_sim(shifted)
    legacy.sim.physics_sim_view.update_articulations_kinematic()
    world.refresh(sensors=False)
    if not torch.allclose(world.wp,shifted[:,:3].double(),atol=1e-6,rtol=0):
        raise ValueError('tensor asymmetric object mapping failed')
    before=world.wp.clone()
    selected=torch.tensor([world.n//2],device=world.device)
    world.reset(selected)
    others=torch.arange(world.n,device=world.device)!=selected[0]
    if not torch.equal(before[others],world.wp[others]):
        raise ValueError('tensor selected reset altered other environments')
    world.reset(torch.arange(world.n,device=world.device))
    result=dict(maximum_reference_errors=errors,asymmetric_mapping=True,selected_reset_isolation=True)
    if tensor_only:result['tensor_only_startup_fields']=tensor_only
    return result


def observations(value):
    return TensorDict({'policy':value},batch_size=[len(value)],device=value.device)


def make_algorithm(obs, steps=24,action_size=20):
    if action_size not in (20,26):raise ValueError('supported policy action sizes are20 or26')
    policy = ActorCritic(obs,{'policy':['policy'],'critic':['policy']},action_size,
        actor_hidden_dims=[128,128],critic_hidden_dims=[128,128],activation='tanh',
        init_noise_std=math.exp(-1.2),noise_std_type='log').to(obs.device)
    # Preserve bounded policy mean and initially gentle requested load. Gaussian
    # actions are clipped only at the actuator boundary, as in the SB3 baseline.
    for network in (policy.actor,policy.critic):
        linear = [m for m in network if isinstance(m,torch.nn.Linear)]
        for layer in linear:
            torch.nn.init.orthogonal_(layer.weight,math.sqrt(2)); torch.nn.init.zeros_(layer.bias)
        torch.nn.init.orthogonal_(linear[-1].weight,.01 if network is policy.actor else 1.)
    with torch.no_grad(): policy.actor[-1].bias[-2:] = math.atanh(-.4 if action_size==26 else -.95)
    policy.actor.append(torch.nn.Tanh())
    minibatches = min(8,len(obs))
    algorithm = PPO(policy,num_learning_epochs=5,num_mini_batches=minibatches,
        learning_rate=3e-4,gamma=.995,lam=.95,clip_param=.2,entropy_coef=.005,
        value_loss_coef=.5,max_grad_norm=.5,schedule='adaptive',desired_kl=.02,
        device=str(obs.device))
    algorithm.init_storage('rl',len(obs),steps,obs,[action_size])
    install_noise_projection(algorithm)
    return algorithm


def reposition_telemetry(adjustment):
    """Return monitor fields supported by a palm-reposition implementation.

    ``LoadedPalmReposition`` predates force-direction diagnostics and only
    exposes state/attempt fields.  ForceDirectionReposition additionally
    exposes signed force samples and timeout flags.  Keep the common fields
    stable, while publishing optional diagnostics only when the controller
    implements them so the default release-reposition path cannot crash the
    training loop.
    """
    row = {
        name: float((adjustment.state == index).double().mean())
        for index, name in enumerate((
            'idle_hand_fraction', 'release_hand_fraction',
            'move_hand_fraction', 'reclose_hand_fraction'))
    }
    row['aborted_hand_fraction'] = float(adjustment.aborted.double().mean())
    attempts = getattr(adjustment, 'attempts', None)
    active = attempts > 0 if attempts is not None else None
    force_fields = (
        ('force_before_release_n', 'force_before'),
        ('force_after_release_n', 'force_release'),
        ('force_during_move_n', 'force_move'),
        ('force_after_reclose_n', 'force_reclose'),
    )
    if active is not None:
        for output_name, attribute_name in force_fields:
            values = getattr(adjustment, attribute_name, None)
            if values is None:
                continue
            row[output_name] = (
                float(values[active].mean()) if bool(active.any()) else 0.
            )
    for output_name, attribute_name in (
        ('exhausted_reposition_fraction', 'exhausted_reposition'),
        ('release_timeout_fraction', 'release_timeout'),
        ('recontact_abort_fraction', 'recontact_abort'),
        ('reclose_timeout_fraction', 'reclose_timeout'),
    ):
        values = getattr(adjustment, attribute_name, None)
        if values is not None:
            row[output_name] = float(values.double().mean())
    return row


def write_reward_scalars(writer,terms,total,iteration):
    for name,value in terms.items():writer.add_scalar('reward/'+name,value,iteration)
    writer.add_scalar('reward/total',total,iteration)


def training_budget_reached(args,iteration,timesteps,min_episodes):
    maximum=getattr(args,'max_iterations',None)
    if maximum is not None:return iteration>=maximum
    if args.episodes_per_env is not None:return min_episodes>=args.episodes_per_env
    return timesteps>=args.timesteps


def save_checkpoint(path, algorithm, contract, iteration, timesteps, env):
    payload=dict(version=VERSION,contract=contract,iteration=iteration,timesteps=timesteps,
        policy=algorithm.policy.state_dict(),optimizer=algorithm.optimizer.state_dict(),
        learning_rate=algorithm.learning_rate,episodes=env.episodes,successes=env.successes,
        recovered=env.recovered,reached_counts=env.reached_counts,transition_counts=env.transition_counts,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    if hasattr(env,'combined_failure_resets'):
        payload['combined_failure_resets']=env.combined_failure_resets
    if hasattr(env,'joint_limit_resets'):
        payload['joint_limit_resets']=env.joint_limit_resets
    if hasattr(env,'object_drop_resets'):
        payload['object_drop_resets']=env.object_drop_resets
    if hasattr(env,'numerical_resets'):
        payload['numerical_resets']=env.numerical_resets
    if getattr(algorithm,'policy_action_recovery',None) is not None:
        payload['policy_action_recovery']=dict(algorithm.policy_action_recovery.counts)
    if hasattr(env,'close_completion_events'):
        payload['close_completion_event_index']=env.close_completion_events.event_index
    tmp=Path(str(path)+'.tmp'); torch.save(payload,tmp); os.replace(tmp,path)


def run_tensor_training(world, template, args, contract):
    from .tabletop_tensor_task import TensorTask, REWARD_VERSION, FAILURE_NAMES, measured_readiness_contract, CLOSE_DOWNWARD_PENALTY_CONTRACT
    from .tabletop_tensor_world import TensorWorld, ROOT_POSE_WRITE_CONTRACT
    from .tabletop_tensor_env import TensorEnv, VELOCITY_GUARD_CONTRACT, TERMINAL_RESET_CONTRACT, PHYSICS_PEAK_NAMES, LIVE_VELOCITY_GUARD_CONTRACT, TRAINING_JOINT_LIMIT_CONTRACT, BOUNDED_FAILURE_RESET_CONTRACT, JOINT_LIMIT_RESET_CONTRACT
    from torch.utils.tensorboard import SummaryWriter
    report_contract=contract
    torch.manual_seed(args.seed)
    device_world=TensorWorld(world)
    window=getattr(args,'contact_reward_window',None)
    task=TensorTask(template,args.num_envs,args.device,pressure_relief=args.pressure_relief,
                    contact_reward_window=window,grasp_readiness=getattr(args,'grasp_readiness',None),
                    vertical_force_reward=getattr(args,'vertical_force_reward',False),
                    vertical_force_reward_mode=getattr(args,'vertical_force_reward_mode','window'),
                    force_reward_start=getattr(args,'force_reward_start','episode_start'),
                    skip_hold=getattr(args,'skip_hold',False),
                    close_downward_penalty=getattr(args,'close_downward_penalty',False),
                    grasp_shaping=getattr(args,'grasp_shaping',False) and not getattr(args,'free_wrist_grasp',False),
                    palm_exploration_axis=getattr(args,'palm_exploration_axis','xyz'))
    probe_contact_z=(getattr(args,'control_probe_mode',None)=='grasp_teacher_contact_z_sweep'
                     and getattr(args,'control_probe_checkpoint',None))
    if getattr(args,'grasp_controller','legacy')=='coordinated' or probe_contact_z:
        task.enable_coordinated_control(torch.ones(args.num_envs,dtype=torch.bool,device=args.device))
    if getattr(args,'grasp_controller','legacy')=='policy':
        task.enable_policy_control()
    if getattr(args,'force_region_control',False):
        from .tabletop_policy_force_region import enable
        enable(task)
        task.last_observation=task.observe().clone()
    if getattr(args,'policy_lift_hold',False):
        from .tabletop_policy_lift_hold import PolicyLiftHold
        task.policy_lift_hold=PolicyLiftHold(args.num_envs,args.device)
        task.last_observation=task.observe().clone()
    if getattr(args,'stable_lift_reward',False):
        from .tabletop_stable_lift_reward import StableLiftReward
        task.stable_lift_reward=StableLiftReward(args.num_envs,args.device)
        task.terms['stable_lift_hold']=torch.zeros_like(task.reward)
        task.last_observation=task.observe().clone()
    if getattr(args,'phase_force_reward',False):
        task.phase_force_reward=True
        for name in ('grasp_upward_force','lift_upward_force'):
            task.terms[name]=torch.zeros_like(task.reward)
    if getattr(args,'free_grasp_lift',False):
        from .tabletop_free_grasp_lift import enable
        enable(task)
    if getattr(args,'free_wrist_grasp',False):
        from .tabletop_free_wrist_grasp import enable
        enable(task,device_world)
    if getattr(args,'contact_effort_feedback',False):
        from .tabletop_contact_effort import ContactEffortFeedback
        task.contact_effort=ContactEffortFeedback(args.num_envs,args.device,effort_limit_nm=task.parameters.effort_limit_nm)
        device_world.contact_effort_enabled=True
    if getattr(args,'release_reposition',False):
        from .tabletop_loaded_palm_motion import LoadedPalmReposition
        task.palm_reposition=LoadedPalmReposition(torch.ones(args.num_envs,dtype=torch.bool,device=args.device))
    if getattr(args,'force_direction_reposition',False):
        from .tabletop_force_direction_reposition import ForceDirectionReposition
        task.palm_reposition=ForceDirectionReposition(torch.ones(args.num_envs,dtype=torch.bool,device=args.device))
        task.force_direction_reposition=True
    env=TensorEnv(device_world,task,recovery=args.training_episode_resets,
                  fault_trace_path=args.output/'physics_fault_trace.json',
                  training_joint_limit=args.training_episode_resets and not bool(getattr(args,'control_probe_checkpoint',None)))
    if getattr(args, 'grasp_acquisition_shaping', False):
        if getattr(args, 'object_kind', 'solid_cylinder') != 'solid_cylinder':
            raise ValueError('acquisition geometry currently requires a solid cylinder')
        env.grasp_acquisition = True
        env.acquisition_radius_m = args.solid_cylinder_radius_m
        env.acquisition_length_m = .5
    if task.shaping is not None:
        from .tabletop_grasp_shaping import LIFT_GATE_NAMES,lift_gate_summary
        env.lift_gate_counts=torch.zeros(len(LIFT_GATE_NAMES),device=args.device,dtype=torch.long)
    from .tabletop_vertical_force import VerticalForceStats, LiftEntryForceStats, CONTRACT as VERTICAL_FORCE_CONTRACT
    env.vertical_force_stats=VerticalForceStats(args.device)
    env.lift_entry_force_stats=LiftEntryForceStats(args.device)
    obs=observations(env.reset())
    validation=validate_world_adapter(world,device_world,task)
    obs=observations(env.reset())
    if getattr(args,'require_initial_below_axis',False):
        from .tabletop_initial_contact_check import validate_initial_contact
        write_json(args.output/'initial_contact_validation.json',validate_initial_contact(task.m))
    write_json(args.output/'tensor_world_validation.json',validation)
    steps=args.rollout_steps or 24
    from .tabletop_tensor_task import (POLICY_MIN_FORCE_N, POLICY_MAX_FORCE_N,
        POLICY_INITIAL_FORCE_ACTION_MEAN, POLICY_INITIAL_FORCE_TARGET_N,
        PALM_XY_SEARCH_LIMIT_M, PALM_Z_SEARCH_LIMIT_M, PALM_XY_EXPLORATION_WEIGHT)
    contract=dict(contract,tensor_training=VERSION,phase_reward_version=REWARD_VERSION,
        force_action_range_n=[POLICY_MIN_FORCE_N,POLICY_MAX_FORCE_N],
        force_action_semantics=dict(version='phase_force_channels_v1',close='signed_object_filtered_normal_vertical_sum',
            hold='signed_object_filtered_normal_vertical_sum',lift='signed_object_filtered_normal_plus_friction_vertical_sum'),
        initial_force_action_mean=POLICY_INITIAL_FORCE_ACTION_MEAN,
        initial_force_target_n=POLICY_INITIAL_FORCE_TARGET_N,
        palm_exploration=dict(version='synchronized_z_v1',xy_bounds_m=[-PALM_XY_SEARCH_LIMIT_M,PALM_XY_SEARCH_LIMIT_M],
            z_bounds_m=[-PALM_Z_SEARCH_LIMIT_M,0.],xy_weight=PALM_XY_EXPLORATION_WEIGHT),
        vector_backend='device_resident_rigid_task_v1',rollout_steps=steps,
        optimizer='rsl_rl.PPO',kl_schedule='adaptive',epochs=5,
        minibatches=min(8,args.num_envs),learning_rate=3e-4,
        reset_velocity_guard=VELOCITY_GUARD_CONTRACT,terminal_reset=TERMINAL_RESET_CONTRACT,
        live_velocity_guard=LIVE_VELOCITY_GUARD_CONTRACT,
        bounded_failure_reset=dict(BOUNDED_FAILURE_RESET_CONTRACT),
        joint_limit_reset=dict(JOINT_LIMIT_RESET_CONTRACT),
        root_pose_write=ROOT_POSE_WRITE_CONTRACT,policy_noise=POLICY_NOISE_CONTRACT,
        vertical_force_telemetry=VERTICAL_FORCE_CONTRACT)
    from .tabletop_tensor_env import OBJECT_DROP_RESET_CONTRACT
    contract['object_drop_reset']=dict(OBJECT_DROP_RESET_CONTRACT)
    if env.training_joint_limit:
        contract['training_joint_limit']=dict(TRAINING_JOINT_LIMIT_CONTRACT)
    if task.grasp_readiness is not None:
        contract['grasp_readiness']=measured_readiness_contract(task.skip_hold)
    if task.close_downward_penalty:
        contract['close_downward_penalty']=dict(CLOSE_DOWNWARD_PENALTY_CONTRACT)
    if args.pressure_relief:
        from .tabletop_pressure_relief import CONTRACT
        contract['pressure_relief']=CONTRACT
        from .tabletop_tensor_task import LIFT_PRESSURE_TIMING_CONTRACT
        contract['lift_pressure_timing']=dict(LIFT_PRESSURE_TIMING_CONTRACT)
    if window is not None:
        from .tabletop_contact_window import reward_contract,REWARD_VERSION as WINDOW_REWARD,OBSERVATION_VERSION
        contract.update(reward=WINDOW_REWARD,phase_reward_version=WINDOW_REWARD,
            observation_version=OBSERVATION_VERSION,observation_size=150,
            contact_reward_window=reward_contract(window))
    if task.vertical_force_reward:
        from .tabletop_vertical_force import vertical_reward_contract
        REWARD_CONTRACT=vertical_reward_contract(task.vertical_force_reward_mode,task.force_reward_start,skip_hold=task.skip_hold)
        if task.vertical_force_reward_mode=='continuous':
            contract['contact_reward_window']=dict(enabled=False,legacy_clock='observation_only')
        contract.update(reward=REWARD_CONTRACT['version'],phase_reward_version=REWARD_CONTRACT['version'],
                        vertical_force_reward=dict(REWARD_CONTRACT))
    if task.force_reward_start!='episode_start':
        contract.update(observation_version='dh116-after-close-force-v3',observation_size=151)
    if task.shaping is not None:
        from .tabletop_grasp_shaping import CONTRACT as SHAPING_CONTRACT
        contract.update(grasp_shaping=dict(SHAPING_CONTRACT),observation_size=156,
                        observation_version='dh116-force-progress-milestones-v1')
    contract=tensor_contract_metadata(contract,task.parameters.max_force_n)
    if getattr(env, 'grasp_acquisition', False):
        from .tabletop_grasp_acquisition import CONTRACT as ACQUISITION
        contract['grasp_acquisition'] = dict(ACQUISITION,
            radius_m=env.acquisition_radius_m, length_m=env.acquisition_length_m)
    apply_palm_exploration_contract(contract, task.palm_exploration_axis)
    if hasattr(task,'coordinated_control'):
        from .tabletop_coordinated_grasp import CONTRACT as CONTROL_CONTRACT
        contract['grasp_controller']=dict(CONTROL_CONTRACT)
    apply_lift_grasp_hold_contract(contract,task)
    apply_palm_target_contract(contract,task)
    apply_close_completion_contract(contract,task)
    if getattr(args,'palm_transport','fixed_pose')=='driven_carriage' and (
            not args.control_probe_checkpoint or getattr(args,'grasp_controller','legacy') in ('coordinated','policy')):
        from .tabletop_palm_mount import CONTRACT as WRIST_CONTRACT
        contract['palm_transport']=dict(WRIST_CONTRACT)
    if getattr(args,'release_reposition',False):
        from .tabletop_loaded_palm_motion import CONTRACT as REPOSITION_CONTRACT
        contract['palm_reposition']=dict(REPOSITION_CONTRACT)
    if getattr(args,'force_direction_reposition',False):
        from .tabletop_force_direction_reposition import CONTRACT as REPOSITION_CONTRACT
        contract['palm_reposition']=dict(REPOSITION_CONTRACT)
        contract['force_direction_reposition']=True

    if hasattr(task,'contact_effort'):
        from .tabletop_contact_effort import CONTRACT as CONTACT_EFFORT_CONTRACT
        from .tabletop_coordinated_grasp import CONTRACT as CONTROL_CONTRACT
        contract['contact_effort_feedback']=dict(CONTACT_EFFORT_CONTRACT,
            total_motor_limit_nm=task.parameters.effort_limit_nm,
            coupling_permission_rad=CONTROL_CONTRACT['severe_coupling_rad'],
            position_pause_disables_effort=False)

    if getattr(task,'policy_control',False):
        from .tabletop_policy_control import apply_contract
        apply_contract(contract,task)
    if hasattr(task,'policy_lift_hold'):
        from .tabletop_policy_lift_hold import CONTRACT as POLICY_LIFT_HOLD
        contract['policy_lift_hold']=dict(POLICY_LIFT_HOLD)
        contract['observation_size']=obs['policy'].shape[-1]
        contract['observation_version']='dh116-policy-lift-hold-v1'
        contract['policy_control']=dict(contract['policy_control'],
            retained_pose_override=True,automatic_force_correction='lift_only',
            ordinary_contact_pause='lift_grip_loss_or_tracking',automatic_ascent=False)
        contract['force_feedback']=dict(enabled=True,scope='accepted_lift_grasp_only',
            target='captured_measured_hand_load_with_bounded_recovery')
        contract['palm_motion']='policy_close_then_policy_requested_common_lift'
        contract['lift_gate_semantics']='accepted_grasp_retained_bilateral_load_slip_and_tracking_gated'

    if hasattr(task,'stable_lift_reward'):
        from .tabletop_stable_lift_reward import CONTRACT as STABLE_LIFT_CONTRACT
        contract['stable_lift_reward']=dict(STABLE_LIFT_CONTRACT)
        contract['observation_size']=obs['policy'].shape[-1]
        contract['observation_version']='dh116-stable-lift-reward-v1'
    if getattr(task,'force_region',False):
        from .tabletop_policy_force_region import CONTRACT as FORCE_REGION_CONTRACT
        from .tabletop_policy_lift_hold import FORCE_HOLD_CONTRACT
        contract['force_region_control']=dict(FORCE_REGION_CONTRACT)
        contract['constant_force_hold']=dict(FORCE_HOLD_CONTRACT)
        contract['action']='20D_joint12_wrist6_force2_v1'
        contract['force_action_range_n']=[2.,12.]
        contract['force_feedback']=dict(enabled=True,scope='close_and_lift',
            measurement='actual_per_hand_object_contact_load',physical_force_guarantee=False)
        contract['policy_control']=dict(contract['policy_control'],automatic_force_correction='close_and_lift')
        if 'policy_lift_hold' in contract:
            contract['policy_lift_hold']=dict(contract['policy_lift_hold'],observation_size=34,
                force_capture='accepted_close_requested_force_2_to_12',reference_recovery='immutable')
        contract['stiffness_action']=dict(enabled=False,fixed_nm_rad=10.)
        contract['grasp_readiness']=dict(contract['grasp_readiness'],min_hand_load_n=2.,max_hand_load_n=12.,
            contact_region='all_valid_force_bearing_contacts_below_actual_local_Y_axis')
        contract['observation_size']=obs['policy'].shape[-1]
        contract['observation_version']='dh116-force-region-control-v1'
        if 'grasp_acquisition' in contract:
            contract['grasp_acquisition']['contact_region_weights']=dict(above=.5,below=2.)
    if getattr(task,'phase_force_reward',False):
        from .tabletop_phase_force_reward import CONTRACT as PHASE_FORCE_CONTRACT
        contract['phase_force_reward']=dict(PHASE_FORCE_CONTRACT)
    if getattr(args,'require_initial_below_axis',False):
        from .tabletop_initial_contact_check import CONTRACT as INITIAL_CONTACT_CONTRACT
        contract['initial_contact_validation']=dict(INITIAL_CONTACT_CONTRACT)
    if getattr(task,'free_grasp_lift',False):
        from .tabletop_free_grasp_lift import apply_contract
        apply_contract(contract,task)

    apply_continuous_motion_contract(contract,env.continuous_motion)
    if getattr(task,'free_wrist_grasp',False):
        from .tabletop_free_wrist_grasp import apply_contract
        apply_contract(contract,task)
    # The caller's report holds this original dict. Publish the fully resolved
    # tensor contract there as well, including manual/probe/error exits.
    report_contract.clear()
    report_contract.update(contract)
    if getattr(task,'free_wrist_grasp',False):
        publish_resolved_tensor_configuration(args.output,contract)

    if getattr(args,'manual_control',False):
        from .tabletop_manual_control import run_manual_control
        write_json(args.output/'tensor_contract.json',contract)
        return run_manual_control(env,args,contract)
    if getattr(args,'teacher_search_cases',None):
        from .tabletop_success_teacher import run_success_teacher
        return run_success_teacher(env,args,contract,obs)

    algorithm=make_algorithm(obs,steps,action_size=getattr(task,'action_size',20))
    policy_recovery=None
    if env.continuous_motion:
        from .tabletop_policy_recovery import PolicyActionRecovery
        policy_recovery=PolicyActionRecovery()
        algorithm.policy_action_recovery=policy_recovery
    actor_initialization=None
    if getattr(args,'actor_initialization',None):
        from .tabletop_behavior_cloning import load_actor_initialization
        actor_initialization=load_actor_initialization(algorithm,args.actor_initialization,contract)
        from .tabletop_student_validation import validate_student
        passed,student_validation=validate_student(env,algorithm,args,contract,obs)
        if not passed or getattr(args,'student_validation_only',False):
            return dict(status='student_validated' if passed else 'student_validation_failed',
                student_validation=student_validation,training_skipped=not passed,optimization_performed=False)
        # Validation uses ordinary resets, but optimization starts a fresh task
        # and fresh training counters with the unchanged actor and optimizer.
        obs=observations(env.reset())
        for name in ('episodes','successes','recovered','reached_counts','transition_counts'):
            getattr(env,name).zero_()
        actor_initialization['closed_loop_student_validation']=student_validation
    if getattr(args,'control_probe_checkpoint',None):
        from .tabletop_control_probe import run_control_probe
        return run_control_probe(env,algorithm,args,contract,obs)
    if task.shaping is not None and algorithm.gamma!=SHAPING_CONTRACT['gamma']:
        raise ValueError('potential shaping discount must match PPO gamma')
    iteration=timesteps=0
    close_event_index=0
    if args.resume:
        saved=torch.load(args.resume,map_location=args.device,weights_only=False)
        if saved['version']!=VERSION:
            raise ValueError('tensor checkpoint contract mismatch; legacy SB3 resume is unsupported')
        changes=checkpoint_contract_changes(saved['contract'],contract)
        if 'force_action_range_2_to_12' in changes:
            raise ValueError('tensor checkpoint contract mismatch; legacy force-action range is not resume-compatible')
        if changes:
            write_json(args.output/'checkpoint_migration.json',dict(source=str(args.resume),
                changes=changes,policy_and_optimizer_retained='bounded_policy_noise_v1' not in changes,
                policy_noise_projection=('log_std_and_its_changed_adam_moments_only'
                    if 'bounded_policy_noise_v1' in changes else None),
                previous_iteration=saved['iteration'],
                added_contract_fields={k:v for k,v in contract.items() if k not in saved['contract']},
                changed_contract_fields={k:dict(old=saved['contract'][k],new=v)
                    for k,v in contract.items() if k in saved['contract'] and v!=saved['contract'][k]},
                policy_input_extension=(dict(old=saved['contract'].get('observation_size',148),new=contract['observation_size'],new_weights_and_moments='zero')
                    if set(changes)&{'contact_window_reward_v2','force_scoring_after_close_v3','force_progress_lift_milestones_v1'} else None)))
        policy,optimizer=(extend_checkpoint_observations(saved,algorithm)
            if set(changes)&{'contact_window_reward_v2','force_scoring_after_close_v3','force_progress_lift_milestones_v1'} else (saved['policy'],saved['optimizer']))
        algorithm.policy.load_state_dict(policy)
        algorithm.optimizer.load_state_dict(optimizer)
        previous_std=algorithm.policy.log_std.detach().exp().cpu().tolist()
        changed_noise=project_policy_noise(algorithm)
        write_json(args.output/'policy_noise_resume.json',dict(
            source=str(args.resume),previous_std=previous_std,
            current_std=algorithm.policy.log_std.detach().exp().cpu().tolist(),
            changed_coordinates=changed_noise.cpu().tolist(),
            actor_critic_weights_retained=True,other_optimizer_state_retained=True))
        algorithm.learning_rate=saved['learning_rate']
        iteration,timesteps=saved['iteration'],saved['timesteps']
        close_event_index=int(saved.get('close_completion_event_index',0))
        for name in ('episodes','successes','recovered','reached_counts','transition_counts'):
            getattr(env,name).copy_(saved[name])
        if 'combined_failure_resets' in saved:
            env.combined_failure_resets.copy_(saved['combined_failure_resets'])
        if 'joint_limit_resets' in saved:
            env.joint_limit_resets.copy_(saved['joint_limit_resets'])
        if 'object_drop_resets' in saved:
            env.object_drop_resets.copy_(saved['object_drop_resets'])
        if 'numerical_resets' in saved:
            env.numerical_resets.copy_(saved['numerical_resets'])
        if policy_recovery is not None:
            policy_recovery=PolicyActionRecovery(saved.get('policy_action_recovery'))
            algorithm.policy_action_recovery=policy_recovery
        torch.set_rng_state(saved['torch_rng'].cpu())
        if saved['cuda_rng']: torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    write_json(args.output/'tensor_contract.json',contract)
    write_json(args.output/'training_start.json',dict(
        resume_source=str(args.resume) if args.resume else None,actor_initialization=actor_initialization,
        initial_iteration=iteration,initial_timesteps=timesteps,
        initial_episodes=int(env.episodes.sum()),initial_successes=int(env.successes.sum()),
        optimizer_state_entries=len(algorithm.optimizer.state),
        initial_policy_std=algorithm.policy.log_std.detach().exp().cpu().tolist()))
    writer=SummaryWriter(str(args.output/'tensorboard'))
    if task.vertical_force_reward and task.skip_hold and not getattr(task,'free_grasp_lift',False):
        from .tabletop_vertical_force import CloseCompletionForceEvents
        env.close_completion_events=CloseCompletionForceEvents(writer,
            args.output/'close_completion_events.jsonl',close_event_index)
        # The same optional writer serves endpoint and rollout telemetry; an
        # endpoint I/O fault must not reappear at the next rollout flush/close.
        writer=env.close_completion_events.writer
        writer.add_text('close_completion/说明',
            'force_N/close完成：每个点是一个环境完成 close、进入 lift 当步的纯法向 Z 力；'
            '横轴是完成事件编号，不是 PPO 轮次；不对时间或环境取平均。'
            '无完成事件就没有力点。环境/回合对应关系见 close_completion 和 JSONL。'
            'rollout_force_N 是单独保留的全过程统计。查看原始瞬时值请将 TensorBoard smoothing 设为 0。',0)
    checkpoint=args.output/'tensor_ppo.pt'
    started=time.monotonic()
    stop=None
    stopped_telemetry=None
    failure_counts=torch.zeros(max(FAILURE_NAMES)+1,dtype=torch.long,device=args.device)
    row=dict(backend=VERSION,num_envs=args.num_envs,iteration=iteration,timesteps=timesteps,
             iteration_budget=getattr(args,'max_iterations',None),successes=int(env.successes.sum()))
    if training_budget_reached(args,iteration,timesteps,int(env.episodes.min())):
        save_checkpoint(checkpoint,algorithm,contract,iteration,timesteps,env)
    try:
        while not training_budget_reached(args,iteration,timesteps,int(env.episodes.min())):
            collect=time.monotonic()
            if hasattr(env,'close_completion_events'):
                env.close_completion_events.iteration=iteration+1
            env.vertical_force_stats.clear()
            env.lift_entry_force_stats.clear()
            if task.shaping is not None:env.lift_gate_counts.zero_()
            reward_sum=torch.zeros((),device=args.device)
            terms={}
            occupancy=torch.zeros(4,device=args.device)
            relief_fraction=torch.zeros((),device=args.device)
            clipped_fraction=torch.zeros((),device=args.device)
            with torch.inference_mode():
                if policy_recovery is not None:
                    def before_step(control_step):
                        if hasattr(env,'close_completion_events'):
                            env.close_completion_events.control_step=control_step
                    collected=policy_recovery.collect(algorithm,env,obs,steps,before_step)
                    obs=collected['obs']
                    reward_sum=collected['reward_sum'];terms=collected['terms']
                    occupancy=collected['occupancy'];relief_fraction=collected['relief_fraction']
                    clipped_fraction=collected['clipped_fraction']
                    if collected['failures'] is not None:
                        counts=collected['failures']
                        failure_counts[:len(counts)]+=counts
                else:
                    for rollout_step in range(steps):
                        action=algorithm.act(obs)
                        if not bool(torch.isfinite(action).all()):
                            raise RuntimeError('nonfinite policy action before actuator clipping')
                        clipped_fraction+=(action.abs()>1.).float().mean()
                        if hasattr(env,'close_completion_events'):
                            env.close_completion_events.control_step=timesteps//args.num_envs+rollout_step+1
                        next_obs,reward,done,extra=env.step(action.clamp(-1,1))
                        obs=observations(next_obs)
                        algorithm.process_env_step(obs,reward,done,{})
                        reward_sum+=reward.mean()
                        occupancy+=torch.bincount(extra['phase'],minlength=4)
                        relief_fraction+=extra['pressure_limited'].float().mean()
                        for name,value in extra.get('terms',{}).items():
                            terms[name]=terms.get(name,torch.zeros((),device=args.device))+value.mean()
                        failures=extra.get('failure')
                        if failures is not None: failure_counts+=torch.bincount(failures[done],minlength=len(failure_counts))
                    algorithm.compute_returns(obs)
            if str(args.device).startswith('cuda'): torch.cuda.synchronize()
            collect_seconds=time.monotonic()-collect
            learning=time.monotonic()
            losses=algorithm.update()
            if str(args.device).startswith('cuda'): torch.cuda.synchronize()
            learn_seconds=time.monotonic()-learning
            if not all(math.isfinite(float(v)) for v in losses.values()):
                raise RuntimeError('nonfinite PPO loss')
            iteration+=1; timesteps+=args.num_envs*steps
            row=dict(status='training',backend=VERSION,num_envs=args.num_envs,
                iteration=iteration,rollout_steps=steps,timesteps=timesteps,
                collection_seconds=collect_seconds,learning_seconds=learn_seconds,
                iteration_seconds=collect_seconds+learn_seconds,
                transitions_per_second=args.num_envs*steps/(collect_seconds+learn_seconds),
                elapsed_seconds=time.monotonic()-started,episodes=int(env.episodes.sum()),
                episodes_per_env_min=int(env.episodes.min()),episodes_per_env_max=int(env.episodes.max()),
                episode_budget_per_env=(None if getattr(args,'max_iterations',None) is not None else args.episodes_per_env),
                iteration_budget=getattr(args,'max_iterations',None),successes=int(env.successes.sum()),
                recovered_physical_episodes=int(env.recovered.sum()),
                combined_failure_resets=int(env.combined_failure_resets),
                joint_limit_resets=int(env.joint_limit_resets),
                object_drop_resets=int(env.object_drop_resets),
                numerical_resets=int(env.numerical_resets),
                phase_occupancy=(occupancy/(steps*args.num_envs)).cpu().tolist(),
                phase_reached_episodes=env.reached_counts.cpu().tolist(),
                phase_transition_counts=env.transition_counts.reshape(4,4).cpu().tolist(),
                mean_step_reward=float(reward_sum/steps),reward_terms={k:float(v/steps) for k,v in terms.items()},
                failures={FAILURE_NAMES[i]:int(v) for i,v in enumerate(failure_counts.cpu()) if i and int(v)},
                hand_load_mean_n=task.m['hand_object_loads'].mean(0).cpu().tolist(),
                max_object_bottom_m=float(task.m['bottom'].max()),losses=losses,
                learning_rate=algorithm.learning_rate,grasp_qualified=False)
            std=algorithm.policy.log_std.detach().exp()
            if getattr(task,'free_grasp_lift',False):
                from .tabletop_free_grasp_lift import bilateral_contact
                for key in ('phase_occupancy','phase_reached_episodes','phase_transition_counts'):
                    row.pop(key,None)
                row['exploration_mode']='continuous_free_grasp_lift'
                row['bilateral_contact_env_fraction']=float(bilateral_contact(task).float().mean())
            simulated_seconds=steps*device_world.decimation*device_world.dt
            row.update(simulated_seconds_per_env_per_iteration=simulated_seconds,
                collection_realtime_factor=simulated_seconds/collect_seconds,
                training_realtime_factor=simulated_seconds/(collect_seconds+learn_seconds),
                physics_hz=args.physics_hz, control_hz=args.physics_hz/device_world.decimation,
                wall_clock_pacing=False)
            if policy_recovery is not None:
                row['policy_action_recovery']=dict(policy_recovery.counts)
            if 'grasp_controller' in contract:row['grasp_controller']=contract['grasp_controller']
            if 'palm_transport' in contract:row['palm_transport']=contract['palm_transport']
            if 'palm_reposition' in contract:
                row['palm_reposition']=contract['palm_reposition']
                row['force_direction_reposition']=bool(getattr(task,'force_direction_reposition',False))
                adjustment=task.palm_reposition
                row['reposition']=reposition_telemetry(adjustment)
            row['policy_noise']=dict(POLICY_NOISE_CONTRACT)
            row['vertical_contact_force']=env.vertical_force_stats.summary()
            if getattr(task,'force_region',False):
                actual=task.m['hand_object_loads']
                target=task.f_requested
                valid_hand=task.m['contact_valid']
                below_hand=valid_hand&task.m['contact_all_below_centerline']
                row['force_control']=dict(requested_min_n=float(target.min()),requested_max_n=float(target.max()),
                    below_axis_contact_fraction=float(below_hand.sum()/valid_hand.sum().clamp_min(1)),
                    active_hold_envs=int(task.policy_lift_hold.active.sum()) if hasattr(task,'policy_lift_hold') else 0)
                for i,side in enumerate(('left','right')):
                    row['force_control'][side+'_requested_mean_n']=float(target[:,i].mean())
                    row['force_control'][side+'_measured_mean_n']=float(actual[:,i].mean())
                    row['force_control'][side+'_error_mean_n']=float((target[:,i]-actual[:,i]).abs().mean())
            row['lift_entry_force']=env.lift_entry_force_stats.summary()
            if hasattr(env,'close_completion_events'):
                row['close_completion_event_index']=env.close_completion_events.event_index
                row['close_completion_telemetry']=env.close_completion_events.diagnostics()
            if task.vertical_force_reward:
                row['vertical_force_reward']=contract['vertical_force_reward']
                row['force_reward_enabled_env_fraction']=float(task.force_scoring_started.float().mean())
                row['downward_penalty_enabled_env_fraction']=row['force_reward_enabled_env_fraction']
            if task.close_downward_penalty and not getattr(task,'free_grasp_lift',False):
                row['close_downward_penalty']=contract['close_downward_penalty']
            if task.shaping is not None:
                row['grasp_shaping']=contract['grasp_shaping']
                row['lift_gates']=lift_gate_summary(env.lift_gate_counts)
                row['max_lift_command_offset_m']=(float(device_world.palms[:,:,2].max())
                    if getattr(task,'policy_control',False) else
                    float((task.offset-device_world.initial_offset).max()))
            row['policy_std_min']=float(std.min())
            row['policy_std_max']=float(std.max())
            row['action_clipped_fraction']=float(clipped_fraction/steps)
            if task.grasp_readiness is not None and not getattr(task,'free_grasp_lift',False):
                row['grasp_readiness']=contract['grasp_readiness']
                row['load_ready_env_fraction']=float(task.grip_force_ready(task.m['hand_object_loads']).float().mean())
                row['max_grasp_contact_time_s']=float(task.contact_time.max())
            row['physics_peaks_since_start']=dict(zip(PHYSICS_PEAK_NAMES,env.physics_peaks.cpu().tolist()))
            row['velocity_diagnostics']=env.velocity_diagnostics()
            row['fault_trace_record_error']=env.fault_trace_error
            row['pressure_relief_hand_fraction']=float(relief_fraction/steps)
            row['observation_size']=obs['policy'].shape[-1]
            if window is not None:
                row['contact_reward_window']=contract['contact_reward_window']
                if task.vertical_force_reward_mode!='continuous':
                    row['contact_window_expired_fraction']=float((task.contact_reward_clock>=1.).float().mean())
                row['contact_reward_paid_max']=float(task.contact_reward_paid.max())
            if torch.cuda.is_available():
                free,total=torch.cuda.mem_get_info(); row['gpu_memory_used_gib']=(total-free)/1024**3
            save_checkpoint(checkpoint,algorithm,contract,iteration,timesteps,env)
            for name,value in dict(collection_seconds=collect_seconds,learning_seconds=learn_seconds,
                fps=row['transitions_per_second'],mean_step_reward=row['mean_step_reward'],
                successes=row['successes'],episodes=row['episodes']).items(): writer.add_scalar(name,value,iteration)
            for name,value in losses.items(): writer.add_scalar('loss/'+name,value,iteration)
            if getattr(task,'free_grasp_lift',False):
                writer.add_scalar('exploration/bilateral_contact_env_fraction',row['bilateral_contact_env_fraction'],iteration)
                writer.add_scalar('exploration/max_actual_lift_m',max(0.,row['max_object_bottom_m']-.30),iteration)
            if task.vertical_force_reward:
                writer.add_scalar('force_reward/enabled_env_fraction',row['force_reward_enabled_env_fraction'],iteration)
                writer.add_scalar('force_reward/downward_penalty_enabled_env_fraction',row['downward_penalty_enabled_env_fraction'],iteration)
            write_reward_scalars(writer,row['reward_terms'],row['mean_step_reward'],iteration)
            for key,value in row.get('force_control',{}).items():
                writer.add_scalar('force_control/'+key,value,iteration)
            writer.add_scalar('recovery/combined_failure_resets',row['combined_failure_resets'],iteration)
            writer.add_scalar('recovery/joint_limit_resets',row['joint_limit_resets'],iteration)
            writer.add_scalar('recovery/object_drop_resets',row['object_drop_resets'],iteration)
            writer.add_scalar('recovery/numerical_resets',row['numerical_resets'],iteration)
            for name,value in row.get('reposition',{}).items():writer.add_scalar('reposition/'+name,value,iteration)
            if task.shaping is not None:
                for name,value in row['lift_gates'].items():
                    if name.endswith('_fraction') and value is not None:writer.add_scalar('lift_gates/'+name,value,iteration)
                writer.add_scalar('lift_gates/lift_substeps',row['lift_gates']['lift_substeps'],iteration)
                writer.add_scalar('lift_gates/max_command_offset_m',row['max_lift_command_offset_m'],iteration)
            if not getattr(task,'free_grasp_lift',False):
                for phase in range(1,4): writer.add_scalar('phase/'+('approach','close','hold','lift')[phase],row['phase_occupancy'][phase],iteration)
            for name,value in row['physics_peaks_since_start'].items():writer.add_scalar('physics_peak/'+name,value,iteration)
            for name,value in row['velocity_diagnostics'].items():writer.add_scalar('velocity_warning/'+name,value,iteration)
            for name in ('policy_std_min','policy_std_max','action_clipped_fraction'):
                writer.add_scalar('policy/'+name,row[name],iteration)
            vertical=row['vertical_contact_force']
            from .tabletop_vertical_force import write_directional_force_scalars,write_lift_entry_force_scalars
            write_directional_force_scalars(writer,vertical,iteration)
            write_lift_entry_force_scalars(writer,row['lift_entry_force'],iteration)
            if hasattr(env,'close_completion_events'):
                writer.add_scalar('close_completion_count/累计完成事件数',env.close_completion_events.event_index,iteration)
            for name in ('hands_total_mean_n','weight_mean_n','force_minus_weight_mean_n',
                         'above_weight_sample_fraction','hands_total_peak_n'):
                if vertical[name] is not None:writer.add_scalar('vertical_force/'+name,vertical[name],iteration)
            for hi,side in enumerate(('left','right')):
                for key in ('hand_normal_mean_n','hand_friction_mean_n','hand_total_mean_n'):
                    writer.add_scalar('vertical_force/'+side+'/'+key,vertical[key][hi],iteration)
            writer.flush()
            for key in ('collection_realtime_factor','training_realtime_factor',
                        'simulated_seconds_per_env_per_iteration'):
                writer.add_scalar('timing/'+key,row[key],iteration)
            if hasattr(env,'close_completion_events'):
                row['close_completion_telemetry']=env.close_completion_events.diagnostics()
            write_json(args.output/'monitor.json',row)
            with (args.output/'iterations.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            print('TENSOR_PPO_ITERATION',json.dumps(row),flush=True)
            if training_budget_reached(args,iteration,timesteps,int(env.episodes.min())):break
    except BaseException as error:
        stop=error
        # Save only the last completed optimizer iteration. The partial rollout
        # is intentionally discarded; its counters must not overwrite checkpoint.
        # Publish the physical fault before potentially slow logger cleanup.
        stopped_telemetry=(env.close_completion_events.diagnostics()
            if hasattr(env,'close_completion_events') else None)
        publish_stopped_monitor(args.output/'monitor.json',error,env.physics_peaks,
            iteration,timesteps,args.num_envs,close_completion_telemetry=stopped_telemetry,
            velocity_diagnostics=env.velocity_diagnostics,
            policy_action_recovery=policy_recovery.counts if policy_recovery is not None else None)
        raise
    finally:
        try:
            writer.close()
        finally:
            final_telemetry=(env.close_completion_events.diagnostics()
                if hasattr(env,'close_completion_events') else None)
            if stop is not None and final_telemetry!=stopped_telemetry:
                publish_stopped_monitor(args.output/'monitor.json',stop,env.physics_peaks,
                    iteration,timesteps,args.num_envs,
                    close_completion_telemetry=final_telemetry,velocity_diagnostics=env.velocity_diagnostics,
                    policy_action_recovery=policy_recovery.counts if policy_recovery is not None else None)
    row['status']='training_budget_complete'
    if hasattr(env,'close_completion_events'):
        row['close_completion_telemetry']=env.close_completion_events.diagnostics()
    write_json(args.output/'monitor.json',row)
    result=dict(status='training_budget_complete',tensor_training=VERSION,iteration=iteration,
                timesteps=timesteps,successes=int(env.successes.sum()),grasp_qualified=False,
                tensor_setup_resolved=True)
    if policy_recovery is not None:
        result['policy_action_recovery']=dict(policy_recovery.counts)
    return result
