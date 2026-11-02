"""Canonical surface action semantics and opt-in completed-interval feedback.

Legacy artifacts require an explicit, audited derivation. Dimensions alone are
never sufficient evidence for importing a Teacher action sequence.
"""
import copy
import hashlib
import json

from .linear_coupling import ACTIVE

LEGACY_OBSERVATION_VERSION = 'dh116-surface-sharing-lift-objective-v2'
OBSERVATION_VERSION = 'dh116-surface-sharing-feedback-371-v1'
ACTION_NAME = '24D_free_fingers_worldXYZ_intrinsicXYZ_surface_sharing_v1'
ACTION_CONTRACT = dict(
    version='dh116-surface-motion24-v1', size=24,
    joint_order=[hand + '.' + joint for hand in ('left', 'right') for joint in ACTIVE],
    slices=dict(joint_velocity=[0,12], world_translation_velocity=[12,18], intrinsic_rotation_velocity=[18,24]),
    hand_order=['left','right'], axis_order=['X','Y','Z'],
    joint_rate_rad_s=.3, translation_rate_m_s=.01, rotation_rate_rad_s=.5,
    normalized_range=[-1.,1.], wrist_vector_norm_limit=1.,
    translation_frame='world;positive_Z_up',
    rotation='seed_wxyz_times_intrinsic_X_then_Y_then_Z',
    integration='existing_native_physics_dt;targets_integrated_at_each_substep',
    zero_action='retain_current_targets_subject_to_existing_joint_lead_and_table_safety_projection',
    guards='existing_joint_limits_lead_velocity_table_clearance;no_added_force_servo',
    force_action_dimensions=0)

_FIELDS = (
    ('support_n',2,10.,'N','accepted_pad_normal_plus_attributable_native_friction'),
    ('support_margin',1,1.,'weight_fraction','(sum_signed_support-object_weight)/object_weight'),
    ('support_imbalance',1,1.,'ratio','absolute_bilateral_support_difference/sum_absolute_support'),
    ('paired',2,1.,'boolean','native_effective_opposed_pad_pair_per_hand'),
    ('ordinary_fraction',1,1.,'ratio','rejected_native_load/(accepted+rejected)'),
    ('pad_depth',1,.0008,'m','maximum_accepted_native_pad_depth'),
    ('anchor_drift',1,.0005,'m','native_fixed_anchor_drift'),
    ('slip_speed',1,.01,'m/s','rod_points_in_measured_wrists_velocity'),
    ('hold_duration',1,5.,'s','last_completed_native_success_window'),
    ('window_good',1,1.,'boolean','last_completed_native_success_window'),
    ('pressure_limited',2,1.,'boolean','actual_existing_controller_pressure_limited_flags'),
    ('mimic_blocked',2,1.,'boolean','actual_existing_block_flags_with_severe_measured_mimic_error'),
    ('wrist_z_request',2,.01,'m/s','submitted_policy_action_14_17_times_existing_wrist_rate'),
    ('wrist_z_increment',2,.001,'m','measured_wrist_pose_end_minus_start'),
)


def feedback_schema():
    start=351; fields=[]
    for name,size,scale,unit,source in _FIELDS:
        fields.append(dict(name=name,slice=[start,start+size],scale=scale,unit=unit,source=source))
        start+=size
    assert start==371
    return fields


def semantic_sha256(contract):
    """Digest full semantics; only known batch-size repetitions are canonical."""
    value=copy.deepcopy(contract)
    value.pop('surface_contract_sha256',None)
    for key in ('num_envs','rollout_steps','minibatches'):
        value.pop(key,None)
    for record,key in ((value,'success_bottom_m'),(value.get('surface_force_sharing',{}),'lift_reference_bottom_m')):
        heights=record.get(key)
        if isinstance(heights,list) and heights and all(type(x) in (int,float) and x==heights[0] for x in heights):
            record[key]={'uniform_value':heights[0]}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _schema():
    return dict(version='dh116-surface-contract-v1',action=copy.deepcopy(ACTION_CONTRACT),
        base_observation=dict(size=351,version=LEGACY_OBSERVATION_VERSION,
            normalization='unchanged_legacy_351_columns;no_running_statistics',
            previous_action='existing_351_free_wrist_action_columns'),
        feedback_fields=feedback_schema(),
        alignment='observation_t_contains_completed_interval_t-1;action_t_executes_next_interval',
        aggregation='native_dt_weighted_mean;depth_and_drift_max;hold_at_last_completed_window',
        reset='selected_rows_zero;no_pre_reset_samples_in_next_interval',
        invalid='zero_diagnostic_features_and_abort_rollout_before_any_PPO_or_BC_update',
        clipping='none;physical_scales_only;finite_out_of_scale_values_remain_observable')


def _validate_motion(contract):
    if not isinstance(contract,dict):raise ValueError('surface contract dictionary required')
    dimensions=[contract[k] for k in ('action_size','policy_command_size') if k in contract]
    if not dimensions or any(type(x) is not int or x!=24 for x in dimensions):
        raise ValueError('explicit 24D motion contract required; no 26D conversion')
    if contract.get('action')!=ACTION_NAME:raise ValueError('unknown surface action order/semantics')
    expected_order='joint_velocity12_world_translation_velocity6_intrinsic_XYZ_velocity6'
    for key in ('policy_control','grasp_controller','free_wrist_grasp'):
        nested=contract.get(key,{})
        if ('action_size' in nested and nested['action_size']!=24
                or 'action_order' in nested and nested['action_order']!=expected_order):
            raise ValueError('contradictory 24D action order: '+key)
        for field,expected in (('joint_rate_rad_s',.3),('wrist_rate_m_s',.01),('rotation_rate_rad_s',.5)):
            if field in nested and nested[field]!=expected:raise ValueError('action scale mismatch: '+field)
    if contract.get('control_hz',60)!=60 or contract.get('physics_hz',480)!=480:
        raise ValueError('audited 480Hz physics / 60Hz control required')


def bind_contract(legacy_contract):
    """Explicit metadata derivation; never converts actions or fabricates inputs."""
    _validate_motion(legacy_contract)
    if legacy_contract.get('observation_size')!=351 or legacy_contract.get('observation_version')!=LEGACY_OBSERVATION_VERSION:
        raise ValueError('only the original versioned 351D input can be explicitly extended')
    if any(k in legacy_contract for k in ('surface_contract','surface_contract_sha256')):
        raise ValueError('source already has surface schema metadata; ambiguous migration')
    value=copy.deepcopy(legacy_contract)
    value.update(observation_size=371,observation_version=OBSERVATION_VERSION,surface_contract=_schema())
    value['surface_contract_sha256']=semantic_sha256(value)
    return value


def validate_contract(contract):
    _validate_motion(contract)
    if contract.get('observation_size')!=371 or contract.get('observation_version')!=OBSERVATION_VERSION:
        raise ValueError('versioned 371D feedback contract required')
    if contract.get('surface_contract')!=_schema():raise ValueError('surface action/observation schema mismatch')
    if contract.get('surface_contract_sha256')!=semantic_sha256(contract):raise ValueError('surface contract SHA mismatch')
    return contract


def validate_actions(actions):
    import torch
    if (not isinstance(actions,torch.Tensor) or actions.ndim!=2 or actions.shape[-1]!=24
            or not actions.is_floating_point() or not torch.isfinite(actions).all()
            or (actions.abs()>1).any()):
        raise ValueError('finite bounded normalized [N,24] actions required; no silent conversion')
    return actions
