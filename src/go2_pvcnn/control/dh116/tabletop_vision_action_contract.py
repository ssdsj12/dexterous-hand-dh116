"""Explicit v3 action semantics shared by training and offline derivation.

The inherited critic contract is provenance only. No feedback column is added
to the 3844D sensor actor and the physical 24D command contract is unchanged.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

LEGACY_VERSION = 'dh116-vision-student-ppo-v2-bounded'
SQUASHED_VERSION = 'dh116-vision-student-ppo-v3-squashed-gaussian'
_ADDED = ('action_distribution', 'rollout_storage', 'action_diagnostics',
          'latent_noise', 'physics_reward_provenance', 'rng_coverage')


def upgrade_action_contract(base_contract):
    """Derive v3 metadata from an otherwise unchanged, valid v2 contract."""
    from .tabletop_vision_resume import _validate_contract
    from .tabletop_squashed_gaussian import DISTRIBUTION_CONTRACT
    from .tabletop_vision_squashed_ppo import STORAGE_CONTRACT
    _validate_contract(base_contract)
    if any(key in base_contract for key in _ADDED) or 'actor_output' in base_contract.get('network', {}):
        raise ValueError('action derivation requires an unmodified v2 contract')
    value = deepcopy(base_contract)
    value['version'] = SQUASHED_VERSION
    value.setdefault('network', {})['actor_output'] = 'unbounded_latent_mean'
    value['action_distribution'] = deepcopy(DISTRIBUTION_CONTRACT)
    value['rollout_storage'] = deepcopy(STORAGE_CONTRACT)
    value['latent_noise'] = dict(parameter='log_std', initial_std=.1, min_std=.001,
        max_std=.3, units='pre_tanh_latent', projection='after_each_Adam_step',
        physical_scales='unchanged_ACTION_CONTRACT')
    value['action_diagnostics'] = dict(saturation_threshold=.99,
        saturation_comparison='abs_action_greater_than_or_equal_to_threshold',
        bounded_action_statistics='population_over_rollout_steps_and_environments',
        groups=dict(fingers=list(range(12)), world_xyz=list(range(12,18)),
                    intrinsic_xyz_rotation=list(range(18,24))),
        frozen_log_prob_check='complete_rollout_before_first_Adam_step',
        new_log_prob_statistics='sample_weighted_evaluations_during_optimizer_updates',
        post_update_latent_std_per_dim='after_final_Adam_step_and_noise_projection')
    critic = value.get('privileged_critic_contract', {})
    digest = hashlib.sha256(json.dumps(critic, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    value['physics_reward_provenance'] = dict(
        source='unchanged_inherited_privileged_critic_contract',
        privileged_critic_contract_sha256=digest,
        authored_configuration_sha256={key: deepcopy(base_contract[key]) for key in
            ('camera_config_sha256', 'primary_reset_sha256', 'secondary_reset_sha256')
            if key in base_contract},
        critic_authored_configuration_sha256={key: deepcopy(item) for key, item in critic.items()
            if 'sha256' in key}, physical_action_contract='unchanged')
    value['rng_coverage'] = dict(saved=['torch_cpu', 'torch_cuda_all_available_devices'],
        python_random='not_saved_by_historical_v2_or_this_v3',
        numpy_random='not_saved_by_historical_v2_or_this_v3',
        exact_resume_scope='saved_model_Adam_LR_Torch_RNG_counters;scene_resets_at_launch')
    return value


def assert_finite_state(value, path='checkpoint'):
    """Reject invalid numeric state without modifying any supplied value."""
    import torch
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'nonfinite saved state at {path}')
    elif isinstance(value, dict):
        for key, item in value.items(): assert_finite_state(item, f'{path}.{key}')
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value): assert_finite_state(item, f'{path}[{index}]')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'nonfinite saved state at {path}')


def validate_squashed_checkpoint(saved, current_contract=None):
    """Validate exact semantics and finite saved state before scene or file I/O."""
    import torch
    from .tabletop_vision_resume import _same, _validate_saved
    from .tabletop_vision_checkpoint_state import validate_visual_checkpoint_state
    if not isinstance(saved, dict) or saved.get('version') != SQUASHED_VERSION:
        raise ValueError('visual resume requires v3 tanh_gaussian; use explicit offline v2 derivation')
    contract = saved.get('contract')
    if not isinstance(contract, dict) or contract.get('version') != SQUASHED_VERSION:
        raise ValueError('visual resume requires the v3 action contract')
    from .tabletop_vision_point_activation import point_activation_from_contract
    point_activation_from_contract(contract)
    base = deepcopy(contract)
    base['version'] = LEGACY_VERSION
    for name in _ADDED: base.pop(name, None)
    if isinstance(base.get('network'), dict): base['network'].pop('actor_output', None)
    if not _same(contract, upgrade_action_contract(base)):
        raise ValueError('visual resume incompatible v3 action contract')
    if current_contract is not None and not _same(contract, current_contract):
        raise ValueError('visual resume requires the same complete v3 contract')
    legacy = dict(saved, version=LEGACY_VERSION, contract=base)
    _validate_saved(legacy)
    assert_finite_state(saved)
    rng = saved['torch_rng']
    if rng.dtype != torch.uint8 or rng.ndim != 1:
        raise ValueError('visual resume requires Torch CPU RNG byte state')
    cuda = saved['cuda_rng']
    if not isinstance(cuda, list) or any(not isinstance(x, torch.Tensor)
        or x.dtype != torch.uint8 or x.ndim != 1 for x in cuda):
        raise ValueError('visual resume requires Torch CUDA RNG byte states')
    std = saved['policy'].get('log_std')
    if not isinstance(std, torch.Tensor) or std.shape != (24,):
        raise ValueError('visual resume requires a 24D latent log_std')
    validate_visual_checkpoint_state(saved)
    return deepcopy(saved)


def _validate_static_resume_contract(args, contract, reset_sha256):
    """Compare launch-known fields; native render/scene readbacks stay deferred.

    The front launcher can supply hashes of reset JSON serialized in memory.
    Candidate JSON is never mistaken for the physical initialization payload.
    """
    from .tabletop_vision_resume import _same
    from .tabletop_vision_quaternion import validate_wrist_quaternion_preprocessing
    requested=validate_wrist_quaternion_preprocessing(getattr(args,'vision_wrist_quaternion_preprocessing','raw'))
    authored=validate_wrist_quaternion_preprocessing(contract.get('actor_wrist_quaternion_preprocessing','raw'))
    if requested!=authored:
        raise ValueError('visual resume incompatible wrist quaternion preprocessing contract')
    missing = object()
    def check(path, expected):
        current = contract
        for part in path.split('.'):
            current = current.get(part, missing) if isinstance(current, dict) else missing
        if not _same(current, expected):
            raise ValueError('visual resume incompatible static contract field: '+path)
    def sha(path):
        try: return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError as error: raise ValueError('visual resume cannot read configuration: '+str(path)) from error
    n = getattr(args,'num_envs',None)
    if n is not None:
        check('num_envs',n)
        check('groups',[dict(label='pinch_thumb_index',num_envs=n//2),
                        dict(label='tripod_thumb_index_middle',num_envs=n//2)])
        check('ppo.minibatches',min(8,n))
        check('memory_contract.logical_minibatches',min(8,n))
    if hasattr(args,'rollout_steps'):
        check('rollout_steps',args.rollout_steps if args.rollout_steps is not None else 24)
    if hasattr(args,'vision_width') and hasattr(args,'vision_height'):
        check('camera_resolution_wh',[args.vision_width,args.vision_height])
    for name, field in [('vision_ppo_microbatch_size','microbatch_size'),
                        ('vision_inference_batch_size','inference_batch_size')]:
        check('memory_contract.'+field,getattr(args,name,128))
    fixed = getattr(args,'vision_camera_mode','bounded') == 'fixed'
    if fixed:
        check('camera_mode','fixed_per_env')
    elif 'camera_mode' in contract:
        raise ValueError('visual resume incompatible static contract field: camera_mode')
    if n is not None:
        check('camera_batch_size',n if fixed else min(n,getattr(args,'vision_camera_batch_size',64)))
    visibility = 'all' if fixed else getattr(args,'vision_visibility_mode','all')
    check('visibility_mode',visibility)
    check('pinned_source_environment',0 if visibility=='batch' else None)
    check('camera_batching','fixed_per_environment_real_RGBD' if fixed
          else 'same_frozen_physics_state_sequential_real_RGBD')
    for name, field in [('depth_camera_config','camera_config_sha256'),
                        ('grasp_initialization','primary_reset_sha256'),
                        ('mixed_grasp_initialization','secondary_reset_sha256')]:
        if field in reset_sha256: check(field,reset_sha256[field])
        elif getattr(args,name,None) is not None: check(field,sha(getattr(args,name)))
    critic = contract.get('privileged_critic_contract',{})
    from .tabletop_lift_reward_experiment import experiment_contract
    lift_profile=getattr(args,'vision_lift_reward_experiment','R0')
    check('privileged_critic_contract.surface_lift_reward_experiment',
          missing if lift_profile=='R0' else experiment_contract(lift_profile))
    # These values are authored CLI/config inputs, not PhysX-derived measurements.
    for name in ('physics_hz','solver_type','position_iterations','passive_velocity_limit_rad_s',
                 'env_spacing_m','reset_height_jitter_m'):
        if hasattr(args,name) and name in critic:
            check('privileged_critic_contract.'+name,getattr(args,name))
    if hasattr(args,'velocity_iterations') and 'physics_hz' in critic:
        if not _same(critic.get('velocity_iterations',4),args.velocity_iterations):
            raise ValueError('visual resume incompatible static contract field: velocity_iterations')
    if hasattr(args,'mimic_frequency') and 'mimic_natural_frequency_s_inv' in critic:
        check('privileged_critic_contract.mimic_natural_frequency_s_inv',float(args.mimic_frequency))
    if getattr(args,'pose_summary',None) is not None and 'pose_sha256' in critic:
        check('privileged_critic_contract.pose_sha256',sha(args.pose_summary))
    if hasattr(args,'load_profile'):
        from .tabletop_load_parameters import read_load_parameters
        parameters = read_load_parameters(args.load_profile)
        check('privileged_critic_contract.load_parameters',parameters.as_dict() if parameters else missing)
    if hasattr(args,'surface_reward_profile'):
        from .tabletop_surface_reward_ablation import ablation_contract
        expected = ({} if args.surface_reward_profile=='baseline' else
                    {'surface_reward_ablation':ablation_contract(args.surface_reward_profile)})
        check('privileged_critic_contract.surface_reward_ablation',expected.get('surface_reward_ablation',missing))
    for name, key, function in [('object_material','object_contact_material','material_contract'),
                                ('hand_material','hand_contact_material','hand_material_contract')]:
        if hasattr(args,name):
            from . import tabletop_material
            expected = getattr(tabletop_material,function)(getattr(args,name))
            check('privileged_critic_contract.'+key,expected.get(key,missing))
    if hasattr(args,'object_kind'):
        from .tabletop_adaptive_initialization import object_spec
        _,_,physical=object_spec(Path(__file__).resolve().parents[4],args.object_kind,
            solid_radius_m=getattr(args,'solid_cylinder_radius_m',.008),
            solid_density_kg_m3=getattr(args,'solid_cylinder_density_kg_m3',None))
        # The original .008 / density=None solid returns {}. Absence is its
        # historical contract, rather than a reason to omit this comparison.
        for key in ('object','object_kind','object_radius_m','object_height_m',
                    'object_mass_kg','object_density_kg_m3'):
            check('privileged_critic_contract.'+key,physical.get(key,missing))
    if hasattr(args,'solve_articulation_contact_last'):
        from .tabletop_solver_order import CONTRACT as SOLVER_ORDER
        check('privileged_critic_contract.articulation_solver_order',
              SOLVER_ORDER if args.solve_articulation_contact_last else missing)
    if hasattr(args,'end_support_clearance_m'):
        clearance=float(args.end_support_clearance_m)
        if clearance==0:
            check('privileged_critic_contract.end_support_scene',None)
        else:
            check('privileged_critic_contract.end_support_scene.clearance_m',clearance)
            radius=(getattr(args,'solid_cylinder_radius_m',.008)
                    if getattr(args,'object_kind','solid_cylinder')=='solid_cylinder' else .008)
            check('privileged_critic_contract.end_support_scene.wire_radius_m',float(radius))
            check('privileged_critic_contract.end_support_scene.wire_length_m',.5)


def validate_vision_action_options(args, *, reset_sha256=None):
    """Pre-Kit validation of saved state and every launch-known visual field.

    Full native critic and rendering readbacks are validated again by runtime
    before its camera, reset, or visual metadata. The v2 default is unchanged.
    """
    mode = getattr(args, 'vision_action_distribution', 'gaussian_clip')
    if mode not in ('gaussian_clip', 'tanh_gaussian'):
        raise ValueError('unknown vision action distribution')
    from .tabletop_vision_quaternion import validate_wrist_quaternion_preprocessing
    preprocessing=validate_wrist_quaternion_preprocessing(getattr(args,'vision_wrist_quaternion_preprocessing','raw'))
    if preprocessing!='raw' and mode!='tanh_gaussian':
        raise ValueError('versioned wrist quaternion preprocessing requires tanh_gaussian')
    initial_audit=getattr(args,'vision_experiment_initial_state_audit',False)
    initial_reference=getattr(args,'vision_experiment_initial_state_reference',None)
    if initial_reference and not initial_audit:
        raise ValueError('initial-state reference requires initial-state audit')
    if initial_audit and (mode!='tanh_gaussian' or not getattr(args,'vision_student',False)
            or getattr(args,'vision_evaluation_checkpoint',None)):
        raise ValueError('initial-state audit requires native v3 vision training')
    if initial_reference and not getattr(args,'resume',None):
        raise ValueError('initial-state audit reference requires an explicit resume source')
    from .tabletop_lift_reward_experiment import PROFILES
    profile=getattr(args,'vision_lift_reward_experiment','R0')
    if profile not in PROFILES:
        raise ValueError('unknown isolated lift reward experiment')
    if profile!='R0' and (mode!='tanh_gaussian' or
            not getattr(args,'vision_student',True) or
            getattr(args,'surface_reward_profile',None)!='support_deficit'):
        raise ValueError('lift reward experiments require v3 visual PPO and support_deficit')
    warm = getattr(args, 'vision_warm_start_optimizer', 'preserve')
    if warm not in ('preserve', 'reset'):
        raise ValueError('unknown visual optimizer warm-start mode')
    interval = getattr(args, 'vision_checkpoint_interval', 10)
    if type(interval) is not int or interval < 1:
        raise ValueError('visual checkpoint interval must be a positive integer')
    if warm == 'reset' and (mode != 'tanh_gaussian' or not getattr(args, 'resume', None)):
        raise ValueError('reset Adam requires an explicit v3 tanh_gaussian warm-start checkpoint')
    if mode == 'gaussian_clip': return
    if not getattr(args, 'vision_student', True):
        raise ValueError('--vision-action-distribution tanh_gaussian requires --vision-student')
    if getattr(args, 'vision_resume_downsize', False):
        raise ValueError('v3 tanh_gaussian downsize resume is unsupported')
    if getattr(args, 'resume', None):
        import torch
        saved=validate_squashed_checkpoint(torch.load(args.resume,map_location='cpu',weights_only=False))
        _validate_static_resume_contract(args,saved['contract'],reset_sha256 or {})
        budget=getattr(args,'max_iterations',getattr(args,'updates',saved['iteration']+1))
        if warm == 'reset' and budget <= saved['iteration']:
            raise ValueError('reset Adam warm-start requires at least one complete new update')


def validate_new_visual_output(output, *, preflight_path=None):
    """Accept empty direct-API outputs or this launcher's preflighted new run."""
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        # The native launcher creates bootstrap telemetry after rejecting every
        # existing output, before constructing Isaac. Bind that authorization to
        # its resolved path; never infer it from monitor contents.
        if preflight_path != str(output.resolve()) or (output/'vision_student_contract.json').exists():
            raise FileExistsError(output)
