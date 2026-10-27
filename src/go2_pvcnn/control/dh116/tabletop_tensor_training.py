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


def tensor_contract_metadata(contract, configured_max_force_n):
    """Add the tensor policy contract fields to an existing full-hand contract."""
    from .tabletop_tensor_task import (PALM_XY_SEARCH_LIMIT_M, PALM_Z_SEARCH_LIMIT_M,
        PALM_XY_EXPLORATION_WEIGHT, POLICY_INITIAL_FORCE_ACTION_MEAN,
        POLICY_INITIAL_FORCE_TARGET_N, POLICY_MIN_FORCE_N, POLICY_MAX_FORCE_N)
    enriched=copy.deepcopy(contract)
    enriched.update(
        force_action_range_n=[POLICY_MIN_FORCE_N, POLICY_MAX_FORCE_N],
        force_action_semantics=dict(version='phase_force_channels_v1',
            close='signed_object_filtered_normal_vertical_sum',
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
        contract['palm_exploration']=exploration
    return contract


def publish_stopped_monitor(path,error,physics_peaks,iteration,timesteps,num_envs):
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
        error.add_note('Could not publish stopped monitor: '+str(publication_error))


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
    if (saved.get('force_action_range_n')==[.1,12.] and
            current.get('force_action_range_n')==[POLICY_MIN_FORCE_N,POLICY_MAX_FORCE_N]):
        changes.append('force_action_range_2_to_12')
        candidate['force_action_range_n'] = current['force_action_range_n']
    from .tabletop_coordinated_grasp import CONTRACT as CONTROL_CONTRACT
    if 'grasp_controller' not in saved and current.get('grasp_controller')==CONTROL_CONTRACT:
        candidate['grasp_controller']=dict(CONTROL_CONTRACT);changes.append(CONTROL_CONTRACT['version'])
    from .tabletop_palm_mount import CONTRACT as WRIST_CONTRACT
    if ('palm_transport' not in saved and current.get('palm_transport')==WRIST_CONTRACT
            and current.get('grasp_controller')==CONTROL_CONTRACT):
        candidate['palm_transport']=dict(WRIST_CONTRACT);changes.append(WRIST_CONTRACT['version'])
    from .tabletop_loaded_palm_motion import CONTRACT as REPOSITION_CONTRACT
    if ('palm_reposition' not in saved and current.get('palm_reposition')==REPOSITION_CONTRACT
            and current.get('grasp_controller')==CONTROL_CONTRACT):
        candidate['palm_reposition']=dict(REPOSITION_CONTRACT);changes.append(REPOSITION_CONTRACT['version'])

    from .tabletop_vertical_force import vertical_reward_contract
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
        if all(saved.get(k)==current.get(k) for k in (saved.keys()|current.keys())-excluded):
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
        ('reset_velocity_guard',VELOCITY_GUARD_CONTRACT,'add_reset_velocity_tolerance'),
        ('live_velocity_guard',LIVE_VELOCITY_GUARD_CONTRACT,'add_live_substep_velocity_guard'),
        ('terminal_reset',TERMINAL_RESET_CONTRACT,'add_immediate_validated_terminal_reset')):
        if name not in saved and current.get(name)==expected:
            candidate[name]=expected;changes.append(change)
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
    for key,value in actual.items():
        if key=='joint_failure': continue
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
    return dict(maximum_reference_errors=errors,asymmetric_mapping=True,selected_reset_isolation=True)


def observations(value):
    return TensorDict({'policy':value},batch_size=[len(value)],device=value.device)


def make_algorithm(obs, steps=24):
    policy = ActorCritic(obs,{'policy':['policy'],'critic':['policy']},20,
        actor_hidden_dims=[128,128],critic_hidden_dims=[128,128],activation='tanh',
        init_noise_std=math.exp(-1.2),noise_std_type='log').to(obs.device)
    # Preserve bounded policy mean and initially gentle requested load. Gaussian
    # actions are clipped only at the actuator boundary, as in the SB3 baseline.
    for network in (policy.actor,policy.critic):
        linear = [m for m in network if isinstance(m,torch.nn.Linear)]
        for layer in linear:
            torch.nn.init.orthogonal_(layer.weight,math.sqrt(2)); torch.nn.init.zeros_(layer.bias)
        torch.nn.init.orthogonal_(linear[-1].weight,.01 if network is policy.actor else 1.)
    with torch.no_grad(): policy.actor[-1].bias[-2:] = math.atanh(-.95)
    policy.actor.append(torch.nn.Tanh())
    minibatches = min(8,len(obs))
    algorithm = PPO(policy,num_learning_epochs=5,num_mini_batches=minibatches,
        learning_rate=3e-4,gamma=.995,lam=.95,clip_param=.2,entropy_coef=.005,
        value_loss_coef=.5,max_grad_norm=.5,schedule='adaptive',desired_kl=.02,
        device=str(obs.device))
    algorithm.init_storage('rl',len(obs),steps,obs,[20])
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
        ('release_timeout_fraction', 'release_timeout'),
        ('recontact_abort_fraction', 'recontact_abort'),
        ('reclose_timeout_fraction', 'reclose_timeout'),
    ):
        values = getattr(adjustment, attribute_name, None)
        if values is not None:
            row[output_name] = float(values.double().mean())
    return row


def save_checkpoint(path, algorithm, contract, iteration, timesteps, env):
    payload=dict(version=VERSION,contract=contract,iteration=iteration,timesteps=timesteps,
        policy=algorithm.policy.state_dict(),optimizer=algorithm.optimizer.state_dict(),
        learning_rate=algorithm.learning_rate,episodes=env.episodes,successes=env.successes,
        recovered=env.recovered,reached_counts=env.reached_counts,transition_counts=env.transition_counts,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    tmp=Path(str(path)+'.tmp'); torch.save(payload,tmp); os.replace(tmp,path)


def run_tensor_training(world, template, args, contract):
    from .tabletop_tensor_task import TensorTask, REWARD_VERSION, FAILURE_NAMES, measured_readiness_contract, CLOSE_DOWNWARD_PENALTY_CONTRACT
    from .tabletop_tensor_world import TensorWorld, ROOT_POSE_WRITE_CONTRACT
    from .tabletop_tensor_env import TensorEnv, VELOCITY_GUARD_CONTRACT, TERMINAL_RESET_CONTRACT, PHYSICS_PEAK_NAMES, LIVE_VELOCITY_GUARD_CONTRACT
    from torch.utils.tensorboard import SummaryWriter
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
                    grasp_shaping=getattr(args,'grasp_shaping',False),
                    palm_exploration_axis=getattr(args,'palm_exploration_axis','xyz'))
    probe_contact_z=(getattr(args,'control_probe_mode',None)=='grasp_teacher_contact_z_sweep'
                     and getattr(args,'control_probe_checkpoint',None))
    if getattr(args,'grasp_controller','legacy')=='coordinated' or probe_contact_z:
        task.enable_coordinated_control(torch.ones(args.num_envs,dtype=torch.bool,device=args.device))
    if getattr(args,'release_reposition',False):
        from .tabletop_loaded_palm_motion import LoadedPalmReposition
        task.palm_reposition=LoadedPalmReposition(torch.ones(args.num_envs,dtype=torch.bool,device=args.device))
    if getattr(args,'force_direction_reposition',False):
        from .tabletop_force_direction_reposition import ForceDirectionReposition
        task.palm_reposition=ForceDirectionReposition(torch.ones(args.num_envs,dtype=torch.bool,device=args.device))
        task.force_direction_reposition=True
    env=TensorEnv(device_world,task,recovery=args.training_episode_resets)
    if task.shaping is not None:
        from .tabletop_grasp_shaping import LIFT_GATE_NAMES,lift_gate_summary
        env.lift_gate_counts=torch.zeros(len(LIFT_GATE_NAMES),device=args.device,dtype=torch.long)
    from .tabletop_vertical_force import VerticalForceStats, LiftEntryForceStats, CONTRACT as VERTICAL_FORCE_CONTRACT
    env.vertical_force_stats=VerticalForceStats(args.device)
    env.lift_entry_force_stats=LiftEntryForceStats(args.device)
    obs=observations(env.reset())
    validation=validate_world_adapter(world,device_world,task)
    obs=observations(env.reset())
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
        root_pose_write=ROOT_POSE_WRITE_CONTRACT,policy_noise=POLICY_NOISE_CONTRACT,
        vertical_force_telemetry=VERTICAL_FORCE_CONTRACT)
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
    apply_palm_exploration_contract(contract, task.palm_exploration_axis)
    if hasattr(task,'coordinated_control'):
        from .tabletop_coordinated_grasp import CONTRACT as CONTROL_CONTRACT
        contract['grasp_controller']=dict(CONTROL_CONTRACT)
    if getattr(args,'palm_transport','fixed_pose')=='driven_carriage' and (
            not args.control_probe_checkpoint or getattr(args,'grasp_controller','legacy')=='coordinated'):
        from .tabletop_palm_mount import CONTRACT as WRIST_CONTRACT
        contract['palm_transport']=dict(WRIST_CONTRACT)
    if getattr(args,'release_reposition',False):
        from .tabletop_loaded_palm_motion import CONTRACT as REPOSITION_CONTRACT
        contract['palm_reposition']=dict(REPOSITION_CONTRACT)
    if getattr(args,'force_direction_reposition',False):
        from .tabletop_force_direction_reposition import CONTRACT as REPOSITION_CONTRACT
        contract['palm_reposition']=dict(REPOSITION_CONTRACT)
        contract['force_direction_reposition']=True

    algorithm=make_algorithm(obs,steps)
    if getattr(args,'control_probe_checkpoint',None):
        from .tabletop_control_probe import run_control_probe
        return run_control_probe(env,algorithm,args,contract,obs)
    if task.shaping is not None and algorithm.gamma!=SHAPING_CONTRACT['gamma']:
        raise ValueError('potential shaping discount must match PPO gamma')
    iteration=timesteps=0
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
        for name in ('episodes','successes','recovered','reached_counts','transition_counts'):
            getattr(env,name).copy_(saved[name])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if saved['cuda_rng']: torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    write_json(args.output/'tensor_contract.json',contract)
    write_json(args.output/'training_start.json',dict(
        resume_source=str(args.resume) if args.resume else None,
        initial_iteration=iteration,initial_timesteps=timesteps,
        initial_episodes=int(env.episodes.sum()),initial_successes=int(env.successes.sum()),
        optimizer_state_entries=len(algorithm.optimizer.state),
        initial_policy_std=algorithm.policy.log_std.detach().exp().cpu().tolist()))
    writer=SummaryWriter(str(args.output/'tensorboard'))
    checkpoint=args.output/'tensor_ppo.pt'
    started=time.monotonic()
    stop=None
    failure_counts=torch.zeros(max(FAILURE_NAMES)+1,dtype=torch.long,device=args.device)
    try:
        while True:
            collect=time.monotonic()
            env.vertical_force_stats.clear()
            env.lift_entry_force_stats.clear()
            if task.shaping is not None:env.lift_gate_counts.zero_()
            reward_sum=torch.zeros((),device=args.device)
            terms={}
            occupancy=torch.zeros(4,device=args.device)
            relief_fraction=torch.zeros((),device=args.device)
            clipped_fraction=torch.zeros((),device=args.device)
            with torch.inference_mode():
                for _ in range(steps):
                    action=algorithm.act(obs)
                    if not bool(torch.isfinite(action).all()):
                        raise RuntimeError('nonfinite policy action before actuator clipping')
                    clipped_fraction+=(action.abs()>1.).float().mean()
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
                episode_budget_per_env=args.episodes_per_env,successes=int(env.successes.sum()),
                recovered_physical_episodes=int(env.recovered.sum()),
                phase_occupancy=(occupancy/(steps*args.num_envs)).cpu().tolist(),
                phase_reached_episodes=env.reached_counts.cpu().tolist(),
                phase_transition_counts=env.transition_counts.reshape(4,4).cpu().tolist(),
                mean_step_reward=float(reward_sum/steps),reward_terms={k:float(v/steps) for k,v in terms.items()},
                failures={FAILURE_NAMES[i]:int(v) for i,v in enumerate(failure_counts.cpu()) if i and int(v)},
                hand_load_mean_n=task.m['hand_object_loads'].mean(0).cpu().tolist(),
                max_object_bottom_m=float(task.m['bottom'].max()),losses=losses,
                learning_rate=algorithm.learning_rate,grasp_qualified=False)
            std=algorithm.policy.log_std.detach().exp()
            if 'grasp_controller' in contract:row['grasp_controller']=contract['grasp_controller']
            if 'palm_transport' in contract:row['palm_transport']=contract['palm_transport']
            if 'palm_reposition' in contract:
                row['palm_reposition']=contract['palm_reposition']
                row['force_direction_reposition']=bool(getattr(task,'force_direction_reposition',False))
                adjustment=task.palm_reposition
                row['reposition']=reposition_telemetry(adjustment)
            row['policy_noise']=dict(POLICY_NOISE_CONTRACT)
            row['vertical_contact_force']=env.vertical_force_stats.summary()
            row['lift_entry_force']=env.lift_entry_force_stats.summary()
            if task.vertical_force_reward:
                row['vertical_force_reward']=contract['vertical_force_reward']
                row['force_reward_enabled_env_fraction']=float(task.force_scoring_started.float().mean())
                row['downward_penalty_enabled_env_fraction']=row['force_reward_enabled_env_fraction']
            if task.close_downward_penalty:
                row['close_downward_penalty']=contract['close_downward_penalty']
            if task.shaping is not None:
                row['grasp_shaping']=contract['grasp_shaping']
                row['lift_gates']=lift_gate_summary(env.lift_gate_counts)
                row['max_lift_command_offset_m']=float(task.offset.max())
            row['policy_std_min']=float(std.min())
            row['policy_std_max']=float(std.max())
            row['action_clipped_fraction']=float(clipped_fraction/steps)
            if task.grasp_readiness is not None:
                row['grasp_readiness']=contract['grasp_readiness']
                row['load_ready_env_fraction']=float(task.grip_force_ready(task.m['hand_object_loads']).float().mean())
                row['max_grasp_contact_time_s']=float(task.contact_time.max())
            row['physics_peaks_since_start']=dict(zip(PHYSICS_PEAK_NAMES,env.physics_peaks.cpu().tolist()))
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
            write_json(args.output/'monitor.json',row)
            with (args.output/'iterations.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            for name,value in dict(collection_seconds=collect_seconds,learning_seconds=learn_seconds,
                fps=row['transitions_per_second'],mean_step_reward=row['mean_step_reward'],
                successes=row['successes'],episodes=row['episodes']).items(): writer.add_scalar(name,value,iteration)
            for name,value in losses.items(): writer.add_scalar('loss/'+name,value,iteration)
            if task.vertical_force_reward:
                writer.add_scalar('force_reward/enabled_env_fraction',row['force_reward_enabled_env_fraction'],iteration)
                writer.add_scalar('force_reward/downward_penalty_enabled_env_fraction',row['downward_penalty_enabled_env_fraction'],iteration)
            for name,value in row['reward_terms'].items(): writer.add_scalar('reward/'+name,value,iteration)
            for name,value in row.get('reposition',{}).items():writer.add_scalar('reposition/'+name,value,iteration)
            if task.shaping is not None:
                for name,value in row['lift_gates'].items():
                    if name.endswith('_fraction') and value is not None:writer.add_scalar('lift_gates/'+name,value,iteration)
                writer.add_scalar('lift_gates/lift_substeps',row['lift_gates']['lift_substeps'],iteration)
                writer.add_scalar('lift_gates/max_command_offset_m',row['max_lift_command_offset_m'],iteration)
            for phase in range(1,4): writer.add_scalar('phase/'+('approach','close','hold','lift')[phase],row['phase_occupancy'][phase],iteration)
            for name,value in row['physics_peaks_since_start'].items():writer.add_scalar('physics_peak/'+name,value,iteration)
            for name in ('policy_std_min','policy_std_max','action_clipped_fraction'):
                writer.add_scalar('policy/'+name,row[name],iteration)
            vertical=row['vertical_contact_force']
            from .tabletop_vertical_force import write_directional_force_scalars,write_lift_entry_force_scalars
            write_directional_force_scalars(writer,vertical,iteration)
            write_lift_entry_force_scalars(writer,row['lift_entry_force'],iteration)
            for name in ('hands_total_mean_n','weight_mean_n','force_minus_weight_mean_n',
                         'above_weight_sample_fraction','hands_total_peak_n'):
                if vertical[name] is not None:writer.add_scalar('vertical_force/'+name,vertical[name],iteration)
            for hi,side in enumerate(('left','right')):
                for key in ('hand_normal_mean_n','hand_friction_mean_n','hand_total_mean_n'):
                    writer.add_scalar('vertical_force/'+side+'/'+key,vertical[key][hi],iteration)
            writer.flush()
            print('TENSOR_PPO_ITERATION',json.dumps(row),flush=True)
            if args.episodes_per_env is not None:
                if int(env.episodes.min())>=args.episodes_per_env: break
            elif timesteps>=args.timesteps: break
    except BaseException as error:
        stop=str(error)
        # Save only the last completed optimizer iteration. The partial rollout
        # is intentionally discarded; its counters must not overwrite checkpoint.
        publish_stopped_monitor(args.output/'monitor.json',error,env.physics_peaks,
            iteration,timesteps,args.num_envs)
        raise
    finally:
        writer.close()
    row['status']='training_budget_complete'
    write_json(args.output/'monitor.json',row)
    return dict(status='training_budget_complete',tensor_training=VERSION,iteration=iteration,
                timesteps=timesteps,successes=int(env.successes.sum()),grasp_qualified=False)
