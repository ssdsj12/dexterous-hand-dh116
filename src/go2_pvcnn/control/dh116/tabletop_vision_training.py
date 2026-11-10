"""Independent RGB-D actor PPO with a training-only privileged critic."""
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from .tabletop_vision_student import SensorHistory, make_vision_algorithm, POLICY_DIM
from .tabletop_surface_contract import ACTION_CONTRACT
from .tabletop_vision_squashed_policy import make_squashed_vision_algorithm
from .tabletop_vision_action_contract import (upgrade_action_contract,
    validate_squashed_checkpoint, validate_vision_action_options,
    validate_new_visual_output, assert_finite_state)

VERSION = 'dh116-vision-student-ppo-v2-bounded'


def _write_json(path, value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def _contract(args, critic_contract):
    sha=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    contract=dict(version=VERSION,actor_observation_size=POLICY_DIM,critic_observation_size=371,
        actor_uses_privileged_state=False,critic_source='native_371D_training_only',
        actor_sources=['RGB-visible object pixels','optical depth','calibrated camera pose',
            'joint encoders','palm FK','previous issued action','sensor timing and validity'],
        point_cloud_shape=[2,128,3],point_frame='actual_palm_body_link',position_unit='m',
        optical_frame='x_right_y_down_z_forward',history_frames=4,
        point_selection='visible_RGB_brown_mask;3x3_depth_continuity_20mm;uniform_retained_pixel_indices;zero_pad_with_mask',
        sensor_rendering=dict(antialiasing_mode='Off',dlssg=False,
            verified_native_readback=getattr(args,'vision_render_readback',None)),
        pixel_ray_convention='RTX_pixel_centers_u_plus_0.5_v_plus_0.5_with_K_center_W_over_2_H_over_2',
        missing_observation='zero_points_with_false_mask;no_simulator_truth_fallback',
        teacher_distillation='disabled_without_synchronized_visual_labels',
        network=dict(type='masked_PointNet_two_hand_shared_encoder_plus_four_frame_MLP',
            point_mlp=[3,32,64,64],action_mlp=[772,128,128,24]),
        ppo=dict(gamma=.995,lam=.95,clip_param=.2,epochs=5,minibatches=min(8,args.num_envs),
            learning_rate=3e-4,schedule='adaptive',desired_kl=.02,entropy_coef=.005,
            value_loss_coef=.5,max_grad_norm=.5,initial_std=.1,min_std=.001,max_std=.3),
        memory_contract=dict(storage='cpu_float32',gradient_accumulation=True,
            rgbd_processing_env_batch=64,
            microbatch_size=getattr(args,'vision_ppo_microbatch_size',128),
            inference_batch_size=getattr(args,'vision_inference_batch_size',128),
            logical_minibatches=min(8,args.num_envs)),
        camera_batch_size=min(args.num_envs,getattr(args,'vision_camera_batch_size',64)),
        camera_batching='same_frozen_physics_state_sequential_real_RGBD',
        visibility_mode=getattr(args,'vision_visibility_mode','all'),
        pinned_source_environment=0 if getattr(args,'vision_visibility_mode','all')=='batch' else None,
        action_contract=ACTION_CONTRACT,num_envs=args.num_envs,rollout_steps=args.rollout_steps,
        groups=[dict(label='pinch_thumb_index',num_envs=args.num_envs//2),
                dict(label='tripod_thumb_index_middle',num_envs=args.num_envs//2)],
        camera_config_sha256=sha(args.depth_camera_config),
        camera_resolution_wh=[args.vision_width,args.vision_height],
        primary_reset_sha256=sha(args.grasp_initialization),
        secondary_reset_sha256=sha(args.mixed_grasp_initialization),
        privileged_critic_contract=critic_contract)
    if getattr(args,'vision_camera_mode','bounded')=='fixed':
        contract.update(camera_mode='fixed_per_env',camera_batch_size=args.num_envs,
            camera_batching='fixed_per_environment_real_RGBD',visibility_mode='all',
            pinned_source_environment=None)
    from .tabletop_vision_quaternion import validate_wrist_quaternion_preprocessing
    preprocessing=validate_wrist_quaternion_preprocessing(
        getattr(args,'vision_wrist_quaternion_preprocessing','raw'))
    if preprocessing!='raw':
        if getattr(args,'vision_action_distribution','gaussian_clip')!='tanh_gaussian':
            raise ValueError('versioned quaternion preprocessing requires tanh_gaussian')
        contract['actor_wrist_quaternion_preprocessing']=preprocessing
    if getattr(args,'vision_action_distribution','gaussian_clip')=='tanh_gaussian':
        return upgrade_action_contract(contract)
    return contract


def _observations(packet, device_world, previous_action, history, critic, reset_mask=None):
    """Only named encoder/FK fields enter the actor; critic data is packed separately."""
    n=device_world.n
    ids=device_world.w.ids
    q=torch.cat([device_world.q[:,h,ids[h]] for h in range(2)],dim=1).float()
    qd=torch.cat([device_world.qd[:,h,ids[h]] for h in range(2)],dim=1).float()
    wrists=device_world.roots.clone()
    wrists[:,:,:3]-=device_world.origins[:,None]
    if not bool(packet['image_valid'].all()):
        raise RuntimeError('camera frame invalid; no privileged fallback is permitted')
    policy=history.update(points_in_hands=packet['points'],
        point_valid=packet['point_valid'],q=q,qd=qd,wrist_poses=wrists.float(),
        previous_action=previous_action,dt=1./60.,
        image_valid=packet['image_valid'],object_valid=packet['object_valid'],reset_mask=reset_mask)
    if critic.shape!=(n,371) or not bool(torch.isfinite(critic).all()):
        raise RuntimeError('invalid privileged critic observation')
    return TensorDict({'policy':policy,'critic':critic.float()},batch_size=[n],device=policy.device)


def _mirror_preprocessing_metadata(mode):
    if mode == 'm0_raw':
        return dict(mode=mode,actor_observation_transform='identity',
                    physical_action_transform='identity',offline_only=False,
                    default_path=True)
    if mode == 'm2_full_mirror':
        return dict(mode=mode,
                    actor_observation_transform='mirror_observation_then_quaternion_canonicalize',
                    physical_action_transform='mirror_action24_inverse',
                    inverse_is_involution=True,offline_only=False,
                    default_path=False,checkpoint_contract_unchanged=True)
    raise ValueError(f'unsupported vision mirror preprocessing: {mode}')


def _apply_mirror_preprocessing(obs, mode):
    metadata=_mirror_preprocessing_metadata(mode)
    if mode == 'm0_raw':
        return obs, metadata
    from .tabletop_vision_mirror_ablation import m2_observation_for_actor
    transformed=obs.clone()
    transformed['policy']=m2_observation_for_actor(obs['policy'])
    return transformed, metadata


def _save(path, algorithm, contract, iteration, timesteps, env):
    data=dict(version=contract['version'],contract=contract,iteration=iteration,timesteps=timesteps,
        policy=algorithm.policy.state_dict(),optimizer=algorithm.optimizer.state_dict(),
        learning_rate=algorithm.learning_rate,episodes=env.episodes,successes=env.successes,
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available()
            else getattr(algorithm,'vision_saved_cuda_rng',[]))
    migration=getattr(algorithm,'vision_resume_migration',None)
    if migration is not None:data['resume_migration']=migration
    derivation=getattr(algorithm,'vision_action_derivation',None)
    if derivation is not None:data['action_derivation']=derivation
    for name in ('experiment_projection','optimizer_reset_history','reward_experiment_derivation',
                 'visual_supervision_derivation','visual_supervision_history'):
        value=getattr(algorithm,'vision_'+name,None)
        if value is not None:data[name]=copy.deepcopy(value)
    if contract['version']!=VERSION:assert_finite_state(data)
    temporary=Path(str(path)+'.tmp')
    torch.save(data,temporary)
    os.replace(temporary,path)


def run_vision_training(world, device_world, task, env, args, critic_contract, *, camera=None):
    """Train a new sensor-only actor; only same-version visual checkpoints resume."""
    if args.rollout_steps is None:
        args=copy.copy(args)
        args.rollout_steps=24
    output=Path(args.output)
    mirror_mode=getattr(args,'vision_mirror_preprocessing','m0_raw')
    mirror_metadata=_mirror_preprocessing_metadata(mirror_mode)
    action_diagnostics=None
    diagnostic_episode_ids=None
    diagnostic_stream=None
    physics_diagnostic_stream=None
    policy_diagnostics=None
    policy_rollout_episode_ids=None
    if getattr(args,'vision_action_diagnostics',False):
        from .tabletop_vision_action_diagnostics import VisionActionDiagnosticRecorder
        action_diagnostics=VisionActionDiagnosticRecorder()
        diagnostic_episode_ids=torch.zeros(args.num_envs,dtype=torch.long)
        diagnostic_stream=(output/'action_diagnostics.jsonl').open('x',encoding='utf-8')
        physics_diagnostic_stream=(output/'physics_action_diagnostics.jsonl').open('x',encoding='utf-8')
        if not hasattr(env,'enable_physics_action_diagnostics'):
            raise RuntimeError('vision action diagnostics require TensorEnv physics telemetry')
        env.enable_physics_action_diagnostics(physics_diagnostic_stream)
    if getattr(args,'vision_policy_diagnostics',False):
        from .tabletop_vision_policy_diagnostics import VisionPolicyDiagnosticRecorder
        policy_diagnostics=VisionPolicyDiagnosticRecorder(output/'policy_ppo_diagnostics.jsonl')
        if diagnostic_episode_ids is None:
            diagnostic_episode_ids=torch.zeros(args.num_envs,dtype=torch.long)
        policy_rollout_episode_ids=diagnostic_episode_ids
    squashed=getattr(args,'vision_action_distribution','gaussian_clip')=='tanh_gaussian'
    saved=None
    validate_vision_action_options(args)
    if squashed:
        validate_new_visual_output(output,preflight_path=getattr(args,'_vision_new_output',None))
    contract=_contract(args,critic_contract)
    if squashed and args.resume:
        resume_sha256=hashlib.sha256(Path(args.resume).read_bytes()).hexdigest()
        saved=validate_squashed_checkpoint(
            torch.load(args.resume,map_location='cpu',weights_only=False),contract)
        if hashlib.sha256(Path(args.resume).read_bytes()).hexdigest()!=resume_sha256:
            raise ValueError('warm-start source changed while loading')
        if (getattr(args,'vision_warm_start_optimizer','preserve')=='reset'
                and args.max_iterations<=saved['iteration']):
            raise ValueError('reset Adam warm-start requires at least one complete new update')
    initial_audit=None
    if getattr(args,'vision_experiment_initial_state_audit',False):
        from .tabletop_vision_initial_state_audit import TrainingInitialStateAudit
        initial_audit=TrainingInitialStateAudit(world,device_world,env,args,contract,
            native_camera=camera is None,saved=saved,
            source_resume_sha256=resume_sha256 if saved is not None else None)
    _write_json(output/'vision_student_contract.json',contract)
    native_camera=camera is None
    if camera is None:
        if getattr(args,'vision_camera_mode','bounded')=='fixed':
            from .tabletop_vision_fixed_camera import FixedVisionCamera
            camera=FixedVisionCamera(world,device_world,args.depth_camera_config,
                width=args.vision_width,height=args.vision_height,num_points=128,
                debug_output_dir=output/'camera_debug',debug_env_ids=(0,args.num_envs//2))
        else:
            from .tabletop_vision_batched_camera import BoundedVisionCamera
            camera=BoundedVisionCamera(world,device_world,args.depth_camera_config,
                width=args.vision_width,height=args.vision_height,num_points=128,
                batch_size=getattr(args,'vision_camera_batch_size',64),
                hide_inactive=getattr(args,'vision_visibility_mode','all')=='batch',
                debug_output_dir=output/'camera_debug',
                debug_env_ids=tuple(i for i in (3388,3450) if i<args.num_envs))
    history=SensorHistory(args.num_envs,device=args.device)
    previous=torch.zeros((args.num_envs,24),device=args.device)
    if hasattr(env,'enable_palm_lift_telemetry'):
        env.enable_palm_lift_telemetry()
    critic=env.reset()
    if initial_audit is not None:initial_audit.before_camera()
    packet=camera.sample()
    if initial_audit is not None:initial_audit.after_camera()
    if native_camera:
        from .tabletop_vision_memory import memory_snapshot
        memory_snapshot(args,'initial_camera_sample',device_world.w)
    _write_json(output/'initial_camera_observation.json',dict(
        num_envs=args.num_envs,image_valid=packet['image_valid'].cpu().tolist(),
        object_valid=packet['object_valid'].cpu().tolist(),
        point_counts=packet['point_valid'].sum(1).cpu().tolist(),
        frame_id=packet.get('frame_id'),simulation_time=packet.get('simulation_time'),
        rendered_simulation_time=packet.get('rendered_simulation_time'),
        reference_time_numerator=packet.get('reference_time_numerator'),
        reference_time_denominator=packet.get('reference_time_denominator'),
        time_mapping=packet.get('time_mapping'),camera_batch_count=packet.get('render_batches'),
        batch_frames=packet.get('batch_frame_ids'),
        batch_environment_ranges=packet.get('batch_environment_ranges'),
        batch_rendered_simulation_times=packet.get('batch_rendered_simulation_times'),
        pinned_source_environment=packet.get('pinned_source_environment'),
        batch_wall_seconds=packet.get('batch_wall_seconds'),
        debug_saved_files=packet.get('debug_saved_files')))
    if not bool(packet['image_valid'].all() & packet['object_valid'].all()):
        raise RuntimeError('initial RGB-D validation requires a visible object in every reset environment')
    if native_camera:
        from .tabletop_vision_audit import audit_initial_points,VisionGeometryError
        try:
            geometry=audit_initial_points(packet,device_world,
                radius=getattr(args,'solid_cylinder_radius_m',.008),length=.5)
        except VisionGeometryError as error:
            _write_json(output/'initial_camera_geometry_validation.json',error.report)
            sensor=dict(points_in_hands=packet['points'].detach().cpu(),
                point_valid=packet['point_valid'].detach().cpu(),
                hand_poses= device_world.roots.detach().cpu())
            if 'selected_pixels_uv' in packet:
                sensor['selected_pixels_uv']=packet['selected_pixels_uv'].detach().cpu()
            torch.save(sensor,output/'initial_camera_sensor_failure.pt')
            torch.save(dict(position=device_world.wp.detach().cpu(),quaternion=device_world.wq.detach().cpu(),
                radius_m=getattr(args,'solid_cylinder_radius_m',.008),length_m=.5),
                output/'initial_camera_truth_validation.pt')
            raise
        _write_json(output/'initial_camera_geometry_validation.json',geometry)
    # Resume references are checked before constructing PPO/Adam/storage.
    if initial_audit is not None and saved is not None:initial_audit.finish(saved['policy'])
    obs,mirror_metadata=_apply_mirror_preprocessing(
        _observations(packet,device_world,previous,history,critic),mirror_mode)
    start_metadata_mirror=mirror_metadata
    _write_json(output/'vision_mirror_preprocessing.json',mirror_metadata)
    maker=make_squashed_vision_algorithm if squashed else make_vision_algorithm
    actor_options={}
    if squashed and contract.get('actor_wrist_quaternion_preprocessing','raw')!='raw':
        actor_options['wrist_quaternion_preprocessing']=contract['actor_wrist_quaternion_preprocessing']
    algorithm=maker(obs,args.rollout_steps,device=args.device,
        microbatch_size=getattr(args,'vision_ppo_microbatch_size',128),
        inference_batch_size=getattr(args,'vision_inference_batch_size',128),**actor_options)
    if native_camera:
        from .tabletop_vision_memory import memory_snapshot
        memory_snapshot(args,'camera_and_PPO_ready',device_world.w)
    iteration=timesteps=0
    if args.resume:
        if not squashed:
            saved=torch.load(args.resume,map_location=args.device,weights_only=False)
            if saved.get('version')!=VERSION:
                raise ValueError('visual resume requires the same visual actor/critic/camera/reset contract')
            from .tabletop_vision_resume import prepare_visual_resume
            saved=prepare_visual_resume(saved,contract,
                allow_downsize=getattr(args,'vision_resume_downsize',False))
        if saved.get('resume_migration') is not None:
            algorithm.vision_resume_migration=saved['resume_migration']
            _write_json(output/'resume_migration.json',saved['resume_migration'])
        if squashed:
            algorithm.vision_saved_cuda_rng=saved['cuda_rng']
            for name in ('experiment_projection','optimizer_reset_history','reward_experiment_derivation',
                         'visual_supervision_derivation','visual_supervision_history'):
                if name in saved:setattr(algorithm,'vision_'+name,copy.deepcopy(saved[name]))
            if saved.get('action_derivation') is not None:
                algorithm.vision_action_derivation=saved['action_derivation']
                _write_json(output/'action_derivation.json',saved['action_derivation'])
        algorithm.policy.load_state_dict(saved['policy'])
        algorithm.optimizer.load_state_dict(saved['optimizer'])
        algorithm.learning_rate=saved['learning_rate']
        iteration,timesteps=saved['iteration'],saved['timesteps']
        if squashed and getattr(args,'vision_warm_start_optimizer','preserve')=='reset':
            algorithm.optimizer.state.clear()
            history_resets=copy.deepcopy(getattr(algorithm,'vision_optimizer_reset_history',[]))
            history_resets.append(dict(iteration=iteration,source_sha256=resume_sha256,
                source_path=str(args.resume),operation='clear_Adam_moments_and_steps_only',
                policy_learning_rate_rng_and_counters_preserved=True))
            algorithm.vision_optimizer_reset_history=history_resets
        for name in ('episodes','successes'):
            getattr(env,name).copy_(saved[name])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if saved.get('cuda_rng') and (not squashed or torch.cuda.is_available()):
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    if initial_audit is not None:
        if saved is None:initial_audit.finish(algorithm.policy.state_dict())
        initial_audit.verify_loaded_policy(algorithm.policy.state_dict())
    start_metadata=dict(
        observation_mode='vision_sensor_only_actor',privileged_actor_inputs=False,
        initial_iteration=iteration,initial_timesteps=timesteps,
        initial_episodes=int(env.episodes.sum()),initial_successes=int(env.successes.sum()),
        resume_source=str(args.resume) if args.resume else None,
        optimizer_state_entries=len(algorithm.optimizer.state),
        actor_observation_size=POLICY_DIM,critic_observation_size=371,
        teacher_distillation=contract['teacher_distillation'])
    if squashed:
        warm_mode=getattr(args,'vision_warm_start_optimizer','preserve')
        start_metadata.update(resume_environment_state='cold_reset_after_resume' if args.resume
            else 'fresh_scene_reset',resume_restored_state=(
                ['policy']+(['Adam'] if warm_mode=='preserve' else [])+
                ['learning_rate','Torch_RNG','counters'] if args.resume else []),
            rng_coverage=contract['rng_coverage'],optimizer_warm_start_mode=warm_mode)
        for name in ('experiment_projection','optimizer_reset_history','reward_experiment_derivation',
                     'visual_supervision_derivation','visual_supervision_history'):
            value=getattr(algorithm,'vision_'+name,None)
            if value is not None:start_metadata[name]=copy.deepcopy(value)
    start_metadata['mirror_preprocessing']=start_metadata_mirror
    _write_json(output/'training_start.json',start_metadata)
    # Save real sampled actor inputs separately from privileged critic state.
    torch.save(dict(policy=obs['policy'].detach().cpu(),contract=contract),
        output/'initial_visual_actor_inputs.pt')
    baseline_episodes=env.episodes.clone()
    baseline_successes=env.successes.clone()
    writer=SummaryWriter(str(output/'tensorboard'))
    checkpoint=output/'vision_student_ppo.pt'
    # A no-op resume still creates a real checkpoint in its output directory.
    if iteration>=args.max_iterations:
        _save(checkpoint,algorithm,contract,iteration,timesteps,env)
    started=time.monotonic()
    row=dict(status='initializing_visual_policy',backend=contract['version'],iteration=iteration,
             iteration_budget=args.max_iterations,successes=int(env.successes.sum()))
    _write_json(output/'monitor.json',row)
    try:
        while iteration<args.max_iterations:
            begin=time.monotonic()
            reward_sum=0.
            failure_counts={}
            visible_fraction=0.
            peak_lift=torch.full((args.num_envs,),-torch.inf,device=args.device)
            palm_peak=palm_last=palm_valid=None
            policy_rollout_records=[]
            with torch.inference_mode():
                for step in range(args.rollout_steps):
                    action=algorithm.act(obs)
                    if not bool(torch.isfinite(action).all()):
                        raise RuntimeError('nonfinite visual action before physics')
                    if squashed:
                        if action.shape!=(args.num_envs,24) or not bool((action.abs()<=1).all()):
                            raise ValueError('bounded visual action must be [N,24] within [-1,1] before physics')
                        issued=action.clone()
                    else:
                        issued=action.clamp(-1,1)
                    if mirror_mode == 'm2_full_mirror':
                        from .tabletop_vision_mirror_ablation import m2_action_to_physics
                        issued=m2_action_to_physics(issued)
                    if policy_diagnostics is not None:
                        transition=algorithm.transition
                        sigma=getattr(transition,'action_sigma',None)
                        if sigma is None:
                            raise RuntimeError('vision policy diagnostics require action_sigma')
                        policy_rollout_records.append(dict(
                            env_id=np.arange(args.num_envs,dtype=np.int64),
                            episode_id=policy_rollout_episode_ids.detach().cpu().numpy().copy(),
                            action24=transition.actions.detach().cpu().numpy().copy(),
                            latent_actions=(getattr(transition,'latent_actions',None).detach().cpu().numpy().copy()
                                            if getattr(transition,'latent_actions',None) is not None else None),
                            latent_mean=transition.action_mean.detach().cpu().numpy().copy(),
                            log_std=sigma.detach().cpu().numpy().copy(),
                            log_prob=transition.actions_log_prob.detach().cpu().numpy().copy(),
                            value=transition.values.detach().cpu().numpy().copy(),
                            timestamp_s=float(getattr(getattr(getattr(device_world,'w',None),'sim',None),
                                                      'current_time',time.time())),
                        ))
                    if physics_diagnostic_stream is not None:
                        env.set_physics_diagnostic_policy_step(iteration*args.rollout_steps+step)
                    critic,reward,done,extra=env.step(issued.clone() if squashed else issued)
                    if policy_diagnostics is not None:
                        policy_rollout_records[-1]['timestamp_s']=float(getattr(
                            getattr(getattr(device_world,'w',None),'sim',None),
                            'current_time',policy_rollout_records[-1]['timestamp_s']))
                        policy_rollout_records[-1]['reward']=reward.detach().cpu().numpy().copy()
                        policy_rollout_records[-1]['done']=done.detach().cpu().numpy().copy()
                        policy_rollout_records[-1]['reward_terms']={
                            str(name): value.detach().cpu().numpy().copy()
                            for name,value in extra.get('terms',{}).items()}
                    if action_diagnostics is not None and hasattr(device_world,'last_action_diagnostics'):
                        audit=device_world.last_action_diagnostics
                        diagnostic_batch=action_diagnostics.record(
                            env_id=torch.arange(args.num_envs).cpu().numpy(),
                            episode_id=diagnostic_episode_ids.numpy(),
                            physics_step=np.full(args.num_envs,int(audit['physics_step']),dtype=np.int64),
                            timestamp_s=np.full(args.num_envs,float(audit['timestamp_s']),dtype=np.float64),
                            raw_action24=audit['raw_action24'].detach().cpu().numpy(),
                            requested_palm_target_m=audit['requested_palm_target_m'].detach().cpu().numpy(),
                            pre_projection_palm_target_m=audit['pre_projection_palm_target_m'].detach().cpu().numpy(),
                            projected_palm_target_m=audit['projected_palm_target_m'].detach().cpu().numpy(),
                            final_sdk_palm_target_m=audit['final_sdk_palm_target_m'].detach().cpu().numpy(),
                            actual_palm_pose_m=audit['actual_palm_pose_m'].detach().cpu().numpy(),
                            palm_displacement_m=audit['palm_displacement_m'].detach().cpu().numpy(),
                            control_frame_sdk_target_m=audit['control_frame_sdk_target_m'].detach().cpu().numpy(),
                            actual_palm_velocity_m_s=audit['actual_palm_velocity_m_s'].detach().cpu().numpy(),
                            table_clearance_m=audit['table_clearance_m'].detach().cpu().numpy(),
                            safe_z_m=audit['safe_z_m'].detach().cpu().numpy(),
                            contact_loads_n=audit['contact_loads_n'].detach().cpu().numpy(),
                            contact_valid=audit['contact_valid'].detach().cpu().numpy(),
                            projection_reasons=audit['projection_reasons'].detach().cpu().tolist())
                        for diagnostic_row in diagnostic_batch['rows']:
                            diagnostic_stream.write(json.dumps(diagnostic_row,separators=(',',':'))+'\n')
                        diagnostic_stream.flush()
                    if diagnostic_episode_ids is not None:
                        diagnostic_episode_ids += done.detach().cpu().long()
                    # Read the pre-task-reset diagnostic, including terminal samples.
                    step_lift=extra.get('surface_diagnostics',{}).get('net_lift_m')
                    if step_lift is None:
                        step_lift=task.m['bottom']-task.surface_reference_bottom_m
                    step_lift=torch.as_tensor(step_lift,device=args.device).float()
                    if step_lift.shape!=(args.num_envs,) or not torch.isfinite(step_lift).all():
                        raise RuntimeError('invalid privileged net-lift diagnostic')
                    peak_lift=torch.maximum(peak_lift,step_lift)
                    palms=extra.get('palm_lift_telemetry')
                    if palms is not None:
                        measured_peak=palms['peak_m'];measured_last=palms['last_m'];valid=palms['valid']
                        if (measured_peak.shape!=(args.num_envs,2) or measured_last.shape!=measured_peak.shape
                                or valid.shape!=measured_peak.shape or valid.dtype!=torch.bool):
                            raise RuntimeError('invalid measured palm-lift diagnostic shape')
                        if palm_peak is None:
                            palm_peak=torch.full_like(measured_peak,-torch.inf)
                            palm_last=torch.full_like(measured_last,torch.nan)
                            palm_valid=torch.zeros_like(valid)
                        palm_peak=torch.where(valid,torch.maximum(palm_peak,measured_peak),palm_peak)
                        palm_last=torch.where(valid,measured_last,palm_last)
                        palm_valid|=valid
                    if not bool(torch.isfinite(reward).all()):
                        raise RuntimeError('nonfinite reward during visual collection')
                    previous=issued.clone()
                    previous[done]=0.
                    packet=camera.sample()
                    collection=dict(phase='collecting',iteration=iteration+1,rollout_step=step+1,
                        rollout_steps=args.rollout_steps,num_envs=args.num_envs,wall_unix_s=time.time(),
                        camera_simulation_time=packet.get('simulation_time'),
                        camera_frame_first=packet.get('camera_frame_first'),
                        camera_frame_last=packet.get('camera_frame_last'),
                        camera_batch_count=packet.get('render_batches'))
                    _write_json(output/'collection_status.json',collection)
                    obs, _ = _apply_mirror_preprocessing(
                        _observations(packet,device_world,previous,history,critic,done),mirror_mode)
                    algorithm.process_env_step(obs,reward,done,{})
                    reward_sum+=float(reward.mean())
                    visible_fraction+=float(packet['object_valid'].float().mean())
                    codes=extra.get('failure')
                    if codes is not None:
                        for code,count in zip(*torch.unique(codes[done],return_counts=True)):
                            if int(code):
                                key=str(int(code))
                                failure_counts[key]=failure_counts.get(key,0)+int(count)
                algorithm.compute_returns(obs)
            if policy_diagnostics is not None:
                storage=algorithm.storage
                for policy_step, record in enumerate(policy_rollout_records):
                    policy_diagnostics.record(
                        env_id=record['env_id'], episode_id=record['episode_id'],
                        iteration=iteration+1,
                        policy_step=iteration*args.rollout_steps+policy_step,
                        timestamp_s=np.full(args.num_envs,record['timestamp_s'],dtype=np.float64),
                        action24=record['action24'], latent_actions=record['latent_actions'],
                        latent_mean=record['latent_mean'],
                        log_std=np.log(np.maximum(record['log_std'],1e-12)),
                        log_prob=record['log_prob'], value=record['value'],
                        advantage=storage.advantages[policy_step].detach().cpu().numpy(),
                        returns=storage.returns[policy_step].detach().cpu().numpy(),
                        reward=record['reward'], done=record['done'],
                        reward_terms=record['reward_terms'])
            collection.update(phase='optimizing',wall_unix_s=time.time())
            _write_json(output/'collection_status.json',collection)
            losses=algorithm.update()
            if not all(torch.isfinite(torch.tensor(float(v))) for v in losses.values()):
                raise RuntimeError('nonfinite visual PPO loss')
            iteration+=1
            timesteps+=args.num_envs*args.rollout_steps
            duration=time.monotonic()-begin
            net_lift=peak_lift
            groups={}
            offset=0
            for group in contract['groups']:
                count=group['num_envs']
                slots=slice(offset,offset+count)
                offset+=count
                groups[group['label']]=dict(num_envs=count,
                    episodes=int(env.episodes[slots].sum()),successes=int(env.successes[slots].sum()),
                    episodes_since_launch=int((env.episodes[slots]-baseline_episodes[slots]).sum()),
                    successes_since_launch=int((env.successes[slots]-baseline_successes[slots]).sum()),
                    net_rod_lift_max_m=float(net_lift[slots].max()))
                if palm_peak is not None:
                    from .tabletop_palm_lift_summary import summarize_palm_lift
                    groups[group['label']].update(summarize_palm_lift(
                        palm_peak,palm_last,palm_valid,start=slots.start,stop=slots.stop))
            row=dict(status='training',backend=contract['version'],observation_mode='vision_sensor_only_actor',
                privileged_actor_inputs=False,actor_observation_size=POLICY_DIM,
                critic_observation_size=371,num_envs=args.num_envs,iteration=iteration,
                iteration_budget=args.max_iterations,rollout_steps=args.rollout_steps,
                timesteps=timesteps,episodes=int(env.episodes.sum()),
                episodes_per_env_min=int(env.episodes.min()),successes=int(env.successes.sum()),
                grasp_styles=groups,surface_sharing=dict(net_rod_lift_max_m=float(net_lift.max())),
                losses={k:float(v) for k,v in losses.items()},
                mean_step_reward=reward_sum/args.rollout_steps,
                object_visible_env_fraction=visible_fraction/args.rollout_steps,
                camera_image_valid_env_fraction=float(packet['image_valid'].float().mean()),
                sampled_object_points_mean=float(packet['point_valid'].float().sum(1).mean()),
                camera_frame_id=packet.get('frame_id'),camera_simulation_time=packet.get('simulation_time'),
                camera_rendered_simulation_time=packet.get('rendered_simulation_time'),
                camera_render_passes=packet.get('render_passes'),
                pinned_source_environment=packet.get('pinned_source_environment'),
                memory_contract=contract['memory_contract'],camera_batch_count=packet.get('render_batches'),
                camera_frame_first=packet.get('camera_frame_first'),camera_frame_last=packet.get('camera_frame_last'),
                failures=failure_counts,iteration_seconds=duration,
                elapsed_seconds=time.monotonic()-started)
            if squashed:
                row['ppo_diagnostics']=copy.deepcopy(algorithm.last_update_diagnostics)
                row['ppo_diagnostics']['post_update_latent_std_per_dim']=(
                    algorithm.policy.log_std.detach().exp().cpu().tolist())
                assert_finite_state(row['ppo_diagnostics'])
            if palm_peak is not None:
                row['palm_lift_measurement']=dict(
                    source='measured_palm_body_link_world_z',unit='m',
                    baseline='measured_after_physical_episode_reset',
                    max_scope='rollout_live_physics_substeps',
                    last_mean_scope='last_valid_sample_per_environment_in_rollout',
                    left_right_maxima_may_use_different_environments=True)
            if action_diagnostics is not None:
                row['action_diagnostics_summary']=action_diagnostics.summary()
            if policy_diagnostics is not None:
                row['policy_diagnostics_summary']=policy_diagnostics.summary()
            if torch.cuda.is_available() and str(args.device).startswith('cuda'):
                free,total=torch.cuda.mem_get_info()
                row['torch_peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
                row['torch_peak_reserved_gib']=torch.cuda.max_memory_reserved()/1024**3
                row['rollout_observation_device']=str(algorithm.storage.observations['policy'].device)
                row['rollout_observation_pinned']=algorithm.storage.observations['policy'].is_pinned()
                row['gpu_memory_used_gib']=(total-free)/1024**3
            # Only a complete optimizer update commits policy and counters.
            _save(checkpoint,algorithm,contract,iteration,timesteps,env)
            if squashed and (iteration%getattr(args,'vision_checkpoint_interval',10)==0
                              or iteration==args.max_iterations):
                from .tabletop_vision_experiment_checkpoint import retain_checkpoint_bundle
                retain_checkpoint_bundle(checkpoint,output/'checkpoints')
            _write_json(output/'monitor.json',row)
            collection.update(phase='optimizer_complete',wall_unix_s=time.time())
            _write_json(output/'collection_status.json',collection)
            with (output/'iterations.jsonl').open('a') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
            for name in ('successes','episodes','mean_step_reward','object_visible_env_fraction',
                         'iteration_seconds','sampled_object_points_mean'):
                writer.add_scalar(name,row[name],iteration)
            if squashed:
                for name,value in row['ppo_diagnostics'].items():
                    if isinstance(value,list):
                        for index,item in enumerate(value):
                            writer.add_scalar(f'ppo_diagnostics/{name}/{index:02d}',item,iteration)
                    else:
                        writer.add_scalar(f'ppo_diagnostics/{name}',value,iteration)
            for label,metrics in groups.items():
                for name,value in metrics.items():
                    if value is not None:
                        writer.add_scalar(f'grasp_style/{label}/{name}',value,iteration)
                writer.add_scalar(f'lift_mm/{label}/rod_peak',metrics['net_rod_lift_max_m']*1000.,iteration)
                for side in ('left','right'):
                    for suffix,tag in [('max_m','peak'),('last_mean_m','last_mean')]:
                        value=metrics.get(f'{side}_palm_lift_{suffix}')
                        if value is not None:writer.add_scalar(f'lift_mm/{label}/{side}_palm_{tag}',value*1000.,iteration)
            print('VISION_STUDENT_PPO_ITERATION',json.dumps(row),flush=True)
    except BaseException as error:
        row.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',
                   error=type(error).__name__+': '+str(error))
        _write_json(output/'monitor.json',row)
        raise
    finally:
        if action_diagnostics is not None:
            if diagnostic_stream is not None:
                diagnostic_stream.flush()
                diagnostic_stream.close()
            _write_json(output/'action_diagnostics_summary.json',action_diagnostics.summary())
        if policy_diagnostics is not None:
            policy_diagnostics.close()
            _write_json(output/'policy_ppo_diagnostics_summary.json',policy_diagnostics.summary())
        if physics_diagnostic_stream is not None:
            physics_diagnostic_stream.flush()
            physics_diagnostic_stream.close()
        writer.close()
    row.update(status='training_budget_complete')
    _write_json(output/'monitor.json',row)
    return dict(status='training_budget_complete',iteration=iteration,timesteps=timesteps,
        successes=int(env.successes.sum()),visual_student=True,privileged_actor_inputs=False,
        actor_observation_size=POLICY_DIM,critic_observation_size=371,
        checkpoint=str(checkpoint))
