"""Independent RGB-D actor PPO with a training-only privileged critic."""
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from .tabletop_vision_student import SensorHistory, make_vision_algorithm, POLICY_DIM
from .tabletop_surface_contract import ACTION_CONTRACT

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


def _save(path, algorithm, contract, iteration, timesteps, env):
    data=dict(version=VERSION,contract=contract,iteration=iteration,timesteps=timesteps,
        policy=algorithm.policy.state_dict(),optimizer=algorithm.optimizer.state_dict(),
        learning_rate=algorithm.learning_rate,episodes=env.episodes,successes=env.successes,
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    migration=getattr(algorithm,'vision_resume_migration',None)
    if migration is not None:data['resume_migration']=migration
    temporary=Path(str(path)+'.tmp')
    torch.save(data,temporary)
    os.replace(temporary,path)


def run_vision_training(world, device_world, task, env, args, critic_contract, *, camera=None):
    """Train a new sensor-only actor; only same-version visual checkpoints resume."""
    if args.rollout_steps is None:
        args=copy.copy(args)
        args.rollout_steps=24
    output=Path(args.output)
    contract=_contract(args,critic_contract)
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
    critic=env.reset()
    packet=camera.sample()
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
    obs=_observations(packet,device_world,previous,history,critic)
    algorithm=make_vision_algorithm(obs,args.rollout_steps,device=args.device,
        microbatch_size=getattr(args,'vision_ppo_microbatch_size',128),
        inference_batch_size=getattr(args,'vision_inference_batch_size',128))
    if native_camera:
        from .tabletop_vision_memory import memory_snapshot
        memory_snapshot(args,'camera_and_PPO_ready',device_world.w)
    iteration=timesteps=0
    if args.resume:
        saved=torch.load(args.resume,map_location=args.device,weights_only=False)
        if saved.get('version')!=VERSION:
            raise ValueError('visual resume requires the same visual actor/critic/camera/reset contract')
        from .tabletop_vision_resume import prepare_visual_resume
        saved=prepare_visual_resume(saved,contract,
            allow_downsize=getattr(args,'vision_resume_downsize',False))
        if saved.get('resume_migration') is not None:
            algorithm.vision_resume_migration=saved['resume_migration']
            _write_json(output/'resume_migration.json',saved['resume_migration'])
        algorithm.policy.load_state_dict(saved['policy'])
        algorithm.optimizer.load_state_dict(saved['optimizer'])
        algorithm.learning_rate=saved['learning_rate']
        iteration,timesteps=saved['iteration'],saved['timesteps']
        for name in ('episodes','successes'):
            getattr(env,name).copy_(saved[name])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if saved.get('cuda_rng'):
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    _write_json(output/'training_start.json',dict(
        observation_mode='vision_sensor_only_actor',privileged_actor_inputs=False,
        initial_iteration=iteration,initial_timesteps=timesteps,
        initial_episodes=int(env.episodes.sum()),initial_successes=int(env.successes.sum()),
        resume_source=str(args.resume) if args.resume else None,
        optimizer_state_entries=len(algorithm.optimizer.state),
        actor_observation_size=POLICY_DIM,critic_observation_size=371,
        teacher_distillation=contract['teacher_distillation']))
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
    row=dict(status='initializing_visual_policy',backend=VERSION,iteration=iteration,
             iteration_budget=args.max_iterations,successes=int(env.successes.sum()))
    _write_json(output/'monitor.json',row)
    try:
        while iteration<args.max_iterations:
            begin=time.monotonic()
            reward_sum=0.
            failure_counts={}
            visible_fraction=0.
            peak_lift=torch.full((args.num_envs,),-torch.inf,device=args.device)
            with torch.inference_mode():
                for step in range(args.rollout_steps):
                    action=algorithm.act(obs)
                    if not bool(torch.isfinite(action).all()):
                        raise RuntimeError('nonfinite visual action before physics')
                    issued=action.clamp(-1,1)
                    critic,reward,done,extra=env.step(issued)
                    # Read the pre-task-reset diagnostic, including terminal samples.
                    step_lift=extra.get('surface_diagnostics',{}).get('net_lift_m')
                    if step_lift is None:
                        step_lift=task.m['bottom']-task.surface_reference_bottom_m
                    step_lift=torch.as_tensor(step_lift,device=args.device).float()
                    if step_lift.shape!=(args.num_envs,) or not torch.isfinite(step_lift).all():
                        raise RuntimeError('invalid privileged net-lift diagnostic')
                    peak_lift=torch.maximum(peak_lift,step_lift)
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
                    obs=_observations(packet,device_world,previous,history,critic,done)
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
            row=dict(status='training',backend=VERSION,observation_mode='vision_sensor_only_actor',
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
            if torch.cuda.is_available() and str(args.device).startswith('cuda'):
                free,total=torch.cuda.mem_get_info()
                row['torch_peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
                row['torch_peak_reserved_gib']=torch.cuda.max_memory_reserved()/1024**3
                row['rollout_observation_device']=str(algorithm.storage.observations['policy'].device)
                row['rollout_observation_pinned']=algorithm.storage.observations['policy'].is_pinned()
                row['gpu_memory_used_gib']=(total-free)/1024**3
            # Only a complete optimizer update commits policy and counters.
            _save(checkpoint,algorithm,contract,iteration,timesteps,env)
            _write_json(output/'monitor.json',row)
            collection.update(phase='optimizer_complete',wall_unix_s=time.time())
            _write_json(output/'collection_status.json',collection)
            with (output/'iterations.jsonl').open('a') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
            for name in ('successes','episodes','mean_step_reward','object_visible_env_fraction',
                         'iteration_seconds','sampled_object_points_mean'):
                writer.add_scalar(name,row[name],iteration)
            for label,metrics in groups.items():
                for name,value in metrics.items():
                    writer.add_scalar(f'grasp_style/{label}/{name}',value,iteration)
            print('VISION_STUDENT_PPO_ITERATION',json.dumps(row),flush=True)
    except BaseException as error:
        row.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',
                   error=type(error).__name__+': '+str(error))
        _write_json(output/'monitor.json',row)
        raise
    finally:
        writer.close()
    row.update(status='training_budget_complete')
    _write_json(output/'monitor.json',row)
    return dict(status='training_budget_complete',iteration=iteration,timesteps=timesteps,
        successes=int(env.successes.sum()),visual_student=True,privileged_actor_inputs=False,
        actor_observation_size=POLICY_DIM,critic_observation_size=371,
        checkpoint=str(checkpoint))
