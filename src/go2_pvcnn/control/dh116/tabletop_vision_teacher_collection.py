"""Independent, bounded native Teacher/MPC supervision; never a PPO rollout."""
import hashlib
import json
import math
from pathlib import Path
import time

import torch


def validate_options(args):
    source=getattr(args,'vision_teacher_source',None)
    if not source:
        if getattr(args,'vision_teacher_mpc',False):
            raise ValueError('vision Teacher MPC requires a Teacher source')
        return
    if (not getattr(args,'vision_student',False)
            or getattr(args,'vision_camera_mode',None)!='fixed'
            or getattr(args,'vision_action_distribution',None)!='tanh_gaussian'
            or not 2<=args.num_envs<=16 or args.num_envs%2):
        raise ValueError('Teacher collection requires fixed visual tanh_gaussian and 2..16 even environments')
    if any(getattr(args,name,None) for name in ('resume','vision_evaluation_checkpoint',
            'vision_evaluation_manifest','grasp_contact_probe','surface_mpc',
            'vision_experiment_initial_state_audit')):
        raise ValueError('Teacher supervision collection excludes resume, PPO, evaluation and inference MPC modes')
    if (getattr(args,'vision_lift_reward_experiment','R0')!='R0'
            or getattr(args,'vision_warm_start_optimizer','preserve')!='preserve'):
        raise ValueError('Teacher collection cannot change rewards or PPO Adam')
    if not 1<=args.vision_teacher_max_control_steps<=10000:
        raise ValueError('Teacher collection requires a finite 1..10000 control-step budget')
    if not 1<=args.vision_teacher_raw_chunk_frames<=64:
        raise ValueError('raw RGB-D chunks require 1..64 frames')


def _exclusive_json(path,data):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(data,stream,indent=2,allow_nan=False)


class RawRGBDChunks:
    """CPU-owned real frames, bounded memory, exclusive evidence files."""
    def __init__(self,output,*,chunk_frames=16):
        if type(chunk_frames) is not int or not 1<=chunk_frames<=64:
            raise ValueError('raw chunk frame budget must be 1..64')
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.chunk_frames=chunk_frames;self.pending=[];self.chunks=[]
        self.frames=0;self.last_frame=-1

    def __call__(self,frame):
        actual=float(frame['simulation_time']);rendered=float(frame['rendered_simulation_time'])
        if not math.isfinite(actual) or not math.isfinite(rendered) or abs(actual-rendered)>1e-6:
            raise ValueError('raw RGB-D time mismatch')
        if type(frame['frame_id']) is not int or frame['frame_id']<=self.last_frame:
            raise ValueError('raw RGB-D frame identity must increase')
        copied={k:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v
                for k,v in frame.items()}
        self.pending.append(copied);self.last_frame=frame['frame_id'];self.frames+=1
        if len(self.pending)>=self.chunk_frames:self._flush()

    def _flush(self):
        if not self.pending:return
        path=self.output/f'chunk_{len(self.chunks):05d}.pt'
        with path.open('xb') as stream:torch.save(self.pending,stream)
        self.chunks.append(dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            frame_ids=[row['frame_id'] for row in self.pending]))
        self.pending.clear()

    def close(self):
        self._flush()
        _exclusive_json(self.output/'manifest.json',dict(version='dh116-real-rgbd-chunks-v1',
            frames=self.frames,chunks=self.chunks,source='native_same_pre_action_frame',
            sensor_only=True,depth_unit='m',rgb_unit='uint8'))


def _choose_references(source,n):
    selected={}
    for item in source['episodes']:
        selected.setdefault(item['grasp_style'],item)
    styles=['pinch_thumb_index']*(n//2)+['tripod_thumb_index_middle']*(n//2)
    if any(style not in selected for style in styles):
        raise ValueError('Teacher source must contain both canonical grasp styles')
    return [selected[style] for style in styles]


@torch.no_grad()
def run_teacher_collection(world,device_world,task,env,args,critic_contract,*,camera=None):
    """Capture actions actually executed at the recorded pre-action visual state.

    No RSL algorithm, optimizer, log probability or PPO storage is constructed.
    Historical time-indexed actions are only references for this physical replay,
    never a state-query expert for a different Student trajectory.
    """
    from .tabletop_vision_training import _contract,_observations,_write_json
    from .tabletop_vision_student import SensorHistory
    from .tabletop_vision_supervision_dataset import load_teacher_reference_dataset,build_dataset,save_dataset
    validate_options(args)
    native_camera=camera is None
    genuine=False
    if native_camera:
        from .tabletop_tensor_world import TensorWorld
        from .tabletop_tensor_env import TensorEnv
        genuine=(type(device_world) is TensorWorld and type(env) is TensorEnv
            and type(world).__module__.endswith('tabletop_gpu_isaac')
            and hasattr(world,'sim') and type(world.sim).__module__.startswith('isaaclab.'))
        if not genuine:
            raise ValueError('native Teacher supervision requires the genuine Isaac TensorWorld/TensorEnv')
    output=Path(args.output)
    source=load_teacher_reference_dataset(args.vision_teacher_source,critic_contract)
    references=_choose_references(source,args.num_envs)
    contract=_contract(args,critic_contract)
    _exclusive_json(output/'vision_student_contract.json',contract)
    _exclusive_json(output/'teacher_reference_identity.json',{k:v for k,v in source.items()
        if k not in ('episodes','references')})
    raw=RawRGBDChunks(output/'raw_rgbd',chunk_frames=args.vision_teacher_raw_chunk_frames)
    if camera is None:
        from .tabletop_vision_fixed_camera import FixedVisionCamera
        camera=FixedVisionCamera(world,device_world,args.depth_camera_config,
            width=args.vision_width,height=args.vision_height,num_points=128)
    camera.set_raw_frame_sink(raw)
    history=SensorHistory(args.num_envs,device=args.device)
    previous=torch.zeros((args.num_envs,24),device=args.device)
    trace_offset=len(getattr(device_world,'surface_trace',[]))
    if hasattr(device_world,'surface_trace_limit'):
        device_world.surface_trace_limit=max(device_world.surface_trace_limit,
            trace_offset+args.vision_teacher_max_control_steps*device_world.decimation)
    critic=env.reset()
    if native_camera:
        initial=dict(q=device_world.q.detach().cpu().clone(),qd=device_world.qd.detach().cpu().clone(),
            command=task.command.detach().cpu().clone(),roots=device_world.roots.detach().cpu().clone(),
            object_position=device_world.wp.detach().cpu().clone(),object_quaternion=device_world.wq.detach().cpu().clone(),
            reference_bottom=task.surface_reference_bottom_m.detach().cpu().clone(),seed=args.seed,
            source='actual_native_cold_reset_before_camera')
        with (output/'initial_native_state.pt').open('xb') as stream:torch.save(initial,stream)
    if hasattr(env,'enable_palm_lift_telemetry'):env.enable_palm_lift_telemetry()
    optimizer=None
    if args.vision_teacher_mpc:
        from .tabletop_vision_teacher_mpc import TeacherReferenceMPC
        optimizer=TeacherReferenceMPC(env,source['source_sha256'])
    ended=torch.zeros(args.num_envs,dtype=torch.bool,device=args.device)
    recovered=ended.clone();accepted=ended.clone();observed_success=ended.clone()
    terminal_records={};rows=[];issued_rows=[];reference_rows=[];metadata=[]
    start=time.monotonic();mpc_active=mpc_changed=0
    max_steps=min(args.vision_teacher_max_control_steps,max(len(item['actions']) for item in references))
    diagnostics_path=output/'teacher_mpc_diagnostics.jsonl'
    physics_path=output/'teacher_native_physics.jsonl'
    failure=None
    with diagnostics_path.open('x',encoding='utf-8') as diagnostics,physics_path.open('x',encoding='utf-8') as physics:
        try:
            for tick in range(max_steps):
                packet=camera.sample()
                obs=_observations(packet,device_world,previous,history,critic,
                    reset_mask=ended if tick==0 else done)
                if tick==0 and not bool(packet['object_valid'].all()):
                    raise RuntimeError('native Teacher collection requires initially visible objects')
                if tick==0 and native_camera:
                    from .tabletop_vision_audit import audit_initial_points
                    _exclusive_json(output/'initial_camera_geometry_validation.json',
                        audit_initial_points(packet,device_world,
                            radius=getattr(args,'solid_cylinder_radius_m',.008),length=.5))
                reference=torch.stack([item['actions'][min(tick,len(item['actions'])-1)]
                    for item in references]).to(device=args.device)
                # Exhausted/ended trajectories execute zero, never repeat a terminal source action.
                alive=~ended&torch.tensor([tick<len(item['actions']) for item in references],device=args.device)
                reference[~alive]=0.
                issued=reference.clone()
                if optimizer is not None:
                    issued,detail=optimizer.refine(reference,context=dict(tick=tick,
                        physics_step=int(world.sim.current_time_step_index),
                        simulation_time=float(world.sim.current_time),episode_id=list(range(args.num_envs))))
                    issued[~alive]=0.
                    detail['recorded_env_ids']=alive.nonzero().flatten().cpu().tolist()
                    diagnostics.write(json.dumps(detail,allow_nan=False)+'\n')
                    mpc_active+=sum(bool(row['active']) for row in detail['per_env'] if alive[row['env']])
                    mpc_changed+=sum(bool(row['changed']) for row in detail['per_env'] if alive[row['env']])
                if issued.shape!=(args.num_envs,24) or not bool(torch.isfinite(issued).all()&(issued.abs()<=1).all()):
                    raise RuntimeError('Teacher/MPC issued action must be finite bounded canonical 24D')
                for index in alive.nonzero().flatten().cpu().tolist():
                    item=references[index]
                    rows.append(obs['policy'][index].detach().cpu().clone())
                    issued_rows.append(issued[index].cpu().clone());reference_rows.append(reference[index].cpu().clone())
                    metadata.append(dict(env_id=index,episode_id=index,tick=tick,frame_id=int(packet['frame_id']),
                        simulation_time_s=float(packet['simulation_time']),physics_step=int(world.sim.current_time_step_index),
                        grasp_style=item['grasp_style'],source_episode_id=item['source_episode_id'],
                        source_trajectory_id=item['source_trajectory_id'],
                        previous_action=previous[index].cpu().tolist()))
                critic,reward,done,extra=env.step(issued.clone())
                if optimizer is not None:optimizer.observe_result(done|~alive)
                recovered|=~ended&extra['recovered'].bool()
                first=done&~ended
                first_success=first&extra['success'].bool()&(extra['failure']==0)&~recovered
                observed_success|=first_success
                # Do not qualify a short reference that succeeds during an unrecorded zero-action tail.
                accepted|=first_success&alive
                # Data is read before reset from the TensorEnv's terminal diagnostics.
                record=dict(tick=tick,physics_step=int(world.sim.current_time_step_index),
                    simulation_time_s=float(world.sim.current_time),done=done.cpu().tolist(),
                    success=extra['success'].bool().cpu().tolist(),failure=extra['failure'].cpu().tolist(),
                    recovered=extra['recovered'].bool().cpu().tolist())
                for key,value in extra.get('surface_diagnostics',{}).items():
                    if isinstance(value,torch.Tensor):record[key]=value.detach().cpu().tolist()
                physics.write(json.dumps(record,allow_nan=False)+'\n')
                for index in first.nonzero().flatten().cpu().tolist():
                    terminal_records[index]=dict(control_steps=tick+1,success=bool(first_success[index]),
                        reference_sequence_complete=bool(alive[index]),
                        failure=int(extra['failure'][index]),recovered=bool(recovered[index]))
                ended|=done
                previous=issued.clone();previous[done]=0.
                if (tick+1)%60==0 or bool(ended.all()) or tick+1==max_steps:
                    diagnostics.flush();physics.flush()
                    progress=dict(status='collecting_teacher_visual_supervision',control_steps=tick+1,
                        max_control_steps=max_steps,successful_episodes=int(accepted.sum()) if genuine else 0,
                        first_episode_done=ended.cpu().tolist(),mpc_active_samples=mpc_active,
                        mpc_changed_samples=mpc_changed,elapsed_wall_s=time.monotonic()-start,
                        ppo_updates=0)
                    _write_json(output/'monitor.json',progress)
                    print('VISION_TEACHER_COLLECTION',json.dumps(progress),flush=True)
                if bool(ended.all()):break
        except Exception as error:
            failure=repr(error)
            raise
        finally:
            camera.set_raw_frame_sink(None);raw.close()
            if failure is not None:
                _exclusive_json(output/'collection_failure.json',dict(error=failure,samples=len(rows),ppo_updates=0))
                with (output/'partial_unqualified_samples.pt').open('xb') as stream:
                    torch.save(dict(observations=rows,actions=issued_rows,reference_actions=reference_rows,
                        sample_metadata=metadata,error=failure,qualification=False,
                        last_action_execution_may_be_incomplete=True),stream)
    native_accepted=observed_success.clone()
    style_audit={}
    traces=getattr(device_world,'surface_trace',[])[trace_offset:]
    if traces:
        from .tabletop_surface_demonstrations import contact_style
        native_physics={key:torch.stack([row[key] for row in traces]) for key in traces[0]}
        with (output/'native_physics_substeps.pt').open('xb') as stream:torch.save(native_physics,stream)
        for index in accepted.nonzero().flatten().cpu().tolist():
            hi=terminal_records[index]['control_steps']*device_world.decimation
            lo=max(0,hi-300*device_world.decimation)
            loads=native_physics['effective_pad_loads_n'][lo:hi,index]
            style_audit[index]=contact_style(loads.reshape(-1,device_world.decimation,2,5).mean(1))
            accepted[index]=style_audit[index]['style']==references[index]['grasp_style']
    elif native_camera:
        accepted.zero_()  # Native successful labels need the independently recorded contact-style audit.
    if not genuine:
        accepted.zero_()  # CPU-injected cameras/worlds cannot qualify physical demonstrations.
    episodes={str(index):dict(env_id=index,first_episode=True,success=bool(accepted[index]),
        recovered=bool(recovered[index]),native_physics=genuine,
        grasp_style=item['grasp_style'],source_trajectory_id=item['source_trajectory_id'],
        source_episode_id=item['source_episode_id'],proof_path=str(output/'teacher_collection_report.json'),
        terminal=terminal_records.get(index),native_contact_style_audit=style_audit.get(index,
            'not_audited_failed_or_incomplete_episode' if genuine else 'fixture_not_native'))
        for index,item in enumerate(references)}
    dataset=build_dataset(torch.stack(rows),torch.stack(issued_rows),torch.stack(reference_rows),metadata,episodes,
        visual_contract=contract,teacher_source_sha256=source['source_sha256'],require_successful=False)
    # Always keep the failed/unfinished physical experiment separate from successful demonstrations.
    with (output/'all_executed_supervision.pt').open('xb') as stream:torch.save(dataset,stream)
    success_ids=set(accepted.nonzero().flatten().cpu().tolist())
    keep=[i for i,row in enumerate(metadata) if row['env_id'] in success_ids]
    successful=None
    if keep:
        qualified=build_dataset(dataset['observations'][keep],dataset['actions'][keep],
            dataset['reference_actions'][keep],[metadata[i] for i in keep],
            {key:value for key,value in episodes.items() if int(key) in success_ids},
            visual_contract=contract,teacher_source_sha256=source['source_sha256'])
        successful=save_dataset(qualified,output/'successful_visual_supervision.pt')
    report=dict(status='completed',mode='teacher_reference_mpc' if optimizer else 'teacher_reference',
        source_sha256=source['source_sha256'],control_steps=tick+1,samples=len(rows),
        successful_episodes=len(success_ids),native_runtime_verified=genuine,
        native_successful_episodes=int(native_accepted.sum()) if genuine else 0,
        observed_terminal_successes=int(native_accepted.sum()),
        contact_style_audit=style_audit,episodes=episodes,successful_dataset=successful,
        mpc_active_samples=mpc_active,mpc_changed_samples=mpc_changed,
        actor_observation_size=3844,critic_observation_size=371,action_size=24,
        ppo_updates=0,student_policy_trained=False,teacher_success_is_student_success=False,
        dagger_state_query_oracle=False,raw_rgbd_frames=raw.frames,elapsed_wall_s=time.monotonic()-start)
    _exclusive_json(output/'teacher_collection_report.json',report)
    _write_json(output/'monitor.json',report)
    return dict(report,grasp_qualified=False)
