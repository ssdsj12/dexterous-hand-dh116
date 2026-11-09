"""Opt-in independent original/unilateral/common full-first-episode experiment.

No simulator is started here. The caller owns the single native Isaac instance.
This runner never uses the production evaluation writer or issues a Student
certificate. Original raw Student and privileged hybrid outcomes stay separate.
"""
from copy import copy
import gzip
import json
import math
from pathlib import Path
import uuid

import numpy as np
import torch

from .tabletop_joint_target_readback import BACKEND_SOURCE
from .tabletop_vision_evaluation_contract import file_sha256

VERSION='dh116-full-episode-tracking-experiment-v1'
ARMS=('original','unilateral','common')
BASELINE_C58_SHA256='c58f9cf0718d36cd9feec181c9180a9657e97b54d6fa8e99ce1b3f16244f6d97'


def _plain(value):
    if isinstance(value,torch.Tensor):return _plain(value.detach().cpu().tolist())
    if isinstance(value,np.ndarray):return _plain(value.tolist())
    if isinstance(value,np.generic):return _plain(value.item())
    if isinstance(value,dict):return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return {'nonfinite_original':str(value)}
    return value


def _new_json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(_plain(value),stream,indent=2,allow_nan=False)


def _line(stream,value):
    stream.write(json.dumps(_plain(value),allow_nan=False,separators=(',',':'))+'\n')


def _identity(value,base):
    if not isinstance(value,dict) or not isinstance(value.get('path'),str):raise ValueError('explicit input path/SHA required')
    path=Path(value['path']);path=path if path.is_absolute() else base/path
    sha=value.get('sha256')
    if not isinstance(sha,str) or len(sha)!=64 or file_sha256(path)!=sha.lower():raise ValueError('actual immutable input SHA mismatch')
    return dict(value,path=str(path.resolve()),sha256=sha.lower())


def load_tracking_config(path):
    path=Path(path).resolve();config=json.loads(path.read_text(encoding='utf-8'))
    if (config.get('version')!=VERSION or config.get('arm') not in ARMS
            or config.get('scene_seed')!=42 or config.get('num_envs')!=8
            or config.get('initial_state_tolerance',0.)!=0.):
        raise ValueError('explicit original/unilateral/common eight-environment seed42 exact-reset config required')
    for key in ('checkpoint','source_initial_state'):config[key]=_identity(config.get(key),path.parent)
    if config['checkpoint']['sha256']!=BASELINE_C58_SHA256:raise ValueError('complete original c58 SHA256 required; no short-prefix model substitution')
    if config['arm']=='common':config['calibration']=_identity(config.get('calibration'),path.parent)
    config['config_identity']=dict(path=str(path),sha256=file_sha256(path))
    return config


def audit_no_force_unload_profile(task,world):
    """Admit only the original surface branch with no force-unload branch.

    Source audit: free_wrist_grasp.control's surface branch uses
    surface_final_guard and sets over=zeros. The non-surface pressure_limited
    branch is not executed. finger_pre_guard's policy-owned path returns before
    pinch-force/mimic servo code. This does not infer priority from action signs.
    """
    from .tabletop_vision_command_shadow import controller_contract
    audit=controller_contract(task,world)
    if audit['surface_policy_owned_targets'] is not True or audit['pinch_force_hold'] is not False:
        raise ValueError('only audited surface-owned/no-pinch profile has known no force-unload branch')
    audit.update(automatic_force_unload_branch=False,
        mandatory_source='source-audited current surface-owned/no-redistribution/no-pinch/no-callback branch;not_action_sign')
    return audit


def no_force_unload_feedback(audit):
    if audit.get('automatic_force_unload_branch') is not False:raise ValueError('audited original profile required')
    def feedback(world,task,live):
        from .tabletop_vision_command_shadow import UNSUPPORTED_CALLBACKS
        if (not getattr(task,'surface_policy_owned_targets',False) or getattr(task,'surface_redistribution',None) is not False
                or getattr(task,'pinch_force_hold',False) or not getattr(task,'surface_force_sharing',False)
                or any(getattr(task,name,None) is not None for name in UNSUPPORTED_CALLBACKS)):
            raise ValueError('original priority profile changed; unknown mandatory unloading refuses the experiment')
        return dict(mandatory_release_mask=torch.zeros(world.n,12,device=live.device,dtype=torch.bool),
            mandatory_wrist_release_mask=torch.zeros(world.n,2,device=live.device,dtype=torch.bool),
            overload=torch.zeros(world.n,device=live.device,dtype=torch.bool))
    return feedback


def load_qualified_calibration(identity,audit):
    identity=_identity(identity,Path.cwd())
    report=json.loads(Path(identity['path']).read_text(encoding='utf-8'))
    if (identity.get('source_run')!='02' or report.get('version')!='dh116-teacher-tracking-calibration-v1'
            or report.get('qualified') is not True or report.get('target_source')!=BACKEND_SOURCE
            or report.get('feedback_source')!='original_advance_returned_refresh_before_TensorEnv_terminal_reset'):
        raise ValueError('actual 02 qualified PhysX/post-physics calibration report required')
    for source in ('dataset','context'):
        if file_sha256(report[source+'_path'])!=report[source+'_sha256']:raise ValueError('calibration source bytes changed')
    for key in ('controller_source_sha256','action_contract_sha256'):
        if report.get(key)!=audit.get(key):raise ValueError('calibration original controller/action identity differs')
    p=torch.as_tensor(report.get('teacher_error_p999_rad'),dtype=torch.float64)
    trigger=torch.as_tensor(report.get('trigger_rad'),dtype=torch.float64)
    if p.shape!=(12,) or not torch.isfinite(p).all() or bool((p<0).any()) or not torch.equal(trigger,torch.maximum(1.5*p,p.new_full((12,),.01))):
        raise ValueError('qualified per-joint P99.9/trigger calibration mismatch')
    return report


def project_unilateral_tracking_action(guard,action,world,task):
    """Use the unchanged controller's dtype/device before the legacy guard.

    TensorTask.begin_step already makes this exact cast in the native loop.
    Keeping it explicit at this opt-in boundary also admits a direct float32
    Actor request without changing the legacy guard's strict SDK chart check.
    The cast changes no gain, bound or action value; raw and executed requests
    remain separate evidence.
    """
    from .tabletop_bimanual_experiment_runtime import apply_tracking_guard
    controller_action=action.to(task.command)
    result,info=apply_tracking_guard(guard,controller_action,world,task)
    info.update(adapter_input_action24=action.detach().clone(),adapter_input_action24_dtype=str(action.dtype),
        original_controller_action24=controller_action.detach().clone(),
        original_controller_action24_dtype=str(controller_action.dtype),
        projected_controller_action24=result.detach().clone(),projected_controller_action24_dtype=str(result.dtype))
    return result,info


def body_link_snapshot(world):
    """Owned named physical body poses; no root/target substitution for palms."""
    rows=[]
    for hand in world.hands:
        names=list(getattr(hand,'body_names',[]));poses=getattr(hand.data,'body_link_pose_w',None)
        available=(isinstance(poses,torch.Tensor) and poses.shape==(world.n,len(names),7)
            and bool(names) and len(set(names))==len(names))
        rows.append(dict(body_names=names,available=available,
            poses_w=poses.detach().cpu().clone() if available else None,
            finite_env=torch.isfinite(poses).flatten(1).all(1).detach().cpu() if available else None,
            source='actual hand.data.body_link_pose_w;world_xyz_m_wxyz'))
    return rows


def body_link_motion(world,initial):
    current=body_link_snapshot(world);palms=[];displacements=[];available=[]
    for row,start in zip(current,initial):
        matching=(row['available'] and start['available'])
        common=[name for name in row['body_names'] if name in start['body_names']] if matching else []
        row['reset_displacement_by_body_name_m']={name:
            row['poses_w'][:,row['body_names'].index(name),:3]-start['poses_w'][:,start['body_names'].index(name),:3]
            for name in common}
        palm=matching and 'palm_link' in common
        available.append(bool(palm))
        if palm:
            palms.append(row['poses_w'][:,row['body_names'].index('palm_link')])
            displacements.append(row['reset_displacement_by_body_name_m']['palm_link'])
    return dict(actual_named_body_links=current,palm_body_available=available,
        palm_body_pose_w=torch.stack(palms,1) if all(available) else None,
        palm_body_displacement_m=torch.stack(displacements,1) if all(available) else None,
        palm_body_source='explicit palm_link physical body only;missing_is_NA;no_base_or_CAD_center_fallback',
        legacy_named_palm_diagnostic_meaning='original TensorWorld actual_palm_pose_m/palm_displacement_m are base_link origin displacement in driven mode')


class FullEpisodeTrackingRecorder:
    """Freeze all eight first native terminals; never treat candidate stop as one."""
    def __init__(self,arm,*,native,sink):
        if arm not in ARMS or not callable(sink):raise ValueError('explicit tracking arm and raw sink required')
        self.arm=arm;self.native=native;self.sink=sink
        self.run_uuid=str(uuid.uuid4());self.episode_ids=None;self.episode_uids=None
        self.closed=torch.zeros(8,dtype=torch.bool);self.first_terminal={};self.guard_aborts=[]
        self.pause=torch.zeros(8,dtype=torch.float64);self.max_pause=self.pause.clone()
        self.candidate_terminated=torch.zeros(8,dtype=torch.bool)
        self.max_hold=[None]*8;self.max_slip=[None]*8;self.max_lift=[None]*8
        self.max_left_link_force=[None]*8;self.max_right_link_force=[None]*8
        self.left_force_over_30=torch.zeros(8,dtype=torch.long);self.right_force_over_30=self.left_force_over_30.clone()
        self.left_joint_boundary=torch.zeros(8,dtype=torch.long);self.right_joint_boundary=self.left_joint_boundary.clone()
        self.joint_boundary_available=torch.zeros(8,dtype=torch.long)
        self.samples=torch.zeros(8,dtype=torch.long);self.policy_step=-1

    def begin(self,episode_ids):
        if self.episode_ids is not None:raise RuntimeError('first episode identities already frozen')
        if not isinstance(episode_ids,torch.Tensor) or episode_ids.shape!=(8,) or episode_ids.dtype!=torch.long:
            raise ValueError('actual diagnostic_episode_ids int64[8] required')
        self.episode_ids=episode_ids.detach().cpu().clone()
        self.episode_uids=[f'{self.run_uuid}:{i}:{int(self.episode_ids[i])}' for i in range(8)]

    def identity(self,episode_ids=None):
        if episode_ids is None:
            if self.native:raise ValueError('actual pre-reset diagnostic episode identity required')
            episode_ids=self.episode_ids if self.episode_ids is not None else torch.zeros(8,dtype=torch.long)
        if self.episode_ids is None:self.begin(episode_ids)
        if episode_ids.shape!=(8,) or episode_ids.dtype!=torch.long:raise ValueError('actual int64[8] episode identity required')
        current=episode_ids.detach().cpu().clone()
        if bool(((current!=self.episode_ids)&~self.closed).any()):raise RuntimeError('physical episode identity changed before owned first terminal')
        return dict(run_uuid=self.run_uuid,env_ids=list(range(8)),episode_ids=self.episode_ids.clone(),
            episode_uids=list(self.episode_uids),diagnostic_episode_ids_at_sample=current,
            diagnostic_episode_uids_at_sample=[f'{self.run_uuid}:{i}:{int(current[i])}' for i in range(8)],
            identity_source='actual TensorEnv.diagnostic_episode_ids before terminal reset;counter increments at control tick end;first_episode_live controls ownership')

    def observe(self,*,physics_step,timestamp_s,live,done,success,failure,measurement,guard_info,metrics,raw,episode_ids=None):
        identity=self.identity(episode_ids)
        live=live.detach().cpu();done=done.detach().cpu();success=success.detach().cpu();failure=failure.detach().cpu()
        active=live&~self.closed
        state=guard_info.get('state');pause=guard_info.get('pause_hand')
        paused=torch.zeros(8,dtype=torch.bool) if pause is None else pause.detach().cpu().any(-1)
        if state is not None:
            terminal_candidate=state.detach().cpu()==4
            self.candidate_terminated |= terminal_candidate&active
            paused |= terminal_candidate
        self.pause=torch.where(active&paused,self.pause+1/480,torch.zeros_like(self.pause))
        self.max_pause=torch.maximum(self.max_pause,self.pause)
        self.samples+=active.long()
        for i in active.nonzero().flatten().tolist():
            joint_fault=[False,False];joint_available=False
            low,high=raw.get('original_joint_lower'),raw.get('original_joint_upper')
            if isinstance(low,torch.Tensor) and isinstance(high,torch.Tensor) and measurement.get('q') is not None:
                from .tabletop_tensor_env import TRAINING_JOINT_LIMIT_CONTRACT
                margin=TRAINING_JOINT_LIMIT_CONTRACT['outside_nominal_limit_rad']
                low=low.detach().cpu();high=high.detach().cpu();q=measurement['q'][i].detach().cpu()
                if low.ndim==2:low=low[i]
                if high.ndim==2:high=high[i]
                joint_available=bool(torch.isfinite(q).all())
                if joint_available:
                    for h in range(2):joint_fault[h]=bool(((q[6*h:6*h+6]<low[6*h:6*h+6]-margin)|(q[6*h:6*h+6]>high[6*h:6*h+6]+margin)).any())
            full,limits=raw.get('full_actual_q'),raw.get('original_sdk_limits')
            if isinstance(full,torch.Tensor) and isinstance(limits,torch.Tensor):
                for h in range(2):
                    q=full[i,h].detach().cpu();bound=limits[i,h].detach().cpu()
                    joint_fault[h] |= bool(((q<bound[:,0]-.02)|(q>bound[:,1]+.02)).any())  # unchanged physical_guard line
            self.joint_boundary_available[i]+=int(joint_available)
            self.left_joint_boundary[i]+=int(joint_fault[0]);self.right_joint_boundary[i]+=int(joint_fault[1])
            for field,target in (('hold_s',self.max_hold),('drift_m',self.max_slip),('net_lift_m',self.max_lift)):
                value=metrics.get(field)
                if value is not None and math.isfinite(float(value[i])):
                    target[i]=max(float(value[i]),target[i] if target[i] is not None else -math.inf)
            loads=measurement.get('object_link_loads')
            if loads is not None:
                for h,target,counter in ((0,self.max_left_link_force,self.left_force_over_30),(1,self.max_right_link_force,self.right_force_over_30)):
                    value=float(loads[i,h].amax())
                    if math.isfinite(value):
                        target[i]=max(value,target[i] if target[i] is not None else -math.inf)
                        counter[i]+=int(value>30.)  # unchanged original strict physical-guard line
            if bool(done[i]):
                self.first_terminal[i]=dict(env_id=i,grasp_style='pinch' if i<4 else 'tripod',
                    run_uuid=self.run_uuid,episode_id=int(self.episode_ids[i]),episode_uid=self.episode_uids[i],
                    policy_step=self.policy_step,physics_step=physics_step,timestamp_s=timestamp_s,
                    original_native_success_flag=bool(success[i]),original_failure_code=int(failure[i]),
                    native_runtime_verified=self.native,joint_boundary_evidence_available=joint_available,
                    left_joint_boundary_evidence=joint_fault[0] if joint_available else None,
                    right_joint_boundary_evidence=joint_fault[1] if joint_available else None)
                self.closed[i]=True
        self.sink(dict(**identity,event_kind='physics_substep_before_terminal_reset',arm=self.arm,policy_step=self.policy_step,
            physics_step=physics_step,timestamp_s=timestamp_s,first_episode_live=active,original_live=live,
            original_done=done,original_native_success=success,original_failure=failure,
            measurement=measurement,guard_state=guard_info,original_hold_slip_diagnostics=metrics,raw=raw,
            execution_classification='raw_student_original_control' if self.arm=='original' else 'privileged_hybrid_experiment',
            production_evaluation_certificate_issued=False))

    def summary(self):
        observed=sum(row['original_native_success_flag'] and row['original_failure_code']==0 for row in self.first_terminal.values())
        qualified=observed if self.native else 0
        styles={name:dict(evaluated=4,completed=sum(i in self.first_terminal for i in ids),
            observed_success=sum(self.first_terminal.get(i,{}).get('original_native_success_flag',False)
                and self.first_terminal.get(i,{}).get('original_failure_code')==0 for i in ids))
            for name,ids in (('pinch',range(4)),('tripod',range(4,8)))}
        return dict(arm=self.arm,evaluated_first_episodes=8,completed_first_episodes=len(self.first_terminal),
            run_uuid=self.run_uuid,env_ids=list(range(8)),episode_ids=None if self.episode_ids is None else self.episode_ids.tolist(),
            episode_uids=self.episode_uids,first_episode_identity_available=self.episode_ids is not None,
            all_first_episodes_completed=bool(self.closed.all()),styles=styles,first_terminal=list(self.first_terminal.values()),
            observed_original_native_success_count=observed,pure_student_original_native_success_count=qualified if self.arm=='original' else 0,
            privileged_hybrid_native_success_count=qualified if self.arm!='original' else 0,
            execution_classification='raw_student_original_control' if self.arm=='original' else 'privileged_hybrid_experiment',
            production_evaluation_certificate_issued=False,teacher_data_qualified=False,ppo_updates=0,
            max_qualified_hold_s=self.max_hold,max_fixed_anchor_slip_m=self.max_slip,max_net_lift_m=self.max_lift,
            max_left_link_force_n=self.max_left_link_force,max_right_link_force_n=self.max_right_link_force,
            left_force_over_original_30N_sample_count=self.left_force_over_30.tolist(),
            right_force_over_original_30N_sample_count=self.right_force_over_30.tolist(),
            left_joint_boundary_crossed_sample_count=self.left_joint_boundary.tolist(),
            right_joint_boundary_crossed_sample_count=self.right_joint_boundary.tolist(),
            joint_boundary_available_sample_count=self.joint_boundary_available.tolist(),
            left_joint_failure_terminal_count=sum(row['original_failure_code']==3 and row['left_joint_boundary_evidence'] is True for row in self.first_terminal.values()),
            right_joint_failure_terminal_count=sum(row['original_failure_code']==3 and row['right_joint_boundary_evidence'] is True for row in self.first_terminal.values()),
            side_attribution='original_task_failure_plus_measured_joint_boundary_evidence;not_causal_proof',
            candidate_terminated_env_ids=self.candidate_terminated.nonzero().flatten().tolist(),
            maximum_contiguous_pause_s=self.max_pause.tolist(),pause_over_2s_env_ids=(self.max_pause>2.).nonzero().flatten().tolist(),
            deadlock_diagnostic='pause_over_2s_is_suspected_lock_only_not_a_new_task_terminal_or_proof',
            observed_physics_samples_per_env=self.samples.tolist(),original_guard_aborts=self.guard_aborts)


@torch.no_grad()
def run_full_episode_tracking_experiment(world,device_world,task,env,args,critic_contract,*,camera=None,policy=None):
    """One arm per fresh native process/output; all eight original first episodes.

    Explicit CLI integration: call this function and return before PPO or any
    standard Student evaluation writer. ``camera``/``policy`` injection is only
    a CPU fixture path and can never set native_runtime_verified=True.
    """
    from .tabletop_vision_branching_eval import _native_execution,_reset_snapshot,compare_source_reset,_clock
    from .tabletop_vision_training import _contract,_observations
    from .tabletop_vision_student import SensorHistory
    from .tabletop_bimanual_tracking_guard import read_common_pause_targets,BimanualTrackingGuard
    from .tabletop_bimanual_tracking_pause_runtime import install_common_tracking_pause
    from .tabletop_tensor_env import BOUNDED_FAILURE_RESET_CONTRACT,JOINT_LIMIT_RESET_CONTRACT,LIVE_VELOCITY_GUARD_CONTRACT
    config=load_tracking_config(args.vision_tracking_experiment_config);arm=config['arm']
    if args.num_envs!=8 or device_world.n!=8 or args.seed!=42 or device_world.dt!=1/480 or device_world.decimation!=8:
        raise ValueError('tracking comparison requires unchanged eight-env seed42 480Hz/8substep source')
    if (getattr(args,'vision_mirror_preprocessing','m0_raw')!='m0_raw'
            or args.vision_camera_mode!='fixed' or args.vision_action_distribution!='tanh_gaussian'):
        raise ValueError('tracking comparison requires original M0 fixed-camera Tanh24 source')
    if any(getattr(args,key,None) for key in ('resume','vision_teacher_source','vision_teacher_mpc','surface_mpc',
            'tripod_recovery_stage','surface_teacher_replay','vision_m2_replay','bimanual_explicit_sdk_adapter')):
        raise ValueError('tracking experiment is an independent raw-Student/hybrid branch; Teacher/MPC priority is unsupported')
    if not torch.equal(device_world.mixed_reset_style_ids.detach().cpu(),torch.tensor([0]*4+[1]*4)):
        raise ValueError('original four pinch plus four tripod template order required')
    if getattr(task,'episode_timeout_s',None)!=30.:
        raise ValueError('unchanged original 30s surface episode timeout required')
    if getattr(env,'joint_replay_observer',None) is not None or getattr(env,'control_probe',None) is not None or getattr(env,'bimanual_action_adapter',None) is not None:
        raise ValueError('independent tracking experiment cannot stack observers or control adapters')
    audit=audit_no_force_unload_profile(task,device_world)
    calibration=load_qualified_calibration(config['calibration'],audit) if arm=='common' else None
    native=_native_execution(world,device_world,env,camera is None and policy is None)
    if policy is None:
        from .tabletop_vision_inference import load_vision_inference
        from .tabletop_vision_evaluation import _verify_native_physical_adapter
        contract_args=copy(args)
        if getattr(contract_args,'rollout_steps',None) is None:contract_args.rollout_steps=24
        policy=load_vision_inference(config['checkpoint']['path'],_contract(contract_args,critic_contract),device=args.device,allow_physical_adapter=True)
        _verify_native_physical_adapter(policy,device_world)
    if policy.checkpoint_sha256!=config['checkpoint']['sha256']:raise ValueError('c58 model changed during inference loading')
    if camera is None:
        from .tabletop_vision_fixed_camera import FixedVisionCamera
        if [args.vision_width,args.vision_height]!=[224,96]:raise ValueError('original fixed camera224x96 required')
        camera=FixedVisionCamera(world,device_world,args.depth_camera_config,width=224,height=96,num_points=128)
    expected=torch.load(config['source_initial_state']['path'],map_location='cpu',weights_only=False)
    if expected.get('seed')!=42 or expected.get('source')!='actual_native_cold_reset_before_camera':
        raise ValueError('actual original eight-env cold-reset source reference required')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    _new_json(output/'tracking_experiment_config.json',config)
    _new_json(output/'tracking_original_thresholds.json',dict(controller=audit,joint_lower=task.low,joint_upper=task.high,
        bounded_failure=BOUNDED_FAILURE_RESET_CONTRACT,joint_limit=JOINT_LIMIT_RESET_CONTRACT,velocity=LIVE_VELOCITY_GUARD_CONTRACT,
        unchanged_success=dict(net_lift_m=.03,hold_s=5.,fixed_anchor_drift_m=.0005),calibration=calibration))
    error=None;caught=None;steps=0;common=None;guard_info={};issued=None;actor_raw=None;initial_bodies=[]
    prior_observer_present='joint_replay_observer' in vars(env);prior_observer=getattr(env,'joint_replay_observer',None)
    prior_adapter_present='bimanual_action_adapter' in vars(env);prior_adapter=getattr(env,'bimanual_action_adapter',None)
    prior_probe_present='control_probe' in vars(env);prior_probe=getattr(env,'control_probe',None)
    with gzip.open(output/'tracking_physics_480hz.jsonl.gz','xt',encoding='utf-8',compresslevel=1) as raw:
        recorder=FullEpisodeTrackingRecorder(arm,native=native,sink=lambda row:_line(raw,row))
        def observe(active_env,measurement,live,newly_terminal):
            info=common.last_info if common is not None else guard_info
            info={} if info is None else info
            metrics={}
            if getattr(task,'surface_lift_success',None) is not None:
                from .tabletop_surface_student_evaluation import diagnostic_snapshot
                metrics=diagnostic_snapshot(task)
            backend=read_common_pause_targets(device_world)
            recorder.observe(physics_step=device_world._physics_steps,timestamp_s=float(world.sim.current_time),
                live=live,done=task.done,success=task.success,failure=task.failure,measurement=measurement,guard_info=info,metrics=metrics,
                episode_ids=active_env.diagnostic_episode_ids,
                raw=dict(actor_action24=issued,raw_actor_action24=actor_raw,raw_actor_action24_dtype=str(actor_raw.dtype),
                    env_step_action24=issued,env_step_action24_dtype=str(issued.dtype),
                    original_controller_action24=issued.to(task.command),original_controller_action24_dtype=str(task.command.dtype),
                    backend_and_cache=backend,full_actual_q=device_world.q,
                    full_actual_qd=device_world.qd,actual_base_root_pose_w=device_world.roots,
                    actual_body_link_motion=body_link_motion(device_world,initial_bodies),
                    logical_command_cache=task.command,sdk_expected_float32=device_world.commands.float(),
                    original_joint_lower=task.low,original_joint_upper=task.high,
                    original_sdk_limits=getattr(device_world,'limits',None),
                    original_transport_diagnostics=getattr(device_world,'last_action_diagnostics',None)))
        class Probe:
            def before_step(self):pass
            def after_step(self,*unused):pass
            def on_physics_failure(self,mask,measurement,context):
                event=dict(event_kind='original_physical_guard_abort',mask=mask,context=context,
                    **recorder.identity(env.diagnostic_episode_ids),
                    physics_step=device_world._physics_steps,measurement=measurement,original_task_done=task.done,
                    original_failure=task.failure,actor_action24=issued,raw_actor_action24=actor_raw,
                    env_step_action24=issued,actual_full_q=device_world.q,
                    actual_full_qd=device_world.qd,actual_base_root_pose_w=device_world.roots,
                    actual_body_link_motion=body_link_motion(device_world,initial_bodies),
                    guard_state=common.last_info if common is not None else guard_info)
                try:event['backend_and_cache']=read_common_pause_targets(device_world)
                except Exception as read_error:event['backend_unavailable_without_fallback']=repr(read_error)
                recorder.guard_aborts.append(_plain(event));_line(raw,event);raw.flush()
        try:
            critic=env.reset();recorder.begin(env.diagnostic_episode_ids);before=_reset_snapshot(device_world,task)
            initial_bodies=body_link_snapshot(device_world)
            with (output/'tracking_body_link_reset_readback.pt').open('xb') as stream:
                torch.save(dict(**recorder.identity(env.diagnostic_episode_ids),actual_body_links=initial_bodies,
                    actual_base_root_pose_w=device_world.roots.detach().cpu().clone(),clock=_clock(device_world)),stream)
            reset=compare_source_reset(expected,before);_new_json(output/'tracking_reset_comparison.json',reset)
            if not reset['matched']:raise RuntimeError('exact original c58 source reset mismatch; no tolerance widened')
            before_clock=_clock(device_world);packet=camera.sample();after=_reset_snapshot(device_world,task)
            warmup=compare_source_reset(before,after)
            _new_json(output/'tracking_camera_warmup.json',dict(comparison=warmup,before_clock=before_clock,after_clock=_clock(device_world)))
            if not warmup['matched'] or before_clock!=_clock(device_world):raise RuntimeError('camera changed original reset state/clock')
            env.joint_replay_observer=observe;env.control_probe=Probe()
            if arm=='unilateral':
                old_guard=BimanualTrackingGuard(8,enabled=True,coupled_pause=False)
                def unilateral(action):
                    nonlocal guard_info
                    result,guard_info=project_unilateral_tracking_action(old_guard,action,device_world,task)
                    return result
                env.bimanual_action_adapter=unilateral
            elif arm=='common':
                p=torch.tensor(calibration['teacher_error_p999_rad'],device=task.command.device,dtype=task.command.dtype)
                common=install_common_tracking_pause(device_world,task,teacher_error_p999_rad=p,source='teacher_hybrid',
                    original_guard_feedback=no_force_unload_feedback(audit),sink=lambda row:_line(raw,dict(
                        **recorder.identity(env.diagnostic_episode_ids),event_kind='common_pause_target_record',record=row)))
            history=SensorHistory(8,device=args.device);previous=torch.zeros(8,24,device=args.device)
            obs=_observations(packet,device_world,previous,history,critic)
            with gzip.open(output/'tracking_policy_60hz.jsonl.gz','xt',encoding='utf-8',compresslevel=1) as controls:
                # One final policy tick can observe the original strict 30s
                # timeout despite floating accumulation; task timeout is never
                # increased and completed first episodes remain retired.
                for tick in range(1801):
                    recorder.policy_step=tick;actor_raw=policy.act(obs)
                    if actor_raw.shape!=(8,24) or not torch.isfinite(actor_raw).all() or bool((actor_raw.abs()>1).any()):raise ValueError('actual bounded24D Student action required')
                    actor_raw=actor_raw.detach().clone();issued=actor_raw.clone();issued[recorder.closed.to(issued.device)]=0.
                    _line(controls,dict(**recorder.identity(env.diagnostic_episode_ids),policy_step=tick,arm=arm,
                        actor_action24=issued,raw_actor_action24=actor_raw,raw_actor_action24_dtype=str(actor_raw.dtype),
                        env_step_action24=issued,env_step_action24_dtype=str(issued.dtype),
                        original_controller_action24=issued.to(task.command),original_controller_action24_dtype=str(task.command.dtype),
                        actual_previous_env_step_action24=previous))
                    critic,_,done,_=env.step(issued);steps=tick+1
                    previous=issued.clone();previous[done]=0.
                    if bool(recorder.closed.all()):break
                    obs=_observations(camera.sample(),device_world,previous,history,critic,reset_mask=done)
                    if steps%60==0:print('TRACKING_FULL_EPISODE',json.dumps(dict(arm=arm,control_steps=steps,first_terminals=int(recorder.closed.sum()))),flush=True)
        except BaseException as failure:
            caught=failure;error=f'{type(failure).__name__}: {failure}'
        finally:
            if common is not None:common.uninstall()
            if prior_observer_present:env.joint_replay_observer=prior_observer
            elif hasattr(env,'joint_replay_observer'):del env.joint_replay_observer
            if prior_adapter_present:env.bimanual_action_adapter=prior_adapter
            elif hasattr(env,'bimanual_action_adapter'):del env.bimanual_action_adapter
            if prior_probe_present:env.control_probe=prior_probe
            elif hasattr(env,'control_probe'):del env.control_probe
            raw.flush()
    for key in ('checkpoint','source_initial_state','calibration'):
        if key in config and file_sha256(config[key]['path'])!=config[key]['sha256']:error='immutable tracking input changed: '+key
    if file_sha256(config['config_identity']['path'])!=config['config_identity']['sha256']:error='immutable tracking config changed'
    if calibration is not None:
        for key in ('dataset','context'):
            if file_sha256(calibration[key+'_path'])!=calibration[key+'_sha256']:error='immutable calibration source changed: '+key
    report=recorder.summary();report.update(version=VERSION,status='interrupted' if isinstance(caught,(KeyboardInterrupt,SystemExit)) else 'error' if error else 'completed' if recorder.closed.all() else 'incomplete_at_original_30s_budget',
        error=error,native_runtime_verified=native,control_steps=steps,physics_hz=480,policy_hz=60,maximum_control_steps=1801,
        original_episode_timeout_s=30.,extra_tick_scope='observe_original_strict_timeout_only;no_new_first_episode_authority',
        scene_seed=42,checkpoint=config['checkpoint'],source_initial_state=config['source_initial_state'],
        actual_palm_body_diagnostic='explicit palm_link from physical body_link_pose_w;NA when absent;all body/base evidence retained',
        legacy_palm_named_fields='original driven-mode base_link origin displacement;not rewritten as palm_link motion',
        calibration_identity=config.get('calibration'),controller_source_audit=audit,
        native_execution_not_started_by_module=True,inference_provenance=policy.provenance,
        actor_input_size=3844,critic_input_size=371,action_size=24,student_privileged_inputs_added=False,
        new_independent_initial_states=0,heldout_generalization_claimed=False,
        reset_reference_scope='actual cold-reset measured state equality;not a PhysX solver/contact-state checkpoint',
        raw_physics_path=str(output/'tracking_physics_480hz.jsonl.gz'),raw_physics_sha256=file_sha256(output/'tracking_physics_480hz.jsonl.gz'))
    _new_json(output/'tracking_full_episode_report.json',report)
    if isinstance(caught,(KeyboardInterrupt,SystemExit)):raise caught
    if error:raise RuntimeError(error)
    return report
