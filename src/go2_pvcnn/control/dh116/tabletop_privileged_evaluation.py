"""Frozen-policy cold-trial evaluation using the original reset-buffer kernels.

Load the policy once against its unchanged base contract before calling here.
Only explicitly authored reset buffers change; no checkpoint identity, physics,
controller, material, actuator limit or success tolerance is projected away.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .tabletop_evaluation_trials import GRASP_STYLES,validate_manifest,_digest
from .tabletop_stereo_contract import ROOT,file_sha256,require_sha,write_json_new
from .tabletop_stereo_observation import StereoHistoryBatch
from .tabletop_vision_evaluation_contract import evaluation_physics_sha256
from .tabletop_vision_evaluation_reset import (
    TASK_RESET_FIELDS,_install_reset_trials,_readback_reset_trials,_validate_coupling,
)

VERSION='dh116-stereo-cold-trial-evaluation-v1'


def _read_manifest(value):
    return json.loads(Path(value).read_text(encoding='utf-8')) if isinstance(value,(str,Path)) else copy.deepcopy(value)


def prepare_cold_trial_batches(manifest,batch_size,*,minimum_per_style=100,schema_only=False):
    if type(batch_size) is not int or batch_size<2 or batch_size%2:raise ValueError('positive even native batch size >=2 required')
    if type(minimum_per_style) is not int or minimum_per_style<1:raise ValueError('positive minimum_per_style required')
    if not schema_only and minimum_per_style<100:raise ValueError('formal evaluation requires at least100 independent trials per style')
    value=validate_manifest(_read_manifest(manifest),min_trials_per_style=minimum_per_style)
    groups={style:[row for row in value['trials'] if row['grasp_style']==style] for style in GRASP_STYLES}
    half=batch_size//2
    if len(groups[GRASP_STYLES[0]])!=len(groups[GRASP_STYLES[1]]) or len(groups[GRASP_STYLES[0]])%half:
        raise ValueError('balanced cohort counts must be divisible by half the batch; no padding clones or dropped trials')
    for row in value['trials']:_validate_coupling(row['physical_state'])
    chunks=[groups[GRASP_STYLES[0]][i:i+half]+groups[GRASP_STYLES[1]][i:i+half]
            for i in range(0,len(groups[GRASP_STYLES[0]]),half)]
    return dict(manifest=value,manifest_sha256=_digest(value),batches=chunks,
        original_split=value['metadata']['split'],minimum_per_style=minimum_per_style)


def _clone(value):
    if isinstance(value,torch.Tensor):return value.detach().clone()
    if isinstance(value,dict):return {k:_clone(v) for k,v in value.items()}
    if isinstance(value,list):return [_clone(v) for v in value]
    return copy.deepcopy(value)


def _json(value):
    if isinstance(value,torch.Tensor):return _json(value.detach().cpu().tolist())
    if isinstance(value,dict):return {k:_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json(v) for v in value]
    if isinstance(value,float) and not torch.isfinite(torch.tensor(value)):return None
    return value


def _staging_reset_state(world,task):
    """Only buffer copies are writable; native handles/names/limits are read-only."""
    shadow=SimpleNamespace(n=world.n,w=world.w,hands=world.hands,limits=world.limits,
        origins=world.origins,initial_offset=world.initial_offset,
        seed_q=[v.clone() for v in world.seed_q],seed_qd=[v.clone() for v in world.seed_qd],
        root_seed=world.root_seed.clone(),seed_wire=world.seed_wire.clone(),seed_velocity=world.seed_velocity.clone())
    other=SimpleNamespace(n=task.n)
    for name in TASK_RESET_FIELDS:setattr(other,name,getattr(task,name).clone())
    if hasattr(task,'_per_env_reset_seed'):other._per_env_reset_seed=_clone(task._per_env_reset_seed)
    return shadow,other


def _save_reset_buffers(world,task):
    return dict(seed_q=[v.clone() for v in world.seed_q],seed_qd=[v.clone() for v in world.seed_qd],
        root_seed=world.root_seed.clone(),seed_wire=world.seed_wire.clone(),seed_velocity=world.seed_velocity.clone(),
        had_task_seed=hasattr(task,'_per_env_reset_seed'),task_seed=_clone(getattr(task,'_per_env_reset_seed',{})))


def _restore_reset_buffers(world,task,saved):
    for key in ('seed_q','seed_qd'):
        for current,original in zip(getattr(world,key),saved[key]):current.copy_(original)
    for key in ('root_seed','seed_wire','seed_velocity'):getattr(world,key).copy_(saved[key])
    if saved['had_task_seed']:task._per_env_reset_seed=_clone(saved['task_seed'])
    elif hasattr(task,'_per_env_reset_seed'):delattr(task,'_per_env_reset_seed')


def _policy_digest(policy):
    if not hasattr(policy,'state_dict'):return None
    digest=hashlib.sha256()
    for key,value in sorted(policy.state_dict().items()):
        if not isinstance(value,torch.Tensor):raise ValueError('frozen policy must have tensor state_dict')
        tensor=value.detach().cpu().contiguous()
        digest.update((key+str(tensor.dtype)+str(tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _native_bridge(bridge):
    from .tabletop_stereo_native import NativeTensorBridge
    from .tabletop_tensor_world import TensorWorld
    return (type(bridge) is NativeTensorBridge and type(bridge.device_world) is TensorWorld
            and type(bridge.world.sim).__module__.startswith('isaaclab.'))


def _clock(bridge):
    return (int(getattr(bridge.device_world,'_physics_steps',getattr(bridge,'_physics_steps',0))),
            float(bridge.world.sim.current_time),int(bridge.policy_tick))


@torch.no_grad()
def evaluate_cold_trials(bridge,policy,manifest,output,*,max_control_steps=1200,minimum_per_style=100,
                         owner='teacher',checkpoint_path=None,observation_callback=None,on_batch_reset=None,
                         step_callback=None,schema_only=False):
    if owner not in ('teacher','student'):raise ValueError('independent Teacher or Student owner required')
    if type(schema_only) is not bool or type(max_control_steps) is not int or max_control_steps<1:
        raise ValueError('explicit schema mode and positive bounded control budget required')
    if owner=='student' and observation_callback is None:raise ValueError('Student sensor-history callback required')
    if owner=='student' and step_callback is None:raise ValueError('Student sensor history execution requires explicit step_callback to update sensor caches')
    plan=prepare_cold_trial_batches(manifest,bridge.n,minimum_per_style=minimum_per_style,schema_only=schema_only)
    native=False if schema_only else _native_bridge(bridge)
    if not schema_only and not native:raise ValueError('formal evaluation requires actual NativeTensorBridge; CPU fixtures must be schema_only')
    context=copy.deepcopy(bridge.contract);context_sha=_digest(context)
    physics_sha=evaluation_physics_sha256(context)
    if plan['manifest']['metadata']['physics_sha256']!=physics_sha:
        raise ValueError('cold manifest physical-context fingerprint differs from unchanged base contract')
    source=file_sha256(checkpoint_path) if checkpoint_path is not None else getattr(bridge,'checkpoint_sha256',None)
    require_sha(source,'frozen policy checkpoint/source SHA256')
    if plan['manifest']['metadata']['source_sha256']!=source:raise ValueError('manifest source checkpoint SHA mismatch')
    inputs={str(Path(p).resolve()):file_sha256(p) for p in (manifest,checkpoint_path) if isinstance(p,(str,Path))}
    output=Path(output)
    if output.exists() or output.is_symlink():raise FileExistsError(output)
    # Validate EVERY trial against actual articulation names and native limits
    # before touching the first real buffer or advancing the first policy step.
    stage,stage_task=_staging_reset_state(bridge.device_world,bridge.task)
    for chunk in plan['batches']:_install_reset_trials(stage,stage_task,chunk)
    output.mkdir(parents=True,exist_ok=False)
    write_json_new(output/'frozen_manifest.json',plan['manifest'])
    write_json_new(output/'base_contract_binding.json',dict(base_contract=context,base_contract_sha256=context_sha,
        evaluation_physics_sha256=physics_sha,checkpoint_sha256=source,manifest_sha256=plan['manifest_sha256'],
        immutable_inputs=inputs,checkpoint_contract_rewritten=False))
    saved=_save_reset_buffers(bridge.device_world,bridge.task)
    tensor_before=_policy_digest(policy);was_training=getattr(policy,'training',None)
    if hasattr(policy,'eval'):policy.eval()
    rows=[];uids=set();batch_records=[]
    if hasattr(bridge,'set_execution_context'):bridge.set_execution_context(owner=owner,checkpoint_sha256=source)
    try:
        with (output/'control_intervals.jsonl').open('x',encoding='utf-8') as intervals:
            for batch_index,chunk in enumerate(plan['batches']):
                batch_output=output/f'batch_{batch_index:05d}';batch_output.mkdir(exist_ok=False)
                _install_reset_trials(bridge.device_world,bridge.task,chunk)
                bridge.bind_cold_trials(copy.deepcopy(chunk))
                bridge.reset()
                readback=_readback_reset_trials(bridge.device_world,chunk)
                current=[bridge.identities.uid(i) for i in range(bridge.n)]
                if len(set(current))!=bridge.n or set(current)&uids:raise RuntimeError('cold batch reused a global episode UID')
                if any(uid in bridge._terminal_rows for uid in current):raise RuntimeError('new cold first episode already has terminal evidence')
                uids.update(current)
                audit=dict(outer_manifest_sha256=plan['manifest_sha256'],original_split=plan['original_split'],
                    trial_ids=[row['trial_id'] for row in chunk],trial_indices=[next(i for i,r in enumerate(plan['manifest']['trials']) if r['trial_id']==row['trial_id']) for row in chunk],
                    first_episode_uids=current,readback=readback,source_scope='native' if native else 'cpu_fixture',
                    native_validation_complete=False,writer='existing__install_reset_trials',reader='existing__readback_reset_trials')
                write_json_new(batch_output/'reset_readback.json',audit)
                batch_records.append(dict(batch_index=batch_index,trial_ids=audit['trial_ids'],readback_path=str(batch_output/'reset_readback.json'),
                    readback_sha256=file_sha256(batch_output/'reset_readback.json')))
                if on_batch_reset is not None:
                    before=_clock(bridge);on_batch_reset(bridge)
                    if _clock(bridge)!=before:raise RuntimeError('sensor history reset callback advanced physics or policy time')
                complete=torch.zeros(bridge.n,device=bridge.device,dtype=torch.bool)
                first=[None]*bridge.n;steps=[0]*bridge.n
                for _ in range(max_control_steps):
                    pre_inference=_clock(bridge)
                    if owner=='teacher':observation=bridge.critic.detach().clone()
                    else:
                        observation=observation_callback()
                        if _clock(bridge)!=pre_inference:
                            raise RuntimeError('Student observation callback advanced the physical or policy clock')
                        if type(observation) is not StereoHistoryBatch:raise TypeError('Student callback must return sensor StereoHistoryBatch only')
                        observation.validate()
                    try:device=next(policy.parameters()).device
                    except (AttributeError,StopIteration):device=bridge.device
                    observation=observation.to(device)
                    action=policy.deterministic(observation) if hasattr(policy,'deterministic') else policy(observation)
                    if (not isinstance(action,torch.Tensor) or action.shape!=(bridge.n,24) or not action.is_floating_point()
                            or not bool(torch.isfinite(action).all()) or bool((action.abs()>1).any())):
                        raise ValueError('frozen policy must return finite bounded canonical24 actions')
                    action=action.to(bridge.device).clone();action[complete]=0.
                    action_owners=['diagnostic_zero' if bool(complete[i]) else owner for i in range(bridge.n)]
                    if hasattr(bridge,'set_action_owners'):bridge.set_action_owners(action_owners)
                    before=_clock(bridge)
                    if owner=='student' and before!=pre_inference:
                        raise RuntimeError('Student inference or action ownership advanced the physical or policy clock')
                    # Keep the Actor output separate from callback-owned storage.
                    result=(bridge.step if step_callback is None else step_callback)(action.clone())
                    from .tabletop_stereo_teacher_collector import StepResult
                    if owner=='student' and type(result) is not StepResult:
                        raise TypeError('Student execution requires StepResult with exact issued action evidence')
                    if type(result) is StepResult:
                        if (not isinstance(result.issued_actions,torch.Tensor) or result.issued_actions.dtype!=action.dtype
                                or not torch.equal(result.issued_actions.to(action.device),action)):
                            raise ValueError('Student execution changed the frozen actor issued action')
                        reward,done=result.reward,result.done
                    elif isinstance(result,tuple) and len(result)==4:_,reward,done,_=result
                    else:raise TypeError('execution must return unchanged native tuple or sensor StepResult')
                    if (done.shape!=(bridge.n,) or done.dtype!=torch.bool or reward.shape!=(bridge.n,)
                            or not bool(torch.isfinite(reward).all())):raise ValueError('invalid native reward/terminal shape or finite state')
                    for i in range(bridge.n):
                        if bool(complete[i]):continue
                        steps[i]+=1
                        terminal=bridge._terminal_rows.get(current[i])
                        if bool(done[i]) and terminal is None:raise RuntimeError('first terminal lacks preserved pre-reset 480Hz evidence')
                        if terminal is not None:
                            first[i]=copy.deepcopy(terminal);complete[i]=True
                    intervals.write(json.dumps(_json(dict(batch=batch_index,trial_ids=audit['trial_ids'],first_episode_uids=current,
                        pre_clock=before,post_clock=_clock(bridge),issued_actions=action,action_owners=action_owners,
                        first_complete=complete,source_scope='native' if native else 'cpu_fixture')),allow_nan=False)+'\n');intervals.flush()
                    if bool(complete.all()):break
                for i,trial in enumerate(chunk):
                    actual=readback['actual_states'][i]
                    rows.append(dict(trial_id=trial['trial_id'],grasp_style=trial['grasp_style'],first_episode_uid=current[i],
                        authored_state_sha256=trial['state_sha256'],actual_readback_state_sha256=actual['readback_state_sha256'],
                        status='completed' if first[i] is not None else 'incomplete',control_steps=steps[i],
                        first_terminal=_json(first[i]),native_success=bool(first[i].get('native_success')) if native and first[i] is not None else None))
        if _policy_digest(policy)!=tensor_before:raise RuntimeError('frozen evaluation policy tensor state changed')
        if _digest(bridge.contract)!=context_sha:raise RuntimeError('base physical/controller contract changed during evaluation')
        if any(file_sha256(path)!=sha for path,sha in inputs.items()):raise RuntimeError('immutable manifest/checkpoint input changed')
    finally:
        _restore_reset_buffers(bridge.device_world,bridge.task,saved)
        if was_training is not None:policy.train(was_training)
        if hasattr(bridge,'_raw'):bridge._raw.flush()
    completed=sum(row['status']=='completed' for row in rows)
    report=dict(version=VERSION,status='bounded_cold_trial_evaluation',owner=owner,deterministic=True,
        batch_size=bridge.n,batch_count=len(plan['batches']),minimum_per_style=minimum_per_style,
        manifest_sha256=plan['manifest_sha256'],physics_sha256=physics_sha,base_contract_sha256=context_sha,
        checkpoint_sha256=source,checkpoint_contract_rewritten=False,policy_tensor_sha256=tensor_before,
        expected_trials=len(rows),completed_trials=completed,incomplete_trials=len(rows)-completed,trials=rows,batches=batch_records,
        native_execution_performed=native,native_successes=sum(row['native_success'] is True for row in rows) if native else None,
        native_qualified=False,force_qualification=None,support_qualification=None,teacher_executed=owner=='teacher',
        critic_deployed=False,reset_buffers_restored=True,physics_hz=480,control_hz=60,
        raw_480hz_source=str(Path(bridge.output)/'native_physics.jsonl') if hasattr(bridge,'output') else None,
        source_scope='native_runtime_capture_unqualified' if native else 'cpu_fixture',schema_only=schema_only,
        limitation='cold authored reset/readback and first terminals; separate raw qualification gate remains mandatory')
    if native:
        identity=bridge.task_identity
        report.update(task_identity=identity,task_kind=identity['task_kind'],
                      task_identity_sha256=identity['task_identity_sha256'])
    write_json_new(output/'trial_evaluation.json',report)
    return report


def _write_manifest_and_provenance(output,value,provenance):
    output=Path(output);sidecar=output.with_suffix('.provenance.json')
    if output.exists() or sidecar.exists() or output.is_symlink() or sidecar.is_symlink():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True);created=[]
    try:
        write_json_new(output,value);created.append(output)
        provenance=copy.deepcopy(provenance);provenance['manifest_file_sha256']=file_sha256(output)
        write_json_new(sidecar,provenance);created.append(sidecar)
    except BaseException:
        for path in reversed(created):path.unlink(missing_ok=True)
        raise
    return dict(manifest_path=str(output),provenance_path=str(sidecar),manifest_sha256=file_sha256(output),native_qualified=False)


def derive_policy_trial_manifest(source_manifest,source_checkpoint,target_checkpoint,output,*,minimum_per_style=100,schema_only=False):
    """New policy-source binding; preserve every declared physical trial byte-value.

    This is an explicit derivation, not a checkpoint conversion or qualification.
    Both policy artifacts and the source manifest are SHA-bound in its sidecar.
    The actual evaluated policy must still pass its original strict loader.
    """
    source_path=Path(source_manifest);old_path=Path(source_checkpoint);new_path=Path(target_checkpoint)
    identities={str(path.resolve()):file_sha256(path) for path in (source_path,old_path,new_path)}
    if Path(output).resolve() in {path.resolve() for path in (source_path,old_path,new_path)}:
        raise ValueError('derived output cannot overwrite source manifest/checkpoints')
    if not schema_only and minimum_per_style<100:raise ValueError('formal cohorts require100 per style')
    value=validate_manifest(_read_manifest(source_path),min_trials_per_style=minimum_per_style)
    if value['metadata']['source_sha256']!=file_sha256(old_path):raise ValueError('source manifest policy artifact SHA mismatch')
    derived=copy.deepcopy(value);derived['metadata']['source_sha256']=file_sha256(new_path)
    validate_manifest(derived,min_trials_per_style=minimum_per_style)
    if derived['trials']!=value['trials']:raise RuntimeError('policy derivation changed physical trial states')
    result=_write_manifest_and_provenance(output,derived,dict(version='dh116-cold-trial-policy-derivation-v1',
        operation='policy_source_binding_only',immutable_inputs=identities,source_manifest_sha256=_digest(value),
        derived_manifest_sha256=_digest(derived),source_policy_sha256=file_sha256(old_path),
        target_policy_sha256=file_sha256(new_path),unchanged_physical_cohort_sha256=_digest(value['trials']),
        physics_sha256=value['metadata']['physics_sha256'],all_trial_ids_and_state_sha256_preserved=True,
        checkpoint_contract_rewritten=False,native_execution_performed=False,native_qualified=False))
    if any(file_sha256(path)!=sha for path,sha in identities.items()):raise RuntimeError('immutable input changed during derivation')
    return result


def inspect_trained_teacher_checkpoint(path, checkpoint_sha256, config, ppo, *, contract=None, task_identity=None):
    """Authenticate the frozen artifact and distinguish BC initialization from PPO."""
    from dataclasses import asdict
    from .tabletop_privileged_checkpoint import VERSION as CHECKPOINT_VERSION, _finite_tree, _validate_adam
    from .tabletop_privileged_cli import training_identity
    from .tabletop_privileged_teacher import PrivilegedTeacher

    require_sha(checkpoint_sha256, 'reviewed R0 checkpoint SHA256')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != checkpoint_sha256:
        raise ValueError('frozen Teacher checkpoint SHA256 differs from the reviewed artifact')
    payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or payload.get('version') != CHECKPOINT_VERSION:
        raise ValueError('versioned privileged Teacher PPO checkpoint required')
    counters = payload.get('counters')
    if (not isinstance(counters, dict) or type(counters.get('updates')) is not int or counters['updates'] < 1
            or type(payload.get('policy_version')) is not int or payload['policy_version'] < counters['updates']):
        raise ValueError('frozen evaluation requires accepted PPO updates after any Actor BC')
    environments, rollout = payload.get('num_envs'), payload.get('rollout_steps')
    if (type(environments) is not int or environments < 1 or type(rollout) is not int or rollout < 1
            or type(counters.get('environment_steps')) is not int
            or counters['environment_steps'] < environments * rollout * counters['updates']):
        raise ValueError('frozen PPO checkpoint has incomplete owned rollout counters')
    identity = payload.get('identity', {})
    if (not isinstance(identity, dict) or identity.get('ppo_config') != asdict(ppo)
            or ppo.target_kl is None or config.get('reward_profile') != 'teacher_r0_existing'
            or identity.get('observation_size') != 371 or identity.get('action_size') != 24):
        raise ValueError('frozen checkpoint must match the KL-guarded 371D/24D R0 configuration')
    if contract is not None and identity != training_identity(ppo, contract, task_identity):
        raise ValueError('frozen checkpoint does not match the actual native R0 contract')
    optimizer, policy = payload.get('optimizer'), payload.get('policy')
    if (not isinstance(optimizer, dict) or not isinstance(optimizer.get('state'), dict)
            or not isinstance(policy, dict)):
        raise ValueError('trained PPO checkpoint requires complete Adam and policy state')
    with torch.random.fork_rng(devices=[]):
        model = PrivilegedTeacher(**config['model'])
    _finite_tree(policy)
    _finite_tree(optimizer)
    _validate_adam(optimizer, model, ppo, counters['updates'])
    try:
        model.load_state_dict(policy, strict=True)
    except (TypeError, RuntimeError) as error:
        raise ValueError('frozen checkpoint policy topology differs from Teacher config') from error
    return dict(checkpoint_sha256=checkpoint_sha256, ppo_updates=counters['updates'],
        ppo_environment_steps=counters['environment_steps'], policy_version=payload['policy_version'],
        ppo_training_environments=environments, ppo_rollout_steps=rollout,
        physics_sha256=evaluation_physics_sha256(contract) if contract is not None else None,
        native_grasp_qualification=False)


def freeze_teacher_cold_manifest(checkpoint,base_contract,primary_initialization,secondary_initialization,
                                 output,*,seed,split='test',count_per_style=100,camera_sha256='0'*64,
                                 wrist_joint_names=('wrist_x','wrist_y','wrist_z','wrist_roll','wrist_pitch','wrist_yaw')):
    """Freeze authored trials from real init files without loading a new scene."""
    from .tabletop_surface_contract import semantic_sha256
    from .tabletop_privileged_checkpoint import VERSION as CHECKPOINT_VERSION
    from .tabletop_adaptive_initialization import adaptive_initialization
    from .tabletop_end_supported_scene import translate_end_support_seed
    from .tabletop_vision_evaluation_reset import template_from_initialization,generate_cold_reset_manifest
    paths=[Path(p).resolve() for p in (checkpoint,primary_initialization,secondary_initialization)]
    if Path(output).resolve() in set(paths):raise ValueError('manifest output cannot overwrite immutable source')
    hashes=[file_sha256(path) for path in paths]
    contract=json.loads(Path(base_contract).read_text(encoding='utf-8')) if isinstance(base_contract,(str,Path)) else copy.deepcopy(base_contract)
    contract=contract.get('inherited_contract',contract)
    payload=torch.load(paths[0],map_location='cpu',weights_only=False)
    if (payload.get('version')!=CHECKPOINT_VERSION
            or payload.get('identity',{}).get('source_sha256',{}).get('native_task_contract')!=semantic_sha256(contract)):
        raise ValueError('Teacher checkpoint must retain the exact original native task contract')
    expected_primary=contract.get('grasp_initialization_sha256',contract.get('close_seed_sha256'))
    if hashes[1]!=expected_primary:raise ValueError('primary initialization differs from frozen base contract source')
    initializations=[json.loads(path.read_text(encoding='utf-8')) for path in paths[1:]]
    source_contract=initializations[0].get('source_contract')
    if not isinstance(source_contract,dict) or initializations[1].get('source_contract')!=source_contract:
        raise ValueError('both grasp templates require the same asset/pose/physics source context')
    templates={};scene=contract.get('end_support_scene')
    for style,data in zip(GRASP_STYLES,initializations):
        layout=dict(hand_roots={side:list(pose) for side,pose in zip(('left','right'),data['roots'])},
            active_joints={},wire_center=data['wire_pose'][:3])
        layout,reset=adaptive_initialization(data,layout,source_contract,object_radius_m=contract.get('object_radius_m',.008))
        if scene is not None:
            reset=translate_end_support_seed(reset,scene)
            for pose in layout['hand_roots'].values():pose[2]+=scene['dz_m']
        templates[style]=template_from_initialization(layout,reset,wrist_joint_names=wrist_joint_names)
    value=generate_cold_reset_manifest(templates,seed=seed,split=split,count_per_style=count_per_style,
        provenance=dict(source_sha256=hashes[0],physics_sha256=evaluation_physics_sha256(contract),camera_sha256=require_sha(camera_sha256)))
    provenance=dict(version='dh116-privileged-cold-trial-generator-v1',immutable_inputs=[dict(path=str(p),sha256=s) for p,s in zip(paths,hashes)],
        base_contract=contract,base_contract_sha256=semantic_sha256(contract),template_source_contract=source_contract,
        authored_cohort_sha256=_digest(value['trials']),source_sha256=hashes[0],native_execution_performed=False,native_qualified=False,
        checkpoint_identity_validation='strict_original_policy_loader_required_before_execution')
    result=_write_manifest_and_provenance(output,value,provenance)
    if [file_sha256(path) for path in paths]!=hashes:raise RuntimeError('immutable source changed during manifest freezing')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/stereo_trial_evaluation_v1.json')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--trained-checkpoint-sha256',
        help='require an authenticated Teacher checkpoint with accepted R0 PPO updates')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--owner',choices=('teacher','student'),default='teacher')
    parser.add_argument('--batch-size','--num-envs',dest='batch_size',type=int)
    parser.add_argument('--max-control-steps',type=int)
    parser.add_argument('--minimum-per-style',type=int)
    parser.add_argument('--seed',type=int,default=116)
    parser.add_argument('--teacher-config',type=Path,default=ROOT/'configs/dh116_training/privileged_teacher_ppo_v1.json')
    parser.add_argument('--camera-config',type=Path,default=ROOT/'configs/dh116_training/stereo_camera_640x480_v2.json')
    parser.add_argument('--native-task-config',type=Path,default=ROOT/'configs/dh116_training/stereo_native_task_v1.json')
    parser.add_argument('--enable-experiment',action='store_true')
    parser.add_argument('--execute-native',action='store_true')
    parser.add_argument('--check-config',action='store_true')
    args=parser.parse_args(argv)
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if config.get('version')!=VERSION or config.get('physics_hz')!=480 or config.get('control_hz')!=60 or config.get('deterministic') is not True:
        raise ValueError('frozen native480Hz/control60Hz deterministic trial configuration required')
    minimum=config['minimum_per_style'] if args.minimum_per_style is None else args.minimum_per_style
    batch=config['batch_size'] if args.batch_size is None else args.batch_size
    budget=config['max_control_steps'] if args.max_control_steps is None else args.max_control_steps
    if minimum<100 or batch not in config['allowed_batch_sizes'] or type(budget) is not int or budget<1:
        raise ValueError('at least100 trials/style, explicit small resource batch and positive step budget required')
    checked=dict(version=VERSION,status='configuration_checked',owner=args.owner,batch_size=batch,
        minimum_per_style=minimum,max_control_steps=budget,native_execution_performed=False,native_qualified=False)
    teacher_config = ppo = None
    if args.trained_checkpoint_sha256 is not None:
        if args.owner != 'teacher' or args.checkpoint is None or args.manifest is None:
            raise ValueError('trained checkpoint verification requires Teacher owner, checkpoint and held-out manifest')
        from .tabletop_privileged_cli import load_config
        teacher_config, ppo = load_config(args.teacher_config)
        checked['frozen_ppo_checkpoint'] = inspect_trained_teacher_checkpoint(args.checkpoint,
            args.trained_checkpoint_sha256, teacher_config, ppo)
    if args.manifest is not None:
        plan=prepare_cold_trial_batches(args.manifest,batch,minimum_per_style=minimum)
        checked.update(manifest_sha256=plan['manifest_sha256'],trials=len(plan['manifest']['trials']),batch_count=len(plan['batches']))
        metadata = plan['manifest']['metadata']
        if args.checkpoint is not None and metadata['source_sha256']!=file_sha256(args.checkpoint):
            raise ValueError('evaluation checkpoint differs from explicitly bound manifest policy source')
        if args.trained_checkpoint_sha256 is not None:
            from .tabletop_stereo_calibration import StereoCalibration
            if metadata['split'] != 'test' or metadata['camera_sha256'] != StereoCalibration.from_json(args.camera_config).sha256:
                raise ValueError('frozen Teacher needs a test split bound to the declared evaluation calibration')
    else:checked['manifest_required_for_execution']=True
    if args.check_config or not args.execute_native:
        print(json.dumps(checked,indent=2));return 0
    if not args.enable_experiment:raise ValueError('native execution requires explicit --enable-experiment')
    if any(value is None for value in (args.output,args.checkpoint,args.manifest)):raise ValueError('native evaluation requires new output, checkpoint and complete manifest')
    from .tabletop_stereo_contract import new_run
    from .tabletop_stereo_native import launch_native
    new_run(args.output)
    def callback(bridge):
        if args.owner=='teacher':
            from .tabletop_privileged_cli import load_config,_policy_for_evaluation
            selected_config, selected_ppo = (teacher_config, ppo) if teacher_config is not None else load_config(args.teacher_config)
            frozen = None
            if args.trained_checkpoint_sha256 is not None:
                frozen = inspect_trained_teacher_checkpoint(args.checkpoint, args.trained_checkpoint_sha256,
                    selected_config, selected_ppo, contract=bridge.contract, task_identity=bridge.task_identity)
                write_json_new(args.output / 'frozen_teacher_checkpoint.json', frozen)
            policy,_=_policy_for_evaluation(bridge,selected_config,selected_ppo,args.checkpoint)
            return evaluate_cold_trials(bridge,policy,args.manifest,args.output/'trials',owner='teacher',
                checkpoint_path=args.checkpoint,max_control_steps=budget,minimum_per_style=minimum)
        from .tabletop_stereo_calibration import StereoCalibration
        from .tabletop_stereo_checkpoint import load_deployment_actor
        from .tabletop_stereo_runtime import NativeStereoEnvironment
        calibration=StereoCalibration.from_json(args.camera_config)
        policy=load_deployment_actor(args.checkpoint,expected_calibration_sha256=calibration.sha256).to(bridge.device)
        environment=NativeStereoEnvironment(bridge,args.camera_config,history=policy.network_config['history'],
            owner='student',checkpoint_sha256=file_sha256(args.checkpoint))
        return evaluate_cold_trials(bridge,policy,args.manifest,args.output/'trials',owner='student',
            checkpoint_path=args.checkpoint,max_control_steps=budget,minimum_per_style=minimum,
            observation_callback=lambda:environment.snapshot().sensor_history,on_batch_reset=environment.clear_sensor_history,
            step_callback=environment.step)
    result=launch_native(callback,native_arguments=args.native_task_config,output=args.output/'native',
        num_envs=batch,seed=args.seed,rollout_steps=1,enabled=True)
    print(json.dumps(_json(result),indent=2));return 0
