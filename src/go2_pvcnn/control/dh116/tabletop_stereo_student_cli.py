"""Explicit Student training, pure actor inference and independent actor export.

Training/native imports are lazy; offline inference imports no Teacher/PPO.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import traceback

import torch

from .tabletop_stereo_contract import ROOT, file_sha256, make_contract, new_run, require_opt_in, write_json_new
from .tabletop_stereo_calibration import StereoCalibration

VERSION = 'dh116-stereo-student-cli-v1'
DEFAULT_CAMERA = ROOT/'configs/dh116_training/stereo_camera_640x480_v2.json'
DEFAULT_CONFIG = ROOT/'configs/dh116_training/stereo_student_ppo_accuracy640_v2.json'


def _json(value):
    if isinstance(value, torch.Tensor): return _json(value.detach().cpu().tolist())
    if isinstance(value, dict): return {key:_json(item) for key,item in value.items()}
    if isinstance(value, (tuple,list)): return [_json(item) for item in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def resolve_config(path, camera_path, *, rollout_steps=None, image_processing_batch_size=None, minibatches=None, validate_ppo=True):
    from .tabletop_stereo_student_ppo import normalize_network_config
    from .tabletop_stereo_preprocessing import processing_contract
    config=json.loads(Path(path).read_text(encoding='utf-8'))
    if config.get('version') not in ('dh116-stereo-student-ppo-config-v1','dh116-stereo-student-ppo-config-v2'):
        raise ValueError('independent stereo Student config required; no visual3844 conversion')
    if config.get('actor_input')!='StereoHistoryBatch_only' or config.get('critic_observation_size')!=371 or config.get('action_size')!=24:
        raise ValueError('stereo-only actor /371D critic /24D action configuration required')
    if config.get('reward_profile')!='teacher_r0_existing':
        raise ValueError('Student CLI uses audited native R0 reward only')
    if config.get('auxiliary_bc',{}).get('enabled',False):
        raise ValueError('implicit auxiliary BC is disabled; use qualified actor-package warm start')
    calibration=StereoCalibration.from_json(camera_path)
    config=deepcopy(config)
    config['preprocessor']=processing_contract(camera_path)
    config['network']['image_resolution_wh']=[calibration.width,calibration.height]
    batch=image_processing_batch_size
    if batch is None: batch=config['network'].get('image_processing_batch_size',2)
    config['network']['image_processing_batch_size']=batch
    config['network']=normalize_network_config(config['network'])
    if rollout_steps is not None: config['rollout_steps']=rollout_steps
    if type(config.get('rollout_steps')) is not int or config['rollout_steps']<1:
        raise ValueError('positive rollout_steps required')
    if minibatches is not None: config['ppo']['minibatches']=minibatches
    if validate_ppo:
        from .tabletop_privileged_ppo import PPOConfig
        PPOConfig(**config['ppo'])
    return config,calibration


def _launch_native(callback,**kwargs):
    from .tabletop_stereo_native import launch_native
    return launch_native(callback,**kwargs)


def _environment(bridge,camera_config,*,history,checkpoint_sha256=None):
    from .tabletop_stereo_runtime import NativeStereoEnvironment
    return NativeStereoEnvironment(bridge,camera_config,history=history,owner='student',checkpoint_sha256=checkpoint_sha256)


def _snapshot(state,calibration_sha256):
    state.validate()
    # privileged_valid is eligibility for supervised Student targets, not the
    # validity of the unchanged finite native numeric critic observation.
    if not bool(torch.isfinite(state.privileged).all()):
        raise ValueError('finite unchanged native371 critic observation required')
    if any(row['calibration_sha256']!=calibration_sha256 for row in state.identities):
        raise ValueError('snapshot/runtime calibration SHA mismatch')
    return state


def _feedback(info):
    for key in ('native_feedback_valid','measurement_valid'):
        value=info.get(key)
        if value is not None:
            valid=bool(value.all()) if isinstance(value,torch.Tensor) else value is True
            if not valid: raise ValueError('native physics/feedback invalid; PPO update aborted')


def _execution_context(environment,checkpoint):
    bridge=getattr(environment,'bridge',None)
    if bridge is not None and callable(getattr(bridge,'set_execution_context',None)):
        bridge.set_execution_context(owner='student',checkpoint_sha256=file_sha256(checkpoint))


def _warmstart_processing(actor,target,*,diagnostic_only):
    from .tabletop_stereo_preprocessing import validate_processing_contract
    validate_processing_contract(target)
    binding=actor.deployment_contract.get('preprocessor',{})
    source=binding.get('processing')
    if binding.get('binding')=='bound_raw_stereo':
        validate_processing_contract(source)
        if source==target:
            return dict(scope='same_bound_raw_stereo_processing',
                source_processing_sha256=source['processing_sha256'],target_processing_sha256=target['processing_sha256'])
    if not diagnostic_only:
        raise ValueError('formal warm start requires matching bound raw stereo processing; history-input-only/changed processing is ineligible')
    return dict(scope='diagnostic_rebind_changed_processing' if source is not None else 'diagnostic_rebind_unspecified_history_input',
        source_processing_sha256=source.get('processing_sha256') if isinstance(source,dict) else None,
        target_processing_sha256=target['processing_sha256'],native_qualified=False)


def _runtime_descriptor(environment,descriptor):
    bridge=getattr(environment,'bridge',None)
    setter=getattr(bridge,'set_runtime_descriptor',None)
    if not callable(setter):
        raise ValueError('runtime bridge must freeze the explicit Student driver descriptor before physical evidence')
    setter(descriptor)


def train_student(environment,config,*,output,updates,calibration_sha256,checkpoint=None,
                  actor_package=None,checkpoint_interval=10,diagnostic_only=False):
    from .tabletop_stereo_student_ppo import make_student_trainer,history_to_tree
    from .tabletop_stereo_checkpoint import save_student_checkpoint,load_student_checkpoint,load_deployment_actor
    from .tabletop_privileged_reward import compose_native_reward
    from .tabletop_stereo_teacher_collector import _validate_step
    if any(type(v) is not int or v<1 for v in (updates,checkpoint_interval)):
        raise ValueError('positive update/checkpoint budgets required')
    if checkpoint is not None and actor_package is not None:
        raise ValueError('exact resume and actor-only warm start are distinct initialization sources')
    if diagnostic_only and (updates!=1 or environment.n>2 or config['rollout_steps']>8):
        raise ValueError('diagnostic-only is limited to one update,at most2 environments,and at most8 control steps')
    output=Path(output)
    trainer=make_student_trainer(config,environment.n,device=environment.device)
    contract=make_contract('student',config,calibration_sha256=calibration_sha256)
    initialization='fresh_random_actor_and_critic'
    training_provenance=deepcopy(config.get('provenance',{}))
    from .tabletop_stereo_preprocessing import validate_processing_contract
    target_processing=validate_processing_contract(config.get('preprocessor'))
    processing_binding=dict(scope='actual_camera_processing_bound',target_processing_sha256=target_processing['processing_sha256'])
    if checkpoint is not None:
        load_student_checkpoint(checkpoint,trainer,contract,history_reset='clear')
        training_provenance=deepcopy(trainer.training_provenance)
        references=training_provenance.setdefault('resume_checkpoint_artifacts',[])
        binding=dict(path=str(Path(checkpoint).resolve()),sha256=file_sha256(checkpoint))
        if binding not in references:references.append(binding)
        initialization='exact_optimization_resume;history_clear;physical_world_not_restored'
    elif actor_package is not None:
        actor=load_deployment_actor(actor_package,expected_calibration_sha256=calibration_sha256)
        for key in ('embed_dim','history','num_heads','num_layers','dropout'):
            if actor.network_config[key]!=trainer.model.network_config[key]:
                raise ValueError('actor-package warm start network semantics mismatch: '+key)
        if actor.deployment_contract['state_schema_sha256']!=contract['state_schema_sha256']:
            raise ValueError('actor-package warm start current371 schema SHA mismatch')
        source_size=actor.network_config.get('image_resolution_wh')
        if source_size is not None and source_size!=trainer.model.network_config.get('image_resolution_wh'):
            raise ValueError('actor-package image resolution differs from selected camera')
        processing_binding=_warmstart_processing(actor,target_processing,diagnostic_only=diagnostic_only)
        training_provenance.update(input_actor_package_artifact=deepcopy(actor.deployment_package_artifact),
            input_actor_training_provenance=deepcopy(actor.deployment_contract['training_provenance']))
        trainer.model.actor.load_state_dict(actor.state_dict(),strict=True)
        trainer.project_noise()
        initialization='actor_package_warm_start;fresh_critic_and_PPO_Adam;not_exact_resume'
    source_paths=training_provenance.setdefault('raw_episode_source_paths',[])
    bridge_output=getattr(getattr(environment,'bridge',None),'output',None)
    if bridge_output is not None:
        raw_path=str((Path(bridge_output)/'native_physics.jsonl').resolve())
        if raw_path not in source_paths:source_paths.append(raw_path)
    training_provenance.update(source_inventory_complete=False,
        source_inventory_scope='registration_after_native_close_required;not_native_or_heldout_qualification')
    training_provenance['runtime_source_artifacts'] = [
        dict(path=str(Path(__file__).with_name(name).resolve()),
             sha256=file_sha256(Path(__file__).with_name(name)))
        for name in ('tabletop_stereo_student_cli.py', 'tabletop_stereo_native.py',
                     'tabletop_stereo_runtime.py', 'tabletop_privileged_reward.py')]
    trainer.training_provenance=training_provenance
    write_json_new(output/'resolved_config.json',config)
    write_json_new(output/'student_contract.json',contract)
    write_json_new(output/'initialization.json',dict(initialization=initialization,
        source_sha256=file_sha256(checkpoint or actor_package) if checkpoint or actor_package else None,
        diagnostic_only=diagnostic_only,processing_binding=processing_binding,native_qualified=False))
    write_json_new(output/'training_source_registration_seed.json',dict(
        version='dh116-training-source-registration-seed-v1',training_provenance=training_provenance,
        raw_artifact_hashing='only_after_native_producer_close',complete_source_inventory=False,native_qualified=False))
    def save():
        path=output/f'student_update_{trainer.counters["updates"]:06d}.pt'
        save_student_checkpoint(path,trainer,contract,network_config=config['network'],history_reset='clear')
        _execution_context(environment,path)
        return path
    latest=save()
    _runtime_descriptor(environment,dict(execution_mode='student_PPO_training',
        teacher_execution_enabled=False,mpc_enabled=False,critic_deployed=True,critic_training=True,
        privileged_actor_inputs=False,actor_input_source='stereo_RGB_and_proprio_only',
        depth_source='computed_from_synchronized_left_right_RGB',model_checkpoint_sha256=None,
        initialization_checkpoint_sha256=file_sha256(latest),
        model_identity_scope='mutable_training_policy;not_frozen_evaluation',
        processing_sha256=target_processing['processing_sha256'],native_qualified=False))
    state=_snapshot(environment.snapshot(),calibration_sha256)
    started=time.monotonic()
    scope=getattr(environment,'evidence_scope','unqualified')
    with (output/'control_transitions.jsonl').open('x',encoding='utf-8') as controls,(output/'iterations.jsonl').open('x',encoding='utf-8') as iterations:
        for update_index in range(updates):
            terms={}
            for _ in range(config['rollout_steps']):
                actor_tree=history_to_tree(state.sensor_history)
                sample=trainer.collect_step(actor_tree,state.privileged)
                result=environment.step(sample['actions'])
                if not torch.equal(result.issued_actions.to(sample['actions']),sample['actions']):
                    raise ValueError('issued normalized actions differ from sampled Student actions')
                _validate_step(state,result,sample['actions'],1/60)
                _feedback(result.info)
                next_state=_snapshot(result.next_state,calibration_sha256)
                source_terms=result.info.get('reward_terms',result.info.get('terms'))
                if not isinstance(source_terms,dict):raise ValueError('actual native reward_terms required')
                native_terms={name:value.to(result.reward) for name,value in source_terms.items()}
                reward,trace=compose_native_reward(result.reward,native_terms)
                with torch.no_grad(): next_value=trainer.model.value(next_state.privileged)
                trainer.record_step(reward,result.done,next_value,executed_actions=result.issued_actions,
                    collection_token=sample['collection_token'])
                for name,item in trace['terms'].items(): terms[name]=terms.get(name,0.)+float(item['total_contribution'])
                row=dict(policy_version=sample['policy_version'],collection_token=sample['collection_token'],
                    identities=state.identities,actions24=sample['actions'],latent_actions=sample['latent_actions'],
                    old_log_prob=sample['log_prob'],old_latent_mean=sample['latent_mean'],old_latent_std=sample['latent_std'],
                    reward=reward,done=result.done,actually_issued_actions=result.issued_actions,
                    applied_targets=result.applied_targets,supervision_valid_fraction=state.privileged_valid.float().mean(),
                    evidence_scope=scope,native_qualified=False,owner='student')
                controls.write(json.dumps(_json(row),allow_nan=False)+'\n');controls.flush()
                state=next_state
            metrics=trainer.update()
            row=dict(**metrics,update=trainer.counters['updates'],reward_term_totals=terms,
                elapsed_wall_s=time.monotonic()-started,evidence_scope=scope,native_qualified=False,
                qualification='training_updates_do_not_qualify_frozen_Student_success')
            iterations.write(json.dumps(row,allow_nan=False)+'\n');iterations.flush()
            if trainer.counters['updates']%checkpoint_interval==0 or update_index==updates-1: latest=save()
    return dict(version=VERSION,status='diagnostic_updates_completed' if diagnostic_only else 'student_updates_completed',
        updates=trainer.counters['updates'],checkpoint=str(latest),checkpoint_sha256=file_sha256(latest),
        evidence_scope=scope,diagnostic_only=diagnostic_only,native_qualified=False,
        teacher_or_mpc_executed=False,critic_deployment=False)


@torch.no_grad()
def infer_actor(package,history,calibration_sha256=None):
    from .tabletop_stereo_checkpoint import load_deployment_actor
    from .tabletop_stereo_observation import StereoHistoryBatch
    from .tabletop_stereo_student_ppo import history_to_tree
    actor=load_deployment_actor(package,expected_calibration_sha256=calibration_sha256)
    return actor.deterministic(StereoHistoryBatch(**history_to_tree(history))).detach().cpu()


@torch.no_grad()
def run_student(environment,actor_package,*,output,steps,calibration_sha256):
    from .tabletop_stereo_checkpoint import load_deployment_actor
    from .tabletop_stereo_preprocessing import require_raw_processing
    from .tabletop_stereo_teacher_collector import _validate_step
    if type(steps) is not int or steps<1:raise ValueError('positive bounded inference step budget required')
    actor=load_deployment_actor(actor_package,expected_calibration_sha256=calibration_sha256).to(environment.device)
    processing=require_raw_processing(actor.deployment_contract,environment.camera_config)
    _execution_context(environment,actor_package)
    package_sha=file_sha256(actor_package)
    _runtime_descriptor(environment,dict(execution_mode='frozen_student_actor_evaluation',
        teacher_execution_enabled=False,mpc_enabled=False,critic_deployed=False,critic_training=False,
        privileged_actor_inputs=False,actor_input_source='stereo_RGB_and_proprio_only',
        depth_source='computed_from_synchronized_left_right_RGB',model_checkpoint_sha256=package_sha,
        actor_package_sha256=package_sha,actor_package_artifact=dict(path=str(Path(actor_package).resolve()),sha256=package_sha),
        training_contract_sha256=actor.deployment_contract['training_contract_sha256'],
        processing_sha256=processing['processing_sha256'],native_qualified=False))
    state=_snapshot(environment.snapshot(),calibration_sha256)
    complete=torch.zeros(environment.n,dtype=torch.bool,device=environment.device)
    with (Path(output)/'actor_execution.jsonl').open('x',encoding='utf-8') as log:
        for tick in range(steps):
            action=actor.deterministic(state.sensor_history)
            result=environment.step(action)
            if not torch.equal(result.issued_actions.to(action),action): raise ValueError('issued action differs from pure Student actor')
            _validate_step(state,result,action,1/60)
            _feedback(result.info)
            log.write(json.dumps(_json(dict(identities=state.identities,actions24=action,
                issued_actions=result.issued_actions,done=result.done,owner='student',
                actor_input_source='stereo_RGB_and_proprio_only',critic_executed=False,
                teacher_executed=False,native_qualified=False)),allow_nan=False)+'\n');log.flush()
            complete|=result.done
            state=_snapshot(result.next_state,calibration_sha256)
            if bool(complete.all()):break
    return dict(status='bounded_student_actor_execution',steps=tick+1,first_episodes_complete=int(complete.sum()),
        evidence_scope=getattr(environment,'evidence_scope','unqualified'),native_qualified=False,
        teacher_or_mpc_executed=False,critic_deployed=False,actor_package_sha256=package_sha,
        checkpoint_sha256=package_sha,processing_sha256=processing['processing_sha256'])


def _formal_admission(args,calibration):
    from .tabletop_stereo_promotion import require_teacher_qualification,verify_stereo_qualification
    if args.teacher_proof is None or args.teacher_checkpoint is None or args.stereo_proof is None:
        raise ValueError('formal Student PPO requires Teacher proof + actual Teacher checkpoint + stereo qualification proof')
    teacher=require_teacher_qualification(args.teacher_proof,file_sha256(args.teacher_checkpoint))
    stereo=verify_stereo_qualification(args.stereo_proof)
    if stereo['calibration_sha256']!=calibration.sha256:
        raise ValueError('formal stereo qualification calibration differs from selected camera')
    if args.checkpoint is None and args.actor_package is None:
        raise ValueError('formal Student PPO requires explicit actor-package warm start or exact Student resume')
    return dict(teacher=teacher,stereo=stereo)


def main(mode='train',argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--camera-config',type=Path,default=DEFAULT_CAMERA)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--seed',type=int,default=116)
    parser.add_argument('--num-envs',type=int,default=2)
    parser.add_argument('--rollout-steps',type=int)
    parser.add_argument('--updates',type=int,default=1)
    parser.add_argument('--minibatches',type=int)
    parser.add_argument('--image-processing-batch-size',type=int)
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--checkpoint','--resume',dest='checkpoint',type=Path)
    source.add_argument('--actor-package',type=Path)
    parser.add_argument('--teacher-proof',type=Path)
    parser.add_argument('--teacher-checkpoint',type=Path)
    parser.add_argument('--stereo-proof',type=Path)
    parser.add_argument('--checkpoint-interval',type=int,default=10)
    parser.add_argument('--max-control-steps',type=int,default=1200)
    parser.add_argument('--observation',type=Path)
    parser.add_argument('--calibration-sha256')
    parser.add_argument('--native-task-config',type=Path,default=ROOT/'configs/dh116_training/stereo_native_task_v1.json')
    parser.add_argument('--experiment-profile',default='teacher_r0_existing')
    parser.add_argument('--diagnostic-only',action='store_true')
    parser.add_argument('--enable-experiment',action='store_true')
    parser.add_argument('--execute-native',action='store_true')
    parser.add_argument('--check-config',action='store_true')
    args=parser.parse_args(argv)
    if mode not in ('train','run','export'): raise ValueError('unknown Student CLI mode')
    if mode=='export':
        if args.checkpoint is None or args.output is None: parser.error('--checkpoint and new --output required')
        from .tabletop_stereo_checkpoint import export_actor
        export_actor(args.checkpoint,args.output);return 0
    if mode=='run' and not args.execute_native:
        if args.actor_package is None or args.observation is None or args.output is None:
            parser.error('offline run requires --actor-package --observation and new --output')
        values=torch.load(args.observation,map_location='cpu',weights_only=False)
        actions=infer_actor(args.actor_package,values,calibration_sha256=args.calibration_sha256)
        if args.output.suffix.lower()=='.pt':
            with args.output.open('xb') as out: torch.save(actions,out)
        else: write_json_new(args.output,dict(version='dh116-stereo-actor-actions24-v1',actions24=actions.tolist(),
            actor_package_sha256=file_sha256(args.actor_package),critic_or_teacher_executed=False))
        return 0
    config,calibration=resolve_config(args.config,args.camera_config,rollout_steps=args.rollout_steps,
        image_processing_batch_size=args.image_processing_batch_size,minibatches=args.minibatches,validate_ppo=mode=='train')
    if mode=='run':
        if args.actor_package is None: parser.error('native inference requires --actor-package')
        from .tabletop_stereo_checkpoint import load_deployment_actor
        bound_actor=load_deployment_actor(args.actor_package,expected_calibration_sha256=calibration.sha256)
        from .tabletop_stereo_preprocessing import require_raw_processing
        require_raw_processing(bound_actor.deployment_contract,args.camera_config)
        if args.image_processing_batch_size is not None and args.image_processing_batch_size!=bound_actor.network_config.get('image_processing_batch_size'):
            raise ValueError('native inference uses the bound actor-package image processing batch size')
        config['network']=deepcopy(bound_actor.network_config)
        del bound_actor
    if args.experiment_profile!=config['reward_profile']: parser.error('experiment profile must match explicit native R0 config')
    if any(type(v) is not int or v<1 for v in (args.num_envs,args.updates,args.checkpoint_interval,args.max_control_steps)):
        parser.error('positive resource/update/checkpoint/step budgets required')
    if args.check_config:
        print(json.dumps(dict(mode=mode,validated=True,native_started=False,config=config,
            calibration_sha256=calibration.sha256),indent=2));return 0
    require_opt_in(args.enable_experiment)
    if not args.execute_native or args.output is None: parser.error('explicit --execute-native and new --output required')
    if args.diagnostic_only and (args.num_envs>2 or args.updates!=1 or config['rollout_steps']>8):
        raise ValueError('diagnostic-only requires n<=2,updates=1,rollout_steps<=8; never qualifies formal training')
    if mode=='run' and args.actor_package is None: parser.error('native inference requires --actor-package')
    admission=None if args.diagnostic_only or mode=='run' else _formal_admission(args,calibration)
    torch.manual_seed(args.seed)
    run=new_run(args.output)
    write_json_new(run/'launch.json',dict(version=VERSION,mode=mode,diagnostic_only=args.diagnostic_only,
        seed=args.seed,num_envs=args.num_envs,input_config_sha256=file_sha256(args.config),
        cli_source_sha256=file_sha256(__file__),
        camera_config_sha256=file_sha256(args.camera_config),calibration_sha256=calibration.sha256,
        native_task_config_sha256=file_sha256(args.native_task_config),native_qualified=False,
        formal_admission=_json(admission)))
    def callback(bridge):
        checkpoint_sha=file_sha256(args.actor_package) if mode=='run' else None
        environment=_environment(bridge,args.camera_config,history=config['network']['history'],checkpoint_sha256=checkpoint_sha)
        if mode=='train': return train_student(environment,config,output=run,updates=args.updates,
            calibration_sha256=calibration.sha256,checkpoint=args.checkpoint,actor_package=args.actor_package,
            checkpoint_interval=args.checkpoint_interval,diagnostic_only=args.diagnostic_only)
        return run_student(environment,args.actor_package,output=run,steps=args.max_control_steps,calibration_sha256=calibration.sha256)
    try:
        result=_launch_native(callback,native_arguments=args.native_task_config,output=run/'native',
            num_envs=args.num_envs,seed=args.seed,rollout_steps=config['rollout_steps'],enabled=True)
        write_json_new(run/'completion.json',result)
    except Exception as error:
        write_json_new(run/'failure.json',dict(status='failed',exception_type=type(error).__name__,
            message=str(error),traceback=traceback.format_exc(),native_qualified=False,diagnostic_only=args.diagnostic_only))
        raise
    return 0
