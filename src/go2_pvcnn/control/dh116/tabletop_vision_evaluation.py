"""Bounded deterministic visual evaluation with pre-reset first-terminal data."""
import json
import gzip
import math
from copy import deepcopy
from pathlib import Path
import sys
from copy import copy
import torch

from .tabletop_evaluation_trials import (FirstTerminalAggregator, validate_manifest,
    manifest_sha256, _digest)
from .tabletop_vision_evaluation_contract import (file_sha256, validate_evaluation_output,
    evaluation_physics_sha256, PHYSICS_IDENTITY_VERSION)
from .tabletop_vision_evaluation_reset import install_evaluation_reset, readback_evaluation_reset


def _production_manifest_eligible(manifest, *, min_trials_per_style):
    """Test-only reduced schema minima can never produce native certificates."""
    if type(min_trials_per_style) is not int or min_trials_per_style<100: return False
    try:
        value=validate_manifest(manifest,min_trials_per_style=100)
    except ValueError:
        return False
    return value['metadata']['split'] in ('validation','test')


def _required_torque(payload,index):
    """Keep all twelve native active joints; a partial vector is not a sample."""
    fields=('estimated_torque_nm','effort_limit_nm','torque_valid')
    values=[payload.get(k) for k in fields]
    estimate,limits,mask=[None if v is None else v[index].detach().cpu() for v in values]
    unavailable=any(v is None for v in (estimate,limits,mask))
    complete=(not unavailable and estimate.shape==(12,) and limits.shape==(12,) and mask.shape==(12,)
        and mask.dtype==torch.bool and bool(mask.all())
        and bool(torch.isfinite(estimate).all()) and bool(torch.isfinite(limits).all())
        and bool((limits>0).all()))
    def finite_json(value):
        if isinstance(value,list): return [finite_json(v) for v in value]
        return None if isinstance(value,float) and not math.isfinite(value) else value
    first=next((v for v in (estimate,limits,mask) if v is not None),None)
    diagnostic=dict(original_shape=None if first is None else list(first.shape),
        estimated_torque_nm=None if estimate is None else finite_json(estimate.tolist()),
        effort_limit_nm=None if limits is None else finite_json(limits.tolist()),
        valid_mask=None if mask is None else finite_json(mask.tolist()),
        reason=('valid' if complete else 'required_torque_unavailable' if unavailable
            else 'partial_or_invalid_required_torque'))
    return (estimate.tolist(),limits.tolist(),diagnostic) if complete else (None,None,diagnostic)


def _cpu_payload_snapshot(payload):
    """Transfer each already-cloned physical batch once before row extraction.

    Default blocking copies finish before Python reads any field. Keep tensor
    dtype/shape/values and non-tensor metadata intact; never mutate the source
    packet or change its pre-reset sampling clock and first-terminal ordering.
    """
    return {key: value.detach().cpu() if isinstance(value,torch.Tensor) else value
            for key,value in payload.items()}


class EvaluationTelemetryCollector:
    def __init__(self, manifest, *, checkpoint_sha256, min_trials_per_style=100):
        self.manifest=validate_manifest(manifest,min_trials_per_style=min_trials_per_style)
        self.aggregator=FirstTerminalAggregator(self.manifest,checkpoint_sha256=checkpoint_sha256,
            deterministic=True,min_trials_per_style=min_trials_per_style)
        self._begun=False
        self._raw_sink=None
        self._conditional={r['trial_id']:dict(trial_id=r['trial_id'],observed_duration_s=0.,
            slip_applicable_duration_s=0.,slip_valid_applicable_duration_s=0.,
            slip_inapplicable_duration_s=0.,slip_applicable_missing_samples=0,
            slip_applicability_missing_samples=0,slip_inapplicable_reason_counts={})
            for r in self.manifest['trials']}
        self._raw_sample_count=0

    def begin_trials(self):
        if self._begun: raise RuntimeError('evaluation trials have already begun')
        self._begun=True

    def attach_raw_sink(self,sink):
        if self._begun or self._raw_sink is not None: raise RuntimeError('raw evidence sink must be attached once before trials')
        self._raw_sink=sink

    def summary(self):
        report=self.aggregator.summary()
        rows=list(self._conditional.values())
        report['conditional_telemetry']=dict(per_trial=deepcopy(rows),raw_sample_count=self._raw_sample_count)
        base=report['trial_rows']
        report['conditional_telemetry_complete']=(report['incomplete_count']==0 and all(
            r['observed_duration_s']>0 and not r['slip_applicable_missing_samples']
            and not r['slip_applicability_missing_samples'] for r in rows) and all(
            not r['invalid_measurement_samples'] and all(r[f'{m}_valid_coverage']==1.
                for m in ('lift','hold','contact','torque','action')) for r in base))
        return report

    def __call__(self, payload):
        if not self._begun: return
        payload=_cpu_payload_snapshot(payload)
        def valid(key,i):
            value=payload.get(key)
            return value is not None and bool(value[i].all())
        def scalar(key,mask,i):
            return float(payload[key][i]) if valid(mask,i) else None
        for i,trial in enumerate(self.manifest['trials']):
            if not bool(payload['live'][i]): continue
            sample=dict(dt_s=float(payload['dt_s']),measurement_valid=bool(payload['measurement_valid'][i]),
                lift_m=scalar('lift_m','lift_valid',i),
                qualified_hold_s=scalar('qualified_hold_s','qualified_hold_valid',i),
                slip_m=scalar('slip_m','slip_valid',i),
                bilateral_opposed_contact=(bool(payload['bilateral_opposed_contact'][i])
                    if valid('contact_valid',i) else None),
                action=payload['action'][i].detach().cpu().tolist() if valid('action_valid',i) else None)
            # A valid physical torque sample requires every native active joint.
            # Never hide an invalid element by shortening the vector to eleven.
            torque,limits,torque_diagnostics=_required_torque(payload,i)
            sample['estimated_torque_nm']=torque; sample['effort_limit_nm']=limits
            if torque is None: sample['measurement_valid']=False
            terminal=bool(payload['done'][i]); success=bool(payload['native_success'][i])
            consumed=self.aggregator.observe(i,trial['trial_id'],sample,
                terminal=terminal,native_success=success)
            if not consumed: continue
            supplied=payload.get('slip_applicable')
            applicable=None if supplied is None else bool(supplied[i])
            reasons=payload.get('slip_inapplicable_reason')
            reason=None if reasons is None else reasons[i]
            codes=payload.get('slip_reason_code')
            slip_reason=reason
            if codes is not None:
                names={number:name for name,number in payload.get('slip_reason_codes',{}).items()}
                slip_reason=names.get(int(codes[i]),'unknown')
                reason=slip_reason if applicable is False else None
            anchors=payload.get('prior_anchor_latched')
            prior_anchor=None if anchors is None else bool(anchors[i])
            slip_valid=valid('slip_valid',i)
            row=self._conditional[trial['trial_id']]; dt=sample['dt_s']
            row['observed_duration_s']+=dt
            if applicable is None:
                row['slip_applicability_missing_samples']+=1
            elif applicable:
                row['slip_applicable_duration_s']+=dt
                if slip_valid: row['slip_valid_applicable_duration_s']+=dt
                else: row['slip_applicable_missing_samples']+=1
            else:
                row['slip_inapplicable_duration_s']+=dt
                if reason not in ('no_anchor','no_contact','inactive') or slip_valid:
                    row['slip_applicability_missing_samples']+=1
                else:
                    counts=row['slip_inapplicable_reason_counts']; counts[reason]=counts.get(reason,0)+1
            self._raw_sample_count+=1
            if self._raw_sink is not None:
                self._raw_sink(dict(env_id=i,trial_id=trial['trial_id'],terminal=terminal,
                    native_success=success,sample=sample,slip_applicable=applicable,
                    slip_valid=slip_valid,slip_inapplicable_reason=reason,
                    slip_reason=slip_reason,prior_anchor_latched=prior_anchor,
                    torque_diagnostics=torque_diagnostics))


def prepare_vision_evaluation(device_world, task, env, args):
    """Attach inactive callback before any startup reset, then author final reset."""
    minimum=100
    if getattr(args,'grasp_contact_probe',False):
        from .tabletop_grasp_contact_native import evaluation_minimum
        minimum=evaluation_minimum(args)
    manifest=validate_manifest(json.loads(Path(args.vision_evaluation_manifest).read_text()),
                               min_trials_per_style=minimum)
    collector=EvaluationTelemetryCollector(manifest,
        checkpoint_sha256=file_sha256(args.vision_evaluation_checkpoint),min_trials_per_style=minimum)
    env.enable_evaluation_telemetry(collector)
    return collector


def _json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def _verify_native_physical_adapter(policy, device_world):
    """Check the actual spawned USDs and named SDK order once before act.

    V4 preserves the existing SDK24 chart. Verification adds no per-substep
    action conversion, physics write or force/control callback.
    """
    if not getattr(policy, 'requires_physical_adapter_verification', False):
        return
    from .linear_coupling import ACTIVE
    from .tabletop_bimanual_physical_action_adapter import HandPhysicalKinematics, BimanualPhysicalActionAdapter
    native = device_world.w
    source_hands = native.source['hands']
    if len(source_hands) != 2 or len(device_world.hands) != 2 or len(native.ids) != 2:
        raise ValueError('V4 runtime requires both actual DH116 source hands')
    models, joint_names, source_paths = {}, {}, {}
    for hi, side in enumerate(('left', 'right')):
        hand = device_world.hands[hi]
        names = [hand.joint_names[index] for index in native.ids[hi]]
        if names != list(ACTIVE) or hand.body_names.count('base_link') != 1:
            raise ValueError('V4 runtime SDK active joint order or base link mismatch: ' + side)
        spawn = source_hands[hi].cfg.spawn
        path = getattr(spawn, 'usd_path', None)
        if not path:
            raise ValueError('V4 runtime requires the actual spawned source USD: ' + side)
        source_paths[side] = str(path)
        joint_names[side] = names
        models[side] = HandPhysicalKinematics.from_usd(path, side)
    actual = BimanualPhysicalActionAdapter(models).contract()
    policy.verify_physical_adapter(actual)
    policy.provenance['physical_action_adapter_runtime'] = dict(
        source_assets=source_paths, sdk_active_joint_names=joint_names,
        execution='original_SDK24_identity_mapping', additional_per_substep_hooks=0)


def run_vision_evaluation(world, device_world, task, env, args, critic_contract, *,
                          collector=None, camera=None, min_trials_per_style=100):
    """Use camera.sample/SensorHistory/_observations; no training algorithm.

    A test-injected camera cannot issue a native certificate. The native path
    additionally checks concrete runtime types, actual complete readback and
    independent truth-only geometry before enabling trial collection.
    """
    from .tabletop_vision_student import SensorHistory
    from .tabletop_vision_training import _observations, _contract
    from .tabletop_vision_inference import load_vision_inference
    from .tabletop_vision_evaluation_contract import validate_physical_adapter_evaluation_mode
    physical_adapter = validate_physical_adapter_evaluation_mode(args)
    if getattr(args,'grasp_contact_probe',False):
        from .tabletop_grasp_contact_native import evaluation_minimum
        min_trials_per_style=evaluation_minimum(args)
    output=validate_evaluation_output(args.output,args.vision_evaluation_checkpoint,
        args.vision_evaluation_manifest,precreated=getattr(args,'_vision_new_output',None))
    maximum=getattr(args,'vision_evaluation_max_control_steps',0)
    if type(maximum) is not int or not 1<=maximum<=100000:
        raise ValueError('bounded positive vision evaluation max control steps required')
    manifest=validate_manifest(json.loads(Path(args.vision_evaluation_manifest).read_text()),
        min_trials_per_style=min_trials_per_style)
    if len(manifest['trials'])!=device_world.n: raise ValueError('manifest/environment count mismatch')
    checkpoint_sha=file_sha256(args.vision_evaluation_checkpoint)
    if collector is not None:
        if collector.manifest!=manifest: raise ValueError('evaluation manifest changed after preparation')
        if collector.summary()['checkpoint_sha256']!=checkpoint_sha:
            raise ValueError('evaluation checkpoint changed after preparation')
    contract_args=copy(args)
    if contract_args.rollout_steps is None: contract_args.rollout_steps=24
    current=_contract(contract_args,critic_contract)
    policy=load_vision_inference(args.vision_evaluation_checkpoint,current,device=args.device,
                                 allow_physical_adapter=physical_adapter)
    if policy.checkpoint_sha256!=checkpoint_sha:
        raise ValueError('evaluation checkpoint changed during inference load')
    _verify_native_physical_adapter(policy, device_world)
    source=policy.provenance['source_contract']
    if manifest['metadata']['camera_sha256']!=source['camera_config_sha256']:
        raise ValueError('manifest camera provenance mismatch')
    if manifest['metadata']['physics_sha256']!=evaluation_physics_sha256(source['privileged_critic_contract']):
        raise ValueError('manifest physics provenance mismatch')
    output.mkdir(parents=True,exist_ok=True)
    if collector is None:
        collector=EvaluationTelemetryCollector(manifest,checkpoint_sha256=checkpoint_sha,
            min_trials_per_style=min_trials_per_style)
        env.enable_evaluation_telemetry(collector)
        install_evaluation_reset(device_world,task,manifest,min_trials_per_style=min_trials_per_style)
    native_camera=camera is None
    if native_camera:
        from .tabletop_vision_fixed_camera import FixedVisionCamera
        if args.vision_camera_mode!='fixed' or [args.vision_width,args.vision_height]!=[224,96]:
            raise ValueError('native fixed evaluation requires original calibrated 224x96 cameras')
        camera=FixedVisionCamera(world,device_world,args.depth_camera_config,
            width=args.vision_width,height=args.vision_height,num_points=128)
    # Final reset is after startup/camera construction and before any trial.
    critic=env.reset()
    history=SensorHistory(args.num_envs,device=args.device)
    previous=torch.zeros((args.num_envs,24),device=args.device)
    readback=readback_evaluation_reset(device_world,manifest,min_trials_per_style=min_trials_per_style)
    packet=camera.sample()
    if not bool(packet['image_valid'].all() & packet['object_valid'].all()):
        raise RuntimeError('evaluation reset requires valid visible RGB-D for every case')
    from .tabletop_vision_audit import audit_initial_points
    geometry=audit_initial_points(packet,device_world,
        radius=getattr(args,'solid_cylinder_radius_m',.008),length=.5)
    # Re-read after rendering: a camera warmup may not mutate the physical reset.
    after=readback_evaluation_reset(device_world,manifest,min_trials_per_style=min_trials_per_style)
    if after!=readback: raise RuntimeError('physical reset changed during evaluation camera warmup')
    obs=_observations(packet,device_world,previous,history,critic)
    evidence={}
    for name,filename,value in (
            ('frozen_manifest','evaluation_manifest.json',manifest),
            ('raw_reset_readback','evaluation_reset_readback.json',readback),
            ('initial_camera_geometry','evaluation_initial_camera_geometry.json',geometry)):
        path=output/filename; _json(path,value)
        evidence[name]=dict(path=str(path),sha256=file_sha256(path))
    actor_path=output/'evaluation_initial_visual_actor_inputs.pt'
    with actor_path.open('xb') as stream:
        torch.save(dict(policy=obs['policy'].detach().cpu(),previous_action=previous.detach().cpu(),
            history=history.frames.detach().cpu(),contract=current,
            sensor_packet={k:v.detach().cpu() for k,v in packet.items() if isinstance(v,torch.Tensor)}),stream)
    evidence['initial_visual_actor_inputs']=dict(path=str(actor_path),sha256=file_sha256(actor_path))
    if getattr(args, 'paired_bias_arm', None):
        if not native_camera or min_trials_per_style<100:
            raise ValueError('paired experiment requires native camera and full cohort')
        from .tabletop_vision_paired_bias import run_paired_arm
        return run_paired_arm(args, env, task, device_world, camera, history, obs, policy,
                              readback, manifest, checkpoint_sha, output)
    raw_path=output/'evaluation_conditional_telemetry_samples.jsonl.gz'
    # Millions of physical samples stay on disk, not in the Python heap.
    raw_stream=gzip.open(raw_path,'xt',encoding='utf-8',compresslevel=1)
    raw_stream.write(json.dumps(dict(schema_version='dh116-conditional-telemetry-samples-v1',
        format='gzip_jsonl',manifest_sha256=manifest_sha256(manifest,min_trials_per_style=min_trials_per_style),
        checkpoint_sha256=checkpoint_sha),allow_nan=False)+'\n')
    collector.attach_raw_sink(lambda row:raw_stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n'))
    contact_observer=None
    if getattr(args,'grasp_contact_diagnostics',False):
        if not native_camera:
            raw_stream.close()
            raise ValueError('contact evidence requires a real native camera/runtime')
        from .tabletop_grasp_contact_native import NativeGraspContactObserver
        contact_observer=NativeGraspContactObserver(device_world,env,manifest,output,
            checkpoint_sha256=checkpoint_sha)
        previous_callback=env.evaluation_telemetry.callback
        def contact_callback(payload):
            contact_observer.sample(payload)
            previous_callback(payload)
        env.evaluation_telemetry.callback=contact_callback
    collector.begin_trials()
    control_steps=0; error=None
    try:
        with torch.inference_mode():
            while control_steps<maximum and collector.summary()['completed_count']<len(manifest['trials']):
                action=policy.act(obs)
                critic,_,done,_=env.step(action)
                control_steps+=1
                previous=action.clone(); previous[done]=0.
                if collector.summary()['completed_count']==len(manifest['trials']): break
                obs=_observations(camera.sample(),device_world,previous,history,critic,reset_mask=done)
    except Exception as failure:
        error=f'{type(failure).__name__}: {failure}'
    finally:
        raw_stream.close()
        if contact_observer is not None:
            env.evaluation_telemetry.callback=previous_callback
            contact_observer.finish(reason='evaluation_error' if error else 'bounded_evaluation_end')
    evidence['conditional_telemetry_samples']=dict(path=str(raw_path),sha256=file_sha256(raw_path))
    report=collector.summary()
    report.update(status=('not_completed' if error or report['incomplete_count']
        or report['invalid_measurement_count'] else 'completed'),
        control_steps=control_steps,max_control_steps=maximum,error=error,
        evaluation_seed=manifest['metadata']['seed'],scene_seed=getattr(args,'seed',None),
        checkpoint_sha256_after=file_sha256(args.vision_evaluation_checkpoint),
        inference_provenance=policy.provenance,evidence_files=evidence,
        physics_identity=dict(version=PHYSICS_IDENTITY_VERSION,
            canonical_sha256=manifest['metadata']['physics_sha256'],
            source_privileged_critic_contract_sha256=_digest(source['privileged_critic_contract']),
            execution_privileged_critic_contract_sha256=_digest(critic_contract)))
    if report['checkpoint_sha256_after']!=checkpoint_sha:
        raise RuntimeError('immutable evaluation checkpoint changed')
    report_path=output/'vision_evaluation_report.json'; _json(report_path,report)
    from .tabletop_tensor_world import TensorWorld
    from .tabletop_tensor_env import TensorEnv
    genuine=(native_camera and type(device_world) is TensorWorld and type(env) is TensorEnv
        and type(world).__module__.endswith('tabletop_gpu_isaac')
        and hasattr(world,'sim') and type(world.sim).__module__.startswith('isaaclab.'))
    if (not getattr(args,'grasp_contact_probe',False)
            and genuine and report['status']=='completed' and report['conditional_telemetry_complete']
            and _production_manifest_eligible(manifest,min_trials_per_style=min_trials_per_style)):
        certificate=dict(schema_version='dh116-native-visual-evaluation-certificate-v1',
            attestation_status='native_verified',execution='native_isaac_tensor_world',
            native_execution_complete=True,conditional_telemetry_complete=True,
            report_sha256=file_sha256(report_path),checkpoint_sha256=checkpoint_sha,
            manifest_sha256=manifest_sha256(manifest,min_trials_per_style=min_trials_per_style),
            split=manifest['metadata']['split'],expected_count=len(manifest['trials']),
            runner_source_sha256=file_sha256(__file__),command=sys.argv.copy(),
            physics_sha256=manifest['metadata']['physics_sha256'],camera_sha256=manifest['metadata']['camera_sha256'],
            physics_identity=deepcopy(report['physics_identity']),
            evidence_files=evidence,
            raw_reset_readback_sha256=evidence['raw_reset_readback']['sha256'],
            initial_camera_geometry_sha256=evidence['initial_camera_geometry']['sha256'],
            initial_visual_actor_inputs_sha256=evidence['initial_visual_actor_inputs']['sha256'])
        _json(output/'vision_evaluation_certificate.json',certificate)
    return report
