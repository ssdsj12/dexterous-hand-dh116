"""Explicit stereo camera probes and offline depth-reference measurement."""
import argparse
import copy
import json
from pathlib import Path

import torch

from .tabletop_stereo_calibration import StereoCalibration
from .tabletop_stereo_contract import ROOT, action_sha256, digest, file_sha256, new_run, require_opt_in, state_contract, write_json_new


def depth_metrics(prediction, reference, valid, reference_valid):
    if (prediction.shape != reference.shape or valid.shape != prediction.shape
            or reference_valid.shape != prediction.shape or valid.dtype != torch.bool
            or reference_valid.dtype != torch.bool):
        raise ValueError('aligned stereo/reference depth and explicit masks required')
    mask = valid & reference_valid & torch.isfinite(prediction) & torch.isfinite(reference)
    if not mask.any():
        return dict(count=0, mae_m=None, rmse_m=None, p95_abs_m=None, validity_coverage=0.)
    error = (prediction[mask]-reference[mask]).float()
    return dict(count=int(mask.sum()), mae_m=float(error.abs().mean()),
        rmse_m=float(error.square().mean().sqrt()), p95_abs_m=float(error.abs().quantile(.95)),
        validity_coverage=float(mask.float().mean()))


def evaluate_capture(capture_path, reference_path):
    captured = torch.load(capture_path, map_location='cpu', weights_only=False)
    reference = torch.load(reference_path, map_location='cpu', weights_only=False)
    if (captured.get('version') != 'dh116-stereo-rgb-probe-v1'
            or reference.get('version') != 'dh116-stereo-depth-reference-v1'):
        raise ValueError('versioned stereo probe and independent depth reference required')
    fields = ('calibration_sha256', 'frame_id', 'simulation_time')
    packet = captured['frames'][0]
    if any(reference.get(key) != packet.get(key) for key in fields):
        raise ValueError('depth reference calibration/frame/timestamp mismatch')
    if packet.get('depth_source') != 'computed_from_left_right_RGB':
        raise ValueError('rendered depth cannot be reported as stereo prediction')
    metrics = depth_metrics(packet['depth_m'], reference['depth_m'],
        packet['valid'], reference['valid'])
    return dict(version='dh116-stereo-depth-evaluation-v1', metrics=metrics,
        capture_sha256=file_sha256(capture_path), reference_sha256=file_sha256(reference_path),
        evidence_scope='offline_reference_comparison', native_calibration_qualification=False,
        reference_used_as_actor_input=False)


def main(mode='probe', argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/dh116_training/stereo_camera_640x480_v2.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=116)
    parser.add_argument('--num-envs', type=int, default=2)
    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--warmup-frames',type=int,default=0)
    parser.add_argument('--advance-control-between-samples',action='store_true',
        help='caller executes one explicit zero24 control step BETWEEN complete RGB/reference pairs; no Teacher/MPC')
    parser.add_argument('--image-processing-batch-size',type=int,default=1)
    parser.add_argument('--capture-validation-reference',action='store_true',
        help='independent native optical-Z reference; validation only, never an Actor input')
    parser.add_argument('--actor-package',type=Path,
        help='optional matching actor-only package for measured inference latency; absent means actor_ms=NA')
    parser.add_argument('--controlled-depth',action='store_true',help='controlled resolution comparison: blockmatch5/subpixel; original config bytes preserved')
    parser.add_argument('--integer-disparity',action='store_true',help='explicit integer disparity ablation, recorded separately')
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--reference-depth', type=Path)
    parser.add_argument('--native-task-config', type=Path, default=ROOT / 'configs/dh116_training/stereo_native_task_v1.json')
    parser.add_argument('--check-config', action='store_true')
    parser.add_argument('--enable-experiment', action='store_true')
    parser.add_argument('--execute-native', action='store_true')
    args = parser.parse_args(argv)
    calibration = StereoCalibration.from_json(args.config)
    if args.num_envs < 1 or args.frames < 1 or args.warmup_frames<0 or args.image_processing_batch_size<1:
        parser.error('positive num-envs and frame count required')
    if mode=='probe' and args.frames+args.warmup_frames>1 and not args.advance_control_between_samples:
        parser.error('multi-frame probe requires explicit --advance-control-between-samples; frozen ReferenceTime need not advance')
    if args.check_config:
        print(json.dumps(dict(validated=True, native_started=False, calibration_sha256=calibration.sha256,
            width=calibration.width, height=calibration.height, baseline_m=calibration.baseline_m)))
        return 0
    require_opt_in(args.enable_experiment)
    if args.output is None:
        parser.error('new --output required')
    if mode == 'evaluate':
        if args.capture is None or args.reference_depth is None:
            parser.error('--capture and --reference-depth required; no simulator depth fallback')
        report = evaluate_capture(args.capture, args.reference_depth)
        run = new_run(args.output)
        write_json_new(run / 'depth_evaluation.json', report)
        return 0
    if not args.execute_native:
        parser.error('camera acquisition requires explicit --execute-native via scripts/python.sh')
    run = new_run(args.output)
    from .tabletop_stereo_native import launch_native
    from .tabletop_stereo_camera import StereoCamera

    def callback(bridge):
        write_json_new(run/'native_camera_callback_entered.json',dict(source_scope='native_runtime_callback_entered',
            entrypoint='tabletop_stereo_camera_cli.main.callback',num_envs=bridge.n,
            source_sha256=file_sha256(__file__),native_accuracy_qualification=False))
        camera = StereoCamera.from_native(bridge.world.sim, bridge.device_world.origins,
            args.config, device=bridge.device)
        from .tabletop_stereo_depth import StereoDepth
        from .tabletop_stereo_reference import ValidationDepthReference,capture_validation_pair,compare_reference,archive_instance_reference
        from .tabletop_stereo_resolution_benchmark import TimedStereoDepth,gpu_memory_snapshot,timed_call
        camera_config=json.loads(args.config.read_text(encoding='utf-8'))
        depth_config=StereoDepth.from_config(camera_config).to_config()
        if args.controlled_depth:depth_config.update(method='blockmatch',block_size=5,subpixel=True)
        if args.integer_disparity:depth_config['subpixel']=False
        depth_config['image_processing_batch_size']=args.image_processing_batch_size
        estimator=StereoDepth(**depth_config)
        camera.depth_estimator=TimedStereoDepth(estimator,bridge.device)
        # Read actual per-environment RigidObject roots; no name/seed heuristic.
        wire_prim_paths=list(bridge.device_world.wire.root_physx_view.prim_paths) if args.capture_validation_reference else None
        reference_reader=ValidationDepthReference.from_camera(camera,wire_prim_paths) if args.capture_validation_reference else None
        bridge.set_execution_context(owner='diagnostic_zero',checkpoint_sha256=None)
        bridge.set_action_owners(['diagnostic_zero']*bridge.n)
        actor=None;actor_error=None;history=None;extractor=None
        if args.actor_package is not None:
            from .tabletop_stereo_checkpoint import load_deployment_actor
            from .tabletop_stereo_observation import StereoHistory
            from .tabletop_stereo_perception import WirePointExtractor
            actor=load_deployment_actor(args.actor_package,expected_calibration_sha256=camera.calibration.sha256).to(bridge.device)
            history=StereoHistory(bridge.n,4,device=bridge.device);extractor=WirePointExtractor()
        frames=[];references=[];metrics=[];timings=[];memory=[];alignment_failures=[];zero_steps=[];schedule=[]
        initial_conditions=[copy.deepcopy(bridge._initial_conditions[bridge.identities.uid(i)]) for i in range(bridge.n)]
        initial_sha=[bridge._initial_sha[bridge.identities.uid(i)] for i in range(bridge.n)]
        rig=dict(baseline_m=camera.calibration.baseline_m,T_world_left=camera.calibration.T_world_left.tolist(),
                 T_world_right=camera.calibration.T_world_right.tolist())
        initial_tick=int(bridge.device_world._physics_steps);initial_time=float(bridge.world.sim.current_time)
        initial_reset_ids=list(bridge.identities.reset_ids)
        if torch.device(bridge.device).type=='cuda':torch.cuda.reset_peak_memory_stats(bridge.device)
        for index in range(args.warmup_frames+args.frames):
            if index>0 and args.advance_control_between_samples:
                start_tick=int(bridge.device_world._physics_steps);start_time=float(bridge.world.sim.current_time)
                (_,_,done,extra),advance_ms=timed_call(bridge.step,bridge.device,torch.zeros(bridge.n,24,device=bridge.device))
                zero_steps.append(dict(owner='diagnostic_zero',normalized_action=[0.]*24,
                    physics_tick_start=start_tick,physics_tick_end=int(bridge.device_world._physics_steps),
                    time_start=start_time,time_end=float(bridge.world.sim.current_time),physics_advance_ms=advance_ms,
                    terminal=done.detach().cpu().tolist(),failure=extra['failure'].detach().cpu().tolist(),
                    grasp_success_claimed=False))
            identity=bridge.pre_action_identities(camera.calibration.sha256)
            before=len(camera.depth_estimator.samples)
            if reference_reader is not None:
                (pair,reference,failed),total_ms=timed_call(capture_validation_pair,bridge.device,camera,reference_reader,identity)
                reference['evaluation_depth_range_m']=[estimator.near,estimator.far]
                alignment_failures.extend(failed)
            else:
                pair,total_ms=timed_call(camera.sample,bridge.device,identity=identity);reference=None
            depth_ms=sum(camera.depth_estimator.samples[before:])
            actor_ms=None;preprocessing_ms=None;actor_forward_ms=None
            if actor is not None:
                from .tabletop_stereo_observation import SensorBatch
                def prepare():
                    proposal=extractor.extract(pair['left_rgb'],pair['depth_m'],pair['valid'],pair['K_left'],pair['T_world_left'])
                    sensor=SensorBatch(left_rgb=pair['left_rgb'].float(),right_rgb=pair['right_rgb'].float(),
                        depth_m=pair['depth_m'].float(),depth_valid=pair['valid'],depth_confidence=pair['confidence'].float(),
                        points_world=proposal['points_world'].float(),point_valid=proposal['point_valid'],
                        proprio=bridge.proprioception(),dt=torch.full((bridge.n,),1/60,device=bridge.device),
                        image_age=torch.zeros(bridge.n,device=bridge.device),
                        image_valid=torch.ones(bridge.n,2,dtype=torch.bool,device=bridge.device))
                    return history.update(sensor)
                inputs,preprocessing_ms=timed_call(prepare,bridge.device)
                with torch.no_grad():_,actor_forward_ms=timed_call(actor.deterministic,bridge.device,inputs)
                actor_ms=preprocessing_ms+actor_forward_ms
            if index<args.warmup_frames:continue
            schedule.append(dict(policy_tick=bridge.policy_tick,physics_tick_from_initial=int(bridge.device_world._physics_steps)-initial_tick,
                simulation_time_from_initial=round(float(bridge.world.sim.current_time)-initial_time,9),
                reset_counts_from_initial=[row['reset_id']-start for row,start in zip(pair['identity'],initial_reset_ids)],
                sampled_physical_initial_sha256=[row['physical_initial_condition_sha256'] for row in pair['identity']],
                episode_uid=[row['episode_uid'] for row in pair['identity']]))
            if reference is not None:metrics.append(compare_reference(pair,reference))
            timings.append(dict(capture_ms=max(0.,total_ms-depth_ms),depth_ms=depth_ms,
                                actor_ms=actor_ms,preprocessing_ms=preprocessing_ms,actor_forward_ms=actor_forward_ms))
            memory.append(gpu_memory_snapshot(bridge.device))
            def cpu(value):
                if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
                if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
                if isinstance(value,list):return [cpu(v) for v in value]
                return value
            pair=cpu(pair)
            pair['depth_source'] = 'computed_from_left_right_RGB'
            pair['actual_palm_fk_pose_wxyz'] = bridge.palm_poses().cpu()
            pair['environment_origins'] = bridge.device_world.origins.detach().cpu().clone()
            frames.append(pair)
            if reference is not None:
                reference=cpu(reference);references.append(reference)
                sdk=reference.get('sdk_camera_view_source_artifact')
                if isinstance(sdk,dict):
                    snapshot=run/'native_sdk_camera_view_source.py'
                    if not snapshot.exists():
                        with snapshot.open('xb') as stream:stream.write(Path(sdk['path']).read_bytes())
                    if file_sha256(snapshot)!=sdk['sha256']:raise ValueError('SDK camera source changed during capture')
                    reference['sdk_camera_view_source_artifact']={'path':str(snapshot),'sha256':sdk['sha256']}
                reference['wire_mask_source_artifact']=archive_instance_reference(reference,
                    run/f'wire_instance_reference_{len(references)-1:04d}.pt')
                with (run/f'depth_reference_{len(references)-1:04d}.pt').open('xb') as stream:torch.save(reference,stream)
        schedule_contract=dict(advance_control_between_samples=args.advance_control_between_samples,warmup_frames=args.warmup_frames,
            frames=args.frames,policy_ticks=[row['policy_tick'] for row in schedule],
            physics_ticks_from_initial=[row['physics_tick_from_initial'] for row in schedule],
            simulation_times_from_initial=[row['simulation_time_from_initial'] for row in schedule],
            reset_counts_from_initial=[row['reset_counts_from_initial'] for row in schedule],
            sampled_physical_initial_sha256=[row['sampled_physical_initial_sha256'] for row in schedule])
        cohort_sha=digest(dict(physical_initial_sha256=sorted(set(initial_sha)),rig=rig,capture_schedule=schedule_contract))
        performance=dict(source_scope='native',measurement_device='cuda' if torch.device(bridge.device).type=='cuda' else 'cpu',
            batch_size=bridge.n,frame_count=len(frames),timing_samples=timings,memory_samples=memory,
            actor_package_sha256=file_sha256(args.actor_package) if args.actor_package else None,
            actor_input_scope='frozen_physics_RGB_history_latency_only;not_policy_rollout',
            actor_latency_scope='wire RGB proposal/3D/history preprocessing + Actor forward; forward separately recorded',
            qualification='measurement_only_not_task_success')
        for name in ('capture_ms','depth_ms','actor_ms','preprocessing_ms','actor_forward_ms'):
            values=[row[name] for row in timings if row[name] is not None]
            performance[name]=sum(values)/len(values) if values else None
        for name in ('gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','gpu_device_used_bytes'):
            values=[row[name] for row in memory if row[name] is not None];performance[name]=max(values) if values else None
        values=[row['gpu_free_bytes'] for row in memory if row['gpu_free_bytes'] is not None]
        performance['gpu_free_bytes']=min(values) if values else None
        performance['memory_scope']='Validation depth render product active; Torch selected-device allocator peak; device used/free sampled, may exclude separate renderer GPU; not pure production RGB-only peak'
        write_json_new(run/'performance.json',performance)
        with (run / 'stereo_probe.pt').open('xb') as stream:
            torch.save(dict(version='dh116-stereo-rgb-probe-v1',frames=frames,
                environment_origins=bridge.device_world.origins.detach().cpu().clone(),
                initial_condition_source='actual_sdk_readback',physical_initial_conditions=initial_conditions,
                physical_initial_condition_sha256=initial_sha,cohort_sha256=cohort_sha,
                capture_schedule=schedule_contract,actual_schedule=schedule,zero_control_steps=zero_steps,
                initial_physics_tick=initial_tick,initial_simulation_time=initial_time,initial_reset_ids=initial_reset_ids,
                performance=performance,reference_alignment_failures=alignment_failures),stream)
        benchmark_metrics={}
        for name in ('depth_mae_m','depth_rmse_m','depth_p95_m','depth_validity_coverage',
                     'rgb_candidate_point_cloud_completeness','wire_point_cloud_completeness'):
            values=[row[name] for row in metrics if row[name] is not None]
            benchmark_metrics[name]=sum(values)/len(values) if values else None
        benchmark_metrics.update({key:performance[key] for key in ('capture_ms','depth_ms','actor_ms',
            'gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','gpu_free_bytes','gpu_device_used_bytes')})
        kwargs=dict(physics_hz=480,control_hz=60,width=camera.calibration.width,height=camera.calibration.height,
            num_envs=bridge.n,frames=args.frames,warmup_frames=args.warmup_frames,
            image_processing_batch_size=args.image_processing_batch_size,depth_estimator=depth_config,
            calibration_sha256=camera.calibration.sha256,actor_package_sha256=performance['actor_package_sha256'],
            advance_control_between_samples=args.advance_control_between_samples)
        write_json_new(run/'producer_config.json',dict(resolved_kwargs=kwargs))
        validation=json.loads((ROOT/'configs/dh116_training/stereo_validation_v1.json').read_text(encoding='utf-8'))
        validation.update(depth_estimator=depth_config,cohort_sha256=cohort_sha)
        write_json_new(run/'validation_config.json',validation)
        if references:
            proof=dict(version='dh116-native-stereo-proof-v1',source_scope='native',schema_only=False,
                process_exit_code=None,qualification_status='pending_actual_external_process_exit',
                action_contract_sha256=action_sha256(),state_contract_sha256=digest(state_contract()),
                producer=dict(source_artifact={'path':str(Path(__file__).resolve()),'sha256':file_sha256(__file__)},
                    config_artifact={'path':str(run/'producer_config.json'),'sha256':file_sha256(run/'producer_config.json')},
                    entrypoint='tabletop_stereo_camera_cli.main.callback',kwargs=kwargs,kwargs_sha256=digest(kwargs)),
                sample_count=len(frames),cohort_sha256=cohort_sha,
                calibration_artifact={'path':str(args.config.resolve()),'sha256':file_sha256(args.config)},
                validation_config_artifact={'path':str(run/'validation_config.json'),'sha256':file_sha256(run/'validation_config.json')},
                capture_artifact={'path':str(run/'stereo_probe.pt'),'sha256':file_sha256(run/'stereo_probe.pt')},
                performance_artifact={'path':str(run/'performance.json'),'sha256':file_sha256(run/'performance.json')},
                actor_package_artifact={'path':str(args.actor_package.resolve()),'sha256':file_sha256(args.actor_package)} if args.actor_package else None,
                reference_artifacts=[{'path':str(run/f'depth_reference_{i:04d}.pt'),'sha256':file_sha256(run/f'depth_reference_{i:04d}.pt')} for i in range(len(references))])
            write_json_new(run/'native_stereo_proof_pending.json',proof)
        report = dict(status='bounded_stereo_probe_completed', frames=len(frames),
            source_scope='native',schema_only=False,width=camera.calibration.width,height=camera.calibration.height,
            cohort_sha256=cohort_sha,benchmark_metrics=benchmark_metrics,
            stereo_qualification=str(run/'native_stereo_proof_pending.json') if references else None,
            depth_source='computed_from_left_right_RGB', render_advances_physics=False,
            native_geometry_accuracy_qualification=False, grasp_success_claimed=False,
            mean_valid_fraction=[float(f['valid'].float().mean()) for f in frames],
            limitation='Runtime measurements only; RGB candidate mask is not GT; absence of Actor package means actor_ms=NA; common verifier still required')
        write_json_new(run / 'stereo_probe.json', report)
        return report

    launch_native(callback, native_arguments=args.native_task_config, output=run / 'native',
        num_envs=args.num_envs, seed=args.seed, enabled=True)
    return 0
