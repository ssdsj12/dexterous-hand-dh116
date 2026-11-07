"""Independent native stage-A evidence runner, with no Student/PPO storage.

This deliberately does not route into qualified BC/DAgger collection. Passing
source geometry, Teacher or MPC experiments never counts as Student success.
Later stages are gated and must have their own frozen experiments.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .tabletop_tripod_recovery_runtime import (TripodRecoveryRuntime,
    PREPARATION_TICKS, validate_staged_options, _safe_json)


def _write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(_safe_json(value),stream,indent=2,allow_nan=False)


def _source_models():
    from .tabletop_tripod_nonlinear_geometry import SourceHandGeometry
    root=Path(__file__).resolve().parents[4]
    manifest=json.loads((root/'assets/fixed_dual_panda_dh116_v1/manifest.json').read_text())
    result=[]
    for side in ('left','right'):
        row=next(r for r in manifest['sources'] if r['kind']=='dh116' and r['side']==side)
        source=Path(row['path'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('original source hand asset hash changed')
        result.append(SourceHandGeometry.from_usd(source,side))
    return result


def _verify_original_scene(world,device_world):
    """Read authored static collision bounds and original mimic relationship.

    No collision/material/drive attributes are written. The native source
    manifold and independent conservative source hulls remain separate models.
    """
    from pxr import Usd,UsdGeom,UsdPhysics
    from .linear_coupling import MIMIC
    stage=world.sim.stage;cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
    rows=[]
    layout=world.layout
    expected=[('Table',np.asarray(layout['table_center']),np.asarray(layout['table_size']))]
    for item in (world.end_support_scene or {}).get('supports',[]):
        expected.append(('EndSupports/'+item['name'],np.asarray(item['center_m']),np.asarray(item['size_m'])))
    for index,path in enumerate(world.paths):
        origin=np.asarray(world.origins[index])
        for suffix,center,size in expected:
            prim=stage.GetPrimAtPath(path+'/'+suffix)
            colliders=([p for p in Usd.PrimRange(prim,Usd.TraverseInstanceProxies())
                if p.HasAPI(UsdPhysics.CollisionAPI)
                and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False] if prim else [])
            if not colliders:
                raise ValueError('original static collider absent: '+path+'/'+suffix)
            # IsaacLab Cuboid creates an Xform parent with collision geometry
            # below it. Verify enabled physical shapes, excluding visual-only
            # descendants, without authoring any scene property.
            boxes=[cache.ComputeWorldBound(p).ComputeAlignedRange() for p in colliders]
            actual=np.stack((np.min([np.asarray(b.GetMin()) for b in boxes],axis=0),
                             np.max([np.asarray(b.GetMax()) for b in boxes],axis=0)))
            wanted=np.stack((origin+center-size/2,origin+center+size/2))
            if not np.allclose(actual,wanted,atol=2e-6,rtol=0):
                raise ValueError('original static collider world bounds mismatch: '+path+'/'+suffix)
            rows.append(dict(path=str(prim.GetPath()),bounds_w_m=actual.tolist(),
                collider_paths=[str(p.GetPath()) for p in colliders]))
    # Already validated native mimic contract is read back and recorded by the
    # original scene setup. Preserve its report; no ideal q is imposed here.
    mimic=[]
    for path in world.paths:
        for side in ('left','right'):
            for follower,master,ratio,offset in MIMIC:
                joint=stage.GetPrimAtPath(f'{path}/{side}Hand/joints/{follower}')
                gearing=joint.GetAttribute('physxMimicJoint:rotX:gearing').Get()
                bias=joint.GetAttribute('physxMimicJoint:rotX:offset').Get()
                axis=joint.GetAttribute('physxMimicJoint:rotX:referenceJointAxis').Get()
                targets=joint.GetRelationship('physxMimicJoint:rotX:referenceJoint').GetTargets()
                if (gearing is None or bias is None or axis!='rotX'
                        or not np.isclose(gearing,-ratio,atol=1e-6,rtol=0)
                        or not np.isclose(bias,-np.degrees(offset),atol=1e-5,rtol=0)
                        or len(targets)!=1 or str(targets[0])!=f'{path}/{side}Hand/joints/{master}'):
                    raise ValueError('original authored native mimic identity mismatch: '+str(joint.GetPath()))
                mimic.append(dict(path=str(joint.GetPath()),gearing=float(gearing),offset_degrees=float(bias),
                    reference_axis=axis,reference_joint=str(targets[0])))
    report=dict(static_colliders=rows,source='actual authored USD world bounds',
        expected_original_affine_mimic=[list(v) for v in MIMIC],
        actual_joint_names=[list(h.joint_names) for h in device_world.hands],
        native_mimic_identity=mimic,
        physical_parameters_modified=False,actuator_limits_modified=False,
        source_certificate_is_not_cooked_PhysX_identity=True)
    return report


@torch.no_grad()
def run_staged_tripod_experiment(world,device_world,task,env,args,critic_contract):
    validate_staged_options(args)
    from .tabletop_tensor_world import TensorWorld
    from .tabletop_tensor_env import TensorEnv
    if not (type(device_world) is TensorWorld and type(env) is TensorEnv
            and type(world).__module__.endswith('tabletop_gpu_isaac')
            and type(world.sim).__module__.startswith('isaaclab.')):
        raise ValueError('native staged experiment requires genuine Isaac TensorWorld/TensorEnv')
    if (task.command.shape!=(args.num_envs,12) or not getattr(task,'surface_policy_owned_targets',False)
            or task.surface_redistribution is not False or device_world.decimation!=8):
        raise ValueError('unchanged policy-owned canonical24D/480Hz controller required')
    if hasattr(env,'control_probe'):
        raise ValueError('staged contact evidence cannot replace an existing control probe')
    from .tabletop_vision_training import _contract,_observations
    from .tabletop_vision_student import SensorHistory
    from .tabletop_vision_fixed_camera import FixedVisionCamera
    from .tabletop_vision_supervision_dataset import load_teacher_reference_dataset
    from .tabletop_vision_teacher_collection import RawRGBDChunks,_choose_references,_teacher_optimizer
    output=Path(args.output);start=time.monotonic()
    source=load_teacher_reference_dataset(args.vision_teacher_source,critic_contract)
    references=_choose_references(source,args.num_envs)
    models=_source_models()
    _write(output/'tripod_scene_readback.json',_verify_original_scene(world,device_world))
    _write(output/'vision_student_contract.json',_contract(args,critic_contract))
    _write(output/'tripod_experiment_identity.json',dict(stage='A',native_type_check_passed=True,
        teacher_source_sha256=source['source_sha256'],source_hands=[m.spec['source_manifest'] for m in models],
        actor_size=3844,critic_size=371,action_size=24,ppo_updates=0,student_successes=0,
        preparation='published pressure_reference ticks0..436, no contact manipulation',
        fixed_wrist='actual SDK frame captured at437; target freeze plus measured settling',
        native_success_thresholds=dict(net_lift_m=.03,hold_s=5.,slip_m=.0005),
        raw_physics_hz=480,control_hz=60,
        native_cold_reset_slots=args.num_envs,unique_randomized_initial_states_attested=False,
        counts_as_student_success=False,no_external_force=True,no_attachment=True,no_new_support=True))
    runtime=TripodRecoveryRuntime(env,models,source['source_sha256'],output)
    raw=RawRGBDChunks(output/'raw_rgbd',chunk_frames=args.vision_teacher_raw_chunk_frames)
    camera=None;failure=None;result=None;first_terminal={};ticks=0;decision='recording_budget_reached'
    try:
        critic=env.reset()
        _write(output/'initial_full_sdk_readback.json',runtime.last_batch['packets'])
        camera=FixedVisionCamera(world,device_world,args.depth_camera_config,
            width=args.vision_width,height=args.vision_height,num_points=128)
        camera.set_raw_frame_sink(raw)
        history=SensorHistory(args.num_envs,device=args.device)
        previous=torch.zeros(args.num_envs,24,device=args.device)
        done=torch.zeros(args.num_envs,device=args.device,dtype=torch.bool)
        optimizer=_teacher_optimizer(env,args,source,references)
        with (output/'control_actions.jsonl').open('x',encoding='utf-8') as actions:
            for tick in range(args.vision_teacher_max_control_steps):
                runtime.tick=tick
                if tick==PREPARATION_TICKS:runtime.start_recovery()
                frame=camera.sample()
                obs=_observations(frame,device_world,previous,history,critic,reset_mask=done)
                if obs['policy'].shape!=(args.num_envs,3844):raise RuntimeError('visual Actor contract changed')
                if tick==0:
                    from .tabletop_vision_audit import audit_initial_points
                    _write(output/'initial_camera_geometry_validation.json',
                        audit_initial_points(frame,device_world,radius=args.solid_cylinder_radius_m,length=.5))
                reference=previous.new_zeros(args.num_envs,24)
                for index,item in enumerate(references):
                    if tick<len(item['actions']) and not bool(runtime.ended[index]):
                        reference[index]=item['actions'][tick].to(reference)
                alive=~runtime.ended
                pressure,detail=optimizer.refine(reference,context=dict(tick=tick,
                    physics_step=int(world.sim.current_time_step_index),simulation_time=float(world.sim.current_time),
                    episode_id=runtime.episode_ids.tolist()),alive=alive)
                issued=runtime.refine(pressure);issued[~alive]=0.
                if not bool(torch.isfinite(issued).all() and (issued.abs()<=1).all()):
                    raise RuntimeError('noncanonical24D staged command')
                actions.write(json.dumps(_safe_json(dict(tick=tick,frame_id=int(frame['frame_id']),
                    physics_step=device_world._physics_steps,simulation_step_index=int(world.sim.current_time_step_index),
                    simulation_time_s=float(world.sim.current_time),physical_episode_ids=runtime.episode_ids,
                    reference=reference,pressure_reference=pressure,issued=issued,
                    pressure_diagnostics=detail)),allow_nan=False,separators=(',',':'))+'\n')
                critic,reward,done,extra=env.step(issued.clone())
                optimizer.observe_result(done|~alive);previous=issued.clone();ticks=tick+1
                for index in (done&alive).nonzero().flatten().tolist():
                    first_terminal[index]=dict(control_tick=tick,native_success=bool(extra['success'][index]),
                        failure_code=int(extra['failure'][index]),counts_as_student_success=False)
                if runtime.active and len(runtime.geometry_blocked)==int(runtime.tripod.sum()):
                    decision='stage_A_blocked_all_actual_source_geometry_uncertified';break
                if runtime.active and all(runtime.latest_rows.get(i,{}).get('stage_gate',{}).get('pass',False)
                        for i in runtime.tripod.nonzero().flatten().tolist()):
                    decision='stage_A_passed_requires_independent_audit_before_B';break
                if bool(runtime.ended[runtime.tripod].all()):decision='stage_A_first_native_episodes_terminated';break
                if tick%20==0:
                    runtime.physics.flush();runtime.events.flush();runtime.decisions.flush();actions.flush()
                    print(json.dumps(dict(stage='A',completed_controls=ticks,
                        physics_steps=device_world._physics_steps,active=runtime.active,
                        confirmed_approach_submissions=runtime.executions)),flush=True)
    except BaseException as error:
        failure=repr(error)
        decision='stage_A_interrupted' if isinstance(error,KeyboardInterrupt) else 'stage_A_exception'
        raise
    finally:
        try:result=runtime.close()
        finally:raw.close()
        _write(output/'tripod_stage_report.json',dict(stage='A',decision=decision,failure=failure,
            completed_controls=ticks,elapsed_wall_s=time.monotonic()-start,first_terminal=first_terminal,
            result=result,teacher_mpc_only=True,student_successes=0,ppo_updates=0,
            later_stages=dict(B='not_run_pending_A_and_independent_evidence',
                C='not_run_pending_B',D='not_run_pending_C',E='not_run_pending_native_A_B_C_D'),
            native_success_standard_unchanged=True))
    return dict(mode='native_tripod_stage_A',ppo_updates=0,student_successes=0,
        stage_report=str(output/'tripod_stage_report.json'),decision=decision)
