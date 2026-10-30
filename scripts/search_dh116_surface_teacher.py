#!/usr/bin/env python3
"""One Isaac instance validates geometric pregrasps; only measured lifts export."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback
import torch


def entry():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--maximum-candidates',type=int,default=4)
    parser.add_argument('--controls',type=int,default=600)
    parser.add_argument('--wall-seconds',type=float,default=600.)
    parser.add_argument('--ascent-start-s',type=float,default=2.)
    parser.add_argument('--pinch-hold',action='store_true',
        help='Keep the chosen clamp force and continue force feedback without scheduled ascent')
    options=parser.parse_args()
    if options.output.exists():raise FileExistsError('preserve existing physical evidence')
    payload=json.loads(options.candidates.read_text())
    rows=[r for r in payload['candidates'] if r['geometry']['penetration']<=.3
        and r['geometry']['object_in_hand']<=.3 and r['geometry']['self']<=.3
        and r['geometry']['minimum_clearance_m']>=.00029][:options.maximum_candidates]
    if not rows:raise RuntimeError('No collision-safe pregrasp candidate; nothing will be simulated')
    from go2_pvcnn.control.dh116 import tabletop_tensor_training as training

    def run(world,template,args,contract):
        from go2_pvcnn.control.dh116.tabletop_tensor_world import TensorWorld
        from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
        from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv
        from go2_pvcnn.control.dh116.tabletop_policy_force_region import enable as force
        from go2_pvcnn.control.dh116.tabletop_free_grasp_lift import enable as free
        from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import enable as wrist,apply_contract
        from go2_pvcnn.control.dh116.tabletop_surface_kinematics import SurfaceHandModel,full_angles,pose_matrix,CONTRACT as SURFACE_MODEL
        from go2_pvcnn.control.dh116.tabletop_surface_candidates import object_surface_points
        from go2_pvcnn.control.dh116.tabletop_surface_objectives import finite_cylinder_sdf
        from go2_pvcnn.control.dh116.tabletop_surface_teacher import Qualification,export_qualified,initial_geometry_mask,implementation_identity
        from go2_pvcnn.control.dh116.tabletop_surface_teacher_control import SurfaceTeacherControl
        from go2_pvcnn.control.dh116.tabletop_surface_support import install_support_servo,CONTRACT as SUPPORT_CONTRACT
        from go2_pvcnn.control.dh116.tabletop_surface_force_goals import CONTRACT as GOAL_CONTRACT
        from go2_pvcnn.control.dh116.linear_coupling import JOINTS
        runtime_implementation=implementation_identity(root)
        training.write_json(args.output/'runtime_source_identity.json',dict(
            captured_before_candidate_physics=True,sha256_lf=runtime_implementation,
            proposal_implementation_hashes=payload.get('implementation_hashes')))
        w=TensorWorld(world)
        t=TensorTask(template,args.num_envs,args.device,pressure_relief=True,
            contact_reward_window='close_start',grasp_readiness='measured_load',
            vertical_force_reward=True,vertical_force_reward_mode='continuous',
            force_reward_start='downward_after_close',skip_hold=True,close_downward_penalty=True,
            grasp_shaping=False,palm_exploration_axis='xyz')
        t.enable_policy_control();force(t);free(t);wrist(t,w);t.phase_force_reward=True
        for name in ('grasp_upward_force','lift_upward_force'):t.terms[name]=torch.zeros_like(t.reward)
        env=TensorEnv(w,t,recovery=True,training_joint_limit=True);env.reset()
        startup=training.validate_world_adapter(world,w,t)
        names=['d134680ab619_DH116_left.usd','3046ad2ed855_DH116_right.usd']
        sources=[root/'assets/fixed_dual_panda_dh116_v1/dependencies'/name for name in names]
        hashes={side:hashlib.sha256(source.read_bytes()).hexdigest() for side,source in zip(('left','right'),sources)}
        if payload['source_hashes']!=hashes:raise ValueError('candidate source USD identity mismatch')
        models=[SurfaceHandModel.from_usd(source,side,device=w.device) for source,side in zip(sources,('left','right'))]
        if payload.get('surface_model_contract') is not None and payload['surface_model_contract']!=SURFACE_MODEL:
            raise ValueError('candidate surface geometry contract mismatch')
        candidate_q=w.t([r['active_angles'] for r in rows]).reshape(w.n,2,6)
        candidate_roots=w.t([r['wrist_pose_w'] for r in rows]);candidate_roots[:,:,:3]+=w.origins[:,None]
        w.root_seed=candidate_roots.clone();w.initial_offset=0.
        w.goal_roots=candidate_roots[:,:,:3].clone()
        for h in range(2):
            w.seed_q[h].zero_();w.seed_qd[h].zero_()
            w.seed_q[h][:,world.named_ids[h]]=full_angles(candidate_q[:,h]).float()
        original_reset=t.reset
        def reset(ids,measurement):
            original_reset(ids,measurement)
            for name in ('command','reference','previous','old_command','joint_requested_target','joint_governed_target'):
                if hasattr(t,name):getattr(t,name)[ids]=candidate_q.flatten(1)[ids]
            t.offset[ids]=0.;t.last_observation[ids]=t.observe()[ids]
        t.reset=reset
        env.reset()
        contacts=([r['seed_contact_points_local'] for r in rows] if all('seed_contact_points_local' in r for r in rows) else None)
        seed_links=([r['seed_contact_body_names'] for r in rows] if all('seed_contact_body_names' in r for r in rows) else None)
        controller=SurfaceTeacherControl(t,w,models,ascent_start_s=options.ascent_start_s,
            seed_local_contacts=contacts,seed_body_names=seed_links,pinch_hold=options.pinch_hold)
        actual=controller.actual_poses();errors=[]
        for h,model in enumerate(models):
            predicted=model.fk(t.m['q'].reshape(w.n,2,6)[:,h],w.roots[:,h])
            error=max(float((predicted[body]-actual[h][body]).abs().max()) for body in predicted)
            errors.append(error)
        if max(errors)>2e-5:raise RuntimeError('offline/native FK mismatch: '+str(errors))
        sampled=[model.sample_poses(p) for model,p in zip(models,actual)]
        initial_objects=object_surface_points(w.wp,w.wq)
        hand_depth=(-finite_cylinder_sdf(torch.cat([s['collision_points'] for s in sampled],1),w.wp,w.wq)).clamp_min(0).amax(-1)
        object_depth=torch.stack([model.object_penetration_poses(p,initial_objects) for model,p in zip(models,actual)],-1).amax(-1)
        self_depth=torch.stack([model.self_penetration_poses(p) for model,p in zip(models,actual)],-1).amax(-1)
        initial_safe=initial_geometry_mask(hand_depth,object_depth,self_depth,w.clearances)
        initial_report=dict(checked_before_any_candidate_physics=True,current_geometry_contract=dict(SURFACE_MODEL),
            candidate_contract_declared=payload.get('surface_model_contract') is not None,
            hand_in_object_mm=(hand_depth*1000).cpu().tolist(),object_in_hand_mm=(object_depth*1000).cpu().tolist(),
            self_penetration_mm=(self_depth*1000).cpu().tolist(),clearance_mm=(w.clearances*1000).cpu().tolist(),
            safe=initial_safe.cpu().tolist())
        training.write_json(args.output/'initial_geometry_preflight.json',initial_report)
        if not initial_safe.all():raise RuntimeError('current measured candidate geometry failed preflight: '+str(initial_report))
        apply_contract(contract,t);contract['surface_teacher']=dict(
            source_hashes=hashes,candidate_file_sha256=hashlib.sha256(options.candidates.read_bytes()).hexdigest(),
            teacher=False,scheduled_ascent_s=options.ascent_start_s,normal_force_goal_exploration=dict(GOAL_CONTRACT),
            maximum_offline_native_fk_error=errors,raw_contact_sampling_hz=args.physics_hz,
            geometry_sampling_hz=60,geometry_contract=dict(SURFACE_MODEL),ppo_updates=0)
        contract['surface_teacher'].update(pinch_hold=options.pinch_hold,
            force_goal_mode='latch_chosen_force_and_keep_feedback' if options.pinch_hold else 'bounded_exploration',
            scheduled_ascent_enabled=not options.pinch_hold,
            hold_wrist_command=options.pinch_hold,finger_force_feedback_always_enabled=True,
            pinch_table_clearance_m=.0003 if options.pinch_hold else None,
            mimic_tracking=('original_passive_drives_without_master_target_retreat_for_pinch_digits'
                if options.pinch_hold else 'existing_policy_master_correction'))
        contract['surface_teacher']['normal_force_goal_exploration'].update(
            initial_range_n=[2.,3.] if options.pinch_hold else [2.,12.])
        support_contract=dict(SUPPORT_CONTRACT)
        if options.pinch_hold:
            support_contract.update(normal_feedback='60Hz_compliance_scaled_force_to_finger_target;480Hz_original_drives',
                normal_feedback_filter='mean_of_previous_physics_interval;raw_samples_retained',
                force_correction_fraction=.2,active_position_stiffness_nm_rad=10.,
                tangent_feedback='disabled_in_pinch_hold',target='no_scheduled_vertical_support_in_pinch_hold',
                force_application='original_implicit_joint_drives',extra_motor_feedforward=False,
                motor_budget='full_original_implicit_drive_effort_limit')
        contract['surface_support']=dict(enabled=True,contract=support_contract,
            executor=('pinch_position_force_feedback_with_original_drives' if options.pinch_hold else
                'continuous_normal_and_vertical_force_servo_with_shared_motor_budget'),
            action_replay='26D_commands_plus_same_recorded_force_executor')
        install_support_servo(w,t,controller)
        training.publish_resolved_tensor_configuration(args.output,contract)
        training.write_json(args.output/'native_geometry_audit.json',dict(startup=startup,
            maximum_fk_matrix_errors=errors,active_finger_count=12,total_finger_count=22,
            hull_counts=[len(m.shapes) for m in models],patches_per_hand=2,samples_per_patch=24))

        class IntervalProbe:
            def __init__(self):self.begin()
            def begin(self):
                self.good=torch.ones(w.n,device=w.device,dtype=torch.bool)
                self.fault=torch.zeros(w.n,device=w.device,dtype=torch.long);self.samples=[]
            def before_step(self):pass
            def after_step(self,live,previous_phase):
                m=t.m;loads=m['grouped_object_loads'][:,:,:2];hands=m['hand_object_loads']
                valid=live&(t.failure==0)&~t.done
                valid &= (loads>.05).flatten(1).all(-1)&(loads.amax(-1)<=1.5*loads.amin(-1)).all(-1)
                valid &= ((hands>=2)&(hands<=12)).all(-1)&(m['bottom']>=.31-1e-10)
                valid &= m['object_velocity'][:,:3].norm(dim=-1)<.03
                tolerance=torch.maximum(torch.full_like(t.f_requested,.1),.05*t.f_requested)
                valid &= ((hands-t.f_requested).abs()<=tolerance).all(-1)
                valid &= w.vertical_components.sum((1,2))>=.9*w.object_weight_n
                self.good &= valid;self.fault|=t.failure
                self.samples.append(dict(physics_digit_normal_load_n=loads.clone(),physics_hand_normal_load_n=hands.clone(),
                    physics_bottom=m['bottom'].clone(),physics_object_pose_w=torch.cat((w.wp,w.wq),-1).clone(),
                    physics_object_velocity=m['object_velocity'].clone(),physics_fault=t.failure.clone(),
                    physics_requested_hand_force_n=t.f_requested.clone(),physics_object_weight_n=w.object_weight_n.clone(),
                    physics_vertical_components_n=w.vertical_components.clone(),
                    physics_support_torque_command_nm=w.effort_feedforward.clone(),
                    physics_support_target_total_fz_n=w.surface_support_diagnostic['target_total_fz_n'].clone()))
        probe=IntervalProbe();env.control_probe=probe
        qualifier=Qualification(w.n,w.device);frames=[];begin=time.monotonic();completed=0
        initial=[dict(active_angles=candidate_q[i].flatten().cpu().tolist(),wrist_pose_w=candidate_roots[i].cpu().tolist(),
            object_pose_w=w.seed_wire[i].cpu().tolist(),candidate_geometry=rows[i]['geometry']) for i in range(w.n)]
        for tick in range(options.controls):
            observation=t.observe().clone();action=controller.actions(tick/60)
            probe.begin();_,_,done,_=env.step(action)
            actual=controller.actual_poses();samples=[m.sample_poses(p) for m,p in zip(models,actual)]
            object_points=object_surface_points(w.wp,w.wq)
            hand_in_object=(-finite_cylinder_sdf(torch.cat([s['collision_points'] for s in samples],1),w.wp,w.wq)).clamp_min(0).amax(-1)
            object_in_hand=torch.stack([model.object_penetration_poses(p,object_points) for model,p in zip(models,actual)],-1).amax(-1)
            self_depth=torch.stack([model.self_penetration_poses(p) for model,p in zip(models,actual)],-1).amax(-1)
            m=t.m
            row=dict(object_pose_w=torch.cat((w.wp,w.wq),-1).clone(),bottom=m['bottom'].clone(),
                digit_normal_load_n=m['grouped_object_loads'][:,:,:2].clone(),hand_normal_load_n=m['hand_object_loads'].clone(),
                object_velocity=m['object_velocity'].clone(),hand_clearance_m=w.clearances.clone(),
                object_penetration_m=torch.maximum(hand_in_object,object_in_hand),self_penetration_m=self_depth,
                fault=probe.fault.clone(),reset=done.clone(),physics_interval_qualified=probe.good.clone(),
                observations=observation,actions=action.clone(),requested_hand_force_n=t.f_requested.clone(),
                object_weight_n=w.object_weight_n.clone(),vertical_components_n=w.vertical_components.clone(),
                support_normal_integral_n=w.surface_support_servo.normal_integral_n.clone(),
                support_tangent_integral_n=w.surface_support_servo.tangent_integral_n.clone(),
                controller_feedback_digit_load_n=(controller.force_feedback_loads if options.pinch_hold else
                    m['grouped_object_loads'][:,:,:2]).clone())
            for key in probe.samples[0]:row[key]=torch.stack([s[key] for s in probe.samples],1)
            frames.append(row);qualifier.update(row,1/60);completed=tick+1
            if tick%30==0:
                summary=qualifier.summary();summary.update(control_steps=completed,elapsed_s=time.monotonic()-begin,
                    loads=m['grouped_object_loads'][:,:,:2].cpu().tolist(),episodes=env.episodes.cpu().tolist(),
                    requested_normal_n=t.f_requested.cpu().tolist(),actual_normal_n=m['hand_object_loads'].cpu().tolist(),
                    total_fz_n=w.vertical_components.sum((1,2)).cpu().tolist(),weight_n=w.object_weight_n.cpu().tolist(),
                    support_target_fz_n=w.surface_support_diagnostic['target_total_fz_n'].cpu().tolist(),
                    actual_wrist_height_mm=((w.roots[:,:,2]-candidate_roots[:,:,2])*1000).cpu().tolist())
                training.write_json(args.output/'status.json',summary)
                print('SURFACE_PHYSICS',json.dumps(summary),flush=True)
            if qualifier.qualified.any() or time.monotonic()-begin>=options.wall_seconds:break
        trace={key:torch.stack([r[key] for r in frames]) for key in frames[0]}
        metadata=dict(active_dof_count=12,passive_dof_count=10,finger_dof_count=22,
            action_size=26,observation_size=t.observe().shape[-1],source_hashes=hashes,
            implementation_hashes=runtime_implementation,proposal_implementation_hashes=payload.get('implementation_hashes'),
            physics=dict(gravity=list(world.sim.cfg.gravity),object_mass_kg=float(w.object_weight_n[0]/9.81),
                object_material=args.object_material,hand_material=args.hand_material,effort_limit_nm=w.parameters.effort_limit_nm),
            contract=contract,sampling_hz=60,physics_substeps_per_action=w.decimation)
        metadata['surface_support']=contract['surface_support']
        # Retain every raw trial before any publisher validation can fail.
        torch.save({key:value.detach().cpu() for key,value in trace.items()},args.output/'all_trials.pt')
        training.write_json(args.output/'trial_initial_cases.json',initial)
        training.write_json(args.output/'trial_metadata.json',metadata)
        teachers=export_qualified(args.output/'qualified',trace,initial,metadata,dt=1/60)
        report=qualifier.summary();report.update(status='surface_teacher_validation_complete',control_steps=completed,
            elapsed_s=time.monotonic()-begin,teacher_files=[str(p) for p in teachers],ppo_updates=0,
            formal_successes=int(env.successes.sum()),episodes=env.episodes.cpu().tolist(),physics_unchanged=True,
            geometry_feasible_candidates_in_generator=payload['geometry_feasible_count'],
            started_from='collision_safe_pregrasp;not_qualified_grasp',native_fk_error=errors)
        training.write_json(args.output/'surface_teacher_result.json',report)
        print('SURFACE_TEACHER_RESULT',json.dumps(report),flush=True)
        sys.stdout.flush();os._exit(0)

    training.run_tensor_training=run
    sys.argv=[str(root/'scripts/train_dh116_tabletop_gpu_ppo.py'),
        '--pose-summary','configs/dh116_training/pose_reference.json','--adaptive-grasp','--force-feedback','--full-hand-contact',
        '--grasp-initialization','configs/dh116_training/grasp_initialization_below_axis_2mm_v1.json',
        '--load-profile','configs/dh116_load_profiles/v2_pressure_12n_instant_candidate.json',
        '--object-kind','solid_cylinder','--object-material','extreme_grip','--hand-material','grip_compliant',
        '--num-envs',str(len(rows)),'--rollout-steps','16','--max-iterations','1',
        '--physics-hz','480','--mimic-frequency','200','--solver-type','0','--position-iterations','64','--velocity-iterations','16',
        '--reset-height-jitter-m','0','--cpu-threads','6','--headless','--device','cuda:0',
        '--direct-training','--tensor-pipeline','--training-episode-resets','--pressure-relief',
        '--grasp-controller','policy','--palm-transport','driven_carriage','--palm-exploration-axis','xyz',
        '--passive-velocity-limit-rad-s','2.97','--contact-reward-window','close_start','--grasp-readiness','measured_load',
        '--skip-hold','--close-downward-penalty','--grasp-shaping','--vertical-force-reward',
        '--vertical-force-reward-mode','continuous','--force-reward-start','downward_after_close',
        '--force-region-control','--phase-force-reward','--free-grasp-lift','--continuous-motion','--free-wrist-grasp',
        '--output',str(options.output),'--kit_args=--allow-root']
    runpy.run_path(sys.argv[0],run_name='__main__')


if __name__=='__main__':
    try:entry()
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
