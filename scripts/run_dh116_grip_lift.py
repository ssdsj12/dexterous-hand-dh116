"""Close and keep joint targets, then physically lift; no force-goal gate."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time
import traceback
from types import SimpleNamespace
import torch


def awake_launcher():
    """Use the existing force-probe convention before any asset is spawned."""
    import isaaclab.app
    original=isaaclab.app.AppLauncher
    class AwakeLauncher(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            import isaaclab.sim as sim_utils
            root_type=sim_utils.ArticulationRootPropertiesCfg
            body_type=sim_utils.RigidBodyPropertiesCfg
            def root(**values):
                values['sleep_threshold']=0.;return root_type(**values)
            def body(**values):
                values['sleep_threshold']=0.;return body_type(**values)
            sim_utils.ArticulationRootPropertiesCfg=root
            sim_utils.RigidBodyPropertiesCfg=body
    isaaclab.app.AppLauncher=AwakeLauncher


def entry():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=10.)
    parser.add_argument('--wall-seconds',type=float,default=180.)
    parser.add_argument('--maximum-poses',type=int,default=4)
    parser.add_argument('--case-indices',type=int,nargs='+',help='replay selected expanded pose/closure cases')
    parser.add_argument('--reference',choices=('original','seed'),default='original')
    parser.add_argument('--closure-margins',type=float,nargs='+',default=[.08,.18])
    parser.add_argument('--thumb-spreads',type=float,nargs='+',default=[-.2,0.,.2,.4,.6])
    parser.add_argument('--mimic-frequency',type=int,choices=(200,400),default=400)
    parser.add_argument('--solve-articulation-contact-last',action='store_true',
        help='test contact-last solver ordering; require physics readback before stepping')
    parser.add_argument('--velocity-iterations',type=int,choices=(16,64),default=16,
        help='explicit numerical convergence comparison; default unchanged')
    parser.add_argument('--device',choices=('cuda:0',),default='cuda:0')
    options=parser.parse_args()
    if options.output.exists():raise FileExistsError('preserve existing physical evidence')
    if not 1<=options.maximum_poses<=8 or not 1<=options.seconds<=30 or not 10<=options.wall_seconds<=1800:
        raise ValueError('bounded positive diagnostic duration and pose count required')
    if (not all(math.isfinite(v) and 0<=v<=.3 for v in options.closure_margins)
            or not all(math.isfinite(v) and -.8<=v<=.8 for v in options.thumb_spreads)):
        raise ValueError('finite closure margins in0..0.3rad and thumb spreads in-0.8..0.8rad required')
    payload=json.loads(options.candidates.read_text())
    poses=[row for row in payload['candidates'] if max(row['geometry'][k] for k in
        ('penetration','object_in_hand','self'))<=.3 and row['geometry']['minimum_clearance_m']>=.00029][:options.maximum_poses]
    if not poses:raise ValueError('no collision-safe initial pose')
    combinations=[(margin,spread) for margin in options.closure_margins for spread in options.thumb_spreads]
    cases=[dict(pose=row,margin=margin,thumb_spread=spread) for row in poses for margin,spread in combinations]
    if options.case_indices is not None:
        if len(set(options.case_indices))!=len(options.case_indices) or any(i<0 or i>=len(cases) for i in options.case_indices):
            raise ValueError('unique case indices must refer to expanded candidates')
        cases=[cases[i] for i in options.case_indices]
    awake_launcher()
    from go2_pvcnn.control.dh116 import tabletop_tensor_training as training

    def run(world,template,args,contract):
        from pxr import PhysxSchema,UsdPhysics
        from go2_pvcnn.control.dh116.tabletop_tensor_world import TensorWorld
        from go2_pvcnn.control.dh116.tabletop_surface_kinematics import full_angles
        from go2_pvcnn.control.dh116.tabletop_surface_teacher import implementation_identity
        from go2_pvcnn.control.dh116.tabletop_grip_lift import closing_targets,advance_program,LiftDwell,valid_readback
        identity=implementation_identity(root)
        names=('d134680ab619_DH116_left.usd','3046ad2ed855_DH116_right.usd')
        assets={side:hashlib.sha256((root/'assets/fixed_dual_panda_dh116_v1/dependencies'/name).read_bytes()).hexdigest()
            for side,name in zip(('left','right'),names)}
        if assets!=payload['source_hashes']:raise ValueError('candidate source USD identity mismatch')
        w=TensorWorld(world);ids=torch.arange(w.n,device=w.device)
        drive_readback=[]
        for h,hand in enumerate(w.hands):
            forces=hand.root_physx_view.get_dof_max_forces()[:,world.ids[h]]
            velocities=hand.root_physx_view.get_dof_max_velocities()[:,world.ids[h]]
            if not torch.isfinite(forces).all() or ((forces-.2).abs()>1e-6).any():
                raise RuntimeError('native active drive torque limit readback mismatch')
            drive_readback.append(dict(max_force_nm=forces.cpu().tolist(),
                max_velocity_rad_s=velocities.cpu().tolist()))
        if getattr(world,'mounts',None) is None or any(len(mount.angular_ids)!=3 for mount in world.mounts):
            raise RuntimeError('physical six-axis wrist supports required')
        seed=w.t([row['pose']['active_angles'] for row in cases])
        roots=w.t([row['pose']['wrist_pose_w'] for row in cases]);roots[:,:,:3]+=w.origins[:,None]
        w.root_seed=roots.clone();w.initial_offset=0.;w.goal_roots=roots[:,:,:3].clone()
        for h in range(2):
            w.seed_q[h].zero_();w.seed_qd[h].zero_()
            w.seed_q[h][:,world.named_ids[h]]=full_angles(seed.reshape(w.n,2,6)[:,h]).float()
        def readable(measurement):
            return valid_readback(torch.cat((w.wp,w.wq),-1),w.wv,measurement['q'],measurement['qd'],
                measurement['hand_object_loads'],w.poses,w.qd,measurement['joint_failure'],
                w.adapter_fault_bits,w.clearances)
        m=w.reset(ids)
        if not readable(m).all():raise RuntimeError('initial reset readback invalid; no physics command submitted')
        if (m['q']-seed).abs().max()>2e-4:raise RuntimeError('initial joint readback mismatch')
        pose_error=float((w.roots[:,:,:3]-roots[:,:,:3]).abs().max())
        orientation_error=float((1-(w.roots[:,:,3:]*roots[:,:,3:]).sum(-1).abs()).abs().max())
        if pose_error>2e-5 or orientation_error>1e-5:raise RuntimeError('initial wrist pose readback mismatch')
        if ((w.bottom-.30).abs()>.0001).any():raise RuntimeError('object must start on the original table')
        sleep=[]
        for path in world.paths:
            for side in ('left','right'):
                p=world.sim.stage.GetPrimAtPath(path+'/'+side+'Hand/wrist_fixed')
                value=PhysxSchema.PhysxArticulationAPI(p).GetSleepThresholdAttr().Get()
                if value!=0.:raise RuntimeError('articulation sleep readback mismatch')
                sleep.append(value)
            p=world.sim.stage.GetPrimAtPath(path+'/Wire')
            if PhysxSchema.PhysxRigidBodyAPI(p).GetSleepThresholdAttr().Get()!=0.:
                raise RuntimeError('object sleep readback mismatch')
            if PhysxSchema.PhysxRigidBodyAPI(p).GetDisableGravityAttr().Get():raise RuntimeError('object gravity disabled')
        wire_paths={path+'/Wire' for path in world.paths}
        for prim in world.sim.stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                joint=UsdPhysics.Joint(prim)
                if any(str(p) in wire_paths for p in joint.GetBody0Rel().GetTargets()+joint.GetBody1Rel().GetTargets()):
                    raise RuntimeError('object has an attachment joint')
        initialization=json.loads(args.grasp_initialization.read_text())
        margin=w.t([row['margin'] for row in cases]);spread=w.t([row['thumb_spread'] for row in cases])
        reference=(seed.clone() if options.reference=='seed' else
            w.t(initialization['grasp_command']).expand(w.n,-1).clone())
        reference[:,[0,6]]+=spread.clamp_max(0.)[:,None]
        closed=closing_targets(seed,reference,margin,thumb_spread=spread.clamp_min(0.))
        task=SimpleNamespace(policy_control=True,free_wrist_grasp=True,pinch_force_hold=True,
            command=seed.clone(),palm_targets=torch.zeros(w.n,2,3,device=w.device,dtype=torch.float64),
            palm_actions=torch.zeros(w.n,2,3,device=w.device,dtype=torch.float64),
            rotation_targets=torch.zeros(w.n,2,3,device=w.device,dtype=torch.float64),
            policy_stiffness=torch.full((w.n,2),10.,device=w.device,dtype=torch.float64),
            offset=torch.zeros(w.n,device=w.device,dtype=torch.float64))
        elapsed=torch.zeros(w.n,device=w.device,dtype=torch.float64)
        episodes=torch.zeros(w.n,device=w.device,dtype=torch.long);streak=episodes.clone()
        dwell=LiftDwell(w.n,w.device);live=torch.ones(w.n,device=w.device,dtype=torch.bool)
        raw=[];control=[];fault_events=[];begin=time.monotonic();error=None;steps=0
        first_success=torch.full((w.n,),-1,device=w.device,dtype=torch.long)
        contract=dict(scene_setup=contract,policy_controller_executed=False,force_controller_executed=False,
            mode='joint_close_hold_then_measured_lift',force_goal=None,
            finger_active_dofs=12,finger_passive_dofs=10,finger_total_dofs=22,external_wrist_drives=12,
            force_limits_used_as_success_gate=False,force_error_used_as_success_gate=False,
            close_s=2.,lift_start_s=2.5,lift_speed_m_s=.01,lift_goal_m=.05,
            acceptance='actual_bottom>=table+30mm;real_contact;speed<0.03m/s;continuous0.5s',
            sleep_disabled=True,source_identity_lf=identity,source_asset_sha256=assets,ppo_updates=0,
            link_body_names=[list(hand.body_names) for hand in w.hands],
            native_active_drive_readback=drive_readback,
            applied_torque_field='implicit actuator estimate;not measured PhysX solver torque',
            source_case_indices=options.case_indices,
            closure_reference=options.reference,candidate_input_sha256=hashlib.sha256(options.candidates.read_bytes()).hexdigest(),
            initial_wrist_position_error_m=pose_error,initial_wrist_quaternion_error=orientation_error)
        training.write_json(args.output/'resolved_grip_lift.json',contract)
        initial=[dict(active_angles=seed[i].cpu().tolist(),closed_targets=closed[i].cpu().tolist(),
            environment_origin_w=w.origins[i].cpu().tolist(),
            wrist_pose_w=roots[i].cpu().tolist(),object_pose_w=w.seed_wire[i].cpu().tolist(),
            closure_margin_rad=cases[i]['margin'],thumb_spread_rad=cases[i]['thumb_spread'],
            candidate_geometry=cases[i]['pose']['geometry']) for i in range(w.n)]
        training.write_json(args.output/'initial_cases.json',initial)
        try:
            for step in range(round(options.seconds/w.dt)):
                elapsed[live]+=w.dt
                previous=task.command.clone()
                task.command,z=advance_program(task.command,seed,closed,elapsed,w.dt,
                    lift_speed_m_s=.01,lift_height_m=.05)
                task.palm_targets.zero_();task.palm_targets[:,:,2]=z[:,None]
                m=w.advance(task,live,live)
                actual_pose=torch.cat((w.wp,w.wq),-1)
                bad=~readable(m)
                reset=bad&live
                valid=live&~bad
                dwell.update(actual_pose,w.wv,m['hand_object_loads'],valid,reset,w.dt)
                newly=dwell.qualified&(first_success<0);first_success[newly]=step
                raw.append(dict(object_pose_w=actual_pose.clone(),object_velocity=w.wv.clone(),bottom=w.bottom.clone(),
                    q_target=task.command.clone(),
                    active_applied_torque_estimate_nm=torch.cat([hand.data.applied_torque[:,world.ids[h]]
                        for h,hand in enumerate(w.hands)],1).double().clone(),
                    q_measured=m['q'].clone(),qd_measured=m['qd'].clone(),full_link_pose_w=w.poses.clone(),
                    full_joint_velocity=w.qd.clone(),joint_failure=m['joint_failure'].clone(),
                    adapter_fault_bits=w.adapter_fault_bits.clone(),
                    hand_normal_load_n=m['hand_object_loads'].clone(),digit_normal_load_n=m['grouped_object_loads'][:,:,:2].clone(),
                    vertical_components_n=w.vertical_components.clone(),hand_clearance_m=w.clearances.clone(),
                    valid=valid.clone(),reset=reset.clone(),episode=episodes.clone(),elapsed_s=elapsed.clone()))
                if step%w.decimation==0:
                    control.append(dict(q_target=task.command.clone(),q_measured=m['q'].clone(),qd_measured=m['qd'].clone(),
                        mimic_error=m['mimic_errors'].clone(),
                        root_target=w.root_targets.clone(),root_measured=w.roots.clone(),
                        requested_joint_rate=(task.command-previous)/w.dt))
                if reset.any():
                    selected=reset.nonzero().flatten();streak[selected]+=1
                    fault_events.append(dict(step=step,ids=selected.cpu().tolist(),
                        joint_failure=m['joint_failure'][selected].cpu().tolist(),adapter_bits=w.adapter_fault_bits[selected].cpu().tolist()))
                    recovered=w.reset(selected);task.command[selected]=seed[selected];task.palm_targets[selected]=0.
                    elapsed[selected]=0.;episodes[selected]+=1
                    if not readable(recovered)[selected].all():
                        # PhysX shares one scene: an invalid articulation cannot be
                        # safely hidden by merely masking its next control target.
                        raise RuntimeError('reset failed readback; stopped before submitting another physics step')
                streak[valid]=0
                steps=step+1
                if step%240==0:
                    status=dwell.summary();status.update(physics_steps=steps,elapsed_wall_s=time.monotonic()-begin,
                        actual_normal_n=m['hand_object_loads'].cpu().tolist(),episodes=episodes.cpu().tolist())
                    training.write_json(args.output/'status.json',status)
                    print('GRIP_LIFT',json.dumps(status),flush=True)
                if not live.any() or time.monotonic()-begin>=options.wall_seconds or (args.output/'stop_requested').exists():break
        except BaseException as failure:
            error=repr(failure);traceback.print_exc()
        trace={key:torch.stack([row[key] for row in raw]).detach().cpu() for key in raw[0]} if raw else {}
        controls={key:torch.stack([row[key] for row in control]).detach().cpu() for key in control[0]} if control else {}
        torch.save(trace,args.output/'raw_physics.pt');torch.save(controls,args.output/'control_trace.pt')
        # Independently replay raw evidence before writing any success case.
        replay=LiftDwell(w.n,'cpu');first_replay={}
        for i in range(len(raw)):
            audit=valid_readback(trace['object_pose_w'][i],trace['object_velocity'][i],
                trace['q_measured'][i],trace['qd_measured'][i],trace['hand_normal_load_n'][i],
                trace['full_link_pose_w'][i],trace['full_joint_velocity'][i],
                trace['joint_failure'][i],trace['adapter_fault_bits'][i],trace['hand_clearance_m'][i])
            if not torch.equal(audit&~trace['reset'][i],trace['valid'][i]):
                raise RuntimeError('raw guard replay mismatch')
            replay.update(trace['object_pose_w'][i],trace['object_velocity'][i],trace['hand_normal_load_n'][i],
                trace['valid'][i],trace['reset'][i],w.dt)
            for index in replay.qualified.nonzero().flatten().tolist():
                if index not in first_replay:
                    first_replay[index]=dict(global_physics_step=i,episode=int(trace['episode'][i,index]),
                        episode_elapsed_s=float(trace['elapsed_s'][i,index]),
                        episode_physics_steps=round(float(trace['elapsed_s'][i,index])/w.dt),
                        replay_stop='start_from_initial_case;execute_program_for_episode_physics_steps;hold_last_targets')
        successful=replay.ever_qualified.nonzero().flatten().tolist()
        for index in successful:
            training.write_json(args.output/f'lift_case_{index:03d}.json',dict(
                qualified_measured_lift=True,force_target_required=False,initial_case=initial[index],
                teacher_eligible=None,geometry_review_required=True,
                evidence='raw_physics.pt',case_index=index,physics_sampling_hz=args.physics_hz,
                first_qualified=first_replay[index],
                retained_at_end=bool(replay.qualified[index]),
                source_identity_lf=identity,acceptance=contract['acceptance'],
                use='replayable_joint_target_motion_program;not_plain_PPO_BC',
                mass_kg=float(w.object_weight_n[index]/9.81),gravity=list(world.sim.cfg.gravity)))
        result=replay.summary();result.update(status='grip_lift_complete' if error is None else 'grip_lift_interrupted',
            error=error,physics_steps=steps,elapsed_wall_s=time.monotonic()-begin,
            successful_cases=successful,episodes=episodes.cpu().tolist(),fault_events=fault_events,
            force_goal=None,ppo_updates=0,sleep_disabled=True,
            unchanged='object_mass_gravity_materials_table_collision_and_actuator_angle_speed_torque_limits')
        training.write_json(args.output/'grip_lift_result.json',result)
        print('GRIP_LIFT_RESULT',json.dumps(result),flush=True);sys.stdout.flush();os._exit(0 if error is None else 1)

    training.run_tensor_training=run
    sys.argv=[str(root/'scripts/train_dh116_tabletop_gpu_ppo.py'),
        '--pose-summary','configs/dh116_training/pose_reference.json','--adaptive-grasp','--force-feedback','--full-hand-contact',
        '--grasp-initialization','configs/dh116_training/grasp_initialization_below_axis_2mm_v1.json',
        '--load-profile','configs/dh116_load_profiles/v2_pressure_12n_instant_candidate.json',
        '--object-kind','solid_cylinder','--object-material','extreme_grip','--hand-material','grip_compliant',
        '--num-envs',str(len(cases)),'--rollout-steps','16','--max-iterations','1',
        '--physics-hz','480','--mimic-frequency',str(options.mimic_frequency),'--solver-type','0','--position-iterations','64','--velocity-iterations',str(options.velocity_iterations),
        '--reset-height-jitter-m','0','--cpu-threads','6','--headless','--device',options.device,
        '--direct-training','--tensor-pipeline','--training-episode-resets','--pressure-relief',
        '--grasp-controller','policy','--palm-transport','driven_carriage','--palm-exploration-axis','xyz',
        '--passive-velocity-limit-rad-s','2.97','--contact-reward-window','close_start','--grasp-readiness','measured_load',
        '--skip-hold','--close-downward-penalty','--grasp-shaping','--vertical-force-reward',
        '--vertical-force-reward-mode','continuous','--force-reward-start','downward_after_close',
        '--force-region-control','--phase-force-reward','--free-grasp-lift','--continuous-motion','--free-wrist-grasp',
        '--output',str(options.output),'--kit_args=--allow-root']
    if options.solve_articulation_contact_last:
        sys.argv.append('--solve-articulation-contact-last')
    runpy.run_path(sys.argv[0],run_name='__main__')


if __name__=='__main__':
    try:entry()
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
