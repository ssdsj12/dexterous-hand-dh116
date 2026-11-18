"""Isaac translation-only probe; fixed grasp joint targets, no PPO learning."""
import json
from pathlib import Path
import runpy
import sys
import torch
from go2_pvcnn.control.dh116 import tabletop_tensor_training as training
from go2_pvcnn.control.dh116 import tabletop_policy_force_region as force_region


def fixed_targets(task,target,requested,live,dt):
    task.f_target[live]=requested[live];task.f_requested[live]=requested[live]
    load=task.m['hand_object_loads']
    severe=task.m['mimic_errors'].reshape(-1,2,5).abs().amax(-1)>=.012
    change=(.4*(requested-load)*dt).clamp(-.12*dt,.12*dt)
    change=torch.where(severe&(change>0),0.,change)
    task.f_offset[live]=(task.f_offset+change).clamp(-.08,.08)[live]
    return target+(task.f_offset[:,:,None]*task.direction).flatten(1)


def probe(world,template,args,contract):
    from go2_pvcnn.control.dh116.tabletop_tensor_world import TensorWorld
    from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
    from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv
    from go2_pvcnn.control.dh116.tabletop_free_grasp_lift import enable
    from go2_pvcnn.control.dh116.tabletop_control_probe import joint_actions
    w=TensorWorld(world)
    t=TensorTask(template,args.num_envs,args.device,pressure_relief=True,
        contact_reward_window='close_start',grasp_readiness='measured_load',
        vertical_force_reward=True,vertical_force_reward_mode='continuous',
        force_reward_start='downward_after_close',skip_hold=True,
        close_downward_penalty=True,grasp_shaping=True,palm_exploration_axis='xyz')
    t.enable_policy_control();force_region.enable(t);enable(t)
    env=TensorEnv(w,t,recovery=True,training_joint_limit=True)
    env.reset()
    force_region.adjust=fixed_targets
    cases=[(x,z) for x in (-.5,0.,.5) for z in (-.5,0.,.5)]
    if args.num_envs!=len(cases):raise ValueError('probe requires 9 environments')
    a=torch.zeros((args.num_envs,20),device=args.device)
    a[:,:12]=joint_actions(t.reference,t.lower,t.center,t.upper).to(a)
    a[:,18:]=-.4
    offsets=torch.tensor(cases,device=args.device,dtype=a.dtype)/1000.
    a[:,12]=offsets[:,0]/.003
    a[:,14]=offsets[:,1]/torch.where(offsets[:,1]>=0,.05,.004)
    a[:,15]=0.
    a[:,17]=.0005/.05
    loads=[];zs=[];below=[];valid=[];failures=[];actual_offsets=[];groups=[];normals=[]
    for step in range(180):
        _,_,done,extra=env.step(a)
        if step>=150:
            loads.append(t.m['hand_object_loads'].clone())
            centers=torch.where(t.m['contact_valid'][:,:,None],t.m['contact_center_relative_m'],0.)
            zs.append(force_region._radial_z(centers,t.m['object_quat']).clone())
            below.append(t.m['contact_all_below_centerline'].clone())
            valid.append(t.m['contact_valid'].clone())
            failures.append(done.clone());actual_offsets.append(t.m['adaptive_palm_offset'].clone())
            groups.append(t.m['grouped_object_loads'].clone());normals.append(t.m['vertical_hand_force_n'].clone())
        if step%30==29:
            print('POSITION_PROBE_STEP',step+1,'load',t.m['hand_object_loads'].mean(0).cpu().tolist(),
                'offset',t.f_offset.mean(0).cpu().tolist(),flush=True)
    mean_load=torch.stack(loads).mean(0).cpu()
    valid=torch.stack(valid);zs=torch.stack(zs)
    count=valid.sum(0).clamp_min(1)
    contact_z=torch.where(valid,zs,0.).sum(0)/count
    lower=torch.stack(below).double().mean(0).cpu()
    seen=valid.double().mean(0).cpu()
    actual=torch.stack(actual_offsets).mean(0).cpu()
    report=dict(fixed_joint_target_rad=t.reference[0].cpu().tolist(),
        effort_limit_nm=t.parameters.effort_limit_nm,steps=180,micro_closing_feedback=True,
        failures=torch.stack(failures).sum(0).cpu().tolist(),cases=[])
    report['sensors']=[dict(path=s.cfg.prim_path,filter=s.cfg.filter_prim_paths_expr,
        net_force_first_env_n=s.data.net_forces_w[0].cpu().tolist(),
        object_force_first_env_n=s.data.force_matrix_w[0].cpu().tolist()) for s in world.sensors]
    report['joint_tracking_max_rad']=(t.command-t.m['q']).abs().amax(1).cpu().tolist()
    report['minimum_clearance_m']=t.m['clearance'].cpu().tolist()
    report['force_offsets_rad']=t.f_offset.cpu().tolist()
    report['mimic_max_rad']=t.m['mimic_errors'].reshape(-1,2,5).abs().amax(-1).cpu().tolist()
    report['raw_contacts']=[]
    for si,sensor in enumerate(world.sensors):
        magnitude,point,normal,separation,count,start=sensor.contact_physx_view.get_contact_data(dt=w.dt)
        for i in range(args.num_envs):
            k=int(count.reshape(args.num_envs,-1)[i,0]);lo=int(start.reshape(args.num_envs,-1)[i,0])
            if k:
                p=point[lo:lo+k].double()-w.wp[i]
                report['raw_contacts'].append(dict(case=i,sensor=sensor.cfg.prim_path,
                    relative_point_m=p.cpu().tolist(),normal_on_hand=normal[lo:lo+k].cpu().tolist(),
                    magnitude_n=magnitude[lo:lo+k].cpu().tolist()))
    for i,(x,z) in enumerate(cases):
        report['cases'].append(dict(index=i,palm_x_mm=x,palm_z_mm=z,
            load_n=mean_load[i].tolist(),contact_z_mm=(contact_z[i]*1000).cpu().tolist(),
            below_fraction=lower[i].tolist(),contact_fraction=seen[i].tolist(),
            grouped_loads_n=torch.stack(groups).mean(0)[i].cpu().tolist(),
            normal_fz_n=torch.stack(normals).mean(0)[i].cpu().tolist(),
            actual_palm_offsets_mm=(actual[i]*1000).tolist()))
    (args.output/'fixed_pose_position_probe.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    for hi,side in enumerate(('left','right')):
        order=sorted(report['cases'],key=lambda r:(-r['below_fraction'][hi],
            abs(r['contact_z_mm'][hi]+2)+abs(r['load_n'][hi]-5)))
        print('BEST_FIXED_POSE',side,json.dumps(order[:4]),flush=True)
    return dict(status='fixed_pose_translation_probe_complete',learning_performed=False)


training.run_tensor_training=probe
sys.argv=['scripts/train_dh116_tabletop_gpu_ppo.py',
    '--pose-summary','configs/dh116_training/pose_reference.json','--adaptive-grasp','--force-feedback','--full-hand-contact',
    '--joint-envelope','configs/dh116_joint_envelopes/free_grasp_geometry_v1.json',
    '--grasp-initialization','configs/dh116_training/grasp_initialization_below_axis_2mm_v1.json',
    '--require-initial-below-axis','--load-profile','configs/dh116_load_profiles/manual_fixed_pose_04nm.json',
    '--object-kind','solid_cylinder','--object-material','extreme_grip','--hand-material','grip_compliant',
    '--num-envs','9','--rollout-steps','24','--max-iterations','1','--physics-hz','480','--mimic-frequency','200',
    '--position-iterations','64','--velocity-iterations','16','--reset-height-jitter-m','0','--cpu-threads','6',
    '--headless','--device','cuda:0','--direct-training','--tensor-pipeline','--training-episode-resets','--pressure-relief',
    '--grasp-controller','policy','--palm-transport','driven_carriage','--palm-exploration-axis','xyz',
    '--passive-velocity-limit-rad-s','2.97','--contact-reward-window','close_start','--grasp-readiness','measured_load',
    '--skip-hold','--close-downward-penalty','--grasp-shaping','--vertical-force-reward',
    '--vertical-force-reward-mode','continuous','--force-reward-start','downward_after_close',
    '--free-grasp-lift','--stable-lift-reward','--force-region-control','--phase-force-reward',
    '--output','runs/fixed_orientation_force_probe_v2','--kit_args=--allow-root']
runpy.run_path(sys.argv[0],run_name='__main__')
