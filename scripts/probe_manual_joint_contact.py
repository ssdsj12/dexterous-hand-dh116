"""Single Isaac process, six placement diagnostics; no PPO or optimizer."""
import json
from pathlib import Path
import runpy
import sys
import torch
import argparse
import isaaclab.app
OriginalLauncher=isaaclab.app.AppLauncher
class CpuDiagnosticLauncher(OriginalLauncher):
    def __init__(self,args,*extra,**kwargs):
        args.device='cpu'
        super().__init__(args,*extra,**kwargs)
        import isaaclab.sim as sim_utils
        old_root=sim_utils.ArticulationRootPropertiesCfg
        old_body=sim_utils.RigidBodyPropertiesCfg
        def awake_root(**values):
            values['sleep_threshold']=0.
            return old_root(**values)
        def awake_body(**values):
            values['sleep_threshold']=0.
            return old_body(**values)
        sim_utils.ArticulationRootPropertiesCfg=awake_root
        sim_utils.RigidBodyPropertiesCfg=awake_body
isaaclab.app.AppLauncher=CpuDiagnosticLauncher
from go2_pvcnn.control.dh116 import tabletop_tensor_training as training


def raw_case(env,index):
    from go2_pvcnn.control.dh116.tabletop_manual_digit_contacts import summarize
    from go2_pvcnn.control.dh116.tabletop_object_contact import link_groups
    w=env.world;groups=link_groups(w.w.object_contact_links)
    loads=env.task.m['object_link_loads'][index].detach().cpu().numpy()
    records=[];grouped={}
    for hi,hand in enumerate(('left','right')):
        grouped[hand]={}
        for gi,digit in enumerate(('thumb','index')):
            grouped[hand][digit]=float(loads[hi,groups==gi].sum())
            for li in (groups==gi).nonzero()[0]:
                if loads[hi,li]<.05:continue
                _,point,_,sep,count,start=w.w.sensors[hi*len(groups)+int(li)].contact_physx_view.get_contact_data(dt=w.dt)
                k=int(count.reshape(w.n,-1)[index,0]);lo=int(start.reshape(w.n,-1)[index,0])
                for p,s in zip(point[lo:lo+k].cpu().tolist(),sep[lo:lo+k].flatten().cpu().tolist()):
                    records.append(dict(hand=hand,digit=digit,position_w=p,separation_m=s))
    qw,qx,qy,qz=w.wq[index].cpu().tolist()
    axis=[2*(qx*qy-qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz+qw*qx)]
    return summarize(records,grouped,w.wp[index].cpu().tolist(),axis)


def probe(world,template,args,contract):
    actual_solver=world.sim.get_physics_context().get_solver_type()
    if actual_solver!='PGS':raise RuntimeError(f'PGS diagnostic readback failed: {actual_solver!r}')
    from pxr import PhysxSchema
    scenes=[prim for prim in world.sim.stage.Traverse() if prim.GetTypeName()=='PhysicsScene']
    if len(scenes)!=1:raise RuntimeError('one native physics scene required')
    gpu=PhysxSchema.PhysxSceneAPI(scenes[0]).GetEnableGPUDynamicsAttr().Get()
    if gpu or args.device!='cpu':raise RuntimeError('CPU physics readback mismatch')
    for path in world.paths:
        for side in ('left','right'):
            api=PhysxSchema.PhysxArticulationAPI(world.sim.stage.GetPrimAtPath(path+'/'+side+'Hand/wrist_fixed'))
            if api.GetSleepThresholdAttr().Get()!=0.:raise RuntimeError('articulation sleep threshold readback mismatch')
        if PhysxSchema.PhysxRigidBodyAPI(world.sim.stage.GetPrimAtPath(path+'/Wire')).GetSleepThresholdAttr().Get()!=0.:
            raise RuntimeError('object sleep threshold readback mismatch')
    from go2_pvcnn.control.dh116.tabletop_tensor_world import TensorWorld
    from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
    from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv
    from go2_pvcnn.control.dh116.tabletop_policy_force_region import enable as enable_force
    from go2_pvcnn.control.dh116.tabletop_manual_digit_force_balance import enable as enable_digits
    from go2_pvcnn.control.dh116.tabletop_control_probe import joint_actions
    from go2_pvcnn.control.dh116.tabletop_manual_thumb_stability import ThumbStability
    w=TensorWorld(world)
    t=TensorTask(template,args.num_envs,args.device,pressure_relief=True,
        contact_reward_window='close_start',grasp_readiness='measured_load',
        vertical_force_reward=True,vertical_force_reward_mode='continuous',
        force_reward_start='downward_after_close',skip_hold=True,
        close_downward_penalty=True,grasp_shaping=True,palm_exploration_axis='xyz')
    t.enable_policy_control();enable_force(t);enable_digits(t)
    from go2_pvcnn.control.dh116.tabletop_manual_pinch_geometry import enable as enable_pinch
    initialization=json.loads(args.grasp_initialization.read_text())
    t.manual_control=True;t.manual_fast_force=True;t.manual_lightweight=True
    t.manual_lift_qualified=torch.zeros(t.n,device=t.device,dtype=torch.bool)
    t.last_observation=t.observe().clone()
    env=TensorEnv(w,t,recovery=True,training_joint_limit=True);env.reset()
    from go2_pvcnn.control.dh116.tabletop_manual_joint_contact import JointContactController
    controller=JointContactController(t,w)
    # Audit the actual PhysX COM-to-contact shift and mimic folding before
    # applying a controller. Kinematic finite differences do not step physics.
    base_model,base_points=controller.contacts()
    differences=[]
    from go2_pvcnn.control.dh116.tabletop_tensor_world import rotated_points
    for hi,hand in enumerate(world.hands):
        base=hand.data.joint_pos.clone()
        for ji in range(3):
            samples=[]
            for sign in (-1,1):
                pose=base+sign*.001*controller.folds[hi][:,ji].float()[None]
                hand.write_joint_state_to_sim(pose,torch.zeros_like(pose))
                world.sim.physics_sim_view.update_articulations_kinematic()
                w.refresh(sensors=False)
                p=(w.pad_relative+w.wp[:,None]).reshape(t.n,2,2,3)[:,hi].clone()
                samples.append(p)
            fd=(samples[1]-samples[0])/.002
            expected=base_model[:,hi,:,:,3+ji]
            differences.append(dict(hand=hi,joint=ji,max_error_m_rad=float((fd-expected).abs().max()),
                                    finite_difference=fd.tolist(),native_shifted=expected.tolist()))
        hand.write_joint_state_to_sim(base,torch.zeros_like(base))
        world.sim.physics_sim_view.update_articulations_kinematic()
        w.refresh(sensors=False)
    (args.output/'jacobian_audit.json').write_text(json.dumps(differences,indent=2))
    print('JACOBIAN_AUDIT',json.dumps(differences),flush=True)
    if max(row['max_error_m_rad'] for row in differences)>.0002:
        raise RuntimeError('contact Jacobian finite-difference validation failed')
    env.reset()
    right_z=[0.]
    gains=[.005]
    requested_forces=[5.]
    armatures=[0.]
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS
    for hand in world.hands:
        ids=[hand.joint_names.index(name) for name in JOINTS]
        values=torch.tensor(armatures,device=t.device,dtype=torch.float32)[:,None].expand(-1,len(ids)).clone()
        hand.write_joint_armature_to_sim(values,joint_ids=ids)
        actual=hand.root_physx_view.get_dof_armatures().to(values.device)[:,ids]
        if not torch.allclose(actual,values,atol=1e-9,rtol=0):raise RuntimeError('joint armature readback mismatch')
    stiffnesses=[10.]
    t.manual_digit_force_gain=t.command.new_tensor(gains)[:,None,None]
    a=torch.zeros((t.n,20),device=t.device)
    a[:,:12]=joint_actions(t.reference,t.lower,t.center,t.upper)
    a[:,18:]=(torch.tensor(requested_forces,device=t.device)[:,None]-2.)/5.-1.
    palms=torch.zeros((t.n,2,3),device=t.device)
    a[:,12:18]=(palms/torch.where(palms>=0,palms.new_tensor([.003,.003,.05]),palms.new_tensor([.003,.003,.004]))).flatten(1)
    trackers=[ThumbStability() for _ in right_z]
    history=[];trace=[];original=w.advance
    def recorded_advance(task,allowed,live):
        m=original(task,allowed,live)
        trace.append(torch.cat((m['grouped_object_loads'][:,:,:2].flatten(1),m['mimic_errors'],
            m['q'],task.joint_governed_target,task.manual_digit_offsets.flatten(1)),1).clone())
        return m
    w.advance=recorded_advance
    original_control=t.control
    frozen=None
    def control(action,live,dt):
        if frozen is None:return original_control(action,live,dt)
        t.command[live]=frozen[live]
        t.palm_targets[live]=palms.to(t.palm_targets)[live]
        t.policy_stiffness[live]=t.command.new_tensor(stiffnesses)[:,None].expand(-1,2)[live]
        return live
    t.control=control
    try:
        for step in range(600):
            _,_,done,_=env.step(a)
            loads=t.m['grouped_object_loads'][:,:,:2].cpu().tolist()
            for i,tracker in enumerate(trackers):
                if bool(done[i]):tracker.reset()
                else:tracker.update(loads[i],1/60)
            net=torch.stack([s.data.net_forces_w[:,0,:]-s.data.force_matrix_w[:,0,0,:] for s in world.sensors],1).double().reshape(t.n,2,-1,3)
            other_groups=net.norm(dim=-1)@w.group_matrix
            history.append(torch.cat((t.m['grouped_object_loads'][:,:,:2].flatten(1),
                t.m['adaptive_palm_offset'].flatten(1),t.m['mimic_errors'],t.m['bottom'][:,None]-.30,
                other_groups[:,:,:2].flatten(1),t.m['hand_clearance']),1).clone())
            if step%30==29:
                print('THUMB_PROBE_STEP',step+1,'loads',loads,'resets',env.episodes.cpu().tolist(),
                      'contactZ',controller.diagnostic.get('contact_radial_z_m'),
                      'palm',t.m['adaptive_palm_offset'].tolist(),flush=True)
        samples=torch.stack(history[-60:]).cpu()
        report=dict(learning_performed=False,steps=600,solver_readback=actual_solver,physics_device='cpu',gpu_dynamics=False,sleep_disabled=True,cases=[],
                    joint_controller={key:value.cpu().tolist() for key,value in controller.diagnostic.items()})
        for i,z in enumerate(right_z):
            forces=samples[:,i,:4].reshape(-1,2,2)
            ratio=forces.amax(-1)/forces.amin(-1).clamp_min(1e-9)
            contact=raw_case(env,i)
            report['cases'].append(dict(index=i,right_z_mm=z*1000,force_gain=gains[i],requested_hand_force_n=requested_forces[i],armature_kg_m2=armatures[i],failures=int(env.episodes[i]),
                load_mean_n=forces.mean(0).tolist(),thumb_range_n=(forces[:,:,0].amax(0)-forces[:,:,0].amin(0)).tolist(),
                ratio_max=ratio.amax(0).tolist(),ratio_pass_fraction=(ratio<=1.5).double().mean(0).tolist(),
                actual_palm_offsets_mm=(t.m['adaptive_palm_offset'][i]*1000).cpu().tolist(),
                actual_joint_rad=t.m['q'][i].tolist(),commanded_joint_rad=t.command[i].tolist(),
                requested_joint_rad=t.reference[i].tolist(),
                contact_body_names=[[world.hands[h].body_names[int(controller.loaded_link_ids[i,h,d])] for d in range(2)] for h in range(2)],
                other_contact_mean_n=samples[:,i,21:25].mean(0).reshape(2,2).tolist(),
                hand_clearance_min_mm=(samples[:,i,25:27].amin(0)*1000).tolist(),
                mimic_max_rad=float(t.m['mimic_errors'][i].abs().max()),
                digit_contacts=contact,thumb_stability=trackers[i].result,
                final_offset_rad=t.manual_digit_offsets[i].cpu().tolist(),
                grasp_stable=all(trackers[i].result['thumb_force_stable']) and bool((ratio[-1]<=1.5).all()) and contact['all_four_digits_below_axis']))
        (args.output/'thumb_probe.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        for row in report['cases']:print('THUMB_RESULT',json.dumps(row),flush=True)
        return dict(status='manual_thumb_probe_complete',learning_performed=False)
    finally:
        torch.save(dict(samples=torch.stack(trace).cpu(),fields=['digit_loads4','mimic10','q12','governed_target12','offset4']),args.output/'thumb_physics_trace.pt')
        if hasattr(env,'manual_last_failure'):
            (args.output/'last_probe_failure.json').write_text(json.dumps(env.manual_last_failure,indent=2,allow_nan=False))


training.run_tensor_training=probe
# Diagnostic-only CLI extension. Production training remains restricted to PGS.
original_add_argument=argparse.ArgumentParser.add_argument
def diagnostic_argument(self,*names,**kwargs):
    if '--solver-type' in names:kwargs['choices']=(0,1)
    return original_add_argument(self,*names,**kwargs)
argparse.ArgumentParser.add_argument=diagnostic_argument
sys.argv=['scripts/train_dh116_tabletop_gpu_ppo.py',
    '--pose-summary','configs/dh116_training/pose_reference.json','--adaptive-grasp','--force-feedback','--full-hand-contact',
    '--joint-envelope','configs/dh116_joint_envelopes/free_grasp_geometry_v1.json',
    '--grasp-initialization','configs/dh116_training/grasp_initialization_below_axis_2mm_v1.json',
    '--require-initial-below-axis','--load-profile','configs/dh116_load_profiles/manual_fixed_pose_04nm.json',
    '--object-kind','solid_cylinder','--object-material','extreme_grip','--hand-material','grip_compliant',
    '--num-envs','1','--rollout-steps','24','--max-iterations','1','--physics-hz','480','--mimic-frequency','0',
    '--solver-type','0','--position-iterations','64','--velocity-iterations','16','--reset-height-jitter-m','0','--cpu-threads','6',
    '--headless','--device','cuda:0','--direct-training','--tensor-pipeline','--training-episode-resets','--pressure-relief',
    '--grasp-controller','policy','--palm-transport','driven_carriage','--palm-exploration-axis','xyz',
    '--passive-velocity-limit-rad-s','2.97','--contact-reward-window','close_start','--grasp-readiness','measured_load',
    '--skip-hold','--close-downward-penalty','--grasp-shaping','--vertical-force-reward',
    '--vertical-force-reward-mode','continuous','--force-reward-start','downward_after_close',
    '--force-region-control','--policy-lift-hold',
    '--output','runs/manual_joint_contact_rigid_v4','--kit_args=--allow-root']
runpy.run_path(sys.argv[0],run_name='__main__')
