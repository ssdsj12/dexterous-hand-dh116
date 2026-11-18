"""Bounded native motion trial through the production 24D surface controller.

No force goal, no PPO update, no object manipulation after reset. All trial
results remain unqualified until independent raw stability/geometry replay.
"""
from pathlib import Path
import argparse,json,hashlib,os,runpy,sys,time,traceback
import numpy as np
import torch

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
os.chdir(root)
parser=argparse.ArgumentParser()
parser.add_argument('--candidates',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--seconds',type=float,default=10.)
parser.add_argument('--wall-seconds',type=float,default=660.)
parser.add_argument('--record-invalid-pad-buffers',action='store_true')
parser.add_argument('--record-contact-atlas',action='store_true')
parser.add_argument('--source-freeze',default='not_provided')
parser.add_argument('--candidate-index',type=int,default=0)
parser.add_argument('--close-seconds',type=float,default=2.)
parser.add_argument('--lift-speed',type=float,default=.006)
parser.add_argument('--preloads-mm',type=float,nargs=2,default=[.15,.3])
parser.add_argument('--batched-contacts',action='store_true')
options=parser.parse_args()
if options.output.exists():raise FileExistsError(options.output)
if not 1<=options.seconds<=18 or not 30<=options.wall_seconds<=1200:raise ValueError('bounded trial required')
if not 1<=options.close_seconds<=5 or not 0<options.lift_speed<=.01:raise ValueError('bounded motion required')
payload=json.loads(options.candidates.read_text())
row=payload['candidates'][options.candidate_index]
if not row['static_wrench']['static_feasible'] or payload['support_clearance_m']!=.06:
    raise ValueError('strict pad/support/wrench-screened candidate required')
options.output.parent.mkdir(parents=True,exist_ok=True)
from go2_pvcnn.control.dh116.tabletop_surface_kinematics import full_angles
initial=json.loads((root/'configs/dh116_training/grasp_initialization_below_axis_2mm_v1.json').read_text())
roots=np.array(row['wrist_pose_w']);roots[:,2]-=.06
initial['roots']=roots.tolist()
initial['joint_positions']=full_angles(torch.tensor(row['active_angles'],dtype=torch.float64).reshape(2,6)).tolist()
initial['grasp_command']=row['active_angles']
initial['surface_candidate']=dict(qualified=False,geometry_only=True,requires_end_support_clearance_m=.06,
    input_sha256=hashlib.sha256(options.candidates.read_bytes()).hexdigest())
initial_path=options.output.with_name(options.output.name+'_geometry_reset_only.json')
if initial_path.exists():raise FileExistsError(initial_path)
initial_path.write_text(json.dumps(initial,indent=2))

contact_profiles=[]
if options.batched_contacts:
    from go2_pvcnn.control.dh116.tabletop_effective_pad_runtime import PadContactSampler
    from go2_pvcnn.control.dh116.tabletop_effective_pad_batch import sample_native_batched
    scalar_sample=PadContactSampler.sample
    def batched_sample(self,*,contacts=True):
        if contacts:
            result=sample_native_batched(self)
            if result is not None:return result
        return scalar_sample(self,contacts=contacts)
    PadContactSampler.sample=batched_sample
if options.record_invalid_pad_buffers or options.record_contact_atlas:
    from go2_pvcnn.control.dh116.tabletop_effective_pad_runtime import PadContactSampler
    original_sample=PadContactSampler.sample
    faults=[]
    def diagnosed_sample(self,**kw):
        result=original_sample(self,**kw)
        self.profile_tick=getattr(self,'profile_tick',0)+int(kw.get('contacts',True))
        if options.record_contact_atlas and kw.get('contacts',True) and self.profile_tick%48==0:
            rows=[]
            for hi,bi,di,sensor in self.bindings:
                rows.append(dict(hand=hi,body=bi,digit=di,
                    buffer=[a.detach().cpu().clone() for a in sensor.contact_physx_view.get_contact_data(dt=self.world.dt)]))
            contact_profiles.append(dict(tick=self.profile_tick,rows=rows,
                object_pose_w=torch.cat((self.world.wp,self.world.wq),-1).cpu().clone(),
                poses=[h.data.body_link_pose_w.detach().cpu().clone() for h in self.world.hands]))
        if result['effective_pad_bad_env'].any() and len(faults)<4 and kw.get('contacts',True):
            rows=[]
            for hi,bi,di,sensor in self.bindings:
                rows.append(dict(hand=hi,body=bi,digit=di,
                    buffer=[a.detach().cpu().clone() for a in sensor.contact_physx_view.get_contact_data(dt=self.world.dt)]))
            faults.append(dict(bad_env=result['effective_pad_bad_env'].cpu(),rows=rows,
                poses=[h.data.body_link_pose_w.detach().cpu().clone() for h in self.world.hands]))
            torch.save(faults,options.output/'invalid_pad_buffers.pt')
        return result
    PadContactSampler.sample=diagnosed_sample

from go2_pvcnn.control.dh116 import tabletop_tensor_training as training
from go2_pvcnn.control.dh116 import tabletop_tensor_env as env_module
from go2_pvcnn.control.dh116.tabletop_manual_joint_contact import point_jacobian
from go2_pvcnn.control.dh116.tabletop_manual_joint_contact import bounded_dls
from go2_pvcnn.control.dh116.tabletop_surface_force_sharing import surface_targets
state={};old_init=env_module.TensorEnv.__init__;old_run=training.run_tensor_training
def capture(self,*a,**kw):
    old_init(self,*a,**kw);state['env']=self
env_module.TensorEnv.__init__=capture

def direct_trial(obs,steps,action_size):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_reference import closure_step,rise_step,anchor_step,ascent_floor
    from go2_pvcnn.control.dh116.tabletop_effective_pad_runtime import _rotation
    from go2_pvcnn.control.dh116.tabletop_grip_lift import valid_readback
    from go2_pvcnn.control.dh116.tabletop_tensor_env import physical_guard
    from go2_pvcnn.control.dh116.tabletop_multifinger_monitor import rod_points_in_hands
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,MIMIC
    env=state['env'];t=env.task;w=env.world;n=w.n
    if action_size!=24 or n!=2:raise ValueError('two-case 24D diagnostic required')
    folds=[];bodies=[];local_points=[];local_normals=[];eligible=[];start_points=[];normal_goals=[]
    names=('finger14_link','finger23_link','finger33_link','finger43_link','finger53_link')
    geo=row['geometry']['per_hand']
    for h,hand in enumerate(w.hands):
        fold=w.t(torch.zeros(len(hand.joint_names),12))
        for j,name in enumerate(ACTIVE):fold[hand.joint_names.index(name),j]=1.
        for follower,master,ratio,_ in MIMIC:fold[hand.joint_names.index(follower),ACTIVE.index(master)]=ratio
        for j,ji in enumerate(w.w.mounts[h].angular_ids):fold[ji,9+j]=1.
        folds.append(fold);bids=[hand.body_names.index(name) for name in names];bodies.append(bids)
        poses=hand.data.body_link_pose_w[:,bids].double();r=_rotation(poses)
        point=w.t(geo[h]['contact_points'])[None]+w.origins[:,None]
        local=torch.einsum('ndc,ndck->ndk',point-poses[...,:3],r)
        center=w.t([w.pad_sampler.atlas['sides'][('left','right')[h]][name]['center_local_m'] for name in names])
        local=.8*local+.2*center[None]
        normal=w.t(geo[h]['hand_normals'])[None].expand(n,-1,-1)
        normal_local=torch.einsum('ndc,ndck->ndk',normal,r)
        local_points.append(local);local_normals.append(normal_local)
        start_points.append(torch.einsum('ndck,ndk->ndc',r,local)+poses[...,:3])
        # Cylinder tangency: pad outward normal points towards the rod.
        normal_goals.append(normal)
        eligible.append(w.t([name in geo[h]['contact_digits'] for name in ('thumb','index','middle','ring','little')]).bool())
    eligible=torch.stack(eligible)[None].expand(n,-1,-1)
    local_points=torch.stack(local_points,1);local_normals=torch.stack(local_normals,1)
    start_points=torch.stack(start_points,1);normal_goals=torch.stack(normal_goals,1)
    radial_goals=normal_goals.clone();radial_goals[...,1]=0
    radial_goals=radial_goals/radial_goals.norm(dim=-1,keepdim=True)
    rod_start=w.wp.clone();preload=w.t(options.preloads_mm)/1000
    final_points=start_points.clone();final_points[...,0]=rod_start[:,None,None,0]-(.008-preload[:,None,None])*radial_goals[...,0]
    final_points[...,2]=rod_start[:,None,None,2]-(.008-preload[:,None,None])*radial_goals[...,2]
    t.surface_redistribution=False # teacher uses bounded Cartesian allocation below
    t.pinch_force_hold=True # do not cancel pressure closure with master-target mimic correction
    w.surface_trace=[];controls=[];offset=w.t(torch.zeros(n,2,5));age=w.t(torch.zeros(n))
    hold=age.clone();longest=age.clone();episodes=torch.zeros(n,device=w.device,dtype=torch.long)
    pressure=preload[:,None,None].expand(n,2,5).clone();grasp_time=age.clone();rise=age.clone()
    filtered_load=t.m['effective_pad_loads_n'].clone();filtered_vectors=t.m['effective_pad_vectors_w'].clone()
    acquired=torch.zeros(n,device=w.device,dtype=torch.bool)
    grip_root_z=w.roots[:,:,2].clone();grip_rise=rise.clone()
    anchored=torch.zeros(n,device=w.device,dtype=torch.bool);anchor=None;pending=anchored.clone()
    start=time.monotonic();error=None;stop='sim_budget';last_a=None
    capture=dict(joint_names=[h.joint_names for h in w.hands],body_names=[h.body_names for h in w.hands],
        jac=[h.root_physx_view.get_jacobians().cpu().clone() for h in w.hands],
        com=[h.data.body_com_pos_w.cpu().clone() for h in w.hands],
        poses=[h.data.body_link_pose_w.cpu().clone() for h in w.hands],q=w.q.cpu().clone(),
        wrist_pose=t.m['wrist_pose'].cpu().clone(),folds=[f.cpu() for f in folds])
    torch.save(capture,options.output/'initial_jacobians.pt')
    def readable(m):
        return valid_readback(torch.cat((w.wp,w.wq),-1),w.wv,m['q'],m['qd'],m['hand_object_loads'],
            w.poses,w.qd,m['joint_failure'],w.adapter_fault_bits,w.clearances)&physical_guard(w,m)[0]
    try:
        for tick in range(round(options.seconds*60)):
            elapsed=tick/60
            if time.monotonic()-start>options.wall_seconds:stop='wall_budget';break
            a=surface_targets(filtered_load,filtered_vectors,valid=filtered_load>.0)
            grasp_time=torch.where(a['paired'].all(-1),grasp_time+1/60,0.)
            redistribution=torch.where(a['paired'][...,None]&eligible,
                (2e-5*(a['desired']-filtered_load)).clamp(-5e-6,5e-6),0.)
            fraction=(age/options.close_seconds).clamp_max(1.)[:,None,None,None]
            new_acquired=(grasp_time>=.05)&~acquired
            grip_root_z[new_acquired]=w.roots[new_acquired,:,2];grip_rise[new_acquired]=rise[new_acquired]
            acquired |= new_acquired
            missing=filtered_load<=.05
            lag=rise-(w.bottom-w.lift_reference_bottom_m)
            pressure=closure_step(pressure,redistribution,missing,eligible,age>=options.close_seconds,
                (lag>.0003)&acquired,t.m['effective_pad_depth_m'],1/60)
            # Advance the teacher reference only as the actual rod follows;
            # this does not add a stage/force band to the free PPO task.
            rise,up=rise_step(rise,w.bottom-w.lift_reference_bottom_m,a['paired'].all(-1),
                acquired,options.lift_speed,1/60)
            desired=start_points+fraction*(final_points-start_points)
            desired+=(pressure-preload[:,None,None])[...,None]*radial_goals
            desired[...,2]+=rise[:,None,None]
            point_j=w.t(torch.zeros(n,2,5,3,12));normal_j=point_j.clone();actual=start_points.clone();actual_n=normal_goals.clone()
            for h,hand in enumerate(w.hands):
                poses=hand.data.body_link_pose_w[:,bodies[h]].double();r=_rotation(poses)
                actual[:,h]=torch.einsum('ndck,ndk->ndc',r,local_points[:,h])+poses[...,:3]
                actual_n[:,h]=torch.einsum('ndck,ndk->ndc',r,local_normals[:,h])
                native=hand.root_physx_view.get_jacobians().double()
                for d,b in enumerate(bodies[h]):
                    j=point_jacobian(native[:,b-1],actual[:,h,d],hand.data.body_com_pos_w[:,b].double(),folds[h])
                    j[:,:,6:9]=torch.eye(3,device=w.device,dtype=torch.float64)
                    point_j[:,h,d]=j
                    angular=native[:,b-1,3:,:]@folds[h]
                    normal_j[:,h,d]=torch.linalg.cross(angular.transpose(-1,-2),actual_n[:,h,d,None].expand(-1,12,-1)).transpose(-1,-2)
            velocity=(4*(desired-actual)).clamp(-.003,.003);velocity[...,2]+=up[:,None,None]
            normal_v=2*(normal_goals-(normal_goals*actual_n).sum(-1,keepdim=True)*actual_n)
            mask=eligible[...,None,None]
            A=torch.cat(((point_j*mask).flatten(2,3),(.0024*normal_j*mask).flatten(2,3)),2)
            B=torch.cat(((velocity*eligible[...,None]).flatten(2,3),(.0024*normal_v*eligible[...,None]).flatten(2,3)),2)
            scale=w.t([.02]*6+[.001]*3+[.05]*3)
            limit=w.t([.15]*6+[.008]*3+[.2]*3)
            bias=w.t(torch.zeros(n,2,12));bias[:,:,8]=up[:,None]
            lower=(-limit).expand_as(bias).clone();upper=limit.expand_as(bias).clone()
            lower[:,:,8]=ascent_floor(w.roots[:,:,2],grip_root_z,rise-grip_rise,up,acquired)
            rate=bounded_dls(A*scale,B,bias/scale,lower/scale,upper/scale,1e-5)*scale
            action=w.t(torch.zeros(n,24));action[:,:12]=rate[:,:,:6].flatten(1)/.3
            action[:,12:18]=rate[:,:,6:9].flatten(1)/.01;action[:,18:24]=rate[:,:,9:].flatten(1)/.5
            reset_this_tick=torch.zeros(n,device=w.device,dtype=torch.bool)
            frame_load=torch.zeros_like(filtered_load);frame_vectors=torch.zeros_like(filtered_vectors)
            for substep in range(w.decimation):
                live=torch.ones(n,device=w.device,dtype=torch.bool)
                safe_action=torch.where(reset_this_tick[:,None],0.,action)
                allowed=t.control(safe_action,live,w.dt);m=w.advance(t,allowed,live);t.m=m
                good=readable(m)
                if (pending&~good).any():raise RuntimeError('teacher reset failed next-step readback')
                pending.zero_();reset=~good
                w.surface_trace[-1].update(full_link_pose_w=w.poses.cpu().clone(),full_joint_velocity=w.qd.cpu().clone(),
                    reset=reset.cpu(),episode=episodes.cpu().clone(),elapsed_s=age.cpu().clone(),
                    wrist_goal_rise_m=rise.cpu().clone(),geometric_preload_m=pressure.cpu().clone())
                if reset.any():
                    ids=reset.nonzero().flatten();m=w.reset(ids)
                    if not readable(m)[ids].all():raise RuntimeError('teacher reset invalid')
                    t.reset(ids,m);t.m=m;episodes[ids]+=1;offset[ids]=0;pending[ids]=True;anchored[ids]=False
                    pressure[ids]=preload[ids,None,None];grasp_time[ids]=0;acquired[ids]=False;rise[ids]=0
                    age[ids]=-w.dt;hold[ids]=0
                    reset_this_tick|=reset
                    filtered_load[ids]=0;filtered_vectors[ids]=0;frame_load[ids]=0;frame_vectors[ids]=0
                age+=w.dt
                last_a=surface_targets(m['effective_pad_loads_n'],m['effective_pad_vectors_w'],valid=m['effective_pad_valid'])
                paired=last_a['paired'].all(-1)&good&~reset
                relative=rod_points_in_hands(torch.cat((w.wp,w.wq),-1),w.roots)
                if anchor is None:anchor=relative.clone()
                anchor,anchored=anchor_step(anchor,anchored,relative,
                    a['paired'].all(-1)&acquired,good,reset)
                drift=(relative-anchor).norm(dim=-1).flatten(1).amax(-1)
                ordinary=m['rejected_contact_loads_n'].sum(-1);load=m['effective_pad_loads_n'].sum((1,2))
                ready=(paired&(ordinary/(ordinary+load).clamp_min(1e-12)<=.05)
                    &(m['effective_pad_depth_m'].amax((1,2))<=.0008))
                frame_load+=torch.where(good[:,None,None]&~reset_this_tick[:,None,None],m['effective_pad_loads_n'],0.)
                frame_vectors+=torch.where(good[:,None,None,None]&~reset_this_tick[:,None,None,None],m['effective_pad_vectors_w'],0.)
                qualified=(paired&anchored&last_a['balanced'].all(-1)&(drift<=.0005)&(w.bottom>=w.lift_reference_bottom_m+.03)
                    &(w.wv[:,:3].norm(dim=-1)<.03)&(ordinary/(ordinary+load).clamp_min(1e-12)<=.05)
                    &(m['effective_pad_depth_m'].amax((1,2))<=.0008))
                hold=torch.where(qualified,hold+w.dt,0.);longest=torch.maximum(longest,hold)
                w.surface_trace[-1].update(rod_anchor_drift_m=drift.cpu(),qualified_frame=qualified.cpu(),hold_s=hold.cpu().clone())
            filtered_load=.8*filtered_load+.2*frame_load/w.decimation
            filtered_vectors=.8*filtered_vectors+.2*frame_vectors/w.decimation
            controls.append(dict(action=action.cpu(),normal_offsets_m=offset.cpu().clone(),desired_points_w=desired.cpu(),
                measured_points_w=actual.cpu(),desired_digit_loads_n=last_a['desired'].cpu(),
                face_loads_n=last_a['face_loads'].cpu(),paired=last_a['paired'].cpu(),balanced=last_a['balanced'].cpu(),
                control_filtered_loads_n=filtered_load.cpu(),control_filtered_vectors_w=filtered_vectors.cpu()))
            if tick%60==59:
                status=dict(sim_seconds=(tick+1)/60,wall_seconds=time.monotonic()-start,episodes=episodes.cpu().tolist(),
                    actual_net_bottom_mm=((w.bottom-w.lift_reference_bottom_m)*1000).cpu().tolist(),
                    pad_loads_n=m['effective_pad_loads_n'].cpu().tolist(),ordinary_loads_n=m['rejected_contact_loads_n'].cpu().tolist(),
                    paired=last_a['paired'].cpu().tolist(),hold_s=hold.cpu().tolist(),
                    acquired=acquired.cpu().tolist(),grasp_stable_s=grasp_time.cpu().tolist(),
                    rise_goal_mm=(rise*1000).cpu().tolist(),preload_mm=(pressure*1000).cpu().tolist(),
                    control_filtered_loads_n=filtered_load.cpu().tolist())
                (options.output/'probe_status.json').write_text(json.dumps(status,indent=2));print('POINT_TEACHER',json.dumps(status),flush=True)
            if elapsed>10. and not acquired.any():stop='no_stable_bilateral_grasp';break
            if elapsed>10. and (w.bottom-w.lift_reference_bottom_m).max()<.001:stop='no_actual_lift';break
            if (longest>=5).any():stop='five_second_candidate_requires_independent_audit';break
    except Exception as exc:error=repr(exc);traceback.print_exc()
    keys=set.intersection(*(set(s) for s in w.surface_trace)) if w.surface_trace else set()
    torch.save({k:torch.stack([s[k] for s in w.surface_trace]) for k in keys},options.output/'surface_physics.pt')
    if contact_profiles:
        torch.save(dict(samples=contact_profiles,atlas=w.pad_sampler.atlas,
            body_names=[list(h.body_names) for h in w.hands]),options.output/'pad_contact_profiles.pt')
    torch.save({k:torch.stack([s[k] for s in controls]) for k in controls[0]} if controls else {},options.output/'surface_control.pt')
    (options.output/'effective_pad_atlas.json').write_text(json.dumps(w.pad_sampler.atlas,indent=2))
    result=dict(status='native_point_teacher_complete',stop_reason=stop,error=error,force_goal=None,ppo_updates=0,
        controller='planned_inner_point_full_xyz_and_normal_with_original_position_drives',
        candidate_index=options.candidate_index,preloads_mm=options.preloads_mm,close_seconds=options.close_seconds,
        lift_speed_m_s=options.lift_speed,sim_seconds=len(controls)/60,wall_seconds=time.monotonic()-start,
        episodes=episodes.cpu().tolist(),longest_qualified_window_s=longest.cpu().tolist(),
        raw_window_candidate=(longest>=5).cpu().tolist(),qualified=False,teacher_export=False,
        source_freeze=options.source_freeze,batched_contacts=options.batched_contacts,
        raw_contact_profiles=len(contact_profiles),raw_trace_keys=sorted(keys),
        pressure_goal='free_measured_load;geometric_closure_adapts_to_missing_contact_and_actual_slip',
        teacher_acquisition='opposed_EMA_of_native_480Hz_step_mean_0.05s;raw_qualification_unchanged',
        control_filter='480Hz_native_8step_mean_then_60Hz_EMA_alpha0.2;not_a_native_force_replacement',
        wrist_ascent='bounded_physical_wrist_Z_lower_rate;not_finger_tip_substitution',
        mimic_target_correction='thumb/index_master_override_off;original_physical_passive_drives_on',
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (options.output/'probe_result.json').write_text(json.dumps(result,indent=2));print('POINT_TEACHER_RESULT',json.dumps(result),flush=True)
    os._exit(0 if error is None else 1)
training.make_algorithm=direct_trial

flags=(root/'scripts/smoke_dh116_surface_grasp.sh').read_text()
import shlex
cmd=shlex.split(flags[flags.index('exec bash scripts/python.sh ')+len('exec bash scripts/python.sh '):].replace('\\\n',' '))
cmd[cmd.index('--grasp-initialization')+1]=str(initial_path)
cmd[cmd.index('--output')+1]=str(options.output)
cmd[cmd.index('--surface-diagnostic-steps')+1]=str(min(10000,round(options.seconds*480)+16))
sys.argv=cmd
try:runpy.run_path(str(root/cmd[0]),run_name='__main__')
except BaseException:traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
