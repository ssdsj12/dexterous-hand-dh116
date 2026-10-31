"""Bounded fixed 6N-total teacher using the physical 24D motion actuators.

Each hand requests 3N normal compression, split by opposed surface then digit.
No PPO update or object manipulation after reset. Independent raw auditing is
required before qualification; a requested force is never an applied-force claim.
"""
from pathlib import Path
import argparse,json,hashlib,os,runpy,sys,time,traceback
import numpy as np
import torch

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
os.chdir(root)
parser=argparse.ArgumentParser()
parser.add_argument('--candidates',type=Path,default=root/'configs/dh116_training/effective_pad_center_pregrasp_v1.json')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--seconds',type=float,default=28.)
parser.add_argument('--wall-seconds',type=float,default=1200.)
parser.add_argument('--record-invalid-pad-buffers',action='store_true')
parser.add_argument('--record-contact-atlas',action='store_true')
parser.add_argument('--source-freeze',default='not_provided')
parser.add_argument('--candidate-index',type=int,default=0)
parser.add_argument('--lift-speed',type=float,default=.003)
parser.add_argument('--explore-lift-after-s',type=float,default=6.)
parser.add_argument('--batched-contacts',action='store_true')
parser.add_argument('--batched-native-sensors',action='store_true')
parser.add_argument('--pad-centering',action='store_true')
parser.add_argument('--retain-airborne-pose',action='store_true')
parser.add_argument('--retain-prelift-pose',action='store_true',
    help='teacher only: retain a genuinely loaded opposed grip before wrist ascent')
parser.add_argument('--profile-cpu',action='store_true')
parser.add_argument('--target-hand-force-n',type=float,default=3.)
parser.add_argument('--force-tolerance-n',type=float,default=.3)
parser.add_argument('--admittance-gains',type=float,nargs=2,default=[.0004,.0008])
parser.add_argument('--joint-target-leads-rad',type=float,nargs=2,default=[.02,.04])
options=parser.parse_args()
if options.output.exists():raise FileExistsError(options.output)
if not 1<=options.seconds<=35 or not 30<=options.wall_seconds<=1200:raise ValueError('bounded trial required')
if not 0<options.lift_speed<=.01:raise ValueError('bounded motion required')
if options.explore_lift_after_s is not None and not (np.isfinite(options.explore_lift_after_s) and 0<=options.explore_lift_after_s<=20):
    raise ValueError('exploratory release time must be finite and within 0..20s')
if not np.isfinite(options.target_hand_force_n) or options.target_hand_force_n<=0:raise ValueError('positive finite hand force required')
if not 0<options.force_tolerance_n<=.3:raise ValueError('current user error limit: at most 0.3N')
if not all(0<x<=.002 for x in options.admittance_gains):raise ValueError('bounded positive admittance gains required')
if not all(.3/60<x<=.1 for x in options.joint_target_leads_rad):raise ValueError('target leads must fit the original 0.1rad envelope')
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
    from go2_pvcnn.control.dh116.tabletop_native_teacher_reference import scheduled_ascent,anchor_step,ascent_floor,force_tracking,hybrid_velocity,reference_velocity_box,interval_force_state,acquisition_dwell,lift_motion_permission
    from go2_pvcnn.control.dh116.tabletop_prioritized_velocity import prioritized_velocity
    from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import CONTRACT as WRIST
    from go2_pvcnn.control.dh116.tabletop_effective_pad_runtime import _rotation
    from go2_pvcnn.control.dh116.tabletop_grip_lift import valid_readback
    from go2_pvcnn.control.dh116.tabletop_tensor_env import physical_guard
    from go2_pvcnn.control.dh116.tabletop_multifinger_monitor import rod_points_in_hands
    from go2_pvcnn.control.dh116.tabletop_lift_hold_window import hold_window_step
    from go2_pvcnn.control.dh116.tabletop_pose_carry import carry_latch,carry_action,prelift_retention
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,MIMIC
    env=state['env'];t=env.task;w=env.world;n=w.n
    if options.batched_native_sensors:
        if not options.batched_contacts:raise ValueError('batched native sensors require batched pad sampling')
        from go2_pvcnn.control.dh116.tabletop_native_sensor_batch import read_native_sensors
        w.native_sensor_reader=read_native_sensors
    centering=None
    if options.pad_centering:
        from go2_pvcnn.control.dh116.tabletop_pad_centering import PadCentering
        centering=PadCentering(w.pad_sampler)
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
    planned=surface_targets(eligible.double(),-normal_goals,valid=eligible)
    shares=planned['desired']/eligible.sum(-1,keepdim=True)
    force_reference=w.t(torch.full((n,2),options.target_hand_force_n))
    admittance_gain=w.t(options.admittance_gains)[:,None,None]
    native_depth=t.m['effective_pad_depth_m'].clone()
    t.surface_redistribution=False # teacher uses bounded Cartesian allocation below
    t.pinch_force_hold=True # do not cancel pressure closure with master-target mimic correction
    t.teacher_force_controlled_digits=eligible.clone()
    w.surface_trace=[];w.surface_trace_limit=round(options.seconds*480)+32
    controls=[];age=w.t(torch.zeros(n))
    hold=age.clone();longest=age.clone();episodes=torch.zeros(n,device=w.device,dtype=torch.long)
    grasp_time=age.clone();rise=age.clone()
    filtered_load=t.m['effective_pad_loads_n'].clone();filtered_vectors=t.m['effective_pad_vectors_w'].clone()
    mean_load=filtered_load.clone();mean_vectors=filtered_vectors.clone()
    target_lead=w.t(options.joint_target_leads_rad)[:,None,None]
    acquired=torch.zeros(n,device=w.device,dtype=torch.bool)
    previous_interval_good=acquired.clone();previous_interval_reset=acquired.clone()
    previous_paired=torch.zeros(n,2,device=w.device,dtype=torch.bool)
    grip_root_z=w.roots[:,:,2].clone();grip_rise=rise.clone()
    anchored=torch.zeros(n,device=w.device,dtype=torch.bool);anchor=None;pending=anchored.clone()
    hold_latched=anchored.clone();hold_anchor=None;initial_wrist_z=w.roots[:,:,2].clone()
    pose_carry=anchored.clone();carry_ever=anchored.clone()
    prelift_dwell=w.t(torch.zeros(n))
    carry_command=t.command.reshape(n,2,6).clone()
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
    profiler=None
    if options.profile_cpu:
        import cProfile
        profiler=cProfile.Profile();profiler.enable()
    try:
        for tick in range(round(options.seconds*60)):
            elapsed=tick/60
            if (options.output/'STOP').exists():stop='operator_stop';break
            if time.monotonic()-start>options.wall_seconds:stop='wall_budget';break
            if options.retain_airborne_pose or options.retain_prelift_pose:
                if options.retain_prelift_pose:
                    next_carry,prelift_dwell=prelift_retention(pose_carry,prelift_dwell,mean_load,
                        previous_paired,eligible,previous_interval_reset,1/60)
                else:
                    next_carry=carry_latch(pose_carry,w.bottom-w.lift_reference_bottom_m,
                        previous_paired,previous_interval_reset)
                captured=next_carry&~pose_carry
                carry_command[captured]=t.command.reshape(n,2,6)[captured]
                pose_carry=next_carry;carry_ever|=pose_carry
                t.teacher_force_controlled_digits[pose_carry]=True
            grasp_time=acquisition_dwell(grasp_time,previous_interval_good,previous_interval_reset,1/60)
            new_acquired=lift_motion_permission(grasp_time>=.2,age,options.explore_lift_after_s)&~acquired
            if options.retain_prelift_pose:new_acquired=pose_carry&~acquired
            grip_root_z[new_acquired]=w.roots[new_acquired,:,2];grip_rise[new_acquired]=rise[new_acquired]
            acquired |= new_acquired
            # An acquired teacher continues the physical wrist schedule while
            # correcting force. Actual rod height is independently measured.
            rise,up=scheduled_ascent(rise,acquired,options.lift_speed,1/60)
            desired=start_points.clone()
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
            actual_inward=-filtered_vectors/filtered_vectors.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            actual_inward=torch.where((filtered_load>.05)[...,None],actual_inward,actual_n)
            force_desired=force_reference[...,None]*shares
            velocity=hybrid_velocity(desired-actual,actual_inward,filtered_load,force_desired,native_depth,up,gain=admittance_gain)
            centering_diag=None
            if centering is not None:
                transport=w.t(torch.zeros(n,3));transport[:,2]=up
                velocity,centering_diag=centering.adjust(velocity,actual_inward,
                    torch.cat((w.wp,w.wq),-1).double(),w.wv.double(),1/60,transport_velocity_w=transport)
            normal_v=2*(normal_goals-(normal_goals*actual_n).sum(-1,keepdim=True)*actual_n)
            mask=eligible[...,None,None]
            A=torch.cat(((point_j*mask).flatten(2,3),(.0008*normal_j*mask).flatten(2,3)),2)
            B=torch.cat(((velocity*eligible[...,None]).flatten(2,3),(.0008*normal_v*eligible[...,None]).flatten(2,3)),2)
            scale=w.t([.02]*6+[.001]*3+[.05]*3)
            # Inscribe the XYZ box in the production 10mm/s norm bound so the
            # downstream command limiter cannot change the normal priority.
            limit=w.t([.15]*6+[.004,.004,.008]+[.2]*3)
            bias=w.t(torch.zeros(n,2,12));bias[:,:,8]=up[:,None]
            command=t.command.reshape(n,2,6);measured=t.m['q'].reshape(n,2,6)
            current=torch.cat((command,w.palms,w.wrist_angles),-1)
            margin=.3/60
            low_ref=torch.cat(((measured-target_lead+margin).maximum(t.low.reshape(1,2,6)),
                w.t(WRIST['palm_lower_m']).expand(n,2,3),torch.full_like(w.wrist_angles,-np.pi/2)),-1)
            high_ref=torch.cat(((measured+target_lead-margin).minimum(t.high.reshape(1,2,6)),
                w.t(WRIST['palm_upper_m']).expand(n,2,3),torch.full_like(w.wrist_angles,np.pi/2)),-1)
            original_lower,upper=reference_velocity_box(current,low_ref,high_ref,limit,1/60)
            lower=original_lower.clone()
            lower[:,:,8]=torch.maximum(lower[:,:,8],torch.minimum(upper[:,:,8],
                ascent_floor(w.roots[:,:,2],grip_root_z,rise-grip_rise,up,acquired)))
            normal_J=torch.einsum('nhdc,nhdck->nhdk',actual_inward,point_j)*eligible[...,None]
            normal_request=(velocity*actual_inward).sum(-1)*eligible
            allocated,solver_info=prioritized_velocity(normal_J*scale,normal_request,
                A*scale,B,bias/scale,lower/scale,upper/scale,1e-5)
            # Lift transport remains inside the original physical box. A force
            # tracking residual is diagnostic, not permission to cancel lift.
            relaxed=solver_info['primary_direction_infeasible']&~acquired[:,None]
            if relaxed.any():
                lower[:,:,8]=torch.where(relaxed,original_lower[:,:,8],lower[:,:,8])
                allocated,solver_info=prioritized_velocity(normal_J*scale,normal_request,
                    A*scale,B,bias/scale,lower/scale,upper/scale,1e-5)
            unresolved=solver_info['primary_direction_infeasible']
            neutral=torch.zeros_like(allocated).maximum(lower/scale).minimum(upper/scale)
            allocated=torch.where((unresolved&~acquired[:,None])[...,None],neutral,allocated)
            rate=allocated*scale
            action=w.t(torch.zeros(n,24));action[:,:12]=rate[:,:,:6].flatten(1)/.3
            action[:,12:18]=rate[:,:,6:9].flatten(1)/.01;action[:,18:24]=rate[:,:,9:].flatten(1)/.5
            if options.retain_airborne_pose or options.retain_prelift_pose:
                action,carry_up=carry_action(action,pose_carry,w.bottom-w.lift_reference_bottom_m,
                    original_lower,upper,options.lift_speed)
            submitted_velocity=torch.cat((action[:,:12].reshape(n,2,6)*.3,
                action[:,12:18].reshape(n,2,3)*.01,action[:,18:24].reshape(n,2,3)*.5),-1)
            carry_this_tick=pose_carry.clone()
            reset_this_tick=torch.zeros(n,device=w.device,dtype=torch.bool)
            frame_load=torch.zeros_like(filtered_load);frame_vectors=torch.zeros_like(filtered_vectors)
            frame_depth=torch.zeros_like(native_depth)
            interval_loads=[];interval_vectors=[];interval_ordinary=[];geometry_frames=[];relative_frames=[]
            for substep in range(w.decimation):
                live=torch.ones(n,device=w.device,dtype=torch.bool)
                safe_action=torch.where(reset_this_tick[:,None],0.,action)
                allowed=t.control(safe_action,live,w.dt);m=w.advance(t,allowed,live);t.m=m
                good=readable(m)
                if (pending&~good).any():raise RuntimeError('teacher reset failed next-step readback')
                pending.zero_();reset=~good
                w.surface_trace[-1].update(full_link_pose_w=w.poses.cpu().clone(),full_joint_velocity=w.qd.cpu().clone(),
                    reset=reset.cpu(),episode=episodes.cpu().clone(),elapsed_s=age.cpu().clone(),
                    wrist_command_pose_w=w.root_targets.cpu().clone(),wrist_command_angles=w.wrist_angles.cpu().clone(),
                    wrist_goal_rise_m=rise.cpu().clone(),normal_force_request_n=force_desired.cpu().clone())
                w.surface_trace[-1]['pose_carry_active']=carry_this_tick.cpu()
                if reset.any():
                    ids=reset.nonzero().flatten();m=w.reset(ids)
                    if not readable(m)[ids].all():raise RuntimeError('teacher reset invalid')
                    t.reset(ids,m);t.m=m;episodes[ids]+=1;pending[ids]=True;anchored[ids]=False
                    if centering is not None:centering.reset(ids)
                    grasp_time[ids]=0;acquired[ids]=False;rise[ids]=0
                    age[ids]=-w.dt;hold[ids]=0
                    hold_latched[ids]=False
                    pose_carry[ids]=False;t.teacher_force_controlled_digits[ids]=eligible[ids]
                    prelift_dwell[ids]=0
                    reset_this_tick|=reset
                    filtered_load[ids]=0;filtered_vectors[ids]=0;frame_load[ids]=0;frame_vectors[ids]=0
                    mean_load[ids]=0;mean_vectors[ids]=0
                    force_reference[ids]=options.target_hand_force_n;frame_depth[ids]=0
                age+=w.dt
                last_a=surface_targets(m['effective_pad_loads_n'],m['effective_pad_vectors_w'],valid=m['effective_pad_valid'])
                paired=last_a['paired'].all(-1)&good&~reset
                relative=rod_points_in_hands(torch.cat((w.wp,w.wq),-1),w.roots)
                relative_frames.append(relative.clone())
                if anchor is None:anchor=relative.clone()
                anchor,anchored=anchor_step(anchor,anchored,relative,
                    previous_paired.all(-1)&acquired,good,reset)
                drift=(relative-anchor).norm(dim=-1).flatten(1).amax(-1)
                ordinary=m['rejected_contact_loads_n'].sum(-1);load=m['effective_pad_loads_n'].sum((1,2))
                raw_force_ok,hand_error,digit_error=force_tracking(m['effective_pad_loads_n'],force_desired,
                    eligible,options.target_hand_force_n,options.force_tolerance_n,face_loads=last_a['face_loads'])
                frame_load+=torch.where(good[:,None,None]&~reset_this_tick[:,None,None],m['effective_pad_loads_n'],0.)
                frame_vectors+=torch.where(good[:,None,None,None]&~reset_this_tick[:,None,None,None],m['effective_pad_vectors_w'],0.)
                frame_depth=torch.maximum(frame_depth,m['effective_pad_depth_m'])
                interval_loads.append(m['effective_pad_loads_n'].clone())
                interval_vectors.append(m['effective_pad_vectors_w'].clone())
                interval_ordinary.append(m['rejected_contact_loads_n'].clone())
                qualified=(good&~reset&anchored&(drift<=.0005)&(w.bottom>=w.lift_reference_bottom_m+.03)
                    &(w.wv[:,:3].norm(dim=-1)<.03)
                    &(m['effective_pad_depth_m'].amax((1,2))<=.0008))
                physical_lift=(good&~reset&(w.bottom>=w.lift_reference_bottom_m+.03)
                    &(w.wv[:,:3].norm(dim=-1)<.03)&(m['effective_pad_depth_m'].amax((1,2))<=.0008))
                geometry_frames.append(physical_lift)
                w.surface_trace[-1].update(rod_anchor_drift_m=drift.cpu(),geometry_qualified_frame=qualified.cpu(),
                    physical_lift_frame=physical_lift.cpu(),
                    hand_force_error_n=hand_error.cpu(),digit_force_error_n=digit_error.cpu(),raw_force_in_tolerance=raw_force_ok.cpu())
                w.surface_trace[-1]['face_force_error_n']=(last_a['face_loads']-options.target_hand_force_n/2).cpu()
            filtered_load=.8*filtered_load+.2*frame_load/w.decimation
            filtered_vectors=.8*filtered_vectors+.2*frame_vectors/w.decimation
            native_depth=frame_depth
            interval_state=interval_force_state(torch.stack(interval_loads),torch.stack(interval_vectors),force_desired,
                eligible,options.target_hand_force_n,options.force_tolerance_n)
            mean_load=interval_state['mean_loads'];mean_vectors=interval_state['mean_vectors']
            previous_interval_good=interval_state['good'];previous_interval_reset=reset_this_tick
            previous_paired=interval_state['paired']
            ordinary_mean=torch.stack(interval_ordinary).mean(0).sum(-1);normal_mean=mean_load.sum((1,2))
            physical_ok=(interval_state['paired'].all(-1)&torch.stack(geometry_frames).all(0)&~reset_this_tick
                &(ordinary_mean/(ordinary_mean+normal_mean).clamp_min(1e-12)<=.05))
            prior_hold=hold.clone()
            if hold_anchor is None:hold_anchor=relative_frames[0].clone()
            hold,hold_anchor,hold_latched,window_diag=hold_window_step(hold,hold_anchor,hold_latched,
                torch.stack(relative_frames),physical_ok,1/60)
            interval_good=window_diag['good'];longest=torch.maximum(longest,hold)
            for j,record in enumerate(w.surface_trace[-w.decimation:]):
                record.update(qualified_frame=interval_good.cpu(),
                    hold_s=torch.where(interval_good,prior_hold+(j+1)*w.dt,0.).cpu(),
                    interval_force_in_tolerance=interval_state['good'].cpu(),force_interval_mean_n=mean_load.cpu())
                record['hold_window_drift_m']=window_diag['drift_m'].cpu()
            controls.append(dict(action=action.cpu(),desired_points_w=desired.cpu(),
                measured_points_w=actual.cpu(),desired_digit_loads_n=force_desired.cpu(),
                face_loads_n=interval_state['face_loads'].cpu(),paired=interval_state['paired'].cpu(),balanced=interval_state['balanced'].cpu(),
                force_mean_loads_n=mean_load.cpu(),mean_force_in_tolerance=interval_state['good'].cpu(),
                mean_hand_force_error_n=interval_state['hand_error'].cpu(),mean_face_force_error_n=interval_state['face_error'].cpu(),
                mean_digit_force_error_n=interval_state['digit_error'].cpu(),
                control_filtered_loads_n=filtered_load.cpu(),control_filtered_vectors_w=filtered_vectors.cpu()))
            controls[-1].update(normal_J=normal_J.cpu(),normal_request_m_s=normal_request.cpu(),allocated_velocity=rate.cpu(),
                submitted_velocity=submitted_velocity.cpu(),pose_carry_active=carry_this_tick.cpu(),
                carry_finger_command=carry_command.cpu().clone(),
                velocity_lower=lower.cpu(),velocity_upper=upper.cpu(),primary_status=solver_info['status'].cpu(),
                primary_preservation_error=solver_info['normal_preservation_error'].cpu(),normal_priority_unresolved=unresolved.cpu())
            if centering_diag is not None:
                controls[-1].update({'centering_'+k:v.cpu() for k,v in centering_diag.items()})
            if tick%60==59:
                status=dict(sim_seconds=(tick+1)/60,wall_seconds=time.monotonic()-start,episodes=episodes.cpu().tolist(),
                    actual_net_bottom_mm=((w.bottom-w.lift_reference_bottom_m)*1000).cpu().tolist(),
                    mean_pad_loads_n=mean_load.cpu().tolist(),mean_hand_force_n=mean_load.sum(-1).cpu().tolist(),
                    instant_pad_loads_n=m['effective_pad_loads_n'].cpu().tolist(),ordinary_loads_n=m['rejected_contact_loads_n'].cpu().tolist(),
                    paired=interval_state['paired'].cpu().tolist(),hold_s=hold.cpu().tolist(),
                    acquired=acquired.cpu().tolist(),grasp_stable_s=grasp_time.cpu().tolist(),
                    motion_released=acquired.cpu().tolist(),explore_lift_after_s=options.explore_lift_after_s,
                    actual_wrist_rise_mm=((w.roots[:,:,2]-initial_wrist_z)*1000).cpu().tolist(),
                    pose_carry_active=pose_carry.cpu().tolist(),actual_height_target_mm=31.5,
                    prelift_grip_dwell_s=prelift_dwell.cpu().tolist(),
                    rise_goal_mm=(rise*1000).cpu().tolist(),normal_force_request_n=force_desired.cpu().tolist(),
                    control_filtered_loads_n=filtered_load.cpu().tolist())
                (options.output/'probe_status.json').write_text(json.dumps(status,indent=2));print('POINT_TEACHER',json.dumps(status),flush=True)
            if (longest>=5).any():stop='five_second_candidate_requires_independent_audit';break
    except Exception as exc:error=repr(exc);traceback.print_exc()
    if profiler is not None:
        profiler.disable();profiler.dump_stats(str(options.output/'cpu_profile.prof'))
    keys=set.intersection(*(set(s) for s in w.surface_trace)) if w.surface_trace else set()
    torch.save({k:torch.stack([s[k] for s in w.surface_trace]) for k in keys},options.output/'surface_physics.pt')
    if contact_profiles:
        torch.save(dict(samples=contact_profiles,atlas=w.pad_sampler.atlas,
            body_names=[list(h.body_names) for h in w.hands]),options.output/'pad_contact_profiles.pt')
    torch.save({k:torch.stack([s[k] for s in controls]) for k in controls[0]} if controls else {},options.output/'surface_control.pt')
    (options.output/'effective_pad_atlas.json').write_text(json.dumps(w.pad_sampler.atlas,indent=2))
    result=dict(status='native_hybrid_teacher_complete',stop_reason=stop,error=error,ppo_updates=0,
        force_goal_source='user_fixed_total;per_hand_then_equal_opposed_faces_then_equal_digits',
        requested_total_n=2*options.target_hand_force_n,
        force_goal_final_n=force_reference.cpu().tolist(),force_tolerance_n=options.force_tolerance_n,
        admittance_gains=options.admittance_gains,joint_target_leads_rad=options.joint_target_leads_rad,
        controller='physical_lift_transport_then_relative_normal_force_and_tangential_centering_original_drives',
        candidate_index=options.candidate_index,
        lift_speed_m_s=options.lift_speed,sim_seconds=len(controls)/60,wall_seconds=time.monotonic()-start,
        explore_lift_after_s=options.explore_lift_after_s,
        retain_airborne_pose=options.retain_airborne_pose,pose_carry_ever=carry_ever.cpu().tolist(),
        carry_force_goal_usage='preload_servo;force_reference_diagnostic_during_acquired_pose_retention',
        carry_action_semantics='allocated_velocity_is_solver_proposal;submitted_velocity_is_actual_after_pose_retention',
        pose_carry_helper_sha256=hashlib.sha256(Path(sys.modules[carry_action.__module__].__file__).read_bytes()).hexdigest(),
        episodes=episodes.cpu().tolist(),longest_qualified_window_s=longest.cpu().tolist(),
        raw_window_candidate=(longest>=5).cpu().tolist(),qualified=False,teacher_export=False,
        source_freeze=options.source_freeze,batched_contacts=options.batched_contacts,
        batched_native_sensors=options.batched_native_sensors,
        pad_centering=options.pad_centering,
        centering_sha256=hashlib.sha256(Path(sys.modules[PadCentering.__module__].__file__).read_bytes()).hexdigest() if centering else None,
        raw_contact_profiles=len(contact_profiles),raw_trace_keys=sorted(keys),
        pressure_goal='fixed_hand_goal;native_compression_feedback;friction_excluded',
        pose_retention=('loaded_native_grip_before_lift' if options.retain_prelift_pose else
                        'after_real_3mm_lift' if options.retain_airborne_pose else None),
        priority_solver_sha256=hashlib.sha256((root/'src/go2_pvcnn/control/dh116/tabletop_prioritized_velocity.py').read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(sys.modules[hybrid_velocity.__module__].__file__).read_bytes()).hexdigest(),
        teacher_acquisition=('opposed_real_8substep_mean_hand_face_digit_tolerance_0.2s' if options.explore_lift_after_s is None
            else 'force_readiness_or_explicit_elapsed_motion_exploration;force_qualification_unchanged'),
        force_assessment='per_60Hz_control_interval_mean_of_8_native_480Hz_samples;no_extra_EMA;instant_errors_diagnostic',
        success_criterion='actual_net_bottom_30mm;5s_continuous_effective_opposed_grip;0.5mm_hold_window_slip;native_depth_and_validity;force_balance_error_diagnostic_only',
        lift_hold_helper_sha256=hashlib.sha256(Path(sys.modules[hold_window_step.__module__].__file__).read_bytes()).hexdigest(),
        control_filter='480Hz_native_8step_mean_then_60Hz_EMA_alpha0.2;not_a_native_force_replacement',
        wrist_ascent='scheduled_original_physical_wrist_after_force_readiness;force_correction_continues',
        mimic_target_correction='selected_teacher_bearing_digits_master_override_off;original_physical_passive_drives_on',
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
