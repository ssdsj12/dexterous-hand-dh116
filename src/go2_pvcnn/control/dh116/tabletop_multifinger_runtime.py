"""Explicit native five-digit force sharing and measured nonslip lift diagnostic."""
import torch
from .linear_coupling import ACTIVE,MIMIC,active_limits
from .tabletop_surface_kinematics import pose_matrix,_surface
from .tabletop_surface_objectives import finite_cylinder_sdf
from .tabletop_manual_joint_contact import point_jacobian
from .tabletop_multifinger_control import (sharing_targets,force_joint_rates,table_safe_rates,
    compliance_from_physx_material,projected_contact_stiffness_count)
from .tabletop_multifinger_monitor import GraspStabilityMonitor,rod_points_in_hands
from .tabletop_multifinger_motion import cancel_queued_ascent
from .tabletop_grip_lift import valid_readback


class MultifingerLiftControl:
    def __init__(self,world,hand_force_range=(2.,3.),include_contact_compliance=False):
        self.w=world;self.n=world.n;self.device=world.device;self.dt=world.dt
        self.monitor=GraspStabilityMonitor(self.n,self.device,window_steps=round(.3/self.dt),
            hand_force_range=hand_force_range,max_digit_ratio=1.5)
        self.force_range=tuple(hand_force_range);self.initial_pressure=sum(self.force_range)/2
        self.history_steps=round(1/(60*self.dt))
        self.history=world.t(torch.zeros(self.history_steps,self.n,2,5));self.count=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self.tick=0;self.dirty=True;self.samples=[];self.models=[];self.folds=[];self.shapes=[]
        self.rate=world.t(torch.zeros(self.n,2,6));self.pressure=world.t(torch.full((self.n,),self.initial_pressure))
        self.palm_load=world.t(torch.zeros(self.n,2))
        self.z=world.t(torch.zeros(self.n));self.started=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.hold=world.t(torch.zeros(self.n));self.longest=self.hold.clone();self.success=self.started.clone()
        self.attempt=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self.phase=torch.zeros_like(self.attempt);self.age=world.t(torch.zeros(self.n))
        self.paused=self.started.clone()
        self.last={};self.ready=self.started.clone();self.anchor_error=world.t(torch.full((self.n,),float('inf')))
        self.valid=torch.ones(self.n,device=self.device,dtype=torch.bool)
        self.include_contact_compliance=bool(include_contact_compliance);self.body_compliance=None
        for h,hand in enumerate(world.hands):
            fold=world.t(torch.zeros(len(hand.joint_names),6))
            for j,name in enumerate(ACTIVE):fold[hand.joint_names.index(name),j]=1.
            for follower,master,ratio,_ in MIMIC:fold[hand.joint_names.index(follower),ACTIVE.index(master)]=ratio
            self.folds.append(fold)
        for hi,body,vertices in world.shapes:
            surface,_,_=_surface(vertices.cpu().numpy(),count=96,seed=body)
            self.shapes.append((hi,body,torch.cat((vertices,world.t(surface)))))

    def reset(self,ids):
        self.history[:,ids]=0.;self.count[ids]=0;self.rate[ids]=0.;self.pressure[ids]=self.initial_pressure;self.palm_load[ids]=0.
        self.z[ids]=0.;self.started[ids]=False;self.hold[ids]=0.;self.ready[ids]=False
        self.phase[ids]=0;self.age[ids]=0.;self.attempt[ids]+=1;self.dirty=True
        self.paused[ids]=False
        if hasattr(self.monitor,'reset'):self.monitor.reset(ids)

    def _read_body_compliance(self):
        w=self.w;n=self.n
        result=[w.t(torch.zeros(n,len(hand.body_names))) for hand in w.hands]
        links=w.w.sensor_links
        for i,sensor in enumerate(w.w.sensors):
            h=i//len(links);b=w.hands[h].body_names.index(links[i%len(links)])
            properties=sensor.body_physx_view.get_material_properties().to(self.device).double()
            compliance=compliance_from_physx_material(properties).reshape(n,-1)
            if ((compliance-compliance[:,:1]).abs()>1e-12).any():
                raise RuntimeError('mixed native collider stiffness within a selected finger body')
            result[h][:,b]=compliance[:,0]
        return result

    def contact_model(self):
        w=self.w;n=self.n;rows=torch.arange(n,device=self.device)
        if self.include_contact_compliance and self.body_compliance is None:
            self.body_compliance=self._read_body_compliance()
        points=w.t(torch.zeros(n,2,5,3));outward=points.clone();body=torch.zeros(n,2,5,device=self.device,dtype=torch.long)
        gap=w.t(torch.full((n,2,5),float('inf')));depth=torch.zeros_like(gap);source_depth=torch.zeros_like(gap)
        observed=torch.zeros(n,2,5,device=self.device,dtype=torch.bool)
        stiffness_count=torch.ones_like(gap)
        table_points=[[],[]];table_bodies=[[],[]];clearances=[[],[]]
        for h,b,local in self.shapes:
            transform=pose_matrix(w.poses[:,h,b]);ps=local[None]@transform[:,:3,:3].transpose(-1,-2)+transform[:,:3,3,None].transpose(-1,-2)
            low=ps[:,:,2].argmin(-1);table_points[h].append(ps[rows,low]);table_bodies[h].append(b)
            clearances[h].append(ps[rows,low,2]-.30)
            name=w.hands[h].body_names[b]
            if not name.startswith('finger'):continue
            digit=int(name[6])-1
            sdf=finite_cylinder_sdf(ps,w.wp,w.wq);closest=sdf.abs().argmin(-1);distance=sdf[rows,closest]
            take=distance.abs()<gap[:,h,digit].abs()
            points[:,h,digit]=torch.where(take[:,None],ps[rows,closest],points[:,h,digit])
            body[:,h,digit]=torch.where(take,torch.full_like(body[:,h,digit],b),body[:,h,digit])
            gap[:,h,digit]=torch.where(take,distance,gap[:,h,digit])
            source_depth[:,h,digit]=torch.maximum(source_depth[:,h,digit],(-sdf).clamp_min(0).amax(-1))
        # Unloaded fallback is a geometric approach reference, not a contact.
        qw,qx,qy,qz=w.wq.unbind(-1)
        axis=torch.stack((2*(qx*qy-qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz+qw*qx)),-1)
        rel=points-w.wp[:,None,None];radial=rel-(rel*axis[:,None,None]).sum(-1,keepdim=True)*axis[:,None,None]
        outward=radial/radial.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        best_force=torch.full_like(gap,-1.);best_native_gap=torch.full_like(gap,float('inf'))
        for i,sensor in enumerate(w.w.sensors):
            name=w.w.sensor_links[i%len(w.w.sensor_links)]
            if not name.startswith('finger'):continue
            h=i//len(w.w.sensor_links);digit=int(name[6])-1;b=w.hands[h].body_names.index(name)
            force,p,normal,separation,count,start=sensor.contact_physx_view.get_contact_data(dt=w.dt)
            for env in range(n):
                if self.count[env]==0:continue
                number=int(count.reshape(-1)[env]);begin=int(start.reshape(-1)[env])
                if not number:continue
                selected=slice(begin,begin+number)
                f=force[selected].reshape(-1).double();pp=p[selected].double();nn=normal[selected].double();sep=separation[selected].reshape(-1).double()
                if not all(torch.isfinite(v).all() for v in (f,pp,nn,sep)):continue
                observed[env,h,digit]=True;depth[env,h,digit]=torch.maximum(depth[env,h,digit],(-sep.min()).clamp_min(0))
                load=f.clamp_min(0).sum();closest=sep.min()
                take=(load>best_force[env,h,digit]+1e-9) or (load<=1e-9 and best_force[env,h,digit]<=1e-9 and closest<best_native_gap[env,h,digit])
                if not take:continue
                weights=f.clamp_min(0);weights=weights/weights.sum() if weights.sum()>0 else torch.full_like(weights,1/number)
                point=(pp*weights[:,None]).sum(0);norm=(nn*weights[:,None]).sum(0)
                if norm.norm()<1e-8:continue
                points[env,h,digit]=point;outward[env,h,digit]=norm/norm.norm();body[env,h,digit]=b
                # Parallel active normal springs contribute their projected
                # stiffness along the selected aggregate contact normal.
                direction=norm/norm.norm()
                stiffness_count[env,h,digit]=projected_contact_stiffness_count(nn,f,direction)
                gap[env,h,digit]=closest;best_force[env,h,digit]=load;best_native_gap[env,h,digit]=closest
        jac=w.t(torch.zeros(n,2,5,3,6));zjac=[]
        for h,hand in enumerate(w.hands):
            native=hand.root_physx_view.get_jacobians().double();ids=body[:,h]
            for digit in range(5):
                b=ids[:,digit];j=native[rows,(b-1).clamp_min(0)]
                jac[:,h,digit]=point_jacobian(j,points[:,h,digit],hand.data.body_com_pos_w.double()[rows,b],self.folds[h])
            table=[]
            for b,point in zip(table_bodies[h],table_points[h]):
                j=native[:,max(b-1,0)]
                table.append(point_jacobian(j,point,hand.data.body_com_pos_w.double()[:,b],self.folds[h])[:,2])
            zjac.append(torch.stack(table,1))
        contact_compliance=torch.zeros_like(gap)
        if self.include_contact_compliance:
            contact_compliance=torch.stack([self.body_compliance[h][rows[:,None],body[:,h]] for h in range(2)],1)/stiffness_count
        self.last=dict(points=points,outward=outward,body_ids=body,jac=jac,normal_jac=torch.einsum('nhdc,nhdck->nhdk',-outward,jac),
            contact_compliance_m_n=contact_compliance,effective_native_patch_stiffness_count=stiffness_count,
            observed=observed,gap=gap,native_depth=depth,source_depth=source_depth,
            table_jac=torch.stack(zjac,1),table_clearance=torch.stack([torch.stack(v,1) for v in clearances],1),
            eligible=(gap.abs()<.030)|observed)

    def targets(self,command,seed,dt):
        w=self.w;self.age+=dt
        if self.dirty or self.tick%w.decimation==0:
            self.contact_model();self.dirty=False
            feedback=self.history.sum(0)/self.count.clamp_min(1)[:,None,None]
            finger_budget=(self.pressure[:,None].expand(-1,2)-self.palm_load).clamp_min(0)
            desired,sharing,seeking=sharing_targets(finger_budget,feedback,self.last['gap'],self.last['eligible'])
            q=command.reshape(self.n,2,6);low,high=(w.t(a)[None,None].expand_as(q) for a in active_limits())
            lower=((low-q)/(1/60)).clamp_min(-.12);upper=((high-q)/(1/60)).clamp_max(.12)
            retreat=(self.last['native_depth']>.0008)|(self.last['source_depth']>.0015)
            active=self.last['eligible']|sharing
            rate=force_joint_rates(self.last['normal_jac'],feedback,desired,active,lower,upper,1/60,retreat=retreat,seeking=seeking,
                contact_compliance_m_n=self.last['contact_compliance_m_n'] if self.include_contact_compliance else None)
            self.rate=table_safe_rates(rate,self.last['table_jac'],self.last['table_clearance'],1/60)
            self.last.update(desired=desired,feedback=feedback,retreat=retreat,sharing=sharing,seeking=seeking)
            model=dict(physics_tick=torch.tensor(self.tick,device=self.device),points_w=self.last['points'],
                outward=self.last['outward'],body_ids=self.last['body_ids'],normal_jac=self.last['normal_jac'],
                gap_m=self.last['gap'],native_depth_m=self.last['native_depth'],source_depth_m=self.last['source_depth'],
                contact_compliance_m_n=self.last['contact_compliance_m_n'],
                effective_native_patch_stiffness_count=self.last['effective_native_patch_stiffness_count'],
                feedback_n=feedback,desired_n=desired,sharing=sharing,seeking=seeking,retreat=retreat,
                rate_before_table=rate,rate_after_table=self.rate,q_command=command,q_measured=w.q,
                body_poses=w.poses,body_com_positions=torch.stack([hand.data.body_com_pos_w for hand in w.hands],1).double(),
                roots=w.roots,object_pose=torch.cat((w.wp,w.wq),-1),table_clearance_m=self.last['table_clearance'])
            self.models.append({k:v.detach().clone() for k,v in model.items()})
        newly=self.ready&~self.started
        if newly.any():
            points=rod_points_in_hands(torch.cat((w.wp,w.wq),-1),w.roots)
            self.monitor.latch(newly,points);self.started |= newly;self.phase[newly]=1
        okay=self.ready&self.started&(self.anchor_error<=.0005)
        # The new anchor is the current measured state, so it has zero error.
        okay |= newly
        measured_offsets=w.roots[:,:,:3]-w.root_seed[:,:,:3]
        measured=measured_offsets[:,:,2].amin(-1)
        rod_lift=(w.bottom-.30).clamp_min(0)
        up_rate=torch.where(self.phase==2,(.0305-rod_lift).clamp(0,.005),torch.full_like(rod_lift,.005))
        self.z=torch.where(okay,self.z+up_rate*dt,self.z)
        slipping=self.started&~okay
        newly_paused=slipping&~self.paused
        self.z=torch.where(newly_paused,torch.minimum(self.z,measured),self.z)
        w.palms=cancel_queued_ascent(w.palms,measured_offsets,newly_paused)
        self.paused=slipping
        # Recover position as well as pressure; never keep pulling away from
        # an object which remained on the table. All movement uses native drives.
        self.z=torch.where(slipping&(measured>rod_lift+.0005),self.z-.003*dt,self.z).clamp(0,.05)
        reached=self.started&(rod_lift>=.03)
        self.phase=torch.where(reached,torch.full_like(self.phase,2),self.phase)
        # Start at the user's band midpoint. Recovery may increase the command
        # inside that band; only raw measured force can qualify stability.
        margin=.1*(self.force_range[1]-self.force_range[0])
        self.pressure=(self.pressure+slipping*.05*dt).clamp(self.force_range[0]+margin,self.force_range[1]-margin)
        low,high=(w.t(a).repeat(2) for a in active_limits())
        return (command+self.rate.flatten(1)*dt).maximum(low).minimum(high),self.z.clone()

    def record_loads(self,loads):
        self.history[self.tick%self.history_steps]=loads
        self.count=(self.count+1).clamp_max(self.history_steps);self.tick+=1

    def after_physics(self,measurement):
        w=self.w;loads=measurement['grouped_object_loads'][:,:,:5]
        self.palm_load=measurement['grouped_object_loads'][:,:,5].clone()
        self.record_loads(loads)
        obj=torch.cat((w.wp,w.wq),-1)
        finite=valid_readback(obj,w.wv,measurement['q'],measurement['qd'],measurement['hand_object_loads'],
            w.poses,w.qd,measurement['joint_failure'],w.adapter_fault_bits,w.clearances)
        finite &= torch.isfinite(w.roots).flatten(1).all(-1)&((w.roots[:,:,3:].norm(dim=-1)-1).abs()<1e-3).all(-1)
        safe_obj=obj.clone();safe_roots=w.roots.clone()
        safe_obj[~finite]=w.t([0.,0.,0.,1.,0.,0.,0.]);safe_roots[~finite]=w.t([0.,0.,0.,1.,0.,0.,0.])
        relative=rod_points_in_hands(safe_obj,safe_roots)
        vectors=torch.stack([sensor.data.force_matrix_w[:,0,0,:] for sensor in w.w.sensors],1).double().reshape(self.n,2,-1,3)
        groups=torch.einsum('nhlc,ld->nhdc',vectors,w.group_matrix)[:,:,:5]
        normals=groups/groups.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        good=finite&(measurement['joint_failure']==0)&(w.adapter_fault_bits==0)
        result=self.monitor.update(loads,relative,normals,w.dt,good,torch.zeros_like(good),
            hand_total_load_n=measurement['hand_object_loads'])
        self.ready=result['ready'];self.anchor_error=self.monitor.anchor_error(relative)
        qualified=self.started&self.ready&(self.anchor_error<=.0005)&(w.bottom>=.33)&(w.wv[:,:3].norm(dim=-1)<.03)&good
        self.hold=torch.where(qualified,self.hold+w.dt,0.);self.longest=torch.maximum(self.longest,self.hold)
        self.success |= self.hold>=5.
        record=dict(loads_digit_n=loads,raw_digit_outward_normals_w=normals,relative_rod_points=relative,
            hand_total_load_n=measurement['hand_object_loads'],palm_load_n=self.palm_load,
            hand_force_in_range=result['hand_force_in_range'],hand_force_min_n=result['hand_force_min_n'],hand_force_max_n=result['hand_force_max_n'],
            digit_ratio_ok=result['digit_ratio_ok'],
            ready=self.ready,force_cv=result['force_cv'],loaded=result['loaded'],relative_drift_m=result['relative_drift_m'],
            slip_rate_m_s=result['slip_rate_m_s'],anchor_error_m=self.anchor_error,phase=self.phase,
            requested_hand_pressure_n=self.pressure[:,None].expand(-1,2),requested_digit_load_n=self.last.get('desired',torch.zeros_like(loads)),
            sharing=self.last['sharing'],seeking=self.last['seeking'],
            commanded_joint_rate=self.rate,commanded_wrist_height_m=self.z,hold_s=self.hold,qualified=self.hold>=5.,attempt=self.attempt,
            native_depth_m=self.last.get('native_depth',torch.zeros_like(loads)),source_depth_m=self.last.get('source_depth',torch.zeros_like(loads)))
        self.samples.append({k:v.detach().clone() for k,v in record.items()})

    def summary(self):
        return dict(stable_now=self.ready.cpu().tolist(),hold_s=self.hold.cpu().tolist(),longest_nonslip_hold_s=self.longest.cpu().tolist(),
            qualified=(self.hold>=5.).cpu().tolist(),ever_qualified=self.success.cpu().tolist(),
            successful_cases=self.success.nonzero().flatten().cpu().tolist(),required_hold_s=5.,
            force_goal='per_hand_summed_normal_force',hand_force_range_n=list(self.force_range),
            active_digit_policy='any_of_five;budget_shared_by_bearing_digits')
