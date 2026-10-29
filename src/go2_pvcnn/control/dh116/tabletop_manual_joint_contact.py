"""Manual hybrid contact-position/normal-force controller; no PPO path.

The six variables per hand are wrist translation and the three thumb/index
active angles. There is no palm rotation variable. PhysX COM Jacobians are
shifted to measured manifold points, then folded through the mimic ratios.
"""
import torch
from .linear_coupling import ACTIVE,MIMIC

CONTRACT=dict(version='manual_joint_contact_v1',target_radial_z_m=-.002,
    max_measured_digit_ratio=1.5,target_ratio_limit=1.4,
    force_velocity_gain_m_n_s=.00015,position_gain_s_inv=1.,
    max_contact_velocity_m_s=.002,max_joint_velocity_rad_s=.12,
    max_wrist_velocity_m_s=.003,joint_budget_rad=.08,
    regularization_m=.00001,physical_validation='pending',
    force_source='object_filtered_per_digit_normal_magnitude; friction_excluded',
    contact_source='strongest_loaded_link_raw_manifold_mean; fallback_source_pad_FK',
    stability_source='independent_raw_all_manifold_points_and_thumb_load_window',
    palm_rotation='unchanged',scope='explicit_manual_only')


def point_jacobian(com_jac,point,com,fold):
    angular=com_jac[...,3:,:].transpose(-1,-2)
    lever=(point-com)[...,None,:].expand_as(angular)
    at_point=com_jac[...,:3,:]+torch.linalg.cross(angular,lever).transpose(-1,-2)
    return at_point@fold


def bounded_dls(a,b,bias,lower,upper,damping):
    """Active-set bounded least squares, re-solving after each saturated DOF."""
    fixed=torch.zeros_like(bias);free=torch.ones_like(bias,dtype=torch.bool)
    eye=torch.eye(a.shape[-2],device=a.device,dtype=a.dtype)
    result=bias
    for _ in range(a.shape[-1]+1):
        af=a*free[...,None,:]
        prior=torch.where(free,bias,0.)
        residual=b-torch.einsum('...ik,...k->...i',a,fixed+prior)
        dual=torch.linalg.solve(af@af.transpose(-1,-2)+damping**2*eye,residual)
        result=fixed+prior+torch.einsum('...ki,...i->...k',af.transpose(-1,-2),dual)
        violation=torch.maximum(lower-result,result-upper).clamp_min(0.)*free
        largest=violation.argmax(-1)
        pin=torch.nn.functional.one_hot(largest,a.shape[-1]).bool()&(violation.amax(-1,keepdim=True)>1e-12)
        fixed=torch.where(pin,result.maximum(lower).minimum(upper),fixed)
        free &= ~pin
    return result.maximum(lower).minimum(upper)


def solve_velocity(jac,relative,loads,requested,bias,*,lift_velocity=None,axis=None,
                   angular_jac=None,contact_valid=None,velocity_lower=None,velocity_upper=None,
                   total_hand_load=None,requested_hand_force=None):
    """One damped least-squares solve for both contact forces and heights.

    jac: (N,2,2,3,6); relative: radial contact vectors; all in world axes.
    bias supplies bounded posture/manual wrist tracking, projected through the
    same solve. The solver never changes the measured load used for stability.
    """
    if not all(torch.isfinite(x).all() for x in (jac,relative,loads,requested,bias)):
        raise ValueError('finite measured contact model required')
    if axis is None:
        axis=relative.new_tensor([0.,1.,0.]).expand(relative.shape[0],3)
    axis=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    radial=relative-(relative*axis[:,None,None]).sum(-1,keepdim=True)*axis[:,None,None]
    inward=-radial/radial.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    pair=loads.sum(-1)
    fraction=torch.where(pair>.05,loads[:,:,0]/pair.clamp_min(.05),.5)
    fraction=fraction.clamp(1/2.4,1.4/2.4)
    desired=requested[:,:,None]*torch.stack((fraction,1-fraction),-1)
    normal_v=(CONTRACT['force_velocity_gain_m_n_s']*(desired-loads)).clamp(-.002,.002)
    normal_v=torch.where((desired-loads).abs()<=.03,0.,normal_v)
    # Raw height is perpendicular to the object's current centerline.
    vertical=relative.new_tensor([0.,0.,1.])-axis*axis[:,2:3]
    z_v=(CONTRACT['position_gain_s_inv']*(CONTRACT['target_radial_z_m']-radial[:,:,:,2])).clamp(-.002,.002)
    normal_j=torch.einsum('nhdc,nhdck->nhdk',inward,jac)
    height_j=torch.einsum('nc,nhdck->nhdk',vertical,jac)
    if angular_jac is not None:
        outward=-inward
        change_normal=torch.linalg.cross(angular_jac.transpose(-1,-2),
            outward[:,:,:,None,:].expand_as(angular_jac.transpose(-1,-2))).transpose(-1,-2)
        tangent_j=.008*torch.einsum('nc,nhdck->nhdk',vertical,change_normal)
        height_j=torch.where(contact_valid[:,:,:,None],tangent_j,height_j)
    if lift_velocity is not None:
        ff=lift_velocity[:,:,None].expand_as(z_v)
        if contact_valid is not None:ff=torch.where(contact_valid,0.,ff)
        z_v=z_v+ff
    total=loads.sum(-1) if total_hand_load is None else total_hand_load
    full_request=requested if requested_hand_force is None else requested_hand_force
    high_load=total>torch.minimum(full_request+2.,torch.full_like(full_request,12.))
    # Suspend placement/lift correction while unloading. Pressure correction
    # still uses the same constrained wrist/joint solve and linkage limits.
    height_j=torch.where(high_load[:,:,None,None],0.,height_j)
    z_v=torch.where(high_load[:,:,None],0.,z_v)
    bias=torch.where(high_load[:,:,None],0.,bias)
    along_j=torch.einsum('nc,nhdck->nhdk',axis,jac).mean(2,keepdim=True)
    a=torch.cat((normal_j,height_j,along_j),2)
    b=torch.cat((normal_v,z_v,torch.zeros_like(normal_v[:,:,:1])),2)
    # Normalize metres and radians before regularization; 0.02rad is roughly
    # 1mm at these fingers. This avoids treating 1rad as cheaply as 1metre.
    scale=jac.new_tensor([.001,.001,.001,.02,.02,.02])
    scaled=a*scale
    limits=jac.new_tensor([.003,.003,.003,.12,.12,.12])
    low=-limits.expand_as(bias) if velocity_lower is None else velocity_lower
    high=limits.expand_as(bias) if velocity_upper is None else velocity_upper
    # Slow the whole contact task before enforcing rate bounds. Clipping a
    # saturated least-squares solution can otherwise reverse one finger's
    # height correction to satisfy the faster, infeasible task on its partner.
    eye=torch.eye(a.shape[-2],device=a.device,dtype=a.dtype)
    residual=b-torch.einsum('nhik,nhk->nhi',a,bias)
    ideal=bias+scale*torch.einsum('nhki,nhi->nhk',scaled.transpose(-1,-2),
        torch.linalg.solve(scaled@scaled.transpose(-1,-2)+CONTRACT['regularization_m']**2*eye,residual))
    task_rate=torch.maximum(ideal[:,:,:3].norm(dim=-1)/.003,
                             ideal[:,:,3:].abs().amax(-1)/.12).clamp_min(1.)
    velocity=scale*bounded_dls(scaled,b/task_rate[:,:,None],bias/scale/task_rate[:,:,None],
                                low/scale,high/scale,CONTRACT['regularization_m'])
    ratio=torch.maximum(velocity[:,:,:3].norm(dim=-1)/CONTRACT['max_wrist_velocity_m_s'],
                        velocity[:,:,3:].abs().amax(-1)/CONTRACT['max_joint_velocity_rad_s']).clamp_min(1.)
    velocity=velocity/ratio[:,:,None]
    return velocity,dict(target_load_n=desired,contact_radial_z_m=radial[:,:,:,2],
                         requested_normal_velocity_m_s=normal_v,requested_z_velocity_m_s=z_v,
                         contact_height_jacobian=height_j,task_rate_reduction=task_rate,high_hand_load=high_load,
                         model_singular_values=torch.linalg.svdvals(scaled),
                         modeled_contact_velocity_m_s=torch.einsum('nhik,nhk->nhi',a,velocity))


class JointContactController:
    def __init__(self,t,w):
        if not getattr(t,'manual_control',False):
            raise ValueError('joint contact controller is manual only')
        self.t,self.w=t,w
        self.folds=[]
        for h in w.hands:
            fold=t.command.new_zeros((len(h.joint_names),3))
            for i,name in enumerate(ACTIVE[:3]):fold[h.joint_names.index(name),i]=1.
            for follower,master,ratio,_ in MIMIC[:2]:
                fold[h.joint_names.index(follower),ACTIVE.index(master)]=ratio
            self.folds.append(fold)
        self.previous_nominal=t.command.new_zeros((t.n,2,3))
        self.nominal_valid=torch.zeros(t.n,device=t.device,dtype=torch.bool)
        self.accepted_reference=t.reference.clone()
        self.diagnostic={}
        self.loaded_link_ids=None
        t.manual_joint_contact=self

    def contacts(self):
        t,w=self.t,self.w
        groups=w.group_matrix.argmax(-1)
        points=(w.pad_relative+w.wp[:,None]).reshape(t.n,2,2,3).clone()
        body_ids=torch.tensor([w.w.pad_ids[h] for h in range(2)],device=t.device).expand(t.n,-1,-1).clone()
        valid=torch.zeros((t.n,2,2),device=t.device,dtype=torch.bool)
        native=[h.root_physx_view.get_jacobians().double() for h in w.hands]
        for hi in range(2):
            for di in range(2):
                candidates=(groups==di).nonzero().flatten()
                selected=t.m['object_link_loads'][:,hi,candidates].argmax(-1)
                for ei in range(t.n):
                    li=int(candidates[selected[ei]])
                    if float(t.m['object_link_loads'][ei,hi,li])<.05:continue
                    sensor=w.w.sensors[hi*len(groups)+li]
                    _,p,_,_,count,start=sensor.contact_physx_view.get_contact_data(dt=w.dt)
                    k=int(count.reshape(t.n,-1)[ei,0]);lo=int(start.reshape(t.n,-1)[ei,0])
                    if k and torch.isfinite(p[lo:lo+k]).all():
                        points[ei,hi,di]=p[lo:lo+k].mean(0).double()
                        body_ids[ei,hi,di]=w.hands[hi].body_names.index(w.w.sensor_links[li])
                        valid[ei,hi,di]=True
        model=t.command.new_zeros((t.n,2,2,3,6))
        angular=torch.zeros_like(model)
        model[:,:,:,:,:3]=torch.eye(3,device=t.device,dtype=model.dtype)
        for hi,hand in enumerate(w.hands):
            for di in range(2):
                ids=body_ids[:,hi,di]
                rows=torch.arange(t.n,device=t.device)
                com=hand.data.body_com_pos_w.double()[rows,ids]
                j=native[hi][rows,ids-1]
                model[:,hi,di,:,3:]=point_jacobian(j,points[:,hi,di],com,self.folds[hi])
                angular[:,hi,di,:,3:]=j[:,3:,:]@self.folds[hi]
        self.loaded_link_ids=body_ids
        self.angular_model=angular;self.contact_valid=valid
        return model,points

    def control(self,actions,live,dt):
        from .tabletop_policy_force_region import requested_force
        from .tabletop_policy_control import CONTRACT as POLICY
        t,w=self.t,self.w
        requested=t.center+actions[:,:12]*torch.where(actions[:,:12]>=0,t.upper-t.center,t.center-t.lower)
        requested=requested.maximum(t.low).minimum(t.high)
        user_requested=requested.clone()
        reference_step=(requested-self.accepted_reference).clamp(-.12*dt,.12*dt)
        errors=t.m['mimic_errors'].reshape(t.n,2,5)
        for hi in range(2):
            for di,(_,master,_,_) in enumerate(MIMIC[:2]):
                k=hi*6+ACTIVE.index(master)
                reference_step[:,k]=torch.where(errors[:,hi,di].abs()>=.008,0.,reference_step[:,k])
        self.accepted_reference[live]+=reference_step[live]
        requested=self.accepted_reference
        lower=t.command.new_tensor(POLICY['palm_lower_m']);upper=t.command.new_tensor(POLICY['palm_upper_m'])
        action=actions[:,12:18].reshape(t.n,2,3)
        nominal=action*torch.where(action>=0,upper,-lower)
        nominal_velocity=((nominal-self.previous_nominal)/dt).clamp(-.01,.01)
        nominal_velocity=torch.where(self.nominal_valid[:,None,None],nominal_velocity,0.)
        self.previous_nominal[live]=nominal[live];self.nominal_valid[live]=True
        jac,points=self.contacts()
        current=t.command.reshape(t.n,2,6)[:,:,:3]
        reference=requested.reshape(t.n,2,6)[:,:,:3]
        bias=t.command.new_zeros((t.n,2,6))
        bias[:,:,:3]=(nominal-w.palms)*.1+nominal_velocity
        bias[:,:,3:]=(reference-current)*.1
        qw,qx,qy,qz=w.wq.unbind(-1)
        axis=torch.stack((2*(qx*qy-qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz+qw*qx)),-1)
        loads=t.m['grouped_object_loads'][:,:,:2]
        force=requested_force(actions)
        available=(force-t.m['grouped_object_loads'][:,:,2:].sum(-1)).clamp_min(0.)
        vl=t.command.new_full((t.n,2,6),-.12);vu=-vl
        vl[:,:,:3]=((lower-w.palms)/dt).clamp(-.003,.003)
        vu[:,:,:3]=((upper-w.palms)/dt).clamp(-.003,.003)
        qlow=torch.maximum(reference-.08,t.low.reshape(2,6)[None,:,:3])
        qhigh=torch.minimum(reference+.08,t.high.reshape(2,6)[None,:,:3])
        vl[:,:,3:]=((qlow-current)/dt).clamp(-.12,.12)
        vu[:,:,3:]=((qhigh-current)/dt).clamp(-.12,.12)
        errors=t.m['mimic_errors'].reshape(t.n,2,5)
        for di,(_,master,ratio,_) in enumerate(MIMIC[:2]):
            col=3+ACTIVE.index(master);error=errors[:,:,di]
            taper=((.012-error.abs())/.004).clamp(0.,1.)
            vu[:,:,col]=torch.where(error<0,vu[:,:,col]*taper,vu[:,:,col])
            vl[:,:,col]=torch.where(error>0,vl[:,:,col]*taper,vl[:,:,col])
            release=(error/ratio).clamp(-.12*dt,.12*dt)/dt
            release=release.maximum(vl[:,:,col]).minimum(vu[:,:,col])
            vl[:,:,col]=torch.where(error.abs()>.012,release,vl[:,:,col])
            vu[:,:,col]=torch.where(error.abs()>.012,release,vu[:,:,col])
        velocity,diag=solve_velocity(jac,points-w.wp[:,None,None],loads,available,bias,
                                     lift_velocity=nominal_velocity[:,:,2],axis=axis,
                                     angular_jac=getattr(self,'angular_model',None),
                                     contact_valid=getattr(self,'contact_valid',None),
                                     velocity_lower=vl,velocity_upper=vu,
                                     total_hand_load=t.m['hand_object_loads'],requested_hand_force=force)
        next_q=current+velocity[:,:,3:]*dt
        next_q=next_q.maximum(reference-.08).minimum(reference+.08)
        target=requested.clone().reshape(t.n,2,6)
        target[:,:,:3]=next_q
        # Continuous, bounded soft linkage governor; the unchanged 0.02rad
        # physical guard remains the final authority and isolates bad rows.
        target=target.flatten(1).maximum(t.low).minimum(t.high)
        applied=t.command+(target-t.command).clamp(-.12*dt,.12*dt)
        palms=(w.palms+velocity[:,:,:3]*dt).maximum(lower).minimum(upper)
        over=t.m['hand_object_loads']>torch.minimum(force.new_full(force.shape,12.),force+2.)
        applied=applied.maximum(t.low).minimum(t.high)
        t.command[live]=applied[live]
        t.joint_requested_target[live]=user_requested[live]
        t.joint_governed_target[live]=target[live]
        t.palm_targets[live]=palms[live];t.palm_actions[live]=action[live]
        t.policy_stiffness[live]=10.
        t.f_target[live]=force[live];t.f_requested[live]=force[live]
        t.f_filtered[live]=t.m['hand_object_loads'][live]
        t.manual_digit_targets[live]=diag['target_load_n'][live]
        t.manual_digit_offsets[live]=torch.stack(((applied.reshape(t.n,2,6)-requested.reshape(t.n,2,6))[:,:,:2].norm(dim=-1),
                                                  (applied.reshape(t.n,2,6)-requested.reshape(t.n,2,6))[:,:,2]),-1)[live]
        t.manual_mimic_limited[live]=(errors[:,:,:2].abs()>.008)[live]
        t.pressure_limited[live]=over[live];t.paused[live]=False;t.f_blocked[live]=False
        t.palm_clipped[live]=False
        self.diagnostic={k:v.detach() for k,v in diag.items()}
        self.diagnostic['contact_point_w']=points.detach()
        return live

    def reset(self,ids):
        # Prevent a reset or initial placement from becoming an artificial lift
        # velocity. No stale commanded displacement is replayed after reset.
        self.nominal_valid[ids]=False
        self.accepted_reference[ids]=self.t.reference[ids]
