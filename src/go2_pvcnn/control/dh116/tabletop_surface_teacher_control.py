"""Diagnostic force/position teacher in the existing26D command space.

No production PPO dispatch is changed. A timed demonstration attempts ascent
without contact admission; failed contact is recorded by independent acceptance.
"""
import math
import torch
from .tabletop_manual_joint_contact import bounded_dls
from .linear_coupling import ACTIVE,MIMIC


def transform_contact_seeds(poses,body_ids,local_points):
    """Transform each candidate's local point on its own actual rigid link."""
    from .tabletop_surface_kinematics import pose_matrix
    n=poses.shape[0]
    if (poses.ndim!=4 or poses.shape[1]!=2 or poses.shape[-1]!=7
            or body_ids.shape!=(n,2,2) or local_points.shape!=(n,2,2,3)
            or body_ids.dtype!=torch.long or (body_ids<0).any() or (body_ids>=poses.shape[2]).any()
            or not torch.isfinite(poses).all() or not torch.isfinite(local_points).all()):
        raise ValueError('finite per-candidate body poses and valid contact seed links required')
    selected=poses.gather(2,body_ids[...,None].expand(-1,-1,-1,7))
    matrix=pose_matrix(selected.reshape(-1,7)).reshape(n,2,2,4,4)
    return (matrix[...,:3,:3]@local_points[...,None]).squeeze(-1)+matrix[...,:3,3]


def solve_surface_velocity(jac,inward,loads,up_velocity,lower,upper,*,desired=None,object_velocity=None,axis=None,wrist_locked=False,control_dt=1/60):
    n=jac.shape[0]
    if jac.shape!=(n,2,2,3,12) or inward.shape!=(n,2,2,3) or loads.shape!=(n,2,2):
        raise ValueError('two-hand two-digit twelve-variable contact model required')
    if lower.shape!=(n,2,12) or upper.shape!=lower.shape or up_velocity.shape!=(n,2):raise ValueError('matching velocity bounds required')
    if not all(torch.isfinite(v).all() for v in (jac,inward,loads,up_velocity,lower,upper)) or (lower>upper).any():
        raise ValueError('finite consistent contact model and bounds required')
    if wrist_locked:
        lower=lower.clone();upper=upper.clone()
        lower[:,:,:6]=0.;upper[:,:,:6]=0.;up_velocity=torch.zeros_like(up_velocity)
    desired=torch.full_like(loads,2.5) if desired is None else desired.to(loads)
    axis=jac.new_tensor([0.,1.,0.]).expand(n,-1) if axis is None else axis.to(jac)
    object_velocity=jac.new_zeros((n,3)) if object_velocity is None else object_velocity.to(jac)
    normal_j=torch.einsum('nhdc,nhdck->nhdk',inward,jac)
    normal_v=(.0006*(desired-loads)).clamp(-.003,.003)
    if wrist_locked:
        if not math.isfinite(control_dt) or control_dt<=0:raise ValueError('positive control dt required')
        # The real free-wrist position drives use 10Nm/rad. Convert force
        # error through their Cartesian compliance, instead of requesting the
        # same displacement for every lever arm (unstable near small J).
        finger_j=normal_j[...,6:]
        compliance=(finger_j@finger_j.transpose(-1,-2))/10.
        held_velocity=.2/control_dt*(compliance@(desired-loads).to(jac)[...,None]).squeeze(-1)
        normal_v=torch.where(loads>.05,held_velocity,normal_v).clamp(-.003,.003)
    normal_v+=torch.einsum('nhdc,nc->nhd',inward,object_velocity)
    z_j=jac[:,:,:,2,:]
    along_j=torch.einsum('nc,nhdck->nhdk',axis,jac)
    along_v=(axis*object_velocity).sum(-1)[:,None,None].expand(-1,2,2)
    a=torch.cat((normal_j,z_j,along_j),2)
    b=torch.cat((normal_v,up_velocity[:,:,None].expand(-1,-1,2),along_v),2)
    if wrist_locked:
        # A coupled finger closes on an arc. Fixing its Cartesian height and
        # axial position competes with its only closing DOF. In pinch/hold,
        # keep the wrist command and let real finger kinematics follow the
        # normal force error, within the original joint/torque limits.
        a,b=normal_j,normal_v
    bias=jac.new_zeros((n,2,12));bias[:,:,2]=up_velocity
    # The demonstration explicitly moves the external wrist upward. Without
    # this lower bound, the redundant solve can satisfy fingertip Z motion
    # entirely with flexion while cancelling the intended wrist ascent.
    lower=lower.clone()
    planned=torch.minimum(up_velocity,upper[:,:,2])
    lower[:,:,2]=torch.where(up_velocity>0,torch.maximum(lower[:,:,2],planned),lower[:,:,2])
    scale=jac.new_tensor([.001]*3+[.02]*9)
    v=bounded_dls(a*scale,b,bias/scale,lower/scale,upper/scale,1e-5)*scale
    # Translation and rotation have independent actuator budgets. Preserve the
    # scheduled Z component and allocate the remaining Cartesian norm to X/Y.
    xy_budget=(.01**2-v[:,:,2].square()).clamp_min(0).sqrt()
    xy_norm=v[:,:,:2].norm(dim=-1).clamp_min(1e-12)
    xy=v[:,:,:2]*(xy_budget/xy_norm).clamp_max(1.)[:,:,None]
    rotation=v[:,:,3:6]/(v[:,:,3:6].norm(dim=-1,keepdim=True)/.5).clamp_min(1.)
    return torch.cat((xy,v[:,:,2:3],rotation,v[:,:,6:].clamp(-.3,.3)),-1)


class SurfaceTeacherControl:
    def __init__(self,task,world,models,ascent_start_s=2.,seed_local_contacts=None,seed_body_names=None,pinch_hold=False):
        if not math.isfinite(ascent_start_s) or ascent_start_s<0:raise ValueError('finite nonnegative ascent time required')
        self.ascent_start_s=ascent_start_s
        self.t,self.w,self.models=task,world,models;self.folds=[]
        from .tabletop_surface_force_goals import ForceGoalExplorer
        self.pinch_hold=bool(pinch_hold)
        self.force_goals=ForceGoalExplorer(task.n,task.device,initial_max_n=3. if self.pinch_hold else 12.)
        task.pinch_force_hold=self.pinch_hold
        if self.pinch_hold:self.force_goals.hold()
        self.seed_local_contacts=None if seed_local_contacts is None else task.command.new_tensor(seed_local_contacts)
        if self.seed_local_contacts is not None and (self.seed_local_contacts.shape!=(task.n,2,2,3)
                or not torch.isfinite(self.seed_local_contacts).all()):raise ValueError('finite N x 2 x 2 local contact seeds required')
        self.seed_body_ids=None
        if seed_body_names is not None and self.seed_local_contacts is None:
            raise ValueError('local contact points required with seed links')
        if self.seed_local_contacts is not None:
            names=seed_body_names if seed_body_names is not None else [
                [[p['body'] for p in model.patches] for model in models] for _ in range(task.n)]
            if len(names)!=task.n or any(len(row)!=2 or any(len(pair)!=2 for pair in row) for row in names):
                raise ValueError('N x 2 hands x 2 digits seed links required')
            indices=[]
            for row in names:
                pair=[]
                for h,digits in enumerate(row):
                    links=[name.rsplit('/',1)[-1] for name in digits]
                    if any(not link.startswith('finger'+str(d+1)) for d,link in enumerate(links)):
                        raise ValueError('seed contacts must belong to thumb and index respectively')
                    pair.append([world.hands[h].body_names.index(link) for link in links])
                indices.append(pair)
            self.seed_body_ids=torch.tensor(indices,device=task.device,dtype=torch.long)
        self.steps=0;self.last={}
        for hi,hand in enumerate(world.hands):
            fold=task.command.new_zeros((len(hand.joint_names),9))
            for i,j in enumerate(world.w.mounts[hi].angular_ids):fold[j,i]=1.
            for i,name in enumerate(ACTIVE):fold[hand.joint_names.index(name),3+i]=1.
            for follower,master,ratio,_ in MIMIC:fold[hand.joint_names.index(follower),3+ACTIVE.index(master)]=ratio
            self.folds.append(fold)

    def actual_poses(self):
        from .tabletop_surface_kinematics import pose_matrix
        return [{body:pose_matrix(self.w.poses[:,h,self.w.hands[h].body_names.index(body.rsplit('/',1)[-1])])
            for body in model.bodies} for h,model in enumerate(self.models)]

    def refresh_contact_model(self,measurement=None):
        """Read the current contact link, point and native Jacobian each substep."""
        t,w=self.t,self.w;n=t.n
        measurement=t.m if measurement is None else measurement
        if self.seed_local_contacts is not None:
            points=transform_contact_seeds(w.poses,self.seed_body_ids,self.seed_local_contacts)
            body_ids=self.seed_body_ids.clone()
        else:
            from .tabletop_surface_kinematics import pose_matrix,transform
            from .tabletop_surface_objectives import finite_cylinder_sdf
            points=[];body_ids=[]
            for hi,model in enumerate(self.models):
                ids=torch.tensor([w.hands[hi].body_names.index(p['body'].rsplit('/',1)[-1])
                    for p in model.patches],device=t.device).expand(n,-1).clone()
                sampled=torch.stack([transform(pose_matrix(w.poses[:,hi,ids[0,digit]]),p['points'])
                    for digit,p in enumerate(model.patches)],1)
                distance=finite_cylinder_sdf(sampled,w.wp,w.wq).abs()
                selected=distance.topk(3,dim=-1,largest=False).indices
                points.append(sampled.gather(2,selected[:,:,:,None].expand(-1,-1,-1,3)).mean(2))
                body_ids.append(ids)
            points=torch.stack(points,1);body_ids=torch.stack(body_ids,1)
        # When bearing, follow the actual strongest loaded link contact report.
        groups=w.group_matrix.argmax(-1);count=len(groups);rows=torch.arange(n,device=t.device)
        contact_valid=torch.zeros(n,2,2,device=t.device,dtype=torch.bool)
        for hi,hand in enumerate(w.hands):
            for digit in range(2):
                candidates=(groups==digit).nonzero().flatten()
                value,choice=measurement['object_link_loads'][:,hi,candidates].max(-1)
                selected=candidates[choice]
                p=torch.stack([s.data.contact_pos_w[:,0,0,:] for s in w.w.sensors[hi*count:(hi+1)*count]],1).double()[rows,selected]
                ids=torch.tensor([hand.body_names.index(link) for link in w.w.sensor_links],device=t.device)[selected]
                # A finger may distribute >0.05N over several weaker links.
                # Select any actual positive strongest-link report; feedforward
                # still requires a bearing digit and a finite actual point.
                good=(value>0)&torch.isfinite(p).all(-1)
                contact_valid[:,hi,digit]=good
                points[:,hi,digit]=torch.where(good[:,None],p,points[:,hi,digit])
                body_ids[:,hi,digit]=torch.where(good,ids,body_ids[:,hi,digit])
        jac=t.command.new_zeros((n,2,2,3,12));jac[:,:,:,:,:3]=torch.eye(3,device=t.device,dtype=jac.dtype)
        for hi,hand in enumerate(w.hands):
            native=hand.root_physx_view.get_jacobians().double()
            for digit in range(2):
                ids=body_ids[:,hi,digit];j=native[rows,ids-1]
                com=hand.data.body_com_pos_w.double()[rows,ids]
                lever=(points[:,hi,digit]-com)[:,None,:].expand(-1,j.shape[-1],-1)
                shifted=j[:,:3,:]+torch.linalg.cross(j[:,3:,:].transpose(-1,-2),lever).transpose(-1,-2)
                jac[:,hi,digit,:,3:]=shifted@self.folds[hi]
        qw,qx,qy,qz=w.wq.unbind(-1)
        axis=torch.stack((2*(qx*qy-qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz+qw*qx)),-1)
        relative=points-w.wp[:,None,None]
        radial=relative-(relative*axis[:,None,None]).sum(-1,keepdim=True)*axis[:,None,None]
        inward=-radial/radial.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        self.last.update(points=points,body_ids=body_ids,jacobian=jac,inward=inward,axis=axis,
            contact_model_valid=contact_valid)
        return self.last

    def actions(self,elapsed_s,dt=1/60):
        t,w=self.t,self.w;n=t.n
        normal_goals=self.force_goals.step(dt)
        self.refresh_contact_model()
        jac,inward,axis=(self.last[key] for key in ('jacobian','inward','axis'))
        lo=t.command.new_full((n,2,12),-.3);hi=-lo
        lo[:,:,:3]=((t.command.new_tensor([-.03,-.03,-.004])-w.palms)/dt).clamp(-.01,.01)
        hi[:,:,:3]=((t.command.new_tensor([.03,.03,.05])-w.palms)/dt).clamp(-.01,.01)
        lo[:,:,2]=torch.maximum(lo[:,:,2],-(w.clearances-.0003).clamp_min(0.)/dt)
        lo[:,:,3:6]=((-torch.pi/2-w.wrist_angles)/dt).clamp(-.5,.5)
        hi[:,:,3:6]=((torch.pi/2-w.wrist_angles)/dt).clamp(-.5,.5)
        cmd=t.command.reshape(n,2,6)
        lo[:,:,6:]=((t.low.reshape(2,6)-cmd)/dt).clamp(-.3,.3)
        hi[:,:,6:]=((t.high.reshape(2,6)-cmd)/dt).clamp(-.3,.3)
        ascending=elapsed_s>=self.ascent_start_s and not self.pinch_hold
        up=torch.full((n,2),.003 if ascending else 0.,device=t.device,dtype=t.command.dtype)
        up=torch.where(w.palms[:,:,2]>=.025,0.,up)
        other=t.m['grouped_object_loads'][:,:,2:].sum(-1)
        desired=((normal_goals-other).clamp_min(0.)/2)[:,:,None].expand(-1,-1,2)
        loads=(getattr(self,'force_feedback_loads',t.m['grouped_object_loads'][:,:,:2])
            if self.pinch_hold else t.m['grouped_object_loads'][:,:,:2])
        velocity=solve_surface_velocity(jac,inward,loads,up,lo,hi,
            desired=desired,object_velocity=w.wv[:,:3],axis=axis,wrist_locked=self.pinch_hold,control_dt=dt)
        action=t.command.new_zeros((n,26));action[:,-2:]=self.force_goals.actions()
        action[:,12:18]=(velocity[:,:,:3]/.01).flatten(1)
        action[:,18:24]=(velocity[:,:,3:6]/.5).flatten(1)
        action[:,:12]=(velocity[:,:,6:]/.3).flatten(1)
        self.last.update(up_velocity=up,desired=desired,requested_normal_n=normal_goals,
            lift_requested=torch.full((n,),ascending,device=t.device,dtype=torch.bool))
        self.steps+=1
        return action.clamp(-1,1)
