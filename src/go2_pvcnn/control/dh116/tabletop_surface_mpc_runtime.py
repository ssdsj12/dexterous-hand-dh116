"""Optional inference-only local grip MPC around the student's wrist motion.

The model predicts pad kinematics, not object dynamics or frictional lift.
Native lift qualification remains in TensorEnv. Invalid/poor predictions use
the unchanged nominal student action and are recorded as model fallbacks.
"""
from types import SimpleNamespace

import torch
from .tabletop_effective_pad_runtime import _rotation


def intrinsic_world_velocity(seed_quat,angles,rates):
    x,y,_=angles.unbind(-1);rx,ry,rz=rates.unbind(-1)
    local=torch.stack((rx+y.sin()*rz,x.cos()*ry-x.sin()*y.cos()*rz,
                       x.sin()*ry+x.cos()*y.cos()*rz),-1)
    pose=torch.cat((torch.zeros_like(seed_quat[...,:3]),seed_quat),-1)
    return (_rotation(pose)@local[...,None]).squeeze(-1)


def nominal_joint_trajectory(command,actions,horizon,lower,upper):
    result=[];current=command.clone()
    for _ in range(horizon):
        current=(current+actions[:,:12].reshape_as(command).clamp(-1,1)*(.3/60)).maximum(lower).minimum(upper)
        result.append(current)
    return torch.stack(result,1)


def action_from_joint_target(actions,command,target):
    result=actions.clone()
    result[:,:12]=((target-command)/(.3/60)).flatten(1).clamp(-1,1)
    return result


def wrist_point_trajectories(actions,palms,angles,seed_quat,pivots,
                             points,targets,center,horizon):
    """Transport world points by bounded *command increments*, not tracking error.

    Wrist commands use exactly the free-wrist norm budget, limits and intrinsic
    quaternion composition at every 60 Hz step. The averaged two-wrist transport
    of target references is only a local geometric heuristic, not object motion
    or friction prediction. Measured wrist drive lag is not forcibly recovered.
    """
    from .tabletop_free_wrist_grasp import _limited_step,compose_rotation,CONTRACT
    low=palms.new_tensor(CONTRACT['palm_lower_m'])
    high=palms.new_tensor(CONTRACT['palm_upper_m'])
    def rotation(value):
        quat=compose_rotation(seed_quat,value)
        return _rotation(torch.cat((torch.zeros_like(palms),quat),-1))
    initial_rotation=rotation(angles)
    current_palms=palms.clone();current_angles=angles.clone()
    lever=points-pivots[:,:,None]
    target_lever=targets-center[:,None,None]
    motions=[];references=[]
    for _ in range(horizon):
        current_palms=(current_palms+_limited_step(actions[:,12:18].reshape_as(palms),
            CONTRACT['wrist_rate_m_s'],1/60)).maximum(low).minimum(high)
        current_angles=(current_angles+_limited_step(actions[:,18:24].reshape_as(angles),
            CONTRACT['rotation_rate_rad_s'],1/60)).clamp(
                -CONTRACT['rotation_limit_rad'],CONTRACT['rotation_limit_rad'])
        relative_rotation=rotation(current_angles)@initial_rotation.transpose(-1,-2)
        translation=current_palms-palms
        motions.append(translation[:,:,None]+
            (relative_rotation[:,:,None]@lever[...,None]).squeeze(-1)-lever)
        mean_rotation=relative_rotation.mean(1)
        references.append(center[:,None,None]+translation.mean(1)[:,None,None]+
            (mean_rotation[:,None,None]@target_lever[...,None]).squeeze(-1))
    return torch.stack(motions,1),torch.stack(references,1)


class NativeSurfaceMPC:
    def __init__(self,env):
        from .tabletop_surface_mpc import SurfaceGripMPC
        from .linear_coupling import ACTIVE,MIMIC
        self.env=env;self.w=env.world;self.t=env.task
        self.planner=SurfaceGripMPC()
        self.horizon=4
        w=self.w
        self.bodies=[];self.folds=[]
        for hand in w.hands:
            self.bodies.append([hand.body_names.index(name) for name in
                ('finger14_link','finger23_link','finger33_link','finger43_link','finger53_link')])
            fold=w.t(torch.zeros(len(hand.joint_names),6))
            for j,name in enumerate(ACTIVE):fold[hand.joint_names.index(name),j]=1.
            for follower,master,ratio,_ in MIMIC:
                fold[hand.joint_names.index(follower),ACTIVE.index(master)]=ratio
            self.folds.append(fold)
        self.local_hand=w.t(torch.zeros(w.n,2,5,3));self.local_rod=self.local_hand.clone()
        self.anchored=torch.zeros(w.n,2,5,device=w.device,dtype=torch.bool)
        self.last_prediction=None;self.pending=torch.zeros(w.n,device=w.device,dtype=torch.bool)
        self.bad_predictions=torch.zeros(w.n,device=w.device,dtype=torch.long)
        self.stats=dict(active_env_steps=0,changed_env_steps=0,model_fallback_env_steps=0,
            skipped_env_steps=0,prediction_peak_rad=0.,prediction_checks=0,
            nominal_cost_sum=0.,optimized_cost_sum=0.,
            model='local_pad_kinematics_not_object_dynamics',
            wrist_intent='unchanged_student_translation_and_intrinsic_XYZ',
            activation='actual_net_lift_at_least3mm_and_effective_opposition',
            force_prediction=False,force_precision_gate=False)

    def reset(self,ids):
        self.anchored[ids]=False;self.pending[ids]=False;self.bad_predictions[ids]=0
        self.local_hand[ids]=0;self.local_rod[ids]=0
        if self.last_prediction is not None:self.last_prediction[ids]=0
        self.planner.reset(ids)

    @torch.no_grad()
    def refine(self,nominal):
        """Keep nominal rows intact unless their own local model succeeds."""
        self.pending.zero_()
        result=nominal.clone()
        failed=torch.zeros_like(self.pending)
        ids=torch.arange(self.w.n,device=self.w.device)
        self._refine_ids(ids,nominal,result,failed)
        self.stats['model_fallback_env_steps']+=int(failed.sum())
        self.stats['skipped_env_steps']+=int((~self.pending&~failed).sum())
        return result

    def _reject(self,ids,failed,error):
        if not ids.numel():return
        failed[ids]=True
        self.anchored[ids]=False
        self.stats['last_model_error']=str(error)

    def _refine_ids(self,ids,nominal,result,failed):
        """Batch the normal path; split unclassified exceptions down to their row.

        Raw finite/feasibility masks remove known bad rows before calling core.
        Anchor captures and predictions are committed only after a successful
        solve, so retrying an exceptional sibling cannot recapture good anchors.
        """
        from .tabletop_surface_exploration import allocation
        from .tabletop_manual_joint_contact import point_jacobian
        w,t=self.w,self.t
        try:
            def take(value,shape,name,boolean=False):
                if (not isinstance(value,torch.Tensor) or value.shape!=shape
                        or value.device!=self.pending.device
                        or (value.dtype!=torch.bool if boolean else not value.is_floating_point())):
                    raise ValueError('invalid native MPC '+name+' shape/dtype/device')
                return value[ids]

            def finite_rows(values):
                mask=torch.ones(len(ids),device=w.device,dtype=torch.bool)
                for value in values:
                    mask &= torch.isfinite(value).reshape(len(ids),-1).all(-1)
                return mask

            loads=take(t.m['effective_pad_loads_n'],(w.n,2,5),'pad loads')
            vectors=take(t.m['effective_pad_vectors_w'],(w.n,2,5,3),'pad vectors')
            valid=take(t.m['effective_pad_valid'],(w.n,2,5),'pad validity',True)
            bottom=take(t.m['bottom'],(w.n,),'bottom')
            reference=torch.as_tensor(t.surface_reference_bottom_m,device=w.device,dtype=bottom.dtype)
            if reference.ndim==0:reference=reference.expand(w.n)
            reference=take(reference,(w.n,),'reference bottom')
            healthy=finite_rows((loads,vectors,bottom,reference))&(loads>=0).flatten(1).all(-1)
            self._reject(ids[~healthy],failed,'invalid contact/activation inputs')
            ids=ids[healthy]
            if not ids.numel():return
            loads=loads[healthy];vectors=vectors[healthy];valid=valid[healthy]
            bottom=bottom[healthy];reference=reference[healthy]
            valid=valid&(loads>.05)
            # Clear lost contacts even when every environment will return early.
            self.anchored[ids] &= valid
            subset=SimpleNamespace(m=dict(effective_pad_loads_n=loads,
                effective_pad_vectors_w=vectors,effective_pad_valid=valid))
            paired=allocation(subset)['paired'].all(-1)
            active=paired&(bottom>=reference+.003)&(self.bad_predictions[ids]<3)
            self.anchored[ids] &= active[:,None,None]
            ids=ids[active];valid=valid[active]
            if not ids.numel():return

            command=take(t.command,(w.n,12),'command').reshape(-1,2,6)
            raw=dict(q=take(t.m['q'],(w.n,12),'q').reshape(-1,2,6),command=command,
                actions=take(nominal,(w.n,24),'actions').to(command).clamp(-1,1),
                measured=take(t.m['effective_pad_points_w'],(w.n,2,5,3),'pad points').to(command),
                relative=take(t.m['object_relative'],(w.n,3),'object relative').to(command),
                object_quat=take(t.m['object_quat'],(w.n,4),'object quaternion').to(command),
                wrist_pose=take(t.m['wrist_pose'],(w.n,2,7),'wrist pose').to(command),
                roots=take(w.roots,(w.n,2,7),'roots').to(command),
                seed=take(w.root_seed,(w.n,2,7),'root seed').to(command),
                palms=take(w.palms,(w.n,2,3),'palms').to(command),
                angles=take(w.wrist_angles,(w.n,2,3),'wrist angles').to(command))
            hand_data=[]
            for h,hand in enumerate(w.hands):
                bids=self.bodies[h]
                poses=hand.data.body_link_pose_w[ids][:,bids].to(command)
                com=hand.data.body_com_pos_w[ids][:,bids].to(command)
                jacobian=hand.root_physx_view.get_jacobians()[ids][:,[b-1 for b in bids]].to(command)
                hand_data.append((poses,com,jacobian))
            healthy=finite_rows(raw.values())
            for poses,com,jacobian in hand_data:
                healthy &= finite_rows((poses,com,jacobian))
                healthy &= (poses[...,3:].norm(dim=-1)>0).all(-1)
            healthy &= raw['object_quat'].norm(dim=-1)>0
            for name in ('wrist_pose','roots','seed'):
                healthy &= (raw[name][...,3:].norm(dim=-1)>0).all(-1)
            from .linear_coupling import active_limits
            lower=t.low.reshape(2,6).to(command);upper=t.high.reshape(2,6).to(command)
            physical_low,physical_high=(command.new_tensor(v) for v in active_limits())
            lo=lower.maximum(physical_low);hi=upper.minimum(physical_high)
            if (not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or (lo>hi).any()):
                raise ValueError('infeasible MPC joint limits')
            healthy &= ((command+.005>=lo)&(command-.005<=hi)).flatten(1).all(-1)
            self._reject(ids[~healthy],failed,'invalid native MPC state/kinematics/command')
            ids=ids[healthy];valid=valid[healthy]
            if not ids.numel():return
            raw={key:value[healthy] for key,value in raw.items()}
            hand_data=[tuple(value[healthy] for value in row) for row in hand_data]
            q=raw['q'].to(command);command=raw['command'];actions=raw['actions']
            planned=nominal_joint_trajectory(command,actions,self.horizon,lower,upper)
            center=raw['relative']+raw['wrist_pose'][...,:3].mean(1)
            rod_rotation=_rotation(torch.cat((center,raw['object_quat']),-1))
            measured=raw['measured']
            local_hand=self.local_hand[ids].clone();local_rod=self.local_rod[ids].clone()
            points=[];jac=[]
            for h,(poses,com,native) in enumerate(hand_data):
                rotation=_rotation(poses)
                capture=valid[:,h]&~self.anchored[ids,h]
                local=(rotation.transpose(-1,-2)@(measured[:,h]-poses[...,:3])[...,None]).squeeze(-1)
                local_hand[:,h]=torch.where(capture[...,None],local,local_hand[:,h])
                rod_local=(rod_rotation[:,None].transpose(-1,-2)@(measured[:,h]-center[:,None])[...,None]).squeeze(-1)
                local_rod[:,h]=torch.where(capture[...,None],rod_local,local_rod[:,h])
                material=(rotation@local_hand[:,h,...,None]).squeeze(-1)+poses[...,:3]
                hand_j=[]
                for d in range(5):
                    hand_j.append(point_jacobian(native[:,d],material[:,d],com[:,d],self.folds[h].to(command)))
                points.append(material);jac.append(torch.stack(hand_j,1))
            points=torch.stack(points,1);jac=torch.stack(jac,1)
            targets=(rod_rotation[:,None,None]@local_rod[...,None]).squeeze(-1)+center[:,None,None]
            wrist_motion,target_points=wrist_point_trajectories(actions,raw['palms'],raw['angles'],
                raw['seed'][...,3:],raw['roots'][...,:3],points,targets,center,self.horizon)
            weights=valid.to(q.dtype)/valid.sum(-1,keepdim=True).clamp_min(1)
            target,diag=self.planner.plan(q=q,command=command,nominal=planned,points=points,
                jacobian=jac,target_points=target_points,wrist_motion=wrist_motion,
                weights=weights,lower=lower,upper=upper,response=.5)
            proposed=action_from_joint_target(nominal[ids],command,target)
            if not torch.isfinite(proposed).all():raise ValueError('nonfinite MPC correction')
            prediction=diag['predicted_joints'][:,0]
            if prediction.shape!=q.shape or not torch.isfinite(prediction).all():
                raise ValueError('invalid MPC joint prediction')
            nominal_cost=float(diag['nominal_cost'].sum());optimized_cost=float(diag['optimized_cost'].sum())
            if self.last_prediction is None:self.last_prediction=q.new_zeros(w.n,2,6)
            self.local_hand[ids]=local_hand;self.local_rod[ids]=local_rod
            self.anchored[ids] |= valid
            result[ids]=proposed;self.pending[ids]=True
            self.last_prediction[ids]=prediction
            self.stats['active_env_steps']+=len(ids)
            self.stats['changed_env_steps']+=int(((proposed-nominal[ids]).abs().amax(-1)>1e-6).sum())
            self.stats['nominal_cost_sum']+=nominal_cost
            self.stats['optimized_cost_sum']+=optimized_cost
        except (ValueError,RuntimeError,IndexError,KeyError) as error:
            if len(ids)>1:
                for subset_ids in ids.chunk(2):self._refine_ids(subset_ids,nominal,result,failed)
            else:self._reject(ids,failed,error)

    @torch.no_grad()
    def observe_result(self,done):
        mask=self.pending&~done
        if self.last_prediction is not None and bool(mask.any()):
            error=(self.t.m['q'].reshape_as(self.last_prediction)-self.last_prediction).abs().amax((1,2))
            self.stats['prediction_checks']+=int(mask.sum())
            self.stats['prediction_peak_rad']=max(self.stats['prediction_peak_rad'],float(error[mask].max()))
            self.bad_predictions[mask]=(self.bad_predictions[mask]+(error[mask]>.005).long())
            self.stats['model_fallback_env_steps']+=int((mask&(error>.005)).sum())
        self.reset(done.nonzero().flatten())
