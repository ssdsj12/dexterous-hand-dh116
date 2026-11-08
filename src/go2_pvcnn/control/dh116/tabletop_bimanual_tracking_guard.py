"""Opt-in simulation wrist projection for measured finger tracking failure.

The guard withdraws that hand's wrist requests while its original finger
controller continues. Returned target cancellations use real measurements in
the existing SDK chart. The caller applies them before ordinary control and
all existing independent physical guards.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch

from .linear_coupling import ACTIVE
from .tabletop_free_wrist_grasp import CONTRACT as WRIST, compose_rotation, multiply
from .tabletop_policy_control import CONTRACT as POLICY


CONTRACT = dict(
    version='dh116-bimanual-tracking-wrist-projection-v2', default_enabled=False,
    default_coupled_pause=False,
    source='privileged_native_simulation_measurements', hardware_validated=False,
    measurement_phase='current_native_readback_before_the_next_physics_substep',
    action_size=24, actor_size=3844, critic_size=371,
    tracking_error_limit_rad=.03,
    tracking_limit_source='same_early_motion_admission_as_Z3_coordination_experiment',
    mimic_error_limit_rad=POLICY['severe_coupling_rad'],
    mimic_limit_source='unchanged_policy_owned_grasp_v2_severe_coupling_rad',
    rotation_chart='authored_root_seed_wxyz_times_intrinsic_X_then_Y_then_Z',
    rotation_limit_rad=WRIST['rotation_limit_rad'],
    rotation_reproduction_tolerance_rad=1e-6,
    rotation_precision_floor='128_times_floating_dtype_epsilon',
    projection='per_hand_wrist_translation_and_rotation_zero;finger_requests_preserved',
    coupled_projection='optional_same_environment_partner_wrist_pause_on_either_hand_tracking_failure',
    cancellation='measured_pose_in_original_SDK_target_chart;unavailable_keeps_existing_target',
    translation_cancellation_bounds_source='unchanged_free_wrist_grasp_v1_policy_palm_bounds',
    checkpoint_activation='explicit_experiment_only;no_automatic_checkpoint_activation',
)


@dataclass
class TrackingGuardState:
    active_tracking_error_rad: torch.Tensor
    mimic_error_rad: torch.Tensor
    live: torch.Tensor
    measured_palm_offset_m: torch.Tensor
    actual_base_pose_w: torch.Tensor
    wrist_seed_pose_w: torch.Tensor
    current_palm_target_m: torch.Tensor
    current_wrist_angles_rad: torch.Tensor
    measured_wrist_angles_rad: torch.Tensor | None = None
    tracking_target_source: str = 'caller_supplied_simulation_target_comparison'
    measurement_physics_step: int | None = None
    measurement_timestamp_s: float | None = None


def _quaternion(q):
    finite=torch.isfinite(q).all(-1)
    safe=torch.where(finite[...,None],q,0.)
    norm=safe.norm(dim=-1,keepdim=True)
    valid=finite & torch.isfinite(norm[...,0]) & ((norm[...,0]-1).abs()<=1e-5)
    identity=torch.zeros_like(q)
    identity[...,0]=1.
    return torch.where(valid[...,None],safe/norm.clamp_min(torch.finfo(q.dtype).tiny),identity),valid


def _conjugate(q):
    return torch.cat((q[...,:1],-q[...,1:]),-1)


def intrinsic_xyz_from_measured_pose(seed_wxyz, actual_wxyz, current_angles_rad,
                                    measured_angles_rad=None):
    """Recover a legal intrinsic XYZ chart and verify it against actual pose.

    Native carriage angles disambiguate the physical configuration at gimbal
    lock. Without them, the inverse chooses the closest current legal chart.
    An out-of-bounds native angle never switches to a different Euler branch.
    """
    seed,actual,current=seed_wxyz,actual_wxyz,current_angles_rad
    if (not isinstance(seed,torch.Tensor) or not seed.is_floating_point()
            or seed.shape[-1]!=4 or actual.shape!=seed.shape
            or current.shape!=seed.shape[:-1]+(3,)
            or actual.dtype!=seed.dtype or current.dtype!=seed.dtype
            or actual.device!=seed.device or current.device!=seed.device):
        raise ValueError('matching floating wxyz quaternions and intrinsic XYZ target charts required')
    seed,seed_valid=_quaternion(seed)
    actual,actual_valid=_quaternion(actual)
    limit=WRIST['rotation_limit_rad']
    if measured_angles_rad is not None:
        if (measured_angles_rad.shape!=current.shape or measured_angles_rad.dtype!=current.dtype
                or measured_angles_rad.device!=current.device):
            raise ValueError('matching measured intrinsic wrist angles required')
        candidate=measured_angles_rad.clone()
    else:
        relative=multiply(_conjugate(seed),actual)
        w,x,y,z=relative.unbind(-1)
        r00=1-2*(y*y+z*z)
        r01=2*(x*y-w*z)
        r02=2*(x*z+w*y)
        r12=2*(y*z-w*x)
        r22=1-2*(x*x+y*y)
        cosine=torch.sqrt(r00.square()+r01.square())
        b=torch.atan2(r02,cosine)
        a=torch.atan2(-r12,r22)
        c=torch.atan2(-r01,r00)
        # At b=+/-pi/2 the chart constrains a+sign(b)*c. Solve its
        # closest point to the current physical target inside the old box.
        locked=cosine<=32*torch.finfo(current.dtype).eps
        total=torch.atan2(2*(y*z+w*x),1-2*(x*x+z*z))
        sign=torch.where(r02>=0,1.,-1.)
        reference=torch.where(torch.isfinite(current),current,0.)
        free_c=(reference[...,2]+sign*(total-reference[...,0]))/2
        low=(sign*total-limit).clamp_min(-limit)
        high=(sign*total+limit).clamp_max(limit)
        locked_c=free_c.maximum(low).minimum(high)
        locked_a=total-sign*locked_c
        a=torch.where(locked,locked_a,a)
        c=torch.where(locked,locked_c,c)
        candidate=torch.stack((a,b,c),-1)
    finite=torch.isfinite(candidate).all(-1)
    within=finite & (candidate.abs()<=limit).all(-1)
    safe_candidate=torch.where(finite[...,None],candidate,0.)
    reproduced=compose_rotation(seed,safe_candidate)
    delta=multiply(_conjugate(actual),reproduced)
    error=2*torch.atan2(delta[...,1:].norm(dim=-1),delta[...,0].abs())
    tolerance=max(CONTRACT['rotation_reproduction_tolerance_rad'],128*torch.finfo(current.dtype).eps)
    pose_valid=seed_valid & actual_valid & finite
    valid=pose_valid & within & (error<=tolerance)
    error=torch.where(pose_valid,error,float('nan'))
    return torch.where(valid[...,None],candidate,current).clone(),valid,error


def tracking_state_from_native(world,task):
    """Read actual SDK buffers, physical joint names and authored wrist seed.

    The next native target is root_seed + palm_target + current task.offset.
    Its measured cancellation offset is computed in that same current chart.
    """
    root=world.roots
    n=world.n
    targets,sources=[],[]
    for hi,hand in enumerate(world.hands):
        ids=world.w.ids[hi]
        if tuple(hand.joint_names[i] for i in ids)!=ACTIVE:
            raise ValueError('native active joint names do not match the physical action contract')
        native=getattr(hand.data,'joint_pos_target',None)
        targets.append(native[:,ids].to(root) if native is not None else world.commands[:,hi].float().to(root))
        sources.append('native_articulation_joint_pos_target' if native is not None
                       else 'float32_submission_reconstructed')
    target=torch.stack(targets,1)
    actual=task.m['q'].reshape(n,2,6).to(root)
    measured_offset=root[:,:,:3]-world.root_seed[:,:,:3].to(root)
    offset=torch.as_tensor(task.offset,device=root.device,dtype=root.dtype)
    if offset.shape!=(n,):
        raise ValueError('one current common lift offset per environment required')
    measured_offset[:,:,2]-=offset[:,None]
    measured_angles=None
    mounts=getattr(world.w,'mounts',None)
    if mounts is not None:
        if len(mounts)!=2 or any(len(m.angular_ids)!=3 for m in mounts):
            raise ValueError('complete actual intrinsic XYZ wrist joints required')
        measured_angles=torch.stack([world.q[:,hi,m.angular_ids].to(root) for hi,m in enumerate(mounts)],1)
    return TrackingGuardState(
        active_tracking_error_rad=(target-actual).abs().amax(-1),
        mimic_error_rad=task.m['mimic_errors'].reshape(n,2,5).to(root).abs().amax(-1),
        live=(~task.done).clone(), measured_palm_offset_m=measured_offset,
        actual_base_pose_w=root.clone(), wrist_seed_pose_w=world.root_seed.to(root).clone(),
        current_palm_target_m=world.palms.to(root).clone(),
        current_wrist_angles_rad=world.wrist_angles.to(root).clone(),
        measured_wrist_angles_rad=measured_angles,
        tracking_target_source=sources[0] if sources[0]==sources[1] else '|'.join(sources),
        measurement_physics_step=getattr(world,'_physics_steps',None),
        measurement_timestamp_s=float(world.w.sim.current_time) if hasattr(world.w,'sim') else None)


class BimanualTrackingGuard:
    """Stateless per-substep projection; the default leaves commands exact."""
    def __init__(self,n,*,enabled=False,coupled_pause=False):
        if (type(n) is not int or n<1 or type(enabled) is not bool
                or type(coupled_pause) is not bool):
            raise ValueError('positive environment count and explicit boolean enable/profile required')
        self.n,self.enabled,self.coupled_pause=n,enabled,coupled_pause
        self._contract=dict(deepcopy(CONTRACT),enabled=enabled,coupled_pause=coupled_pause,
                            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def contract(self):
        return deepcopy(self._contract)

    def _state(self,state,action):
        if not isinstance(state,TrackingGuardState):
            raise ValueError('current native TrackingGuardState required when enabled')
        shapes=dict(active_tracking_error_rad=(2,),mimic_error_rad=(2,),
                    measured_palm_offset_m=(2,3),actual_base_pose_w=(2,7),
                    wrist_seed_pose_w=(2,7),current_palm_target_m=(2,3),
                    current_wrist_angles_rad=(2,3))
        for name,tail in shapes.items():
            value=getattr(state,name)
            if (not isinstance(value,torch.Tensor) or value.shape!=(self.n,)+tail
                    or value.device!=action.device or value.dtype!=action.dtype):
                raise ValueError(f'{name} requires matching action dtype, device and shape')
        if (state.live.shape!=(self.n,) or state.live.dtype!=torch.bool
                or state.live.device!=action.device):
            raise ValueError('one native live boolean per environment required')

    @torch.no_grad()
    def project(self,action24,state=None):
        raw=action24
        if (not isinstance(raw,torch.Tensor) or raw.shape!=(self.n,24)
                or not raw.is_floating_point() or not torch.isfinite(raw).all()
                or (raw.abs()>1+1e-12).any()):
            raise ValueError('finite normalized physical N x 24 actions required')
        output=raw.clone()
        info=self.contract()
        info['pause_hand']=torch.zeros((self.n,2),dtype=torch.bool,device=raw.device)
        info['local_pause_hand']=torch.zeros_like(info['pause_hand'])
        info['propagated_partner_pause']=torch.zeros_like(info['pause_hand'])
        if not self.enabled:
            return output,info
        self._state(state,raw)
        tracking_valid=torch.isfinite(state.active_tracking_error_rad) & torch.isfinite(state.mimic_error_rad)
        pose_valid=torch.isfinite(state.actual_base_pose_w).all(-1) & torch.isfinite(state.wrist_seed_pose_w).all(-1)
        _,actual_quat_valid=_quaternion(state.actual_base_pose_w[...,3:])
        _,seed_quat_valid=_quaternion(state.wrist_seed_pose_w[...,3:])
        measurement_valid=tracking_valid & pose_valid & actual_quat_valid & seed_quat_valid
        tracking_bad=state.active_tracking_error_rad.abs()>CONTRACT['tracking_error_limit_rad']
        mimic_bad=state.mimic_error_rad.abs()>=CONTRACT['mimic_error_limit_rad']
        local_pause=(tracking_bad | mimic_bad | ~measurement_valid) & state.live[:,None]
        pause=(local_pause.any(-1,keepdim=True).expand(-1,2).clone()
               if self.coupled_pause else local_pause)
        translation=output[:,12:18].reshape(self.n,2,3)
        rotation=output[:,18:24].reshape(self.n,2,3)
        translation[pause]=0.
        rotation[pause]=0.
        low=raw.new_tensor(WRIST['palm_lower_m'])
        high=raw.new_tensor(WRIST['palm_upper_m'])
        measured=state.measured_palm_offset_m
        translation_valid=torch.isfinite(measured).all(-1) & (measured>=low).all(-1) & (measured<=high).all(-1)
        hold_offset=torch.where(translation_valid[...,None],measured,state.current_palm_target_m)
        existing_translation_valid=(torch.isfinite(state.current_palm_target_m).all(-1)
                                    & (state.current_palm_target_m>=low).all(-1)
                                    & (state.current_palm_target_m<=high).all(-1))
        existing_rotation_valid=(torch.isfinite(state.current_wrist_angles_rad).all(-1)
                                 & (state.current_wrist_angles_rad.abs()<=WRIST['rotation_limit_rad']).all(-1))
        angles,rotation_valid,rotation_error=intrinsic_xyz_from_measured_pose(
            state.wrist_seed_pose_w[...,3:],state.actual_base_pose_w[...,3:],
            state.current_wrist_angles_rad,state.measured_wrist_angles_rad)
        info.update(pause_hand=pause,local_pause_hand=local_pause,
            propagated_partner_pause=pause & ~local_pause,measurement_valid=measurement_valid,
            tracking_error_trigger=tracking_bad,mimic_error_trigger=mimic_bad,
            cancel_translation=pause & translation_valid,
            hold_offset_m=hold_offset.clone(),translation_cancellation_unavailable=pause & ~translation_valid,
            cancel_rotation=pause & rotation_valid,hold_intrinsic_xyz_rad=angles,
            rotation_cancellation_unavailable=pause & ~rotation_valid,
            rotation_reproduction_error_rad=rotation_error,
            rotation_source='native_wrist_joint_angles_validated_with_actual_base_and_seed' if state.measured_wrist_angles_rad is not None
                else 'measured_base_quaternion_inverse_nearest_current_intrinsic_chart',
            tracking_target_source=state.tracking_target_source,
            existing_translation_target_within_policy_bounds=existing_translation_valid,
            existing_rotation_target_within_bounds=existing_rotation_valid,
            measurement_physics_step=state.measurement_physics_step,
            measurement_timestamp_s=state.measurement_timestamp_s,
            native_targets_written=False,finger_requests_modified=False)
        return output,info
