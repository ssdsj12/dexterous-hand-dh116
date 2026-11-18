"""Action-only force-region teacher: measured close exploration and held lift.

No task writes, success injection, drive writes or simulator imports. Feedback
timing uses task.elapsed deltas when supplied, otherwise one 1/60 s control tick.
"""
import math
import torch
from .tabletop_control_probe import joint_actions
from .tabletop_policy_force_region import _radial_z


CONTRACT = dict(version='force_region_teacher_v1',
    action='20D_joint12_wrist6_force2', force_range_n=[2.,12.],
    wrist_lower_z_m=-.004, wrist_upper_z_m=.05, wrist_xy_limit_m=.003,
    measured_contact_z_rate_m_s=.001, fallback_control_dt_s=1/60,
    contact_frame='world_radial_Z_after_removing_rotated_local_Y_axis_projection',
    missing_contact='retain_current_wrist_offset_without_synthetic_feedback',
    ramp='reference_plus_case_joint_offset_times_elapsed_over_closing_ramp_s',
    lift='phase3_and_policy_lift_hold_active; captured_command_and_bilateral_anchor_z_plus_.04_capped_.05',
    force='constant_case_force_in_close_and_lift; not_stiffness',
    reset='selected_rows_only', task_mutation=False, physical_validation=False)


class ForceTeacherController:
    def __init__(self,cases,reference,lower,center,upper,device):
        if not cases or len({c['name'] for c in cases}) != len(cases):
            raise ValueError('teacher cases require unique names')
        self.cases=cases
        n=len(cases)
        def tensor(value,shape,name):
            try:
                result=torch.as_tensor(value,device=device,dtype=torch.float64)
                result=torch.broadcast_to(result,shape).clone()
            except (ValueError,TypeError,RuntimeError) as exc:
                raise ValueError('invalid '+name) from exc
            if not torch.isfinite(result).all():
                raise ValueError(name+' must be finite')
            return result
        self.reference=tensor(reference,(n,12),'reference')
        self.lower=tensor(lower,(n,12),'lower')
        self.center=tensor(center,(n,12),'center')
        self.upper=tensor(upper,(n,12),'upper')
        self.offset=torch.zeros_like(self.reference)
        self.ramp=torch.zeros(n,device=device,dtype=torch.float64)
        self.base=torch.zeros((n,20),device=device,dtype=torch.float32)
        self.target=torch.zeros((n,2),device=device,dtype=torch.float64)
        for i,case in enumerate(cases):
            depth=tensor(case['contact_z_m'],(2,),'contact_z_m')
            if ((depth < -.004)|(depth >= 0)).any():
                raise ValueError('contact_z_m must be in [-.004,0)')
            force=tensor(case['force_n'],(2,),'force_n')
            if ((force < 2)|(force > 12)).any():
                raise ValueError('force_n must be within [2,12]')
            # Broadcast is allowed only for the specified scalar contact depth.
            if torch.as_tensor(case['force_n']).shape!=(2,):
                raise ValueError('force_n requires two hand targets')
            offset=case.get('joint_offset_rad',[0.]*12)
            if torch.as_tensor(offset).shape!=(12,):
                raise ValueError('joint_offset_rad requires twelve values')
            self.offset[i]=tensor(offset,(12,),'joint_offset_rad')
            xy=case.get('xy_m',[[0.,0.],[0.,0.]])
            if torch.as_tensor(xy).shape!=(2,2):
                raise ValueError('xy_m requires two XY pairs')
            xy=tensor(xy,(2,2),'xy_m')
            if (xy.abs() > .003).any():
                raise ValueError('xy_m exceeds +/- .003m')
            ramp=float(case.get('closing_ramp_s',0.))
            if not math.isfinite(ramp) or ramp < 0:
                raise ValueError('closing_ramp_s must be finite and nonnegative')
            self.ramp[i]=ramp
            self.target[i]=depth
            self.base[i,12:14]=(xy[0]/.003).float()
            self.base[i,15:17]=(xy[1]/.003).float()
            self.base[i,18:]=((force-2)/5-1).float()
        # Reuse production encoding/validation for every joint and envelope.
        joint_actions(self.reference,self.lower,self.center,self.upper)
        full=(self.reference+self.offset).maximum(self.lower).minimum(self.upper)
        self.base[:,:12]=joint_actions(full,self.lower,self.center,self.upper).float()
        self.base[:,14]=(self.target[:,0]/.004).float()
        self.base[:,17]=(self.target[:,1]/.004).float()
        self.z=self.target.clone()
        self.time=torch.zeros_like(self.ramp)
        self.last_elapsed=torch.zeros_like(self.ramp)

    def reset(self,done):
        self.z[done]=self.target[done]
        self.time[done]=0.
        self.last_elapsed[done]=0.

    def actions(self,measurement,task=None):
        if task is not None and hasattr(task,'elapsed'):
            now=task.elapsed.to(self.time)
            if not torch.isfinite(now).all():
                raise ValueError('elapsed must be finite')
            dt=(now-self.last_elapsed).clamp_min(0.)
            self.last_elapsed.copy_(now)
            self.time.copy_(now.clamp_min(0.))
        else:
            dt=torch.full_like(self.time,1/60)
            self.time+=dt
        fraction=torch.where(self.ramp>0,
            (self.time/self.ramp.clamp_min(1e-12)).clamp(0.,1.),1.)
        requested=(self.reference+self.offset*fraction[:,None]).maximum(self.lower).minimum(self.upper)
        held=torch.zeros_like(self.time,dtype=torch.bool)
        hold=getattr(task,'policy_lift_hold',None)
        if hold is not None:
            held=(task.phase==3)&hold.active
            requested=torch.where(held[:,None],hold.command.to(requested),requested)
        # Feedback is independent per hand; invalid/missing contacts are not
        # converted into a zero-height measurement.
        if all(k in measurement for k in ('contact_valid','contact_center_relative_m','object_quat')):
            centers=measurement['contact_center_relative_m'].to(self.z)
            valid=measurement['contact_valid'].bool() & torch.isfinite(centers).all(-1)
            safe=torch.where(valid[:,:,None],centers,0.)
            radial=_radial_z(safe,measurement['object_quat'].to(self.z))
            error=self.target-radial
            step=error.maximum(-.001*dt[:,None]).minimum(.001*dt[:,None])
            self.z=torch.where(valid & ~held[:,None],(self.z+step).clamp(-.004,0.),self.z)
        palms_z=self.z.clone()
        if hold is not None:
            lift=(hold.palm_offset[:,:,2].to(self.z)+.04).clamp_max(.05)
            palms_z=torch.where(held[:,None],lift,palms_z)
        actions=self.base.clone()
        actions[:,:12]=joint_actions(requested,self.lower,self.center,self.upper).float()
        encoded=palms_z/torch.where(palms_z>=0,.05,.004)
        actions[:,14]=encoded[:,0].float()
        actions[:,17]=encoded[:,1].float()
        if not torch.isfinite(actions).all():
            raise ValueError('teacher actions must be finite')
        return actions.clamp(-1.,1.)
