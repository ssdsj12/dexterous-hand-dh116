"""Contact-gated effort feedback; measured load closes the pressure loop.

The calibrated closure direction is a candidate motor direction, not a contact
normal or a guarantee of object force. Callers retain coupling, overload,
clearance, release, and phase guards and verify the resulting measured force.
"""
import math
import torch

CONTRACT=dict(version='contact_load_integral_effort_v3',
    measurement='per_hand_measured_object_contact_load_n',
    integral_gain_nm_per_n_s=.02,effort_slew_nm_s=.1,deadband_n=.02,
    min_contact_load_n=.05,output='nonnegative_calibrated_closure_effort_nm',
    direction='positive_components_normalized_by_max_abs_per_hand',
    eligibility='caller_mechanical_coupling_overload_release_conflict_guards',
    contact_loss_grace_s=.02,contact_loss='hold_integral_without_integration_then_clear_at_grace',
    retained_contact_loss='hold_previously_acquired_integral_without_integration_until_permission_revoked_or_reset',
    inactive='ineligible_or_zero_direction_clear_integral_and_contact_history_immediately',
    motor_budget='abs_feedforward_plus_implicit_drive_limit_at_most_total_limit',
    joint_motion_required=False,physical_force_guarantee=False)


def _limit(value):
    try:
        result=float(value)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError('finite positive motor effort limit required') from exc
    if isinstance(value,bool) or not math.isfinite(result) or result<=0:
        raise ValueError('finite positive motor effort limit required')
    return result


def split_effort_budget(feedforward,effort_limit_nm=.2):
    """Return bounded explicit effort and the remaining implicit-drive limit.

    Bounding the drive independently by ``limit - abs(feedforward)`` bounds
    their sum for either drive sign. The caller must write that drive limit
    into PhysX before submitting explicit effort.
    """
    limit=_limit(effort_limit_nm)
    if (not isinstance(feedforward,torch.Tensor) or not feedforward.is_floating_point()
            or not bool(torch.isfinite(feedforward).all())):
        raise ValueError('finite floating point feedforward effort tensor required')
    bounded=feedforward.clamp(-limit,limit)
    return bounded,(limit-bounded.abs()).clamp_min(0.)


class ContactEffortFeedback:
    def __init__(self,n,device,effort_limit_nm=.2):
        if isinstance(n,bool) or not isinstance(n,int) or n<=0:
            raise ValueError('positive integer environment count required')
        self.effort_limit_nm=_limit(effort_limit_nm)
        self.integral_nm=torch.zeros((n,2),device=device,dtype=torch.float64)
        self.contact_loss_s=torch.zeros_like(self.integral_nm)
        self.contact_acquired=torch.zeros((n,2),device=device,dtype=torch.bool)

    def _input(self,value,shape,name,*,boolean=False):
        if (not isinstance(value,torch.Tensor) or value.shape!=shape
                or value.device!=self.integral_nm.device
                or (boolean and value.dtype!=torch.bool)
                or (not boolean and not value.is_floating_point())
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f'finite {name} tensor with shape {shape} on controller device required')
        return value if boolean else value.double()

    def step(self,target_load,measured_load,direction,eligible,dt,*,retained=None):
        """Return (N,2,6) effort; no joint-angle tracking signal is needed."""
        shape=self.integral_nm.shape
        target=self._input(target_load,shape,'target load')
        measured=self._input(measured_load,shape,'measured load')
        closure=self._input(direction,(*shape,6),'closure direction').clamp_min(0.)
        eligible=self._input(eligible,shape,'eligibility',boolean=True)
        retained=(torch.zeros_like(eligible) if retained is None else
                  self._input(retained,shape,'retained grasp',boolean=True))
        if (isinstance(dt,bool) or not isinstance(dt,(int,float))
                or not math.isfinite(dt) or dt<=0
                or bool((target<0).any()) or bool((measured<0).any())):
            raise ValueError('nonnegative loads and finite positive timestep required')
        maximum=closure.abs().amax(dim=2,keepdim=True)
        closure=closure/maximum.clamp_min(torch.finfo(torch.float64).tiny)
        permitted=eligible&(maximum.squeeze(2)>0)
        active=permitted&(measured>=CONTRACT['min_contact_load_n'])
        loss_s=torch.where(permitted&self.contact_acquired&~active,self.contact_loss_s+dt,0.)
        retained=permitted&self.contact_acquired&(retained|(loss_s<CONTRACT['contact_loss_grace_s']))
        acquired=active|retained
        error=target-measured
        error=torch.where(error.abs()<=CONTRACT['deadband_n'],0.,error)
        slew=CONTRACT['effort_slew_nm_s']*dt
        delta=(CONTRACT['integral_gain_nm_per_n_s']*error*dt).clamp(-slew,slew)
        updated=(self.integral_nm+delta).clamp(0.,self.effort_limit_nm)
        # Brief contact loss holds the existing effort without integrating the
        # missing sample. Safety invalidation clears effort and history at once.
        self.integral_nm.copy_(torch.where(active,updated,torch.where(retained,self.integral_nm,0.)))
        self.contact_acquired.copy_(acquired)
        self.contact_loss_s.copy_(torch.where(acquired,loss_s,0.))
        return self.integral_nm[:,:,None]*closure

    def reset(self,ids):
        selected=torch.as_tensor(ids,device=self.integral_nm.device)
        if (selected.ndim!=1 or selected.dtype not in (torch.int32,torch.int64)
                or bool((selected<0).any()) or bool((selected>=len(self.integral_nm)).any())):
            raise ValueError('valid one dimensional integer environment indices required')
        self.integral_nm[selected]=0.
        self.contact_loss_s[selected]=0.
        self.contact_acquired[selected]=False
