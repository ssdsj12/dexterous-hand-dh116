"""Capture each currently qualified close grasp without a readiness dwell."""
import torch

CONTRACT=dict(version='instant_per_hand_retained_close_hold_v3',close_dwell_s=0.,hold_dwell_s=0.,
    readiness='current_per_hand_load_mimic_motion_contact_and_normal_z',
    normal_force='own_normal_z_n >= 0; aggregate_normal_z_n >= 0_for_lift',
    capture='first_current_ready_sample_applied_command_force_target_and_palm',
    lift='both_current_sides_ready_same_physics_substep',
    contact_loss_load_n=.05,contact_loss='retain_pose_until_reset_or_failure;current_readiness_still_required',
    feedback='bounded_tightening_only_from_last_applied_command;physical_governors_retain_priority',
    correction_budget_rad=.08,position_integral='frozen_while_held',
    accepted_pose='soft_lead_and_mimic_limit_new_tightening_only;severe_overpressure_mechanical_recovery_can_override',
    pressure_reference='configured_max_force_n_capped_at_12_with_existing_1.25_band',
    palm='independent_accepted_offsets_with_per_hand_clearance',
    reset='selected_environments_only',rewards_changed=False)


class CloseGraspHold:
    def __init__(self,n,device):
        self.active=torch.zeros((n,2),device=device,dtype=torch.bool)
        self.ready=torch.zeros_like(self.active)
        self.command=torch.zeros((n,12),device=device,dtype=torch.float64)
        self.force_target=torch.zeros((n,2),device=device,dtype=torch.float64)
        self.palm_offset=torch.zeros((n,2,3),device=device,dtype=torch.float64)
        self.contact_loss_s=torch.zeros_like(self.force_target)

    def reset(self,ids):
        for value in vars(self).values():value[ids]=0

    def update(self,task,live,close,speed,palm_offset,dt):
        """Refresh current readiness, then capture newly qualified sides only."""
        from .tabletop_tensor_task import MEASURED_READINESS_CONTRACT
        from .tabletop_coordinated_grasp import CONTRACT as GOVERNOR
        selected=live&task.coordinated_control
        failed=selected&(task.failure!=0)
        self.reset(failed.nonzero(as_tuple=True)[0])
        selected &= task.failure==0
        eligible=selected&close
        m=task.m;loads=m['hand_object_loads']
        if task.grasp_readiness=='measured_load':
            load_ready=(loads>=MEASURED_READINESS_CONTRACT['min_hand_load_n'])
            load_ready &= loads<=1.25*task.parameters.max_force_n+task.parameters.deadband_n
        else:
            load_ready=(loads>=.8*task.f_requested)&(loads<=1.25*task.f_requested+task.parameters.deadband_n)
        ready=eligible[:,None]&load_ready
        ready &= (speed<.02)[:,None]&(task.slip<=.003)[:,None]
        ready &= m['mimic_errors'].reshape(task.n,2,5).abs().amax(2)<GOVERNOR['coupling_budget_rad']
        ready &= m['vertical_hand_force_n']>=0.
        if hasattr(task,'palm_reposition'):
            ready &= ~((task.palm_reposition.state==1)|(task.palm_reposition.state==2))
            ready &= ~task.palm_reposition.aborted
        if task.contact_target_z_m is not None:
            ready &= m['contact_valid']&m['contact_all_below_centerline']
        self.ready[selected]=ready[selected]
        loss=eligible[:,None]&self.active&(loads<CONTRACT['contact_loss_load_n'])
        clock=torch.where(loss,self.contact_loss_s+dt,0.)
        self.contact_loss_s[selected]=clock[selected]
        captured=ready&~self.active
        self.command.reshape(task.n,2,6)[captured]=task.command.reshape(task.n,2,6)[captured]
        self.force_target[captured]=task.f_target[captured]
        self.palm_offset[captured]=palm_offset[captured]
        self.contact_loss_s[captured]=0.
        self.active |= captured
        if hasattr(task,'palm_reposition'):
            for name in ('state','clock','dwell'):
                getattr(task.palm_reposition,name)[captured]=0
        return self.ready
