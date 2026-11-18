"""Retain accepted close references; feedback and physical guards stay active."""
import torch

CONTRACT=dict(version='accepted_close_grasp_recovery_servo_v5',
    capture='accepted_transition_to_lift',
    joints='last_applied_command_plus_bounded_nonnegative_closure_velocity;physical_governors_retain_priority',
    correction_budget_rad=.08,correction_anchor='accepted_applied_command',
    antiwindup='final_applied_command_projection_per_hand',
    pressure_mimic_conflict='neutral_measured_q_request_with_slew_not_instant_physical_recovery',
    force='retained_target_with_latched_grip_recovery_ramp',
    recovery_slip_m=.002,recovery_slip_rate_m_s=.002,
    contact_loss_load_n=.05,contact_loss_trigger_s=.02,
    recovered_hand_load_n=1.,recovery_stable_s=.02,
    recovery_force_rate_n_s=4.,recovery_force_ceiling_n=12.,
    recovery='lift_only_pause_common_ascent_retain_pose_ramp_targets_until_bilateral_contact_and_no_growing_slip',
    recovery_dwell='recovery_exit_only_not_close_admission',
    pressure='held_grasp_uses_configured_max_force_ceiling_with_existing_1.25_band',
    palms='hold_applied_exploration_offset_common_lift_remains_active',
    lift_height_projection='per_hand_captured_offset_and_independent_clearance',
    shared_height_projection='close_and_hold_only',
    policy='ignore_new_grasp_force_and_palm_requests_during_lift',
    accepted_pose='soft_lead_and_mimic_limit_new_tightening_only',
    lift_progress='continuous_after_admission_unless_grip_recovery_severe_coupling_overpressure_or_tracking_fault',
    lift_readmission=False,
    bounds='mechanical_bounds_measured_lead_mimic_pressure_and_slew',
    reset='selected_environments_only',physical_force_constant=False)


class LiftGraspHold:
    def __init__(self,n,device):
        self.active=torch.zeros(n,device=device,dtype=torch.bool)
        self.command=torch.zeros((n,12),device=device,dtype=torch.float64)
        self.feedback_offset=torch.zeros((n,2),device=device,dtype=torch.float64)
        self.force_target=torch.zeros_like(self.feedback_offset)
        self.palm_offset=torch.zeros((n,2,3),device=device,dtype=torch.float64)
        self.recovering=torch.zeros(n,device=device,dtype=torch.bool)
        self.contact_loss_s=torch.zeros(n,device=device,dtype=torch.float64)
        self.recovery_stable_s=torch.zeros_like(self.contact_loss_s)

    def recover(self,task,live,dt):
        """Pause transport for grip loss, keeping the accepted pose owned."""
        selected=live&self.active&(task.phase==3)
        loads=task.m['hand_object_loads']
        growing=(task.slip-task.f_last_slip)/dt>CONTRACT['recovery_slip_rate_m_s']
        slipping=(task.slip>CONTRACT['recovery_slip_m'])&growing
        lost=(loads<CONTRACT['contact_loss_load_n']).any(1)
        loss=torch.where(lost,self.contact_loss_s+dt,0.)
        recovering=self.recovering|slipping|(loss>=CONTRACT['contact_loss_trigger_s'])
        recovered=(loads>=CONTRACT['recovered_hand_load_n']).all(1)&~growing
        stable=torch.where(recovering&recovered,self.recovery_stable_s+dt,0.)
        recovering &= stable<CONTRACT['recovery_stable_s']
        ceiling=min(task.parameters.max_force_n,CONTRACT['recovery_force_ceiling_n'])
        target=(self.force_target+CONTRACT['recovery_force_rate_n_s']*dt).clamp_max(ceiling)
        self.force_target[selected&recovering]=target[selected&recovering]
        self.recovering[selected]=recovering[selected]
        self.contact_loss_s[selected]=loss[selected]
        self.recovery_stable_s[selected]=stable[selected]

    def capture(self,task,mask,palm_offset):
        selected=mask & ~self.active
        self.command[selected]=task.command[selected]
        self.feedback_offset[selected]=task.f_offset[selected]
        self.force_target[selected]=task.f_target[selected]
        self.palm_offset[selected]=palm_offset[selected]
        self.active |= selected

    def reset(self,ids):
        for value in vars(self).values():value[ids]=0
