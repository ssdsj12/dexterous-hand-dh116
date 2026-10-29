"""Optional phase-3 policy stabilizer, independent of physics and success gates.

Caller supplies raw policy palms before targets(), selects only live active phase-3
rows, applies the usual physical governors, then calls project_applied(). A paused
world must submit measured adaptive_palm_offset to cancel queued drive ascent.
All offsets here are already-applied world offsets, never local-frame offsets.
"""
import torch
import math

FORCE_HOLD_CONTRACT=dict(version='latched_reference_filtered_pi_v1',
    reference='last_accepted_close_requested_force_per_hand',range_n=[2.,12.],
    reference_after_close='immutable_including_recovery',filter_tau_s=.03,
    kp_rad_n=.01,ki_rad_n_s=.02,integral_limit_n_s=2.,offset_limit_rad=.08,
    offset_rate_rad_s=.03,invalid_contact='pause_ascent_freeze_filter_and_integral',
    antiwindup='freeze_on_invalid_coupling_tracking_overload_and_output_saturation; backcalculate_applied_projection',
    additional_observation=['filtered_force2_div_12','force_integral2_div_2'],
    tolerance='max(0.5N,10percent_of_reference)',physical_force_guarantee=False)


CONTRACT = dict(
    version='policy_lift_hold_v1', update_physics_dt=True,
    dt_units='seconds; call once per physics update, not once per PPO step',
    no_success_change=True, no_physics_change=True, policy_close_unchanged=True,
    capture='real_valid_transition_to_phase3; idempotent_until_selected_reset',
    selection='live & active & (task.phase == 3); caller masks outputs',
    force_capture='actual_hand_object_loads_clamp_1_to_min_configured_max_force_12',
    correction_rate_rad_s=.03, correction_budget_rad=.08,
    force_feedback_gain_rad_n_s=.1, severe_mimic_rad=.012,
    tracking_pause_rad=.03, contact_loss_load_n=.05, contact_loss_trigger_s=.02,
    recovery_slip_m=.002, recovery_slip_rate_m_s=.002,
    slip_pause='slip_above_threshold_and_growing; tiny_elastic_slip_does_not_pause',
    recovered_load_n=1., recovery_stable_s=.02,
    recovery_stability='bilateral_load_at_least_1N_and_no_thresholded_slipping',
    recovery_force_rate_n_s=4., force_ceiling_n=12.,
    common_lift_rate_m_s=.01, palm_upper_z_m=.05,
    ascent='bilateral_positive_policy_request; min_delta; no_automatic_lift',
    common_height='monotone_during_unpaused_ascent; signed_rebase_to_lower_measured_shared_height_on_pause_and_resume',
    palm_frame='applied_world_offset; captured_XY_and_relative_Z',
    paused_palms='measured_adaptive_palm_offset; cancel_queued_drive_target_no_teleport',
    final_guards='caller_mechanical_pressure_mimic_and_slew_remain_authoritative',
    antiwindup='project_applied_after_final_task_command',
    command_normalization='signed_rad_divided_by_pi',
    observation_fields=(('command',12,3.141592653589793),
        ('palm_offset',6,.05), ('stiffness',2,20.), ('force_target',2,12.),
        ('closure',2,.08), ('active',1,1.), ('recovering',1,1.),
        ('paused',1,1.), ('contact_loss_s',1,.02),
        ('recovery_stable_s',1,.02), ('common_height',1,.05)),
    observation_size=30,
    observation_nonfinite='raise_ValueError_for_any_nonfinite_internal_state',
    task_inputs='command[N,12], policy_stiffness[N,2], direction[N,2,6], phase[N], slip[N], palm_targets[N,2,3], parameters.max_force_n, m:{q[N,12],hand_object_loads[N,2],mimic_errors[N,K],adaptive_palm_offset[N,2,3]}')


class PolicyLiftHold:
    def __init__(self, n, device):
        self.active = torch.zeros(n, device=device, dtype=torch.bool)
        self.command = torch.zeros((n,12), device=device, dtype=torch.float64)
        self.palm_offset = self.command.new_zeros((n,2,3))
        self.stiffness = self.command.new_zeros((n,2))
        self.force_target = torch.zeros_like(self.stiffness)
        self.force_filtered = torch.zeros_like(self.stiffness)
        self.force_integral = torch.zeros_like(self.stiffness)
        self.closure = torch.zeros_like(self.stiffness)
        self.recovering = torch.zeros_like(self.active)
        self.contact_loss_s = self.command.new_zeros(n)
        self.recovery_stable_s = torch.zeros_like(self.contact_loss_s)
        self.last_slip = torch.zeros_like(self.contact_loss_s)
        self.common_height = torch.zeros_like(self.contact_loss_s)
        self.paused = torch.zeros_like(self.active)

    def capture(self, task, mask, palm_offset):
        selected = mask & ~self.active
        self.command[selected] = task.command[selected]
        self.palm_offset[selected] = palm_offset[selected]
        self.stiffness[selected] = task.policy_stiffness[selected]
        ceiling = min(task.parameters.max_force_n,12.)
        if ceiling < 1:
            raise ValueError('policy lift hold requires max_force_n >= 1')
        floor=2. if getattr(task,'force_region',False) else 1.
        reference=(task.f_requested if getattr(task,'force_region',False) else task.m['hand_object_loads'])
        self.force_target[selected] = reference[selected].clamp(floor,ceiling)
        self.force_filtered[selected]=task.m['hand_object_loads'][selected]
        self.last_slip[selected] = task.slip[selected]
        self.active[selected] = True

    def reset(self, ids):
        """Clear every state tensor for integer ids or a boolean row mask."""
        for value in vars(self).values():
            value[ids] = 0

    def targets(self, task, live, dt):
        """Return (joint[N,12], world palms[N,2,3], stiffness[N,2], paused[N])."""
        if dt <= 0:
            raise ValueError('dt must be positive physics seconds')
        selected = live & self.active & (task.phase == 3)
        loads = task.m['hand_object_loads']
        growing = (task.slip-self.last_slip)/dt > .002
        slipping = (task.slip > .002) & growing
        lost = (loads < .05).any(1)
        loss = torch.where(lost, self.contact_loss_s+dt, 0.).clamp_max(.02)
        recovery = self.recovering | slipping | (loss >= .02)
        floor=2. if getattr(task,'force_region',False) else 1.
        region=(task.m['contact_valid']&task.m['contact_all_below_centerline']).all(1) if getattr(task,'force_region',False) else torch.ones_like(selected)
        force_tracking=torch.ones_like(selected)
        if getattr(task,'force_region',False):
            tolerance=torch.maximum(torch.full_like(self.force_target,.5),.1*self.force_target)
            force_tracking=((loads-self.force_target).abs()<=tolerance).all(1)&(loads<=12.).all(1)
        stable = torch.where(recovery & (loads >= floor).all(1) & region & force_tracking & ~slipping,
                             self.recovery_stable_s+dt, 0.).clamp_max(.02)
        recovery = recovery & (stable < .02)
        severe = task.m['mimic_errors'].abs().amax(1) >= .012
        tracking = (task.command-task.m['q']).abs().amax(1) > .03
        paused = (loads < floor).any(1) | ~region | ~force_tracking | slipping | recovery | severe | tracking
        ceiling = min(task.parameters.max_force_n,12.)
        ramp = (self.force_target+4.*dt).clamp_max(ceiling)
        if not getattr(task,'force_region',False):
            self.force_target[selected & recovery] = ramp[selected & recovery]
        error = self.force_target-loads
        change = (.1*error*dt).clamp(-.03*dt,.03*dt)
        # Severe linkage error vetoes tightening but never blocks force release.
        valid_hand=(task.m['contact_valid']&task.m['contact_all_below_centerline']) if getattr(task,'force_region',False) else torch.ones_like(loads,dtype=torch.bool)
        change = torch.where((severe[:,None] | ~valid_hand) & (change > 0), 0., change)
        closure = (self.closure+change).clamp(0.,.08)
        if getattr(task,'force_region',False):
            sensed=selected[:,None]&valid_hand&(loads>.05)
            filtered=self.force_filtered+(1-math.exp(-dt/.03))*(loads-self.force_filtered)
            self.force_filtered.copy_(torch.where(sensed,filtered,self.force_filtered))
            pi_error=self.force_target-self.force_filtered
            permitted=sensed&~severe[:,None]&~tracking[:,None]&(loads<=12.)
            trial=(self.force_integral+pi_error*dt).clamp(-2.,2.)
            raw=.01*pi_error+.02*trial
            winding=((raw>.08)&(pi_error>0))|((raw<-.08)&(pi_error<0))
            self.force_integral.copy_(torch.where(permitted&~winding,trial,self.force_integral))
            desired=(.01*pi_error+.02*self.force_integral).clamp(-.08,.08)
            change=(desired-self.closure).clamp(-.03*dt,.03*dt)
            change=torch.where((~sensed|severe[:,None])&(change>0),0.,change)
            change=torch.where(sensed,change,0.)
            closure=(self.closure+change).clamp(-.08,.08)
        self.closure[selected] = closure[selected]
        measured = task.m['adaptive_palm_offset']
        # Rebase both during pauses and at the first resumed update: old queued
        # heights must never leak back into the drive target after contact loss.
        # Signed displacement also preserves measured heights below the anchor;
        # zero-clamping here would resurrect an unexecuted upward drive target.
        shared = (measured[:,:,2]-self.palm_offset[:,:,2]).amin(1)
        capacity = (.05-self.palm_offset[:,:,2]).amin(1).clamp_min(0.)
        rebase = selected & (paused | self.paused)
        self.common_height[rebase] = shared.minimum(capacity)[rebase]
        request = task.palm_targets[:,:,2]-self.palm_offset[:,:,2]
        requested_height = request.amin(1).clamp_min(0.).minimum(capacity)
        advance = (requested_height-self.common_height).clamp(0.,.01*dt)
        rising = selected & ~paused & (request > 0).all(1)
        self.common_height[rising] += advance[rising]
        for dst, src in ((self.recovering,recovery), (self.paused,paused),
                         (self.contact_loss_s,loss), (self.recovery_stable_s,stable),
                         (self.last_slip,task.slip)):
            dst[selected] = src[selected]
        target = self.command+(self.closure[:,:,None]*task.direction).flatten(1)
        palms = self.palm_offset.clone()
        palms[:,:,2] += self.common_height[:,None]
        if getattr(task,'manual_control',False):
            # Explicit user XYZ requests replace the autonomous fixed-XY,
            # ascent-only path. The world still enforces slew/clearance/guards.
            palms=task.palm_targets.clone()
        palms = torch.where(self.paused[:,None,None], measured, palms)
        return target, palms, self.stiffness.clone(), self.paused.clone()

    def project_applied(self, task, mask):
        """Project final governed command back onto each hand's closing axis."""
        selected = mask & self.active & (task.phase == 3)
        delta = (task.command-self.command).reshape(-1,2,6)
        direction = task.direction
        projected = (delta*direction).sum(2)/direction.square().sum(2).clamp_min(1e-12)
        minimum=-.08 if getattr(task,'force_region',False) else 0.
        projected=projected.clamp(minimum,.08)
        if getattr(task,'force_region',False):
            changed=selected[:,None]&((projected-self.closure).abs()>1e-8)
            corrected=((projected-.01*(self.force_target-self.force_filtered))/.02).clamp(-2.,2.)
            self.force_integral.copy_(torch.where(changed,corrected,self.force_integral))
        self.closure[selected] = projected[selected]

    def observe(self, task):
        """30 bounded finite features in CONTRACT order; append in task.observe()."""
        finite = torch.stack([torch.isfinite(value).all() for value in vars(self).values()])
        if not finite.all():
            raise ValueError('PolicyLiftHold contains nonfinite internal state')
        fields = [getattr(self,name).reshape(self.active.numel(),size)/scale
                  for name,size,scale in CONTRACT['observation_fields']]
        if getattr(task,'force_region',False):
            fields.extend((self.force_filtered/12.,self.force_integral/2.))
        return torch.cat(fields,1).clamp(-1.,1.)
