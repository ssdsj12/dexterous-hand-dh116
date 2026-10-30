"""GPU task stepping with isolated terminal resets and independent safety guards."""
import torch

PHYSICS_PEAK_NAMES=('link_force_n','active_joint_speed_rad_s','all_joint_speed_rad_s','mimic_error_rad')

CONTINUOUS_MOTION_CONTRACT=dict(version='continuous_actuator_limits_v1',
    finite_mimic='diagnostic_and_bounded_correction_not_motion_veto_or_episode_failure',
    actuators='existing_position_velocity_effort_limits_unchanged_not_hardware_calibrated',
    invalid_state='isolated_validated_reset_before_next_physics',
    reset_failure='abort_on_invalid_reset_or_immediate_recurrence',
    terminal_observation='last_finite_task_snapshot_for_invalid_measurement',
    manual_recovery='resume_run_and_requested_lift_after_validated_reset')


def finite_physics(world,m):
    """Raw state validity, excluding documented missing-contact sentinels."""
    finite=torch.ones(world.n,dtype=torch.bool,device=world.device)
    optional={'contact_center_w','contact_center_relative_m','contact_normal_w',
              'contact_normal_load_n','contact_valid','contact_all_below_centerline','contact_count'}
    for name,value in m.items():
        if name not in optional:
            finite &= torch.isfinite(value).reshape(world.n,-1).all(1)
    for value in (world.q,world.qd,world.limits,world.errors):
        finite &= torch.isfinite(value).reshape(world.n,-1).all(1)
    for name in ('vertical_components','object_weight_n'):
        if hasattr(world,name):
            finite &= torch.isfinite(getattr(world,name)).reshape(world.n,-1).all(1)
    if hasattr(world,'object_weight_n'):
        finite &= (world.object_weight_n>0).reshape(world.n,-1).all(1)
    if hasattr(world,'adapter_fault_bits'):
        finite &= world.adapter_fault_bits==0
    return finite

VELOCITY_GUARD_CONTRACT = dict(version='finite_velocity_warning_v2',
    nominal_limit_rad_s=3.,absolute_tolerance_rad_s=1e-5,
    comparison='finite_only',scope='terminal_reset_recovery',
    finite_exceedance='diagnostic_only',nonfinite='fatal')
LIVE_VELOCITY_GUARD_CONTRACT = dict(VELOCITY_GUARD_CONTRACT,
    scope='all_joints_every_physics_substep',action='record_and_penalize_finite_exceedance',
    penalty_coefficient=.1,penalty_normalizer_rad_s=3.,penalty_square_cap=25.,
    penalty='-0.1 * min((max(0, max_abs_qd - 3) / 3)^2, 25) * dt',
    penalty_scope='live_environments_only')
TERMINAL_RESET_CONTRACT = dict(version='validated_terminal_substep_reset_v1',
    timing='after_validated_terminal_substep',
    transition='preserved_until_control_tick_end',next_physics='strict_fresh_guards')
BOUNDED_FAILURE_RESET_CONTRACT = dict(version='joint_bounded_failure_reset_v2',
    strict_force_n=30.,strict_mimic_rad=.02,max_force_exclusive_n=45.,max_mimic_exclusive_rad=.03,
    combined_fault='failed_episode_isolated_reset_with_original_penalty',
    requires='live_finite_independently_safe_and_matching_failure_label',
    reset_validation='immediate_and_next_physics_strict_guards',
    hard_fault='global_abort',physics_thresholds_changed=False)
JOINT_LIMIT_RESET_CONTRACT = dict(version='finite_joint_limit_episode_reset_v1',
    scope='live_finite_joint_limit_failures_all_articulation_joints',
    action='failed_episode_immediate_isolated_reset',penalty_preserved=True,
    joint_margin=.02,early_active_joint_margin=.005,
    other_faults='retain_independent_finite_geometry_force_mimic_guards',
    reset_validation='strict_immediate_and_next_physics')
TRAINING_JOINT_LIMIT_CONTRACT = dict(version='early_active_joint_task_failure_v1',
    scope='ppo_training_active_angular_joints',outside_nominal_limit_rad=.005,
    eligibility='live_strict_physical_safe_no_prior_joint_failure',
    action='joint_limit_failure_then_validated_isolated_reset',
    reset_validation='early_boundary_and_strict_guards_through_next_physics',
    hard_position_tolerance_rad=.02)
OBJECT_DROP_RESET_CONTRACT=dict(version='isolated_object_drop_reset_v1',bottom_threshold_m=.29,
    action='failed_episode_immediate_isolated_reset',penalty_preserved=True,
    other_guards='finite_state_finite_velocity_existing_joint_limit_resets_clearance_and_bounded_force_mimic_unchanged',
    reset_validation='strict_immediate_and_next_physics',physics_thresholds_changed=False)


def reset_velocity_safe(world):
    # The project-defined speed line is diagnostic, not a hardware safety cap.
    # Keep nonfinite motion fatal in every path; never clamp measured motion.
    return torch.isfinite(world.qd).flatten(1).all(1)


def physical_guard(world, m, *, include_joint_candidate=False, include_drop_candidate=False):
    """Return strict guard and bounded recovery masks; no prioritized-label blind spot."""
    finite = torch.ones(world.n,dtype=torch.bool,device=world.device)
    optional_contact_geometry=('contact_center_w','contact_center_relative_m',
                               'contact_normal_w','contact_normal_load_n',
                               'contact_valid','contact_all_below_centerline','contact_count')
    for name,value in m.items():
        # PhysX reports NaN positions/normals when a sensor has no valid
        # contact.  Validity gates consume those fields; they are not a
        # numerical safety failure by themselves.
        if name in optional_contact_geometry: continue
        finite &= torch.isfinite(value).reshape(world.n,-1).all(1)
    for value in (world.q,world.qd,world.limits,world.errors):
        finite &= torch.isfinite(value).reshape(world.n,-1).all(1)
    if getattr(world,'continuous_motion',False) and hasattr(world,'adapter_fault_bits'):
        finite &= world.adapter_fault_bits==0
    joint_out = ((world.q<world.limits[...,0]-.02)|(world.q>world.limits[...,1]+.02)).flatten(1).any(1)
    velocity_safe = reset_velocity_safe(world)
    force = m['object_link_loads'].flatten(1).amax(1)
    mimic = m['mimic_errors'].abs().amax(1)
    geometry = finite & (m['clearance']>=-.001)
    geometry &= (m['object_link_loads']>=0).flatten(1).all(1)
    base = geometry & (m['bottom']>=.29)
    independent = base & ~joint_out
    continuous=getattr(world,'continuous_motion',False)
    mimic_ok=torch.ones_like(finite) if continuous else mimic<=.02
    strict = independent & velocity_safe & (force<=30.) & mimic_ok
    # A failed grasp may cross both soft boundaries on the same sample. Both
    # must remain inside their existing reset bounds; neither fault exempts a
    # hard/nonfinite/geometry violation. The sample is failed, never accepted
    # as strict-safe, and must reset before any further physics is executed.
    mimic_bounded=torch.ones_like(finite) if continuous else mimic<.03
    bounded = independent & velocity_safe & (force<45.) & mimic_bounded
    recover_force = bounded & (force>30.)
    recover_mimic = bounded & (mimic>.02) & ~torch.full_like(finite,continuous)
    if include_joint_candidate:
        from .tabletop_tensor_task import FAILURE_CODES
        declared=m['joint_failure']==FAILURE_CODES['joint_limit']
        joint_candidate=base & velocity_safe & (force<45.) & mimic_bounded & (joint_out|declared)
        if include_drop_candidate:
            # Only the dropped-object height is exempted. Raw measurements and
            # every other physical bound remain authoritative before resetting.
            dropped=(geometry & velocity_safe & (force<45.) & mimic_bounded
                     & (m['bottom']<OBJECT_DROP_RESET_CONTRACT['bottom_threshold_m']))
            joint_candidate |= dropped & (joint_out|declared)
            return strict,recover_force,recover_mimic,joint_candidate,dropped
        return strict,recover_force,recover_mimic,joint_candidate
    return strict, recover_force, recover_mimic


class TensorEnv:
    def __init__(self, world, task, recovery=True, fault_trace_path=None,
                 training_joint_limit=False):
        self.world, self.task = world, task
        args=getattr(getattr(world,'w',None),'args',None)
        self.continuous_motion=bool(getattr(task,'continuous_motion',False)
            or getattr(world,'continuous_motion',False) or getattr(args,'continuous_motion',False))
        self.world.continuous_motion=self.task.continuous_motion=self.continuous_motion
        self.n, self.device = world.n, world.device
        self.recovery = recovery
        self.training_joint_limit = bool(training_joint_limit and recovery)
        self.physics_peaks=torch.zeros(4,device=self.device,dtype=torch.float64)
        self.velocity_exceedance_steps=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self.velocity_exceedance_streak_s=torch.zeros(self.n,device=self.device,dtype=torch.float64)
        self.velocity_exceedance_longest_s=torch.zeros_like(self.velocity_exceedance_streak_s)
        self.velocity_peak=torch.zeros_like(self.velocity_exceedance_streak_s)
        self.pending = torch.zeros(self.n,dtype=torch.bool,device=self.device)
        self.reset_substep = torch.zeros(self.n,dtype=torch.long,device=self.device)
        self.substep = 0
        self.episodes = torch.zeros(self.n,dtype=torch.long,device=self.device)
        self.successes = torch.zeros_like(self.episodes)
        self.recovered = torch.zeros_like(self.episodes)
        self.combined_failure_resets=torch.zeros((),device=self.device,dtype=torch.long)
        self.joint_limit_resets=torch.zeros((),device=self.device,dtype=torch.long)
        self.object_drop_resets=torch.zeros((),device=self.device,dtype=torch.long)
        self.numerical_resets=torch.zeros((),device=self.device,dtype=torch.long)
        self.returns = torch.zeros(self.n,device=self.device)
        self.last_episode_returns = torch.zeros_like(self.returns)
        self.phase_visits = torch.zeros((self.n,4),dtype=torch.bool,device=self.device)
        self.reached_counts = torch.zeros(4,dtype=torch.long,device=self.device)
        self.transition_counts = torch.zeros(16,dtype=torch.long,device=self.device)
        self.fault_trace = None
        self.fault_trace_error = None
        if fault_trace_path is not None:
            from .tabletop_physics_fault_trace import PhysicsFaultTrace
            self.fault_trace = PhysicsFaultTrace(fault_trace_path)

    def training_joint_safe(self,m):
        if not self.training_joint_limit or self.continuous_motion:
            return torch.ones_like(self.pending)
        # Only the twelve active angular joints use this task boundary. Full
        # articulation hard guards (including prismatic wrists) remain separate.
        margin=TRAINING_JOINT_LIMIT_CONTRACT['outside_nominal_limit_rad']
        return ((m['q']>=self.task.low-margin)&(m['q']<=self.task.high+margin)).all(1)

    def state_valid(self,m):
        """Validate the whole returned-state contract, including policy dtype."""
        valid=finite_physics(self.world,m)
        for key in self.task.keys:
            valid &= torch.isfinite(m[key].float()).reshape(self.n,-1).all(1)
        for key in ('object_link_loads','grouped_object_loads','hand_object_loads'):
            valid &= (m[key]>=0).reshape(self.n,-1).all(1)
        valid &= torch.isclose(m['object_link_loads'].sum(2),m['hand_object_loads'],
                               rtol=1e-6,atol=1e-8).all(1)
        expected=torch.stack([m['object_link_loads'][...,self.task.link_groups==g].sum(2)
                              for g in range(6)],2)
        valid &= torch.isclose(expected,m['grouped_object_loads'],rtol=1e-6,atol=1e-9).flatten(1).all(1)
        before=self.task.m
        self.task.m={key:torch.where(valid.reshape((self.n,)+(1,)*(value.ndim-1)),
                                     m.get(key,value),value) for key,value in before.items()}
        try:valid &= torch.isfinite(self.task.observe()).all(1)
        finally:self.task.m=before
        return valid

    def record_velocity(self):
        speed=self.world.qd.abs().flatten(1).amax(1)
        finite=torch.isfinite(speed)
        bound=VELOCITY_GUARD_CONTRACT['nominal_limit_rad_s']+VELOCITY_GUARD_CONTRACT['absolute_tolerance_rad_s']
        exceeded=finite&(speed>bound)
        self.velocity_exceedance_steps+=exceeded.long()
        self.velocity_exceedance_streak_s.copy_(torch.where(exceeded,
            self.velocity_exceedance_streak_s+self.world.dt,0.))
        self.velocity_exceedance_longest_s.copy_(torch.maximum(
            self.velocity_exceedance_longest_s,self.velocity_exceedance_streak_s))
        self.velocity_peak.copy_(torch.maximum(self.velocity_peak,torch.where(finite,speed,0.)))
        return speed

    def velocity_diagnostics(self):
        return dict(threshold_rad_s=3.,exceedance_env_substeps=int(self.velocity_exceedance_steps.sum()),
            affected_envs=int((self.velocity_exceedance_steps>0).sum()),
            peak_abs_rad_s=float(self.velocity_peak.max()),
            longest_exceedance_s=float(self.velocity_exceedance_longest_s.max()),
            total_exceedance_env_s=float(self.velocity_exceedance_steps.sum())*self.world.dt)

    def fail(self, mask, m, context):
        ids = mask.nonzero().flatten()[:8]
        detail = dict(context=context,indices=ids.cpu().tolist(),
            task_done=self.task.done[ids].cpu().tolist(),
            task_failure=self.task.failure[ids].cpu().tolist(),
            task_elapsed_s=self.task.elapsed[ids].cpu().tolist(),
            pending_reset=self.pending[ids].cpu().tolist(),
            physics_substep=self.substep,reset_substep=self.reset_substep[ids].cpu().tolist(),
            max_force_n=m['object_link_loads'][ids].flatten(1).amax(1).cpu().tolist(),
            max_mimic_rad=m['mimic_errors'][ids].abs().amax(1).cpu().tolist(),
            clearance_m=m['clearance'][ids].cpu().tolist(),bottom_m=m['bottom'][ids].cpu().tolist(),
            joint_failure=m['joint_failure'][ids].cpu().tolist(),
            max_joint_velocity_rad_s=self.world.qd[ids].abs().flatten(1).amax(1).cpu().tolist(),
            joint_position_rad=self.world.q[ids].cpu().tolist(),
            joint_velocity_rad_s=self.world.qd[ids].cpu().tolist(),
            joint_limits_rad=self.world.limits[ids].cpu().tolist())
        if self.fault_trace is not None:
            try:
                self.fault_trace.write(ids,context)
                detail['fault_trace_path']=str(self.fault_trace.path)
            except Exception as error:
                detail['fault_trace_error']=str(error)
        if self.fault_trace_error is not None:
            detail['fault_trace_record_error']=self.fault_trace_error
        if hasattr(self.world,'adapter_fault_bits'):
            detail['adapter_fault_bits']=self.world.adapter_fault_bits[ids].cpu().tolist()
        raise RuntimeError(f'tensor physical safety failure: {detail}')

    def reset(self):
        ids = torch.arange(self.n,device=self.device)
        m = self.world.reset(ids)
        safe,_,_ = physical_guard(self.world,m)
        safe &= reset_velocity_safe(self.world)
        safe &= self.training_joint_safe(m)
        if self.continuous_motion:safe &= self.state_valid(m)
        if not bool(safe.all()): self.fail(~safe,m,'initial_reset')
        self.task.reset(ids,m)
        self.velocity_exceedance_streak_s[ids]=0.
        self.pending[:] = False
        self.reset_substep[:] = 0
        self.substep = 0
        self.phase_visits[:] = False
        self.phase_visits[:,3 if getattr(self.task,'free_grasp_lift',False) else 1] = True
        return self.task.observe()

    def step(self, actions):
        from .tabletop_tensor_task import FAILURE_CODES
        from .tabletop_tensor_world import signed_hand_lift_vertical_force
        t, w = self.task, self.world
        acquisition_before = None
        if getattr(self, 'grasp_acquisition', False) and not getattr(t,'manual_lightweight',False):
            from .tabletop_grasp_acquisition import potential
            if getattr(t,'force_region',False):
                from .tabletop_policy_force_region import potential
            acquisition_before = potential(t.m, self.acquisition_radius_m, self.acquisition_length_m)
        self.reset_substep[:] = 0
        self.substep = 0
        actions = t.begin_step(actions)
        velocity_penalty=torch.zeros(self.n,device=self.device,dtype=torch.float64)
        recovered = torch.zeros_like(self.pending)
        invalid = t.failure==FAILURE_CODES['invalid_action']
        if bool(invalid.any()):
            if not (self.continuous_motion and self.recovery):
                self.fail(invalid,w.measure(),'invalid_policy_action')
            if bool((invalid&self.pending).any()):
                self.fail(invalid&self.pending,w.measure(),'post_reset_invalid_action')
            ids=invalid.nonzero().flatten()
            reset_m=w.reset(ids)
            safe=physical_guard(w,reset_m)[0]&reset_velocity_safe(w)&self.training_joint_safe(reset_m)
            safe &= self.state_valid(reset_m)
            if bool((invalid&~safe).any()):self.fail(invalid&~safe,reset_m,'immediate_reset_invalid_action')
            self.velocity_exceedance_streak_s[ids]=0.
            self.pending |= invalid
            recovered |= invalid
            self.numerical_resets+=invalid.sum()
        for substep in range(1,w.decimation+1):
            self.substep = substep
            live = ~t.done
            if hasattr(self,'control_probe'):self.control_probe.before_step()
            allowed = t.control(actions,live,w.dt)
            if hasattr(self,'lift_gate_counts'):self.lift_gate_counts+=t.lift_diagnostic_masks.sum(0)
            m = w.advance(t,allowed,live)
            invalid_physics=(~finite_physics(w,m) if self.continuous_motion else torch.zeros_like(live))
            scoring_live=live&~invalid_physics
            def scoring(value,fill=0.):
                # Fill only excluded scoring rows, never the raw measurement.
                mask=invalid_physics.reshape((self.n,)+(1,)*(value.ndim-1))
                return torch.where(mask,torch.full_like(value,fill),value)
            if self.fault_trace is not None and self.fault_trace_error is None:
                try:
                    self.fault_trace.record(self,m)
                except Exception as error:
                    # Diagnostic allocation/copy failures must not bypass the
                    # independent physical checks on this same substep.
                    self.fault_trace_error=str(error)
            if hasattr(self,'vertical_force_stats'):
                self.vertical_force_stats.add(scoring(w.vertical_components),scoring(w.object_weight_n,1.),scoring_live)
            # Keep substep peaks before a terminal reset removes the transient.
            # Device-only reductions; this telemetry never changes control/guards.
            sample=torch.stack((m['object_link_loads'].amax(),m['qd'].abs().amax(),
                                w.qd.abs().amax(),m['mimic_errors'].abs().amax()))
            if self.continuous_motion:sample=torch.where(torch.isfinite(sample),sample,0.)
            self.physics_peaks.copy_(torch.maximum(self.physics_peaks,sample))
            measured_speed=self.record_velocity()
            safe, force_candidate, mimic_candidate, limit_reset, drop_reset = physical_guard(
                w,m,include_joint_candidate=True,include_drop_candidate=True)
            joint_safe = self.training_joint_safe(m)
            # A reset must survive a real next physics step before it can ever
            # become eligible for another reset. Frozen terminal rows also checked.
            guarded = self.pending | ~live
            velocity_safe = reset_velocity_safe(w)
            guarded_safe=safe&velocity_safe&joint_safe
            if self.continuous_motion and bool(guarded.any()):guarded_safe &= self.state_valid(m)
            if bool((guarded & ~guarded_safe).any()): self.fail(guarded & ~guarded_safe,m,'post_reset_or_terminal')
            if bool((live & ~velocity_safe).any()) and not (self.continuous_motion and self.recovery):
                self.fail(live & ~velocity_safe,m,'nonfinite_velocity')
            self.pending[:] = False
            previous_phase = t.phase.clone()
            # This is an earlier failed-grasp boundary, never an exemption from
            # a hard guard. Keep the world measurement and prior failures intact.
            joint_candidate = live & ((safe & ~joint_safe & (m['joint_failure']==0)) |
                                      (limit_reset & self.recovery))
            task_m = dict(m,joint_failure=torch.where(joint_candidate & (m['joint_failure']==0),
                FAILURE_CODES['joint_limit'],m['joint_failure']))
            if t.vertical_force_reward:
                # One current normal sample drives both close admission and its
                # endpoint telemetry. Friction is used only in the lift channel.
                components=scoring(w.vertical_components)
                normal_vertical_force=components[:,:,0].sum(dim=1)
                t.consume(task_m,scoring_live,w.dt,vertical_force_n=normal_vertical_force,
                          lift_vertical_force_n=signed_hand_lift_vertical_force(components).sum(dim=1),
                          object_weight_n=scoring(w.object_weight_n,1.) if (t.shaping is not None or getattr(t,'phase_force_reward',False)) else None,
                          hand_lift_force_n=(signed_hand_lift_vertical_force(components)
                              if getattr(t,'phase_force_reward',False) else None))
            else:
                t.consume(task_m,scoring_live,w.dt)
            raw_bad=live&invalid_physics
            if bool(raw_bad.any()):
                t.failure[raw_bad]=FAILURE_CODES['nonfinite_state'];t.nonfinite |= raw_bad
                t.reward[raw_bad]-=25.;t.terms['failure'][raw_bad]-=25.
            if bool((t.done&~velocity_safe).any()) and not (self.continuous_motion and self.recovery):
                self.fail(t.done&~velocity_safe,m,'terminal_velocity')
            changed = previous_phase!=t.phase
            self.transition_counts += torch.bincount((4*previous_phase+t.phase)[changed],minlength=16)
            numerical = (t.failure==FAILURE_CODES['nonfinite_state']) | (t.failure==FAILURE_CODES['nonfinite_joint_state'])
            if bool(numerical.any()) and not (self.continuous_motion and self.recovery):
                self.fail(numerical,m,'nonfinite_task_or_observation')
            candidate = ((force_candidate & (t.failure==FAILURE_CODES['excess_force'])) |
                         (mimic_candidate & (t.failure==FAILURE_CODES['mimic_error'])) |
                         (joint_candidate & t.done & (t.failure!=0)) |
                         (drop_reset & t.done & (t.failure!=0))) & live
            if self.continuous_motion:
                # Any invalid physical row gets one validated local reset.
                # A pending/post-reset row has already been rejected above.
                unclassified=live&~safe&~t.done
                if bool(unclassified.any()):
                    t.failure[unclassified]=FAILURE_CODES['nonfinite_state']
                    t.reward[unclassified]-=25.;t.terms['failure'][unclassified]-=25.
                    numerical |= unclassified
                candidate |= (numerical|~safe)&live
            if not self.recovery: candidate[:] = False
            self.combined_failure_resets+=(candidate&force_candidate&mimic_candidate).sum()
            self.joint_limit_resets+=(candidate&joint_candidate).sum()
            self.object_drop_resets+=(candidate&drop_reset).sum()
            self.numerical_resets+=(candidate&numerical).sum()
            fatal = ~safe & ~candidate
            if bool(fatal.any()): self.fail(fatal,m,'independent_guard')
            # A finite speed excursion changes reward only, never termination.
            cfg=LIVE_VELOCITY_GUARD_CONTRACT
            cost=-cfg['penalty_coefficient']*((measured_speed.double()-cfg['nominal_limit_rad_s']).clamp_min(0.)/
                cfg['penalty_normalizer_rad_s']).square().clamp_max(cfg['penalty_square_cap'])*w.dt
            cost=torch.where(scoring_live&torch.isfinite(cost),cost,0.)
            t.reward+=cost
            velocity_penalty+=cost
            if t.vertical_force_reward and hasattr(self,'close_completion_events'):
                entry=live&(t.failure==0)&(previous_phase==1)&(t.phase==3)
                self.close_completion_events.record(w.vertical_components,normal_vertical_force,
                    self.episodes,t.elapsed,entry,substep)
            if hasattr(self,'lift_entry_force_stats'):
                entry=live&(t.failure==0)&(previous_phase==1)&(t.phase==3)
                self.lift_entry_force_stats.add(w.vertical_components,w.object_weight_n,entry)
            # Ordinary timeouts/successes must not keep driving the old grasp
            # after termination. Validate the terminal state first, then reset
            # physics immediately, retaining the task snapshot until tick end.
            if hasattr(self,'control_probe'):self.control_probe.after_step(live,previous_phase)
            newly_terminal = live & t.done
            ids = newly_terminal.nonzero().flatten()
            if ids.numel():
                if getattr(t,'manual_control',False):
                    self.manual_last_failure=dict(
                        failure_codes=t.failure[ids].detach().cpu().tolist(),
                        object_lift_m=(m['bottom'][ids]-.30).detach().cpu().tolist(),
                        palm_offsets_m=m['adaptive_palm_offset'][ids].detach().cpu().tolist(),
                        hand_load_n=m['hand_object_loads'][ids].detach().cpu().tolist(),
                        mimic_errors_rad=m['mimic_errors'][ids].detach().cpu().tolist(),
                        active_joint_rad=m['q'][ids].detach().cpu().tolist(),
                        active_joint_target_rad=t.command[ids].detach().cpu().tolist(),
                        max_mimic_error_rad=m['mimic_errors'][ids].abs().amax(1).detach().cpu().tolist())
                    if hasattr(w,'adapter_fault_bits'):
                        self.manual_last_failure['adapter_fault_bits']=w.adapter_fault_bits[ids].cpu().tolist()
                    if w.n==1:
                        try:
                            from .tabletop_manual_digit_contacts import collect
                            self.manual_last_failure['digit_contacts']=collect(self)[0]
                            if getattr(t,'manual_digit_force_balance',False):
                                from .tabletop_manual_digit_force_balance import summary
                                self.manual_last_failure['digit_force_balance']=summary(t)
                        except (AttributeError,ValueError,RuntimeError) as error:
                            self.manual_last_failure['digit_contacts_error']=str(error)
                reset_m = w.reset(ids)
                self.velocity_exceedance_streak_s[ids]=0.
                self.reset_substep[ids] = substep
                reset_safe,_,_ = physical_guard(w,reset_m)
                reset_safe &= reset_velocity_safe(w)
                reset_safe &= self.training_joint_safe(reset_m)
                if self.continuous_motion:reset_safe &= self.state_valid(reset_m)
                if bool((newly_terminal & ~reset_safe).any()): self.fail(newly_terminal & ~reset_safe,reset_m,'immediate_reset')
                recovered |= candidate
                self.pending |= newly_terminal
            self.phase_visits.scatter_(1,t.phase[:,None],True)
        obs,rewards,dones,extras = t.end_step()
        extras=dict(extras,terms=dict(extras['terms'],velocity_excess=velocity_penalty.clone()))
        rewards, dones = rewards.clone(), dones.clone()
        if acquisition_before is not None:
            from .tabletop_grasp_acquisition import potential, reward
            if getattr(t,'force_region',False):
                from .tabletop_policy_force_region import potential
            after = potential(t.m, self.acquisition_radius_m, self.acquisition_length_m)
            acquisition = reward(acquisition_before, after, dones)
            rewards += acquisition
            extras = dict(extras, terms=dict(extras['terms'], grasp_acquisition=acquisition.clone()))
        self.returns += rewards
        self.episodes += dones.long()
        self.successes += (dones&t.success).long()
        self.recovered += recovered.long()
        self.last_episode_returns = torch.where(dones,self.returns,self.last_episode_returns)
        self.returns = torch.where(dones,0.,self.returns)
        self.reached_counts += (self.phase_visits&dones[:,None]).sum(0)
        extras = dict(extras,terminal_observation=obs.clone(),recovered=recovered.clone())
        ids = dones.nonzero().flatten()
        if ids.numel():
            # Every terminal slot was physically reset exactly once. Preserve
            # its real post-reset evolution and only reset the task state here.
            reset_m = w.measure()
            safe,_,_ = physical_guard(w,reset_m)
            safe &= reset_velocity_safe(w)
            safe &= self.training_joint_safe(reset_m)
            if self.continuous_motion:safe &= self.state_valid(reset_m)
            if bool((dones&~safe).any()): self.fail(dones&~safe,reset_m,'episode_reset')
            t.reset(ids,reset_m)
            self.pending |= dones
            self.phase_visits[ids] = False
            self.phase_visits[ids,3 if getattr(t,'free_grasp_lift',False) else 1] = True
            obs = t.observe()
        return obs,rewards,dones,extras
