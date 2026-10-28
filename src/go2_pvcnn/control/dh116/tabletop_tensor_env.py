"""GPU task stepping with isolated terminal resets and independent safety guards."""
import torch

PHYSICS_PEAK_NAMES=('link_force_n','active_joint_speed_rad_s','all_joint_speed_rad_s','mimic_error_rad')

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
TRAINING_JOINT_LIMIT_CONTRACT = dict(version='early_active_joint_task_failure_v1',
    scope='ppo_training_active_angular_joints',outside_nominal_limit_rad=.005,
    eligibility='live_strict_physical_safe_no_prior_joint_failure',
    action='joint_limit_failure_then_validated_isolated_reset',
    reset_validation='early_boundary_and_strict_guards_through_next_physics',
    hard_position_tolerance_rad=.02)


def reset_velocity_safe(world):
    # The project-defined speed line is diagnostic, not a hardware safety cap.
    # Keep nonfinite motion fatal in every path; never clamp measured motion.
    return torch.isfinite(world.qd).flatten(1).all(1)


def physical_guard(world, m):
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
    joint_out = ((world.q<world.limits[...,0]-.02)|(world.q>world.limits[...,1]+.02)).flatten(1).any(1)
    velocity_safe = reset_velocity_safe(world)
    force = m['object_link_loads'].flatten(1).amax(1)
    mimic = m['mimic_errors'].abs().amax(1)
    independent = finite & ~joint_out & (m['clearance']>=-.001) & (m['bottom']>=.29)
    independent &= (m['object_link_loads']>=0).flatten(1).all(1)
    strict = independent & velocity_safe & (force<=30.) & (mimic<=.02)
    recover_force = independent & velocity_safe & (force>30.) & (force<45.) & (mimic<=.02)
    recover_mimic = independent & velocity_safe & (mimic>.02) & (mimic<.03) & (force<=30.)
    return strict, recover_force, recover_mimic


class TensorEnv:
    def __init__(self, world, task, recovery=True, fault_trace_path=None,
                 training_joint_limit=False):
        self.world, self.task = world, task
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
        if not self.training_joint_limit:
            return torch.ones_like(self.pending)
        # Only the twelve active angular joints use this task boundary. Full
        # articulation hard guards (including prismatic wrists) remain separate.
        margin=TRAINING_JOINT_LIMIT_CONTRACT['outside_nominal_limit_rad']
        return ((m['q']>=self.task.low-margin)&(m['q']<=self.task.high+margin)).all(1)

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
        raise RuntimeError(f'tensor physical safety failure: {detail}')

    def reset(self):
        ids = torch.arange(self.n,device=self.device)
        m = self.world.reset(ids)
        safe,_,_ = physical_guard(self.world,m)
        safe &= reset_velocity_safe(self.world)
        safe &= self.training_joint_safe(m)
        if not bool(safe.all()): self.fail(~safe,m,'initial_reset')
        self.task.reset(ids,m)
        self.velocity_exceedance_streak_s[ids]=0.
        self.pending[:] = False
        self.reset_substep[:] = 0
        self.substep = 0
        self.phase_visits[:] = False
        self.phase_visits[:,1] = True
        return self.task.observe()

    def step(self, actions):
        from .tabletop_tensor_task import FAILURE_CODES
        from .tabletop_tensor_world import signed_hand_lift_vertical_force
        t, w = self.task, self.world
        self.reset_substep[:] = 0
        self.substep = 0
        actions = t.begin_step(actions)
        velocity_penalty=torch.zeros(self.n,device=self.device,dtype=torch.float64)
        invalid = t.failure==FAILURE_CODES['invalid_action']
        if bool(invalid.any()): self.fail(invalid,w.measure(),'invalid_policy_action')
        recovered = torch.zeros_like(self.pending)
        for substep in range(1,w.decimation+1):
            self.substep = substep
            live = ~t.done
            if hasattr(self,'control_probe'):self.control_probe.before_step()
            allowed = t.control(actions,live,w.dt)
            if hasattr(self,'lift_gate_counts'):self.lift_gate_counts+=t.lift_diagnostic_masks.sum(0)
            m = w.advance(t,allowed,live)
            if self.fault_trace is not None and self.fault_trace_error is None:
                try:
                    self.fault_trace.record(self,m)
                except Exception as error:
                    # Diagnostic allocation/copy failures must not bypass the
                    # independent physical checks on this same substep.
                    self.fault_trace_error=str(error)
            if hasattr(self,'vertical_force_stats'):
                self.vertical_force_stats.add(w.vertical_components,w.object_weight_n,live)
            # Keep substep peaks before a terminal reset removes the transient.
            # Device-only reductions; this telemetry never changes control/guards.
            sample=torch.stack((m['object_link_loads'].amax(),m['qd'].abs().amax(),
                                w.qd.abs().amax(),m['mimic_errors'].abs().amax()))
            self.physics_peaks.copy_(torch.maximum(self.physics_peaks,sample))
            measured_speed=self.record_velocity()
            safe, force_candidate, mimic_candidate = physical_guard(w,m)
            joint_safe = self.training_joint_safe(m)
            # A reset must survive a real next physics step before it can ever
            # become eligible for another reset. Frozen terminal rows also checked.
            guarded = self.pending | ~live
            velocity_safe = reset_velocity_safe(w)
            if bool((guarded & ~(safe&velocity_safe&joint_safe)).any()): self.fail(guarded & ~(safe&velocity_safe&joint_safe),m,'post_reset_or_terminal')
            if bool((live & ~velocity_safe).any()): self.fail(live & ~velocity_safe,m,'nonfinite_velocity')
            self.pending[:] = False
            previous_phase = t.phase.clone()
            # This is an earlier failed-grasp boundary, never an exemption from
            # a hard guard. Keep the world measurement and prior failures intact.
            joint_candidate = live & safe & ~joint_safe & (m['joint_failure']==0)
            task_m = dict(m,joint_failure=torch.where(joint_candidate,
                FAILURE_CODES['joint_limit'],m['joint_failure']))
            if t.vertical_force_reward:
                # One current normal sample drives both close admission and its
                # endpoint telemetry. Friction is used only in the lift channel.
                normal_vertical_force=w.vertical_components[:,:,0].sum(dim=1)
                t.consume(task_m,live,w.dt,vertical_force_n=normal_vertical_force,
                          lift_vertical_force_n=signed_hand_lift_vertical_force(w.vertical_components).sum(dim=1),
                          object_weight_n=w.object_weight_n if t.shaping is not None else None)
            else:
                t.consume(task_m,live,w.dt)
            if bool((t.done&~velocity_safe).any()): self.fail(t.done&~velocity_safe,m,'terminal_velocity')
            changed = previous_phase!=t.phase
            self.transition_counts += torch.bincount((4*previous_phase+t.phase)[changed],minlength=16)
            numerical = (t.failure==FAILURE_CODES['nonfinite_state']) | (t.failure==FAILURE_CODES['nonfinite_joint_state'])
            if bool(numerical.any()): self.fail(numerical,m,'nonfinite_task_or_observation')
            candidate = ((force_candidate & (t.failure==FAILURE_CODES['excess_force'])) |
                         (mimic_candidate & (t.failure==FAILURE_CODES['mimic_error'])) |
                         (joint_candidate & (t.failure==FAILURE_CODES['joint_limit']))) & live
            if not self.recovery: candidate[:] = False
            fatal = ~safe & ~candidate
            if bool(fatal.any()): self.fail(fatal,m,'independent_guard')
            # A finite speed excursion changes reward only, never termination.
            cfg=LIVE_VELOCITY_GUARD_CONTRACT
            cost=-cfg['penalty_coefficient']*((measured_speed.double()-cfg['nominal_limit_rad_s']).clamp_min(0.)/
                cfg['penalty_normalizer_rad_s']).square().clamp_max(cfg['penalty_square_cap'])*w.dt
            cost=torch.where(live,cost,0.)
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
                reset_m = w.reset(ids)
                self.velocity_exceedance_streak_s[ids]=0.
                self.reset_substep[ids] = substep
                reset_safe,_,_ = physical_guard(w,reset_m)
                reset_safe &= reset_velocity_safe(w)
                reset_safe &= self.training_joint_safe(reset_m)
                if bool((newly_terminal & ~reset_safe).any()): self.fail(newly_terminal & ~reset_safe,reset_m,'immediate_reset')
                recovered |= candidate
                self.pending |= newly_terminal
            self.phase_visits.scatter_(1,t.phase[:,None],True)
        obs,rewards,dones,extras = t.end_step()
        extras=dict(extras,terms=dict(extras['terms'],velocity_excess=velocity_penalty.clone()))
        rewards, dones = rewards.clone(), dones.clone()
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
            if bool((dones&~safe).any()): self.fail(dones&~safe,reset_m,'episode_reset')
            t.reset(ids,reset_m)
            self.pending |= dones
            self.phase_visits[ids] = False
            self.phase_visits[ids,1] = True
            obs = t.observe()
        return obs,rewards,dones,extras
