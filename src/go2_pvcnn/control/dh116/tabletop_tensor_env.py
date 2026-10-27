"""GPU task stepping with isolated terminal resets and independent safety guards."""
import torch

PHYSICS_PEAK_NAMES=('link_force_n','active_joint_speed_rad_s','all_joint_speed_rad_s','mimic_error_rad')

VELOCITY_GUARD_CONTRACT = dict(nominal_limit_rad_s=3.,absolute_tolerance_rad_s=1e-5,
    comparison='finite_abs_le_limit_plus_atol',scope='terminal_reset_recovery')
LIVE_VELOCITY_GUARD_CONTRACT = dict(VELOCITY_GUARD_CONTRACT,
    scope='all_joints_every_physics_substep',action='fatal_before_reward_or_reset')
TERMINAL_RESET_CONTRACT = dict(version='validated_terminal_substep_reset_v1',
    timing='after_validated_terminal_substep',
    transition='preserved_until_control_tick_end',next_physics='strict_fresh_guards')


def reset_velocity_safe(world):
    # Fixed numerical solver allowance, not float64 conversion error or a
    # change to the physical velocity cap. Never clamp the measured velocity.
    velocity=world.qd.abs().flatten(1).amax(1)
    bound=VELOCITY_GUARD_CONTRACT['nominal_limit_rad_s']+VELOCITY_GUARD_CONTRACT['absolute_tolerance_rad_s']
    return torch.isfinite(velocity)&(velocity<=bound)


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
    def __init__(self, world, task, recovery=True):
        self.world, self.task = world, task
        self.n, self.device = world.n, world.device
        self.recovery = recovery
        self.physics_peaks=torch.zeros(4,device=self.device,dtype=torch.float64)
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
        raise RuntimeError(f'tensor physical safety failure: {detail}')

    def reset(self):
        ids = torch.arange(self.n,device=self.device)
        m = self.world.reset(ids)
        safe,_,_ = physical_guard(self.world,m)
        safe &= reset_velocity_safe(self.world)
        if not bool(safe.all()): self.fail(~safe,m,'initial_reset')
        self.task.reset(ids,m)
        self.pending[:] = False
        self.reset_substep[:] = 0
        self.substep = 0
        self.phase_visits[:] = False
        self.phase_visits[:,1] = True
        return self.task.observe()

    def step(self, actions):
        from .tabletop_tensor_task import FAILURE_CODES
        t, w = self.task, self.world
        self.reset_substep[:] = 0
        self.substep = 0
        actions = t.begin_step(actions)
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
            if hasattr(self,'vertical_force_stats'):
                self.vertical_force_stats.add(w.vertical_components,w.object_weight_n,live)
            # Keep substep peaks before a terminal reset removes the transient.
            # Device-only reductions; this telemetry never changes control/guards.
            sample=torch.stack((m['object_link_loads'].amax(),m['qd'].abs().amax(),
                                w.qd.abs().amax(),m['mimic_errors'].abs().amax()))
            self.physics_peaks.copy_(torch.maximum(self.physics_peaks,sample))
            safe, force_candidate, mimic_candidate = physical_guard(w,m)
            # A reset must survive a real next physics step before it can ever
            # become eligible for another reset. Frozen terminal rows also checked.
            guarded = self.pending | ~live
            velocity_safe = reset_velocity_safe(w)
            if bool((guarded & ~(safe&velocity_safe)).any()): self.fail(guarded & ~(safe&velocity_safe),m,'post_reset_or_terminal')
            # Solver velocity limits are not a guarantee on measured motion.
            # Stop at the first violating substep, before reward or any reset.
            if bool((live & ~velocity_safe).any()): self.fail(live & ~velocity_safe,m,'live_velocity')
            self.pending[:] = False
            previous_phase = t.phase.clone()
            if t.vertical_force_reward:
                # Sum both hands and signed normal/friction BEFORE reset.
                t.consume(m,live,w.dt,vertical_force_n=w.vertical_components.sum(dim=(1,2)),
                          object_weight_n=w.object_weight_n if t.shaping is not None else None)
            else:
                t.consume(m,live,w.dt)
            if bool((t.done&~velocity_safe).any()): self.fail(t.done&~velocity_safe,m,'terminal_velocity')
            changed = previous_phase!=t.phase
            self.transition_counts += torch.bincount((4*previous_phase+t.phase)[changed],minlength=16)
            numerical = (t.failure==FAILURE_CODES['nonfinite_state']) | (t.failure==FAILURE_CODES['nonfinite_joint_state'])
            if bool(numerical.any()): self.fail(numerical,m,'nonfinite_task_or_observation')
            candidate = ((force_candidate & (t.failure==FAILURE_CODES['excess_force'])) |
                         (mimic_candidate & (t.failure==FAILURE_CODES['mimic_error']))) & live
            if not self.recovery: candidate[:] = False
            fatal = ~safe & ~candidate
            if bool(fatal.any()): self.fail(fatal,m,'independent_guard')
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
                self.reset_substep[ids] = substep
                reset_safe,_,_ = physical_guard(w,reset_m)
                reset_safe &= reset_velocity_safe(w)
                if bool((newly_terminal & ~reset_safe).any()): self.fail(newly_terminal & ~reset_safe,reset_m,'immediate_reset')
                recovered |= candidate
                self.pending |= newly_terminal
            self.phase_visits.scatter_(1,t.phase[:,None],True)
        obs,rewards,dones,extras = t.end_step()
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
            if bool((dones&~safe).any()): self.fail(dones&~safe,reset_m,'episode_reset')
            t.reset(ids,reset_m)
            self.pending |= dones
            self.phase_visits[ids] = False
            self.phase_visits[ids,1] = True
            obs = t.observe()
        return obs,rewards,dones,extras
