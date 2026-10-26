"""Signed vertical-force gated release, downward reposition, and reclose."""

import torch


CONTRACT = dict(
    version='force_direction_release_reposition_v1',
    trigger_force_n=-.15,
    release_force_n=-.05,
    reclose_force_n=-.15,
    contact_load_n=.1,
    released_load_n=.05,
    released_dwell_s=.05,
    correction_step_m=.001,
    max_downward_step_m=.002,
    arrival_tolerance_m=.0002,
    arrival_dwell_s=.05,
    adjust_timeout_s=2.,
    reclose_timeout_s=1.,
    max_attempts_per_hand=2,
    release_joints='thumb_and_index',
    goal_table_clearance_m=.0021,
    phases='close_only',
    policy_downward_positioning=True,
    actual_velocity_guaranteed=False,
    rewards_changed=False,
    hard_guards_changed=False,
)


class ForceDirectionReposition:
    """Device-resident state machine for correcting negative object Fz."""

    def __init__(self, mask):
        self.mask = torch.as_tensor(mask, dtype=torch.bool).clone()
        n = len(self.mask)
        device = self.mask.device
        self.state = torch.zeros((n, 2), device=device, dtype=torch.long)
        self.attempts = torch.zeros_like(self.state)
        self.aborted = torch.zeros_like(self.state, dtype=torch.bool)
        self.clock = torch.zeros((n, 2), device=device, dtype=torch.float64)
        self.dwell = torch.zeros_like(self.clock)
        self.goal = torch.zeros((n, 2, 3), device=device, dtype=torch.float64)
        self.hold = torch.zeros_like(self.goal)
        self.goal_clipped = torch.zeros_like(self.aborted)
        self.force_before = torch.zeros((n, 2), device=device, dtype=torch.float64)
        self.force_release = torch.zeros_like(self.force_before)
        self.force_move = torch.zeros_like(self.force_before)
        self.force_reclose = torch.zeros_like(self.force_before)
        self.release_timeout = torch.zeros_like(self.aborted)
        self.recontact_abort = torch.zeros_like(self.aborted)
        self.reclose_timeout = torch.zeros_like(self.aborted)

    @property
    def active(self):
        return self.state != 0

    def reset(self, ids):
        ids = torch.as_tensor(ids, device=self.mask.device, dtype=torch.long)
        for name in (
            'state', 'attempts', 'aborted', 'clock', 'dwell', 'goal', 'hold',
            'goal_clipped', 'force_before', 'force_release', 'force_move',
            'force_reclose', 'release_timeout', 'recontact_abort',
            'reclose_timeout',
        ):
            getattr(self, name)[ids] = 0

    def _clip_goal(self, goal, current, clearance):
        lower = current[:, :, 2] - CONTRACT['max_downward_step_m']
        goal[:, :, 2] = torch.maximum(goal[:, :, 2], lower)
        if clearance is not None:
            safe = current[:, :, 2] - (
                clearance - CONTRACT['goal_table_clearance_m']
            ).clamp_min(0.)
            clipped = goal[:, :, 2] < safe
            self.goal_clipped |= clipped
            goal[:, :, 2] = torch.maximum(goal[:, :, 2], safe)
        return goal

    def apply(self, target, current, loads, phase, live, dt,
              vertical_hand_force_n, *, clearance=None):
        selected = self.mask[:, None] & live[:, None] & (phase == 1)[:, None]
        fz = torch.as_tensor(vertical_hand_force_n, device=current.device,
                             dtype=current.dtype)
        if fz.shape != loads.shape:
            raise ValueError('vertical_hand_force_n must have shape (N, 2)')

        inactive = self.active & ~selected
        self.state[inactive] = 0
        self.clock[inactive] = 0.
        self.dwell[inactive] = 0.

        aggregate_negative = fz.sum(1, keepdim=True) < CONTRACT['trigger_force_n']
        negative = fz < CONTRACT['trigger_force_n']
        trigger = selected & (self.state == 0) & ~self.aborted
        trigger &= self.attempts < CONTRACT['max_attempts_per_hand']
        trigger &= loads > CONTRACT['contact_load_n']
        trigger &= negative | (aggregate_negative & (fz < 0.))
        if bool(trigger.any()):
            goal = target.clone()
            goal[:, :, 2] = torch.minimum(
                goal[:, :, 2], current[:, :, 2] - CONTRACT['correction_step_m'])
            self.goal[:] = self._clip_goal(goal, current, clearance)
            self.hold[trigger] = current[trigger]
            self.force_before[trigger] = fz[trigger]
            self.attempts[trigger] += 1
            self.state[trigger] = 1
            self.clock[trigger] = 0.
            self.dwell[trigger] = 0.

        active_selected = selected & self.active
        self.clock[active_selected] += dt

        releasing = selected & (self.state == 1)
        released_now = (loads <= CONTRACT['released_load_n']) & (
            fz >= CONTRACT['release_force_n'])
        self.dwell[releasing] = torch.where(
            released_now[releasing], self.dwell[releasing] + dt, 0.)
        released = releasing & (
            self.dwell >= CONTRACT['released_dwell_s'] - 1e-12)
        if bool(released.any()):
            self.force_release[released] = fz[released]
            self.state[released] = 2
            self.dwell[released] = 0.

        moving = selected & (self.state == 2) & ~released
        recontact = moving & ((loads > CONTRACT['released_load_n']) | (
            fz < CONTRACT['trigger_force_n']))
        if bool(recontact.any()):
            self.recontact_abort[recontact] = True
            self.hold[recontact] = current[recontact]
            self.state[recontact] = 1
            self.dwell[recontact] = 0.
        moving = selected & (self.state == 2) & ~recontact
        arrived = (self.goal - current).norm(dim=2) <= CONTRACT['arrival_tolerance_m']
        self.dwell[moving] = torch.where(
            arrived[moving] & released_now[moving],
            self.dwell[moving] + dt,
            0.,
        )
        finished = moving & (
            self.dwell >= CONTRACT['arrival_dwell_s'] - 1e-12)
        if bool(finished.any()):
            self.force_move[finished] = fz[finished]
            self.state[finished] = 3
            self.clock[finished] = 0.
            self.dwell[finished] = 0.

        timeout = selected & ((self.state == 1) | (self.state == 2)) & (
            self.clock >= CONTRACT['adjust_timeout_s'])
        if bool(timeout.any()):
            self.release_timeout[timeout] = self.state[timeout] == 1
            self.aborted[timeout] = True
            self.goal[timeout] = current[timeout]
            self.state[timeout] = 0
            self.clock[timeout] = 0.
            self.dwell[timeout] = 0.

        reclose = selected & (self.state == 3)
        reclose_ready = (loads > CONTRACT['contact_load_n']) & (
            fz >= CONTRACT['reclose_force_n'])
        self.dwell[reclose] = torch.where(
            reclose_ready[reclose], self.dwell[reclose] + dt, 0.)
        complete = reclose & (
            self.dwell >= CONTRACT['released_dwell_s'] - 1e-12)
        if bool(complete.any()):
            self.force_reclose[complete] = fz[complete]
            self.state[complete] = 0
            self.clock[complete] = 0.
            self.dwell[complete] = 0.
        reclose_timeout = reclose & (
            self.clock >= CONTRACT['reclose_timeout_s'])
        if bool(reclose_timeout.any()):
            self.reclose_timeout[reclose_timeout] = True
            self.aborted[reclose_timeout] = True
            self.state[reclose_timeout] = 0
            self.clock[reclose_timeout] = 0.
            self.dwell[reclose_timeout] = 0.

        result = target.clone()
        wait = selected & (self.state == 1)
        move = selected & (self.state == 2)
        result[wait] = self.hold[wait]
        result[move] = self.goal[move]
        return result, selected & ((self.state == 1) | (self.state == 2))

