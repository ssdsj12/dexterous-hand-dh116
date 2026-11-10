"""Pure measured per-pad recovery state and conserved execution ledger.

No simulation, planner, force controller or success observer is installed here.
The caller supplies real object-filtered measurements at 480 Hz. Geometry can
measure progress but cannot establish contact. Approved approach is a command
distance after native guards, not measured displacement or object drift.

Example::

    recovery = TripodContactRecovery(n, device)  # right thumb/index/middle
    recovery.reset(ids, episode_ids=physical_episode_ids, source_sha=source_sha)
    recovery.observe_substep(**actual_pre_reset_measurement)
    recovery.request_retry(episode_ids=physical_episode_ids, control_tick=tick,
        observation_revision=step, geometry_eligible=eligible, blocked_reason='none')
    ticket = recovery.reserve_approach(episode_ids=physical_episode_ids,
        control_tick=tick, dt_s=1/480, requested_approach_m=request)
    # Revalidate the encoded candidate, then run the unchanged native guards.
    recovery.commit_execution(episode_ids=physical_episode_ids,
        reservation_id=ticket['reservation_id'], execution_id=execution_id,
        physics_step=step, approved_approach_m=approved_distance,
        executed_mask=executed, requested_action24=requested_action,
        final_target_rad=post_guard_target, approved_action24=approved_action)

20 ms STABLE is recovery qualification only, never native grasp success.
"""
from __future__ import annotations

import math

import torch


STATE_CODES = dict(LOST=0, REACQUIRING=1, CONTACT=2, STABLE=3)
BLOCK_CODES = dict(none=0, geometry_ineligible=1, model_infeasible=2,
    sensor_unknown=3, no_progress=4, distance_exhausted=5, time_exhausted=6,
    terminal=7, waiting_new_measurement=8, pressure_guard=9)


def _owned(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _owned(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_owned(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_owned(item) for item in value)
    return value


class TripodContactRecovery:
    """Independent [N,2,5] state; only real episode reset replenishes caps.

    ``measurement_valid`` is [N] or [N,2,5], independent of ``pad_valid``.
    Healthy load=0/pad_valid=False is known absence, not sensor failure.
    Optional ``sensor_healthy`` isolates individual broken pads. STABLE needs
    load>.05 N, finite nonzero force normal, depth<=.6 mm (the original pressure
    retreat threshold), and rejected hand load<=.05 N, continuously for20 ms.

    ``measured_gap_m`` is actual current pad-region/object separation. Optional
    ``measured_geometry_signature`` [N,2,5,6] contains a measured region witness
    xyz and outward normal xyz; it permits retry after a real pose/normal change
    even when distance is unchanged. A new tick/revision alone cannot restart a
    no-progress round. Invalid geometry never counts as progress.

    Open REACQUIRING rounds spend simulation time even when a transient solver
    failure permits no motion. CONTACT pauses the same round; flicker preserves
    its watchdog. STABLE closes it. A no-progress stop needs a new measured
    condition before another round and never replenishes episode budgets.
    """
    LOST, REACQUIRING, CONTACT, STABLE = (STATE_CODES[k] for k in STATE_CODES)

    def __init__(self, n, device, *, enabled_pad_mask=None, recovery_stable_s=.02):
        if type(n) is not int or n < 1:
            raise ValueError('positive integer n required')
        if (type(recovery_stable_s) not in (int, float)
                or not math.isfinite(recovery_stable_s) or recovery_stable_s < .02):
            raise ValueError('recovery_stable_s must be finite and at least .02')
        self.n, self.device = n, torch.device(device)
        self.recovery_stable_s = float(recovery_stable_s)
        self.shape = (n, 2, 5)
        if enabled_pad_mask is None:
            enabled_pad_mask = torch.zeros(self.shape, dtype=torch.bool, device=self.device)
            enabled_pad_mask[:, 1, :3] = True
        self._tensor(enabled_pad_mask, self.shape, 'enabled_pad_mask', torch.bool)
        self.enabled_pad_mask = enabled_pad_mask.detach().clone()
        self.episode_ids = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.last_physics_step = self.episode_ids.clone()
        self.last_execution_step = self.episode_ids.clone()
        self.episode_ended = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.source_shas = [None] * n
        self.state = torch.full(self.shape, self.LOST, dtype=torch.long, device=self.device)
        for name in ('contact_known', 'measured_contact', 'stable', 'round_active',
                     'round_stopped', 'distance_exhausted', 'time_exhausted',
                     'geometry_valid', 'stop_geometry_valid'):
            setattr(self, name, torch.zeros(self.shape, dtype=torch.bool, device=self.device))
        for name in ('stable_time_s', 'attempt_time_s', 'executed_approach_m',
                     'reserved_approach_m', 'no_progress_s'):
            setattr(self, name, torch.zeros(self.shape, dtype=torch.float64, device=self.device))
        self.round_id = torch.zeros(self.shape, dtype=torch.long, device=self.device)
        self.blocked_reason = torch.zeros_like(self.round_id)
        self.measured_gap_m = torch.full_like(self.attempt_time_s, float('nan'))
        self.best_gap_m = torch.full_like(self.attempt_time_s, float('inf'))
        self.stop_gap_m = self.measured_gap_m.clone()
        self.geometry_signature = torch.full((*self.shape, 6), float('nan'), device=self.device, dtype=torch.float64)
        self.stop_signature = self.geometry_signature.clone()
        self._step_approved_m = torch.zeros_like(self.executed_approach_m)
        # A guard-approved interval remains owed even when that same commit
        # exhausts distance and closes the round before its physical sample.
        self.pending_attempt_step = torch.full_like(self.round_id, -1)
        self.pending_attempt_episode_ids = torch.full_like(self.round_id, -1)
        self.pending_attempt_round_id = torch.full_like(self.round_id, -1)
        self.events = []
        self._reservations, self._commits = {}, {}
        self._next_reservation = 0

    def _tensor(self, value, shape, name, dtype=None):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.device != self.device
                or (value.dtype != dtype if dtype is not None else not value.is_floating_point())):
            raise ValueError(name + ' has invalid shape/dtype/device')
        return value

    def _episodes(self, episode_ids):
        self._tensor(episode_ids, (self.n,), 'episode_ids', torch.long)
        if bool((self.episode_ids < 0).any()) or not torch.equal(episode_ids, self.episode_ids):
            raise ValueError('stale or uninitialized physical episode identity')

    @staticmethod
    def _integer(value, name):
        if type(value) is not int or value < 0:
            raise ValueError(name + ' must be a nonnegative integer')

    @staticmethod
    def _dt(value):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1.:
            raise ValueError('dt_s must be finite in (0,1]')

    def _event(self, kind, mask, **extra):
        for env, hand, digit in mask.nonzero().tolist():
            self.events.append(dict(kind=kind, env=env, hand=hand, digit=digit,
                episode_id=int(self.episode_ids[env]), source_sha=self.source_shas[env],
                physics_step=int(self.last_physics_step[env]), round_id=int(self.round_id[env, hand, digit]),
                **_owned(extra)))

    @torch.no_grad()
    def reset(self, ids, *, episode_ids, source_sha):
        """Clear selected rows only for a strictly newer physical episode ID.

        A new independent run creates a new FSM. Replayed/rolled-back native
        reset packets cannot replenish an existing run's conserved budgets.
        """
        ids = torch.as_tensor(ids, device=self.device)
        if ids.dtype == torch.bool:
            self._tensor(ids, (self.n,), 'ids', torch.bool)
            ids = ids.nonzero().flatten()
        if ids.ndim != 1 or ids.dtype != torch.long or bool(((ids < 0) | (ids >= self.n)).any()):
            raise ValueError('ids must be selected int64 rows')
        if len(ids.unique()) != len(ids):
            raise ValueError('reset ids must be unique')
        if (not isinstance(source_sha, str) or len(source_sha) != 64
                or any(c not in '0123456789abcdef' for c in source_sha)):
            raise ValueError('source_sha must identify the source with SHA256')
        if not isinstance(episode_ids, torch.Tensor) or episode_ids.device != self.device or episode_ids.dtype != torch.long:
            raise ValueError('episode_ids require native int64 identities')
        new = episode_ids[ids] if episode_ids.shape == (self.n,) else episode_ids
        if new.shape != ids.shape or bool((new < 0).any()) or bool((new <= self.episode_ids[ids]).any()):
            raise ValueError('reset requires strictly increasing physical episode identities')
        selected = set(ids.tolist())
        self.events = [e for e in self.events if e['env'] not in selected]
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == self.n and name != 'enabled_pad_mask':
                if name in ('episode_ids', 'last_physics_step', 'last_execution_step',
                            'pending_attempt_step', 'pending_attempt_episode_ids', 'pending_attempt_round_id'):
                    value[ids] = -1
                elif name in ('measured_gap_m', 'stop_gap_m', 'geometry_signature', 'stop_signature'):
                    value[ids] = float('nan')
                elif name == 'best_gap_m':
                    value[ids] = float('inf')
                else:
                    value[ids] = 0
        self.episode_ids[ids] = new
        for env in selected:
            self.source_shas[env] = source_sha
        for reservation in self._reservations.values():
            reservation['allowed_distance_m'][ids] = 0.
            reservation['requested_approach_m'][ids] = 0.
            reservation['episode_ids'][ids] = -1
            reservation['physics_steps'][ids] = -1
            reservation['settled'][ids] = True
        self._reservations = {key: value for key, value in self._reservations.items()
                              if not bool(value['settled'].all())}
        # A batch identity changes on selected reset. Existing reservations stay
        # settled for unchanged siblings, so discarded batch deduplication keys
        # cannot make an old execution chargeable again.
        self._commits.clear()

    @torch.no_grad()
    def observe_substep(self, *, episode_ids, physics_step, dt_s, live, terminal,
            measurement_valid, pad_valid, loads_n, vectors_w, depth_m,
            rejected_loads_n, measured_gap_m, geometry_valid,
            sensor_healthy=None, measured_geometry_signature=None):
        self._episodes(episode_ids)
        self._integer(physics_step, 'physics_step'); self._dt(dt_s)
        if abs(dt_s - 1/480) > 1e-12:
            raise ValueError('contact recovery observations require original 480 Hz dt_s')
        for name, value in (('live', live), ('terminal', terminal)):
            self._tensor(value, (self.n,), name, torch.bool)
        active = live & ~self.episode_ended
        old_step = self.last_physics_step[active]
        if bool(((old_step >= 0) & (old_step + 1 != physics_step)).any()):
            raise ValueError('stale, duplicate or missing physics_step')
        if not isinstance(measurement_valid, torch.Tensor):
            raise ValueError('measurement_valid requires an independent boolean tensor')
        if measurement_valid.shape == (self.n,):
            self._tensor(measurement_valid, (self.n,), 'measurement_valid', torch.bool)
            health = measurement_valid[:, None, None].expand(self.shape)
        else:
            health = self._tensor(measurement_valid, self.shape, 'measurement_valid', torch.bool)
        self._tensor(pad_valid, self.shape, 'pad_valid', torch.bool)
        self._tensor(geometry_valid, self.shape, 'geometry_valid', torch.bool)
        for name, value, shape in (('loads_n', loads_n, self.shape), ('depth_m', depth_m, self.shape),
                ('vectors_w', vectors_w, (*self.shape, 3)), ('measured_gap_m', measured_gap_m, self.shape),
                ('rejected_loads_n', rejected_loads_n, (self.n, 2))):
            self._tensor(value, shape, name)
        if sensor_healthy is not None:
            health = health & self._tensor(sensor_healthy, self.shape, 'sensor_healthy', torch.bool)
        signature = torch.full_like(self.geometry_signature, float('nan'))
        if measured_geometry_signature is not None:
            signature = self._tensor(measured_geometry_signature, (*self.shape, 6), 'measured_geometry_signature')
        # All schema and clock checks precede changes. Bad numerical data is
        # unknown for the affected pad, not a manufactured measured contact loss.
        enabled = self.enabled_pad_mask & active[:, None, None]
        finite = (torch.isfinite(loads_n) & (loads_n >= 0) & torch.isfinite(vectors_w).all(-1)
            & torch.isfinite(depth_m) & (depth_m >= 0)
            & torch.isfinite(rejected_loads_n)[..., None] & (rejected_loads_n >= 0)[..., None])
        known = enabled & health & finite & ~((loads_n > .05) & ~pad_valid)
        contact = known & pad_valid & (loads_n > .05)
        normal = vectors_w.norm(dim=-1)
        usable = contact & torch.isfinite(normal) & (normal > 0) & (depth_m <= .0006) & (rejected_loads_n <= .05)[..., None]
        geom_ok = enabled & health & finite & geometry_valid & torch.isfinite(measured_gap_m) & (measured_gap_m >= 0)
        old_contact, old_known = self.measured_contact.clone(), self.contact_known.clone()
        old_stable = self.stable.clone()
        # Tickets authorize only the immediately following physical substep.
        # Unused candidates cannot be carried to a newer measured state.
        for identity, reservation in list(self._reservations.items()):
            expired = active & ~reservation['settled'] & (reservation['physics_steps'] < physics_step)
            self.reserved_approach_m[expired] = (self.reserved_approach_m[expired]
                - reservation['allowed_distance_m'][expired]).clamp_min(0.)
            reservation['settled'][expired] = True
            if bool(reservation['settled'].all()):
                del self._reservations[identity]
        self._step_approved_m[active] = 0.
        committed_interval = ((self.pending_attempt_step == physics_step)
            & (self.pending_attempt_episode_ids == self.episode_ids[:, None, None])
            & (self.pending_attempt_round_id == self.round_id))
        counting = enabled & ((self.round_active & (self.state == self.REACQUIRING)) | committed_interval)
        progress = counting & geom_ok & torch.isfinite(self.best_gap_m) & (measured_gap_m < self.best_gap_m - 1e-6)
        self.attempt_time_s += counting.to(torch.float64) * dt_s
        self.attempt_time_s.clamp_max_(1.)
        consumed_interval = active[:, None, None] & (self.pending_attempt_step <= physics_step)
        for name in ('pending_attempt_step', 'pending_attempt_episode_ids', 'pending_attempt_round_id'):
            value = getattr(self, name)
            value.masked_fill_(consumed_interval, -1)
        self.no_progress_s = torch.where(progress, 0., self.no_progress_s + counting.to(torch.float64) * dt_s)
        update_best = counting & geom_ok & (progress | ~torch.isfinite(self.best_gap_m))
        self.best_gap_m = torch.where(update_best, measured_gap_m.to(torch.float64), self.best_gap_m)
        self.stable_time_s = torch.where(enabled,
            torch.where(usable & ~terminal[:, None, None], self.stable_time_s + dt_s, 0.), self.stable_time_s)
        self.stable = torch.where(enabled, usable & (self.stable_time_s >= self.recovery_stable_s - 1e-12), self.stable)
        self.contact_known = torch.where(active[:, None, None], known, self.contact_known)
        self.measured_contact = torch.where(active[:, None, None], contact, self.measured_contact)
        self.geometry_valid = torch.where(active[:, None, None], geom_ok, self.geometry_valid)
        self.measured_gap_m = torch.where(active[:, None, None], measured_gap_m.to(torch.float64), self.measured_gap_m)
        self.geometry_signature = torch.where(active[:, None, None, None], signature.to(torch.float64), self.geometry_signature)
        self.last_physics_step[active] = physics_step
        self.time_exhausted |= self.enabled_pad_mask & (self.attempt_time_s >= 1. - 1e-12)
        self.round_active &= ~self.stable
        stopped = counting & ~contact & (self.no_progress_s >= .5 - 1e-12) & ~self.time_exhausted
        self.round_stopped |= stopped
        self.round_active &= ~(stopped | self.time_exhausted | self.distance_exhausted)
        self.stop_gap_m = torch.where(stopped, self.measured_gap_m, self.stop_gap_m)
        self.stop_geometry_valid = torch.where(stopped, self.geometry_valid, self.stop_geometry_valid)
        self.stop_signature = torch.where(stopped[..., None], self.geometry_signature, self.stop_signature)
        new_state = torch.where(self.round_active, self.REACQUIRING, self.LOST)
        new_state = torch.where(contact, self.CONTACT, new_state)
        new_state = torch.where(self.stable, self.STABLE, new_state)
        self.state = torch.where(enabled, new_state, self.state)
        self.blocked_reason = torch.where(enabled & ~known, BLOCK_CODES['sensor_unknown'], self.blocked_reason)
        self.blocked_reason = torch.where(stopped, BLOCK_CODES['no_progress'], self.blocked_reason)
        self.blocked_reason = torch.where(self.time_exhausted, BLOCK_CODES['time_exhausted'], self.blocked_reason)
        ending = enabled & terminal[:, None, None]
        self.round_active &= ~ending
        self.stable &= ~ending
        self.stable_time_s = torch.where(ending, 0., self.stable_time_s)
        self.state = torch.where(ending, self.LOST, self.state)
        self.blocked_reason = torch.where(ending, BLOCK_CODES['terminal'], self.blocked_reason)
        self.episode_ended |= terminal & live
        self._event('contact_lost', enabled & old_contact & known & ~contact & ~ending)
        self._event('contact_acquired', contact & ~old_contact & old_known & ~ending)
        self._event('contact_observed', contact & ~old_contact & ~old_known & ~ending)
        self._event('sensor_unknown', enabled & old_known & ~known & ~ending)
        self._event('stable', self.stable & ~old_stable)
        self._event('no_progress_stop', stopped)
        self._event('episode_terminal', ending)

    @torch.no_grad()
    def request_retry(self, *, episode_ids, control_tick, observation_revision,
                      geometry_eligible, blocked_reason='none'):
        self._episodes(episode_ids)
        self._integer(control_tick, 'control_tick'); self._integer(observation_revision, 'observation_revision')
        self._tensor(geometry_eligible, self.shape, 'geometry_eligible', torch.bool)
        if isinstance(blocked_reason, str):
            if blocked_reason not in BLOCK_CODES:
                raise ValueError('unknown blocked_reason')
            reason = torch.full_like(self.blocked_reason, BLOCK_CODES[blocked_reason])
        else:
            reason = self._tensor(blocked_reason, self.shape, 'blocked_reason', torch.long)
            if bool(((reason < 0) | (reason >= len(BLOCK_CODES))).any()):
                raise ValueError('unknown blocked_reason code')
        base = (self.enabled_pad_mask & ~self.episode_ended[:, None, None]
            & self.contact_known & ~self.measured_contact & self.geometry_valid
            & ~self.distance_exhausted & ~self.time_exhausted)
        changed_gap = (self.geometry_valid & self.stop_geometry_valid
            & (self.measured_gap_m - self.stop_gap_m).abs().ge(1e-6))
        signature_known = torch.isfinite(self.geometry_signature).all(-1) & torch.isfinite(self.stop_signature).all(-1)
        changed_signature = signature_known & ((self.geometry_signature - self.stop_signature).abs().amax(-1) > 1e-6)
        changed = changed_gap | changed_signature | (self.geometry_valid & ~self.stop_geometry_valid)
        waiting = self.round_stopped & ~changed
        eligible = base & geometry_eligible & (reason == 0) & ~waiting
        start = eligible & ~self.round_active
        self.round_active |= start
        self.round_id += start.long()
        self.round_stopped &= ~start
        self.no_progress_s = torch.where(start, 0., self.no_progress_s)
        self.best_gap_m = torch.where(start, self.measured_gap_m, self.best_gap_m)
        self.state = torch.where(start, self.REACQUIRING, self.state)
        self.blocked_reason = torch.where(base, torch.where(geometry_eligible, reason, BLOCK_CODES['geometry_ineligible']), self.blocked_reason)
        self.blocked_reason = torch.where(base & waiting, BLOCK_CODES['waiting_new_measurement'], self.blocked_reason)
        self._event('reacquisition_started', start, control_tick=control_tick, observation_revision=observation_revision)
        return _owned(dict(started=start, retry_allowed=eligible,
            blocked_reason=self.blocked_reason, remaining_distance_m=(.001-self.executed_approach_m).clamp_min(0.),
            remaining_time_s=(1.-self.attempt_time_s).clamp_min(0.)))

    @torch.no_grad()
    def reserve_approach(self, *, episode_ids, control_tick, dt_s, requested_approach_m):
        self._episodes(episode_ids); self._integer(control_tick, 'control_tick'); self._dt(dt_s)
        if dt_s > 1/480 + 1e-12:
            raise ValueError('reservation dt_s cannot exceed one original physics substep')
        request = self._tensor(requested_approach_m, self.shape, 'requested_approach_m')
        if not torch.isfinite(request).all() or bool((request < 0).any()):
            raise ValueError('requested_approach_m must be finite nonnegative')
        available = (.001-self.executed_approach_m-self.reserved_approach_m).clamp_min(0.)
        time_room = (1.-self.attempt_time_s).clamp_min(0.)
        allowed = torch.minimum(request.to(torch.float64), available)
        step_room = (.001 * time_room.clamp_max(dt_s)
            - self.reserved_approach_m - self._step_approved_m).clamp_min(0.)
        allowed = torch.minimum(allowed, step_room)
        active = (self.enabled_pad_mask & self.round_active & (self.state == self.REACQUIRING)
            & self.contact_known & self.geometry_valid & (self.blocked_reason == 0)
            & ~self.episode_ended[:, None, None] & ~self.distance_exhausted & ~self.time_exhausted)
        allowed = torch.where(active, allowed, 0.)
        identity = 'reservation-' + str(self._next_reservation)
        self._next_reservation += 1
        record = dict(reservation_id=identity, episode_ids=self.episode_ids.clone(),
            physics_steps=self.last_physics_step.clone(), control_tick=control_tick,
            requested_approach_m=request.clone(), allowed_distance_m=allowed.clone(),
            settled=self.episode_ended | ~self.enabled_pad_mask.any((1,2)))
        self._reservations[identity] = record
        self.reserved_approach_m += allowed
        return _owned(record)

    @torch.no_grad()
    def cancel_reservation(self, reservation_id):
        if reservation_id not in self._reservations:
            raise ValueError('unknown reservation')
        reservation = self._reservations.pop(reservation_id)
        mask = ~reservation['settled']
        self.reserved_approach_m[mask] = (self.reserved_approach_m[mask] - reservation['allowed_distance_m'][mask]).clamp_min(0.)
        reservation['settled'][:] = True

    @torch.no_grad()
    def commit_execution(self, *, episode_ids, reservation_id, execution_id, physics_step,
            approved_approach_m, executed_mask, requested_action24, final_target_rad, approved_action24):
        """Idempotent post-guard commit, never a candidate/planner callback.

        The caller must report *commanded* approach independently of measured
        progress. Over-reservation is rejected, never clipped to hide execution.
        Both action records are canonical24; final target is the actual guarded
        twelve-joint target. This module does not infer actuator realization.
        """
        self._episodes(episode_ids); self._integer(physics_step, 'physics_step')
        if not isinstance(execution_id, str) or not execution_id:
            raise ValueError('nonempty execution_id required')
        amount = self._tensor(approved_approach_m, self.shape, 'approved_approach_m')
        self._tensor(executed_mask, self.shape, 'executed_mask', torch.bool)
        for name, value, shape in (('requested_action24', requested_action24, (self.n,24)),
                ('approved_action24', approved_action24, (self.n,24)),
                ('final_target_rad', final_target_rad, (self.n,12))):
            self._tensor(value, shape, name)
            if not torch.isfinite(value).all() or (shape[-1] == 24 and bool((value.abs() > 1).any())):
                raise ValueError(name + ' must be finite and canonical actions bounded')
        if not torch.isfinite(amount).all() or bool((amount < 0).any()) or bool(((amount > 0) & ~executed_mask).any()):
            raise ValueError('approved approach must be finite, nonnegative and actually executed')
        key = (tuple(episode_ids.tolist()), execution_id)
        payload = dict(episode_ids=episode_ids, reservation_id=reservation_id, execution_id=execution_id,
            physics_step=physics_step, approved_approach_m=amount, executed_mask=executed_mask,
            requested_action24=requested_action24, final_target_rad=final_target_rad, approved_action24=approved_action24)
        if key in self._commits:
            old = self._commits[key]
            if any(not torch.equal(old[k], v) if isinstance(v, torch.Tensor) else old[k] != v for k, v in payload.items()):
                raise ValueError('conflicting duplicate execution_id')
            return dict(duplicate=True, committed_distance_m=torch.zeros_like(self.executed_approach_m))
        if bool((self.episode_ended & executed_mask.any((1,2))).any()):
            raise ValueError('terminal episode cannot accept late execution')
        if reservation_id not in self._reservations:
            raise ValueError('unknown reservation')
        reservation = self._reservations[reservation_id]
        affected = ~reservation['settled'] | (reservation['allowed_distance_m'] > 0).any((1,2)) | executed_mask.any((1,2))
        if bool((affected & self.episode_ended).any()):
            raise ValueError('terminal episode cannot accept late execution')
        if bool((affected & (reservation['episode_ids'] != self.episode_ids)).any()):
            raise ValueError('stale reservation physical episode')
        if bool((affected & reservation['settled']).any()) or bool((amount > reservation['allowed_distance_m'] + 1e-15).any()):
            raise ValueError('execution exceeds or reuses reservation')
        if bool((affected & ((self.last_execution_step >= physics_step)
                | (reservation['physics_steps'] + 1 != physics_step)
                | (self.last_physics_step + 1 != physics_step))).any()):
            raise ValueError('stale or future execution physics_step')
        if bool((self.executed_approach_m + amount > .001 + 1e-15).any()):
            raise ValueError('execution exceeds remaining episode reservation')
        interval = (self.enabled_pad_mask & self.round_active & (self.state == self.REACQUIRING)
            & affected[:, None, None] & ~self.episode_ended[:, None, None])
        self.pending_attempt_step = torch.where(interval, physics_step, self.pending_attempt_step)
        self.pending_attempt_episode_ids = torch.where(interval, self.episode_ids[:, None, None], self.pending_attempt_episode_ids)
        self.pending_attempt_round_id = torch.where(interval, self.round_id, self.pending_attempt_round_id)
        unsettled = ~reservation['settled']
        self.reserved_approach_m[unsettled] = (self.reserved_approach_m[unsettled] - reservation['allowed_distance_m'][unsettled]).clamp_min(0.)
        reservation['settled'][:] = True
        self.executed_approach_m += amount.to(torch.float64)
        self._step_approved_m += amount.to(torch.float64)
        self.executed_approach_m.clamp_max_(.001)
        self.distance_exhausted |= self.enabled_pad_mask & (self.executed_approach_m >= .001 - 1e-15)
        self.round_active &= ~self.distance_exhausted
        self.state = torch.where(self.distance_exhausted & ~self.measured_contact, self.LOST, self.state)
        self.blocked_reason = torch.where(self.distance_exhausted, BLOCK_CODES['distance_exhausted'], self.blocked_reason)
        self.last_execution_step[affected] = physics_step
        self._commits[key] = _owned(payload)
        del self._reservations[reservation_id]
        # Per-env events let selected reset remove only that episode's evidence.
        for env in unsettled.nonzero().flatten().tolist():
            event = dict(kind='execution_committed', env=env, source_sha=self.source_shas[env])
            for name, value in payload.items():
                event[name] = value[env:env+1].clone() if isinstance(value, torch.Tensor) else value
            event['cumulative_approach_m'] = self.executed_approach_m[env:env+1].clone()
            event['cumulative_attempt_time_s'] = self.attempt_time_s[env:env+1].clone()
            event['reacquiring_interval'] = interval[env:env+1].clone()
            self.events.append(event)
        return dict(duplicate=False, committed_distance_m=amount.detach().clone())

    def snapshot(self):
        """Return owned state/events; changing the result cannot alter control."""
        names = ('episode_ids', 'last_physics_step', 'last_execution_step', 'episode_ended',
            'enabled_pad_mask', 'state', 'contact_known', 'measured_contact', 'stable',
            'stable_time_s', 'attempt_time_s', 'executed_approach_m', 'reserved_approach_m',
            'no_progress_s', 'round_id', 'round_active', 'round_stopped', 'blocked_reason',
            'pending_attempt_step', 'pending_attempt_episode_ids', 'pending_attempt_round_id',
            'distance_exhausted', 'time_exhausted', 'measured_gap_m', 'geometry_valid')
        return _owned(dict(version='tripod_measured_contact_recovery_v1',
            state_codes=STATE_CODES.copy(), blocked_reason_codes=BLOCK_CODES.copy(),
            limits=dict(speed_m_s=.001, distance_m=.001, attempt_time_s=1.,
                no_progress_s=.5, progress_resolution_m=1e-6, recovery_stable_s=self.recovery_stable_s),
            source_shas=list(self.source_shas), events=self.events,
            **{name: getattr(self, name) for name in names}))
