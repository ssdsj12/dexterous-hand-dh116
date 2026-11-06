"""Opt-in privileged diagnostics from actual pre-reset physical substeps.

This observer never advances a success tracker, changes a task, or reads hardware
torque. Implicit-actuator applied/computed torques are simulator estimates.
Synthetic callers of this class do not constitute native evaluation evidence.
"""
import torch

SLIP_REASON_CODES = dict(valid=0, no_anchor=1, no_contact=2,
                         measurement_invalid=3, anchor_invalid=4, inactive=5)


class EvaluationTelemetry:
    """Batch callback and last-live-row snapshot, all tensors detached/cloned.

    Packets have scalar ``dt_s`` and bool [N] ``live``, ``done`` (new terminal),
    ``native_success``, ``measurement_valid``, ``lift_valid``,
    ``qualified_hold_valid``, ``slip_valid``, ``contact_valid``, ``action_valid``.
    ``prior_anchor_latched`` and ``slip_applicable`` are bool [N], independent
    of slip validity. ``slip_reason_code`` is int64 [N], decoded by the immutable
    values in ``slip_reason_codes``. A known anchor with corrupt raw measurement
    or anchor remains applicable and invalid; only no anchor, trustworthy lost
    opposition, or inactivity is legitimate not-applicable data.
    ``lift_m``, ``qualified_hold_s``, ``slip_m`` are float64 [N], with NaN for
    unobservable/invalid values. ``bilateral_opposed_contact`` is bool [N].
    ``action`` is the submitted [N,A] action (A=24 for surface vision policies).
    Torque fields are float64 [N,12], left then right active six joints:
    ``estimated_torque_nm``, optional ``computed_torque_nm``, original
    ``effort_limit_nm``, bool ``torque_valid`` for the applied estimate, and
    optional bool ``computed_torque_valid``. Unavailable torque data is
    None, never inferred from action bounds. ``torque_source`` names the estimate.

    References start at the next physical reset. Snapshot freezes inactive rows
    after their terminal sample, even while their reset physics keeps evolving.
    """
    def __init__(self, n, device, callback):
        if not callable(callback):
            raise TypeError('evaluation telemetry callback must be callable')
        self.n, self.device, self.callback = n, device, callback
        self.reference_bottom_m = torch.full((n,), float('nan'),
                                             dtype=torch.float64, device=device)
        self._last = None

    @torch.no_grad()
    def reset(self, ids, measurement):
        """Record only selected measured physical-reset baselines."""
        bottom = measurement['bottom'][ids].to(self.reference_bottom_m)
        self.reference_bottom_m[ids] = torch.where(torch.isfinite(bottom), bottom, float('nan'))

    @staticmethod
    def _tracker(task):
        tracker = getattr(task, 'surface_slip_tracking', None)
        return tracker if tracker is not None else getattr(task, 'surface_lift_success', None)

    @torch.no_grad()
    def capture_anchor(self, task):
        """Clone before consume can release/reanchor the native fixed window."""
        tracker = self._tracker(task)
        if tracker is None:
            return None
        if tracker.reward_diagnostics:
            # The contact diagnostic anchor remains observable at speeds that
            # disqualify hold. Capture it before consume can reanchor on slip.
            return tracker._slip_anchor.detach().clone(), tracker._slip_latched.detach().clone()
        return tracker._anchor.detach().clone(), tracker._latched.detach().clone()

    def _float(self):
        return torch.full((self.n,), float('nan'), dtype=torch.float64, device=self.device)

    def _mask(self):
        return torch.zeros(self.n, dtype=torch.bool, device=self.device)

    def _finite(self, value):
        return torch.isfinite(value).reshape(self.n, -1).all(-1)

    def _contact(self, m):
        from .tabletop_surface_force_sharing import surface_targets
        names = ('effective_pad_loads_n', 'effective_pad_vectors_w', 'effective_pad_valid')
        if not all(name in m for name in names):
            return self._mask(), self._mask()
        loads, vectors, valid = (m[name] for name in names)
        measured = self._finite(loads) & self._finite(vectors) & (loads >= 0).flatten(1).all(-1)
        # Helpers reject invalid full batches. Isolate bad rows without claiming
        # those replacement zeroes are measured contacts.
        clean_loads = torch.where(measured[:, None, None], loads, 0.)
        clean_vectors = torch.where(measured[:, None, None, None], vectors, 0.)
        paired = surface_targets(clean_loads, clean_vectors, valid=valid)['paired'].all(-1)
        return paired & measured, measured

    def _slip(self, m, measured, contact, contact_valid, live, anchor):
        from .tabletop_multifinger_monitor import rod_points_in_hands
        drift, valid = self._float(), self._mask()
        latched = self._mask() if anchor is None else anchor[1]
        reason = torch.full((self.n,), SLIP_REASON_CODES['no_anchor'],
                            device=self.device, dtype=torch.long)
        reason = torch.where(live, reason, SLIP_REASON_CODES['inactive'])
        applicable = self._mask()
        if anchor is None:
            return drift, valid, latched, applicable, reason
        fixed, _ = anchor
        known = live & latched
        anchor_valid = self._finite(fixed)
        names = ('object_relative', 'object_quat', 'wrist_pose')
        if not all(name in m for name in names):
            reason = torch.where(known, SLIP_REASON_CODES['measurement_invalid'], reason)
            reason = torch.where(known & ~anchor_valid, SLIP_REASON_CODES['anchor_invalid'], reason)
            return drift, valid, latched, known, reason
        wrist, quat = m['wrist_pose'], m['object_quat']
        center = m['object_relative'] + wrist[..., :3].mean(1)
        object_pose = torch.cat((center, quat), -1)
        observable = (measured & contact_valid & self._finite(object_pose) & self._finite(wrist)
                      & (quat.abs().amax(-1) > 0.)
                      & (wrist[..., 3:].abs().amax(-1) > 0.).all(-1))
        # Establish scope from the prior native latch and raw contact/liveness,
        # never by negating slip_valid. Corruption cannot become a legal NA.
        applicable = known & (contact | ~observable | ~anchor_valid)
        reason = torch.where(known, SLIP_REASON_CODES['valid'], reason)
        reason = torch.where(known & ~contact, SLIP_REASON_CODES['no_contact'], reason)
        reason = torch.where(known & ~observable, SLIP_REASON_CODES['measurement_invalid'], reason)
        reason = torch.where(known & ~anchor_valid, SLIP_REASON_CODES['anchor_invalid'], reason)
        valid = known & observable & contact & anchor_valid
        clean_object = torch.where(valid[:, None], object_pose, 0.).clone()
        clean_wrist = torch.where(valid[:, None, None], wrist, 0.).clone()
        clean_object[~valid, 3] = 1.
        clean_wrist[~valid, :, 3] = 1.
        points = rod_points_in_hands(clean_object, clean_wrist)
        raw = (points - fixed).norm(dim=-1).amax((1, 2)).double()
        reason = torch.where(valid & ~torch.isfinite(raw), SLIP_REASON_CODES['measurement_invalid'], reason)
        valid &= torch.isfinite(raw)
        return torch.where(valid, raw, drift), valid, latched, applicable, reason

    def _torque(self, world):
        hands = getattr(world, 'hands', None)
        ids = getattr(getattr(world, 'w', None), 'ids', None)
        limit = getattr(getattr(world, 'parameters', None), 'effort_limit_nm', None)
        if hands is None or ids is None or limit is None:
            return None, None, None, None
        applied = [getattr(hand.data, 'applied_torque', None) for hand in hands]
        if len(applied) != 2 or any(value is None for value in applied):
            return None, None, None, None
        applied = torch.cat([value[:, joint_ids] for value, joint_ids in zip(applied, ids)], -1).double()
        if applied.shape != (self.n, 12):
            raise ValueError('evaluation torque requires two hands with six active joints each')
        computed = [getattr(hand.data, 'computed_torque', None) for hand in hands]
        computed = (None if any(value is None for value in computed) else
                    torch.cat([value[:, joint_ids] for value, joint_ids in zip(computed, ids)], -1).double())
        limits = torch.full_like(applied, float(limit))
        valid = torch.isfinite(applied) & torch.isfinite(limits) & (limits > 0.)
        return applied, computed, limits, valid

    @torch.no_grad()
    def sample(self, task, world, measurement, live, dt, submitted_action, *,
               measurement_valid, anchor=None):
        """Emit once after consume, before any world/task reset of this sample."""
        m = measurement
        measured = measurement_valid.clone()
        # Optional contact sentinels are handled independently below. These raw
        # quantities must never silently reuse task.m's last finite observation.
        for name in ('bottom', 'object_relative', 'object_quat', 'wrist_pose', 'object_velocity'):
            if name in m:
                measured &= self._finite(m[name])
        if 'object_quat' in m:
            measured &= m['object_quat'].abs().amax(-1) > 0.
        if 'wrist_pose' in m:
            measured &= (m['wrist_pose'][..., 3:].abs().amax(-1) > 0.).all(-1)
        lift = m['bottom'].double() - self.reference_bottom_m
        lift_valid = measured & torch.isfinite(lift)
        tracker = getattr(task, 'surface_lift_success', None)
        hold = (tracker.duration if tracker is not None else task.success_time).double()
        hold_valid = measured & torch.isfinite(hold) & (hold >= 0.)
        contact, contact_valid = self._contact(m)
        contact_valid &= measured
        contact &= contact_valid
        slip, slip_valid, prior_latched, slip_applicable, slip_reason = self._slip(
            m, measured, contact, contact_valid, live, anchor)
        applied, computed, limits, torque_valid = self._torque(world)
        action = torch.as_tensor(submitted_action, device=self.device)
        packet = dict(dt_s=float(dt), live=live, done=live & task.done,
            native_success=live & task.success, measurement_valid=measured,
            lift_m=torch.where(lift_valid, lift, float('nan')), lift_valid=lift_valid,
            qualified_hold_s=torch.where(hold_valid, hold, float('nan')), qualified_hold_valid=hold_valid,
            slip_m=slip, slip_valid=slip_valid, bilateral_opposed_contact=contact,
            prior_anchor_latched=prior_latched, slip_applicable=slip_applicable,
            slip_reason_code=slip_reason, slip_reason_codes=SLIP_REASON_CODES.copy(),
            contact_valid=contact_valid, estimated_torque_nm=applied, computed_torque_nm=computed,
            effort_limit_nm=limits, torque_valid=torque_valid,
            computed_torque_valid=None if computed is None else torch.isfinite(computed),
            torque_source='implicit_actuator_estimate_not_hardware_measurement' if applied is not None else None,
            action=action, action_valid=self._finite(action))
        packet = self._clone(packet)
        if self._last is None:
            self._last = {name: (torch.full_like(value, float('nan')) if value.is_floating_point()
                                else torch.zeros_like(value)) if isinstance(value, torch.Tensor)
                          else value for name, value in packet.items()}
            self._last['slip_reason_code'].fill_(SLIP_REASON_CODES['inactive'])
        for name, value in packet.items():
            if isinstance(value, torch.Tensor):
                previous = self._last[name]
                if previous is None:
                    previous = (torch.full_like(value, float('nan')) if value.is_floating_point()
                                else torch.zeros_like(value))
                mask = live.reshape((self.n,) + (1,) * (value.ndim - 1))
                self._last[name] = torch.where(mask, value, previous)
            else:
                self._last[name] = value.copy() if isinstance(value, dict) else value
        self.callback(packet)
        return packet

    @staticmethod
    def _clone(packet):
        return {name: (value.detach().clone() if isinstance(value, torch.Tensor) else
                       value.copy() if isinstance(value, dict) else value)
                for name, value in packet.items()}

    @torch.no_grad()
    def snapshot(self):
        return None if self._last is None else self._clone(self._last)
