"""Independent contact diagnostics around the original conserved recovery ledger.

No observer/controller is installed. Inputs declare actual object-filtered
measurements; source/clock receipts are checked, not Native-attested. Geometry
only labels candidates. The original ledger owns all distance/time/watchdog,
reservation and post-submission execution accounting. No target is written.
"""
from dataclasses import asdict, dataclass
import math

import torch

from .tabletop_bimanual_mpc_state import SampleClock
from .tabletop_tripod_contact_recovery import TripodContactRecovery, _owned


VERSION = 'dh116-bimanual-measured-contact-runtime-v1'
STATE_NAMES = ('NO_CONTACT', 'CONTACT_CANDIDATE', 'VALID_CONTACT', 'STABLE', 'SLIPPING', 'LOST', 'REACQUIRING')
QUALIFICATION = dict(native_attested=False, hardware_measurement_qualified=False,
    planning_eligible=False, dynamics_qualified=False, bilateral_hold_qualified=False,
    candidate_future_dynamics_qualified=False, force_sign_frame_qualified=False, support_qualified=False,
    native_success_qualification_issued=False, new_expert_qualification_issued=False,
    Student_success_qualified=False)
_STYLES = {'pinch_thumb_index': 2, 'tripod_thumb_index_middle': 3}
_DT = 1 / 480


def _sha(value):
    if (not isinstance(value, str) or len(value) != 64 or
            any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('explicit source SHA256 required')
    return value


@dataclass(frozen=True, slots=True)
class SlipThreshold:
    value: float
    unit: str
    source: str
    source_sha: str

    def __post_init__(self):
        if (type(self.value) not in (int, float) or not math.isfinite(self.value) or self.value <= 0
                or self.unit != 'm/s' or not isinstance(self.source, str) or not self.source.strip()):
            raise ValueError('explicit positive slip threshold, m/s and source required')
        _sha(self.source_sha)


class BimanualContactRuntime:
    """Per-digit labels, independent both-hand masks, unchanged ledger limits.

    STABLE means only the old 20ms recovery/pressure gate. Unknown slip or
    opposition stays NA; neither STABLE nor these labels establish bilateral
    200ms stability, original task hold, force-side or hardware qualification.
    """

    def __init__(self, n, device, *, grasp_styles, slip_threshold=None):
        if type(n) is not int or n < 1:
            raise ValueError('positive integer environment count required')
        styles = tuple(grasp_styles)
        if len(styles) != n or any(s not in _STYLES for s in styles):
            raise ValueError('one explicit supported grasp style per environment required')
        if slip_threshold is not None and not isinstance(slip_threshold, SlipThreshold):
            raise ValueError('explicit SlipThreshold or None required; no default zero threshold')
        self.n, self.device = n, torch.device(device)
        self._grasp_styles, self._slip_threshold = styles, slip_threshold
        self.shape = (n, 2, 5)
        self._required = torch.zeros(self.shape, dtype=torch.bool, device=self.device)
        for e, style in enumerate(styles):
            self._required[e, :, :_STYLES[style]] = True
        self._ledger = TripodContactRecovery(n, self.device, enabled_pad_mask=self._required)
        self._last = [None] * n; self._observed = [False] * n
        self._used_uids = [set() for _ in range(n)]; self._run_uuid = None
        self._ever = torch.zeros_like(self._required)
        self._candidate = torch.zeros_like(self._required)
        self._geometry_known = torch.zeros_like(self._required)
        self._slip_known = torch.zeros_like(self._required)
        self._slipping = torch.zeros_like(self._required)
        self._slip_speed = torch.full(self.shape, float('nan'), dtype=torch.float64, device=self.device)
        self._slip_source = [None] * n
        self._opposition = [None] * n
        self._events = []; self._terminals = {}

    @property
    def required_pad_mask(self):
        return self._required.clone()

    @property
    def grasp_styles(self):
        return self._grasp_styles

    @property
    def slip_threshold(self):
        return self._slip_threshold

    def _tensor(self, value, shape, name, dtype=None):
        return self._ledger._tensor(value, shape, name, dtype)

    def _clocks(self, clocks):
        clocks = tuple(clocks)
        if len(clocks) != self.n or any(not isinstance(c, SampleClock) or c.env_id != i
                                       for i, c in enumerate(clocks)):
            raise ValueError('one exact env-associated SampleClock per slot required')
        return clocks

    def _sources(self, source_sha):
        values = (source_sha,) * self.n if isinstance(source_sha, str) else tuple(source_sha)
        if len(values) != self.n:
            raise ValueError('one source SHA per environment required')
        return tuple(_sha(s) for s in values)

    @staticmethod
    def _physical_frame(clock):
        return (clock.run_uuid, clock.world_step, clock.sdk_step,
                clock.world_time_s, clock.sdk_time_s, clock.policy_step)

    @staticmethod
    def _delta(previous, current, *, reset=False):
        delta = current.world_step - previous.world_step
        if (delta < 0 if reset else delta != 1) or current.sdk_step - previous.sdk_step != delta:
            raise ValueError('repeated/missing/misaligned World/SDK physical clock')
        if any(not math.isclose(getattr(current, k) - getattr(previous, k), delta * _DT,
                                abs_tol=1e-7, rel_tol=0) for k in ('world_time_s', 'sdk_time_s')):
            raise ValueError('actual 480Hz World/SDK time alignment changed')
        if (current.policy_step < previous.policy_step or
                not reset and current.policy_step > previous.policy_step + 1):
            raise ValueError('policy clock moved backwards or skipped a control interval')

    def reset(self, ids, *, clocks, source_sha):
        """Only selected, explicit newer physical resets replenish the old ledger."""
        ids = torch.as_tensor(ids, device=self.device)
        if ids.dtype == torch.bool:
            self._tensor(ids, (self.n,), 'ids', torch.bool); ids = ids.nonzero().flatten()
        if (ids.ndim != 1 or ids.dtype != torch.long or len(ids.unique()) != len(ids)
                or bool(((ids < 0) | (ids >= self.n)).any())):
            raise ValueError('unique selected int64 reset rows required')
        if not len(ids):
            return self.snapshot()
        clocks = self._clocks(clocks); _sha(source_sha)
        chosen = ids.tolist(); runs = {clocks[e].run_uuid for e in chosen}
        if len(runs) != 1 or self._run_uuid is not None and runs != {self._run_uuid}:
            raise ValueError('run identity changed; construct a new contact runtime')
        if len({self._physical_frame(clocks[e]) for e in chosen}) != 1:
            raise ValueError('selected reset rows require one shared actual physical frame')
        episodes = self._ledger.episode_ids.clone()
        for e in chosen:
            c = clocks[e]
            if (not c.reset or c.terminal or c.episode_uid in self._used_uids[e]
                    or c.reset_generation <= int(episodes[e])):
                raise ValueError('actual newer physical reset/unused episode UID required')
            if self._last[e] is not None:
                self._delta(self._last[e], c, reset=True)
            episodes[e] = c.reset_generation
        # All new-layer validation precedes the single original reset call.
        self._ledger.reset(ids, episode_ids=episodes, source_sha=source_sha)
        self._run_uuid = next(iter(runs))
        self._events = [v for v in self._events if v['env_id'] not in chosen]
        for e in chosen:
            self._last[e] = clocks[e]; self._observed[e] = False
            self._used_uids[e].add(clocks[e].episode_uid)
            for value in (self._ever, self._candidate, self._geometry_known, self._slip_known, self._slipping):
                value[e] = False
            self._slip_speed[e] = float('nan'); self._slip_source[e] = None; self._opposition[e] = None
            self._terminals.pop(e, None)
            self._events.append(dict(kind='actual_episode_reset', env_id=e, clock=clocks[e], source_sha=source_sha))
        return self.snapshot()

    def _observation_context(self, clocks, sources, active):
        if bool((self._ledger.episode_ids < 0).any()):
            raise ValueError('initialize every physical episode before observation')
        selected = active.nonzero().flatten().tolist()
        frame = {self._physical_frame(clocks[e]) for e in selected}
        if len(frame) != 1:
            raise ValueError('one shared actual physical frame required')
        for e in selected:
            c, previous = clocks[e], self._last[e]
            if (c.run_uuid != self._run_uuid or c.reset or c.episode_uid != previous.episode_uid
                    or c.reset_generation != int(self._ledger.episode_ids[e]) or sources[e] != self._ledger.source_shas[e]):
                raise ValueError('actual run/episode/reset/source identity differs')
            self._delta(previous, c)

    def _capture_events(self, offset, clocks):
        for event in self._ledger.events[offset:]:
            e = event['env']
            self._events.append(dict(kind=event['kind'], env_id=e, clock=clocks[e],
                source_sha=self._ledger.source_shas[e], ledger_event=_owned(event)))

    def observe_substep(self, *, clocks, source_sha, live, measurement_basis,
            measurement_valid, pad_valid, loads_n, vectors_w, depth_m, rejected_loads_n,
            measured_gap_m, geometry_valid, geometry_alignment_cos, sensor_healthy=None,
            measured_geometry_signature=None, slip_speed_m_s=None, slip_valid=None, slip_source=None,
            opposition_confirmed=None, opposition_valid=None, opposition_source=None, opposition_basis=None):
        self._tensor(live, (self.n,), 'live', torch.bool)
        active = live & ~self._ledger.episode_ended
        if not bool(active.any()):
            if bool((~self._ledger.episode_ended).any()):
                clocks = self._clocks(clocks)
                if any(c.terminal and not bool(self._ledger.episode_ended[e]) for e, c in enumerate(clocks)):
                    raise ValueError('first terminal requires an actual live sample')
            return self.snapshot()
        clocks = self._clocks(clocks); sources = self._sources(source_sha)
        if any(c.terminal and not bool(live[e]) and not bool(self._ledger.episode_ended[e])
               for e, c in enumerate(clocks)):
            raise ValueError('first terminal requires an actual live sample')
        self._observation_context(clocks, sources, active)
        if measurement_basis != 'actual_object_filtered_source_patch':
            raise ValueError('actual object-filtered measured contact basis required; geometry is candidate only')
        self._tensor(geometry_alignment_cos, self.shape, 'geometry_alignment_cos')
        have_slip = any(v is not None for v in (slip_speed_m_s, slip_valid, slip_source))
        if have_slip:
            if self.slip_threshold is None:
                raise ValueError('explicit slip threshold required, never implicit zero')
            self._tensor(slip_speed_m_s, self.shape, 'slip_speed_m_s')
            self._tensor(slip_valid, self.shape, 'slip_valid', torch.bool)
            if not isinstance(slip_source, str) or not slip_source.strip() or 'geometry_only' in slip_source:
                raise ValueError('actual measured slip source required')
        have_opposition = any(v is not None for v in (opposition_confirmed, opposition_valid, opposition_source, opposition_basis))
        if have_opposition:
            self._tensor(opposition_confirmed, (self.n,), 'opposition_confirmed', torch.bool)
            self._tensor(opposition_valid, (self.n,), 'opposition_valid', torch.bool)
            if (opposition_basis != 'actual_contact_observation' or not isinstance(opposition_source, str)
                    or not opposition_source.strip()):
                raise ValueError('opposition requires explicit actual-contact provenance')
        old_contact = self._ledger.measured_contact.clone()
        recovering = (self._ledger.round_active & (self._ledger.state == self._ledger.REACQUIRING)
                      & self._required).any(-1)
        terminal = torch.tensor([c.terminal for c in clocks], device=self.device, dtype=torch.bool)
        offset = len(self._ledger.events)
        self._ledger.observe_substep(episode_ids=self._ledger.episode_ids.clone(),
            physics_step=clocks[active.nonzero()[0].item()].world_step, dt_s=_DT, live=live, terminal=terminal,
            measurement_valid=measurement_valid, pad_valid=pad_valid, loads_n=loads_n, vectors_w=vectors_w,
            depth_m=depth_m, rejected_loads_n=rejected_loads_n, measured_gap_m=measured_gap_m,
            geometry_valid=geometry_valid, sensor_healthy=sensor_healthy,
            measured_geometry_signature=measured_geometry_signature)
        self._capture_events(offset, clocks)
        for e in active.nonzero().flatten().tolist():
            self._last[e] = clocks[e]; self._observed[e] = True
            self._ever[e] |= self._ledger.measured_contact[e]
            geom = (geometry_valid[e] & torch.isfinite(measured_gap_m[e]) & (measured_gap_m[e] >= 0)
                    & torch.isfinite(geometry_alignment_cos[e]) & (geometry_alignment_cos[e].abs() <= 1))
            self._geometry_known[e] = geom & self._required[e]
            self._candidate[e] = self._geometry_known[e] & (measured_gap_m[e] <= .001) & (geometry_alignment_cos[e] >= .8)
            if have_slip:
                self._slip_speed[e] = slip_speed_m_s[e]
                self._slip_known[e] = (slip_valid[e] & torch.isfinite(slip_speed_m_s[e]) & (slip_speed_m_s[e] >= 0)
                    & self._ledger.measured_contact[e] & self._ledger.contact_known[e] & self._required[e])
                self._slipping[e] = self._slip_known[e] & (slip_speed_m_s[e] > self.slip_threshold.value)
                self._slip_source[e] = slip_source
            else:
                self._slip_speed[e] = float('nan'); self._slip_known[e] = False
                self._slipping[e] = False; self._slip_source[e] = None
            self._opposition[e] = (dict(declared_valid=bool(opposition_valid[e]),
                declared_confirmed=bool(opposition_confirmed[e]), source=opposition_source)
                if have_opposition else None)
            lost = old_contact[e] & self._ledger.contact_known[e] & ~self._ledger.measured_contact[e] & self._required[e]
            for recovering_hand in range(2):
                other = 1 - recovering_hand
                if recovering[e, recovering_hand] and bool(lost[other].any()):
                    self._events.append(dict(kind='concurrent_coordination_failure', env_id=e, clock=clocks[e],
                        source_sha=sources[e], recovering_hand=('left', 'right')[recovering_hand],
                        lost_hand=('left', 'right')[other], lost_digits=lost[other].nonzero().flatten().tolist(),
                        causal_evidence=False))
        for e in active.nonzero().flatten().tolist():
            if clocks[e].terminal and e not in self._terminals:
                self._terminals[e] = self._row(e)
        return self.snapshot()

    def _context(self, clocks, source_sha):
        clocks = self._clocks(clocks); sources = self._sources(source_sha)
        open_rows = (~self._ledger.episode_ended).nonzero().flatten().tolist()
        if not open_rows:
            raise ValueError('terminal episodes cannot authorize recovery operations')
        for e in open_rows:
            if not self._observed[e] or clocks[e] != self._last[e] or sources[e] != self._ledger.source_shas[e]:
                raise ValueError('recovery operation requires current actual observation clock/source')
        return clocks, max(clocks[e].policy_step for e in open_rows), max(clocks[e].world_step for e in open_rows)

    def request_retry(self, *, clocks, source_sha, geometry_eligible, blocked_reason='none'):
        clocks, tick, revision = self._context(clocks, source_sha); offset = len(self._ledger.events)
        result = self._ledger.request_retry(episode_ids=self._ledger.episode_ids.clone(), control_tick=tick,
            observation_revision=revision, geometry_eligible=geometry_eligible, blocked_reason=blocked_reason)
        self._capture_events(offset, clocks)
        return result

    def reserve_approach(self, *, clocks, source_sha, requested_approach_m, dt_s=_DT):
        _, tick, _ = self._context(clocks, source_sha)
        return self._ledger.reserve_approach(episode_ids=self._ledger.episode_ids.clone(),
            control_tick=tick, dt_s=dt_s, requested_approach_m=requested_approach_m)

    def cancel_reservation(self, *, clocks, source_sha, reservation_id):
        self._context(clocks, source_sha)
        return self._ledger.cancel_reservation(reservation_id)

    def commit_execution(self, *, clocks, source_sha, **execution):
        """Caller reports post-SDK-submission guarded commands, never a target write."""
        clocks, _, _ = self._context(clocks, source_sha); offset = len(self._ledger.events)
        result = self._ledger.commit_execution(episode_ids=self._ledger.episode_ids.clone(), **execution)
        self._capture_events(offset, clocks)
        return result

    def _row(self, e):
        known = self._ledger.contact_known[e] & self._required[e]
        names = []
        for h in range(2):
            hand = []
            for d in range(5):
                if not known[h, d]:
                    name = None
                elif self._ledger.measured_contact[e, h, d]:
                    name = 'SLIPPING' if self._slipping[e, h, d] else 'STABLE' if self._ledger.stable[e, h, d] else 'VALID_CONTACT'
                elif self._ledger.state[e, h, d] == self._ledger.REACQUIRING:
                    name = 'REACQUIRING'
                else:
                    name = 'LOST' if self._ever[e, h, d] else 'CONTACT_CANDIDATE' if self._candidate[e, h, d] else 'NO_CONTACT'
                hand.append(name)
            names.append(tuple(hand))
        required_known = (known | ~self._required[e]).all(-1)
        required_contact = (self._ledger.measured_contact[e] | ~self._required[e]).all(-1) & required_known
        receipt = self._opposition[e]
        opposition_valid = bool(receipt and receipt['declared_valid'] and required_contact.all())
        return _owned(dict(clock=self._last[e], source_sha=self._ledger.source_shas[e], grasp_style=self.grasp_styles[e],
            states=tuple(names), valid=known, required_pad_mask=self._required[e],
            required_contact_known=required_known, required_contact_present=required_contact,
            geometry_candidate=self._candidate[e], geometry_known=self._geometry_known[e],
            slip=dict(threshold=None if self.slip_threshold is None else asdict(self.slip_threshold),
                speed_m_s=self._slip_speed[e], classification_valid=self._slip_known[e], source=self._slip_source[e]),
            opposition=dict(valid=opposition_valid, value=receipt['declared_confirmed'] if opposition_valid else None,
                source=None if receipt is None else receipt['source'], input_receipt=receipt,
                missing_reason=None if opposition_valid else 'actual opposition source or both required-hand contacts unavailable'),
            qualification=dict(QUALIFICATION)))

    def snapshot(self):
        return _owned(dict(version=VERSION, rows=tuple(self._row(e) for e in range(self.n)),
            terminal_samples=self._terminals, ledger=self._ledger.snapshot(), events=self._events,
            state_names=STATE_NAMES, qualification=dict(QUALIFICATION),
            stability_scope='original_20ms_recovery_pressure_only_not_bilateral_200ms_or_task_hold',
            input_scope='caller-declared actual filtered-contact observations; source receipts are not Native attestation',
            execution_scope='original ledger command-distance accounting only; wrapper performs no SDK submission or target write'))
