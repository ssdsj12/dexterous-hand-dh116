"""Fixed-anchor slip lifecycle CPU helper, without native source admission.

Caller contact booleans, public dataclasses and CPU measurements never witness
native contact or qualify a full Teacher source. A later adapter must bind real
readers. This owner retains the first bilateral references through detachment;
geometry drift is independent evidence and never replaces contact measurements.
"""
from dataclasses import dataclass, replace
import math
from typing import Optional, Tuple

from .tabletop_training_reward import Measurement, SourceFailure, SourceStamp


_GROUPS = {(side, kind) for side in ("left", "right") for kind in ("material", "tangential")}
_NA_NEVER = "no_bilateral_contact_since_reset"


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _text(value):
    return type(value) is str and bool(value.strip())


def _tick(value):
    if type(value) is not int or value < 0 or not _finite(value):
        raise SourceFailure("exact finite nonnegative integer tick required")


def _uid(value):
    if not _text(value):
        raise SourceFailure("nonempty primitive episode_uid required")


def _sequence(value, name):
    if type(value) not in (tuple, list):
        raise SourceFailure("owned tuple/list required: " + name)
    return tuple(value)


@dataclass(frozen=True, slots=True)
class FixedAnchorV2:
    side: str
    kind: str
    uid: str
    reference_tick: int
    reference: Tuple[float, ...]

    def __post_init__(self):
        if (type(self.side) is not str or type(self.kind) is not str
                or (self.side, self.kind) not in _GROUPS or not _text(self.uid)):
            raise SourceFailure("fixed left/right material/tangential identity required")
        _tick(self.reference_tick)
        reference = _sequence(self.reference, "anchor reference")
        if not reference or any(not _finite(value) for value in reference):
            raise SourceFailure("nonempty finite primitive anchor reference required")
        object.__setattr__(self, "reference", reference)


def _references(values):
    references = _sequence(values, "references")
    owned = []
    for value in references:
        if type(value) is not FixedAnchorV2:
            raise SourceFailure("exact FixedAnchorV2 required")
        # Revalidate and own a separate snapshot even if a frozen instance was
        # constructed or modified through low-level dataclass bypasses.
        owned.append(FixedAnchorV2(value.side, value.kind, value.uid,
                                   value.reference_tick, value.reference))
    if references and (len(owned) != 4 or {(a.side, a.kind) for a in owned} != _GROUPS):
        raise SourceFailure("all four distinct fixed anchor groups required")
    return tuple(owned)


def _pairs(values):
    pairs = _sequence(values, "contact pairs")
    if len(pairs) != 4:
        raise SourceFailure("all four contact drift/uncertainty pairs required")
    owned = []
    for value in pairs:
        pair = _sequence(value, "drift/uncertainty pair")
        if len(pair) != 2 or any(not _finite(x) or x < 0 for x in pair):
            raise SourceFailure("finite nonnegative contact drift/uncertainty required")
        if not _finite(pair[0] + pair[1]):
            raise SourceFailure("finite contact drift/uncertainty bounds required")
        owned.append(pair)
    return tuple(owned)


def _stamp(value):
    if (type(value) is not SourceStamp
            or any(not _text(v) for v in (value.episode_uid, value.reset_uid,
                value.rod_uid, value.scope, value.source, value.frame))
            or value.scope not in ("cpu_fixture", "unqualified_source")
            or value.frame != "world"
            or type(value.env_id) is not int or value.env_id < 0 or not _finite(value.env_id)
            or not _finite(value.solver_time_s)):
        raise SourceFailure("owned CPU/unqualified source and world clock identity required")
    _tick(value.physics_tick)


def _geometry(values, episode_uid, tick):
    owned = []
    clock = None
    for value in _sequence(values, "geometry drift"):
        if (type(value) is not Measurement or not _finite(value.raw) or value.raw < 0
                or type(value.unit) is not str or value.unit != "m"
                or type(value.valid) is not bool or not value.valid
                or type(value.applicable) is not bool or not value.applicable
                or value.history_reason is not None):
            raise SourceFailure("valid applicable finite nonnegative geometry Measurement in m required")
        _stamp(value.stamp)
        stamp = value.stamp
        if stamp.episode_uid != episode_uid or stamp.physics_tick != tick:
            raise SourceFailure("geometry drift must match observation episode and tick")
        if clock is not None and stamp.clock_identity != clock:
            raise SourceFailure("geometry drift requires one complete clock identity")
        clock = stamp.clock_identity
        # All retained leaves are primitive and immutable. Do not retain caller
        # instances whose fields could later be changed via object.__setattr__.
        copied_stamp = SourceStamp(stamp.episode_uid, stamp.physics_tick, stamp.solver_time_s,
            stamp.scope, stamp.source, stamp.frame, stamp.env_id, stamp.reset_uid, stamp.rod_uid)
        owned.append(Measurement(value.raw, value.unit, copied_stamp, True, True, None))
    return tuple(owned)


@dataclass(frozen=True, slots=True)
class SlipLifecycleV2:
    """Unqualified CPU snapshot; public construction supplies no source witness."""

    state: str
    references: Tuple[FixedAnchorV2, ...]
    contact_pairs: Optional[Tuple[Tuple[float, float], ...]]
    na_reason: Optional[str]
    geometry_drift: Tuple[object, ...] = ()

    @property
    def source_scope(self):
        return "cpu_math_fixture"

    @property
    def native_ready(self):
        return False


def _receipt_snapshot(receipt):
    """Copy validated owner state without exporting any mutable dataclass leaf.

    frozen=True does not block object.__setattr__. Every public return owns new
    anchors, Measurements and SourceStamps; only primitive immutable tuples may
    be shared with the owner's private history or with another returned view.
    """
    return replace(
        receipt,
        references=tuple(replace(anchor) for anchor in receipt.references),
        geometry_drift=tuple(replace(value, stamp=replace(value.stamp))
                             for value in receipt.geometry_drift),
    )


class SlipLifecycleOwnerV2:
    """One environment's fixed-reference history with atomic CPU observations.

    The first observation may lazily bind an episode. Later episodes require an
    explicit reset with a never-used UID. Reset establishes a clock endpoint;
    its next observation, and every later observation, advances exactly one tick.
    Only a new reset clears references; no nearest-point reanchoring is allowed.
    """

    def __init__(self):
        self._seen_episode_uids = set()
        self._episode_uid = None
        self._reset_tick = None
        self._last_tick = None
        self._current_receipt = None

    @property
    def current_receipt(self):
        if self._current_receipt is None:
            raise SourceFailure("successful observation or reset required")
        return _receipt_snapshot(self._current_receipt)

    def reset(self, *, episode_uid, tick):
        _uid(episode_uid)
        _tick(tick)
        if episode_uid in self._seen_episode_uids:
            raise SourceFailure("episode_uid reuse forbidden for slip lifecycle owner")
        receipt = SlipLifecycleV2("never_anchored", (), None, _NA_NEVER)
        self._seen_episode_uids.add(episode_uid)
        self._episode_uid = episode_uid
        self._reset_tick = self._last_tick = tick
        self._current_receipt = receipt
        return self.current_receipt

    def clear(self, *, episode_uid, tick):
        """Clear only by a new Reset UID, retaining lifetime UID history."""
        return self.reset(episode_uid=episode_uid, tick=tick)

    def observe(self, *, episode_uid, tick, contacting, references, pairs, geometry_drift=()):
        _uid(episode_uid)
        _tick(tick)
        if type(contacting) is not bool:
            raise SourceFailure("exact bool contacting required; unknown is not False")
        if self._episode_uid is not None:
            if episode_uid != self._episode_uid:
                raise SourceFailure("new episode requires explicit reset")
            if tick != self._last_tick + 1:
                raise SourceFailure("observation tick must advance exactly one")
        frozen = _references(references)
        geometry = _geometry(geometry_drift, episode_uid, tick)
        previous = () if self._current_receipt is None else self._current_receipt.references
        reset_tick = tick if self._reset_tick is None else self._reset_tick
        if previous:
            if frozen != previous:
                raise SourceFailure("fixed anchor identity changed or disappeared; reanchoring forbidden")
        elif frozen:
            if not contacting:
                raise SourceFailure("never-anchored noncontact cannot invent references")
            if any(not reset_tick <= anchor.reference_tick <= tick for anchor in frozen):
                raise SourceFailure("anchor reference tick must be within current reset history")
        if contacting:
            if not frozen:
                raise SourceFailure("current contact requires all four fixed references")
            contact_pairs = _pairs(pairs)
            state, reason = "contacting", None
        else:
            if pairs is not None:
                raise SourceFailure("noncontact requires contact_pairs None")
            contact_pairs = None
            state, reason = ("detached", "fixed_anchor_detached") if previous else ("never_anchored", _NA_NEVER)
        receipt = SlipLifecycleV2(state, frozen, contact_pairs, reason, geometry)
        # Commit only after the complete endpoint passed all schema/history
        # checks. Invalid lazy first observations must not consume their UID.
        self._seen_episode_uids.add(episode_uid)
        self._episode_uid = episode_uid
        self._reset_tick = reset_tick
        self._last_tick = tick
        self._current_receipt = receipt
        return self.current_receipt
