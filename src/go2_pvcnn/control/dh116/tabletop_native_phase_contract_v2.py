"""Fixed-S0 four-phase CPU mathematics; no native producer or admission.

Boolean samples are caller-provided CPU fixture qualifications. They do not
prove contact, stability, physical time, fixed S0, or native success. W4 must
independently witness those sources before any physical/full-profile admission.
This module owns no anchors and changes no slip, evaluator, or v1 contracts.
"""
from dataclasses import dataclass
import math


PHASES_V2 = ("Close", "PreliftHold", "Lift", "LiftHold")


def _validate_uid(value):
    if type(value) is not str or not value.strip():
        raise ValueError("nonempty primitive episode_uid required")


def _validate_tick(value):
    if type(value) is not int or value < 0:
        raise ValueError("exact nonnegative integer physics_tick required")


@dataclass(frozen=True, slots=True)
class PhaseSampleV2:
    """One CPU fixture endpoint, never proof of a native measurement source."""

    episode_uid: str
    physics_tick: int
    bilateral: bool
    prelift_stable: bool
    final_hold_eligible: bool

    def __post_init__(self):
        _validate_uid(self.episode_uid)
        _validate_tick(self.physics_tick)
        for value in (self.bilateral, self.prelift_stable, self.final_hold_eligible):
            if type(value) is not bool:
                raise ValueError("exact bool phase qualification required; unknown is not False")


@dataclass(frozen=True, slots=True)
class PhaseReceiptV2:
    """Immutable CPU-math snapshot; success_candidate never admits native success.

    Public construction or replacement of a receipt provides no owner/source
    witness. A later source/context adapter must verify its actual phase owner.
    """

    episode_uid: str
    physics_tick: int
    phase: str
    prelift_stable_s: float
    final_hold_s: float
    success_candidate: bool

    @property
    def source_scope(self):
        return "cpu_math_fixture"

    @property
    def native_ready(self):
        return False


class PhaseOwnerV2:
    """One independent environment's CPU phase state, without source admission.

    Episode UID reuse is forbidden for this owner's lifetime. The external
    run/environment/reset owner must allocate globally unique episode UIDs.
    """

    def __init__(self, *, prelift_stable_s: float):
        try:
            valid = (type(prelift_stable_s) in (int, float)
                and math.isfinite(prelift_stable_s) and prelift_stable_s > 0)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("positive finite prelift_stable_s required")
        self._prelift_required_s = float(prelift_stable_s)
        self._seen_episode_uids = set()
        self._current_receipt = None
        self._prelift_start = None
        self._final_start = None

    @property
    def current_receipt(self):
        """Return this owner's latest immutable CPU receipt after Reset."""
        if self._current_receipt is None:
            raise ValueError("phase owner reset required")
        return self._current_receipt

    def reset(self, *, episode_uid, physics_tick):
        _validate_uid(episode_uid)
        _validate_tick(physics_tick)
        if episode_uid in self._seen_episode_uids:
            raise ValueError("episode_uid reuse forbidden for phase owner")
        receipt = PhaseReceiptV2(episode_uid, physics_tick, "Close", 0., 0., False)
        self._seen_episode_uids.add(episode_uid)
        self._prelift_start = None
        self._final_start = None
        self._current_receipt = receipt
        return receipt

    def observe(self, sample: PhaseSampleV2):
        if type(sample) is not PhaseSampleV2:
            raise ValueError("exact PhaseSampleV2 required; no caller receipt/source flags")
        sample.__post_init__()
        previous = self.current_receipt
        if sample.episode_uid != previous.episode_uid:
            raise ValueError("phase sample episode_uid differs from current episode")
        if sample.physics_tick != previous.physics_tick + 1:
            raise ValueError("phase sample physics_tick must advance exactly one")
        # Validate the entire endpoint before calculating or committing state.
        phase = previous.phase
        prelift_start = self._prelift_start
        final_start = self._final_start
        prelift_s = previous.prelift_stable_s
        final_s = 0.
        candidate = False
        tick = sample.physics_tick

        if not sample.bilateral:
            phase = "Close"
            prelift_start = final_start = None
            prelift_s = 0.
        elif phase in ("Close", "PreliftHold"):
            phase = "PreliftHold"
            final_start = None
            if sample.prelift_stable:
                if prelift_start is None:
                    prelift_start = tick
                prelift_s = (tick - prelift_start) / 480
                if prelift_s >= self._prelift_required_s:
                    phase = "Lift"
            else:
                prelift_start = None
                prelift_s = 0.
            # This endpoint belongs to PreliftHold, including the transition.
            # A caller's final=True cannot skip ahead or reuse this clock.
        elif sample.final_hold_eligible:
            phase = "LiftHold"
            if final_start is None:
                final_start = tick
            intervals = tick - final_start
            final_s = intervals / 480
            candidate = intervals >= 2400
        else:
            phase = "Lift"
            final_start = None

        receipt = PhaseReceiptV2(
            sample.episode_uid, tick, phase, prelift_s, final_s, candidate)
        self._prelift_start = prelift_start
        self._final_start = final_start
        self._current_receipt = receipt
        return receipt
