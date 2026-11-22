"""Versioned contact-to-lift reward over the existing PPO settlement."""
from dataclasses import replace
from types import MappingProxyType

from .tabletop_deformable_r0_replay import _digest
from .tabletop_native_teacher_profile import digest
from .tabletop_supported_lift_observation import FixedLiftReference
from .tabletop_teacher_lift_progress import SupportedLiftProgress, lift_reward_delta
from .tabletop_training_reward import CONTACT, SCHEMA_SHA256, SourceFailure, TrainingReward, _term


PROFILE = "supported_lift_diagnostic_v2"
SCHEMA = digest({
    "version": PROFILE,
    "base_schema": SCHEMA_SHA256,
    "phase": "measured_palm_and_rod_prelift_v2",
    "lift": {"weight": 4., "target_m": .03, "table_and_ends_unloaded": True},
    "hold_success_event": False,
})


class SupportedLiftDiagnosticReward(TrainingReward):
    """Add measured Lift potential to the contact-only settlement engine.

    The diagnostic uses the original action/GAE transaction. Its isolated
    reward identity cannot label a legacy checkpoint or qualify native Hold.
    """

    def __init__(self, config, *, contact_owner, reference, priming_event, prelift_stable_s):
        if config.profile != CONTACT:
            raise ValueError("loaded contact reward base required")
        super().__init__(config, contact_owner=contact_owner)
        reset = contact_owner._reset_identity
        if (type(reference) is not FixedLiftReference or reference.epoch != reset.epoch
                or reference.physics_step != reset.physics_step
                or (reference.median_z_m, reference.minimum_z_m) != tuple(contact_owner._fixed[:2])):
            raise SourceFailure("Lift must use the same fixed measured Reset as contact")
        self._progress_owner = SupportedLiftProgress(
            reference, grasp=contact_owner.grasp, prelift_stable_s=prelift_stable_s)
        self._diagnostic_sha = digest({"base": self.config.config_sha256, "schema": SCHEMA,
                                      "prelift_stable_s": prelift_stable_s})
        self._observe_lift(priming_event, "priming_physical")

    def _observe_lift(self, event, kind):
        _, contact = self._contact_source()
        if event.get("kind") != kind or _digest(event) != contact["event_sha256"]:
            raise SourceFailure("same recorded contact event required for Lift progression")
        try:
            lifted = self._progress_owner.observe(event["exchange"])
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise SourceFailure("measured Lift progression source: " + str(error)) from error
        if (lifted.observation.raw_rod_sha256 != contact["raw_rod_sha256"]
                or lifted.observation.bilateral != contact["bilateral_support"]):
            raise SourceFailure("Lift and contact must use the same measured scope")
        self._lift = lifted

    def observe_recorded_event(self, event, endpoint):
        """Consume the original verified physical event before reward settlement."""
        if self.quarantined or self._request is None or self._terminal or len(self._costs) >= 8:
            raise RuntimeError("active physical interval required for Lift progression")
        try:
            self._observe_lift(event, "physical")
            return super().observe(endpoint)
        except SourceFailure as error:
            self._fail(str(error), endpoint)

    def _potential(self, row, values):
        base, terms = super()._potential(row, values)
        lift = self._lift
        if (lift.phase.episode_uid != row.stamp.episode_uid
                or lift.phase.physics_tick != row.stamp.physics_tick):
            raise SourceFailure("Lift progression and reward endpoint identity differ")
        height = lift.observation.off_support_lift_m
        terms = dict(terms)
        terms["lift"] = _term(height, "m", "measured_supported_rod_fixed_reset",
            lift.lift_potential, applicable=height is not None,
            history_reason="off_support_lift_unavailable" if height is None else None)
        return base + lift.lift_potential, MappingProxyType(terms)

    def _make_context(self, row):
        base = dict(super()._make_context(row))
        base.update(config_sha256=self._diagnostic_sha, schema_sha256=SCHEMA,
                    lift_phase=self._lift.phase.phase, prelift_stable_s=self._lift.phase.prelift_stable_s,
                    lift_reset_rod_sha256=self._lift.observation.reset_rod_sha256)
        return MappingProxyType(base)

    def begin(self, submitted_request, *, remaining_substeps):
        super().begin(submitted_request, remaining_substeps=remaining_substeps)
        self._pre_lift = self._lift

    def settle_once(self, token, *, truncated=False, buffer_full=False):
        packet = super().settle_once(token, truncated=truncated, buffer_full=buffer_full)
        delta = lift_reward_delta(self._pre_lift, self._lift, gamma=packet.gamma,
                                  terminated=packet.terminated)
        context = MappingProxyType(dict(packet.post_context, lift_shaping_delta=delta))
        return replace(packet, profile=PROFILE, profile_version=2, post_context=context,
                       config_sha256=self._diagnostic_sha, schema_sha256=SCHEMA)
