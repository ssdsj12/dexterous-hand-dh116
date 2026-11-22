"""Contact-only context over sealed Reset and actually recorded physical events.

No scene, App, actuator, model, solver or native qualification is created here.
The first measured contact boundary follows Reset by a separately charged prime.
"""
from copy import deepcopy
import math

from .tabletop_native_teacher_context import (
    TaskContextOwner, ContextIdentity, CONTEXT_FIELDS, CONTACT_PRELIFT,
)
from .tabletop_native_teacher_source_priming import PrimedNativeActionHistory
from .tabletop_native_teacher_contact_measurement import capture_supported_contact
from .tabletop_deformable_r0_replay import _digest


class PrimedContactContextOwner(TaskContextOwner):
    """Own contact durations; reuse the existing pre/post/bootstrap transaction."""

    def __init__(self, *, backend_profile, scope, episode_cap_s, external_cap_s, grasp):
        if grasp not in ("pinch", "tripod"):
            raise ValueError("independent two/three-pad contact context required; Power needs its own palm source")
        super().__init__(backend_profile=backend_profile, mode=CONTACT_PRELIFT, scope=scope,
                         episode_cap_s=episode_cap_s, external_cap_s=external_cap_s, grasp=grasp)
        self._contact_failed = False
        self._pending_reset = False
        self._contact_identity = self._contact = self._contact_event_sha = None
        self._contact_start = None
        self._contact_s = 0.
        self._observed = 0

    def _check_contact(self):
        if self._closed or self._contact_failed:
            raise RuntimeError("closed or failed contact context; no continuation or backfill")

    @property
    def current(self):
        self._check_contact()
        return super().current

    @property
    def pending_post(self):
        self._check_contact()
        return super().pending_post

    @property
    def contact_readback(self):
        self._check_contact()
        if self._contact is None:
            raise ValueError("measured contact required after priming")
        return dict(deepcopy(self._contact), duration_s=self._contact_s,
                    physical_observations=self._observed, event_sha256=self._contact_event_sha)

    def _history(self, owner):
        self._check_contact()
        if type(owner) is not PrimedNativeActionHistory:
            raise ValueError("actual PrimedNativeActionHistory required")
        owner._check()
        if self.scope != "cpu_fixture" and owner._backend != self.profile.binary_sha256:
            raise ValueError("history backend profile differs")
        return owner

    def reset_from_verified_state(self, *args, **kwargs):
        raise ValueError("explicit pending Reset then measured priming required")

    def _remaining(self, state, identity):
        external = state.get("external_budget")
        if (not isinstance(external,dict) or external.get("source") != "external_budget_clock"
                or external.get("identity") != identity or external.get("cap_s") != self.external_cap_s
                or type(external.get("remaining_s")) not in (int,float)
                or not math.isfinite(external["remaining_s"])
                or not 0 <= external["remaining_s"] <= self.external_cap_s):
            raise ValueError("same-clock finite external remaining budget required")
        return external["remaining_s"]

    def reset_pending_from_verified_state(self, state, *, history_owner):
        self._check_contact()
        try:
            history = self._history(history_owner)
            if (self._pending_reset or self._current is not None or self._contact_identity is not None
                    or history._priming is not None or history._decision is not None
                    or history._captured is not None or history._sequence.controls != 0):
                raise ValueError("fresh unprimed Reset required for pending contact context")
            identity = self._state(state)
            remaining = self._remaining(state, identity)
            self._freeze_reset_references(state, history_owner=history)
            self._contact_identity = identity
            self._reset_remaining = remaining
            self._pending_reset = True
        except BaseException:
            self._contact_failed = True
            raise

    def _consume_event(self, event, history, *, priming):
        wanted = "priming_physical" if priming else "physical"
        if type(event) is not dict or event.get("kind") != wanted:
            raise ValueError("original separately recorded physical event required")
        event_sha = _digest(event)
        identity = ContextIdentity.from_clock(history._sequence.clock)
        if priming:
            receipt = history.priming
            expected_sha = receipt["priming_row_sha256"]
            if (ContextIdentity.from_clock(receipt["reset_identity"]) != self._reset_identity
                    or ContextIdentity.from_clock(receipt["primed_identity"]) != identity):
                raise ValueError("actual priming receipt and original Reset identities differ")
        else:
            if not history._row_hashes or history._decision is None or history._settled is not None:
                raise ValueError("original history must record physical row during pending policy first")
            expected_sha = history._row_hashes[-1]
        prior = self._contact_identity
        if (event_sha != expected_sha or identity.epoch != prior.epoch
                or identity.physics_step != prior.physics_step + 1
                or identity.rod_tick != prior.rod_tick + 12
                or not math.isclose(identity.timestamp_s-prior.timestamp_s,1/480,rel_tol=0,abs_tol=2e-7)):
            raise ValueError("same owned event and consecutive physical contact clocks required")
        rod = event["exchange"]["rod"]
        if any(rod.get(k) != getattr(identity,k) for k in ContextIdentity.__dataclass_fields__):
            raise ValueError("raw rod and accepted history clocks differ")
        contact = capture_supported_contact(rod,grasp=self.grasp)
        if contact["bilateral_support"]:
            if self._contact_start is None:
                self._contact_start = identity.physics_step
            self._contact_s = (identity.physics_step-self._contact_start)/480
        else:
            self._contact_start = None
            self._contact_s = 0.
        self._contact = contact
        self._contact_identity = identity
        self._contact_event_sha = event_sha
        self._observed += 1

    def commit_primed_state(self, state, *, history_owner, priming_event):
        self._check_contact()
        try:
            history = self._history(history_owner)
            identity = self._state(state)
            if (not self._pending_reset or self._current is not None
                    or history._owner_uid != self._history_uid or history._decision is not None
                    or history._captured is not None or history._sequence.controls != 0
                    or self._remaining(state,identity) > self._reset_remaining + 1e-7
                    or state.get("target_reference_uid") != dict(self._references)["target_material"]
                    or state.get("rest_length_m") != self._rest
                    or state.get("distance_mean_m") != self._fixed[3]):
                raise ValueError("unchanged pending Reset reference and monotonic priming budget required")
            self._consume_event(priming_event,history,priming=True)
            if identity != self._contact_identity:
                raise ValueError("primed context state differs from measured source identity")
            packet = self._build(identity,history._previous,self._contact_event_sha,state)
            self._current = packet
            self._pending_reset = False
            return self.current
        except BaseException:
            self._contact_failed = True
            raise

    def observe_physical_event(self, event, *, history_owner):
        self._check_contact()
        try:
            history = self._history(history_owner)
            if (self._pending_reset or self._current is None or self._pre is None
                    or self._pending is not None or history._owner_uid != self._history_uid):
                raise ValueError("contact observation requires the current frozen physical policy window")
            self._consume_event(event,history,priming=False)
            return self.contact_readback
        except BaseException:
            self._contact_failed = True
            raise

    def _build(self, identity, previous, token, state, **kwargs):
        self._check_contact()
        if ("physical_fields" in state or "phase" in state or "phase_source" in state
                or self._contact is None or self._contact_identity != identity):
            raise ValueError("contact fields/Close phase must come from this owned measured source")
        fields = {}
        for index, value in ((4,self._contact["coverage_left"]),(5,self._contact["coverage_right"]),
                             (6,self._contact_s)):
            fields[index] = dict(value=value,source=CONTEXT_FIELDS[index]["source"],identity=identity,valid=True)
        references = kwargs.pop("references", self._references)
        references = tuple(references) + (("contact_event",self._contact_event_sha),)
        return super()._build(identity,previous,token,dict(state,physical_fields=fields),
                              references=references,**kwargs)

    def freeze_pre_action(self, identity, *, history_owner):
        self._check_contact()
        if identity != self._contact_identity:
            raise ValueError("last observed contact must match the frozen pre-action state")
        packet = super().freeze_pre_action(identity,history_owner=history_owner)
        self._observed_before = self._observed
        return packet

    def advance_physical_endpoint(self, state, *, history_owner, collection_token,
                                  anchor_owner=None, phase_owner=None):
        self._check_contact()
        try:
            identity = self._state(state)
            if (self._pre is None or anchor_owner is not None or phase_owner is not None
                    or identity != self._contact_identity
                    or self._observed-self._observed_before !=
                       identity.physics_step-self._pre.metadata.identities[0].physics_step):
                raise ValueError("every physical contact endpoint required; anchors/full phase excluded")
            return super().advance_physical_endpoint(state,history_owner=history_owner,
                                                      collection_token=collection_token)
        except BaseException:
            self._contact_failed = True
            raise

    def commit_post_context(self, packet):
        self._check_contact()
        return super().commit_post_context(packet)
