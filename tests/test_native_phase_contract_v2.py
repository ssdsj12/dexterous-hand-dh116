"""Fixed-S0 phase v2 CPU mathematics, never native/source admission."""


def test_reset_is_close_old_phase_schema_unchanged():
    from go2_pvcnn.control.dh116.tabletop_native_phase_contract_v2 import PhaseOwnerV2
    from go2_pvcnn.control.dh116.tabletop_native_teacher_context import CONTEXT_FIELDS

    owner = PhaseOwnerV2(prelift_stable_s=0.25)
    receipt = owner.reset(episode_uid="fixture:reset-schema", physics_tick=2)
    assert (receipt.phase, receipt.prelift_stable_s, receipt.final_hold_s) == ("Close", 0.0, 0.0)
    assert receipt.success_candidate is False
    assert [field["name"] for field in CONTEXT_FIELDS[:4]] == [
        "phase_one_hot.Approach", "phase_one_hot.Close",
        "phase_one_hot.Lift", "phase_one_hot.Hold",
    ]

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest


def phase_module():
    from go2_pvcnn.control.dh116 import tabletop_native_phase_contract_v2
    return tabletop_native_phase_contract_v2


def fresh_owner(uid, tick=2, threshold=.25):
    module = phase_module()
    owner = module.PhaseOwnerV2(prelift_stable_s=threshold)
    owner.reset(episode_uid=uid, physics_tick=tick)
    return owner


def step(owner, tick, bilateral=True, stable=True, final=False):
    module = phase_module()
    return owner.observe(module.PhaseSampleV2(
        owner.current_receipt.episode_uid, tick, bilateral, stable, final))


@pytest.mark.parametrize("value", [True, False, None, "0.25", 0, -1, float("nan"),
    float("inf"), -float("inf"), complex(.25), 10**400])
def test_prelift_parameter_requires_positive_finite_number(value):
    with pytest.raises(ValueError):
        phase_module().PhaseOwnerV2(prelift_stable_s=value)


@pytest.mark.parametrize("value", ["", "   ", None, True, 9])
def test_reset_requires_nonempty_primitive_uid(value):
    owner = fresh_owner("fixture:invalid-reset-uid")
    before = owner.current_receipt
    with pytest.raises(ValueError):
        owner.reset(episode_uid=value, physics_tick=0)
    assert owner.current_receipt is before


@pytest.mark.parametrize("tick", [True, False, -1, 2.0, None, float("nan"), "2"])
def test_reset_requires_exact_nonnegative_integer_tick(tick):
    owner = fresh_owner("fixture:invalid-reset-tick")
    before = owner.current_receipt
    with pytest.raises(ValueError):
        owner.reset(episode_uid="fixture:unused-after-invalid-reset", physics_tick=tick)
    assert owner.current_receipt is before
    assert owner.reset(episode_uid="fixture:unused-after-invalid-reset", physics_tick=0).phase == "Close"


def test_new_episode_reset_clears_timers_and_rejects_reuse():
    owner = fresh_owner("fixture:reset-1")
    step(owner, 3)
    receipt = owner.reset(episode_uid="fixture:reset-2", physics_tick=258)
    assert (receipt.physics_tick, receipt.phase, receipt.prelift_stable_s,
        receipt.final_hold_s, receipt.success_candidate) == (258, "Close", 0., 0., False)
    for reused in ("fixture:reset-1", "fixture:reset-2"):
        with pytest.raises(ValueError):
            owner.reset(episode_uid=reused, physics_tick=0)
        assert owner.current_receipt is receipt


def test_observe_and_current_receipt_require_reset():
    module = phase_module()
    owner = module.PhaseOwnerV2(prelift_stable_s=.25)
    with pytest.raises(ValueError):
        _ = owner.current_receipt
    with pytest.raises(ValueError):
        owner.observe(module.PhaseSampleV2("fixture:unreset", 0, True, True, False))
    assert owner.reset(episode_uid="fixture:unreset", physics_tick=2).physics_tick == 2


@pytest.mark.parametrize("tick", [True, -1, 2.0, 2, 1, 4, None, float("nan")])
def test_wrong_episode_or_skipped_tick_rejected(tick):
    module = phase_module()
    owner = fresh_owner("fixture:clock")
    before = owner.current_receipt
    with pytest.raises(ValueError):
        owner.observe(module.PhaseSampleV2("fixture:clock", tick, True, True, False))
    assert owner.current_receipt is before
    with pytest.raises(ValueError):
        owner.observe(module.PhaseSampleV2("fixture:other-clock", 3, True, True, False))
    assert owner.current_receipt is before
    assert step(owner, 3).physics_tick == 3


@pytest.mark.parametrize("field", ["bilateral", "prelift_stable", "final_hold_eligible"])
@pytest.mark.parametrize("value", [None, 0, 1, "true", float("nan"), []])
def test_unknown_bool_rejected_without_state_change(field, value):
    module = phase_module()
    owner = fresh_owner("fixture:bool")
    before = owner.current_receipt
    # Check observe itself, even for a sample bypassing dataclass construction.
    sample = module.PhaseSampleV2("fixture:bool", 3, True, True, False)
    object.__setattr__(sample, field, value)
    with pytest.raises(ValueError):
        owner.observe(sample)
    assert owner.current_receipt is before
    assert step(owner, 3).physics_tick == 3


def test_only_exact_typed_samples_are_accepted_no_source_flags():
    module = phase_module()
    owner = fresh_owner("fixture:sample-type")
    before = owner.current_receipt
    fields = dict(episode_uid="fixture:sample-type", physics_tick=3,
        bilateral=True, prelift_stable=True, final_hold_eligible=True)
    for sample in (fields, SimpleNamespace(**fields), replace(before, physics_tick=3), True, None):
        with pytest.raises(ValueError):
            owner.observe(sample)
        assert owner.current_receipt is before
    with pytest.raises(TypeError):
        module.PhaseSampleV2(**fields, source_scope="actual")
    with pytest.raises(TypeError):
        module.PhaseSampleV2(**fields, native_ready=True)
    assert step(owner, 3).physics_tick == 3


def test_sample_and_receipt_are_immutable_owned_snapshots():
    module = phase_module()
    assert module.PHASES_V2 == ("Close", "PreliftHold", "Lift", "LiftHold")
    owner = fresh_owner("fixture:immutable")
    before = owner.current_receipt
    sample = module.PhaseSampleV2("fixture:immutable", 3, True, True, False)
    for item, field, value in ((before, "phase", "Lift"), (sample, "bilateral", False)):
        with pytest.raises(FrozenInstanceError):
            setattr(item, field, value)
    after = owner.observe(sample)
    assert before is not after
    assert before.physics_tick == 2
    assert after is owner.current_receipt
    with pytest.raises(AttributeError):
        owner.current_receipt = before


def test_different_owner_episode_state_is_isolated():
    first = fresh_owner("fixture:environment-1")
    second = fresh_owner("fixture:environment-2", tick=258, threshold=.5)
    before = second.current_receipt
    receipt = step(first, 3)
    assert second.current_receipt is before
    with pytest.raises(ValueError):
        second.observe(phase_module().PhaseSampleV2(receipt.episode_uid, 259, True, True, False))
    assert second.current_receipt is before
    assert step(second, 259).physics_tick == 259


def test_receipt_scope_is_always_cpu_math_and_not_native_ready():
    owner = fresh_owner("fixture:scope")
    receipt = owner.current_receipt
    assert receipt.source_scope == "cpu_math_fixture"
    assert receipt.native_ready is False
    for field, value in (("source_scope", "actual"), ("native_ready", True)):
        with pytest.raises((AttributeError, TypeError)):
            setattr(receipt, field, value)
    with pytest.raises(TypeError):
        replace(receipt, native_ready=True)


def enter_lift(owner, start_tick=3):
    assert step(owner, start_tick).phase == "PreliftHold"
    for tick in range(start_tick + 1, start_tick + 120):
        assert step(owner, tick).phase == "PreliftHold"
    receipt = step(owner, start_tick + 120)
    assert receipt.phase == "Lift"
    return receipt


def test_prelift_does_not_count_toward_final_hold():
    owner = fresh_owner("fixture:two-clocks")
    first = step(owner, 3)
    assert (first.phase, first.prelift_stable_s, first.final_hold_s) == ("PreliftHold", 0., 0.)
    for tick in range(4, 123):
        receipt = step(owner, tick)
        assert receipt.phase == "PreliftHold"
        assert receipt.final_hold_s == 0.
        assert receipt.success_candidate is False
    assert receipt.prelift_stable_s == 119/480
    lift = step(owner, 123)
    assert (lift.phase, lift.prelift_stable_s, lift.final_hold_s) == ("Lift", .25, 0.)
    first_final = step(owner, 124, final=True)
    assert (first_final.phase, first_final.final_hold_s) == ("LiftHold", 0.)
    for tick in range(125, 2524):
        receipt = step(owner, tick, final=True)
        assert receipt.success_candidate is False
    assert receipt.final_hold_s == 2399/480
    candidate = step(owner, 2524, final=True)
    assert (candidate.final_hold_s, candidate.success_candidate) == (5., True)
    assert candidate.native_ready is False
    assert candidate.source_scope == "cpu_math_fixture"
    # Candidate is a mathematical condition, not automatic physical termination.
    continuing = step(owner, 2525, final=True)
    assert continuing.phase == "LiftHold"
    assert continuing.final_hold_s == 2401/480
    assert continuing.success_candidate is True
    interrupted = step(owner, 2526)
    assert (interrupted.phase, interrupted.final_hold_s, interrupted.success_candidate) == ("Lift", 0., False)
    lost = step(owner, 2527, bilateral=False, stable=False)
    assert (lost.phase, lost.prelift_stable_s, lost.final_hold_s) == ("Close", 0., 0.)


def test_final_true_cannot_skip_prelift_or_enter_hold_on_lift_transition():
    owner = fresh_owner("fixture:no-skip")
    for tick in range(3, 123):
        receipt = step(owner, tick, final=True)
        assert receipt.phase == "PreliftHold"
        assert receipt.final_hold_s == 0.
        assert receipt.success_candidate is False
    receipt = step(owner, 123, final=True)
    assert receipt.phase == "Lift"
    assert receipt.final_hold_s == 0.
    receipt = step(owner, 124, final=True)
    assert (receipt.phase, receipt.final_hold_s) == ("LiftHold", 0.)


def test_close_requires_bilateral_even_when_other_qualifications_true():
    owner = fresh_owner("fixture:close-gate")
    for tick in range(3, 130):
        receipt = step(owner, tick, bilateral=False, stable=True, final=True)
        assert (receipt.phase, receipt.prelift_stable_s, receipt.final_hold_s,
            receipt.success_candidate) == ("Close", 0., 0., False)
    receipt = step(owner, 130, stable=False, final=True)
    assert (receipt.phase, receipt.prelift_stable_s, receipt.final_hold_s) == ("PreliftHold", 0., 0.)
    receipt = step(owner, 131)
    assert (receipt.phase, receipt.prelift_stable_s) == ("PreliftHold", 0.)


def test_prelift_interruption_restarts_window():
    owner = fresh_owner("fixture:prelift-interruption")
    for tick in range(3, 63):
        step(owner, tick)
    interrupted = step(owner, 63, stable=False, final=True)
    assert (interrupted.phase, interrupted.prelift_stable_s, interrupted.final_hold_s) == ("PreliftHold", 0., 0.)
    assert step(owner, 64).prelift_stable_s == 0.
    for tick in range(65, 184):
        receipt = step(owner, tick)
        assert receipt.phase == "PreliftHold"
    assert receipt.prelift_stable_s == 119/480
    assert step(owner, 184).phase == "Lift"


def test_final_interruption_clears_hold():
    owner = fresh_owner("fixture:final-interruption")
    enter_lift(owner)
    for tick in range(124, 2524):
        step(owner, tick, final=True)
    interrupted = step(owner, 2524, final=False)
    assert (interrupted.phase, interrupted.final_hold_s, interrupted.success_candidate) == ("Lift", 0., False)
    restarted = step(owner, 2525, stable=False, final=True)
    assert (restarted.phase, restarted.final_hold_s) == ("LiftHold", 0.)
    for tick in range(2526, 4925):
        receipt = step(owner, tick, stable=False, final=True)
        assert receipt.success_candidate is False
    assert receipt.final_hold_s == 2399/480
    assert step(owner, 4925, stable=False, final=True).success_candidate is True


@pytest.mark.parametrize("stop_tick", [3, 124, 2524])
def test_bilateral_loss_clears_both_clocks_in_every_nonclose_phase(stop_tick):
    owner = fresh_owner("fixture:bilateral-loss")
    for tick in range(3, stop_tick + 1):
        step(owner, tick, final=True)
    lost = step(owner, stop_tick + 1, bilateral=False, stable=True, final=True)
    assert (lost.phase, lost.prelift_stable_s, lost.final_hold_s,
        lost.success_candidate) == ("Close", 0., 0., False)
    restarted = step(owner, stop_tick + 2, final=True)
    assert (restarted.phase, restarted.prelift_stable_s, restarted.final_hold_s) == ("PreliftHold", 0., 0.)


def test_reset_after_candidate_uses_new_uid_and_independent_tick_origin():
    owner = fresh_owner("fixture:multi-episode-1")
    enter_lift(owner)
    for tick in range(124, 2525):
        receipt = step(owner, tick, final=True)
    assert receipt.success_candidate is True
    reset = owner.reset(episode_uid="fixture:multi-episode-2", physics_tick=3000)
    assert (reset.phase, reset.prelift_stable_s, reset.final_hold_s,
        reset.success_candidate) == ("Close", 0., 0., False)
    assert step(owner, 3001).prelift_stable_s == 0.
    for tick in range(3002, 3121):
        assert step(owner, tick).phase == "PreliftHold"
    assert step(owner, 3121).phase == "Lift"


def test_configured_threshold_is_independent_per_owner():
    first = fresh_owner("fixture:threshold-quarter", threshold=.25)
    second = fresh_owner("fixture:threshold-half", threshold=.5)
    for tick in range(3, 124):
        first_receipt = step(first, tick)
        second_receipt = step(second, tick)
    assert first_receipt.phase == "Lift"
    assert second_receipt.phase == "PreliftHold"
    for tick in range(124, 243):
        assert step(second, tick).phase == "PreliftHold"
    assert step(second, 243).phase == "Lift"


def test_invalid_endpoint_during_hold_is_atomic_and_does_not_reset_timer():
    module = phase_module()
    owner = fresh_owner("fixture:atomic-hold")
    enter_lift(owner)
    step(owner, 124, final=True)
    before = step(owner, 125, final=True)
    for sample in (
        module.PhaseSampleV2("fixture:atomic-hold", 127, False, False, False),
        module.PhaseSampleV2("fixture:wrong-episode", 126, False, False, False),
    ):
        with pytest.raises(ValueError):
            owner.observe(sample)
        assert owner.current_receipt is before
    bad = module.PhaseSampleV2("fixture:atomic-hold", 126, True, True, True)
    object.__setattr__(bad, "final_hold_eligible", None)
    with pytest.raises(ValueError):
        owner.observe(bad)
    assert owner.current_receipt is before
    after = step(owner, 126, final=True)
    assert after.final_hold_s == 2/480
