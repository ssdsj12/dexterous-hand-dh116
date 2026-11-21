"""Fixed-wrist force decision consumes only the preceding native evidence."""

import numpy as np
import pytest

from go2_pvcnn.control.dh116 import tabletop_pinch_force_hold_feedback as feedback


def window(*, end=362, index=.13, signed=True, right_sum=.35, identity=("run","ep",1)):
    return [dict(physics_step=tick, policy_step=(end-3)//8, substep=i, guard_failure=None,
        rod=dict(run_uuid=identity[0], episode_uid=identity[1], reset_id=identity[2], rod_uid="rod0",
                 physics_step=tick, rod_tick=(tick-2)*12, contact_valid=True),
        stage_reaction=dict(physics_step=tick, rod_tick=(tick-2)*12,
            evidence_scope="measured_simulated_reaction", per_substep=[
                dict(substep=j,
                    left=dict(thumb_load_n=.15,index_load_n=index,signed_opposition=signed,
                              opposition_cos=-.5 if signed else None),
                    right=dict(thumb_load_n=.17,index_load_n=right_sum-.17,signed_opposition=True,
                               opposition_cos=-.5)) for j in range(12)]))
        for i,tick in enumerate(range(end-7,end+1))]


def test_uses_only_left_index_and_bounds_total_preload():
    control=feedback.FixedWristPinchForceHold(362)
    proof=lambda delta, row: True
    for end in [362,370]:
        decision=control.decide(window(end=end), certify=proof)
        assert decision["reasons"]==["left_required_load_deficit"]
        assert decision["left_index_delta_rad"]==pytest.approx(.00005)
    clipped=control.decide(window(end=378), certify=proof)
    assert clipped["reasons"]==["preload_limit_reached"] and clipped["left_index_delta_rad"]==0


@pytest.mark.parametrize("fields,reason", [
    ({"signed":False},"opposing_contact_not_continuous"),
    ({"index":.17},"left_load_adequate"),
    ({"index":.049},"required_pad_load_missing"),
    ({"right_sum":.24},"right_required_load_missing"),
])
def test_zero_correction_for_missing_opposition_or_wrong_deficit(fields, reason):
    control=feedback.FixedWristPinchForceHold(362)
    decision=control.decide(window(**fields),certify=lambda *args: pytest.fail("must hold"))
    assert decision["left_index_delta_rad"]==0 and decision["reasons"]==[reason]


def test_failed_interval_proof_holds_and_does_not_spend_budget():
    control=feedback.FixedWristPinchForceHold(362)
    assert control.decide(window(),certify=lambda _,rc:False)["reasons"]==["collision_proof_rejected"]
    assert control.decide(window(end=370),certify=lambda _,rc:True)["left_index_delta_rad"]>0


@pytest.mark.parametrize("fault",["clock","identity","num_substeps","nonfinite"])
def test_rejects_stale_or_incomplete_solver_packets(fault):
    evidence=window()
    if fault=="clock":
        evidence[-1]["stage_reaction"]["physics_step"]-=1
    elif fault=="identity":
        evidence[-1]["rod"]["reset_id"]=2
    elif fault=="num_substeps":
        evidence[-1]["stage_reaction"]["per_substep"].pop()
    else:
        evidence[-1]["stage_reaction"]["per_substep"][0]["left"]["thumb_load_n"]=np.nan
    with pytest.raises(ValueError):
        feedback.FixedWristPinchForceHold(362).decide(evidence,certify=lambda *args:True)


def test_action_bridge_keeps_wrist_right_hand_and_other_fingers_fixed(monkeypatch):
    from scripts import run_dh116_pinch_fixed_force_feedback as pilot

    monkeypatch.setattr(pilot.prelift, "joint_and_wrist",
        lambda row, side: (np.full(12,.1), np.full(12,.1)))
    def certify(row, states, targets):
        assert row["physics_step"] == 362
        expected = np.zeros(12)
        expected[2] = .00005
        assert np.allclose(targets["left"]-states["left"][1], expected)
        assert np.array_equal(targets["right"], states["right"][1])
        return {"left":{"proved":True},"right":{"proved":True}}
    actor = pilot.FeedbackAction(certify=certify)
    original = np.ones(24)/10.
    prefix, diag = actor(pilot.seating.PREFIX_WINDOWS-1, original, [])
    assert np.array_equal(prefix, original) and diag is None
    action, diag = actor(pilot.seating.PREFIX_WINDOWS, np.zeros(24), window())
    assert np.flatnonzero(action).tolist() == [2]
    assert action[2] * pilot.prelift.WINDOW_SCALE[2] == pytest.approx(diag["left_index_delta_rad"])
