"""Do not require a nonexistent hold measurement before any bilateral grasp."""
from copy import deepcopy
import numpy as np
import pytest
from test_dh116_deformable_control_step import inputs
from test_dh116_deformable_grasp_task import snapshot, signals, measurement
from test_dh116_r0_reward_domain import no_bilateral
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import RESET_HOOKS
from go2_pvcnn.control.dh116.tabletop_r0_hold_necessary_domain import (
    NecessaryHoldR0Episode, NecessaryHoldR0ControlStep,
)

def make(grasp="pinch"):
    t, b, w = inputs()
    e = NecessaryHoldR0Episode(t, grasp, env_id=0, diagnostic_coefficients=w)
    e.reset({k: lambda: None for k in RESET_HOOKS},
            lambda: (snapshot(t,2), measurement(.005)))
    return e, t, b, w

@pytest.mark.parametrize("grasp", ["pinch","tripod","power"])
def test_verified_no_bilateral_contact_makes_hold_contribution_inapplicable(grasp):
    e,t,_,_ = make(grasp)
    row=e.step(snapshot(t,3),no_bilateral(3,stable=None))
    term=row["terms"]["hold_progress"]
    assert term["raw"] is None and term["valid"] is False
    assert term["applicable"] is False and term["contribution"] == 0
    assert term["applicability_reason"] == "known_missing_bilateral_hold_precondition"
    assert row["terms"]["instability"]["raw"] is None
    assert "instability" in row["blocked_terms"] and "hold_progress" not in row["blocked_terms"]
    assert row["hold_s"] == 0 and row["success_candidate"] is False
    assert row["training_reward"] is None and row["native_qualification"] is False

def test_unknown_contact_cannot_be_replaced_with_known_no_contact():
    e,t,_,_=make()
    m=no_bilateral(3,stable=None)
    m["right"]=measurement(None)
    row=e.step(snapshot(t,3),m)
    assert row["terms"]["hold_progress"]["contribution"] is None
    assert "hold_progress" in row["blocked_terms"]

def test_bilateral_contact_requires_real_stability_and_slip_sources():
    e,t,_,_=make()
    m=signals(t,step=3)
    m["stable"]=measurement(None)
    row=e.step(snapshot(t,3),m)
    assert row["terms"]["hold_progress"]["contribution"] is None
    assert row["terms"]["hold_progress"]["applicable"] is True

def test_contact_loss_does_not_erase_previous_slip_requirement():
    e,t,_,_=make()
    e.step(snapshot(t,3),signals(t,step=3))
    row=e.step(snapshot(t,4),no_bilateral(4,stable=None))
    assert row["terms"]["hold_progress"]["contribution"] == 0
    assert row["terms"]["material_anchor_slip"]["contribution"] is None
    assert row["bilateral_contact_seen"]

def test_no_new_rewards_or_success_thresholds():
    e,t,_,_=make()
    assert (e.lift_target_m,e.hold_target_s,e.slip_limit_m)==(.03,5.,.0005)
    assert all(x is None for x in t["reward"]["coefficients"].values())
    with pytest.raises(ValueError):
        bad=deepcopy(t);bad["success"]["hold_s"]=.5
        NecessaryHoldR0Episode(bad,"pinch",env_id=0,diagnostic_coefficients={})

def test_control_reuses_original_terminal_and_does_not_zero_fill_raw():
    _,t,b,w=make()
    r=NecessaryHoldR0ControlStep(t,"pinch",env_id=0,budget_config=b,diagnostic_coefficients=w)
    r.reset({k:lambda:None for k in RESET_HOOKS},
            lambda:(snapshot(t,2),measurement(.005)))
    r.begin(control_index=0,recipe_sha256=r.recipe_sha256,action=np.zeros(24))
    for step in range(3,11):
        r.push(snapshot(t,step),no_bilateral(step,stable=None),
               control_index=0,recipe_sha256=r.recipe_sha256)
    out=r.settle()
    term=out["terms"]["hold_progress"]
    assert term["raw_sum"] is None and term["valid_samples"]==0
    assert term["contribution_sum"]==0 and term["not_applicable_samples"]==8
    assert out["training_reward"] is None and "instability" in out["blocked_terms"]

def test_known_instability_is_still_an_immediate_terminal():
    _,t,b,w=make()
    r=NecessaryHoldR0ControlStep(t,"pinch",env_id=0,budget_config=b,diagnostic_coefficients=w)
    r.reset({k:lambda:None for k in RESET_HOOKS},
            lambda:(snapshot(t,2),measurement(.005)))
    r.begin(control_index=0,recipe_sha256=r.recipe_sha256,action=np.zeros(24))
    assert r.push(snapshot(t,3),no_bilateral(3,stable=False),
        control_index=0,recipe_sha256=r.recipe_sha256)
    out=r.settle()
    assert out["terminal_reasons"]==["instability"] and out["physical_samples"]==1
