"""Diagnostic fixtures only; production R0 coefficients and native gates stay closed."""
from copy import deepcopy
import math
import numpy as np
import pytest
from go2_pvcnn.control.dh116.tabletop_deformable_r0_applicability import (
    ApplicableR0Episode, ApplicableR0ControlStep, VERSION,
)
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import RESET_HOOKS
from test_dh116_deformable_control_step import inputs
from test_dh116_deformable_grasp_task import snapshot, signals, measurement
from test_dh116_r0_reward_domain import no_bilateral

def reset_measurement(step=2, *, uid="ep-0", reset_id=0, value=(.02,.04)):
    return dict(sample_identity=dict(env_id=0,episode_uid=uid,reset_id=reset_id,
        rod_uid="rod0",physics_step=step,rod_tick=step*12,timestamp_s=step/480),
        approach_distance_m=measurement(list(value),"cpu_fixture:measured_reset_geometry"))

def make(grasp="pinch", reset_distance=True, coefficients=True):
    task,budget,weights=inputs()
    e=ApplicableR0Episode(task,grasp,env_id=0,
        diagnostic_coefficients=weights if coefficients else None)
    e.reset({n:lambda:None for n in RESET_HOOKS},
        lambda:(snapshot(task,2),measurement(.005)),
        reset_approach=reset_measurement() if reset_distance else None)
    return e,task,weights

@pytest.mark.parametrize("grasp",["pinch","tripod","power"])
def test_pregrasp_defined_contribution_does_not_fabricate_slip_measurement(grasp):
    e,t,w=make(grasp)
    row=e.step(snapshot(t,3),no_bilateral(3,stable=True))
    slip=row["terms"]["material_anchor_slip"]
    assert slip["raw"] is None and slip["valid"] is False
    assert slip["applicable"] is False and slip["contribution"]==0
    assert slip["contribution_valid"] is True
    assert row["terms"]["approach_progress"]["raw"]==pytest.approx(.02)
    assert row["diagnostic_reward"]==pytest.approx(.02*w["approach_progress"]-w["external_support_violation"])
    assert row["diagnostic_reward_without_applicability"] is None
    assert row["material_anchor_slip_m"] is None
    assert row["training_reward"] is None and not row["native_qualification"]
    assert row["version"]==VERSION
    assert row["term_stats"]["material_anchor_slip"]["validity_count"]==0
    assert all(x is None for x in t["reward"]["coefficients"].values())

@pytest.mark.parametrize("side",["left","right"])
def test_unknown_contact_cannot_disable_slip(side):
    e,t,_=make()
    m=no_bilateral(3,stable=True);m[side]=measurement(None)
    row=e.step(snapshot(t,3),m)
    assert row["terms"]["material_anchor_slip"]["applicable"] is None
    assert row["terms"]["material_anchor_slip"]["contribution"] is None
    assert row["diagnostic_reward"] is None

@pytest.mark.parametrize("anchor_available",[False,True])
def test_bilateral_contact_latches_slip_requirement_even_if_anchor_never_available(anchor_available):
    e,t,_=make()
    m=signals(t,step=3)
    if not anchor_available:
        for s in ("left","right"):
            for k in ("material_reference_uid","material_s_m","material_uncertainty_m",
                      "tangential_reference_uid","tangential_slip_m","tangential_uncertainty_m"):
                m[s]["value"][k]=None
    row=e.step(snapshot(t,3),m)
    assert row["bilateral_contact_seen"] is True
    for step in (4,5,6):
        lost=e.step(snapshot(t,step),no_bilateral(step,stable=True))
        assert lost["terms"]["material_anchor_slip"]["applicable"] is True
        assert lost["terms"]["material_anchor_slip"]["raw"] is None
        assert lost["diagnostic_reward"] is None

def test_fresh_reset_clears_contact_latch_but_cannot_reuse_episode():
    e,t,_=make();e.step(snapshot(t,3),signals(t,step=3))
    e.reset({n:lambda:None for n in RESET_HOOKS},
        lambda:(snapshot(t,2,uid="ep-1",reset_id=1),measurement(.005)),
        reset_approach=reset_measurement(uid="ep-1",reset_id=1))
    m=no_bilateral(3,stable=True);m["sample_identity"].update(episode_uid="ep-1",reset_id=1)
    row=e.step(snapshot(t,3,uid="ep-1",reset_id=1),m)
    assert row["terms"]["material_anchor_slip"]["applicable"] is False
    with pytest.raises(ValueError,match="fresh"):
        e.reset({n:lambda:None for n in RESET_HOOKS},
            lambda:(snapshot(t,2,uid="ep-1",reset_id=1),measurement(.005)))

@pytest.mark.parametrize("change",[
    lambda r:r["sample_identity"].update(physics_step=3),
    lambda r:r["sample_identity"].update(timestamp_s=.9),
    lambda r:r["sample_identity"].update(episode_uid="foreign"),
    lambda r:r["approach_distance_m"].update(source=None),
    lambda r:r["approach_distance_m"].update(value=[float("nan"),.1]),
    lambda r:r.update(extra="forbidden"),
])
def test_reset_approach_requires_exact_identity_source_shape_and_no_extras(change):
    task,_,w=inputs();r=reset_measurement();change(r)
    e=ApplicableR0Episode(task,"pinch",env_id=0,diagnostic_coefficients=w)
    with pytest.raises(ValueError):
        e.reset({n:lambda:None for n in RESET_HOOKS},
                lambda:(snapshot(task,2),measurement(.005)),reset_approach=r)
    assert not e.active

def test_missing_reset_or_missing_stability_or_coefficient_still_blocks():
    e,t,_=make(reset_distance=False)
    row=e.step(snapshot(t,3),no_bilateral(3,stable=True))
    assert row["blocked_terms"]==["approach_progress"]
    e,t,_=make()
    row=e.step(snapshot(t,3),no_bilateral(3,stable=None))
    assert set(row["blocked_terms"])=={"hold_progress","instability"}
    e,t,_=make(coefficients=False)
    row=e.step(snapshot(t,3),no_bilateral(3,stable=True))
    assert row["diagnostic_reward"] is None
    assert len(row["blocked_terms"])==10
    assert row["terms"]["material_anchor_slip"]["contribution"] is None

def reducer():
    task,budget,w=inputs()
    r=ApplicableR0ControlStep(task,"pinch",env_id=0,budget_config=budget,diagnostic_coefficients=w)
    r.reset({n:lambda:None for n in RESET_HOOKS},
            lambda:(snapshot(task,2),measurement(.005)),reset_approach=reset_measurement())
    return r,task,w

def test_policy_settlement_counts_contributions_separately_from_raw_measurements():
    r,t,w=reducer()
    r.begin(control_index=0,recipe_sha256=r.recipe_sha256,action=np.zeros(24))
    for s in range(3,11):
        assert r.push(snapshot(t,s),no_bilateral(s,stable=True),
                      control_index=0,recipe_sha256=r.recipe_sha256) is (s==10)
    row=r.settle()
    term=row["terms"]["material_anchor_slip"]
    assert term["raw_sum"] is None and term["valid_samples"]==0
    assert term["defined_contribution_samples"]==8 and term["not_applicable_samples"]==8
    assert term["contribution_sum"]==0
    assert row["diagnostic_reward"]==pytest.approx(.02*w["approach_progress"]-8*w["external_support_violation"])
    assert row["total_accounted_physics_steps"]==10
    assert row["training_reward"] is None and not row["native_qualification"]
    assert row["new_physics_steps"]==row["optimizer_updates"]==0
    assert row["reward_contract_version"]==VERSION
    with pytest.raises(RuntimeError,match="settle"):r.settle()

def test_early_terminal_is_unchanged_and_future_physics_cannot_overwrite():
    r,t,_=reducer()
    r.begin(control_index=0,recipe_sha256=r.recipe_sha256,action=np.zeros(24))
    m=no_bilateral(3,stable=False)
    assert r.push(snapshot(t,3),m,control_index=0,recipe_sha256=r.recipe_sha256)
    with pytest.raises(RuntimeError,match="settle"):
        r.push(snapshot(t,4),m,control_index=0,recipe_sha256=r.recipe_sha256)
    row=r.settle()
    assert row["physical_samples"]==1 and row["terminal_reasons"]==["instability"]
    assert row["partial_interval_due_to_terminal"] and row["physics_step_end"]==3
    with pytest.raises(RuntimeError,match="closed"):
        r.begin(control_index=1,recipe_sha256=r.recipe_sha256,action=np.zeros(24))

def test_wrong_sample_invalidates_new_reducer():
    r,t,_=reducer()
    r.begin(control_index=0,recipe_sha256=r.recipe_sha256,action=np.zeros(24))
    with pytest.raises(ValueError):
        r.push(snapshot(t,4),no_bilateral(4,stable=True),control_index=0,recipe_sha256=r.recipe_sha256)
    with pytest.raises(RuntimeError,match="invalid"):r.settle()

def test_hold_and_slip_thresholds_are_inherited_without_relaxation():
    e,t,_=make()
    assert (e.lift_target_m,e.hold_target_s,e.slip_limit_m)==(.03,5.,.0005)
    for s in range(3,2404):
        row=e.step(snapshot(t,s,dz=.035),signals(t,step=s))
        if s<2403:assert not row["success_candidate"]
    assert row["success_candidate"]
    m=signals(t,step=2404,shift=.0006)
    row=e.step(snapshot(t,2404,dz=.035),m)
    assert not row["success_candidate"]
    assert row["terms"]["material_anchor_slip"]["raw"]==pytest.approx(.0006)

def test_legacy_recipe_does_not_match_new_contract():
    from go2_pvcnn.control.dh116.tabletop_deformable_control_step import FlexibleR0ControlStep
    t,b,w=inputs()
    legacy=FlexibleR0ControlStep(t,"pinch",env_id=0,budget_config=b,diagnostic_coefficients=w)
    current=ApplicableR0ControlStep(t,"pinch",env_id=0,budget_config=b,diagnostic_coefficients=w)
    assert current.recipe_sha256!=legacy.recipe_sha256
