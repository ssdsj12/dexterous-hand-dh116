"""Fail-closed applicability across unobserved contact history."""
from copy import deepcopy
import numpy as np
import pytest
from test_dh116_r0_applicability import make, reducer
from test_dh116_deformable_grasp_task import snapshot, signals, measurement
from test_dh116_r0_reward_domain import no_bilateral

def test_unknown_interval_cannot_later_be_relabelled_as_proven_never_grasped():
    e,t,_=make()
    missing=no_bilateral(3,stable=True);missing["right"]=measurement(None)
    e.step(snapshot(t,3),missing)
    row=e.step(snapshot(t,4),no_bilateral(4,stable=True))
    assert row["terms"]["material_anchor_slip"]["applicable"] is None
    assert row["terms"]["material_anchor_slip"]["contribution"] is None
    assert row["contact_history_unknown"] is True

@pytest.mark.parametrize("grasp",["tripod","power"])
def test_required_pad_profile_is_grasp_specific(grasp):
    e,t,_=make(grasp)
    m=signals(t,step=3)
    row=e.step(snapshot(t,3),m)
    assert not row["bilateral_contact_seen"]  # Pinch's 2 pads are not sufficient.
    assert row["terms"]["material_anchor_slip"]["applicable"] is False
    m=signals(t,step=4)
    for side in ("left","right"):m[side]["value"]["pads"]=t["grasps"][grasp]["required_pads"][side]
    row=e.step(snapshot(t,4),m)
    assert row["bilateral_contact_seen"]
    row=e.step(snapshot(t,5),no_bilateral(5,stable=True))
    assert row["terms"]["material_anchor_slip"]["applicable"] is True
    assert row["diagnostic_reward"] is None

def test_approach_gap_breaks_difference_baseline_and_does_not_fill_zero():
    e,t,_=make()
    first=e.step(snapshot(t,3),no_bilateral(3,stable=True,approach=False))
    second=e.step(snapshot(t,4),no_bilateral(4,stable=True))
    third=e.step(snapshot(t,5),no_bilateral(5,stable=True))
    assert first["terms"]["approach_progress"]["raw"] is None
    assert second["terms"]["approach_progress"]["raw"] is None
    assert third["terms"]["approach_progress"]["raw"]==0

def test_actual_slip_remains_measurement_after_contact_recovers():
    e,t,_=make()
    e.step(snapshot(t,3),signals(t,step=3))
    e.step(snapshot(t,4),no_bilateral(4,stable=True))
    row=e.step(snapshot(t,5),signals(t,step=5,shift=.0007))
    assert row["terms"]["material_anchor_slip"]["raw"]==pytest.approx(.0007)
    assert row["terms"]["material_anchor_slip"]["applicable"] is True
    assert row["terms"]["material_anchor_slip"]["contribution"]<0
    assert row["hold_s"]==0

def test_mixed_interval_counts_na_measured_and_missing_independently():
    r,t,_=reducer()
    r.begin(control_index=0,recipe_sha256=r.recipe_sha256,action=np.zeros(24))
    for s in range(3,11):
        m=no_bilateral(s,stable=True) if s!=6 else signals(t,step=s)
        r.push(snapshot(t,s),m,control_index=0,recipe_sha256=r.recipe_sha256)
    row=r.settle();term=row["terms"]["material_anchor_slip"]
    assert term["valid_samples"]==1
    assert term["not_applicable_samples"]==3
    assert term["defined_contribution_samples"]==4
    assert term["contribution_sum"] is None and term["raw_sum"] is None
    assert row["diagnostic_reward"] is None

def test_diagnostic_output_cannot_mutate_sticky_internal_contact_state():
    e,t,_=make()
    row=e.step(snapshot(t,3),signals(t,step=3))
    row["bilateral_contact_seen"]=False
    row["terms"]["required_contact"]["raw"]=0
    later=e.step(snapshot(t,4),no_bilateral(4,stable=True))
    assert later["bilateral_contact_seen"]
    assert later["terms"]["material_anchor_slip"]["contribution"] is None
