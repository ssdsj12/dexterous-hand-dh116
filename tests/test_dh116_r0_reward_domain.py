"""Characterize the frozen R0 domain using explicit CPU samples, never native claims."""
from copy import deepcopy

import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import FlexibleGraspEpisode, RESET_HOOKS
from go2_pvcnn.control.dh116.tabletop_deformable_control_step import FlexibleR0ControlStep
from test_dh116_deformable_control_step import inputs
from test_dh116_deformable_grasp_task import snapshot

SOURCE = "cpu_r0_reward_domain_fixture:"
SLIP_FIELDS = ("material_reference_uid", "material_s_m", "material_uncertainty_m",
               "tangential_reference_uid", "tangential_slip_m", "tangential_uncertainty_m")


def measured(value, field):
    return dict(value=value, source=None if value is None else SOURCE+field)


def no_bilateral(step, *, stable=None, table=True, approach=True):
    """Complete zero-pad contact, unknown external assistance, explicit stable source."""
    row = dict(sample_identity=dict(env_id=0,episode_uid="ep-0",reset_id=0,rod_uid="rod0",
                    physics_step=step,rod_tick=12*step,timestamp_s=step/480),
        approach_distance_m=measured([.01,.01] if approach else None,"approach_distance"),
        table_contact=measured(table,"rod_table_SDF_not_hand_table"),
        external_support=measured(None,"external_support"),
        authored_force=measured(None,"authored_force"),attachment=measured(None,"attachment"),
        joint_limit=measured(False,"actual_q_original_joint_guard"),stable=measured(stable,"stable_predicate"))
    for side in ("left","right"):
        row[side]=measured(dict(rod_uid="rod0",pads=[],opposed=False,**dict.fromkeys(SLIP_FIELDS)),
                           side+":complete_pair_capture_necessary_opposition_false")
    return row


def episode(radius):
    task,_,weights=inputs()
    before=deepcopy(task)
    value=FlexibleGraspEpisode(task,"pinch",env_id=0,diagnostic_coefficients=weights)
    value.reset({name:lambda:None for name in RESET_HOOKS},
                lambda:(snapshot(task,2),measured(radius,"radius_readback")))
    assert task==before and all(coefficient is None for coefficient in task["reward"]["coefficients"].values())
    return value,task


@pytest.mark.parametrize("radius",[None,.005],ids=["radius-unknown","radius-cpu-known"])
@pytest.mark.parametrize("stable",[None,True,False],ids=["stable-unknown","stable-cpu-true","stable-cpu-false"])
def test_original_radius_stability_matrix_keeps_unknown_distinct_from_defined_zero(radius,stable):
    value,task=episode(radius)
    samples=[]
    for step in (3,4):
        measurements=no_bilateral(step,stable=stable)
        row=value.step(snapshot(task,step),measurements)
        terms=row["terms"]
        assert row["phase"]=="approach" and not row["success_candidate"]
        assert not row["prohibited_support_latched"]
        for name in ("required_contact","contact_persistence","grasp_geometry","joint_limit"):
            assert terms[name]["valid"] and terms[name]["raw"]==0
            assert SOURCE in terms[name]["source"]
        # True rod-table contact determines this OR despite other support values being unknown.
        assert terms["external_support_violation"]["raw"]==1
        assert terms["external_support_violation"]["source"]==SOURCE+"rod_table_SDF_not_hand_table"
        expected_lift=None if radius is None else 0.
        expected_hold=None if radius is None or stable is None else 0.
        expected_instability=None if stable is None else float(not stable)
        for name,expected in (("lift_progress",expected_lift),("hold_progress",expected_hold),
                              ("instability",expected_instability)):
            assert terms[name]["raw"]==expected
            assert terms[name]["valid"] is (expected is not None)
        assert row["hold_valid"] is (expected_hold is not None)
        assert row["hold_steps"]==0 and row["hold_s"]==0
        assert (row["min_surface_z_m"] is None) is (radius is None)
        if radius is not None:assert SOURCE+"radius_readback" in terms["lift_progress"]["source"]
        assert terms["instability"]["source"]==(None if stable is None else SOURCE+"stable_predicate")
        assert terms["material_anchor_slip"]["raw"] is None
        assert terms["material_anchor_slip"]["valid"] is False
        assert row["diagnostic_reward"] is row["training_reward"] is None
        assert row["native_qualification"] is False
        samples.append(row)
    assert samples[0]["terms"]["approach_progress"]["raw"] is None
    assert samples[1]["terms"]["approach_progress"]["raw"]==0
    if radius is not None and stable is not None:
        assert samples[1]["blocked_terms"]==["material_anchor_slip"]


@pytest.mark.parametrize("stable",[None,True,False])
def test_original_reducer_approach_termination_uses_measured_stability_not_contact_presence(stable):
    task,budget,weights=inputs()
    reducer=FlexibleR0ControlStep(task,"pinch",env_id=0,budget_config=budget,diagnostic_coefficients=weights)
    reducer.reset({name:lambda:None for name in RESET_HOOKS},
                  lambda:(snapshot(task,2),measured(.005,"radius_readback")))
    reducer.begin(control_index=0,recipe_sha256=reducer.recipe_sha256,action=np.zeros(24))
    for step in range(3,11):
        ready=reducer.push(snapshot(task,step),no_bilateral(step,stable=stable),
                           control_index=0,recipe_sha256=reducer.recipe_sha256)
        if ready:break
    result=reducer.settle()
    assert result["physical_samples"]==(1 if stable is False else 8)
    assert result["terminated"] is (stable is False)
    assert result["terminal_reasons"]==(["instability"] if stable is False else [])
    assert result["terms"]["lift_progress"]["raw_sum"]==0
    assert result["terms"]["hold_progress"]["raw_sum"]==(None if stable is None else 0)
    assert result["terms"]["instability"]["raw_sum"]==(None if stable is None else float(not stable))
    assert result["terms"]["material_anchor_slip"]["valid_samples"]==0
    assert result["diagnostic_reward"] is result["training_reward"] is None
    assert result["native_qualification"] is False


@pytest.mark.parametrize("field",["radius","stable"])
def test_known_value_without_source_cannot_unlock_the_reward_domain(field):
    if field=="radius":
        task,_,weights=inputs()
        value=FlexibleGraspEpisode(task,"pinch",env_id=0,diagnostic_coefficients=weights)
        with pytest.raises(ValueError,match="explicit measurement source"):
            value.reset({name:lambda:None for name in RESET_HOOKS},
                        lambda:(snapshot(task,2),dict(value=.005,source=None)))
    else:
        value,task=episode(.005)
        measurements=no_bilateral(3,stable=True)
        measurements["stable"]["source"]=None
        with pytest.raises(ValueError,match="explicit measurement source"):
            value.step(snapshot(task,3),measurements)


def test_known_absence_of_table_contact_does_not_fill_unknown_external_support():
    value,task=episode(.005)
    row=value.step(snapshot(task,3),no_bilateral(3,stable=True,table=False))
    for name in ("external_support_violation","lift_progress","hold_progress"):
        assert row["terms"][name]["raw"] is None
    assert row["terms"]["instability"]["raw"]==0


def test_missing_approach_sample_breaks_the_difference_baseline_without_zero_fill():
    value,task=episode(.005)
    raw=[]
    for step,available in enumerate((True,True,False,True,True),3):
        row=value.step(snapshot(task,step),no_bilateral(step,stable=True,approach=available))
        raw.append(row["terms"]["approach_progress"]["raw"])
    assert raw==[None,0.,None,None,0.]
