"""All samples and diagnostic coefficients below are CPU fixtures."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import R0_TERMS, load_flexible_grasp_task
from go2_pvcnn.control.dh116.tabletop_deformable_control_step import FlexibleR0ControlStep
from test_dh116_deformable_grasp_task import measurement, signals, snapshot

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    task = load_flexible_grasp_task(ROOT/"configs/dh116_tasks/flexible_dual_hand_grasp_v1.json", ROOT)
    budget = json.loads((ROOT/"configs/dh116_training/deformable_task_budget_examples_v2.json").read_text())["plans"]["frozen_eval"]
    budget["setup_physics_steps_per_env"] = 2
    budget["physics"]["solver_substeps_per_physics_step"] = 12
    coefficients = {name: float(i+1) for i,name in enumerate(R0_TERMS)}
    return task, budget, coefficients


def runner(**changes):
    task, budget, coefficients = inputs()
    instance = FlexibleR0ControlStep(task, "pinch", env_id=0, budget_config=budget,
                                    diagnostic_coefficients=coefficients, **changes)
    hooks = {name: lambda: None for name in ("rod","joint","controller","filter","contact","camera_history")}
    instance.reset(hooks, lambda: (snapshot(task, 2), measurement(.005)))
    return instance, task


def begin(instance, control=0):
    instance.begin(control_index=control, recipe_sha256=instance.recipe_sha256, action=np.zeros(24))


def push(instance, task, step, control=0, **changes):
    obs = signals(task, step=step)
    obs.update(changes)
    return instance.push(snapshot(task, step), obs, control_index=control,
                         recipe_sha256=instance.recipe_sha256)


def test_full_interval_emits_exactly_once_and_unknown_first_progress_is_not_zero():
    r, task = runner()
    begin(r)
    for step in range(3,11):
        ready = push(r,task,step)
        assert ready is (step == 10)
    receipt = r.settle()
    assert receipt["physical_samples"] == 8 and receipt["control_index"] == 0
    assert receipt["physics_step_start"] == 2 and receipt["physics_step_end"] == 10
    assert receipt["diagnostic_reward"] is None
    assert receipt["blocked_terms"] == ["approach_progress"]
    assert receipt["terms"]["approach_progress"]["valid_samples"] == 7
    assert receipt["training_reward"] is None and receipt["native_qualification"] is False
    with pytest.raises(RuntimeError,match="settle"):
        r.settle()
    with pytest.raises(ValueError,match="control"):
        begin(r,0)
    begin(r,1)
    for step in range(11,19):
        push(r,task,step,1)
    second = r.settle()
    assert second["diagnostic_reward"] == pytest.approx(
        sum(term["contribution_sum"] for term in second["terms"].values()))
    assert second["diagnostic_reward"] == pytest.approx(8*(2 + 3/480 + 4))
    assert second["recipe_sha256"] == receipt["recipe_sha256"]
    assert second["terminated"] is False and second["truncated"] is False


def test_pending_settlement_blocks_extra_physics_and_next_control():
    r, task=runner();begin(r)
    for step in range(3,11):push(r,task,step)
    with pytest.raises(RuntimeError,match="settle"):
        push(r,task,11)
    with pytest.raises(RuntimeError,match="settle"):
        begin(r,1)
    assert r.settle()["physical_samples"] == 8


def test_incomplete_interval_does_not_settle_but_can_continue():
    r,task=runner();begin(r)
    push(r,task,3)
    with pytest.raises(RuntimeError,match="incomplete"):
        r.settle()
    for step in range(4,11):push(r,task,step)
    assert r.settle()["physical_samples"] == 8


@pytest.mark.parametrize("fault,reason", [
    ({"joint_limit":measurement(True)}, "joint_limit"),
    ({"stable":measurement(False)}, "instability"),
    ({"external_support":measurement(True)}, "prohibited_external_support"),
])
def test_known_failure_closes_at_first_bad_physical_sample(fault,reason):
    r,task=runner();begin(r)
    assert push(r,task,3) is False
    assert push(r,task,4,**fault) is True
    result=r.settle()
    assert result["terminated"] is True and result["terminal_reasons"] == [reason]
    assert result["physical_samples"] == 2 and result["physics_step_end"] == 4
    assert result["partial_interval_due_to_terminal"] is True
    assert result["success_candidate"] is False
    with pytest.raises(RuntimeError,match="closed"):
        begin(r,1)


@pytest.mark.parametrize("edit", [
    lambda r,t: r.push(snapshot(t,4),signals(t,step=4),control_index=0,recipe_sha256=r.recipe_sha256),
    lambda r,t: r.push(snapshot(t,3),signals(t,step=4),control_index=0,recipe_sha256=r.recipe_sha256),
    lambda r,t: r.push(snapshot(t,3,uid="foreign"),signals(t,step=3,uid="foreign"),
                       control_index=0,recipe_sha256=r.recipe_sha256),
])
def test_bad_physical_packet_invalidates_window_and_cannot_be_retried(edit):
    r,task=runner();begin(r)
    with pytest.raises(ValueError):edit(r,task)
    with pytest.raises(RuntimeError,match="invalid"):
        push(r,task,3)
    with pytest.raises(RuntimeError,match="invalid"):
        r.settle()


def test_recipe_and_control_identity_reject_before_consuming():
    r,task=runner();begin(r)
    with pytest.raises(ValueError,match="recipe"):
        r.push(snapshot(task,3),signals(task,step=3),control_index=0,recipe_sha256="0"*64)
    with pytest.raises(ValueError,match="control"):
        push(r,task,3,1)
    assert push(r,task,3) is False


def test_grasp_and_coefficient_recipes_cannot_mix_and_input_mutations_do_not_change_recipe():
    task,budget,weights=inputs()
    a=FlexibleR0ControlStep(task,"pinch",env_id=0,budget_config=budget,diagnostic_coefficients=weights)
    b=FlexibleR0ControlStep(task,"tripod",env_id=0,budget_config=budget,diagnostic_coefficients=weights)
    c=FlexibleR0ControlStep(task,"power",env_id=0,budget_config=budget,diagnostic_coefficients=weights)
    assert len({a.recipe_sha256,b.recipe_sha256,c.recipe_sha256}) == 3
    before=a.recipe_sha256
    weights["required_contact"]=999
    task["grasps"]["pinch"]["required_pads"]["left"].clear()
    budget["hold_control_steps"]=999
    assert a.recipe_sha256 == before


@pytest.mark.parametrize("change", [
    lambda c:None,
    lambda c:{k:v for k,v in c.items() if k != "joint_limit"},
    lambda c:{**c,"joint_limit":None},
    lambda c:{**c,"joint_limit":False},
    lambda c:{**c,"joint_limit":float("nan")},
])
def test_missing_or_invalid_coefficients_are_rejected(change):
    task,budget,weights=inputs()
    with pytest.raises(ValueError,match="coefficients"):
        FlexibleR0ControlStep(task,"pinch",env_id=0,budget_config=budget,
                              diagnostic_coefficients=change(weights))


def test_scalar_is_blocked_by_any_unknown_raw_term_even_zero_coefficient():
    task,budget,weights=inputs();weights["external_support_violation"]=0.
    r=FlexibleR0ControlStep(task,"pinch",env_id=0,budget_config=budget,diagnostic_coefficients=weights)
    hooks={name:lambda:None for name in ("rod","joint","controller","filter","contact","camera_history")}
    r.reset(hooks,lambda:(snapshot(task,2),measurement(.005)));begin(r)
    for step in range(3,11):
        push(r,task,step,external_support=measurement(None))
    receipt=r.settle()
    assert "external_support_violation" in receipt["blocked_terms"]
    assert receipt["diagnostic_reward"] is None


def test_table_contact_during_approach_is_not_automatically_a_failure():
    r,task=runner();begin(r)
    for step in range(3,11):push(r,task,step,table_contact=measurement(True))
    assert r.settle()["terminated"] is False


def test_budget_ends_with_truncation_and_cannot_reset_to_reuse_counts():
    r,task=runner()
    for control in range(360):
        begin(r,control)
        for step in range(3+8*control,11+8*control):push(r,task,step,control)
        receipt=r.settle()
        assert receipt["truncated"] is (control == 359)
    assert receipt["consumed_physics_samples"] == 2880
    assert receipt["accounted_setup_physics_steps"] == 2
    assert receipt["total_accounted_physics_steps"] == 2882
    with pytest.raises(RuntimeError,match="closed"):begin(r,360)
    with pytest.raises(RuntimeError,match="one reset"):r.reset({},None)


def test_unknown_setup_and_training_budget_are_not_cpu_settlement_budgets():
    task,budget,weights=inputs()
    budget["setup_physics_steps_per_env"]=None
    with pytest.raises(ValueError,match="setup"):
        FlexibleR0ControlStep(task,"pinch",env_id=0,budget_config=budget,diagnostic_coefficients=weights)
    budget=json.loads((ROOT/"configs/dh116_training/deformable_task_budget_examples_v2.json").read_text())["plans"]["teacher_update_smoke"]
    budget["setup_physics_steps_per_env"]=2
    with pytest.raises(ValueError,match="frozen"):
        FlexibleR0ControlStep(task,"pinch",env_id=0,budget_config=budget,diagnostic_coefficients=weights)


def test_first_original_five_second_success_settles_one_partial_terminal_once():
    r,task=runner()
    receipt=None
    for control in range(301):
        begin(r,control)
        for step in range(3+8*control,11+8*control):
            ready=r.push(snapshot(task,step,dz=.035),signals(task,step=step),
                         control_index=control,recipe_sha256=r.recipe_sha256)
            if ready:
                receipt=r.settle()
                break
        if receipt["terminated"]:
            break
    assert receipt["control_index"] == 300
    assert receipt["physics_step_end"] == 2403
    assert receipt["physical_samples"] == 1
    assert receipt["consumed_physics_samples"] == 2401
    assert receipt["terminated"] and receipt["success_candidate"]
    assert receipt["terminal_reasons"] == ["success_candidate"]
    assert receipt["partial_interval_due_to_terminal"]
    assert receipt["native_qualification"] is False and receipt["training_reward"] is None
    with pytest.raises(RuntimeError,match="settle"):r.settle()
    with pytest.raises(RuntimeError,match="closed"):begin(r,301)


def test_finite_physical_terms_cannot_overflow_into_a_valid_control_reward():
    task,budget,weights=inputs()
    weights["required_contact"]=1e308
    r=FlexibleR0ControlStep(task,"pinch",env_id=0,budget_config=budget,diagnostic_coefficients=weights)
    hooks={name:lambda:None for name in ("rod","joint","controller","filter","contact","camera_history")}
    r.reset(hooks,lambda:(snapshot(task,2),measurement(.005)));begin(r)
    for step in range(3,11):push(r,task,step)
    with pytest.raises(OverflowError):r.settle()
    with pytest.raises(RuntimeError,match="invalid"):r.settle()


def test_recipe_binds_implementation_helpers_as_well_as_configuration(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_deformable_control_step as module
    task,budget,weights=inputs()
    first=FlexibleR0ControlStep(task,"pinch",env_id=0,budget_config=budget,diagnostic_coefficients=weights)
    identity=first.implementation_sha256
    assert set(identity) == {"tabletop_deformable_control_step.py",
                            "tabletop_deformable_grasp_task.py","tabletop_deformable_task_budget.py"}
    assert all(len(sha) == 64 for sha in identity.values())
    changed={**identity,"tabletop_deformable_grasp_task.py":"a"*64}
    monkeypatch.setattr(module,"_implementation_identity",lambda:changed)
    second=FlexibleR0ControlStep(task,"pinch",env_id=0,budget_config=budget,diagnostic_coefficients=weights)
    assert first.recipe_sha256 != second.recipe_sha256
    identity.clear()
    assert len(first.implementation_sha256) == 3
