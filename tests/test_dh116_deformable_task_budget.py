import copy
import json
from pathlib import Path

import pytest

from go2_pvcnn.control.dh116.tabletop_deformable_task_budget import (
    plan_flexible_experiment, require_grasp_evidence_scope,
)


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/dh116_training/deformable_task_budget_examples_v1.json"
)


@pytest.fixture
def plans():
    return json.loads(CONFIG.read_text())["plans"]


def test_calibration_replays_exact_240_step_scope(plans):
    plan = plan_flexible_experiment(plans["contact_calibration"])
    assert plan["physics_steps_per_env"] == plan["total_physics_rows"] == 240
    assert plan["control_steps_per_env"] == plan["updates"] == 0
    assert plan["solver_substeps_per_env"] == 2880
    assert plan["calibration_only"] is True
    assert plan["native_qualification"] is False
    with pytest.raises(ValueError, match="grasp evidence"):
        require_grasp_evidence_scope(plan)


def test_two_environment_teacher_budget_requires_512_rows(plans):
    plan = plan_flexible_experiment(plans["teacher_update_smoke"])
    assert plan["control_steps_per_env"] == 32
    assert plan["physics_steps_per_env"] == 256
    assert plan["total_physics_rows"] == 512
    assert plan["updates"] == 1
    assert plan["requires_separate_native_approval"] is True
    assert plan["native_qualification"] is False
    with pytest.raises(ValueError, match="grasp evidence"):
        require_grasp_evidence_scope(plan)


def test_frozen_eval_budgets_approach_and_five_second_hold(plans):
    example = plans["frozen_eval"]
    plan = plan_flexible_experiment(example)
    assert plan["hold_physics_steps_per_env"] == 2400
    assert plan["physics_steps_per_env"] == 2400 + example["approach_control_steps"] * 8
    assert plan["total_physics_rows"] == plan["physics_steps_per_env"] * example["environments"]
    assert plan["requires_separate_native_approval"] is True
    assert plan["native_qualification"] is False
    with pytest.raises(ValueError, match="measurement sources"):
        require_grasp_evidence_scope(plan)
    locked = copy.deepcopy(example)
    locked["physics"]["source_sha256"] = "a" * 64
    locked["external_support"]["measurement_source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="solver substep"):
        require_grasp_evidence_scope(plan_flexible_experiment(locked))
    locked["physics"]["solver_substeps_per_physics_step"] = 12
    phase_only_plan = plan_flexible_experiment(locked)
    assert phase_only_plan["setup_physics_steps_per_env"] is None
    assert phase_only_plan["physics_steps_per_env"] == phase_only_plan["measured_physics_steps_per_env"]
    with pytest.raises(ValueError, match="setup physics"):
        require_grasp_evidence_scope(phase_only_plan)


@pytest.mark.parametrize(
    "change, match",
    [
        ({"hold_control_steps": 299}, "hold"),
        ({"approach_control_steps": 0}, "approach"),
        ({"updates": 1}, "update"),
        ({"physics_hz": 240}, "480 Hz"),
        ({"control_hz": 30}, "60 Hz"),
    ],
)
def test_frozen_eval_rejects_budget_and_clock_regressions(plans, change, match):
    with pytest.raises(ValueError, match=match):
        plan_flexible_experiment({**plans["frozen_eval"], **change})


def test_evaluation_rejects_calibration_physics_or_ambiguous_support(plans):
    case = plans["frozen_eval"]
    for change in ({"pinned_node_ids": [0, 20]}, {"gravity_world_m_s2": [0, 0, 0]}):
        with pytest.raises(ValueError, match="free rod under"):
            plan_flexible_experiment({**case, "physics": {**case["physics"], **change}})
    with pytest.raises(ValueError, match="external support"):
        plan_flexible_experiment({
            **case, "external_support": plans["contact_calibration"]["external_support"],
        })
    with pytest.raises(ValueError, match="SHA256"):
        plan_flexible_experiment({
            **case, "external_support": {**case["external_support"], "measurement_source_sha256": "0" * 64},
        })


def test_wall_budget_is_independent_of_the_physical_clock(plans):
    case = plans["contact_calibration"]
    plan = plan_flexible_experiment(case)
    assert plan["wall_seconds"] == 180
    shorter = plan_flexible_experiment({**case, "wall_seconds": 60})
    assert shorter["wall_seconds"] == 60
    assert shorter["physics_steps_per_env"] == plan["physics_steps_per_env"] == 240
    assert shorter["physics_config_validation"] == "configured_values_validated"
    assert shorter["native_qualification"] is False
    for wall_seconds in (None, False, 0, -1, 0.5, "180"):
        with pytest.raises(ValueError, match="wall_seconds"):
            plan_flexible_experiment({**case, "wall_seconds": wall_seconds})


@pytest.fixture
def v2_plans():
    path = CONFIG.with_name("deformable_task_budget_examples_v2.json")
    return json.loads(path.read_text())["plans"]


def test_v1_aliases_are_phase_only_and_require_approval(plans):
    calibration = plan_flexible_experiment(plans["contact_calibration"])
    assert calibration["requires_separate_native_approval"] is True
    assert calibration["native_qualification"] is False
    assert calibration["physics_steps_per_env"] == calibration["total_physics_rows"] == 240
    assert calibration["budget_basis"] == "phase_only_setup_unaccounted"
    assert calibration["setup_physics_steps_per_env"] is None
    assert calibration["total_physics_steps_per_env"] is None


def test_v2_contact_calibration_accounts_for_every_setup_tick(v2_plans):
    case = v2_plans["contact_calibration"]
    plan = plan_flexible_experiment(case)
    assert plan["wall_seconds"] == 180
    assert plan["control_steps_per_env"] == plan["updates"] == 0
    assert plan["setup_physics_steps_per_env"] == 2
    assert plan["measured_physics_steps_per_env"] == plan["total_measured_ledger_rows"] == 238
    assert plan["total_physics_steps_per_env"] == plan["total_physics_steps_across_envs"] == 240
    assert plan["measured_solver_substeps_per_env"] == 238 * 12
    assert plan["requires_separate_native_approval"] is True
    assert plan["native_qualification"] is False
    assert "physics_steps_per_env" not in plan
    for update in ({"setup_physics_steps_per_env": None}, {"setup_physics_steps_per_env": 0},
                   {"calibration_measured_physics_steps": 240}):
        with pytest.raises(ValueError, match="setup.*238"):
            plan_flexible_experiment({**case, **update})


@pytest.mark.parametrize(("phase", "steps", "measured", "ledger", "hold"), [
    ("teacher_update_smoke", 32, 256, 512, 0),
    ("frozen_eval", 360, 2880, 2880, 2400),
])
def test_v2_unknown_setup_cannot_be_counted_as_zero(v2_plans, phase, steps, measured, ledger, hold):
    case = v2_plans[phase]
    plan = plan_flexible_experiment(case)
    assert plan["control_steps_per_env"] == steps
    assert plan["measured_physics_steps_per_env"] == measured
    assert plan["total_measured_ledger_rows"] == ledger
    assert plan["hold_physics_steps_per_env"] == hold
    assert plan["setup_physics_steps_per_env"] is None
    assert plan["total_physics_steps_per_env"] is None
    assert plan["total_physics_steps_across_envs"] is None
    assert plan["requires_separate_native_approval"] is True
    assert plan["native_qualification"] is False
    for invalid in (False, -1, 1.5, "2"):
        with pytest.raises(ValueError, match="setup_physics_steps_per_env"):
            plan_flexible_experiment({**case, "setup_physics_steps_per_env": invalid})
    locked = plan_flexible_experiment({**case, "setup_physics_steps_per_env": 2})
    assert locked["total_physics_steps_per_env"] == measured + 2
    assert locked["total_physics_steps_across_envs"] == (measured + 2) * case["environments"]
    assert locked["total_measured_ledger_rows"] == ledger


def test_v2_frozen_evidence_scope_needs_setup_budget(v2_plans):
    case = copy.deepcopy(v2_plans["frozen_eval"])
    case["physics"].update(source_sha256="a" * 64, solver_substeps_per_physics_step=12)
    case["external_support"]["measurement_source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="setup physics"):
        require_grasp_evidence_scope(plan_flexible_experiment(case))
    case["setup_physics_steps_per_env"] = 2
    plan = plan_flexible_experiment(case)
    require_grasp_evidence_scope(plan)
    assert plan["native_qualification"] is False
