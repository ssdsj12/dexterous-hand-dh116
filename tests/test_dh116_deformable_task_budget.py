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
    require_grasp_evidence_scope(plan_flexible_experiment(locked))


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
