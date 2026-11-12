"""Plan flexible-task evidence budgets without authorizing or launching native work."""

import math
import re

from .tabletop_deformable_rod_contact import DT, STEPS, SUBSTEPS


VERSION = "dh116-deformable-task-budget-v1"
PHYSICS_HZ = round(1 / DT)
CONTROL_HZ = 60
HOLD_CONTROL_STEPS = 5 * CONTROL_HZ
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _count(config, key, minimum=0):
    value = config.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{key} requires an integer at least {minimum}")
    return value


def _optional_sha256(value, label):
    if value is not None and (not isinstance(value, str)
                              or not _SHA256.fullmatch(value) or value == "0" * 64):
        raise ValueError(f"{label} requires a nonzero SHA256 or null")
    return value


def plan_flexible_experiment(config):
    """Count rows per environment and across the planned deformable-rod run."""
    if config.get("version") != VERSION or config.get("task_kind") != "deformable_rod":
        raise ValueError("expected the independent deformable_rod budget contract")
    if _count(config, "physics_hz") != PHYSICS_HZ or _count(config, "control_hz") != CONTROL_HZ:
        raise ValueError("expected the 480 Hz physical and 60 Hz control clocks")
    phase = config.get("phase")
    environments = _count(config, "environments", 1)
    updates = _count(config, "updates")
    wall_seconds = _count(config, "wall_seconds", 1)
    approach = _count(config, "approach_control_steps")
    hold = _count(config, "hold_control_steps")
    rollout = _count(config, "rollout_control_steps")
    calibration_steps = _count(config, "calibration_physics_steps")
    physics = config["physics"]
    support = config["external_support"]
    gravity = physics["gravity_world_m_s2"]
    pins = physics["pinned_node_ids"]
    solver_substeps = physics["solver_substeps_per_physics_step"]
    if (not isinstance(gravity, list) or len(gravity) != 3
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in gravity)
            or not isinstance(pins, list) or any(type(value) is not int or value < 0 for value in pins)
            or len(pins) != len(set(pins))):
        raise ValueError("gravity and pinned material nodes must be explicit")
    if solver_substeps is not None:
        _count(physics, "solver_substeps_per_physics_step", 1)
    physical_source = _optional_sha256(physics["source_sha256"], "physical parameters")
    support_source = _optional_sha256(support["measurement_source_sha256"], "external support")
    calibration = phase == "contact_calibration"
    if calibration:
        if (environments, updates, approach, hold, rollout, calibration_steps) != (1, 0, 0, 0, 0, STEPS):
            raise ValueError("contact calibration requires one environment and exactly 240 physics steps")
        if gravity != [0, 0, 0] or pins != [0, 20] or solver_substeps != SUBSTEPS:
            raise ValueError("contact calibration requires the known zero-gravity, two-pin setup")
        if support["policy"] != "calibration_pinned_endpoints":
            raise ValueError("calibration external support must disclose pinned endpoints")
    else:
        if gravity[2] >= 0 or pins:
            raise ValueError("task experiments require a free rod under downward gravity")
        if support["policy"] != "forbidden_during_lift_hold":
            raise ValueError("task experiments require external support forbidden during Lift and Hold")
        if calibration_steps:
            raise ValueError("task experiments cannot count calibration physics steps")
        if phase == "teacher_update_smoke":
            if (environments, updates, rollout, approach, hold) != (2, 1, 32, 0, 0):
                raise ValueError("Teacher smoke requires two environments, 32 control steps and one update")
        elif phase == "frozen_eval":
            if updates or rollout:
                raise ValueError("frozen evaluation cannot request a training update or rollout")
            if not approach:
                raise ValueError("frozen evaluation requires a separate approach budget")
            if hold < HOLD_CONTROL_STEPS:
                raise ValueError("frozen evaluation requires at least five continuous seconds of hold")
        else:
            raise ValueError("unknown deformable task experiment phase")

    decimation = PHYSICS_HZ // CONTROL_HZ
    controls = rollout + approach + hold
    physical = calibration_steps + controls * decimation
    return {
        "phase": phase,
        "task_kind": "deformable_rod",
        "calibration_only": calibration,
        "environments": environments,
        "updates": updates,
        "wall_seconds": wall_seconds,
        "control_steps_per_env": controls,
        "physics_steps_per_env": physical,
        "hold_physics_steps_per_env": hold * decimation,
        "total_physics_rows": physical * environments,
        "solver_substeps_per_env": None if solver_substeps is None else physical * solver_substeps,
        "physics_config_validation": "configured_values_validated",
        "physical_source_sha256": physical_source,
        "external_support_measurement_sha256": support_source,
        "requires_separate_native_approval": not calibration,
        "native_qualification": False,
    }


def require_grasp_evidence_scope(plan):
    """Check plan metadata only. Native qualification requires independent physical evidence."""
    if plan.get("task_kind") != "deformable_rod" or plan.get("phase") != "frozen_eval":
        raise ValueError("grasp evidence requires an independent frozen deformable-rod evaluation")
    if not plan.get("physical_source_sha256") or not plan.get("external_support_measurement_sha256"):
        raise ValueError("grasp evidence requires locked physical and external support measurement sources")
    if plan.get("solver_substeps_per_env") is None:
        raise ValueError("grasp evidence requires a locked solver substep budget")
