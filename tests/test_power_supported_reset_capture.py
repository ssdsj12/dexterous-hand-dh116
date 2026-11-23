"""Exercise the Power supported-scene routing without starting Isaac."""
import hashlib

import pytest

from test_tripod_supported_reset_capture import entry, inputs


def test_power_native_reset_plan_binds_own_scope_and_two_step_budget(tmp_path):
    recorder = entry()
    _, _, extra = inputs(tmp_path)
    task = recorder.ROOT / "configs/dh116_tasks/flexible_power_supported_contact_v1.json"
    digest = hashlib.sha256(task.read_bytes()).hexdigest()
    output = tmp_path / "native"
    plan = recorder.freeze_plan(
        output, task_config=task, task_config_sha256=digest, extra_sources=extra, style="power"
    )
    assert plan["style"] == "power"
    assert plan["task_config"] == {"path": str(task), "sha256": digest}
    assert plan["budget"]["mode"] == "fresh_power_reset17"
    assert plan["budget"]["sdk_steps"] == 2
    assert plan["budget"]["episode_resets"] == 1
    assert all(plan["budget"][name] == 0 for name in ("control_steps", "solver_steps", "ppo_updates"))
    assert plan["mesh_count"] == 17
    assert not output.exists()


def test_power_reset_rejects_tripod_seed_before_any_native_call(tmp_path):
    recorder = entry()
    task, digest, extra = inputs(tmp_path)
    output = tmp_path / "native"
    with pytest.raises(ValueError, match="Power rod"):
        recorder.freeze_plan(
            output, task_config=task, task_config_sha256=digest, extra_sources=extra, style="power"
        )
    assert not output.exists()
