import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_dh116_work_packages as scheduler
from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_launch as actual
from test_dh116_work_packages import repo
from test_native_teacher_actual_launch import actual_fixture


def test_private_recipe_bytes_validate_against_same_scheduler_contract(repo, tmp_path):
    from scripts import prepare_dh116_pinch_ppo_draft as draft

    package, old_spec, _ = actual_fixture(repo, tmp_path)
    training = Path(old_spec["teacher_contract"]["training_config"]["path"])
    environment = json.loads((repo.root / actual.ENVIRONMENT).read_bytes())
    pins = {key: old_spec["teacher_contract"][key] for key in
            ("backend_sha256", "target_sha256", "reset_sha256", "reward_sha256", "semantic_sha256")}
    result = draft.freeze_recipe(repo, training=training, environment=environment,
        limits=draft.window_limits(inner_wall_s=180), pins_for_source=lambda _: pins,
        resource_budget=dict(timeout_seconds=240, min_gpu_free_mib=4096, min_disk_free_bytes=1073741824))
    proposal = result["package"]
    assert proposal["status"] == "pending" and proposal["requires_approval"] is True
    assert not (repo.root / proposal["command_spec"]).exists()
    assert result["spec"]["budget"]["physics_steps"] == 258
    assert result["spec"]["budget"]["updates"] == 1
    assert result["launch"]["role"] == "a_fresh"
    assert result["spec"]["teacher_contract"]["semantic_sha256"] == pins["semantic_sha256"]
    core = scheduler.native_recipe_identity(repo, proposal, result["spec_bytes"], result["spec"])
    assert actual.validate_actual_launch(result["launch"], repository=repo, package=proposal, recipe=core)
    assert scheduler.file_hash(training) == result["spec"]["teacher_contract"]["training_config"]["sha256"]
    assert result["package"]["actual_launch"]["sha256"] == hashlib.sha256(result["launch_bytes"]).hexdigest()

    for relative, payload in [(proposal["command_spec"], result["spec_bytes"]),
                              (proposal["actual_launch"]["path"], result["launch_bytes"])]:
        (repo.root / relative).write_bytes(payload)
    registry = repo.registry()
    registry["packages"] = {draft.PACKAGE: proposal}
    (repo.root / scheduler.REGISTRY).write_text(json.dumps(registry))
    plan = repo.native_plan(scope="all")
    assert plan["blockers"][draft.PACKAGE]["kind"] == "waiting_approval"
    assert draft.PACKAGE not in plan["native_ready"] and draft.PACKAGE not in plan["native_qualified"]
    training.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="input digest"):
        repo.native_recipe(proposal)


@pytest.mark.parametrize("key", ["authorization", "resource_lock", "budget"])
def test_training_draft_refuses_reused_process_metadata(tmp_path, key):
    from scripts import prepare_dh116_pinch_ppo_draft as draft
    from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import candidate_config
    from go2_pvcnn.control.dh116.tabletop_native_teacher_training import VERSION

    config = dict(schema=VERSION, preflight=candidate_config(), rollout_steps=32,
                  episode_cap_s=32 / 60, model_config={"grasp": "pinch"})
    if key == "budget":
        config["preflight"][key] = dict(max_sdk_steps=258, max_resets=1, max_solver_calls=256,
                                       max_episode_count=1, wall_timeout_s=180)
    else:
        config["preflight"][key] = {"path": "/not-authorized", "sha256": "a" * 64}
    path = tmp_path / "template.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unleased"):
        draft.prepare_training(path, hashlib.sha256(path.read_bytes()).hexdigest(), inner_wall_s=180)
