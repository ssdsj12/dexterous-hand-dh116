"""Freeze the actual Tripod acquisition entry; no App or CUDA execution."""
import hashlib
import importlib
import json
from pathlib import Path
import pytest

from test_fresh_supported_builder import EVIDENCE


def inputs(tmp_path):
    task = tmp_path/"tripod.json"
    source = EVIDENCE/"ref902/execution-closure-native-context-fixed/native/frozen_plan.json"
    task.write_text(json.dumps(json.loads(source.read_bytes())["reset_config"]))
    extra = dict(rays=EVIDENCE/"ag2203-power-palm-rays-enabled-native/native/palm_rays.json",
        left_bake=EVIDENCE/"ag2184-left-power-sdf-native/bake",
        right_bake=EVIDENCE/"ag2200-right-power-sdf-native/bake",
        proximal_capture=EVIDENCE/"ag2312-power-proximal-tensor-ray-native",
        proximal_bake=EVIDENCE/"ag2314-power-proximal-sdf-half-mm/bake")
    return task,hashlib.sha256(task.read_bytes()).hexdigest(),extra


def entry():
    name = "scripts.record_dh116_tripod_supported_reset"
    assert importlib.util.find_spec(name) is not None, "missing fresh17 Tripod native Reset acquisition entry"
    return importlib.import_module(name)


def test_freeze_binds_actual_seventeen_sources_and_reset_only_budget_without_creating_output(tmp_path):
    m = entry()
    task,digest,extra = inputs(tmp_path)
    output = tmp_path/"native"
    plan = m.freeze_plan(output,task_config=task,task_config_sha256=digest,extra_sources=extra)
    assert plan["version"] == m.VERSION and plan["style"] == "tripod"
    assert plan["budget"]["sdk_steps"] == plan["budget"]["sdk_setup_steps"] == 2
    assert plan["budget"]["episode_resets"] == 1
    for field in ("control_steps","solver_steps","teacher_queries","ppo_updates","model_loads","auto_retries"):
        assert plan["budget"][field] == 0
    assert plan["budget"]["auto_kill"] is False
    assert plan["task_config"] == dict(path=str(task),sha256=digest)
    assert plan["frozen_files"][str(task)] == digest
    assert plan["mesh_count"] == 17 and plan["pair_capacity"] == 8364
    for name in ("left.base_link","right.base_link","left.finger52_link","right.finger52_link"):
        folder = extra[name.split(".")[0]+"_bake"] if name.endswith("base_link") else extra["proximal_bake"]
        path = folder/(name+".npz")
        assert plan["frozen_files"][str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not output.exists()


def test_new_reset_entry_refuses_changed_candidate_before_creating_or_running_any_output(tmp_path):
    m = entry()
    task,digest,extra = inputs(tmp_path)
    task.write_bytes(task.read_bytes()+b" ")
    with pytest.raises(ValueError,match="SHA"):
        m.freeze_plan(tmp_path/"native",task_config=task,task_config_sha256=digest,extra_sources=extra)
    assert not (tmp_path/"native").exists()
