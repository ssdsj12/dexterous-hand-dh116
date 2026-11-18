import hashlib
import json

import pytest

from scripts import record_dh116_teacher_numeric_reset as reset


def write(path, obj):
    path.write_text(json.dumps(obj))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_seal_requires_natural_exit_and_bound_native_artifacts(tmp_path):
    run = tmp_path/"native"
    run.mkdir()
    plan = {"output": str(run), "budget": {"sdk_setup_steps": 2, "ppo_updates": 0}}
    plan_sha = write(run/"frozen_plan.json", plan)
    actual_files = {name: write(run/name, {}) for name in reset.REQUIRED_EVIDENCE}
    shell = dict(output=str(run), exit_code=0, plan_sha256=plan_sha,
                 style="pinch", artifact_sha256=actual_files)
    exit_file = tmp_path/"outer.json"
    assert reset._verify_outer(run, exit_file, write(exit_file, shell)) == shell
    shell["exit_code"] = -11
    with pytest.raises(ValueError, match="natural exit"):
        reset._verify_outer(run, exit_file, write(exit_file, shell))
    shell["exit_code"] = 0
    write(run/reset.NUMERIC_FILE, {"replaced": True})
    with pytest.raises(ValueError, match="artifact SHA"):
        reset._verify_outer(run, exit_file, write(exit_file, shell))
