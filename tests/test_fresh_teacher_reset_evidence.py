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

@pytest.mark.parametrize("exit_fields", [{}, {"exit_code": -11}, {"exit_code": False}, {"exit_code": 0.0}])
def test_manifest_reader_rejects_non_native_exit_before_examining_reset_data(tmp_path, exit_fields):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_preflight as preflight

    def source(name, payload):
        path = tmp_path / name
        return dict(path=str(path), sha256=write(path, payload))

    refs = dict(original_receipt=source("outer.json", dict(style="pinch", **exit_fields)),
                reset_readback=source("readback.json", {}),
                numeric_reset=source("numeric.json", {}))
    cfg = preflight.candidate_config()
    cfg["resolved_reset_manifest"] = source("manifest.json", dict(
        schema=preflight.RESET_MANIFEST_SCHEMA, style="pinch", pose_preview_only=False,
        target_schema=preflight.TARGET_SCHEMA_VERSION, target_material_ids=list(range(41)),
        source_refs=refs))
    with pytest.raises(ValueError, match="natural exit 0"):
        preflight.load_reset_manifest(cfg, project=tmp_path)
