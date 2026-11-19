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


@pytest.mark.parametrize("passed", [True, False])
def test_explicit_pair_preserves_frozen_bytes_and_static_scope(tmp_path, passed):
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder

    packet = builder.load_pair_proposal()
    packet.update(version="dh116-paired-pinch-reset-preflight-cpu-margin-v1",
                  passes_static_policy_clearance=passed, left_extra_clearance_m=.00005,
                  left_continuous_clearance_m=.002151, right_continuous_clearance_m=.002112)
    path = tmp_path / "candidate.json"
    binding = {"path": str(path), "sha256": write(path, packet)}
    if not passed:
        with pytest.raises(ValueError, match="static clearance"):
            builder.load_bound_pair_proposal(binding)
        return

    assert builder.load_bound_pair_proposal(binding) == packet
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="paired evidence"):
        builder.load_bound_pair_proposal(binding)


def test_new_reset_proposal_requires_matching_outer_and_acquisition_identity(tmp_path):
    run = tmp_path / "native"
    run.mkdir()
    pair = dict(path=str(tmp_path / "candidate.json"), sha256="1" * 64)
    plan_sha = write(run / "frozen_plan.json", {"pair_proposal": pair})
    hashes = {name: write(run / name, {}) for name in reset.REQUIRED_EVIDENCE}
    hashes["reset_capture.json"] = write(run / "reset_capture.json", {"pair_proposal": pair})
    outer = tmp_path / "outer.json"
    receipt = dict(output=str(run), exit_code=0, plan_sha256=plan_sha, style="pinch",
                   artifact_sha256=hashes)
    with pytest.raises(ValueError, match="proposal binding"):
        reset._verify_outer(run, outer, write(outer, receipt))
    receipt["pair_proposal"] = pair
    assert reset._verify_outer(run, outer, write(outer, receipt)) == receipt


def test_fresh_recentered_seed_cannot_omit_proposal_from_manifest_and_receipt(tmp_path):
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_preflight as preflight
    from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import _digest

    seed = builder.load_pair_proposal()["reset_seed"]
    seed["source_description"] += "; different frozen source"
    seed.pop("sha256")
    seed["sha256"] = _digest(seed)

    def source(name, value):
        path = tmp_path / name
        return {"path": str(path), "sha256": write(path, value)}

    cfg = preflight.candidate_config()
    cfg["resolved_reset_manifest"] = source("manifest.json", dict(
        schema=preflight.RESET_MANIFEST_SCHEMA, style="pinch", pose_preview_only=False,
        target_schema=preflight.TARGET_SCHEMA_VERSION, target_material_ids=list(range(41)),
        reset_seed=seed, source_refs=dict(
            original_receipt=source("outer.json", dict(style="pinch", exit_code=0)),
            reset_readback=source("readback.json", {}),
            numeric_reset=source("numeric.json", {}))))
    with pytest.raises(ValueError, match="fresh Pinch requires an explicit paired proposal"):
        preflight.load_reset_manifest(cfg, project=tmp_path)
