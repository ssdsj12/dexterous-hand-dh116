"""CPU admission comparison; isolated proof mocks never grant native qualification."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from go2_pvcnn.control.dh116 import tabletop_stereo_promotion as promotion
from go2_pvcnn.control.dh116 import tabletop_stereo_student_cli as student
from go2_pvcnn.control.dh116 import tabletop_stereo_training_cli as collection
from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116.tabletop_stereo_preprocessing import processing_contract

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = ("student", "collection")


@pytest.fixture
def admission(tmp_path, monkeypatch):
    camera = tmp_path / "camera.json"
    camera.write_bytes((ROOT / "configs/dh116_training/stereo_camera_v1.json").read_bytes())
    calibration = StereoCalibration.from_json(camera)
    teacher_checkpoint = tmp_path / "teacher.pt"
    teacher_checkpoint.write_bytes(b"CPU metadata fixture; no native proof")
    proof = dict(
        calibration_sha256=calibration.sha256,
        depth_estimator_config=deepcopy(processing_contract(camera)["depth_estimator"]),
    )
    real_teacher = promotion.require_teacher_qualification
    real_stereo = promotion.verify_stereo_qualification
    # Assume lower qualification only to isolate configuration admission. Return
    # the same proof object so mutation of measured evidence remains observable.
    monkeypatch.setattr(promotion, "require_teacher_qualification", lambda *a, **k: {"mock_only": True})
    monkeypatch.setattr(promotion, "verify_stereo_qualification", lambda *a, **k: proof)
    arguments = SimpleNamespace(
        teacher_proof={}, teacher_checkpoint=teacher_checkpoint, stereo_proof={},
        checkpoint=None, actor_package=tmp_path / "bound_actor.pt", camera_config=camera,
    )

    def call(entry):
        if entry == "collection":
            return collection._formal_collection_admission({}, teacher_checkpoint, {}, camera)
        return student._formal_admission(arguments, calibration)

    return SimpleNamespace(call=call, proof=proof, args=arguments, camera=camera,
        calibration=calibration, real_teacher=real_teacher, real_stereo=real_stereo)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("key,value", [
    ("subpixel", True), ("method", "census"),
    ("min_disparity", 2), ("max_disparity", 63), ("block_size", 7),
    ("min_depth_m", .25), ("max_depth_m", 1.1),
    ("lr_tolerance_px", .5), ("texture_threshold", .002),
    ("uniqueness_threshold", .1), ("rectify", True),
])
def test_same_calibration_changed_matcher_is_rejected(admission, entry, key, value):
    assert admission.proof["depth_estimator_config"][key] != value
    admission.proof["depth_estimator_config"][key] = value
    with pytest.raises(ValueError, match="matcher differs"):
        admission.call(entry)


@pytest.mark.parametrize("entry", ENTRIES)
def test_changed_selected_camera_matcher_with_same_intrinsics_is_rejected(admission, entry):
    config = json.loads(admission.camera.read_text())
    config["depth_estimator"] = deepcopy(admission.proof["depth_estimator_config"])
    config["depth_estimator"]["subpixel"] = True
    admission.camera.write_text(json.dumps(config))
    assert StereoCalibration.from_json(admission.camera).sha256 == admission.calibration.sha256
    with pytest.raises(ValueError, match="matcher differs"):
        admission.call(entry)


@pytest.mark.parametrize("entry", ENTRIES)
def test_missing_matcher_cannot_reuse_calibration_only_proof(admission, entry):
    admission.proof.pop("depth_estimator_config")
    with pytest.raises(ValueError, match="matcher differs"):
        admission.call(entry)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("matcher", [None, []])
def test_malformed_matcher_cannot_reuse_calibration_only_proof(admission, entry, matcher):
    admission.proof["depth_estimator_config"] = matcher
    with pytest.raises(ValueError, match="matcher differs"):
        admission.call(entry)


@pytest.mark.parametrize("entry", ENTRIES)
def test_string_disparity_is_not_equivalent_to_numeric_matcher_setting(admission, entry):
    admission.proof["depth_estimator_config"]["max_disparity"] = "64"
    with pytest.raises(ValueError, match="matcher differs"):
        admission.call(entry)


@pytest.mark.parametrize("entry", ENTRIES)
def test_unrecognized_proof_matcher_field_is_not_silently_ignored(admission, entry):
    admission.proof["depth_estimator_config"]["unrecognized_semantics"] = True
    with pytest.raises(ValueError, match="matcher differs"):
        admission.call(entry)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("resource_batch", [1, 99])
def test_resource_batch_does_not_change_matcher_or_mutate_proof(admission, entry, resource_batch):
    admission.proof["depth_estimator_config"]["image_processing_batch_size"] = resource_batch
    before = deepcopy(admission.proof)
    result = admission.call(entry)
    assert result["stereo"] is admission.proof
    assert admission.proof == before


@pytest.mark.parametrize("entry", ENTRIES)
def test_calibration_mismatch_remains_rejected(admission, entry):
    admission.proof["calibration_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="calibration differs"):
        admission.call(entry)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("verifier,message", [
    ("require_teacher_qualification", "genuine Teacher proof is missing"),
    ("verify_stereo_qualification", "genuine stereo proof is missing"),
])
def test_external_proof_verifier_rejection_propagates_before_matcher(
    admission, entry, monkeypatch, verifier, message,
):
    admission.proof.pop("depth_estimator_config")

    def reject(*args, **kwargs):
        raise ValueError(message)

    monkeypatch.setattr(promotion, verifier, reject)
    with pytest.raises(ValueError, match=message):
        admission.call(entry)


@pytest.mark.parametrize("entry", ENTRIES)
def test_real_stereo_verifier_still_rejects_summary_only_input(admission, entry, monkeypatch):
    monkeypatch.setattr(promotion, "verify_stereo_qualification", admission.real_stereo)
    with pytest.raises(ValueError, match="native stereo"):
        admission.call(entry)


@pytest.mark.parametrize("entry", ENTRIES)
def test_real_teacher_verifier_still_rejects_summary_only_input(admission, entry, monkeypatch):
    monkeypatch.setattr(promotion, "require_teacher_qualification", admission.real_teacher)
    with pytest.raises(ValueError, match="native evaluation report"):
        admission.call(entry)


@pytest.mark.parametrize("missing", ["teacher_proof", "teacher_checkpoint", "stereo_proof"])
def test_student_requires_explicit_teacher_and_stereo_proofs(admission, missing):
    setattr(admission.args, missing, None)
    with pytest.raises(ValueError, match="Teacher proof.*Teacher checkpoint.*stereo qualification proof"):
        admission.call("student")


@pytest.mark.parametrize("source", ["actor_package", "checkpoint"])
def test_student_accepts_explicit_warm_start_or_resume_at_admission(admission, source):
    admission.args.actor_package = None
    admission.args.checkpoint = None
    setattr(admission.args, source, admission.camera.parent / "explicit_initialization.pt")
    assert admission.call("student")["stereo"] == admission.proof


def test_student_cannot_omit_both_warm_start_and_resume(admission):
    admission.args.actor_package = None
    with pytest.raises(ValueError, match="explicit actor-package warm start or exact Student resume"):
        admission.call("student")


def test_student_retains_explicit_selected_calibration_check(admission):
    with pytest.raises(ValueError, match="calibration differs"):
        student._formal_admission(admission.args, SimpleNamespace(sha256="0" * 64))
