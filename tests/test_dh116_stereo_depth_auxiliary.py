import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_depth_auxiliary import (
    StereoDepthAuxiliary,
    depth_auxiliary_loss,
    native_depth_targets,
)
from go2_pvcnn.control.dh116.tabletop_stereo_state_estimator import StereoStateEstimator


def sample():
    shape = (1, 33, 65)
    sha = "a" * 64
    frame = {
        "frame_id": 4,
        "simulation_time": 0.25,
        "calibration_sha256": sha,
        "left_rgb": torch.rand(1, 3, 33, 65, requires_grad=True),
        "right_rgb": torch.rand(1, 3, 33, 65, requires_grad=True),
    }
    reference = {
        "frame_id": 4,
        "simulation_time": 0.25,
        "calibration_sha256": sha,
        "source_scope": "native",
        "source_units": "optical_axis_z_m",
        "validation_only": True,
        "reference_used_as_actor_input": False,
        "depth_m": torch.full(shape, 0.4),
        "valid": torch.ones(shape, dtype=torch.bool),
        "right_depth_m": torch.full(shape, 0.5),
        "right_valid": torch.ones(shape, dtype=torch.bool),
    }
    return frame, reference


def prediction(model, frame, image_valid=((True, True),)):
    return model(frame["left_rgb"], frame["right_rgb"], torch.tensor(image_valid))


def test_left_reference_trains_both_rgb_streams_without_supervising_right_output():
    torch.manual_seed(12)
    estimator = StereoStateEstimator()
    old_keys = tuple(estimator.state_dict())
    model = StereoDepthAuxiliary(estimator.encoder, min_depth_m=0.2, max_depth_m=1.2)
    frame, reference = sample()
    reference["valid"][0, 0, 0] = False
    reference["depth_m"][0, 0, 0] = float("nan")
    output = prediction(model, frame)
    assert model.encoder is estimator.encoder
    assert tuple(estimator.state_dict()) == old_keys
    assert output.depth_m.shape == (1, 2, 2, 3)
    assert output.y_px.tolist() == [0, 32]
    assert output.x_px.tolist() == [0, 32, 64]
    assert output.valid.all()
    assert torch.all((output.depth_m > 0.2) & (output.depth_m < 1.2))

    targets = native_depth_targets(frame, reference)
    assert targets.valid.sum(dim=(0, 2, 3)).tolist() == [5, 0]
    assert torch.isnan(targets.depth_m[:, 1]).all()
    loss = depth_auxiliary_loss(output, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.project_depth[0].weight.grad.abs().sum() > 0
    assert torch.equal(model.project_depth[1].weight.grad, torch.zeros_like(model.project_depth[1].weight))
    assert next(model.encoder.backbone.parameters()).grad.abs().sum() > 0
    assert frame["left_rgb"].grad.abs().sum() > 0
    assert frame["right_rgb"].grad.abs().sum() > 0


@pytest.mark.parametrize("field,value", [
    ("simulation_time", 0.24),
    ("calibration_sha256", "0" * 64),
    ("source_scope", "synthetic"),
    ("source_units", "range_m"),
    ("reference_used_as_actor_input", True),
])
def test_reference_must_be_aligned_native_label_only(field, value):
    frame, reference = sample()
    reference[field] = value
    if field == "calibration_sha256":
        frame[field] = value
    with pytest.raises(ValueError):
        native_depth_targets(frame, reference)


def test_missing_synchronized_view_does_not_train_and_rejects_empty_mask():
    estimator = StereoStateEstimator()
    model = StereoDepthAuxiliary(estimator.encoder, min_depth_m=0.2, max_depth_m=1.2)
    frame, reference = sample()
    output = prediction(model, frame, ((True, False),))
    assert not output.valid.any()
    with pytest.raises(ValueError, match="no supervised"):
        depth_auxiliary_loss(output, native_depth_targets(frame, reference))

    reference["valid"].zero_()
    with pytest.raises(ValueError, match="no supervised"):
        depth_auxiliary_loss(prediction(model, frame), native_depth_targets(frame, reference))


def test_independently_resolved_right_source_and_calibrated_range_are_enforced():
    frame, reference = sample()
    frame["identity"] = [{"env_id": 0}]
    shape = reference["depth_m"].shape
    wire = torch.ones(shape, dtype=torch.bool)
    ids = wire.to(dtype=torch.int64)
    info = {"idToLabels": {"1": "/World/envs/env_0/Wire"}}
    reference.update({
        "wire_mask_source_scope": "native_resolved_instance_id",
        "wire_instance_resolution": {"wire_inventory_complete": True},
        "raw_instance_ids": ids,
        "raw_instance_info": info,
        "wire_reference_mask": wire,
        "right_raw_instance_ids": ids,
        "right_raw_instance_info": info,
        "right_wire_reference_mask": wire,
        "right_wire_instance_resolution": {"wire_inventory_complete": True},
    })
    reference["right_depth_m"][0, 0, 0] = 1.4
    target = native_depth_targets(frame, reference, min_depth_m=0.2, max_depth_m=1.2)
    assert target.valid.sum(dim=(0, 2, 3)).tolist() == [6, 5]
    assert torch.isnan(target.depth_m[0, 1, 0, 0])
    assert target.depth_m[0, 1, 0, 1] == 0.5
    reference["right_wire_instance_resolution"]["wire_inventory_complete"] = False
    with pytest.raises(ValueError, match="independently resolved"):
        native_depth_targets(frame, reference)


def test_cpu_replay_uses_explicit_bounded_updates_and_keeps_unlabelled_head_fixed(tmp_path, monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace

    from scripts import diagnose_dh116_stereo_depth_auxiliary as diagnostic

    frame, reference = sample()
    frame.update(native_evidence=True, render_passes=1, identity=[{"env_id": 0}],
                 depth_m=torch.full((1, 33, 65), float("nan")),
                 valid=torch.zeros((1, 33, 65), dtype=torch.bool))
    proof = tmp_path / "proof.json"
    proof.write_text("{}")
    digest = hashlib.sha256(proof.read_bytes()).hexdigest()
    proof_metadata = {
        "capture_artifact": {"sha256": "b" * 64},
        "reference_artifacts": [{"sha256": "c" * 64}],
        "cohort_sha256": "d" * 64,
    }
    monkeypatch.setattr(diagnostic, "load_capture", lambda _: (
        {"frames": [frame]}, [reference], SimpleNamespace(sha256=frame["calibration_sha256"]),
        {"min_depth_m": 0.2, "max_depth_m": 1.2}, proof_metadata,
    ))
    common = ["--proof", str(proof), "--proof-sha256", digest]
    default_path = tmp_path / "default.json"
    assert diagnostic.main([*common, "--output", str(default_path)]) == 0
    default = json.loads(default_path.read_bytes())
    assert default["optimizer_updates"] == 0 and "optimization" not in default

    fit_path = tmp_path / "fit.json"
    assert diagnostic.main([*common, "--output", str(fit_path), "--optimizer-steps", "2"]) == 0
    fit = json.loads(fit_path.read_bytes())
    assert fit["optimizer_updates"] == 2
    assert fit["checkpoint_saved"] is False and fit["deployment_qualified"] is False
    assert fit["reference_used_as_input"] is False and fit["native_physics_steps"] == 0
    assert fit["environments"][0]["left"]["supervised_samples"] == 6
    assert fit["environments"][0]["right"]["supervised_samples"] == 0
    result = fit["optimization"]
    assert result["scope"] == "same_archived_frame_optimizer_diagnostic"
    assert len(result["same_frame_loss_m"]) == 3
    assert result["same_frame_loss_m"][0] == pytest.approx(default["loss_m"])
    assert result["same_frame_loss_m"][-1] < result["same_frame_loss_m"][0]
    assert result["head_parameter_max_abs_change"]["left"] > 0
    assert result["head_parameter_max_abs_change"]["right"] == 0
    assert result["shared_backbone_first_parameter_max_abs_change"] > 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["default.json", "fit.json", "proof.json"]


def test_cpu_replay_rejects_over_budget_before_loading_proof(monkeypatch, tmp_path, capsys):
    from scripts import diagnose_dh116_stereo_depth_auxiliary as diagnostic

    def invalid_lookup(_):
        pytest.fail("over-budget CLI must fail before accessing proof")

    monkeypatch.setattr(diagnostic, "file_sha256", invalid_lookup)
    with pytest.raises(SystemExit) as error:
        diagnostic.main(["--proof", str(tmp_path/"unused.json"), "--proof-sha256", "a"*64,
                         "--output", str(tmp_path/"no.json"), "--optimizer-steps", "9"])
    assert error.value.code == 2
    assert "optimizer steps must be between 0 and 8" in capsys.readouterr().err
