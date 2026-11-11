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
