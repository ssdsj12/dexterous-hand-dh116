from types import SimpleNamespace
import copy

import numpy as np
import pytest
import torch

from scripts import audit_dh116_wireseg_depth_error as audit


def sample(name, target=1., valid=True):
    return SimpleNamespace(file_name=name, scene_family="fixture", pair_sha256="a" * 64,
        rgb=np.full((32, 32, 3), 128, np.uint8),
        instance_masks=np.pad(np.ones((1, 16, 16), bool), ((0, 0), (8, 8), (8, 8))),
        depth_m=np.full((32, 32), target, np.float32),
        depth_valid=np.full((32, 32), valid, bool))


class Constant(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(1.2))

    def forward(self, rgb):
        assert not torch.is_grad_enabled()
        size = (rgb.shape[0], *rgb.shape[-2:])
        return {"metric_depth": self.value.expand(size), "visible": torch.zeros(size)}


def test_readonly_metrics_keep_roi_and_invalid_pixels():
    model = Constant()
    before = copy.deepcopy(model.state_dict())
    record, arrays = audit.inspect_sample(model, sample("valid"), "center", 32, 1.)
    assert record["depth"]["all_valid"]["valid_pixels"] == 1024
    assert record["depth"]["wire_roi"]["valid_pixels"] == 256
    assert record["depth"]["all_valid"]["mae_m"] == pytest.approx(.2)
    assert record["constant"]["wire_roi"]["mae_m"] == 0.
    assert record["signed_bias_m"] == pytest.approx(.2)
    assert record["linear_smooth_l1_fraction"] == 1.
    assert record["prediction_quantiles_m"]["50"] == pytest.approx(1.2)
    assert record["rgb_minmax"][0] == pytest.approx(128 / 255)
    assert torch.equal(model.value, before["value"]) and model.value.grad is None
    empty, _ = audit.inspect_sample(model, sample("empty", valid=False), "center", 32, 1.)
    assert empty["depth"]["all_valid"]["mae_m"] is None
    assert empty["loss"] is None and empty["signed_bias_m"] is None


def test_train_only_calibration_does_not_fit_validation():
    model = Constant()
    _, train = audit.inspect_sample(model, sample("train", 1.), "center", 32, 1.)
    _, heldout = audit.inspect_sample(model, sample("heldout", 2.), "center", 32, 1.)
    fitted = audit.fit_train_calibration([train])
    assert fitted["offset_m"] == pytest.approx(-.2)
    report = audit.summarize([heldout], baseline=1., calibration=fitted)
    assert report["offset_diagnostic"]["all_valid"]["mae_m"] == pytest.approx(1.)
    assert report["constant"]["all_valid"]["mae_m"] == pytest.approx(1.)
    assert report["depth"]["all_valid"]["mae_m"] == pytest.approx(.8)
    assert fitted["fit_split"] == "train"
    assert fitted["fit_pixels"] == 1024


def test_pixel_weighted_aggregation_is_not_mean_of_image_metrics():
    a = (np.array([2.]), np.array([1.]), np.array([True]), np.array([True]))
    b = (np.ones(9), np.ones(9), np.ones(9, bool), np.ones(9, bool))
    result = audit.summarize([a, b], baseline=1., calibration=None)
    assert result["depth"]["all_valid"]["mae_m"] == pytest.approx(.1)
    assert result["depth"]["all_valid"]["rmse_m"] == pytest.approx(np.sqrt(.1))


def test_partial_deadline_persists_only_finished_records(tmp_path):
    clock = iter([0., 0., 2.])
    completed = []
    result = audit.inspect_views(Constant(), [sample("one"), sample("two")], size=32, baseline=1.,
                                  deadline=1., clock=lambda: next(clock),
                                  publish=lambda record: completed.append(record))
    assert len(completed) == 2  # one sample, center and full field
    assert result["completed"] is False
    assert all(row["file_name"] == "one" for row in completed)


def test_detects_a_forward_that_mutates_model_weights():
    class Mutating(Constant):
        def forward(self, rgb):
            self.value.add_(1.)
            return super().forward(rgb)
    model = Mutating()
    before = copy.deepcopy(model.state_dict())
    audit.inspect_sample(model, sample("mutated"), "center", 32, 1.)
    with pytest.raises(ValueError, match="weights"):
        audit.verify_unchanged_weights(model, before)
