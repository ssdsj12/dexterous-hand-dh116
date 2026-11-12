import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def module():
    return importlib.import_module("scripts.diagnose_dh116_wireseg_overfit")


def test_crop_baseline_is_identical_and_empty_masks_are_not_dropped():
    from scripts.pretrain_dh116_wireseg import training_sample
    sample = SimpleNamespace(rgb=np.zeros((64, 64, 3), dtype=np.uint8),
        instance_masks=np.zeros((0, 64, 64), dtype=bool),
        depth_m=np.ones((64, 64), dtype=np.float32),
        depth_valid=np.ones((64, 64), dtype=bool))
    actual = module().make_view(sample, "center", 32, "cpu")
    baseline = training_sample(sample, 32, "cpu")
    assert all(torch.equal(a, b) for a, b in zip(actual, baseline))
    assert actual[1].shape == (0, 32, 32)


def test_full_field_keeps_wire_outside_center_and_depth_units():
    sample = SimpleNamespace(rgb=np.zeros((64, 64, 3), dtype=np.uint8),
        instance_masks=np.zeros((1, 64, 64), dtype=bool),
        depth_m=np.full((64, 64), 2.5, dtype=np.float32),
        depth_valid=np.ones((64, 64), dtype=bool))
    sample.instance_masks[0, :12, :12] = True
    center = module().make_view(sample, "center", 32, "cpu")
    resized = module().make_view(sample, "full_field_resize", 32, "cpu")
    assert not center[1].any() and resized[1].any()
    assert torch.all(resized[2] == 2.5)


def test_depth_reports_roi_separately_excluding_invalid_labels():
    prediction = np.array([1., 2., 7., np.nan])
    truth = np.array([2., 2., 2., np.nan])
    valid = np.array([True, True, True, False])
    roi = np.array([True, True, False, True])
    result = module().depth_metrics(prediction, truth, valid, roi)
    assert result["all_valid"]["mae_m"] == 2.
    assert result["wire_roi"]["mae_m"] == .5
    assert result["wire_roi"]["valid_pixels"] == 2
    with pytest.raises(ValueError, match="nonfinite"):
        module().depth_metrics(np.array([np.nan]), np.ones(1), np.ones(1, bool), np.ones(1, bool))


def test_seg_and_depth_losses_do_not_silently_train_other_head():
    from go2_pvcnn.control.dh116.tabletop_wireseg_pretraining import WireSegPretrainer
    net = WireSegPretrainer(embed_dim=32, max_instances=4)
    rgb = torch.rand((1, 3, 32, 32))
    masks = torch.zeros((0, 32, 32), dtype=torch.bool)
    depth, valid = torch.ones((1, 32, 32)), torch.ones((1, 32, 32), dtype=torch.bool)
    module().diagnostic_loss(net(rgb), masks, depth, valid, "seg-only").backward()
    assert net.depth_head.weight.grad is None
    assert net.visible_head.weight.grad is not None
    net.zero_grad(set_to_none=True)
    module().diagnostic_loss(net(rgb), masks, depth, valid, "depth-only").backward()
    assert net.visible_head.weight.grad is None
    assert net.depth_head.weight.grad.abs().sum() > 0


def test_fixed_cohort_never_filters_empty_crop_and_rejects_duplicate():
    cohort = [{"file_name": str(i), "pair_sha256": str(i)} for i in range(8)]
    assert module().validate_fixed_cohort(cohort) == cohort
    with pytest.raises(ValueError, match="unique"):
        module().validate_fixed_cohort(cohort[:-1] + [cohort[0]])
    with pytest.raises(ValueError, match="8"):
        module().validate_fixed_cohort(cohort[:7])


def test_constant_depth_is_fitted_on_training_only():
    train = [np.array([1., 2., 3.]), np.array([2., 8.])]
    assert module().constant_depth(train) == 2.
    with pytest.raises(ValueError, match="valid"):
        module().constant_depth([np.array([np.nan, 0.])])



def test_missing_depth_still_evaluated_but_never_fabricates_an_update():
    rgb = torch.zeros((1, 3, 32, 32))
    empty_mask = torch.zeros((0, 32, 32), dtype=torch.bool)
    missing = (rgb, empty_mask, torch.zeros((1, 32, 32)), torch.zeros((1, 32, 32), dtype=torch.bool))
    assert module().has_supervision(missing, "seg-only")
    assert not module().has_supervision(missing, "depth-only")
