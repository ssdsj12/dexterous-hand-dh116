"""Synthetic RGB/geometry fixtures only; no rendered-depth deployment fallback."""
import importlib

import pytest
import torch


def module():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_stereo_perception')


def fixture():
    rgb = torch.zeros(1, 3, 32, 48)
    rgb[:, 0, 14:19, 4:44] = .7
    rgb[:, 1, 14:19, 4:44] = .4
    rgb[:, 2, 14:19, 4:44] = .1
    depth = torch.full((1, 32, 48), .5)
    valid = torch.ones_like(depth, dtype=torch.bool)
    k = torch.tensor([[100., 0., 24.], [0., 100., 16.], [0., 0., 1.]])
    return rgb, depth, valid, k


def test_sensor_only_skeleton_is_arc_sorted_and_resets_identity():
    m = module(); rgb, depth, valid, k = fixture()
    extractor = m.WirePointExtractor()
    a = extractor.extract(rgb, depth, valid, k, torch.eye(4), reset_mask=torch.tensor([False]))
    assert a['points_world'].shape == (1, 128, 3)
    count = int(a['point_valid'].sum())
    assert 20 < count < 45
    p = a['points_world'][0, a['point_valid'][0]]
    assert (p[1:, 0] > p[:-1, 0]).all()
    assert not a['temporal_identity_valid'].any()
    b = extractor.extract(rgb, depth, valid, k, torch.eye(4), reset_mask=torch.tensor([False]))
    assert b['temporal_identity_valid'].all()
    c = extractor.extract(rgb, depth, valid, k, torch.eye(4), reset_mask=torch.tensor([True]))
    assert not c['temporal_identity_valid'].any()


def test_invalid_depth_never_filled_by_known_geometry():
    rgb, depth, valid, k = fixture()
    valid[:] = False; depth[:] = float('nan')
    result = module().WirePointExtractor().extract(rgb, depth, valid, k, torch.eye(4))
    assert not result['point_valid'].any()
    assert torch.isnan(result['points_world']).all()


def test_non_wire_image_is_missing_not_ground_truth_substitution():
    rgb, depth, valid, k = fixture(); rgb[:] = .5
    result = module().WirePointExtractor().extract(rgb, depth, valid, k, torch.eye(4))
    assert not result['point_valid'].any()


def test_branch_does_not_become_fabricated_unbranched_centerline():
    rgb, depth, valid, k = fixture()
    rgb[:, 0, 2:16, 22:27] = .7; rgb[:, 1, 2:16, 22:27] = .4; rgb[:, 2, 2:16, 22:27] = .1
    result = module().WirePointExtractor().extract(rgb, depth, valid, k, torch.eye(4))
    assert not result['point_valid'].any()
    assert result['rejection_reason'] == ['ambiguous_or_branched_visible_skeleton']
