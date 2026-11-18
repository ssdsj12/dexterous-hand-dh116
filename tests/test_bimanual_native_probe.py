import torch
import pytest

from go2_pvcnn.control.dh116.tabletop_bimanual_native_probe import perturbations


def test_each_sdk_dimension_has_independent_positive_and_negative_probe():
    rows = perturbations(8, 'cpu')
    assert len(rows) == 53
    assert torch.equal(rows[0][1], torch.zeros(8, 24))
    for dim in range(24):
        positive, negative = rows[1+2*dim][1], rows[2+2*dim][1]
        assert torch.equal(positive, -negative)
        assert (positive != 0).sum().item() == 8
        assert positive[:, dim].eq(.03).all()
    assert all(action.abs().max() <= .03 for _, action in rows)


def test_invalid_probe_size_is_rejected():
    with pytest.raises(ValueError):
        perturbations(0, 'cpu')
