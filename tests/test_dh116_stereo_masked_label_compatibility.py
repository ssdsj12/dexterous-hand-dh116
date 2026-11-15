"""Masked label-ledger compatibility only; no native/source qualification."""
import pytest
import torch

from test_dh116_stereo_collection import FakeEnv, record
from go2_pvcnn.control.dh116.tabletop_stereo_supervision_dataset import (
    build_dataset, validate_dataset, batch_dataset,
)
from go2_pvcnn.control.dh116.tabletop_stereo_student_cli import _snapshot


def dataset():
    return build_dataset([record()], source_sha256="f" * 64, schema_only=True)


def test_unknown_371_labels_stay_nan_and_false_in_dataset():
    data = dataset()
    for row in data["rows"]:
        row["privileged"].fill_(float("nan"))
        row["privileged_valid"].zero_()
    checked = validate_dataset(data, allow_schema_only=True)
    assert all(torch.isnan(row["privileged"]).all() for row in checked["rows"])
    assert all(not row["privileged_valid"].any() for row in checked["rows"])
    assert checked["native_qualified"] is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_valid_nonfinite_label_is_still_rejected(value):
    data = dataset()
    data["rows"][0]["privileged"][0] = value
    data["rows"][0]["privileged_valid"][0] = True
    with pytest.raises(ValueError, match="finite|371D"):
        validate_dataset(data, allow_schema_only=True)


def test_pre_action_label_mask_allows_unknown_without_imputation():
    state = FakeEnv().snapshot()
    state.privileged.fill_(float("nan"))
    state.privileged_valid.zero_()
    state.validate()
    assert torch.isnan(state.privileged).all()
    assert not state.privileged_valid.any()


def test_existing_full_371_critic_guard_remains_closed_for_nan():
    state = FakeEnv().snapshot()
    state.privileged.fill_(float("nan"))
    state.privileged_valid.zero_()
    with pytest.raises(ValueError, match="finite unchanged native371"):
        _snapshot(state, "c" * 64)


def test_common66_and_unknown305_survive_batching_without_qualification():
    data = dataset()
    selected = list(range(24)) + list(range(78, 88)) + list(range(153, 185))
    mask = torch.zeros(371, dtype=torch.bool)
    mask[selected] = True
    for row in data["rows"]:
        row["privileged_valid"].copy_(mask)
        row["privileged"][~mask] = torch.nan
    before = [row["privileged"].clone() for row in data["rows"]]
    batch = batch_dataset(data, list(range(len(data["rows"]))))
    actual = batch["targets"]["privileged_state"]
    valid = batch["validity"]["privileged_state"]
    assert torch.equal(valid, mask.expand_as(valid))
    assert torch.isnan(actual[:, ~mask]).all()
    assert torch.equal(actual[:, mask], torch.stack(before)[:, mask])
    assert batch["native_qualified"] is False
    assert all(torch.isnan(row["privileged"][~mask]).all() for row in data["rows"])


def test_label_and_validity_devices_must_match_before_mask_operation():
    data = dataset()
    data["rows"][0]["privileged_valid"] = torch.ones(371, device="meta", dtype=torch.bool)
    with pytest.raises(ValueError, match="371D|device"):
        validate_dataset(data, allow_schema_only=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_pre_action_valid_nonfinite_label_is_still_rejected(value):
    state = FakeEnv().snapshot()
    state.privileged[0, 0] = value
    state.privileged_valid[0, 0] = True
    with pytest.raises(ValueError, match="finite|371D"):
        state.validate()
