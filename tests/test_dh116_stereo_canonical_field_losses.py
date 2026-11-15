"""Canonical field-loss mathematics; CPU masks are not native evidence."""
import math

import pytest
import torch
from torch.nn import functional as F

from go2_pvcnn.control.dh116.tabletop_stereo_losses import StereoEstimationLoss
from go2_pvcnn.control.dh116.tabletop_stereo_state_supervision import _prediction_loss

CLASSIFICATION = (
    149, 150, 151, 152, 255, 256, 257, 258, 303, 304,
    309, 310, 311, 312, 313, 314, 315, 316, 317, 318,
)


def sources_for(criterion):
    sources = {
        key: torch.ones(1, dtype=torch.bool)
        for key in criterion.required
    }
    sources["rotation_symmetry_valid"] = torch.ones(1, dtype=torch.bool)
    return sources


def evaluate(criterion, prediction, target, mask, sources):
    return criterion(
        {"privileged_state": prediction},
        {"privileged_state": target},
        {"privileged_state": mask, "state_sources": sources},
    )


@pytest.mark.parametrize("index", CLASSIFICATION)
@pytest.mark.parametrize("label", [0., 1.])
def test_every_canonical_classification_field_uses_bce(index, label):
    criterion = StereoEstimationLoss()
    assert criterion.state_allowed[index]
    prediction = torch.zeros(1, 371, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[0, index] = label
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[0, index] = True

    result = evaluate(
        criterion, prediction, target, mask, sources_for(criterion),
    )
    independent, _ = _prediction_loss(
        {"privileged_state": prediction},
        {"privileged_state": (target, mask)},
        criterion,
    )

    assert float(result["total"]) == pytest.approx(math.log(2.))
    torch.testing.assert_close(result["total"], independent)
    result["total"].backward()
    assert float(prediction.grad[0, index]) == pytest.approx(.5 - label)
    other = prediction.grad.detach().clone()
    other[0, index] = 0
    assert torch.count_nonzero(other) == 0


def test_unknown_nan_fields_have_zero_loss_and_gradient():
    criterion = StereoEstimationLoss()
    prediction = torch.full((1, 371), float("nan"), requires_grad=True)
    target = torch.full_like(prediction, float("nan"))
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[0, 149] = True
    sources = sources_for(criterion)
    sources["source_valid"].zero_()

    result = evaluate(criterion, prediction, target, mask, sources)
    assert float(result["total"]) == 0.
    assert result["coverage"]["privileged_state"] == 0.
    result["total"].backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) == 0


def test_regression_field_keeps_smooth_l1_scale():
    criterion = StereoEstimationLoss()
    prediction = torch.zeros(1, 371, requires_grad=True)
    target = torch.zeros_like(prediction)
    prediction.data[0, 116] = .2
    target[0, 116] = .8
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[0, 116] = True
    result = evaluate(
        criterion, prediction, target, mask, sources_for(criterion),
    )
    expected = F.smooth_l1_loss(
        prediction[:, 116], target[:, 116],
    )
    torch.testing.assert_close(result["total"], expected)


def test_shared_quaternion_loss_remains_sign_invariant():
    criterion = StereoEstimationLoss()
    lo, hi = criterion.rotations[0]["slice"]
    prediction = torch.zeros(1, 371, requires_grad=True)
    prediction.data[0, lo:hi] = torch.tensor([1., .2, 0., 0.])
    target = torch.zeros_like(prediction)
    target[0, lo] = 1.
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[0, lo:hi] = True

    positive = criterion.canonical_state_loss(prediction, target, mask)
    negative = criterion.canonical_state_loss(prediction, -target, mask)
    torch.testing.assert_close(positive, negative)
    positive.backward()
    assert torch.isfinite(prediction.grad).all()


def test_new_loss_semantics_has_explicit_version():
    assert (
        StereoEstimationLoss().loss_version
        == "dh116-stereo-canonical-field-loss-v2"
    )

def test_classification_partition_is_exact():
    criterion = StereoEstimationLoss()
    assert tuple(criterion.state_classification.nonzero().flatten().tolist()) == CLASSIFICATION


@pytest.mark.parametrize("case", ["prediction_shape", "target_shape", "target_integer", "mask_shape", "mask_float"])
def test_shared_loss_validates_canonical_pairs(case):
    criterion = StereoEstimationLoss()
    prediction = torch.zeros(1, 371)
    target = prediction.clone()
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    if case == "prediction_shape":
        prediction = prediction[:, :370]
    elif case == "target_shape":
        target = target[:, :370]
    elif case == "target_integer":
        target = target.long()
    elif case == "mask_shape":
        mask = mask[:, :370]
    elif case == "mask_float":
        mask = mask.float()
    with pytest.raises(ValueError):
        criterion.canonical_state_loss(prediction, target, mask)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1., 2.])
def test_shared_loss_rejects_bad_valid_classification_label(bad):
    criterion = StereoEstimationLoss()
    prediction = torch.zeros(1, 371)
    target = prediction.clone()
    target[0, 149] = bad
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[0, 149] = True
    with pytest.raises(ValueError):
        criterion.canonical_state_loss(prediction, target, mask)


def test_shared_loss_ignores_incomplete_quaternion_group():
    criterion = StereoEstimationLoss()
    lo, hi = criterion.rotations[0]["slice"]
    prediction = torch.zeros(1, 371, requires_grad=True)
    target = torch.full_like(prediction, float("nan"))
    target[0, lo] = 1.
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[0, lo] = True
    loss = criterion.canonical_state_loss(prediction, target, mask)
    assert float(loss) == 0.
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) == 0
    assert torch.isnan(target[0, lo+1:hi]).all()


def test_shared_loss_matches_standalone_mixed_terms_and_residual_metrics():
    criterion = StereoEstimationLoss()
    prediction = torch.zeros(1, 371, requires_grad=True)
    target = torch.full_like(prediction, float("nan"))
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    lo, hi = criterion.rotations[0]["slice"]
    with torch.no_grad():
        prediction[0, lo:hi] = torch.tensor([1., .2, 0., 0.])
        prediction[0, 116] = .2
    target[0, lo:hi] = torch.tensor([-1., 0., 0., 0.])
    target[0, 116] = .8
    target[0, 149] = 1.
    mask[0, lo:hi] = True
    mask[0, [116, 149]] = True
    shared = criterion.canonical_state_loss(prediction, target, mask)
    standalone, errors = _prediction_loss({"privileged_state": prediction},
        {"privileged_state": (target, mask)}, criterion)
    torch.testing.assert_close(standalone, shared)
    assert float(shared) == pytest.approx(.18 + math.log(2.) + 2*math.atan(.2))
    assert float(errors["privileged_state"][0, 149]) == pytest.approx(.5)
    assert torch.isnan(errors["privileged_state"][0, lo:hi]).all()
    assert float(errors["rotation_rad"][0, 0]) == pytest.approx(2*math.atan(.2))
    a, = torch.autograd.grad(shared, prediction, retain_graph=True)
    b, = torch.autograd.grad(standalone, prediction)
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("symmetry", [None, False])
def test_unqualified_rotation_symmetry_remains_unknown(symmetry):
    criterion = StereoEstimationLoss()
    group = next(g for g in criterion.rotations if g.get("symmetry_mask_required"))
    lo, hi = group["slice"]
    prediction = torch.full((1, 371), float("nan"), requires_grad=True)
    target = torch.full_like(prediction, float("nan"))
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[0, lo:hi] = True
    sources = sources_for(criterion)
    if symmetry is None:
        sources.pop("rotation_symmetry_valid")
    else:
        sources["rotation_symmetry_valid"].zero_()
    result = evaluate(criterion, prediction, target, mask, sources)
    assert float(result["total"]) == 0.
    assert result["coverage"]["privileged_state"] == 0.
    result["total"].backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) == 0
    assert torch.isnan(target).all()


def test_na_fields_stay_unknown_and_additional_heads_keep_their_math():
    criterion = StereoEstimationLoss()
    prediction = torch.full((1, 371), float("nan"), requires_grad=True)
    target = torch.full_like(prediction, float("nan"))
    mask = ~criterion.state_allowed[None]
    logits = torch.zeros(1, 2, requires_grad=True)
    geometry = torch.ones(1, 1, 3, requires_grad=True)
    result = criterion(
        {"privileged_state": prediction, "contact_logits": logits, "geometry": geometry},
        {"privileged_state": target, "contact_logits": torch.ones_like(logits), "geometry": torch.zeros_like(geometry)},
        {"privileged_state": mask, "state_sources": sources_for(criterion),
         "contact_logits": torch.ones_like(logits, dtype=torch.bool), "geometry": torch.ones_like(geometry, dtype=torch.bool)})
    assert float(result["privileged_state"]) == 0.
    assert float(result["total"]) == pytest.approx(math.log(2.)+.5)
    result["total"].backward()
    assert torch.isfinite(prediction.grad).all() and torch.count_nonzero(prediction.grad) == 0
    assert torch.isnan(target).all()


def test_missing_force_proof_stays_blocked_with_canonical_classification():
    criterion = StereoEstimationLoss()
    prediction = torch.zeros(1, 371, requires_grad=True)
    target = prediction.detach().clone()
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[0, 149] = True
    force = torch.ones(1, 2, 3, requires_grad=True)
    result = criterion({"privileged_state": prediction, "force_proxy": force},
        {"privileged_state": target, "force_proxy": torch.full_like(force, float("nan"))},
        {"privileged_state": mask, "state_sources": sources_for(criterion),
         "force_proxy": torch.ones_like(force, dtype=torch.bool)})
    assert result["blocked_reasons"]["force_proxy"] == "missing_force_source_evidence"
    assert result["coverage"]["force_proxy"] == 0.
    assert float(result["total"]) == pytest.approx(math.log(2.))
    result["total"].backward()
    assert torch.count_nonzero(force.grad) == 0


def test_bc_report_and_contract_record_loss_version(tmp_path):
    import json
    from go2_pvcnn.control.dh116 import tabletop_stereo_training_cli as cli
    from test_dh116_stereo_training_cli import dataset
    network = dict(embed_dim=32, history=4, num_heads=4, num_layers=1, dropout=0.,
        image_resolution_wh=[32, 32], image_processing_batch_size=2)
    cli.train_bc(dataset(), network, {}, output=tmp_path, steps=1, batch_size=1,
        heldout_initial_sha256=["b"*64], schema_only=True)
    contract = json.loads((tmp_path/"bc_contract.json").read_text())
    updates = [json.loads(line) for line in (tmp_path/"bc_updates.jsonl").read_text().splitlines()]
    checkpoint = torch.load(tmp_path/"bc_update_000001.pt", map_location="cpu", weights_only=False)
    assert contract["config"]["state_loss_version"] == "dh116-stereo-canonical-field-loss-v2"
    assert updates[0]["state_loss_version"] == contract["config"]["state_loss_version"]
    assert checkpoint["contract"]["config"]["state_loss_version"] == updates[0]["state_loss_version"]


def test_state_schema_report_records_loss_version_without_training():
    from go2_pvcnn.control.dh116 import tabletop_stereo_state_supervision as supervision
    from test_dh116_stereo_state_training_modes import center_rows, estimator
    report = supervision.run_training_mode(estimator(), center_rows(), mode="schema_test")
    assert report["state_loss_version"] == "dh116-stereo-canonical-field-loss-v2"
    assert report["state_training_started"] is False


def test_state_report_and_new_checkpoint_record_loss_version(tmp_path, monkeypatch):
    import json
    from go2_pvcnn.control.dh116 import tabletop_stereo_state_supervision as supervision
    from test_dh116_stereo_state_training_modes import center_rows
    rows = center_rows()
    # Only native file admission is replaced by explicit CPU tensor fixtures.
    monkeypatch.setattr(supervision, "read_candidate", lambda *a, **kw: rows)
    estimator_type = supervision.StereoStateEstimator
    monkeypatch.setattr(supervision, "StereoStateEstimator", lambda **kw:
        estimator_type(embed_dim=32, num_layers=1, **kw))
    output = tmp_path/"new-state-fit"
    code = supervision.main(["--candidate", str(tmp_path/"fixture-candidate.pt"),
        "--candidate-sha256", "a"*64, "--raw", str(tmp_path/"fixture-raw.jsonl"),
        "--raw-sha256", "b"*64, "--output", str(output), "--mode", "diagnostic_fit",
        "--epochs", "1", "--batch-size", "2", "--device", "cpu"])
    assert code == 0
    report = json.loads((output/"state_supervision_report.json").read_text())
    checkpoint = torch.load(output/"state_estimator.pt", map_location="cpu", weights_only=True)
    assert report["state_loss_version"] == "dh116-stereo-canonical-field-loss-v2"
    assert checkpoint["state_loss_version"] == report["state_loss_version"]
    assert report["generalization_qualified"] is False
    assert checkpoint["actor_deployment_qualified"] is False
