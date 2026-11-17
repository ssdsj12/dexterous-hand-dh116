"""Regression for the original physical threshold and FP32 producer identity."""
from copy import deepcopy
import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_deformable_pinch_force_precision import validate_ledger


def row(force=.008):
    nodes = np.zeros((41,3),dtype=np.float32)
    nodes[:,2] = force
    wrench = {name:[0.,0.,0.,0.,0.,0.] for name in (
        "finger14_link","finger23_link","finger33_link","finger43_link","finger53_link","table")}
    wrench["table"][2] = -float(nodes.astype(float).sum(axis=0)[2])
    reported = float(np.linalg.norm(nodes.sum(axis=0) + [0,0,wrench["table"][2]]))
    return dict(rod_node_contact_forces_w=nodes.tolist(), rod_mesh_wrenches_w=wrench,
                pair_force_residual_n=reported)


def test_producer_matches_original_precision_and_validator_gets_owned_canonical_copy(monkeypatch):
    import go2_pvcnn.control.dh116.tabletop_deformable_pinch_force_precision as module
    data = row()
    original = deepcopy(data)
    def prior(rows):
        assert rows[0]["pair_force_residual_n"] == 0.
        rows[0]["rod_node_contact_forces_w"][0][2] = 100
        return {"ledger_validation_passed":True}
    monkeypatch.setattr(module,"validate_pinch_ledger",prior)
    result = validate_ledger([data])
    assert result["ledger_validation_passed"] is True
    assert result["max_independent_force_residual_n"] == 0.
    assert data == original


@pytest.mark.parametrize("fault", ["source", "imbalance", "shape", "nonfinite"])
def test_rejects_forged_source_and_actual_imbalance_before_legacy_validation(monkeypatch, fault):
    import go2_pvcnn.control.dh116.tabletop_deformable_pinch_force_precision as module
    data = row()
    if fault == "source":
        data["pair_force_residual_n"] += 1e-5
    elif fault == "imbalance":
        data["rod_mesh_wrenches_w"]["table"][2] = 0.
        data["pair_force_residual_n"] = float(np.linalg.norm(np.asarray(
            data["rod_node_contact_forces_w"],dtype=np.float32).sum(axis=0)))
    elif fault == "shape":
        data["rod_node_contact_forces_w"].pop()
    else:
        data["rod_node_contact_forces_w"][0][2] = float("nan")
    monkeypatch.setattr(module,"validate_pinch_ledger",lambda _:pytest.fail("invalid ledger reached old validator"))
    with pytest.raises(ValueError,match="force"):
        validate_ledger([data])
