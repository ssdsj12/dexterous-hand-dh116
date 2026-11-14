import numpy as np
import pytest

from scripts.audit_dh116_pinch_v3_evidence import residual_precision


def test_source_fp32_and_independent_fp64_may_differ_without_force_imbalance():
    nodes = np.zeros((41,3),dtype=np.float32)
    nodes[:,2] = .008
    meshes = np.zeros((6,6))
    meshes[-1,2] = -float(nodes.astype(float).sum(axis=0)[2])
    reported = float(np.linalg.norm(nodes.sum((0,)) + meshes[:,:3].sum(0)))
    result = residual_precision(nodes,meshes,reported)
    assert result["matches_producer"] and not result["matches_legacy_fp64"]
    assert result["independent_balance_passed"]


def test_nonfinite_and_large_imbalance_are_not_accepted():
    nodes = np.zeros((41,3))
    meshes = np.zeros((6,6))
    nodes[0,2] = .01
    result = residual_precision(nodes,meshes,.01)
    assert result["matches_producer"] and not result["independent_balance_passed"]
    nodes[0,2] = np.nan
    with pytest.raises(ValueError,match="finite"):
        residual_precision(nodes,meshes,0.)
