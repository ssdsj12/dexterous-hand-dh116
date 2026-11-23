import numpy as np
import pytest

from scripts.audit_dh116_tripod_reset_continuity import array_agreement, window_displacements


def test_reset_comparison_detects_dtype_and_exact_value_changes():
    old = np.zeros((1, 41, 3), dtype=np.float32)
    same = array_agreement(old, old.copy())
    assert same == {"byte_equal": True, "max_abs_difference": 0.0}
    changed = old.copy()
    changed[0, 12, 0] = 0.00025
    assert array_agreement(old, changed)["max_abs_difference"] == pytest.approx(0.00025)
    assert array_agreement(old, changed)["byte_equal"] is False
    assert array_agreement(old, old.astype(np.float64))["byte_equal"] is False
    with pytest.raises(ValueError, match="matching shape"):
        array_agreement(old, old[0])


def test_window_motion_uses_all_eight_exchanges_even_if_endpoint_returns():
    positions = np.zeros((17, 41, 3))
    positions[4, 2, 1] = 0.001
    positions[16, 35, 2] = 0.003
    metrics = window_displacements(positions)
    np.testing.assert_allclose(metrics[:, [2, 35]], [[0.001, 0], [0, 0.003]])
    with pytest.raises(ValueError, match="complete eight-step windows"):
        window_displacements(positions[:-1])
