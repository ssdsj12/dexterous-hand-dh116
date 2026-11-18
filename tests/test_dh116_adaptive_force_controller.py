import numpy as np
import pytest

from go2_pvcnn.control.dh116.adaptive_force_controller import (
    AdaptiveForceController,
    allocate_preload,
)


def test_target_load_update_is_rate_limited_and_does_not_rewrite_measurements():
    controller = AdaptiveForceController(
        [1.0] * 6, max_loads_n=[1.4] * 6, rate_limit_n_s=0.5, preload_step_n=0.2
    )
    measured = np.array([0.2] * 6)
    result = controller.update(measured, 0.1, support_feasible=False)

    assert np.allclose(result["measured_loads_n"], measured)
    assert np.allclose(result["target_loads_n"], 1.05)
    assert result["replan_required"] is False
    assert controller.target_loads_n.shape == (6,)


def test_target_load_update_reports_replan_at_force_cap():
    controller = AdaptiveForceController(
        [1.3] * 6, max_loads_n=[1.3] * 6, rate_limit_n_s=5.0, preload_step_n=0.2
    )
    result = controller.update([1.0] * 6, 0.1, support_feasible=False)

    assert np.allclose(result["target_loads_n"], [1.3] * 6)
    assert result["replan_required"] is True
    assert result["reason"] == "support_insufficient_at_force_cap"


def test_invalid_measurement_or_dt_is_rejected():
    controller = AdaptiveForceController([1.0] * 6, max_loads_n=[2.0] * 6)
    with pytest.raises(ValueError):
        controller.update([1.0] * 5, 0.01)
    with pytest.raises(ValueError):
        controller.update([1.0] * 6, 0.0)


def test_allocate_preload_uses_mass_and_never_claims_grasp():
    points = np.array([[.02, .02, 0.], [-.02, .02, 0.],
                       [-.02, -.02, 0.], [.02, -.02, 0.]])
    normals = np.tile([0., 0., 1.], (4, 1))
    light = allocate_preload(points, normals, np.zeros(3), mass_kg=.1,
                             mu=.5, max_load_n=3.)
    heavy = allocate_preload(points, normals, np.zeros(3), mass_kg=.2,
                             mu=.5, max_load_n=3.)
    assert light['feasible'] and heavy['feasible']
    assert not light['grasp_qualified'] and not heavy['grasp_qualified']
    assert np.sum(heavy['target_loads_n']) > np.sum(light['target_loads_n'])
    capped = allocate_preload(points, normals, np.zeros(3), mass_kg=2.,
                              mu=.5, max_load_n=.5)
    assert not capped['feasible'] and capped['target_loads_n'] is None
