import importlib
import numpy as np
import pytest


def module():
    return importlib.import_module("scripts.verify_dh116_deformx_rigid_bridge")


def test_wrench_com_shift_and_rotated_angular_momentum():
    probe = module()
    wrench = probe.wrench_at_com([4, 0, 0, 0, 0, 3], [0, 0, 0], [0, 0.25, 0])
    np.testing.assert_allclose(wrench, [4, 0, 0, 0, 0, 4])
    q = [2**-0.5, 0, 0, 2**-0.5]
    np.testing.assert_allclose(
        probe.angular_momentum(np.diag([2, 3, 4]), q, [1, 0, 0]),
        [3, 0, 0],
        atol=1e-12,
    )


def test_impulse_uses_delivered_wrench_with_one_step_delay():
    probe = module()
    rows = [
        dict(phase_step=1, sdk_step=11, dt_s=0.25, applied_wrench_com_w=[0]*6,
             object_wrench_com_w=[4, 0, 0, 0, 0, 2],
             delta_momentum_w=[0, 0, 0], delta_angular_momentum_com_w=[0, 0, 0]),
        dict(phase_step=2, sdk_step=12, dt_s=0.25, applied_wrench_com_w=[4, 0, 0, 0, 0, 2],
             object_wrench_com_w=[8, 0, 0, 0, 0, 2],
             delta_momentum_w=[1, 0, 0], delta_angular_momentum_com_w=[0, 0, 0.5]),
    ]
    result = probe.impulse_audit(rows, True)
    assert result["force_impulse_w"] == [1, 0, 0]
    assert result["max_linear_residual"] == result["max_angular_residual"] == 0
    with pytest.raises(ValueError, match="previous"):
        probe.impulse_audit(rows, False)
    rows[1]["sdk_step"] = 13
    with pytest.raises(ValueError, match="clock"):
        probe.impulse_audit(rows, True)


def test_owned_composer_refresh_precedes_world_wrench_conversion():
    from types import SimpleNamespace
    probe = module()
    calls = []
    composer = SimpleNamespace(reset=lambda: calls.append("refresh link pose"))
    bridge = SimpleNamespace(apply_wrenches=lambda: calls.append("convert and stage"))
    probe.stage_owned_wrench(bridge, composer)
    assert calls == ["refresh link pose", "convert and stage"]
