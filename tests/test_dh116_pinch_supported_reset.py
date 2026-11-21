import numpy as np
import pytest


def test_pinch_material_alignment_keeps_transformed_pair_midpoint_and_tangent():
    from scripts.screen_dh116_pinch_supported_reset import material_transfer

    old = np.array([[0., 0., 0.], [0., 1., 0.]])
    new = np.array([[0., 1., 2.], [0., 1., 3.]])
    rotation, center_old, center_new = material_transfer(old,new)
    np.testing.assert_allclose(rotation.apply(old-center_old)+center_new,new,atol=1e-12)
    with pytest.raises(ValueError):
        material_transfer(old,np.zeros((2,3)))


def test_supported_pinch_seed_cannot_enter_default_power_reset():
    from scripts.verify_dh116_pinch_supported_reset import REQUEST
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import prepare_initial_state
    import json

    raw = json.loads(REQUEST.read_bytes())
    config = raw["config"]
    table = {"center_m":[.65,0.,.275], "size_m":[.8,1.,.05]}
    with pytest.raises(ValueError):
        prepare_initial_state(config,table)
    seed, active, rod, poses = prepare_initial_state(config,table,expected_scope=config["scope"])
    assert active.shape == (2,6) and rod.shape == (41,3) and set(poses)=={"support_0","support_1"}
