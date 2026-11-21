import numpy as np
import pytest


def test_archived_rod_tube_rejects_intermediate_intrusion_with_clear_end_samples():
    from scripts.screen_dh116_tripod_supported_reset import archived_tube, separation

    rod = np.zeros((3, 41, 3))
    rod[:, :, 1] = np.linspace(-.25, .25, 41)
    rod[:, :, 0] = np.array([.01, 0., .01])[:, None]
    shape = np.array([[x, y, z] for x in (-.001, .001) for y in (-.001, .001) for z in (-.001, .001)])
    assert separation(shape, *archived_tube(rod[[0, 2]])) > 0.
    assert separation(shape, *archived_tube(rod)) < 0.
    with pytest.raises(ValueError, match="41"):
        archived_tube(rod[:, :-1])


def test_tripod_scene_scope_uses_separate_seed_and_preserves_original_supports():
    from scripts.screen_dh116_tripod_supported_reset import scene_request, SCOPE
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import prepare_initial_state

    request, config = scene_request()
    table = {"center_m": [0., 0., .275], "size_m": [1., 1., .05]}
    assert config["scope"] == request["scope"] == SCOPE
    with pytest.raises(ValueError):
        prepare_initial_state(config, table)
    seed, active, nodes, supports = prepare_initial_state(config, table, expected_scope=SCOPE)
    assert active.shape == (2, 6) and nodes.shape == (41, 3)
    assert set(supports) == {"support_0", "support_1"}
    assert seed["reset_root_pose_w"][0][2] > seed["reset_root_pose_w"][1][2]-.03
