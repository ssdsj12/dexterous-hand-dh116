"""Check minimization and the original world-translation action mapping."""
import importlib

import numpy as np
import pytest


def api():
    return importlib.import_module("scripts.audit_dh116_tripod_wrist_seating")


def test_choose_smallest_motion_with_complete_opposed_geometry_and_original_depth_limit():
    offsets = np.array([[0., 0., 0.], [.0001, 0., 0.], [.00015, 0., 0.], [.0002, 0., 0.]])
    gaps = np.full((4, 2, 3), -.00004)
    gaps[0, 0, 1] = .00002
    gaps[1, 1, 0] = -.000251
    # All three pads touch, but the thumb cannot oppose both fingers.
    thumb = [1., 0., 0.]
    normals = np.broadcast_to([thumb, [-1., 0., 0.], [-1., 0., 0.]], (4, 2, 3, 3)).copy()
    normals[2, 1, 2] = thumb
    chosen, supported = api().choose_offset(offsets, gaps, normals, max_depth_m=.00025)
    assert chosen == 3
    np.testing.assert_array_equal(supported, [False, False, False, True])
    assert api().choose_offset(offsets[:3], gaps[:3], normals[:3], max_depth_m=.00025)[0] is None


def test_candidate_uses_left_world_xy_only_and_preserves_the_original_rate():
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT

    shift = np.array([.000075, -.0002, 0.])
    action = api().world_xy_dose(shift)
    assert action.shape == (2, 24)
    assert np.linalg.norm(action[:, 12:15], axis=1).max() <= 1.
    assert np.count_nonzero(action[:, :12]) == np.count_nonzero(action[:, 14:]) == 0
    np.testing.assert_allclose(action[:, 12:15].sum(axis=0) * ACTION_CONTRACT["translation_rate_m_s"] / 60,
                               shift, rtol=0., atol=1e-14)
    assert api().world_xy_dose(np.zeros(3)).shape == (0, 24)
    with pytest.raises(ValueError, match="World XY"):
        api().world_xy_dose([0., 0., .0001])


def test_extension_retains_exact_prefix_and_proposes_one_xy_dose_then_fixed_wrist_hold():
    mod = api()
    prefix = np.zeros((3, 24))
    prefix[1, 1] = .125
    offset = [.000025, -.000125, 0.]
    rows = mod.seating_schedule(prefix, prefix.tolist(), offset)
    np.testing.assert_array_equal(rows[:3], prefix)
    np.testing.assert_allclose(rows[3:4], mod.world_xy_dose(offset))
    assert rows.shape == (3 + 1 + 16, 24)
    assert not rows[4:].any()
    changed = prefix.copy()
    changed[0, 2] = .1
    with pytest.raises(ValueError, match="prefix"):
        mod.seating_schedule(changed, prefix.tolist(), offset)
