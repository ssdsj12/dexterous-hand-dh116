"""Exercise measured geometry, radius and contact gating for the approach pulse."""
import importlib

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def api():
    name = "go2_pvcnn.control.dh116.tabletop_wire_live_palm_approach"
    assert importlib.util.find_spec(name), "measured wire approach is missing"
    return importlib.import_module(name)


def test_radius_changes_the_sealed_envelope_without_moving_centerline():
    nodes = np.zeros((1, 41, 3))
    nodes[0, :, 1] = np.linspace(-.25, .25, 41)
    a, small = api().wire_tube(nodes, radius_m=.0045)
    b, large = api().wire_tube(nodes, radius_m=.00525)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(large - small, .00075, atol=1e-12, rtol=0.)
    with pytest.raises(ValueError, match="radius"):
        api().wire_tube(nodes, radius_m=True)


def _sample():
    pose = Rotation.from_euler("z", 90, degrees=True)
    actual = dict(left=[.014, .010, .03], right=[-.040, -.020, .01])
    observation = dict(clock=dict(physics_step=10), reaction_valid=True,
        measured_hands={side: dict(xyz_in_palm_m=pose.inv().apply(vec),
                                   tangent_in_palm=pose.inv().apply([0., 1., 0.]),
                                   reaction={"thumb": {"mean_normal_n": 0.}})
                        for side, vec in actual.items()})
    geometry = dict(physics_step=10, poses={f"{side}.base_link": ([0., 0., .4], pose.as_quat())
                                           for side in actual})
    return observation, geometry


def test_world_approach_projects_out_axis_and_uses_both_measured_palms():
    observation, geometry = _sample()
    action = api().choose_pulse(observation, geometry)
    expected = np.zeros(24)
    expected[12], expected[15] = .3, -.3
    np.testing.assert_allclose(action, expected, atol=1e-13)
    with pytest.raises(ValueError, match="same SDK tick"):
        api().choose_pulse(observation, dict(geometry, physics_step=9))


def test_force_or_untrusted_contact_holds_both_targets():
    observation, geometry = _sample()
    observation["measured_hands"]["right"]["reaction"]["thumb"]["mean_normal_n"] = .001
    assert not api().choose_pulse(observation, geometry).any()
    observation["reaction_valid"] = False
    assert not api().choose_pulse(observation, geometry).any()


def test_native_float32_pose_matches_wrist_controller_float64_and_wxyz():
    import torch

    script = importlib.import_module("scripts.verify_dh116_wire_live_palm_approach")
    sdk = 10
    pose = np.array([0., 0., np.sqrt(.5), np.sqrt(.5)], dtype=np.float32)
    pos = np.array([.45, -.12, .4], dtype=np.float32)
    poses = {f"{side}.base_link": (pos.copy(), pose.copy()) for side in ("left", "right")}
    row = script.pack_geometry(sdk, poses, {}, np.array([.04, .04], dtype=np.float32))
    assert row["root_poses_w"].dtype == row["clearance_m"].dtype == torch.float64
    np.testing.assert_array_equal(row["root_poses_w"][0, 0], np.r_[pos, pose[[3, 0, 1, 2]]])
