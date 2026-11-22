"""Protect material alignment and geometric selection in the contact-seeded Reset."""
import importlib

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def api():
    name = "scripts.prepare_dh116_tripod_contact_seed"
    assert importlib.util.find_spec(name), "measured Tripod contact seed is missing"
    return importlib.import_module(name)


def test_aligns_material_point_and_tangent_without_changing_palm_local_geometry():
    source = Rotation.from_euler("z", 20, degrees=True).as_quat()
    current_q = Rotation.from_euler("x", 15, degrees=True).as_quat()
    origin = np.array([.42, .18, .34])
    target = np.array([.44, .18, .365])
    hand = origin + np.array([.04, -.005, .017])
    old = Rotation.from_quat(current_q)
    tangent = Rotation.from_quat(source).apply([0., 1., 0.])
    new = api().align_palm(hand, current_q, origin, tangent, target, [0., 1., 0.])
    rot = Rotation.from_quat(new[[4, 5, 6, 3]])
    np.testing.assert_allclose(rot.inv().apply(target - new[:3]), old.inv().apply(origin - hand), atol=1e-13)
    np.testing.assert_allclose(rot.inv().apply([0., 1., 0.]), old.inv().apply(tangent), atol=1e-13)
    with pytest.raises(ValueError, match="material tangent"):
        api().align_palm(hand, current_q, origin, [0., -1., 0.], target, [0., 1., 0.])


def test_reset_tube_keeps_the_measured_sag_envelope_and_authenticates_seed_frame():
    reset = np.zeros((41, 3))
    reset[:, 1] = np.linspace(-.225, .225, 41)
    sag = reset.copy()
    sag[20, 2] = -.002
    normals, upper = api().reset_tube(reset, sag[None], radius_m=.0045)
    down = np.argmin(normals[:, 2])
    assert upper[9:11, down].max() == pytest.approx(.0065)
    with pytest.raises(ValueError, match="complete measured parked"):
        api().reset_tube(reset, sag[:40][None], radius_m=.0045)
