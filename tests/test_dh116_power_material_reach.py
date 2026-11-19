import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def test_power_material_roll_rotates_transverse_offset_about_same_station():
    from scripts.audit_dh116_power_material_reach import material_wrist_seed

    root = np.array([0.02, .1, .03, 1., 0., 0., 0.])
    station = np.array([0., .1, .03])
    tangent = np.array([0., 1., 0.])
    inward = np.array([1., 0., 0.])
    pose = material_wrist_seed(root, station, tangent, inward, angle_deg=-30.,
                              lateral_m=.004, backoff_m=.004)
    turn = Rotation.from_quat(np.roll(pose[3:], -1))
    delta = (root[:3] - station + [.0, .0, -.004] - [.004, .0, .0])
    np.testing.assert_allclose(pose[:3] - station, turn.apply(delta), atol=1e-14)
    assert np.linalg.det(turn.as_matrix()) == pytest.approx(1.)


def test_power_palm_seating_search_brackets_before_penetration():
    from scripts.audit_dh116_power_material_reach import palm_approach_samples

    def measure(inward):
        return {"palm_gap_m": .0197 - inward}

    rows = palm_approach_samples(measure)
    assert any(0. <= row["palm_gap_m"] <= .0005 for row in rows)
    assert any(row["palm_gap_m"] < 0. for row in rows)
    assert max(row["inward_m"] for row in rows) <= .024
