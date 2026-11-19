import numpy as np
import pytest

from scripts import audit_dh116_tripod_bilateral_closure as closure


def test_bilateral_compilation_drives_both_hands_on_the_same_clock():
    pose = np.array([0., 0., .3, 1., 0., 0., 0.])
    q = np.full(6, .2)
    end = pose.copy()
    end[2] -= .0007
    close = q + [0., .01, .01, .01, 0., 0.]
    endpoints = {side: (pose, end, q, close) for side in ("left", "right")}
    actions, plans = closure.compile_bilateral_actions(endpoints, physics_steps=160)
    assert actions.shape == (20, 24)
    assert np.all(actions[:, [1, 2, 3, 7, 8, 9]] > 0.)
    assert np.all(actions[:, [14, 17]] < 0.)
    assert np.array_equal(actions, sum(np.array(plan["actions"]) for plan in plans.values()))


def test_polyline_path_cannot_certify_a_midpath_rod_violation():
    def query(t):
        return {"table:finger22": .003, "rod:finger13": .002 - .003 * (1 - abs(2*t-1))}

    proof = closure.certify_path(query, dict(zip(query(0.), [0., .006])),
                                dict(zip(query(0.), [.0021, 0.])))
    assert not proof["proved"]
    assert proof["violations"][0]["fraction"] == .5
    assert proof["violations"][0]["shape"] == "rod:finger13"


def test_measured_recenter_endpoint_has_bounded_wrist_and_active_motion():
    from scipy.spatial.transform import Rotation

    root = np.array([0., 0., .3, 1., 0., 0., 0.])
    q = np.full(6, .2)
    rotation = np.roll(Rotation.from_rotvec([0., .003, 0.]).as_quat(), 1)
    recenter = {"root": [0., 0., .3002, *rotation], "q": (.201*np.ones(6)).tolist()}
    pose, angles = closure.recenter_endpoint(root, q, recenter)
    np.testing.assert_allclose(pose, recenter["root"])
    np.testing.assert_allclose(angles, recenter["q"])
    recenter["root"][2] += .005
    with pytest.raises(ValueError, match="bounded"):
        closure.recenter_endpoint(root, q, recenter)


def test_delayed_closure_checks_wrist_and_fingers_independently():
    poses = np.array([[0., 0., height, 1., 0., 0., 0.] for height in (3., 2., 1.)])
    joints = np.array([np.full(6, tick) for tick in (0., 1., 2.)])
    rows = list(closure.delay_samples(poses, joints, 1, [0, 2, 0, 0, 0, 0]))
    assert [(step, pose[2], q[1], q[2]) for step, pose, q in rows] == [
        (4, 1., 2., 2.), (3, 1., 1., 2.), (2, 2., 0., 2.), (1, 3., 0., 1.), (0, 3., 0., 0.),
    ]


def test_tripod_delay_corners_find_a_mixed_failure_hidden_by_two_legacy_cases():
    poses = np.array([[0., 0., height, 1., 0., 0., 0.] for height in (2., 1., 0.)])
    joints = np.zeros((3, 6))
    joints[:, 1:4] = np.arange(3.)[:, None]
    ticks = {name: {"screening_upper_ticks": 1} for name in (*closure.ACTIVE, "wrist_z")}

    def clearance(pose, q):
        failed = pose[2] == 0. and q[1] == 1. and q[2] == q[3] == 2.
        return {"rod:source:finger13_link": -.00001 if failed else .0001}

    assert all(row["violation"] is None for row in closure.screen_delays(poses, joints, ticks, clearance))
    cases = closure.screen_delays(poses, joints, ticks, clearance, corners=True)
    assert len(cases) == 16
    assert len({(row["wrist_delay_ticks"], tuple(row["joint_delays_ticks"])) for row in cases}) == 16
    failed = [row for row in cases if row["violation"]]
    assert len(failed) == 1
    assert failed[0]["joint_delays_ticks"] == [0, 1, 0, 0, 0, 0]
    assert failed[0]["violation"]["sdk_reference_tick"] == 2
    assert failed[0]["violation"]["shape"] == "rod:source:finger13_link"
