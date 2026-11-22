import numpy as np

from go2_pvcnn.control.dh116.linear_coupling import active_limits, expand
from scripts import screen_dh116_power_thumb_clearance as screen


def test_grid_sweeps_both_original_thumb_controls_and_includes_the_measured_pose():
    lo, hi = active_limits()
    start = (lo + hi) / 2
    samples = screen.thumb_joint_grid(start, sample_counts=(8, 10))
    np.testing.assert_allclose(samples.min(axis=0), lo[:2])
    np.testing.assert_allclose(samples.max(axis=0), hi[:2])
    assert any(np.array_equal(q, start[:2]) for q in samples)
    for thumb_angles in samples:
        active = start.copy()
        active[:2] = thumb_angles
        expand(active)


def test_thumb_witness_rejects_free_looking_tip_with_deep_second_node():
    gap = np.array([[0., -.0008, .002, np.nan],
                    [.00001, .0002, .002, np.nan],
                    [.00001, .0003, .002, np.nan],
                    [.001, .001, .0001, np.nan]])
    dots = np.full(gap.shape, -.9)
    dots[1, 1] = 0.
    allowed = np.array([False, True, True, True])
    rows, nodes = screen.opposing_thumb_witnesses(gap, dots, allowed, partner_node=1)
    assert list(zip(rows, nodes, strict=True)) == [(2, 1), (3, 2)]


def test_middle_link_check_uses_the_whole_segment_and_the_original_nontarget_margin():
    from itertools import product

    hull = np.array(list(product((-.001,.001),(-.001,.001),(-.001,.001))))
    nodes = np.array([[-.02,0.,0.],[.02,0.,0.]])
    pierced = screen.middle_link_clearance(hull, nodes, .005)
    assert pierced["status"] == "certified_below_limit"
    assert pierced["upper_m"] < 0.
    passed = screen.middle_link_clearance(hull+[0.,0.,.008], nodes, .005)
    assert passed["status"] == "clear"
    assert passed["lower_m"] > .00025

def test_local_detours_keep_coupled_joint_and_wrist_endpoints():
    from scipy.spatial.transform import Rotation

    f = np.linspace(0., 1., 5)
    source = np.array([.45, .2, .39, 1., 0., 0., 0.])
    target = source.copy()
    target[:3] += [.014, -.001, .006]
    angle = np.array([.08, -.15, .21])
    target[3:] = Rotation.from_euler("XYZ", angle).as_quat()[[3, 0, 1, 2]]
    projection = {"projected_root_poses": np.array([[source, source]]),
                  "projected_angles": np.zeros((1, 2, 3))}
    plain = screen.wrist_samples(source, target, projection, "left", f)
    detour = screen.wrist_samples(source, target, projection, "left", f,
                                  retreat_world_m=np.array([0., 0., .0015]))
    np.testing.assert_allclose(detour[[0, -1]], [source, target], atol=1e-12)
    np.testing.assert_allclose(detour[2, :3] - plain[2, :3], [0., 0., -.0015])
    lo, hi = active_limits()
    start = lo + .1*(hi-lo)
    end = start.copy()
    end[:2] = lo[:2] + .7*(hi-lo)[:2]
    joints = screen.thumb_path(start, end, f, flex_exponent=2.)
    np.testing.assert_allclose(joints[[0, -1]], [start, end])
    assert joints[2,1] > (.5*(start+end))[1]
    for q in joints:
        np.testing.assert_array_equal(q[2:], start[2:])
        expand(q)
