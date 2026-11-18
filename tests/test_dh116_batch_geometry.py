import numpy as np
from scipy.spatial.transform import Rotation
from go2_pvcnn.control.dh116.tabletop_batch_geometry import rotated_points, collision_clearance


def test_batched_rotation_matches_independent_scipy_and_translation():
    rng = np.random.default_rng(7)
    q = Rotation.random(8, random_state=rng).as_quat()
    pose = np.c_[rng.normal(size=(8, 3)), q[:, 3], q[:, :3]]
    points = rng.normal(size=(13, 3))
    got = rotated_points(pose, points)
    expected = np.stack([Rotation.from_quat(v[3:][[1,2,3,0]]).apply(points)+v[:3] for v in pose])
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_clearance_keeps_every_hull_vertex_and_each_hand_separate():
    poses = np.zeros((3, 2, 2, 7)); poses[..., 3] = 1
    poses[..., 2] = .31
    poses[1, 1, 1, 2] = .29
    shapes = [(0, 0, np.array([[0,0,-.001],[0,0,.001]])),
              (1, 1, np.array([[0,0,-.002],[0,0,.002]]))]
    clear, lowest = collision_clearance(poses, shapes)
    np.testing.assert_allclose(clear, [[.009,.008],[.009,-.012],[.009,.008]], atol=1e-12)
    assert lowest.tolist() == [1,1,1]


def test_normalized_cooking_preserves_thin_shell_vertices_exactly():
    from go2_pvcnn.control.dh116.tabletop_batch_geometry import normalized_convex
    from go2_pvcnn.control.dh116.hollow_straw import collision_wedges, straw_parameters
    for points,_ in collision_wedges(straw_parameters()):
        local,center,scale=normalized_convex(points)
        np.testing.assert_allclose(local*scale+center,points,atol=1e-15)
        np.testing.assert_allclose(np.ptp(local,axis=0),2.)


def test_height_only_transform_preserves_every_vertex_for_float32_physics_poses():
    from go2_pvcnn.control.dh116.tabletop_batch_geometry import rotated_heights
    rng = np.random.default_rng(103)
    q = Rotation.random(97,random_state=rng).as_quat()
    poses = np.c_[rng.uniform(-100,100,(97,3)),q[:,3],q[:,:3]].astype(np.float32)
    points = rng.uniform(-.1,.1,(400,3))
    full = rotated_points(poses,points)[...,2]
    np.testing.assert_allclose(rotated_heights(poses,points),full,atol=1e-12,rtol=0)


def test_fast_clearance_matches_independent_rotation_for_asymmetric_hulls():
    rng = np.random.default_rng(17)
    poses = np.zeros((31,2,3,7))
    poses[...,:3] = rng.uniform(-1,1,(31,2,3,3))
    q = Rotation.random(31*2*3,random_state=rng).as_quat().reshape(31,2,3,4)
    poses[...,3:] = q[...,[3,0,1,2]]
    shapes = [(h,b,rng.uniform(-.1,.1,(73+b,3))) for h in range(2) for b in range(3)]
    heights=np.stack([rotated_points(poses[:,h,b],points)[...,2].min(axis=1)
                      for h,b,points in shapes],axis=1)
    expected=np.stack([heights[:,:3].min(axis=1),heights[:,3:].min(axis=1)],axis=1)-.3
    clearance,lowest = collision_clearance(poses,shapes)
    np.testing.assert_allclose(clearance,expected,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(lowest,heights.argmin(axis=1))
