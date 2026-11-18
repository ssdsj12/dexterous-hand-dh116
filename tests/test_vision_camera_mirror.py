import torch

from go2_pvcnn.control.dh116.tabletop_vision_camera import frames_to_points


def test_mirrored_world_camera_and_hand_frames_exchange_local_points():
    # A symmetric brown depth image is enough to exercise the complete
    # camera->world->wrist-local path without an Isaac process.
    rgb = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    rgb[:] = torch.tensor([100., 20., 10.])
    depth = torch.ones((1, 4, 4), dtype=torch.float32)
    K = torch.eye(3).reshape(1, 3, 3)
    T = torch.eye(4).reshape(1, 4, 4)
    hands = torch.tensor([[[.2, 0., 0., 1., 0., 0., 0.],
                           [-.2, 0., 0., 1., 0., 0., 0.]]])
    original = frames_to_points(rgb, depth, K, T, hands, num_points=8,
                                min_depth_m=.1, max_depth_m=2., depth_edge_threshold_m=None)

    mirror = torch.diag(torch.tensor([-1., 1., 1., 1.])).reshape(1, 4, 4)
    mirrored_hands = hands[:, [1, 0]].clone()
    mirrored_hands[:, :, :3] = hands[:, [1, 0], :3] * torch.tensor([-1., 1., 1.])
    mirrored = frames_to_points(rgb, depth, K, mirror, mirrored_hands, num_points=8,
                                min_depth_m=.1, max_depth_m=2., depth_edge_threshold_m=None)

    expected = original['points'][:, [1, 0]].clone()
    expected[..., 0] *= -1.
    assert torch.equal(original['point_valid'], mirrored['point_valid'])
    assert torch.allclose(mirrored['points'], expected, atol=1e-6, rtol=0.)
