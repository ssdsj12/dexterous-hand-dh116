import numpy as np
import torch
from go2_pvcnn.control.dh116.tabletop_batch_geometry import rotated_heights as reference_heights
from go2_pvcnn.control.dh116.tabletop_batch_geometry import rotated_points as reference_points
from go2_pvcnn.control.dh116.tabletop_tensor_world import (
    rotated_heights,
    rotated_points,
    signed_hand_vertical_force,
)


def test_geometry_matches_exact_hull_reference():
    rng = np.random.default_rng(42)
    pose = rng.normal(size=(32,7)); pose[:,3:] /= np.linalg.norm(pose[:,3:],axis=1)[:,None]
    points = rng.normal(size=(125,3))*.03
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        actual = rotated_heights(torch.tensor(pose,device=device),torch.tensor(points,device=device))
        assert actual.device.type == device
        np.testing.assert_allclose(actual.cpu(),reference_heights(pose,points),atol=1e-12,rtol=0)
        actual = rotated_points(torch.tensor(pose,device=device),torch.tensor(points[0],device=device))
        np.testing.assert_allclose(actual.cpu(),reference_points(pose,points[:1])[:,0],atol=1e-12,rtol=0)


def test_signed_hand_vertical_force_sums_normal_and_friction_components():
    components = torch.tensor([[[1., -2.], [3., -4.]]], dtype=torch.float64)
    torch.testing.assert_close(
        signed_hand_vertical_force(components),
        torch.tensor([[-1., -1.]], dtype=torch.float64),
    )
