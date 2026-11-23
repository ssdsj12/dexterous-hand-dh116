"""Device-resident candidate arithmetic for the native Teacher batch.

The live exchange continues to use the original safety validators until the
full, same-owner CUDA readback and contact contracts have native parity.
"""
import numpy as np
import torch

from .linear_coupling import ACTIVE, MIMIC, active_limits
from .tabletop_training_reward import PHYSICS_HZ, WINDOW_SUBSTEPS


class TensorSourceClearance:
    """Evaluate every original source hull vertex from SDK WXYZ poses."""

    def __init__(self, shapes, *, device):
        self._shapes = tuple(shapes)
        if {side for side, _, _ in self._shapes} != {0, 1}:
            raise ValueError("both original collision hull sets required")
        vertices = []
        shape_ids = []
        for index, (side, link, points) in enumerate(self._shapes):
            points = np.asarray(points, dtype=np.float64)
            if (type(link) is not int or link < 0 or points.ndim != 2 or points.shape[1] != 3
                    or not len(points) or not np.isfinite(points).all()):
                raise ValueError("finite original collision hull vertices and link required")
            vertices.append(points)
            shape_ids.extend([index] * len(points))
        self._vertices = torch.as_tensor(np.concatenate(vertices), dtype=torch.float64, device=device)
        self._shape_id = torch.tensor(shape_ids, dtype=torch.long, device=device)
        self._hands = torch.tensor([shape[0] for shape in self._shapes], device=device)
        self._links = torch.tensor([shape[1] for shape in self._shapes], device=device)
        self._link_count = max(shape[1] for shape in self._shapes) + 1

    def __call__(self, poses):
        if (poses.ndim != 4 or poses.shape[1] != 2 or poses.shape[2] < self._link_count
                or poses.shape[3] != 7 or poses.device != self._vertices.device):
            raise ValueError("matching batched measured link poses required")
        selected = poses[:, self._hands, self._links].double()
        w, x, y, z = selected[..., 3:].unbind(-1)
        row = torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x),
                           1 - 2 * (x.square() + y.square())), dim=-1)
        height = (row[:, self._shape_id] * self._vertices).sum(dim=-1) + selected[:, self._shape_id, 2]
        minimum = height.new_full((len(poses), len(self._shapes)), float("inf"))
        minimum.scatter_reduce_(1, self._shape_id.expand(len(poses), -1), height, reduce="amin")
        hands = minimum.new_full((len(poses), 2), float("inf"))
        hands.scatter_reduce_(1, self._hands.expand(len(poses), -1), minimum, reduce="amin")
        return hands - .30, minimum.argmin(dim=1)


def joint_force_guard(q, forces, joint_names):
    """Encode the existing left, right, then force guard reason precedence."""
    if (q.ndim != 3 or q.shape[1:] != (2, len(joint_names)) or forces.ndim != 4
            or forces.shape[:2] != q.shape[:2] or forces.shape[-1] != 3 or forces.device != q.device
            or len(set(joint_names)) != len(joint_names)):
        raise ValueError("matching original named joint and hand force tensors required")
    q = q.double()
    active = q[..., [joint_names.index(name) for name in ACTIVE]]
    followers, masters, ratio, offset = zip(*MIMIC, strict=True)
    errors = q[..., [joint_names.index(name) for name in followers]] - (
        q[..., [joint_names.index(name) for name in masters]] * q.new_tensor(ratio) + q.new_tensor(offset))
    lower, upper = (q.new_tensor(values) for values in active_limits())
    nonfinite = ~(torch.isfinite(active).all(dim=-1) & torch.isfinite(errors).all(dim=-1))
    limit = ((active < lower - .02) | (active > upper + .02)).any(dim=-1)
    mimic = (errors.abs() > .02).any(dim=-1)
    codes = torch.where(nonfinite, 1, torch.where(limit, 2, torch.where(mimic, 3, 0)))
    left, right = codes.unbind(dim=1)
    excess = (torch.linalg.vector_norm(forces.double(), dim=-1) > 30).flatten(1).any(dim=1)
    return torch.where(left != 0, left, torch.where(right != 0, right + 3, excess.long() * 7))


def reward_window(phi_pre, phi_post, rates, gamma, event, terminated):
    """Reduce validated raw reward components using the original 480 Hz sum."""
    size = phi_pre.shape
    if (len(size) != 1 or rates.ndim != 2 or rates.shape[0] != size[0]
            or not 1 <= rates.shape[1] <= WINDOW_SUBSTEPS
            or not 0 < gamma < 1 or terminated.dtype != torch.bool
            or any(v.shape != size for v in (phi_post, event, terminated))
            or any(v.device != phi_pre.device for v in (phi_post, event, terminated, rates))):
        raise ValueError("same-device validated original reward components required")
    return (gamma * torch.where(terminated, 0., phi_post.double()) - phi_pre.double()
            - rates.double().sum(dim=1) / PHYSICS_HZ + event.double())
