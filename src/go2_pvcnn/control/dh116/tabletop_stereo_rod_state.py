"""Read-only node state boundary for an independent deformable solver runtime.

The rigid TensorTask bridge cannot provide this observation. A Cosserat/DeformX
integration must supply one atomic ``read_rod_snapshot`` at the requested
physics boundary. This module validates and owns that snapshot without starting
Isaac, stepping the solver or treating a provenance declaration as qualification.
"""

import math
from pathlib import Path

import torch

from .tabletop_stereo_contract import file_sha256, require_sha


VERSION = 'dh116-deformable-rod-readback-v1'
UNITS = {
    'node_positions': 'm',
    'velocities': 'm/s',
    'centerline': 'm',
    'tangent': '1',
    'curvature': '1/m',
    'distributed_contact': 'N',
    'deformation': '1',
    'rest_lengths_m': 'm',
}


def _validate_clock(env_ids, episode_uids, reset_ids, physics_step, timestamp_s):
    if (not isinstance(env_ids, (list, tuple)) or not env_ids
            or any(type(value) is not int or value < 0 for value in env_ids)
            or len(set(env_ids)) != len(env_ids)):
        raise ValueError('env_ids must identify distinct nonnegative environment slots')
    if (not isinstance(episode_uids, (list, tuple)) or len(episode_uids) != len(env_ids)
            or any(not isinstance(value, str) or not value for value in episode_uids)
            or len(set(episode_uids)) != len(episode_uids)):
        raise ValueError('episode_uids must uniquely identify each environment episode')
    if (not isinstance(reset_ids, (list, tuple)) or len(reset_ids) != len(env_ids)
            or any(type(value) is not int or value < 0 for value in reset_ids)):
        raise ValueError('reset_ids must align with the environment slots')
    if type(physics_step) is not int or physics_step < 0:
        raise ValueError('physics_step must be a nonnegative integer')
    if type(timestamp_s) not in (int, float) or not math.isfinite(timestamp_s) or timestamp_s < 0:
        raise ValueError('timestamp_s must be finite and nonnegative')


def _owned_tensor(packet, name, shape, *, device=None, dtype=None, finite=True):
    tensor = packet.get(name)
    if (not isinstance(tensor, torch.Tensor) or tensor.shape != shape
            or (device is not None and tensor.device != device)
            or (dtype is not None and tensor.dtype != dtype)
            or (dtype is None and tensor.dtype not in (torch.float32, torch.float64))):
        raise ValueError(f'{name} requires actual solver tensor with shape {tuple(shape)} and matching device/dtype')
    tensor = tensor.detach().clone()
    if finite and not bool(torch.isfinite(tensor).all()):
        raise ValueError(f'{name} must be finite at the requested physics boundary')
    return tensor


class DeformableRodStateProvider:
    """Bind an already constructed solver to its backend artifact and readback.

    Arrays follow one stable material-node order for each episode. The
    centerline is sampled at those nodes, curvature is a nonnegative nodal
    magnitude, and deformation is segment axial engineering strain. Contact
    is the distributed force on the rod in world axes. Unavailable nodal
    contact remains NaN with a false validity bit.
    """

    def __init__(self, solver, *, backend, backend_artifact, backend_sha256):
        if backend not in ('Cosserat', 'DeformX'):
            raise ValueError('Cosserat or DeformX solver backend required')
        reader = getattr(solver, 'read_rod_snapshot', None)
        if not callable(reader):
            raise ValueError('Cosserat/DeformX solver must provide atomic read_rod_snapshot')
        self._reader = reader
        self._artifact = Path(backend_artifact).resolve(strict=True)
        self._sha256 = require_sha(backend_sha256, 'solver backend SHA256')
        self._backend = backend
        self._validate_artifact()

    def _validate_artifact(self):
        if (self._sha256 == '0' * 64 or not self._artifact.is_file()
                or file_sha256(self._artifact) != self._sha256):
            raise ValueError('solver backend SHA256 differs from the bound artifact bytes')

    @torch.no_grad()
    def snapshot(self, *, env_ids, episode_uids, reset_ids, physics_step, timestamp_s):
        _validate_clock(env_ids, episode_uids, reset_ids, physics_step, timestamp_s)
        self._validate_artifact()
        packet = self._reader()
        if not isinstance(packet, dict) or packet.get('version') != VERSION:
            raise ValueError('versioned solver node readback packet required')
        if packet.get('coordinate_frame') != 'world':
            raise ValueError('coordinate_frame must be world')
        if packet.get('units') != UNITS:
            raise ValueError('units must match the declared SI node readback contract')

        identity = dict(env_ids=list(env_ids), episode_uids=list(episode_uids),
            reset_ids=list(reset_ids), physics_step=physics_step, timestamp_s=float(timestamp_s))
        for name, expected in identity.items():
            actual = packet.get(name)
            if name == 'timestamp_s':
                matches = type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9)
            elif isinstance(expected, list):
                matches = (isinstance(actual, (list, tuple)) and len(actual) == len(expected)
                    and all(type(a) is type(b) and a == b for a, b in zip(actual, expected)))
            else:
                matches = type(actual) is type(expected) and actual == expected
            if not matches:
                raise ValueError(f'{name} differs from the requested solver physics boundary')

        raw_positions = packet.get('node_positions')
        if (not isinstance(raw_positions, torch.Tensor) or raw_positions.ndim != 3
                or raw_positions.shape[0] != len(env_ids) or raw_positions.shape[1] < 3
                or raw_positions.shape[2] != 3):
            raise ValueError('node_positions requires at least three actual material nodes per environment')
        shape = raw_positions.shape
        node_shape, segment_shape = shape[:2], (shape[0], shape[1] - 1)
        positions = _owned_tensor(packet, 'node_positions', shape)
        opts = dict(device=positions.device, dtype=positions.dtype)
        velocities = _owned_tensor(packet, 'velocities', shape, **opts)
        centerline = _owned_tensor(packet, 'centerline', shape, **opts)
        tangent = _owned_tensor(packet, 'tangent', shape, **opts)
        curvature = _owned_tensor(packet, 'curvature', node_shape, **opts)
        contact = _owned_tensor(packet, 'distributed_contact', shape, finite=False, **opts)
        contact_valid = _owned_tensor(packet, 'contact_valid', node_shape,
            device=positions.device, dtype=torch.bool)
        rest_lengths = _owned_tensor(packet, 'rest_lengths_m', segment_shape, **opts)
        deformation = _owned_tensor(packet, 'deformation', segment_shape, **opts)

        lengths = torch.linalg.vector_norm(torch.diff(positions, dim=1), dim=-1)
        if bool((lengths <= 0).any()):
            raise ValueError('node_positions contains degenerate material segments')
        center_edges = torch.diff(centerline, dim=1)
        if bool((torch.linalg.vector_norm(center_edges, dim=-1) <= 0).any()):
            raise ValueError('centerline contains degenerate segments')
        if (not torch.allclose(torch.linalg.vector_norm(tangent, dim=-1), torch.ones_like(curvature),
                atol=1e-4, rtol=0) or bool(((tangent[:, :-1] * center_edges).sum(-1) <= 0).any())
                or bool(((tangent[:, 1:] * center_edges).sum(-1) <= 0).any())):
            raise ValueError('tangent must be unit length and follow the material node order')
        if bool((curvature < 0).any()):
            raise ValueError('curvature is a nonnegative magnitude in 1/m')
        if (bool((contact_valid & ~torch.isfinite(contact).all(-1)).any())
                or bool((~contact_valid & ~torch.isnan(contact).all(-1)).any())):
            raise ValueError('distributed_contact must be finite when valid and entirely NaN when unavailable')
        if bool((rest_lengths <= 0).any()):
            raise ValueError('rest_lengths_m must be positive per material segment')
        if not torch.allclose(deformation, lengths / rest_lengths - 1, atol=2e-6, rtol=1e-5):
            raise ValueError('deformation must match segment axial engineering strain')

        return dict(version=VERSION, task_kind='deformable_rod', source_scope='solver_readback_unqualified',
            backend=self._backend, backend_sha256=self._sha256, coordinate_frame='world', units=dict(UNITS),
            node_positions=positions, velocities=velocities, centerline=centerline, tangent=tangent,
            curvature=curvature, distributed_contact=contact, contact_valid=contact_valid,
            rest_lengths_m=rest_lengths, deformation=deformation, **identity)
