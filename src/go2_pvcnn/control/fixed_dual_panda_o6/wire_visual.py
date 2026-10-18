"""Tubular render geometry driven by measured Cosserat states, not PhysX.

Directors follow PyElastica's row convention. This is a uniform circular wire
skin, not a CAD asset or a second collision/dynamics representation.
"""

from numbers import Real
import numpy as np


def rod_surface(positions, directors, radius, *, sides=16):
    """Return world vertices, polygon sizes and flattened indices for USD Mesh."""
    if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, Real):
        raise TypeError('radius must be a real scalar')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be finite and positive')
    if isinstance(sides, bool) or not isinstance(sides, int) or sides < 3:
        raise ValueError('sides must be an integer >= 3')
    p = np.asarray(positions, dtype=float)
    d = np.asarray(directors, dtype=float)
    if p.ndim != 2 or p.shape[0] != 3 or p.shape[1] < 2 or not np.isfinite(p).all():
        raise ValueError('positions must be finite (3, nodes>=2)')
    if d.shape != (3, 3, p.shape[1]-1) or not np.isfinite(d).all():
        raise ValueError('directors must be finite (3, 3, nodes-1)')
    frames = np.moveaxis(d, -1, 0)
    if not np.allclose(frames @ frames.transpose(0, 2, 1), np.eye(3), atol=1e-5, rtol=0) or np.any(np.linalg.det(frames) <= 0):
        raise ValueError('directors must be right-handed orthonormal frames')
    if np.any(np.linalg.norm(np.diff(p, axis=1), axis=0) <= 1e-10):
        raise ValueError('rod nodes must be distinct')
    frames = np.concatenate((frames, frames[-1:]), axis=0)
    angles = np.arange(sides) * (2*np.pi/sides)
    radial = np.cos(angles)[None, :, None] * frames[:, None, 0, :] + np.sin(angles)[None, :, None] * frames[:, None, 1, :]
    vertices = (p.T[:, None, :] + radius*radial).reshape(-1, 3)
    faces = []
    for station in range(p.shape[1]-1):
        for j in range(sides):
            nxt = (j+1) % sides
            faces.append([station*sides+j, station*sides+nxt, (station+1)*sides+nxt, (station+1)*sides+j])
    faces.extend([list(reversed(range(sides))), list(range((p.shape[1]-1)*sides, p.shape[1]*sides))])
    return vertices, [len(face) for face in faces], [i for face in faces for i in face]
