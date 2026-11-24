"""Unmerged native hull CUDA ctypes wrapper."""
import ctypes
from pathlib import Path
import numpy as np
import torch

class HullSupport:

    def __init__(self, shapes, normals, *, device, library=None):
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('CUDA device required')
        self.device = torch.empty(0, device=self.device).device
        points = []
        sides = []
        links = []
        offsets = [0]
        keys = set()
        sources = []
        for (side, link, key, vertices) in shapes:
            v = np.array(vertices, dtype=np.float64, copy=True)
            if type(side) is not int or side not in (0, 1) or type(link) is not int or (not 0 <= link < 23) or (not isinstance(key, str)) or (not key) or ((side, key) in keys) or (v.ndim != 2) or (v.shape[1:] != (3,)) or (not len(v)) or (not np.isfinite(v).all()):
                raise ValueError('distinct complete finite original hull and actual side/link required')
            keys.add((side, key))
            points.append(v)
            sides.append(side)
            links.append(link)
            offsets.append(offsets[-1] + len(v))
            sources.append((side, link, torch.tensor(v, dtype=torch.float64, device=self.device)))
        if not points:
            raise ValueError('nonempty complete original source hulls required')
        normal_cpu = normals.detach().cpu().numpy().copy() if isinstance(normals, torch.Tensor) else np.array(normals)
        if normal_cpu.shape != (64, 3) or normal_cpu.dtype != np.float64 or (not np.isfinite(normal_cpu).all()):
            raise ValueError('64 finite float64 normals required')
        if np.any(normal_cpu[:, 1] != 0):
            raise ValueError('locked support normals require exact zero y')
        self.normals = torch.tensor(normal_cpu, dtype=torch.float64, device=self.device)
        self._vertices = torch.tensor(np.concatenate(points), dtype=torch.float64, device=self.device)
        self._offsets = torch.tensor(offsets, dtype=torch.int64, device=self.device)
        self._sides = torch.tensor(sides, dtype=torch.int64, device=self.device)
        self._links = torch.tensor(links, dtype=torch.int64, device=self.device)
        self.oracle_sources = tuple(sources)
        self.vertex_counts = tuple((len(p) for p in points))
        self.total_vertices = offsets[-1]
        self.hulls = len(points)
        self._library_path = Path(library) if library else Path(__file__).parent / 'build/libdh116_hull_support.so'
        self._lib = ctypes.CDLL(str(self._library_path))
        self._launch = self._lib.dh116_hull_support
        self._launch.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int, ctypes.c_int, ctypes.c_int64] + [ctypes.c_void_p] * 5
        self._launch.restype = ctypes.c_int
        self._lib.dh116_hull_error.argtypes = [ctypes.c_int]
        self._lib.dh116_hull_error.restype = ctypes.c_char_p

    def extrema(self, rotations, translations):
        for (tensor, label) in ((rotations, 'rotations'), (translations, 'translations')):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(label + ' must be a CUDA tensor')
            if tensor.device != self.device:
                raise ValueError(label + ' must use source CUDA device')
            if tensor.dtype != torch.float64:
                raise ValueError(label + ' must be float64')
        n = rotations.shape[0] if rotations.ndim else 0
        if n <= 0 or n * self.hulls > 2147483647 or rotations.shape != (n, 2, 23, 3, 3) or (translations.shape != (n, 2, 23, 3)):
            raise ValueError('rotations/translations shape must be [N,2,23,3,3]/[N,2,23,3]')
        if not rotations.is_contiguous() or not translations.is_contiguous():
            raise ValueError('contiguous CUDA inputs required')
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            world = torch.empty((n, self.total_vertices, 3), dtype=torch.float64, device=self.device)
            low = torch.empty((n, self.hulls, 3), dtype=torch.float64, device=self.device)
            high = torch.empty_like(low)
            support = torch.empty((n, self.hulls, 64), dtype=torch.float64, device=self.device)
            tensors = (self._vertices, self._offsets, self._sides, self._links, rotations, translations, self.normals)
            code = self._launch(*(v.data_ptr() for v in tensors), n, self.hulls, self.total_vertices, world.data_ptr(), low.data_ptr(), high.data_ptr(), support.data_ptr(), stream.cuda_stream)
            if code:
                raise RuntimeError('CUDA hull measurement: ' + self._lib.dh116_hull_error(code).decode())
            for tensor in tensors:
                tensor.record_stream(stream)
            return (low, high, support)
