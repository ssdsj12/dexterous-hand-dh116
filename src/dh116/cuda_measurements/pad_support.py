"""Unmerged native pad CUDA ctypes wrapper."""
import ctypes, math
from pathlib import Path
import torch

class PadSupport:

    def __init__(self, pads, roundings, *, device, library=None):
        self.device = torch.empty(0, device=device).device
        if self.device.type != 'cuda':
            raise ValueError('CUDA device required')
        if not pads or len(pads) != len(roundings):
            raise ValueError('complete original pad/roundoff mapping required')
        values = []
        offsets = [0]
        dimensions = []
        bodies = []
        origins = []
        spacings = []
        rounds = []
        for (pad, rounding) in zip(pads, roundings):
            (side, link, _, _, value, origin, spacing) = pad
            if side not in (0, 1) or not 0 <= link < 23 or value.ndim != 3 or (min(value.shape) < 2) or (value.dtype != torch.float64) or (not bool(torch.isfinite(value).all())) or (origin.shape != (3,)) or (not bool(torch.isfinite(origin).all())) or (spacing.shape != ()) or (not bool(torch.isfinite(spacing) & (spacing > 0))) or (rounding.shape != (9,)) or (not bool(torch.isfinite(rounding).all())):
                raise ValueError('finite original locked grid and roundoff metadata required')
            values.append(value.detach().to(self.device).flatten().clone())
            offsets.append(offsets[-1] + value.numel())
            dimensions.append(value.shape)
            bodies.append(side * 23 + link)
            origins.append(origin.detach().to(self.device).clone())
            spacings.append(spacing.detach().to(self.device).clone())
            rounds.append(rounding.detach().to(self.device).clone())
        self.pads = len(pads)
        self._values = torch.cat(values)
        self._offsets = torch.tensor(offsets, dtype=torch.int64, device=self.device)
        self._dimensions = torch.tensor(dimensions, dtype=torch.int64, device=self.device)
        self._bodies = torch.tensor(bodies, dtype=torch.int64, device=self.device)
        self._origins = torch.stack(origins)
        self._spacings = torch.stack(spacings)
        self._roundings = torch.stack(rounds)
        self._threshold_cache = {}
        path = Path(library) if library else Path(__file__).parent / 'build/libdh116_hull_support.so'
        self._lib = ctypes.CDLL(str(path))
        self._launch = self._lib.dh116_pad_metrics
        self._launch.argtypes = [ctypes.c_void_p] * 12 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 5
        self._launch.restype = ctypes.c_int
        self._lib.dh116_hull_error.argtypes = [ctypes.c_int]
        self._lib.dh116_hull_error.restype = ctypes.c_char_p

    def metrics(self, rotations, translations, nodes, radii, thresholds):
        tensors = (rotations, translations, nodes, radii)
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor) or tensor.device != self.device or tensor.dtype != torch.float64 or (not tensor.is_contiguous()):
                raise ValueError('same-device contiguous CUDA float64 pad inputs required')
        n = rotations.shape[0] if rotations.ndim else 0
        if n < 1 or rotations.shape != (n, 2, 23, 3, 3) or translations.shape != (n, 2, 23, 3) or (nodes.shape != (n, 41, 3)) or (radii.shape != (n,)):
            raise ValueError('complete native pad input shape required')
        key = tuple((float(v) for v in thresholds))
        if not key or not all((math.isfinite(v) for v in key)):
            raise ValueError('finite original thresholds required')
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            threshold_tensor = self._threshold_cache.get(key)
            if threshold_tensor is None:
                threshold_tensor = torch.tensor(key, dtype=torch.float64, device=self.device)
                self._threshold_cache[key] = threshold_tensor
            gap = torch.empty((n, self.pads, 41), dtype=torch.float64, device=self.device)
            norm = torch.empty_like(gap)
            valid = torch.empty((n, self.pads, 41), dtype=torch.bool, device=self.device)
            flags = torch.empty((n, self.pads), dtype=torch.bool, device=self.device)
            inputs = (self._values, self._offsets, self._dimensions, self._bodies, self._origins, self._spacings, self._roundings, rotations, translations, nodes, radii, threshold_tensor)
            code = self._launch(*(t.data_ptr() for t in inputs), n, self.pads, len(key), gap.data_ptr(), norm.data_ptr(), valid.data_ptr(), flags.data_ptr(), stream.cuda_stream)
            if code:
                raise RuntimeError('CUDA pad measurement: ' + self._lib.dh116_hull_error(code).decode())
            for tensor in inputs:
                tensor.record_stream(stream)
            return (gap, norm, valid, flags)
