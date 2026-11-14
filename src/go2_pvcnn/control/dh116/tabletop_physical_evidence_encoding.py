"""Byte-equivalent encoder for already-owned numerical physical evidence."""
from contextlib import contextmanager
import math

import numpy as np

from . import tabletop_deformable_dual_evidence as journal
from .tabletop_deformable_r0_replay import encode_owned as _LEGACY
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings


def encode_owned(value):
    """Keep legacy JSON bytes, array dtypes, and nonfinite tags."""
    value_type = type(value)
    if value_type is float:
        if math.isfinite(value):
            return value
        return {"__dh116_nonfinite__": "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"}
    if value is None or value_type in (str, int, bool):
        return value
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("explicit string dictionary keys required")
        return {key: encode_owned(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode_owned(item) for item in value]
    if isinstance(value, np.ndarray):
        kind = value.dtype.kind
        if kind not in "bifu":
            raise ValueError("numeric owned arrays only")
        values = value.tolist()
        # Ordinary finite floats, booleans, and integers need no Python scalar
        # walk. Extended floats and nonfinite arrays follow the original codec.
        if kind == "f" and (value.dtype.itemsize > 8 or not np.isfinite(value).all()):
            values = _LEGACY(values)
        return {"__dh116_array__": values, "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return encode_owned(value.item())
    if hasattr(value, "detach"):
        return encode_owned(value.detach().cpu().numpy())
    return _LEGACY(value)


@contextmanager
def encoding_scope():
    """Use the faster equivalent writer only inside the selected owner."""
    if journal.encode_owned is not _LEGACY:
        raise ValueError("another physical evidence encoding scope is active")
    with bindings((journal, "encode_owned", encode_owned)):
        yield
