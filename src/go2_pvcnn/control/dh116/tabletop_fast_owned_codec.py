"""Opt-in byte-identical finite-array encoder for owned diagnostic snapshots.

The existing codec remains the schema authority. Only exact ordinary numeric
ndarrays with finite builtin-width elements skip redundant scalar recursion.
Missing/nonfinite values, array subclasses and all other inputs use the old
implementation. No value is quantized, rounded, filled, clipped or dropped.
"""
from contextlib import contextmanager
from threading import RLock
from math import isfinite
import numpy as np
from . import tabletop_deformable_r0_replay as codec

VERSION="dh116-owned-finite-array-codec-v1"
_BASE=codec.encode_owned
_LOCK=RLock()

def encode_owned(value):
    kind=type(value)
    if value is None or kind in (str,int,bool):
        return value
    if kind is float:
        if isfinite(value):
            return value
        return _BASE(value)
    if kind is dict:
        if any(not isinstance(k,str) for k in value):
            raise ValueError("explicit string dictionary keys required")
        return {k:encode_owned(v) for k,v in value.items()}
    if kind in (tuple,list):
        return [encode_owned(v) for v in value]
    if hasattr(value,"detach"):
        value=value.detach().cpu().numpy()
    if (type(value) is np.ndarray and value.dtype.kind in "bifu"
            and value.dtype.itemsize<=8 and np.isfinite(value).all()):
        return {"__dh116_array__":value.tolist(),"dtype":str(value.dtype)}
    return _BASE(value)

@contextmanager
def codec_scope():
    """Process-local opt-in in the single owner; restore on every exit.

    Recursive calls made by the legacy encoder also use this byte-equivalent
    finite-array path. A lock serializes overlapping opt-in scopes; unrelated
    callers see the same schema/bytes. No inference/control module is patched.
    """
    with _LOCK:
        before=codec.encode_owned
        if before not in (_BASE,encode_owned):
            raise RuntimeError("unexpected owned codec binding before opt-in")
        codec.encode_owned=encode_owned
        try:
            yield
        finally:
            if codec.encode_owned is not encode_owned:
                raise RuntimeError("owned codec binding changed inside isolated scope")
            codec.encode_owned=before
