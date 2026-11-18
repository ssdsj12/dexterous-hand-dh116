"""Strict JSON evidence: explicit NumPy conversion, never NaN/Infinity tokens."""
import json


def _native(value):
    import numpy as np
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'unsupported evidence type: {type(value).__name__}')


def dumps(value,**kwargs):
    return json.dumps(value,allow_nan=False,default=_native,**kwargs)
