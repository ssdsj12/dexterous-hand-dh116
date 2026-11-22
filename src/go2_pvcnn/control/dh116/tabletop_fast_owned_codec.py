"""Opt-in byte-identical finite-array encoder for owned diagnostic snapshots.

The existing codec remains the schema authority. Only exact ordinary numeric
ndarrays with finite builtin-width elements skip redundant scalar recursion.
Missing/nonfinite values, array subclasses and all other inputs use the old
implementation. No value is quantized, rounded, filled, clipped or dropped.
"""
from contextlib import contextmanager
from collections import OrderedDict
import copy
import hashlib
import json
import sys
import weakref
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


def decode_owned(value):
    """Decode numeric JSON arrays without walking every scalar in Python.

    The dtype probe only excludes dictionaries/tags and irregular structure.
    Conversion always uses the original payload, preserving mixed integer/float
    values before the declared dtype is applied. Tagged or unusual inputs keep
    the original decoder's validation and rejection behavior.
    """
    if isinstance(value,dict):
        if set(value)=={"__dh116_array__","dtype"}:
            dtype=np.dtype(value["dtype"])
            if dtype.kind not in "bifu":
                raise ValueError("invalid numeric array type")
            payload=value["__dh116_array__"]
            try:
                probe=np.asarray(payload)
            except (TypeError,ValueError,OverflowError):
                return codec.decode_owned(value)
            if probe.dtype.kind in "bifu":
                return np.array(payload,dtype=dtype,copy=True)
            return codec.decode_owned(value)
        if set(value)=={"__dh116_nonfinite__"}:
            return codec.decode_owned(value)
        return {key:decode_owned(item) for key,item in value.items()}
    if isinstance(value,list):
        return [decode_owned(item) for item in value]
    return value


# Pure CPU deserialization only. No raw bytes or qualification result is held.
_JSON_CACHE = OrderedDict()
_JSON_CACHE_LOCK = RLock()
_JSON_CACHE_MIN_RAW_BYTES = 32 * 1024**2
_JSON_CACHE_MAX_BYTES = 64 * 1024**2
_JSON_CACHE_MAX_ENTRIES = 2


def _owned_cpu_bytes(value, seen):
    if id(value) in seen:
        return 0
    seen.add(id(value))
    kind = type(value)
    size = sys.getsizeof(value)
    if kind in (type(None), bool, int, float, str):
        return size
    if kind is np.ndarray:
        if value.dtype.kind not in "bifu" or not value.flags.owndata:
            return None
        if id(value.dtype) not in seen:
            seen.add(id(value.dtype))
            size += sys.getsizeof(value.dtype)
        # ndarray.__sizeof__ includes its owned CPU allocation.
        return size
    if kind not in (dict, list):
        return None
    children = (part for pair in value.items() for part in pair) if kind is dict else value
    for child in children:
        amount = _owned_cpu_bytes(child, seen)
        if amount is None:
            return None
        size += amount
    return size


def decode_owned_json(raw):
    """Reuse bounded CPU decoding of identical bytes, always returning owned data.

    The caller still verifies provenance and executes all physical validators.
    This cache records neither those checks nor their outcome. Stdlib JSON and
    the existing decoder remain the only schema authorities. The size limit
    bounds retained cache objects, not raw input, temporary copies or process RSS.
    """
    if type(raw) is not bytes or len(raw) < _JSON_CACHE_MIN_RAW_BYTES:
        return decode_owned(json.loads(raw))
    parser, decoder = json.loads, decode_owned
    try:
        identities = tuple(weakref.ref(fn) for fn in (parser, decoder, codec.decode_owned))
    except TypeError:
        return decoder(parser(raw))
    key = (hashlib.sha256(raw).digest(), *identities,
           sys.get_int_max_str_digits() if hasattr(sys, "get_int_max_str_digits") else None)
    with _JSON_CACHE_LOCK:
        if key in _JSON_CACHE:
            value, _ = _JSON_CACHE[key]
            _JSON_CACHE.move_to_end(key)
            return copy.deepcopy(value)
        value = decoder(parser(raw))
        size = _owned_cpu_bytes(value, set())
        if size is None:
            return value
        size += sys.getsizeof(key) + sum(sys.getsizeof(part) for part in key)
        # The stored size integer is bounded by MAX_BYTES whenever retained.
        size += sys.getsizeof((value, size)) + sys.getsizeof(_JSON_CACHE_MAX_BYTES)
        if size + sys.getsizeof(OrderedDict()) > _JSON_CACHE_MAX_BYTES:
            return value
        # Isolate the first caller too; validators may pop/mutate nested fields.
        result = copy.deepcopy(value)
        _JSON_CACHE[key] = (value, size)
        while (len(_JSON_CACHE) > _JSON_CACHE_MAX_ENTRIES or
               sys.getsizeof(_JSON_CACHE) + sum(item[1] for item in _JSON_CACHE.values())
               > _JSON_CACHE_MAX_BYTES):
            _JSON_CACHE.popitem(last=False)
        return result
