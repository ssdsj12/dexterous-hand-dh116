"""Pinned initial pose for a gravity/contact diagnostic, not a manipulation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .linear_coupling import expand


def load_active_pose(path, sha256, asset_sha256, current_layer_pins):
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=sha256:raise ValueError('active pose SHA mismatch')
    data=json.loads(payload)
    if (data.get('schema')!='dh116-static-pose-v1' or data.get('grasp_qualified') is not False
            or data.get('asset_sha256')!=asset_sha256 or data.get('layers')!=current_layer_pins
            or set(data.get('states',{}))!={'left','right'}):
        raise ValueError('active pose identity or diagnostic scope mismatch')
    states={s:np.asarray(q,dtype=float) for s,q in data['states'].items()}
    for q in states.values():
        if q.shape!=(13,) or not np.isfinite(q).all():raise ValueError('invalid active pose')
        expand(q[7:])
    return states
