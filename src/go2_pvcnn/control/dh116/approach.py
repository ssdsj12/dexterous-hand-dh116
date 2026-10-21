"""Pinned nominal approach segment; runtime collision checks remain mandatory."""
import hashlib
import json
from pathlib import Path
import numpy as np


def _states(value):
    if set(value)!={'left','right'}:raise ValueError('both approach arms required')
    q={s:np.asarray(v,dtype=float) for s,v in value.items()}
    if any(v.shape!=(13,) or not np.isfinite(v).all() for v in q.values()):
        raise ValueError('finite 13D approach references required')
    return q


def approach_target(start,end,time_s,*,settle_s=.5,duration_s=1.2):
    a,b=_states(start),_states(end)
    if not np.isfinite([time_s,settle_s,duration_s]).all() or min(time_s,settle_s)<0 or duration_s<=0:
        raise ValueError('invalid approach time')
    speed_limit=np.r_[np.full(7,.1),np.full(6,.2)]
    if any(np.any(1.5*np.abs(b[s]-a[s])/duration_s>speed_limit) for s in a):
        raise ValueError('approach nominal speed exceeds configured cap')
    u=float(np.clip((time_s-settle_s)/duration_s,0,1));blend=u*u*(3-2*u)
    return {s:a[s]+blend*(b[s]-a[s]) for s in a}


def load_approach(path,sha256,current_layers):
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=sha256:raise ValueError('approach certificate SHA mismatch')
    data=json.loads(payload)
    if (data.get('static_wire_source_sweep_passed') is not True
            or data.get('motion_qualified') is not False or data.get('layers')!=current_layers):
        raise ValueError('invalid source-only approach certificate')
    intervals=sorted((x for x in data.get('covered_intervals',[]) if x.get('leg')==1),key=lambda x:x['start'])
    cursor=0.
    for interval in intervals:
        a,b,clearance=[float(interval[k]) for k in ('start','end','minimum_lower_bound_m')]
        if not np.isfinite([a,b,clearance]).all() or a!=cursor or not a<b<=1 or clearance<.001:
            raise ValueError('incomplete/invalid approach coverage')
        cursor=b
    if cursor!=1.:raise ValueError('missing approach coverage')
    points=data['input']['waypoints']
    start,end=_states(points[1]),_states(points[2])
    approach_target(start,end,0.)
    return start,end
