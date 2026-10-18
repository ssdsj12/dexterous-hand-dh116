"""Pinned stage-initialization artifacts, not opaque solver checkpoints.

The physics producer owns provenance qualification. The external SHA pins the
approved payload; runtime must additionally compare limits with the live asset.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import numpy as np
from .linear_coupling import JOINTS

PHASES = ('settle', 'approach', 'close', 'hold', 'lift')
ROD_SHAPES = dict(rod_position=(3,41), rod_velocity=(3,41),
                  rod_director=(3,3,40), rod_omega=(3,40))
REST_SHAPES = dict(rest_lengths=(40,), rest_voronoi_lengths=(39,),
                   rest_sigma=(3,40), rest_kappa=(3,39))


def array(value, shape):
    raw=np.asarray(value)
    if raw.dtype.kind not in 'fiu' or raw.shape!=shape or not np.isfinite(raw).all():
        raise ValueError('invalid finite physical array')
    return raw.astype(float)


def validate_state(payload, expected_pins):
    # Strict JSON both rejects NaN and makes an independent owned copy.
    p=json.loads(json.dumps(payload,allow_nan=False))
    if p['schema']!='dh116-stage-state-v1' or p['entry_stage'] not in PHASES[1:]:
        raise ValueError('unsupported stage-state schema or entry')
    if (not p['pins'] or p['pins']!=expected_pins
            or any(not isinstance(v,str) or re.fullmatch('[0-9a-f]{64}',v) is None for v in p['pins'].values())):
        raise ValueError('state configuration identity mismatch')
    source=p['source'];previous=list(PHASES[:PHASES.index(p['entry_stage'])])
    if (not isinstance(source['run'],str) or not source['run']
            or type(source['step']) is not int or source['step']<0
            or source['passed_stages']!=previous):
        raise ValueError('missing qualified predecessor stages')
    if set(p['robot'])!={'left','right'}:raise ValueError('two named articulations required')
    for side,r in p['robot'].items():
        names=r['joint_names']
        if len(names)!=18 or len(set(names))!=18 or set(names)!=set(JOINTS)|{f'{side}_panda_joint{i}' for i in range(1,8)}:
            raise ValueError('incorrect physical joint set')
        q,qd,target,lower,upper=[array(r[k],(18,)) for k in ('q','qd','target','lower','upper')]
        if np.any(lower>=upper) or np.any(q<lower) or np.any(q>upper) or np.any(target<lower) or np.any(target>upper):
            raise ValueError('saved joint state outside limits')
        root=array(r['root_state'],(13,))
        if not np.isclose(np.linalg.norm(root[3:7]),1.,atol=1e-6,rtol=0):
            raise ValueError('invalid root quaternion')
    rod=p['rod'];dynamic={k:array(rod['dynamic'][k],s) for k,s in ROD_SHAPES.items()}
    if not isinstance(rod['model_sha256'],str) or re.fullmatch('[0-9a-f]{64}',rod['model_sha256']) is None:
        raise ValueError('rod model identity required')
    rest={k:array(rod['reference'][k],s) for k,s in REST_SHAPES.items()}
    if np.any(rest['rest_lengths']<=0) or np.any(rest['rest_voronoi_lengths']<=0):
        raise ValueError('rest lengths must be positive')
    if np.any(np.linalg.norm(np.diff(dynamic['rod_position'],axis=1),axis=0)<1e-10):
        raise ValueError('collapsed rod element')
    frames=np.moveaxis(dynamic['rod_director'],-1,0)
    if not np.allclose(frames@frames.transpose(0,2,1),np.eye(3),atol=1e-6,rtol=0) or np.any(np.linalg.det(frames)<=0):
        raise ValueError('invalid material frame')
    ids=p['material_ids']
    if len(ids)!=4 or any(type(i)is not int or not 0<=i<=40 for i in ids) or set(ids[:2])&set(ids[2:]):
        raise ValueError('invalid fixed material regions')
    array(p['anchors_local'],(4,3));array(p['settled_heights_m'],(2,))
    loads=array(p['target_loads_n'],(4,))
    if np.any(loads<.005/.9) or np.any(loads>3.):raise ValueError('invalid force targets')
    return p


def save_state(path,payload):
    p=validate_state(payload,payload['pins'])
    data=json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    path=Path(path)
    # Hard-link publication is atomic and fails rather than replacing evidence.
    fd,tmp=tempfile.mkstemp(prefix='.stage-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        os.link(tmp,path)
        directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        os.unlink(tmp)
    return hashlib.sha256(data).hexdigest()


def load_state(path,sha256,expected_pins):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=sha256:raise ValueError('external state SHA mismatch')
    return validate_state(json.loads(data),expected_pins)
