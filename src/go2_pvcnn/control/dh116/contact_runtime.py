"""DH116 identities and COM motion adapter for the generic DeformX impulse bridge."""
import hashlib
import json
from pathlib import Path
import numpy as np
from go2_pvcnn.control.fixed_dual_panda_o6.contact_bridge import BodyState

BODY_NAMES={f'{side}_{digit}':body for side in ('left','right')
            for digit,body in (('thumb','finger14_link'),('index','finger23_link'))}


def probe_rod_start(point,normal,direction,overlap_m=.0001):
    if not np.isfinite(overlap_m) or not 0<overlap_m<=.001:raise ValueError('invalid diagnostic overlap')
    p,n,t=(np.array(x,dtype=float,copy=True) for x in (point,normal,direction))
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in (p,n,t)) or np.linalg.norm(n)<1e-10:
        raise ValueError('invalid diagnostic point/frame')
    n/=np.linalg.norm(n);t-=n*np.dot(t,n)
    if np.linalg.norm(t)<1e-10:raise ValueError('parallel probe tangent/normal')
    t/=np.linalg.norm(t)
    # The backend samples element centers, not the 41 nodes. Center element 20.
    return p+n*(.008-overlap_m)-t*(20.5*.5/40),t


def load_mesh_manifest(path,expected_sha256):
    path=Path(path).resolve();payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected_sha256:raise ValueError('contact manifest identity mismatch')
    report=json.loads(payload)
    if report['schema']!='dh116-contact-hulls-v1':raise ValueError('wrong contact schema')
    for file,digest in report['layers'].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=digest:raise ValueError('source layer identity mismatch')
    records=report['meshes']
    if tuple(records)!=tuple(BODY_NAMES):raise ValueError('wrong contact channels/order')
    result={}
    for channel,record in records.items():
        if record['body_name']!=BODY_NAMES[channel]:raise ValueError('wrong DH116 body identity')
        target=(path.parent/record['path']).resolve()
        if target.parent!=path.parent:raise ValueError('contact path escapes bundle')
        if hashlib.sha256(target.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('derived mesh identity mismatch')
        offset=np.asarray(record['body_local_mesh_com_m'],dtype=float)
        if offset.shape!=(3,) or not np.isfinite(offset).all():raise ValueError('invalid mesh COM')
        result[channel]={**record,'resolved_path':target}
    return result


def _states(states):
    if tuple(states)!=tuple(BODY_NAMES):raise ValueError('wrong body channels/order')
    times=set();tokens=set()
    for c,s in states.items():
        if not isinstance(s,BodyState) or s.body_name!=BODY_NAMES[c]:raise ValueError('wrong DH116 body')
        times.add(s.sample_time_s);tokens.add(s.sample_token)
    if len(times)!=1 or len(tokens)!=1:raise ValueError('asynchronous body samples')
    return next(iter(times)),next(iter(tokens))


def mesh_motions(states,offsets,previous,dt,prediction_order=2):
    from co_sim.moving_contact_models import MeshMotion
    if prediction_order not in (1,2):raise ValueError('prediction order must be 1 or 2')
    if not np.isfinite(dt) or dt<=0:raise ValueError('invalid dt')
    now,token=_states(states)
    if tuple(offsets)!=tuple(BODY_NAMES):raise ValueError('wrong offset channels/order')
    if previous is not None:
        before,old_token=_states(previous)
        if token==old_token or not np.isclose(now-before,dt,atol=1e-12,rtol=1e-9):
            raise ValueError('stale motion sample')
    result={}
    for channel,s in states.items():
        offset=np.asarray(offsets[channel],dtype=float)
        if offset.shape!=(3,) or not np.isfinite(offset).all():raise ValueError('invalid COM offset')
        p=s.link_position_w+s.rotation_wb@offset
        v=s.physical_com_velocity_w+np.cross(s.omega_w,p-s.physical_com_position_w)
        a=np.zeros(3);alpha=np.zeros(3)
        if previous is not None and prediction_order==2:
            old=previous[channel];old_p=old.link_position_w+old.rotation_wb@offset
            old_v=old.physical_com_velocity_w+np.cross(old.omega_w,old_p-old.physical_com_position_w)
            a=(v-old_v)/dt;alpha=(s.omega_w-old.omega_w)/dt
        result[channel]=MeshMotion(p,s.rotation_wb,v,s.omega_w,a,alpha)
    return result
