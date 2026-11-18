"""Explicit simulation contact evidence, separate from controller targets."""
import numpy as np


def normal_contact_sample(*,force_on_mesh,normal,gaps,relative_normal_speeds,contact_duration_s,dt,element_ids=None):
    f=np.asarray(force_on_mesh,dtype=float);n=np.asarray(normal,dtype=float)
    g=np.asarray(gaps,dtype=float);v=np.asarray(relative_normal_speeds,dtype=float)
    if (f.shape!=(3,) or n.shape!=(3,) or g.ndim!=1 or not len(g) or v.shape!=g.shape
            or any(not np.isfinite(a).all() for a in (f,n,g,v))
            or not np.isclose(np.linalg.norm(n),1.,atol=1e-6,rtol=0)
            or not np.isfinite([dt,contact_duration_s]).all() or not 0<dt<=.0001
            or not 0<=contact_duration_s<=dt+1e-12):
        raise ValueError('invalid loaded-contact evidence')
    ids=np.arange(len(g)) if element_ids is None else np.asarray(element_ids)
    if ids.shape!=g.shape or not np.issubdtype(ids.dtype,np.integer) or (ids<0).any():
        raise ValueError('invalid contact quadrature parent IDs')
    i=int(np.argmin(g));load=float(-f@n)
    valid=bool(load>=.005 and contact_duration_s>=.99*dt and g[i]<=1e-7 and v[i]<=.001)
    return dict(valid=valid,normal_load_n=load,closest_element=int(ids[i]),closest_sample=i,gap_m=float(g[i]),
                separation_speed_m_s=float(v[i]),contact_duration_s=float(contact_duration_s),
                noise_floor_n=.005,separation_tolerance_m_s=.001)
