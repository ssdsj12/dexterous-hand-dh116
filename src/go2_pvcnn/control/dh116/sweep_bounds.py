"""Global link-point displacement bounds for serial revolute USD joint trees."""
import numpy as np


def point_rotation_radii(edges, body, vertices, names):
    """Per-physical-joint upper bounds on distance to its rotation axis.

    Sum intervening frame translation lengths and local vertex radius. This
    holds at every joint angle, so sum(radius*abs(delta_q)) bounds a path, not
    only its endpoints. Joint frames/axes are validated by SideKinematics.
    """
    v=np.asarray(vertices,dtype=float)
    if v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all():
        raise ValueError('invalid body geometry')
    parents={e[2]:e for e in edges}
    known={e[1] for e in edges}|set(parents)
    if body not in known or len(parents)!=len(edges) or len(set(names))!=len(names):
        raise ValueError('invalid tree, names or body')
    chain=[];seen=set();current=body
    while current in parents:
        if current in seen:raise ValueError('cyclic joint tree')
        seen.add(current);e=parents[current];chain.append(e);current=e[1]
    radius=float(np.linalg.norm(v,axis=1).max());result=np.zeros(len(names))
    for name,parent,child,t0,t1,axis in chain:
        t0=np.asarray(t0,dtype=float);t1=np.asarray(t1,dtype=float)
        if t0.shape!=(4,4) or t1.shape!=(4,4) or not np.isfinite([t0,t1]).all():
            raise ValueError('invalid joint frame')
        radius+=float(np.linalg.norm(t1[:3,3]))
        if axis is not None:
            if name not in names:raise ValueError('missing physical joint')
            result[names.index(name)]=radius
        radius+=float(np.linalg.norm(t0[:3,3]))
    return result


def relative_motion_bound(radii_a, radii_b, half_delta, *, same_tree):
    """Common ancestors are a shared rigid transform and preserve distance."""
    a=np.asarray(radii_a,dtype=float);b=np.asarray(radii_b,dtype=float);d=np.asarray(half_delta,dtype=float)
    if (a.ndim!=1 or not len(a) or a.shape!=b.shape or a.shape!=d.shape
            or not np.isfinite([a,b,d]).all() or np.any(a<0) or np.any(b<0) or np.any(d<0)
            or type(same_tree) is not bool):
        raise ValueError('invalid radii or half-delta')
    r=a+b
    if same_tree:r=np.where((a>0)&(b>0),0.,r)
    return float(r@d)
