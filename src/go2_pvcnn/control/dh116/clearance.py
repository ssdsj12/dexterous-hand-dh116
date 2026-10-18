"""Conservative source-convex clearance certificates, not cooked PhysX geometry.

Failure means no certificate, not necessarily collision. Callers must include
all required source bodies, validate identity, and separately bound motion.
"""
import numpy as np


def _points(value):
    p=np.asarray(value,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)==0 or not np.isfinite(p).all():
        raise ValueError('expected nonempty finite 3D points')
    return p


def hull_wire_lower_bound(planes, nodes, radius, spacing):
    """Clearance lower bound for every segment of a radius-thick polyline.

    Each unit face plane gives a lower bound on distance to the convex solid.
    Their maximum is 1-Lipschitz. Sampling plus half-spacing subtraction covers
    intermediate points, unlike an endpoint-only collision test.
    Planes must be outward, n*x+b<=0 defining the source convex solid.
    """
    eq=np.asarray(planes,dtype=float);p=_points(nodes)
    if (eq.ndim!=2 or eq.shape[1]!=4 or len(eq)==0 or not np.isfinite(eq).all()
            or not np.allclose(np.linalg.norm(eq[:,:3],axis=1),1.,rtol=0,atol=1e-9)
            or len(p)<2 or not np.isfinite(radius) or radius<=0
            or not np.isfinite(spacing) or spacing<=0):
        raise ValueError('invalid planes, radius, spacing or polyline')
    best=np.inf
    for a,b in zip(p[:-1],p[1:]):
        length=float(np.linalg.norm(b-a))
        if not np.isfinite(length) or length/spacing>100000:
            raise ValueError('excessive/invalid segment sampling')
        count=max(1,int(np.ceil(length/spacing)))
        sample=a+np.linspace(0,1,count+1)[:,None]*(b-a)
        bound=float((sample@eq[:,:3].T+eq[:,3]).max(axis=1).min())
        best=min(best,bound-length/(2*count)-radius)
    if not np.isfinite(best):raise ValueError('nonfinite derived clearance')
    return float(best)


def axis_lower_bound(vertices_a,vertices_b,axis):
    """Re-evaluate a separating direction on CURRENT vertices, never cached poses."""
    a=_points(vertices_a);b=_points(vertices_b);n=np.asarray(axis,dtype=float)
    if n.shape!=(3,) or not np.isfinite(n).all() or not np.isclose(np.linalg.norm(n),1.,atol=1e-9,rtol=0):
        raise ValueError('invalid unit separating witness')
    pa=a@n;pb=b@n
    return float(max(pa.min()-pb.max(),pb.min()-pa.max()))


def separation_lower_bound(vertices_a, vertices_b, axes,*,return_axis=False):
    """Best supplied separating-plane bound; no edge-axis completeness claim."""
    a=_points(vertices_a);b=_points(vertices_b);n=_points(axes)
    lengths=np.linalg.norm(n,axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths<1e-12):
        raise ValueError('invalid projection axes')
    n=n/lengths[:,None]
    pa=a@n.T;pb=b@n.T
    bounds=np.maximum(pa.min(0)-pb.max(0),pb.min(0)-pa.max(0));index=int(np.argmax(bounds))
    bound=float(bounds[index])
    if not np.isfinite(bound):raise ValueError('nonfinite derived separation')
    return (bound,n[index].copy()) if return_axis else bound
