"""Work-conjugate geometry and mobility for the experimental contact solve.

Generalized arrays use node-major world translation followed by element-major
material angular coordinates. These functions never mutate a live rod.
"""
import numpy as np

from .implicit_contact import _array


def rod_mobility(mass, inverse_inertia, dilatation):
    """External-load acceleration response, including rod stretch factors."""
    mass = np.array(mass, dtype=float, copy=True)
    if mass.ndim != 1 or mass.size < 2 or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError('mass must contain positive finite node masses')
    elements = mass.size - 1
    inertia = _array(inverse_inertia, (3,3,elements), 'inverse_inertia')
    stretch = _array(dilatation, (elements,), 'dilatation')
    if np.any(stretch <= 0):
        raise ValueError('dilatation must be positive')
    dimension = 3*(mass.size+elements)
    result = np.zeros((dimension,dimension))
    for node, value in enumerate(mass):
        start = 3*node
        result[start:start+3,start:start+3] = np.eye(3)/value
    offset = 3*mass.size
    for element in range(elements):
        block = inertia[:,:,element]*stretch[element]
        if not np.allclose(block,block.T,rtol=1e-12,atol=1e-14):
            raise ValueError('inverse_inertia must be symmetric positive definite')
        try:
            np.linalg.cholesky(block)
        except np.linalg.LinAlgError as exc:
            raise ValueError('inverse_inertia must be positive definite') from exc
        start = offset+3*element
        result[start:start+3,start:start+3] = block
    if not np.isfinite(result).all():
        raise ValueError('mobility is not finite')
    return result


def contact_jacobians(positions, directors, contact_points, *, subsamples):
    """Map generalized velocity to exact kernel sample velocities.

    Samples are ordered element-major. The one-sample legacy endpoint weights
    deliberately match the existing kernel, including its first-element branch
    for a one-element rod. Quadrature force weights are NOT included in B.
    """
    position = np.array(positions,dtype=float,copy=True)
    if position.ndim != 2 or position.shape[0] != 3 or position.shape[1] < 2 or not np.isfinite(position).all():
        raise ValueError('positions must be finite 3 by node_count')
    if isinstance(subsamples,(bool,np.bool_)) or not isinstance(subsamples,(int,np.integer)) or subsamples < 1:
        raise ValueError('subsamples must be a positive integer')
    nodes=position.shape[1]; elements=nodes-1
    director=_array(directors,(3,3,elements),'directors')
    points=_array(contact_points,(elements*subsamples,3),'contact_points')
    for element in range(elements):
        d=director[:,:,element]
        if not np.allclose(d@d.T,np.eye(3),rtol=0,atol=1e-8) or not np.isclose(np.linalg.det(d),1.,rtol=0,atol=1e-8):
            raise ValueError('directors must be proper orthonormal rotations')
    result=np.zeros((elements*subsamples,3,3*(nodes+elements)))
    for sample, point in enumerate(points):
        element=sample//subsamples
        if subsamples > 1:
            right=(sample%subsamples+.5)/subsamples
        elif element == 0:
            right=2./3.
        elif element == elements-1:
            right=1./3.
        else:
            right=.5
        left=1.-right
        arm=point-(left*position[:,element]+right*position[:,element+1])
        x,y,z=arm
        skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        result[sample,:,3*element:3*element+3]=left*np.eye(3)
        result[sample,:,3*(element+1):3*(element+2)]=right*np.eye(3)
        offset=3*nodes+3*element
        result[sample,:,offset:offset+3]=-skew@director[:,:,element].T
    return result
