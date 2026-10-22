"""Local non-pad wire clearances and derivatives for hand-only planning.

Rigid body poses and 6x6 [linear; angular] Jacobians are world-frame and
referenced at each link origin. Their columns are DH116 ACTIVE only. This
is a source-geometry observation, not a deformable-rod step or a safety gate.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .clearance import hull_wire_lower_bound


def hand_wire_clearance_observations(bodies, wire_nodes, *, radius, activation_distance=.003,
                                     finite_difference_step=1e-5):
    nodes=np.asarray(wire_nodes,dtype=float)
    if (nodes.ndim!=2 or nodes.shape[1]!=3 or len(nodes)<2 or not np.isfinite(nodes).all()
            or not np.isfinite([radius,activation_distance,finite_difference_step]).all()
            or min(radius,activation_distance,finite_difference_step)<=0):
        raise ValueError('finite wire geometry and positive observation settings required')
    distances=[];gradients=[];names=[]
    wire_low=nodes.min(0);wire_high=nodes.max(0)
    for body in bodies:
        if body['side'] not in ('left','right'):raise ValueError('named hand side required')
        vertices=np.asarray(body['vertices'],dtype=float)
        planes=np.asarray(body['planes'],dtype=float)
        rotation=np.asarray(body['rotation'],dtype=float);position=np.asarray(body['position'],dtype=float)
        jac=np.asarray(body['jacobian'],dtype=float)
        if jac.shape!=(6,6) or not np.isfinite(jac).all():
            raise ValueError('body Jacobian must contain only six active hand columns')
        if (position.shape!=(3,) or rotation.shape!=(3,3) or not np.isfinite(position).all()
                or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6,rtol=0)
                or not np.isclose(np.linalg.det(rotation),1.,atol=1e-6,rtol=0)):
            raise ValueError('proper measured body pose required')
        effective_radius=radius+float(body.get('motion_allowance_m',0.))
        world=vertices@rotation.T+position
        separation=np.maximum(np.maximum(world.min(0)-wire_high,wire_low-world.max(0)),0.)
        if np.linalg.norm(separation)-effective_radius>=activation_distance:continue

        def measure(R,p):
            return hull_wire_lower_bound(planes,(nodes-p)@R,effective_radius,.0005,
                                         refine_below=activation_distance,refinement_steps=40)

        distance=measure(rotation,position)
        if distance>=activation_distance:continue
        gradient=np.zeros(12)
        start=0 if body['side']=='left' else 6
        for column in range(6):
            if np.max(np.abs(jac[:,column]))<=1e-12:continue
            h=finite_difference_step
            v=jac[:3,column];omega=jac[3:,column]
            plus=measure(Rotation.from_rotvec(h*omega).as_matrix()@rotation,position+h*v)
            minus=measure(Rotation.from_rotvec(-h*omega).as_matrix()@rotation,position-h*v)
            gradient[start+column]=(plus-minus)/(2*h)
        distances.append(distance);gradients.append(gradient);names.append(body['name'])
    return dict(clearances=np.asarray(distances,dtype=float),
        clearance_jacobians=np.asarray(gradients,dtype=float).reshape(-1,12),bodies=names)
