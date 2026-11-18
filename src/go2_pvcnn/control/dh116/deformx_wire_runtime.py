"""Free Cosserat wire coupling; no Isaac imports or application startup.

Penalty contacts approximate Coulomb friction with coefficient 1.0; no rope
self-contact. All impulses are integrated at rod substeps before Isaac routing.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import ConvexHull
from vendor.DeformX.co_sim.gpu_cosserat import GpuRodConfig, TorchCudaCosseratRod


def paper_rod_config(center, table_center, table_size):
    center = np.asarray(center, dtype=float)
    return GpuRodConfig(n_elem=20, base_length=1., base_radius=.005,
        density=700., youngs_modulus=200000., shear_modulus_ratio=1.5, damping=.1,
        dt=1e-5, rod_start=tuple(center-[0,.5,0]), rod_direction=(0.,1.,0.),
        rod_normal=(0.,0.,1.), table_center=tuple(table_center),table_size=tuple(table_size),
        friction_coefficient=1., contact_subsamples=10,
        contact_stiffness=10000., contact_damping=10.)


def outward_hull(points):
    points = np.unique(np.asarray(points,dtype=float),axis=0)
    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    normals = np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
    flip = np.einsum('ij,ij->i',normals,hull.equations[:,:3]) < 0
    faces[flip] = faces[flip][:,[0,2,1]]
    return points,faces


def reaction_at_com(linear_impulse, angular_origin_impulse, com, dt):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('positive finite dt required')
    impulse = np.asarray(linear_impulse,dtype=float)
    torque = (np.asarray(angular_origin_impulse)-np.cross(com,impulse))/dt
    force = impulse/dt
    if not np.isfinite(force).all() or not np.isfinite(torque).all():
        raise FloatingPointError('nonfinite reaction wrench')
    return force,torque


def link_origin_velocity(v_com, omega, link_origin, com):
    return np.asarray(v_com)+np.cross(omega,np.asarray(link_origin)-com)


class GraspRegions:
    """Intervals selected once in initial material arclength, never reselected."""
    def __init__(self, initial_positions, grasp_points, half_width=.02):
        p = np.asarray(initial_positions,dtype=float).T
        delta = np.diff(p,axis=0)
        lengths = np.linalg.norm(delta,axis=1)
        if np.any(lengths <= 0) or half_width <= 0:
            raise ValueError('positive elements and region half-width required')
        self.s = np.r_[0,np.cumsum(lengths)]
        self.samples = []
        self.intervals = []
        for point in np.asarray(grasp_points):
            fraction = np.clip(np.einsum('ij,ij->i',point-p[:-1],delta)/lengths**2,0,1)
            projected = p[:-1]+fraction[:,None]*delta
            i = np.argmin(np.linalg.norm(projected-point,axis=1))
            center = self.s[i]+fraction[i]*lengths[i]
            lo,hi = max(0,center-half_width),min(self.s[-1],center+half_width)
            self.intervals.append([float(lo),float(hi)])
            self.samples.append(np.unique(np.r_[lo,self.s[(self.s>lo)&(self.s<hi)],hi]))

    def measure(self, positions, velocity, radius):
        p,v = np.asarray(positions),np.asarray(velocity)
        if p.shape != (3,len(self.s)) or v.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(v).all():
            raise FloatingPointError('invalid rod state')
        radii = np.asarray(radius,dtype=float)
        if radii.ndim == 0:
            radii = np.full(len(self.s)-1,float(radii))
        if radii.shape != (len(self.s)-1,) or not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError('finite positive scalar or per-element radius required')
        minimum,speed = [],[]
        for samples in self.samples:
            lo,hi = samples[0],samples[-1]
            bottoms = []
            for i,r in enumerate(radii):
                a,b = max(lo,self.s[i]),min(hi,self.s[i+1])
                if a <= b:
                    bottoms.append(float(np.interp([a,b],self.s,p[2]).min()-r))
            minimum.append(min(bottoms))
            velocities = np.array([np.interp(samples,self.s,a) for a in v])
            speed.append(float(np.linalg.norm(velocities,axis=0).max()))
        return dict(material_intervals_m=self.intervals,local_surface_min_z_m=minimum,
            local_max_speed_m_s=speed,global_surface_min_z_m=float((np.minimum(p[2,:-1],p[2,1:])-radii).min()))


class WireRuntime:
    def __init__(self, config, meshes, grasp_points, device):
        self.engine = TorchCudaCosseratRod(config,device=device)
        self.config,self.meshes = config,dict(meshes)
        for name,points in self.meshes.items():
            vertices,faces = outward_hull(points)
            self.engine.register_contact_mesh(name,vertices,faces)
        initial = self.engine.snapshot(device=False)
        self.regions = GraspRegions(initial['rod_position'],grasp_points)

    def advance(self, motions, centers_of_mass, body_channels, dt):
        state, impulses = self.engine.advance_with_meshes(motions,dt)
        channels = self.engine.last_contact_diagnostics['channels']
        reactions = {}
        max_balance = 0.
        for name,detail in channels.items():
            max_balance = max(max_balance,float(np.linalg.norm(detail['linear_balance_w'])),
                float(np.linalg.norm(detail['angular_origin_balance_w'])))
            body = body_channels[name]
            force,torque = reaction_at_com(detail['mesh_linear_w'],
                detail['mesh_angular_origin_w'],centers_of_mass[body],dt)
            if body not in reactions:
                reactions[body] = [np.zeros(3),np.zeros(3)]
            reactions[body][0] += force
            reactions[body][1] += torque
        if max_balance > 1e-7:
            raise FloatingPointError('contact reaction imbalance')
        maximum_force = max((np.linalg.norm(value[0]) for value in reactions.values()),default=0.)
        if maximum_force > 100:
            raise FloatingPointError(f'per-body mean reaction exceeds 100 N: {maximum_force}')
        segments = np.linalg.norm(np.diff(state['rod_position'],axis=1),axis=0)
        if np.min(segments) < .5*self.config.base_length/self.config.n_elem or np.max(segments) > 2*self.config.base_length/self.config.n_elem or np.max(np.abs(state['rod_velocity'])) > 100:
            raise FloatingPointError('divergent rod state')
        measurement = self.regions.measure(state['rod_position'],state['rod_velocity'],self.engine.radius.detach().cpu().numpy())
        tangent = np.diff(state['rod_position'],axis=1)/segments
        measurement.update(max_segment_length_deviation_m=float(np.max(np.abs(segments-self.config.base_length/self.config.n_elem))),
            max_director_centerline_shear=float(np.max(np.linalg.norm(state['rod_director'][2]-tangent,axis=0))))
        measurement.update(contact_detected=any(i.contact_duration_s>0 for i in impulses.values()),
            max_reaction_force_n=float(maximum_force),max_contact_balance_impulse=float(max_balance),
            contact_channels={n:dict(contact_duration_s=i.contact_duration_s,peak_force_n=i.peak_force_n,
                linear_impulse_w=i.linear_w.tolist(),angular_origin_impulse_w=channels[n]['mesh_angular_origin_w'].tolist())
                for n,i in impulses.items()})
        return state,reactions,measurement
