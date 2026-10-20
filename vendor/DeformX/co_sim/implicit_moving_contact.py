"""Experimental MovingContact backend using one joint semi-implicit kick."""
from __future__ import annotations

import numpy as np
import elastica as ea

from .implicit_contact import solve_contact_velocity
from .implicit_contact_adapter import rod_mobility, contact_jacobians
from .moving_contact import (
    CHANNELS, MovingContactRodEngine, _PrescribedMeshBC,
)


class _ImplicitContactForcing(ea.NoForces):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def apply_forces(self, system, time=np.float64(0.0)):
        self.engine._apply_implicit_contacts(system, float(time))


class ImplicitMovingContactRodEngine(MovingContactRodEngine):
    """Moving-contact rod with joint frozen-geometry velocity contact.

    This is opt-in and deliberately keeps the original MovingContact engine as
    the reference implementation. The original explicit contact operators are
    not registered in this subclass.
    """

    def _setup_simulator(self):
        size = self.table_size
        # Reuse the same finite table construction as TableRodEngine without
        # registering its explicit RodMeshContact operator.
        import open3d as o3d
        table_geometry = o3d.geometry.TriangleMesh.create_box(*map(float, size))
        table_geometry.translate(-0.5 * size)
        table_mesh = ea.Mesh(table_geometry)
        self.table_body = ea.MeshRigidBody(
            mesh=table_mesh, center_of_mass=self.table_center,
            density=1.0, volume=table_mesh.compute_volume(),
        )
        self.sim.append(self.table_body)
        self.sim.constrain(self.table_body).using(
            ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
        )
        self.mesh_bodies = {}
        for channel in CHANNELS:
            _, mesh, volume, inertia = self._prepared_meshes[channel]
            body = ea.MeshRigidBody(
                mesh=mesh, center_of_mass=np.zeros(3), density=1.0,
                volume=volume, mass_second_moment_of_inertia=inertia,
            )
            self.mesh_bodies[channel] = body
            self.sim.append(body)
            self.sim.constrain(body).using(_PrescribedMeshBC, engine=self, channel=channel)
        self.sim.add_forcing_to(self.rod).using(_ImplicitContactForcing, engine=self)

    @staticmethod
    def _sample_mesh(body, samples):
        closest, distances, normals = body.query_closest_points(samples)
        return (np.asarray(closest, dtype=float), np.asarray(distances, dtype=float),
                np.asarray(normals, dtype=float))

    def _apply_implicit_contacts(self, rod, time):
        if self._active_step is None:
            raise RuntimeError('implicit contact callback outside an active step')
        n = self.contact_subsamples
        sample_positions = (rod.position_collection[:, :-1, None] *
                            (1-(np.arange(n)+.5)/n) +
                            rod.position_collection[:, 1:, None] *
                            ((np.arange(n)+.5)/n)).reshape(3, -1).T
        directors = rod.director_collection
        jacobian_one = contact_jacobians(
            rod.position_collection, directors, sample_positions, subsamples=n
        )
        mobility = rod_mobility(rod.mass, rod.inv_mass_second_moment_of_inertia,
                                rod.dilatation)
        velocity = np.concatenate((rod.velocity_collection.T.ravel(), rod.omega_collection.T.ravel()))
        base_force = np.concatenate((rod.internal_forces.T.ravel(), rod.internal_torques.T.ravel()))
        base_force += np.concatenate((rod.external_forces.T.ravel(), rod.external_torques.T.ravel()))
        free_velocity = velocity + float(self._active_step) * (mobility @ base_force)
        all_jacobians=[]; all_boundaries=[]; all_normals=[]; all_penetrations=[]
        mesh_data=[]
        bodies=[('table', self.table_body)] + [(c, self.mesh_bodies[c]) for c in CHANNELS]
        for name, body in bodies:
            cp, distance, normal = self._sample_mesh(body, sample_positions)
            if name == 'table':
                mesh_velocity=np.zeros((len(cp),3)); mesh_body_position=body.position_collection[:,0]
            else:
                mesh_velocity=np.repeat(body.velocity_collection[:,0][None,:],len(cp),axis=0)
                omega_world=body.director_collection[:,:,0].T @ body.omega_collection[:,0]
                mesh_velocity += np.cross(omega_world, cp-body.position_collection[:,0])
                mesh_body_position=body.position_collection[:,0]
            # The same work-conjugate rod Jacobian is used for every mesh.
            all_jacobians.append(jacobian_one)
            all_boundaries.append(mesh_velocity)
            all_normals.append(normal)
            all_penetrations.append(rod.radius.repeat(n)-distance)
            mesh_data.append((name, body, cp, mesh_body_position))
        jac=np.concatenate(all_jacobians); boundary=np.concatenate(all_boundaries)
        normals=np.concatenate(all_normals); penetration=np.concatenate(all_penetrations)
        solve=solve_contact_velocity(
            free_velocity, mobility, jac, boundary, normals, penetration,
            stiffness=self.contact_k, normal_damping=self.contact_nu,
            tangential_damping=self.velocity_damping_coefficient,
            friction=self.friction_coefficient, weights=1./n,
            dt=float(self._active_step),
        )
        generalized=np.einsum('nij,ni->j', jac, solve.forces)
        node_count=rod.position_collection.shape[1]
        rod.external_forces += generalized[:3*node_count].reshape(node_count,3).T
        rod.external_torques += generalized[3*node_count:].reshape(rod.n_elems,3).T
        table_reaction=np.zeros(3)
        for index,(name,body,cp,body_position) in enumerate(mesh_data):
            start=index*len(sample_positions); stop=start+len(sample_positions)
            mesh_force=-solve.forces[start:stop].sum(axis=0)
            mesh_torque_world=np.cross(cp-body_position, -solve.forces[start:stop]).sum(axis=0)
            if name == 'table':
                table_reaction += mesh_force * float(self._active_step)
                continue
            body.external_forces[:,0] += mesh_force
            body.external_torques[:,0] += body.director_collection[:,:,0] @ mesh_torque_world
            acc=self._accumulators[name]
            acc['mesh_linear'] += mesh_force*float(self._active_step)
            acc['rod_linear'] += (-mesh_force)*float(self._active_step)
            acc['mesh_angular_com'] += mesh_torque_world*float(self._active_step)
            acc['evaluations'] += 1
            acc['peak']=max(acc['peak'],float(np.linalg.norm(mesh_force)))
            if np.linalg.norm(mesh_force)>0: acc['duration'] += float(self._active_step)
        self._expected_channel_index=len(CHANNELS)
        self.last_solver_diagnostics={
            'accepted_steps': self._step_count,
            'max_residual': float(solve.residual),
            'iterations': int(solve.iterations),
            'table_linear_impulse_w': table_reaction,
        }
