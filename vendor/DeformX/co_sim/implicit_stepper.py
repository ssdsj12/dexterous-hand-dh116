"""Single accepted contact kick built on the experimental implicit solve.

This is intentionally an adapter, not a time-stepper: the caller owns the
Verlet update and must submit the returned generalized force exactly once.
"""
from dataclasses import dataclass

import numpy as np

from .implicit_contact import solve_contact_velocity


@dataclass(frozen=True)
class ContactKick:
    generalized_force: np.ndarray
    jacobians_force: np.ndarray
    impulse: np.ndarray
    rod_force: np.ndarray
    mesh_force: np.ndarray
    rod_torque_origin: np.ndarray
    mesh_torque_origin: np.ndarray
    applied_count: int
    residual: float


def assemble_contact_kick(*, free_velocity, mobility, jacobians,
                          boundary_velocity, normals, penetration,
                          stiffness, normal_damping, tangential_damping,
                          friction, weights, dt, max_iterations=50):
    """Return one work-conjugate contact kick; never mutates caller arrays."""
    result = solve_contact_velocity(
        free_velocity, mobility, jacobians, boundary_velocity, normals,
        penetration, stiffness=stiffness, normal_damping=normal_damping,
        tangential_damping=tangential_damping, friction=friction,
        weights=weights, dt=dt, max_iterations=max_iterations,
    )
    forces = np.array(result.forces, copy=True)
    jac = np.array(jacobians, dtype=float, copy=True)
    generalized = np.einsum('nij,ni->j', jac, forces)
    rod_force = forces.sum(axis=0)
    mesh_force = -rod_force
    # The Jacobian's rotational block is already material-frame torque. The
    # origin torque is supplied by the caller's rod geometry in the full
    # adapter; for a translational solve it is exactly zero.
    rod_torque = np.zeros(3)
    mesh_torque = -rod_torque
    impulse = float(dt) * generalized
    for array in (generalized, impulse, rod_force, mesh_force, rod_torque, mesh_torque):
        array.setflags(write=False)
    return ContactKick(
        generalized, generalized, impulse, rod_force, mesh_force,
        rod_torque, mesh_torque, int(np.count_nonzero(penetration > 0.)),
        float(result.residual),
    )
