"""Experimental joint velocity solve for frozen-geometry mesh contacts.

This module does not advance a rod or apply a robot reaction. The caller must
provide the actual dynamic mobility and work-conjugate contact Jacobians.
It is not yet enabled in MovingContactRodEngine.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContactVelocityResult:
    velocity: np.ndarray
    forces: np.ndarray
    iterations: int
    residual: float


def _array(value, shape, name):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    return result


def _parameter(value, count, name, *, positive=False):
    value = np.asarray(value, dtype=float)
    if value.ndim == 0:
        if not np.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f'invalid {name}')
        value = np.full(count, float(value))
    value = _array(value, (count,), name)
    if np.any(value < 0) or (positive and np.any(value == 0)):
        raise ValueError(f'invalid {name}')
    return value


def solve_contact_velocity(free_velocity, mobility, jacobians,
                           boundary_velocity, normals, penetration, *,
                           stiffness, normal_damping, tangential_damping,
                           friction, weights, dt, max_iterations=50):
    """Solve one backward-velocity contact kick without mutating any input.

    Residual is the maximum componentwise velocity error normalized by
    1e-8 + 1e-6 * max(abs(velocity), abs(free_velocity)). Forces include
    quadrature weights and act on the rod, in world coordinates.
    """
    free = np.array(free_velocity, dtype=float, copy=True)
    if free.ndim != 1 or free.size == 0 or not np.isfinite(free).all():
        raise ValueError('free_velocity must be a nonempty finite vector')
    dimension = free.size
    response = _array(mobility, (dimension, dimension), 'mobility')
    if not np.allclose(response, response.T, rtol=1e-12, atol=1e-14):
        raise ValueError('mobility must be symmetric positive definite')
    try:
        np.linalg.cholesky(response)
    except np.linalg.LinAlgError as exc:
        raise ValueError('mobility must be positive definite') from exc
    depths = np.asarray(penetration, dtype=float)
    if depths.ndim != 1:
        raise ValueError('penetration must be a vector')
    count = depths.size
    depths = _array(depths, (count,), 'penetration')
    b = _array(jacobians, (count, 3, dimension), 'jacobians')
    boundary = _array(boundary_velocity, (count, 3), 'boundary_velocity')
    normal = _array(normals, (count, 3), 'normals')
    if not np.allclose(np.linalg.norm(normal, axis=1), 1., rtol=0, atol=1e-10):
        raise ValueError('normals must have unit length')
    k = _parameter(stiffness, count, 'stiffness')
    nu = _parameter(normal_damping, count, 'normal_damping')
    c = _parameter(tangential_damping, count, 'tangential_damping')
    mu = _parameter(friction, count, 'friction')
    w = _parameter(weights, count, 'weights', positive=True)
    if np.ndim(dt) != 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 0:
        raise ValueError('max_iterations must be a nonnegative integer')

    def evaluate(velocity):
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            try:
                relative = np.einsum('nij,j->ni', b, velocity) - boundary
                vn = np.sum(relative * normal, axis=1)
                fn = np.where(depths > 0., np.maximum(0., k*depths - nu*vn), 0.)
                tangent = relative - vn[:, None]*normal
                speed = np.linalg.norm(tangent, axis=1)
                ft = -np.minimum(c*speed, mu*fn)[:, None]*tangent/(speed[:, None]+1e-14)
                forces = w[:, None]*(fn[:, None]*normal + ft)
                residual = velocity - free - dt*response @ np.einsum('nij,ni->j', b, forces)
            except FloatingPointError as exc:
                raise RuntimeError('nonfinite contact evaluation') from exc
        if not np.isfinite(residual).all() or not np.isfinite(forces).all():
            raise RuntimeError('nonfinite contact evaluation')
        return residual, forces

    velocity = free.copy()
    for iteration in range(max_iterations + 1):
        residual, forces = evaluate(velocity)
        scale = 1e-8 + 1e-6*np.maximum(np.abs(velocity), np.abs(free))
        error = float(np.max(np.abs(residual)/scale))
        if error <= 1.:
            velocity.setflags(write=False)
            forces.setflags(write=False)
            return ContactVelocityResult(velocity, forces, iteration, error)
        if iteration == max_iterations:
            break
        jacobian = np.empty((dimension, dimension))
        for column in range(dimension):
            step = np.cbrt(np.finfo(float).eps)*max(1., abs(velocity[column]))
            plus, minus = velocity.copy(), velocity.copy()
            plus[column] += step
            minus[column] -= step
            jacobian[:, column] = (evaluate(plus)[0] - evaluate(minus)[0])/(2*step)
        try:
            delta = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError('contact convergence failed: singular Jacobian') from exc
        if not np.isfinite(delta).all():
            raise RuntimeError('contact convergence failed: nonfinite update')
        for backtrack in range(24):
            candidate = velocity + (0.5**backtrack)*delta
            candidate_residual, _ = evaluate(candidate)
            if np.max(np.abs(candidate_residual)/scale) < error:
                velocity = candidate
                break
        else:
            raise RuntimeError('contact convergence failed: line search')
    raise RuntimeError('contact convergence failed: iteration limit')
