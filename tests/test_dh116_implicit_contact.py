"""Analytic tests for the experimental frozen-geometry contact solve."""
import importlib.util

import numpy as np
import pytest


def solve(**changes):
    assert importlib.util.find_spec('co_sim.implicit_contact') is not None, 'implicit solver missing'
    from co_sim.implicit_contact import solve_contact_velocity
    args = dict(free_velocity=np.array([.001, 0., 0.]), mobility=np.eye(3),
                jacobians=np.eye(3)[None], boundary_velocity=np.zeros((1, 3)),
                normals=np.array([[0., 0., 1.]]), penetration=np.array([.0002]),
                stiffness=1e4, normal_damping=10., tangential_damping=1000.,
                friction=.5, weights=1., dt=1e-4)
    args.update(changes)
    return solve_contact_velocity(**args)


def test_implicit_normal_and_unsaturated_friction_have_analytic_solution():
    r = solve()
    np.testing.assert_allclose(r.velocity, [.001/1.1, 0., .0002/1.001], atol=1e-10)
    assert r.residual <= 1
    assert not r.velocity.flags.writeable
    assert not r.forces.flags.writeable


def test_shared_contacts_solve_one_common_velocity():
    r = solve(jacobians=np.repeat(np.eye(3)[None], 2, axis=0),
              boundary_velocity=np.zeros((2, 3)), normals=np.array([[0.,0.,1.]]*2),
              penetration=np.array([.0002]*2))
    np.testing.assert_allclose(r.velocity, [.001/1.2, 0., .0004/1.002], atol=1e-10)


def test_quadrature_weights_scale_normal_and_tangential_force():
    r = solve(weights=.25)
    np.testing.assert_allclose(r.velocity, [.001/1.025, 0., .00005/1.00025], atol=1e-10)


def test_sliding_friction_is_coulomb_limited():
    r = solve(free_velocity=np.array([1., 0., 0.]))
    assert r.forces[0, 0] == pytest.approx(-.5*r.forces[0, 2], abs=1e-10)
    assert r.velocity[0] == pytest.approx(1. + 1e-4*r.forces[0, 0], abs=1e-10)


def test_moving_boundary_uses_relative_velocity():
    r = solve(free_velocity=np.zeros(3), boundary_velocity=np.array([[.001,0.,0.]]))
    assert r.velocity[0] == pytest.approx(.0001/1.1, abs=1e-10)
    assert r.forces[0, 0] > 0


@pytest.mark.parametrize('depth', [0., -.001])
def test_no_contact_even_if_approaching_fast(depth):
    v = np.array([.001, 0., -10.])
    r = solve(penetration=np.array([depth]), free_velocity=v)
    np.testing.assert_array_equal(r.velocity, v)
    np.testing.assert_array_equal(r.forces, np.zeros((1,3)))


def test_no_contacts_preserves_free_velocity():
    r = solve(jacobians=np.zeros((0,3,3)), boundary_velocity=np.zeros((0,3)),
              normals=np.zeros((0,3)), penetration=np.zeros(0))
    np.testing.assert_array_equal(r.velocity, [.001,0.,0.])


def test_anisotropic_mobility_is_included():
    r = solve(mobility=np.diag([2., 1., 3.]))
    np.testing.assert_allclose(r.velocity, [.001/1.2, 0., .0006/1.003], atol=1e-10)


def test_stiff_damping_does_not_reverse_sliding_direction():
    r = solve(tangential_damping=1e7, friction=1e4, dt=.01)
    assert 0 <= r.velocity[0] < 1e-7
    assert r.forces[0,0]*r.velocity[0] <= 0


def test_nonconvergence_raises_and_leaves_inputs_unchanged():
    v = np.array([.001,0.,0.]); original=v.copy()
    with pytest.raises(RuntimeError, match='converg'):
        solve(free_velocity=v, max_iterations=0)
    np.testing.assert_array_equal(v, original)


@pytest.mark.parametrize('changes', [dict(dt=0), dict(friction=-1),
    dict(weights=0), dict(mobility=np.diag([1.,-1.,1.])),
    dict(normals=np.array([[0.,0.,2.]])), dict(penetration=np.array([np.nan])),
    dict(jacobians=np.ones((1,2,3))), dict(max_iterations=1.5)])
def test_invalid_problem_rejected(changes):
    with pytest.raises(ValueError):
        solve(**changes)


def test_world_rotation_preserves_generalized_solution():
    base = solve()
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    rotated = solve(jacobians=rotation[None], normals=(rotation @ np.array([0.,0.,1.]))[None])
    np.testing.assert_allclose(rotated.velocity, base.velocity, atol=1e-10)
    np.testing.assert_allclose(rotated.forces, base.forces @ rotation.T, atol=1e-10)


def test_rotational_dof_and_shared_contacts_satisfy_momentum_equation():
    # Generalized velocity [x,y,z,omega_z], opposite lever arms.
    b = np.zeros((2,3,4)); b[:,:,:3] = np.eye(3)
    b[0,0,3] = -.02; b[1,0,3] = .02
    free = np.array([.001,0.,0.,.01])
    mobility = np.diag([1.,1.,1.,100.])
    r = solve(free_velocity=free, mobility=mobility, jacobians=b,
              boundary_velocity=np.zeros((2,3)), normals=np.array([[0.,0.,1.]]*2),
              penetration=np.array([.0002]*2))
    generalized_force = np.einsum('nij,ni->j', b, r.forces)
    np.testing.assert_allclose(r.velocity-free, 1e-4*mobility @ generalized_force, atol=1e-9)
    assert abs(r.velocity[3]) < abs(free[3])
    point_velocity = np.einsum('nij,j->ni', b, r.velocity)
    assert np.sum(point_velocity[:,:2]*r.forces[:,:2]) <= 0


def test_separating_contact_has_no_adhesion():
    free = np.array([.001,0.,1.])
    r = solve(free_velocity=free)
    np.testing.assert_array_equal(r.velocity, free)
    np.testing.assert_array_equal(r.forces, np.zeros((1,3)))
