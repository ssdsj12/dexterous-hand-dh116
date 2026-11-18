import numpy as np
import pytest
import hashlib
from pathlib import Path
import open3d as o3d

from co_sim.implicit_stepper import assemble_contact_kick, ContactKick
from co_sim.moving_contact import MeshAsset, MovingContactRodEngine
from test_fixed_dual_panda_o6_moving_contact import CHANNELS, _config, _motion


@pytest.fixture
def assets(tmp_path):
    mesh=o3d.geometry.TriangleMesh.create_box(.02,.02,.02)
    mesh.translate([-.01,-.01,-.01]); mesh.compute_triangle_normals()
    path=tmp_path/'box.ply'; assert o3d.io.write_triangle_mesh(str(path),mesh,write_ascii=True)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    return {name:MeshAsset(path,digest) for name in CHANNELS}


def _problem():
    return dict(
        free_velocity=np.array([.001, 0., 0.]),
        mobility=np.eye(3),
        jacobians=np.eye(3)[None],
        boundary_velocity=np.zeros((1, 3)),
        normals=np.array([[0., 0., 1.]]),
        penetration=np.array([.0002]),
        stiffness=1e4, normal_damping=10., tangential_damping=1000.,
        friction=.5, weights=1., dt=1e-4,
    )


def test_kick_returns_work_conjugate_generalized_force_and_impulse():
    result = assemble_contact_kick(**_problem())
    assert isinstance(result, ContactKick)
    np.testing.assert_allclose(result.generalized_force, result.jacobians_force)
    np.testing.assert_allclose(result.impulse, result.generalized_force * 1e-4)
    assert result.applied_count == 1
    assert not result.impulse.flags.writeable


def test_multiple_contacts_share_one_joint_force():
    p = _problem()
    p['jacobians'] = np.repeat(np.eye(3)[None], 2, axis=0)
    p['boundary_velocity'] = np.zeros((2, 3))
    p['normals'] = np.array([[0., 0., 1.]] * 2)
    p['penetration'] = np.array([.0002, .0002])
    result = assemble_contact_kick(**p)
    np.testing.assert_allclose(result.generalized_force, result.jacobians_force)
    assert result.applied_count == 2


def test_reaction_is_equal_and_opposite_for_each_contact():
    p = _problem()
    result = assemble_contact_kick(**p)
    np.testing.assert_allclose(result.rod_force, result.mesh_force * -1.)
    np.testing.assert_allclose(result.rod_torque_origin, result.mesh_torque_origin * -1.)


def test_no_contact_has_zero_kick():
    p = _problem(); p['penetration'] = np.array([-1e-3])
    result = assemble_contact_kick(**p)
    np.testing.assert_array_equal(result.impulse, np.zeros(3))
    assert result.applied_count == 0


def test_invalid_dt_rejected():
    p = _problem(); p['dt'] = 0.
    with pytest.raises(ValueError):
        assemble_contact_kick(**p)


def test_implicit_engine_no_contact_matches_reference_verlet(assets):
    from co_sim.implicit_moving_contact import ImplicitMovingContactRodEngine
    kwargs=dict(table_center=(0,0,-1),table_size=(2,2,.1),meshes=assets,
                contact_subsamples=4,velocity_damping_coefficient=1000.)
    implicit=ImplicitMovingContactRodEngine(_config(py_dt=1e-5),**kwargs)
    explicit=MovingContactRodEngine(_config(py_dt=1e-5),**kwargs)
    motions={c:_motion(position=(10+i,10,10)) for i,c in enumerate(CHANNELS)}
    a,ai=implicit.advance_with_meshes(motions,1e-4)
    b,bi=explicit.advance_with_meshes(motions,1e-4)
    for field in ('rod_position','rod_velocity','rod_director','rod_omega'):
        np.testing.assert_allclose(a[field],b[field],atol=1e-12,rtol=1e-10)
    for c in CHANNELS:
        np.testing.assert_allclose(ai[c].linear_w,bi[c].linear_w,atol=1e-12)


def test_implicit_engine_records_pad_reaction_once(assets):
    from co_sim.implicit_moving_contact import ImplicitMovingContactRodEngine
    e=ImplicitMovingContactRodEngine(_config(py_dt=1e-5),table_center=(0,0,-1),
        table_size=(2,2,.1),meshes=assets,contact_subsamples=4,
        velocity_damping_coefficient=1000.)
    motions={c:_motion(position=(10+i,10,10)) for i,c in enumerate(CHANNELS)}
    motions['left_thumb']=_motion(position=(0,0,.183))
    _,impulses=e.advance_with_meshes(motions,1e-4)
    assert impulses['left_thumb'].peak_force_n>0
    assert e.last_solver_diagnostics['accepted_steps']==10
    assert e.last_solver_diagnostics['max_residual']<=1
