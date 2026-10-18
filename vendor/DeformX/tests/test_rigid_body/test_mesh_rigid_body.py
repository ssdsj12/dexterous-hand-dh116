import numpy as np
from numpy.testing import assert_allclose
from elastica.mesh import Mesh
from elastica.rigidbody.mesh_rigid_body import MeshRigidBody


def test_actual_dh116_convex_pad_outside_point_is_not_negative():
    import hashlib
    from pathlib import Path
    import pytest
    path=Path('/home/xk/coding/M1/assets/dh116_v2/contact_hulls_v2/right_thumb.ply')
    if not path.exists():pytest.skip('local pinned DH116 asset unavailable')
    manifest=__import__('json').loads((path.parent/'manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['meshes']['right_thumb']['sha256']
    body=MeshRigidBody(Mesh(str(path)))
    query=np.array([[-.0163660437423104,.0004152284341645757,-.013940692021180976]])
    closest,distance,normal=body.query_closest_points(query)
    assert distance[0]>.013  # formerly -13.692 mm from a degenerate parity ray
    assert np.dot(query[0]-closest[0],normal[0])>0


def test_convex_halfspaces_preserve_transformed_inside_outside_signs():
    from scipy.spatial.transform import Rotation
    body=MeshRigidBody(Mesh(_centered_box(1.,2.,3.)))
    assert body.signed_distance_method=='convex-halfspaces'
    R=Rotation.from_rotvec([.4,-.2,.1]).as_matrix();p=np.array([.4,.1,1.])
    body.position_collection[:,0]=p;body.director_collection[:,:,0]=R.T
    local=np.array([[0,0,0],[.6,0,0],[0,1.1,0],[0,0,1.6]])
    cp,d,n=body.query_closest_points(local@R.T+p)
    assert d[0]<0 and np.all(d[1:]>0)
    assert_allclose(np.abs(d),[.5,.1,.1,.1],atol=1e-7)


def test_nonconvex_closed_mesh_is_not_silently_replaced_by_hull():
    import open3d as o3d
    mesh=Mesh(o3d.geometry.TriangleMesh.create_torus(torus_radius=.1,tube_radius=.025))
    body=MeshRigidBody(mesh)
    assert body.signed_distance_method=='ray-parity'
    _,d,_=body.query_closest_points(np.zeros((1,3)))
    assert d[0]>.07  # empty hole would be wrongly inside the convex hull


def _centered_box(width: float, height: float, depth: float):
    import open3d as o3d

    box = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
    box.compute_vertex_normals()
    box.compute_triangle_normals()
    box.translate(np.array([-0.5 * width, -0.5 * height, -0.5 * depth]))
    return box


def _uncentered_box(width: float, height: float, depth: float):
    import open3d as o3d

    box = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
    box.compute_vertex_normals()
    box.compute_triangle_normals()
    return box


def test_mesh_rigid_body_initialization():
    mesh = Mesh(_centered_box(2.0, 2.0, 2.0))
    body = MeshRigidBody(mesh=mesh, density=2.0, volume=mesh.compute_volume())

    assert_allclose(body.position_collection[:, 0], np.zeros(3), atol=1e-12)
    assert_allclose(body.mass, 16.0, atol=1e-12)
    assert_allclose(body.radius, 1.0, atol=1e-12)
    assert_allclose(body.length, 2.0, atol=1e-12)


def test_mesh_rigid_body_default_com_is_origin_without_estimation():
    mesh = Mesh(_uncentered_box(2.0, 2.0, 2.0))
    body = MeshRigidBody(mesh=mesh, density=2.0, volume=mesh.compute_volume())

    assert_allclose(body.position_collection[:, 0], np.zeros(3), atol=1e-12)


def test_query_closest_points_with_rotation():
    mesh = Mesh(_centered_box(1.0, 2.0, 3.0))
    body = MeshRigidBody(mesh=mesh, density=1.0, volume=mesh.compute_volume())

    r_mesh_from_world = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    body.director_collection[:, :, 0] = r_mesh_from_world

    query = np.array([[2.0, 0.0, 0.0]])
    closest, dists, normals = body.query_closest_points(query)

    expected_closest = np.array([[1.0, 0.0, 0.0]])
    expected_dist = np.array([1.0])
    assert_allclose(closest, expected_closest, atol=1e-12)
    assert_allclose(dists, expected_dist, atol=1e-12)
    assert_allclose(np.linalg.norm(normals, axis=1), np.ones(1), atol=1e-12)


def test_query_closest_points_with_translation():
    mesh = Mesh(_centered_box(1.0, 1.0, 1.0))
    com = np.array([0.5, 0.0, 0.0])
    body = MeshRigidBody(
        mesh=mesh, center_of_mass=com, density=1.0, volume=mesh.compute_volume()
    )

    query = np.array([[0.5, 0.0, 1.5]])
    closest, dists, _ = body.query_closest_points(query)

    expected_closest = np.array([[0.5, 0.0, 0.5]])
    expected_dist = np.array([1.0])
    assert_allclose(closest, expected_closest, atol=1e-12)
    assert_allclose(dists, expected_dist, atol=1e-12)


def test_query_closest_points_inside_returns_negative_signed_distance():
    mesh = Mesh(_centered_box(1.0, 1.0, 1.0))
    body = MeshRigidBody(mesh=mesh, density=1.0, volume=mesh.compute_volume())

    query = np.array([[0.0, 0.0, 0.0]])
    closest, dists, normals = body.query_closest_points(query)

    assert dists[0] < 0.0
    assert_allclose(np.abs(dists[0]), 0.5, atol=1e-12)
    assert_allclose(np.linalg.norm(normals, axis=1), np.ones(1), atol=1e-12)
    # Normal should point outward from the mesh surface.
    assert np.dot(closest[0] - query[0], normals[0]) > 0.0


def test_update_accelerations_and_energy():
    mesh = Mesh(_centered_box(1.0, 1.0, 1.0))
    body = MeshRigidBody(mesh=mesh, density=3.0, volume=mesh.compute_volume())

    external_forces = np.array([[3.0], [0.0], [0.0]])
    body.external_forces[:] = external_forces
    body.velocity_collection[:] = np.array([[2.0], [0.0], [0.0]])

    body.update_accelerations(time=0.0)
    assert_allclose(
        body.acceleration_collection, external_forces / body.mass, atol=1e-12
    )

    translational_energy = body.compute_translational_energy()
    assert_allclose(translational_energy, 0.5 * body.mass * 4.0, atol=1e-12)

    inertia_diag = np.diag(body.mass_second_moment_of_inertia[:, :, 0])
    body.omega_collection[:] = np.array([[1.0], [0.0], [0.0]])
    rotational_energy = body.compute_rotational_energy()
    assert_allclose(
        rotational_energy,
        0.5 * inertia_diag[0] * 1.0**2,
        atol=1e-12,
    )
