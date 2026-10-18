import pytest
import numpy as np
from numpy.testing import assert_allclose

from elastica.mesh import Mesh
from elastica.rigidbody.mesh_rigid_body import MeshRigidBody
from elastica.contact_forces import RodMeshContact
from elastica._contact_functions import _calculate_contact_forces_rod_mesh


def _centered_box(width: float, height: float, depth: float):
    import open3d as o3d

    box = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
    box.compute_vertex_normals()
    box.compute_triangle_normals()
    box.translate(np.array([-0.5 * width, -0.5 * height, -0.5 * depth]))
    return box


def test_sub_element_contact_finds_small_patch_between_old_midpoints():
    mesh=Mesh(_centered_box(.004,.02,.02))
    body=MeshRigidBody(mesh=mesh,center_of_mass=np.array([.018,0.,.183]),density=1.,volume=mesh.compute_volume())
    nodes=np.array([[-.025,.025],[0.,0.],[.2,.2]])
    old=_MockRod(nodes.copy(),radius=.008)
    RodMeshContact(k=1e4,nu=0.).apply_contact(old,body)
    assert_allclose(old.external_forces,0.,atol=1e-12)
    refined=_MockRod(nodes.copy(),radius=.008)
    RodMeshContact(k=1e4,nu=0.,subsamples_per_element=16).apply_contact(refined,body)
    assert np.linalg.norm(refined.external_forces)>0.
    assert_allclose(refined.external_forces.sum(1)+body.external_forces[:,0],0.,atol=1e-12)
    moment=np.cross(nodes.T,refined.external_forces.T).sum(0)+refined.external_torques.sum(1)
    moment+=np.cross(body.position_collection[:,0],body.external_forces[:,0])+body.director_collection[:,:,0].T@body.external_torques[:,0]
    assert_allclose(moment,0.,atol=1e-12)


@pytest.mark.parametrize('count',[1,4,16,32])
def test_sub_element_weights_preserve_uniform_contact_total_stiffness(count):
    mesh=Mesh(_centered_box(1.,1.,1.))
    body=MeshRigidBody(mesh=mesh,density=1.,volume=mesh.compute_volume())
    rod=_MockRod(np.array([[-.025,.025],[0.,0.],[.508,.508]]),radius=.01)
    RodMeshContact(k=1000.,nu=0.,subsamples_per_element=count).apply_contact(rod,body)
    assert_allclose(rod.external_forces.sum(1),[0,0,2.],atol=1e-5)


@pytest.mark.parametrize('count',[0,-1,1.5,True,256])
def test_invalid_contact_subsample_count_rejected(count):
    with pytest.raises(ValueError):RodMeshContact(k=1000.,nu=0.,subsamples_per_element=count)


class _MockRod:
    def __init__(self, node_positions: np.ndarray, radius: float):
        self.position_collection = node_positions
        self.velocity_collection = np.zeros_like(node_positions)
        self.radius = np.full(node_positions.shape[1] - 1, radius, dtype=np.float64)
        self.external_forces = np.zeros_like(node_positions)
        n_elems = node_positions.shape[1] - 1
        self.external_torques = np.zeros((3, n_elems), dtype=np.float64)
        self.director_collection = np.repeat(np.eye(3)[:, :, None], n_elems, axis=2)
        self.omega_collection = np.zeros((3, n_elems), dtype=np.float64)


@pytest.mark.parametrize('count',[1,4,16])
def test_optional_broad_phase_preserves_rotated_moving_contact_wrench(count):
    from scipy.spatial.transform import Rotation
    mesh=Mesh(_centered_box(.03,.025,.04))
    R=Rotation.from_rotvec([.3,-.2,.4]).as_matrix()
    p=np.array([.43,.15,.3])
    local=np.array([np.linspace(-.25,.25,41),np.zeros(41),np.full(41,.026)])
    output=[]
    for broad in (False,True):
        body=MeshRigidBody(mesh=mesh,center_of_mass=p,density=1.,volume=mesh.compute_volume())
        body.director_collection[:,:,0]=R.T
        body.velocity_collection[:,0]=[.01,-.02,.03]
        body.omega_collection[:,0]=[.2,.1,-.3]
        rod=_MockRod(R@local+p[:,None],radius=.008)
        rod.velocity_collection[:]=np.array([.02,.03,-.01])[:,None]
        rod.omega_collection[:]=np.array([.2,-.1,.3])[:,None]
        RodMeshContact(k=1e4,nu=10.,friction_coefficient=.5,velocity_damping_coefficient=1000.,
            subsamples_per_element=count,broad_phase=broad).apply_contact(rod,body)
        output.append([rod.external_forces.copy(),rod.external_torques.copy(),
                       body.external_forces.copy(),body.external_torques.copy()])
    assert np.linalg.norm(output[0][0])>0
    for a,b in zip(*output):assert_allclose(a,b,atol=1e-12,rtol=1e-12)


def test_broad_phase_keeps_inside_and_near_surface_queries():
    mesh=Mesh(_centered_box(.02,.02,.02))
    body=MeshRigidBody(mesh=mesh,density=1.,volume=mesh.compute_volume())
    points=np.array([[0,0,0],[.018-1e-9,0,0],[.018+1e-9,0,0],[.2,.3,.4]])
    exact=body.query_closest_points(points)
    bounded=body.query_contact_points(points,np.full(4,.008))
    for a,b in zip(exact,bounded):assert_allclose(a[:3],b[:3],atol=1e-12,rtol=0)
    assert bounded[1][3]>.008


@pytest.mark.parametrize('radii',[[-.1], [np.nan], [.01,.02]])
def test_broad_phase_rejects_invalid_radius(radii):
    mesh=Mesh(_centered_box(.02,.02,.02))
    body=MeshRigidBody(mesh=mesh,density=1.,volume=mesh.compute_volume())
    with pytest.raises(ValueError):body.query_contact_points(np.zeros((1,3)),radii)


def _run_single_element_kernel(
    *,
    penetration=1.0,
    rod_velocity=(0.0, 0.0, 0.0),
    mesh_velocity=(0.0, 0.0, 0.0),
    mesh_angular_velocity=(0.0, 0.0, 0.0),
    closest_point=(1.0, 0.0, 0.0),
    normal=(0.0, 0.0, 1.0),
    rod_nodes=None,
    rod_node_velocities=None,
    rod_angular_velocity=(0.0, 0.0, 0.0),
    rod_director=None,
    mesh_director=None,
    mesh_frozen=False,
    k=1.0,
    nu=1.0,
    velocity_damping_coefficient=0.0,
    friction_coefficient=0.0,
    subsamples_per_element=1,
):
    rod_external_forces = np.zeros((3, 2), dtype=np.float64)
    rod_external_torques = np.zeros((3, 1), dtype=np.float64)
    mesh_external_forces = np.zeros((3, 1), dtype=np.float64)
    mesh_external_torques = np.zeros((3, 1), dtype=np.float64)
    radius = 1.0
    if mesh_director is None:
        mesh_director = np.eye(3, dtype=np.float64)
    if rod_director is None:
        rod_director = np.eye(3, dtype=np.float64)
    if rod_nodes is None:
        rod_nodes = np.zeros((3, 2), dtype=np.float64)
    if rod_node_velocities is None:
        rod_node_velocities = np.repeat(
            np.asarray(rod_velocity, dtype=np.float64)[:, None], 2, axis=1
        )

    _calculate_contact_forces_rod_mesh(
        np.repeat(np.asarray([closest_point], dtype=np.float64),subsamples_per_element,axis=0),
        np.full(subsamples_per_element,radius-penetration,dtype=np.float64),
        np.repeat(np.asarray([normal], dtype=np.float64),subsamples_per_element,axis=0),
        np.asarray(rod_nodes, dtype=np.float64),
        np.asarray(rod_node_velocities, dtype=np.float64),
        np.asarray([radius], dtype=np.float64),
        rod_external_forces,
        rod_external_torques,
        np.asarray(rod_angular_velocity, dtype=np.float64)[:, None],
        np.asarray(rod_director, dtype=np.float64)[:, :, None],
        np.zeros(3, dtype=np.float64),
        np.asarray(mesh_velocity, dtype=np.float64),
        np.asarray(mesh_angular_velocity, dtype=np.float64),
        mesh_external_forces,
        mesh_external_torques,
        np.asarray(mesh_director, dtype=np.float64),
        mesh_frozen,
        np.float64(k),
        np.float64(nu),
        np.float64(velocity_damping_coefficient),
        np.float64(friction_coefficient),
        subsamples_per_element,
    )
    return (
        rod_external_forces.sum(axis=1),
        mesh_external_forces[:, 0],
        mesh_external_torques[:, 0],
        rod_external_torques[:, 0],
    )


def test_rod_mesh_no_contact():
    mesh = Mesh(_centered_box(1.0, 1.0, 1.0))
    mesh_body = MeshRigidBody(mesh=mesh, density=1.0, volume=mesh.compute_volume())

    nodes = np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 2.2]])
    rod = _MockRod(nodes, radius=0.1)

    contact = RodMeshContact(k=1e4, nu=10.0)
    contact.apply_contact(rod, mesh_body)

    assert_allclose(rod.external_forces, np.zeros_like(rod.external_forces), atol=1e-12)
    assert_allclose(
        mesh_body.external_forces, np.zeros_like(mesh_body.external_forces), atol=1e-12
    )


def test_rod_mesh_penetration_force_balance():
    mesh = Mesh(_centered_box(1.0, 1.0, 1.0))
    mesh_body = MeshRigidBody(mesh=mesh, density=1.0, volume=mesh.compute_volume())

    nodes = np.array([[0.0, 0.0], [0.0, 0.0], [0.6, 0.8]])
    rod = _MockRod(nodes, radius=0.4)

    contact = RodMeshContact(k=1e3, nu=0.0)
    contact.apply_contact(rod, mesh_body)

    rod_force_sum = rod.external_forces.sum(axis=1)
    mesh_force = mesh_body.external_forces[:, 0]

    assert np.linalg.norm(rod_force_sum) > 0.0
    assert_allclose(rod_force_sum + mesh_force, np.zeros(3), atol=1e-9)


def test_frozen_mesh_uses_zero_velocity():
    mesh = Mesh(_centered_box(1.0, 1.0, 1.0))

    nodes = np.array([[0.0, 0.0], [0.0, 0.0], [0.4, 0.6]])
    rod_fast = _MockRod(nodes.copy(), radius=0.4)
    rod_static = _MockRod(nodes.copy(), radius=0.4)

    mesh_fast = MeshRigidBody(mesh=mesh, density=1.0, volume=mesh.compute_volume())
    mesh_fast.velocity_collection[:] = np.array([[5.0], [0.0], [0.0]])
    mesh_fast.omega_collection[:] = np.array([[3.0], [0.0], [0.0]])

    mesh_static = MeshRigidBody(mesh=mesh, density=1.0, volume=mesh.compute_volume())

    contact = RodMeshContact(k=1e3, nu=5.0, mesh_frozen=True)
    contact.apply_contact(rod_fast, mesh_fast)

    contact_static = RodMeshContact(k=1e3, nu=5.0, mesh_frozen=True)
    contact_static.apply_contact(rod_static, mesh_static)

    assert_allclose(rod_fast.external_forces, rod_static.external_forces, atol=1e-12)
    assert_allclose(mesh_fast.external_forces, np.zeros_like(mesh_fast.external_forces))
    assert_allclose(
        mesh_fast.external_torques, np.zeros_like(mesh_fast.external_torques)
    )


def test_separating_contact_does_not_apply_attractive_normal_force():
    rod_force, mesh_force, _, _ = _run_single_element_kernel(
        penetration=1e-4, rod_velocity=(0.0, 0.0, 0.2), k=1e4, nu=10.0
    )

    assert_allclose(rod_force, np.zeros(3), atol=1e-12)
    assert_allclose(mesh_force, np.zeros(3), atol=1e-12)


def test_separating_contact_with_slip_has_no_friction_when_normal_force_clamps():
    rod_force, _, _, _ = _run_single_element_kernel(
        penetration=1e-4,
        rod_velocity=(3.0, 0.0, 0.2),
        k=1e4,
        nu=10.0,
        velocity_damping_coefficient=100.0,
        friction_coefficient=0.5,
    )

    assert_allclose(rod_force, np.zeros(3), atol=1e-12)


@pytest.mark.parametrize(
    "velocity_damping_coefficient, expected_friction",
    [(0.25, -0.5), (100.0, -1.0)],
    ids=["viscous_limited", "coulomb_limited"],
)
def test_compressed_sliding_contact_friction_opposes_slip_and_is_bounded(
    velocity_damping_coefficient, expected_friction
):
    friction_coefficient = 0.5
    rod_force, _, _, _ = _run_single_element_kernel(
        rod_velocity=(2.0, 0.0, -1.0),
        k=1.0,
        nu=1.0,
        velocity_damping_coefficient=velocity_damping_coefficient,
        friction_coefficient=friction_coefficient,
    )

    normal_force = rod_force[2]
    assert_allclose(rod_force, np.array([expected_friction, 0.0, 2.0]), atol=1e-12)
    assert rod_force[0] * 2.0 < 0.0
    assert abs(rod_force[0]) <= friction_coefficient * normal_force


def test_closing_velocity_adds_positive_normal_damping():
    rod_force, _, _, _ = _run_single_element_kernel(
        rod_velocity=(0.0, 0.0, -1.0), k=1.0, nu=2.0
    )

    assert_allclose(rod_force, np.array([0.0, 0.0, 3.0]), atol=1e-12)


@pytest.mark.parametrize(
    "penetration, normal_velocity",
    [(1.0, 1.0), (0.0, -1.0), (-0.1, -1.0)],
    ids=["exact_zero_normal", "zero_penetration", "negative_penetration"],
)
def test_contact_applies_no_force_at_unilateral_boundary(
    penetration, normal_velocity
):
    rod_force, mesh_force, mesh_torque, _ = _run_single_element_kernel(
        penetration=penetration,
        rod_velocity=(0.0, 0.0, normal_velocity),
        k=1.0,
        nu=1.0,
    )

    assert_allclose(rod_force, np.zeros(3), atol=1e-12)
    assert_allclose(mesh_force, np.zeros(3), atol=1e-12)
    assert_allclose(mesh_torque, np.zeros(3), atol=1e-12)


def test_contact_force_reaction_and_mesh_point_torque_are_balanced():
    director = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rod_force, mesh_force, mesh_torque, _ = _run_single_element_kernel(
        closest_point=(1.0, 0.0, 0.0), mesh_director=director, k=2.0, nu=0.0
    )

    assert_allclose(rod_force + mesh_force, np.zeros(3), atol=1e-12)
    expected_world_torque = np.cross(np.array([1.0, 0.0, 0.0]), mesh_force)
    assert_allclose(mesh_torque, director @ expected_world_torque, atol=1e-12)


def test_frozen_mesh_has_no_reaction_and_ignores_mesh_velocity():
    moving = _run_single_element_kernel(
        mesh_velocity=(0.0, 0.0, 20.0),
        mesh_angular_velocity=(0.0, 10.0, 0.0),
        mesh_frozen=True,
        k=1.0,
        nu=1.0,
    )
    stationary = _run_single_element_kernel(mesh_frozen=True, k=1.0, nu=1.0)

    assert_allclose(moving[0], stationary[0], atol=1e-12)
    assert_allclose(moving[1], np.zeros(3), atol=1e-12)
    assert_allclose(moving[2], np.zeros(3), atol=1e-12)


def test_mesh_contact_uses_world_frame_angular_velocity_for_point_velocity():
    closest_points = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
    contact_distances = np.array([0.0], dtype=np.float64)
    contact_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)

    rod_velocity_collection = np.zeros((3, 2), dtype=np.float64)
    rod_radii = np.array([1.0], dtype=np.float64)
    rod_external_forces = np.zeros((3, 2), dtype=np.float64)

    mesh_position = np.zeros(3, dtype=np.float64)
    mesh_velocity = np.zeros(3, dtype=np.float64)
    # world->mesh rotation: +90 deg about y
    mesh_director = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
    )

    for angular_velocity_z, expected_force_z in [(-1.0, 1.0), (1.0, 0.0)]:
        rod_external_forces.fill(0.0)
        mesh_external_forces = np.zeros((3, 1), dtype=np.float64)
        mesh_external_torques = np.zeros((3, 1), dtype=np.float64)
        _calculate_contact_forces_rod_mesh(
            closest_points,
            contact_distances,
            contact_normals,
            np.zeros((3, 2), dtype=np.float64),
            rod_velocity_collection,
            rod_radii,
            rod_external_forces,
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            np.eye(3, dtype=np.float64)[:, :, None],
            mesh_position,
            mesh_velocity,
            np.array([0.0, 0.0, angular_velocity_z], dtype=np.float64),
            mesh_external_forces,
            mesh_external_torques,
            mesh_director,
            False,
            np.float64(0.0),  # k
            np.float64(1.0),  # nu
            np.float64(0.0),
            np.float64(0.0),
        )

        rod_force_sum = rod_external_forces.sum(axis=1)
        assert_allclose(
            rod_force_sum, np.array([0.0, 0.0, expected_force_z]), atol=1e-12
        )


@pytest.mark.parametrize("contact_index", [0, 1, 2], ids=["first", "interior", "last"])
def test_rod_mesh_contact_balances_common_origin_moment_for_all_element_weights(
    contact_index,
):
    nodes = np.array(
        [[0.0, 1.0, 2.0, 3.0], [0.2, -0.1, 0.3, -0.2], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    n_elems = 3
    weights = ((1.0 / 3.0, 2.0 / 3.0), (0.5, 0.5), (2.0 / 3.0, 1.0 / 3.0))
    w0, w1 = weights[contact_index]
    application_point = w0 * nodes[:, contact_index] + w1 * nodes[:, contact_index + 1]
    contact_point = application_point + np.array([0.2, 0.4, 0.0])
    closest_points = np.zeros((n_elems, 3), dtype=np.float64)
    closest_points[contact_index] = contact_point
    distances = np.full(n_elems, 2.0, dtype=np.float64)
    distances[contact_index] = 0.0
    normals = np.repeat(np.array([[0.0, 0.0, 1.0]]), n_elems, axis=0)
    rod_forces = np.zeros((3, n_elems + 1), dtype=np.float64)
    rod_torques = np.zeros((3, n_elems), dtype=np.float64)
    rod_director = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    rod_directors = np.repeat(rod_director[:, :, None], n_elems, axis=2)
    mesh_director = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
    )
    mesh_position = np.array([-0.3, 0.5, -0.2])
    mesh_forces = np.zeros((3, 1), dtype=np.float64)
    mesh_torques = np.zeros((3, 1), dtype=np.float64)

    _calculate_contact_forces_rod_mesh(
        closest_points, distances, normals, nodes, np.zeros_like(nodes),
        np.ones(n_elems), rod_forces, rod_torques, np.zeros((3, n_elems)),
        rod_directors, mesh_position, np.zeros(3), np.zeros(3), mesh_forces,
        mesh_torques, mesh_director, False, np.float64(2.0), np.float64(0.0),
        np.float64(0.0), np.float64(0.0),
    )

    world_moment = np.zeros(3)
    for node_index in range(n_elems + 1):
        world_moment += np.cross(nodes[:, node_index], rod_forces[:, node_index])
    world_moment += rod_director.T @ rod_torques[:, contact_index]
    world_moment += np.cross(mesh_position, mesh_forces[:, 0])
    world_moment += mesh_director.T @ mesh_torques[:, 0]
    assert_allclose(rod_forces.sum(axis=1) + mesh_forces[:, 0], 0.0, atol=1e-12)
    assert_allclose(world_moment, 0.0, atol=1e-12)


@pytest.mark.parametrize('subsamples',[1,4,16,32])
def test_rod_spin_drives_friction_and_force_torque_mapping_is_work_conjugate(subsamples):
    rod_director = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
    )
    omega_world = np.array([0.0, 0.0, 2.0])
    omega_material = rod_director @ omega_world
    contact_point = np.array([0.0, 1.0, 0.0])
    rod_force, _, _, rod_torque = _run_single_element_kernel(
        closest_point=contact_point,
        rod_angular_velocity=omega_material,
        rod_director=rod_director,
        k=1.0,
        nu=0.0,
        velocity_damping_coefficient=10.0,
        friction_coefficient=0.5,
        subsamples_per_element=subsamples,
    )

    contact_velocity = np.cross(omega_world, contact_point)
    assert rod_force[0] > 0.0
    assert_allclose(
        np.dot(rod_torque, omega_material),
        np.dot(rod_force, contact_velocity),
        atol=1e-12,
    )


@pytest.mark.parametrize('subsamples',[1,4,16,32])
def test_weighted_rod_contact_velocity_matches_rigid_mesh_comotion(subsamples):
    nodes = np.array([[0.0, 3.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    omega_world = np.array([0.0, 0.0, 1.0])
    node_velocities = np.column_stack(
        (np.cross(omega_world, nodes[:, 0]), np.cross(omega_world, nodes[:, 1]))
    )
    contact_point = np.array([2.0, 1.0, 0.0])

    rod_force, _, _, _ = _run_single_element_kernel(
        rod_nodes=nodes,
        rod_node_velocities=node_velocities,
        rod_angular_velocity=omega_world,
        closest_point=contact_point,
        mesh_angular_velocity=omega_world,
        k=1.0,
        nu=1.0,
        velocity_damping_coefficient=10.0,
        friction_coefficient=1.0,
        subsamples_per_element=subsamples,
    )

    assert_allclose(rod_force, np.array([0.0, 0.0, 1.0]), atol=1e-12)


def test_frozen_mesh_still_applies_rod_eccentric_torque_without_reaction():
    rod_force, mesh_force, mesh_torque, rod_torque = _run_single_element_kernel(
        closest_point=(1.0, 0.0, 0.0), mesh_frozen=True, k=2.0, nu=0.0
    )

    assert np.linalg.norm(rod_force) > 0.0
    assert np.linalg.norm(rod_torque) > 0.0
    assert_allclose(mesh_force, 0.0, atol=1e-12)
    assert_allclose(mesh_torque, 0.0, atol=1e-12)
