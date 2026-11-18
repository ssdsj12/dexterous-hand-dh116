import numpy as np
import pytest
import sys
from pathlib import Path

torch = pytest.importorskip("torch")
BACKEND = Path(__file__).resolve().parents[1] / "vendor" / "DeformX"
if BACKEND.is_dir():
    sys.path.insert(0, str(BACKEND))

from co_sim.gpu_cosserat import GpuRodConfig, TorchCudaCosseratRod

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_gpu_cosserat_uses_material_stiffness_and_rest_state_is_force_free():
    cfg = GpuRodConfig(
        n_elem=8, base_length=0.5, base_radius=0.008, density=1000.0,
        youngs_modulus=1.0e6, shear_modulus_ratio=1.5,
        axial_stretch_stiffening=1.0, dt=1.0e-6,
    )
    rod = TorchCudaCosseratRod(cfg, device="cuda:0")

    forces, torques = rod.compute_internal_forces_and_torques()

    assert forces.is_cuda and torques.is_cuda
    torch.testing.assert_close(forces, torch.zeros_like(forces), atol=1e-5, rtol=0)
    torch.testing.assert_close(torques, torch.zeros_like(torques), atol=1e-5, rtol=0)


def test_gpu_cosserat_bending_generates_restoring_internal_force():
    rod = TorchCudaCosseratRod(
        GpuRodConfig(n_elem=8, base_length=0.5, base_radius=0.008,
                     youngs_modulus=1.0e6, dt=1.0e-7),
        device="cuda:0",
    )
    rod.rod_position[2, 4] += 0.01

    forces, torques = rod.compute_internal_forces_and_torques()

    assert torch.isfinite(forces).all() and torch.isfinite(torques).all()
    assert torch.linalg.vector_norm(forces) > 0
    assert torch.linalg.vector_norm(torques) > 0


def test_gpu_internal_wrench_matches_existing_pyelastica_constitutive_model():
    pytest.importorskip("elastica")
    from co_sim.free_rod import FreeRodEngine
    from co_sim.models import CoSimConfig

    config = GpuRodConfig(n_elem=4, base_length=0.5, base_radius=0.008,
                          density=1000.0, youngs_modulus=1.0e4,
                          shear_modulus_ratio=1.5, dt=1.0e-5, damping=0.0,
                          rod_start=(0.0, 0.0, 0.2))
    cpu = FreeRodEngine(CoSimConfig(
        n_elem=config.n_elem, base_length=config.base_length,
        base_radius=config.base_radius, density=config.density,
        youngs_modulus=config.youngs_modulus,
        shear_modulus_ratio=config.shear_modulus_ratio,
        py_dt=config.dt, isaac_dt=1.0e-4, damping_constant=0.0,
        rod_start=np.asarray(config.rod_start),
        rod_direction=np.asarray(config.rod_direction),
        rod_normal=np.asarray(config.rod_normal),
    ))
    cpu.rod.position_collection[2] += np.asarray([0.0, 0.001, 0.002, 0.001, 0.0])
    cpu.rod.compute_internal_forces_and_torques(0.0)
    gpu = TorchCudaCosseratRod(config, device="cuda:0")
    gpu.rod_position.copy_(gpu._tensor(cpu.rod.position_collection))

    force, torque = gpu.compute_internal_forces_and_torques()

    np.testing.assert_allclose(force.detach().cpu().numpy(), cpu.rod.internal_forces,
                               atol=1e-10, rtol=1e-9)
    np.testing.assert_allclose(torque.detach().cpu().numpy(), cpu.rod.internal_torques,
                               atol=1e-10, rtol=1e-9)


def test_gpu_bending_and_twist_wrench_matches_pyelastica_after_frame_rotation():
    pytest.importorskip("elastica")
    from co_sim.free_rod import FreeRodEngine
    from co_sim.models import CoSimConfig

    config = GpuRodConfig(n_elem=5, base_length=0.5, base_radius=0.008,
                          density=1000.0, youngs_modulus=2.0e4,
                          shear_modulus_ratio=1.5, dt=1.0e-5, damping=0.0)
    cpu = FreeRodEngine(CoSimConfig(
        n_elem=config.n_elem, base_length=config.base_length,
        base_radius=config.base_radius, density=config.density,
        youngs_modulus=config.youngs_modulus,
        shear_modulus_ratio=config.shear_modulus_ratio,
        py_dt=config.dt, isaac_dt=1.0e-4, damping_constant=0.0,
        rod_start=np.asarray(config.rod_start), rod_direction=np.asarray(config.rod_direction),
        rod_normal=np.asarray(config.rod_normal),
    ))
    angles = np.linspace(-0.02, 0.03, config.n_elem)
    for i, angle in enumerate(angles):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.asarray([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
        cpu.rod.director_collection[:, :, i] = rotation @ cpu.rod.director_collection[:, :, i]
    cpu.rod.compute_internal_forces_and_torques(0.0)
    gpu = TorchCudaCosseratRod(config, device="cuda:0")
    gpu.rod_director.copy_(gpu._tensor(cpu.rod.director_collection))

    force, torque = gpu.compute_internal_forces_and_torques()

    np.testing.assert_allclose(force.detach().cpu().numpy(), cpu.rod.internal_forces,
                               atol=1e-9, rtol=1e-8)
    np.testing.assert_allclose(torque.detach().cpu().numpy(), cpu.rod.internal_torques,
                               atol=1e-9, rtol=1e-8)


def test_gpu_one_step_tracks_reference_free_rod_under_gravity():
    pytest.importorskip("elastica")
    from co_sim.free_rod import FreeRodEngine
    from co_sim.models import CoSimConfig

    config=GpuRodConfig(n_elem=4,base_length=.5,base_radius=.008,density=1000.,
        youngs_modulus=1e4,dt=1e-6,damping=0.,rod_start=(0.,0.,.2))
    cpu=FreeRodEngine(CoSimConfig(n_elem=config.n_elem,base_length=config.base_length,
        base_radius=config.base_radius,density=config.density,youngs_modulus=config.youngs_modulus,
        shear_modulus_ratio=config.shear_modulus_ratio,py_dt=config.dt,isaac_dt=1e-4,
        damping_constant=0.,rod_start=np.asarray(config.rod_start),
        rod_direction=np.asarray(config.rod_direction),rod_normal=np.asarray(config.rod_normal)))
    gpu=TorchCudaCosseratRod(config,device="cuda:0")
    cpu.advance(config.dt)
    gpu.step(1,dt=config.dt)
    for actual,expected in ((gpu.rod_position,cpu.rod.position_collection),
                            (gpu.rod_velocity,cpu.rod.velocity_collection),
                            (gpu.rod_director,cpu.rod.director_collection),
                            (gpu.rod_omega,cpu.rod.omega_collection)):
        np.testing.assert_allclose(actual.detach().cpu().numpy(),expected,atol=1e-10,rtol=1e-9)


def test_gpu_table_contact_returns_equal_and_opposite_reaction():
    cfg = GpuRodConfig(
        n_elem=8, base_length=0.5, base_radius=0.008, dt=1.0e-6,
        table_center=(0.25, 0.0, -0.004), table_size=(0.8, 0.4, 0.02),
    )
    rod = TorchCudaCosseratRod(cfg, device="cuda:0")
    rod.rod_position[2].fill_(0.0)

    contact = rod.compute_table_contact()

    assert contact["rod_forces"].is_cuda
    assert torch.isfinite(contact["rod_forces"]).all()
    assert torch.linalg.vector_norm(contact["rod_forces"]) > 0
    torch.testing.assert_close(
        contact["rod_forces"].sum(dim=1) + contact["reaction_force"],
        torch.zeros(3, device="cuda:0", dtype=rod.dtype), atol=1e-5, rtol=0,
    )


def test_gpu_mesh_contact_runs_on_device_and_balances_robot_wrench():
    rod = TorchCudaCosseratRod(
        GpuRodConfig(n_elem=4, base_length=0.5, base_radius=0.008, dt=1.0e-6),
        device="cuda:0",
    )
    # A closed box mesh is used as a tiny stand-in for one already-loaded USD pad collider.
    vertices = torch.tensor([
        [0.15, -0.03, 0.004], [0.35, -0.03, 0.004],
        [0.35, 0.03, 0.004], [0.15, 0.03, 0.004],
        [0.15, -0.03, 0.05], [0.35, -0.03, 0.05],
        [0.35, 0.03, 0.05], [0.15, 0.03, 0.05],
    ], device="cuda:0")
    triangles = torch.tensor([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=torch.long, device="cuda:0")
    rod.rod_position[2].fill_(0.0)

    contact = rod.compute_mesh_contact("left_index", vertices, triangles)

    assert contact["rod_forces"].is_cuda and contact["body_force"].is_cuda
    assert int(contact["contact_count"].item()) > 0
    assert torch.linalg.vector_norm(contact["body_force"]) > 0
    torch.testing.assert_close(
        contact["rod_forces"].sum(dim=1) + contact["body_force"],
        torch.zeros(3, device="cuda:0", dtype=rod.dtype), atol=1e-4, rtol=0,
    )
    assert torch.isfinite(contact["rod_torques"]).all()
    assert int(contact["contact_count"].item()) > 0


def test_gpu_outer_step_uploads_reaction_impulses_and_keeps_state_on_cuda():
    from co_sim.moving_contact_models import MeshMotion

    rod = TorchCudaCosseratRod(
        GpuRodConfig(n_elem=4, base_length=0.5, base_radius=0.008,
                     dt=1.0e-6, damping=0.0), device="cuda:0")
    vertices = np.asarray([
        [0.15, -0.03, 0.004], [0.35, -0.03, 0.004],
        [0.35, 0.03, 0.004], [0.15, 0.03, 0.004],
        [0.15, -0.03, 0.05], [0.35, -0.03, 0.05],
        [0.35, 0.03, 0.05], [0.15, 0.03, 0.05],
    ])
    triangles = np.asarray([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ])
    rod.register_contact_mesh("left_index", vertices, triangles)
    rod.rod_position[2].fill_(0.0)
    zero = np.zeros(3)
    motion = MeshMotion(np.array([0.25, 0.0, 0.0]), np.eye(3), zero, zero, zero, zero)

    snapshot, impulses = rod.advance_with_meshes({"left_index": motion}, 2.0e-6)

    assert snapshot["backend"] == "torch_cuda"
    assert isinstance(snapshot["rod_position"], np.ndarray)
    assert impulses["left_index"].peak_force_n > 0
    assert impulses["left_index"].contact_duration_s > 0
    assert rod.rod_position.is_cuda
    geometry=rod.contact_geometry("left_index")
    assert geometry["gaps"].shape == (rod.n_elem,)
    assert np.isfinite(geometry["gaps"]).all()
    assert geometry["relative_normal_speeds"].shape == geometry["gaps"].shape
    assert geometry["positions"].shape == (rod.n_elem,3)
    assert tuple(rod.last_contact_diagnostics["channels"]) == ("left_index",)
    assert rod.last_contact_diagnostics["integrated_duration_s"] == 2.0e-6
    np.testing.assert_allclose(rod.last_contact_diagnostics["channels"]["left_index"]["linear_balance_w"],
                               0., atol=1e-15)


def test_gpu_coupling_diagnostics_match_reaction_bridge_contract():
    from go2_pvcnn.control.fixed_dual_panda_o6.contact_bridge import reaction_wrenches
    from go2_pvcnn.control.dh116.contact_runtime import BODY_NAMES
    from co_sim.moving_contact_models import MeshMotion

    rod = TorchCudaCosseratRod(GpuRodConfig(n_elem=4, base_length=.5, base_radius=.008,
        dt=1e-6, damping=0.), device="cuda:0")
    vertices = np.asarray([
        [.15,-.03,.0079],[.35,-.03,.0079],[.35,.03,.0079],[.15,.03,.0079],
        [.15,-.03,.05],[.35,-.03,.05],[.35,.03,.05],[.15,.03,.05],
    ])
    faces = np.asarray([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
        [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    zero=np.zeros(3)
    motions={}
    for channel in BODY_NAMES:
        rod.register_contact_mesh(channel, vertices, faces)
        z=0. if channel=="left_index" else .5
        motions[channel]=MeshMotion(np.array([.25,0.,z]),np.eye(3),zero,zero,zero,zero)
    _, impulses=rod.advance_with_meshes(motions,2e-6)
    wrenches=reaction_wrenches(impulses,rod.last_contact_diagnostics,
        {name:np.array([.25,0.,.5]) for name in BODY_NAMES},2e-6)
    assert tuple(wrenches)==tuple(BODY_NAMES)
    assert rod.last_contact_diagnostics["evaluation_count"] == 12
    assert impulses["left_index"].peak_force_n > 0
    assert np.isfinite(wrenches["left_index"].torque_com_w).all()


def _box_mesh(lower=(-.1, -.1, -.1), upper=(.1, .1, .1)):
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    vertices = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                         [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    faces = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                      [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return vertices, faces


def test_gpu_contact_uses_world_spin_for_rigid_comotion():
    rod = TorchCudaCosseratRod(GpuRodConfig(
        n_elem=4, rod_start=(0., -.25, 0.), rod_direction=(0., 1., 0.),
        rod_normal=(0., 0., 1.)), device="cuda:0")
    vertices, faces = _box_mesh((-.03, -.3, .007), (.03, .3, .05))
    omega = rod._tensor([.3, -.2, .1])
    translation = rod._tensor([.01, .02, -.03])
    origin = rod._tensor([0., 0., .02])
    rod.rod_velocity.copy_(translation[:, None] + torch.linalg.cross(
        omega.expand_as(rod.rod_position.T), rod.rod_position.T, dim=1).T)
    rod.rod_omega.copy_(torch.einsum("ijn,j->in", rod.rod_director, omega))

    contact = rod.compute_mesh_contact("pad", vertices, faces, body_origin=origin,
        body_velocity=translation + torch.linalg.cross(omega, origin), body_omega=omega)

    torch.testing.assert_close(contact["relative_normal_speeds"],
        torch.zeros_like(contact["relative_normal_speeds"]), atol=1e-13, rtol=0)
    torch.testing.assert_close(contact["body_force"][:2],
        torch.zeros(2, dtype=rod.dtype, device=rod.device), atol=1e-10, rtol=0)


def test_gpu_separating_contact_damping_matches_cpu_contact_law():
    rod = TorchCudaCosseratRod(GpuRodConfig(n_elem=4), device="cuda:0")
    vertices, faces = _box_mesh((.15, -.03, .007), (.35, .03, .05))
    rod.rod_velocity[2].fill_(-.02)

    contact = rod.compute_mesh_contact("pad", vertices, faces)

    # The two middle samples penetrate 1 mm and separate at 0.02 m/s.
    expected = 2 * (rod.cfg.contact_stiffness * .001 - rod.cfg.contact_damping * .02)
    assert contact["body_force"][2].item() == pytest.approx(expected, abs=1e-10)


def test_gpu_rotational_damping_matches_pyelastica_with_nonzero_spin():
    pytest.importorskip("elastica")
    from co_sim.free_rod import FreeRodEngine
    from co_sim.models import CoSimConfig

    cfg = GpuRodConfig(n_elem=4, dt=1e-5, damping=.01)
    cpu = FreeRodEngine(CoSimConfig(n_elem=cfg.n_elem, base_length=cfg.base_length,
        base_radius=cfg.base_radius, density=cfg.density, youngs_modulus=cfg.youngs_modulus,
        shear_modulus_ratio=cfg.shear_modulus_ratio, py_dt=cfg.dt, isaac_dt=1e-4,
        damping_constant=cfg.damping, rod_start=np.array(cfg.rod_start),
        rod_direction=np.array(cfg.rod_direction), rod_normal=np.array(cfg.rod_normal)))
    gpu = TorchCudaCosseratRod(cfg, device="cuda:0")
    cpu.rod.omega_collection[2] = .1
    gpu.rod_omega[2].fill_(.1)

    cpu.advance(cfg.dt)
    gpu.step()

    np.testing.assert_allclose(gpu.rod_omega.cpu().numpy(), cpu.rod.omega_collection,
                               atol=1e-11, rtol=1e-9)
    np.testing.assert_allclose(gpu.rod_director.cpu().numpy(), cpu.rod.director_collection,
                               atol=1e-11, rtol=1e-9)


@pytest.mark.parametrize("start", [(.507, -.25, 0.), (0., -.25, -.057),
                                    (0., -.25, .057), (.503, -.25, .053)])
def test_gpu_finite_table_contact_matches_closed_box_surface(start):
    rod = TorchCudaCosseratRod(GpuRodConfig(n_elem=4, rod_start=start,
        rod_direction=(0., 1., 0.), table_center=(0., 0., 0.),
        table_size=(1., 1., .1)), device="cuda:0")
    vertices, faces = _box_mesh((-.5, -.5, -.05), (.5, .5, .05))
    table = rod.compute_table_contact()
    mesh = rod.compute_mesh_contact("table", vertices, faces, body_origin=(0., 0., 0.))

    assert torch.linalg.vector_norm(mesh["body_force"]) > 0
    torch.testing.assert_close(table["rod_forces"], mesh["rod_forces"], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(table["rod_torques"], mesh["rod_torques"], atol=1e-10, rtol=1e-10)


def test_gpu_outer_step_predicts_moving_colliders_on_cuda(monkeypatch):
    from co_sim import moving_contact_models
    from co_sim.moving_contact_models import MeshMotion, predict_motion

    rod = TorchCudaCosseratRod(GpuRodConfig(n_elem=4, dt=1e-5), device="cuda:0")
    rod.register_contact_mesh("pad", *_box_mesh())
    motion = MeshMotion(np.array([2., 0., .3]), np.eye(3), np.array([.1, -.2, .3]),
        np.array([.4, -.3, .2]), np.array([.2, .1, -.1]), np.array([.1, -.2, .3]))
    expected = predict_motion(motion, 1e-4)

    def forbid_cpu_prediction(*args):
        raise AssertionError("fine-step mesh motion must be predicted on CUDA")
    monkeypatch.setattr(moving_contact_models, "_predict_validated_motion", forbid_cpu_prediction)

    rod.advance_with_meshes({"pad": motion}, 1e-4)

    body = rod.mesh_bodies["pad"]
    np.testing.assert_allclose(body.position_collection[:, 0], expected.position_w, atol=1e-14)
    np.testing.assert_allclose(body.director_collection[:, :, 0].T, expected.rotation_wb, atol=1e-14)
    np.testing.assert_allclose(body.velocity_collection[:, 0], expected.velocity_w, atol=1e-14)
    np.testing.assert_allclose(body.omega_collection[:, 0], expected.omega_w, atol=1e-14)


def test_gpu_contact_reports_rod_moment_from_applied_nodal_and_material_loads():
    rod = TorchCudaCosseratRod(GpuRodConfig(n_elem=4, rod_start=(0., -.25, 0.),
        rod_direction=(0., 1., 0.)), device="cuda:0")
    rod.rod_velocity[0].fill_(.02)
    vertices, faces = _box_mesh((-.03, -.3, .007), (.03, .3, .05))

    contact = rod.compute_mesh_contact("pad", vertices, faces, body_origin=(.1, .2, .3))

    applied_moment = (torch.linalg.cross(rod.rod_position.T, contact["rod_forces"].T, dim=1).sum(0)
        + torch.einsum("jin,jn->in", rod.rod_director, contact["rod_torques"]).sum(1))
    torch.testing.assert_close(contact["rod_angular_origin"], applied_moment, atol=1e-12, rtol=0)
    torch.testing.assert_close(contact["rod_angular_origin"] + contact["body_angular_origin"],
        torch.zeros(3, dtype=rod.dtype, device=rod.device), atol=1e-12, rtol=0)


def test_gpu_contact_observation_keeps_exact_gaps_outside_contact_broad_phase():
    from co_sim.moving_contact_models import MeshMotion

    rod = TorchCudaCosseratRod(GpuRodConfig(n_elem=4), device="cuda:0")
    vertices, faces = _box_mesh((-.03, -.3, .05), (.03, .3, .1))
    rod.register_contact_mesh("pad", vertices, faces)
    zero = np.zeros(3)
    motion = MeshMotion(zero, np.eye(3), zero, zero, zero, zero)
    _, impulses = rod.advance_with_meshes({"pad": motion}, 1e-5)

    geometry = rod.contact_geometry("pad")

    assert impulses["pad"].peak_force_n == 0.
    closest, signed, _ = rod.mesh_bodies["pad"].query_closest_points(geometry["positions"])
    expected = signed - rod.radius.cpu().numpy()[geometry["element_ids"]]
    np.testing.assert_allclose(geometry["gaps"], expected, atol=1e-12, rtol=1e-12)
    assert np.isfinite(closest).all()


def test_production_size_cuda_rod_deforms_and_restores_continuous_dynamics():
    cfg = GpuRodConfig(n_elem=40, base_length=.5, base_radius=.008, dt=1e-5,
        rod_start=(.43, -.25, .308), rod_direction=(0., 1., 0.),
        table_center=(.6, 0., .25), table_size=(.8, 1.2, .1), contact_subsamples=16)
    rod = TorchCudaCosseratRod(cfg, device="cuda:0")
    rod.rod_velocity[2] = .01 * torch.sin(torch.linspace(0., torch.pi, 41,
        dtype=rod.dtype, device=rod.device))
    original = rod.rod_position.clone()
    rod.step(100)
    saved = rod.export_physical_state()
    restored = TorchCudaCosseratRod(cfg, device="cuda:0")
    restored.restore_physical_state(saved)

    rod.step(50)
    restored.step(50)

    for name in ("rod_position", "rod_velocity", "rod_director", "rod_omega"):
        state = getattr(rod, name)
        assert state.is_cuda and torch.isfinite(state).all()
        torch.testing.assert_close(state, getattr(restored, name), atol=1e-12, rtol=1e-10)
    displacement = rod.rod_position[2] - original[2]
    assert (displacement.max() - displacement.min()).item() > 1e-7
    assert torch.linalg.vector_norm(rod.compute_internal_forces_and_torques()[1]).item() > 0.
    assert rod.last_contact_diagnostics["table"]["reaction_force"].is_cuda


def test_gpu_planning_query_returns_actual_closest_mesh_point_without_advancing_rod():
    rod=TorchCudaCosseratRod(GpuRodConfig(n_elem=4,contact_subsamples=16),device="cuda:0")
    rod.register_contact_mesh('pad',*_box_mesh((.15,-.03,.05),(.35,.03,.10)))
    original=rod.rod_position.clone()
    a=rod.closest_contact_point('pad',position=(0.,0.,0.),rotation=np.eye(3))
    b=rod.closest_contact_point('pad',position=(0.,0.,.02),rotation=np.eye(3))
    assert a['point_w'].is_cuda and a['gap_m'].is_cuda
    assert a['point_w'][2].item()==pytest.approx(.05)
    assert a['gap_m'].item()==pytest.approx(.042)
    assert b['point_w'][2].item()==pytest.approx(.07)
    assert b['gap_m'].item()==pytest.approx(.062)
    torch.testing.assert_close(rod.rod_position,original)
