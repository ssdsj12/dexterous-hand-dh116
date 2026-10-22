"""CUDA-resident Cosserat rod state, internal forces, and rigid-mesh contacts.

The rod equations follow the shear/stretch and bend/twist constitutive model
used by the vendored PyElastica backend. Rod state, constitutive evaluation,
table contact, and prescribed closed-mesh contact stay on the selected CUDA
device. USD is only a rendering/source-geometry boundary; it is not rewritten.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np


class _GpuMeshBody:
    """Compatibility view for existing guards; distances are queried on CUDA."""

    def __init__(self, engine, name, vertices, triangles):
        self.engine = engine
        self.name = name
        self.vertices_local = vertices
        self.triangles = triangles
        self.position_collection = np.zeros((3, 1))
        self.director_collection = np.eye(3)[:, :, None]
        self.velocity_collection = np.zeros((3, 1))
        self.omega_collection = np.zeros((3, 1))
        self.signed_distance_method = "cuda_triangle_closest_point_and_solid_angle"

    def set_motion(self, motion):
        self.position_collection[:, 0] = np.asarray(motion.position_w)
        self.director_collection[:, :, 0] = np.asarray(motion.rotation_wb).T
        self.velocity_collection[:, 0] = np.asarray(motion.velocity_w)
        self.omega_collection[:, 0] = np.asarray(motion.omega_w)

    def query_closest_points(self, points):
        torch = self.engine.torch
        rotation = self.engine._tensor(self.director_collection[:, :, 0].T)
        translation = self.engine._tensor(self.position_collection[:, 0])
        vertices = self.vertices_local @ rotation.T + translation[None, :]
        query = self.engine._tensor(points)
        closest, distance, winding = self.engine._closest_mesh(query, vertices[self.triangles])
        signed = torch.where(winding.abs() > 0.5, -distance, distance)
        return (closest.detach().cpu().numpy(), signed.detach().cpu().numpy(),
                winding.detach().cpu().numpy())


@dataclass(frozen=True)
class GpuRodConfig:
    n_elem: int = 40
    base_length: float = 0.5
    base_radius: float = 0.008
    density: float = 1000.0
    youngs_modulus: float = 1.0e6
    shear_modulus_ratio: float = 1.5
    axial_stretch_stiffening: float = 1.0
    dt: float = 1.0e-5
    damping: float = 1.0e-2
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    rod_start: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rod_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    rod_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    table_center: tuple[float, float, float] | None = None
    table_size: tuple[float, float, float] | None = None
    contact_stiffness: float = 1.0e4
    contact_damping: float = 10.0
    friction_coefficient: float = 0.5
    tangential_damping: float = 10.0
    contact_velocity_epsilon: float = 1.0e-4
    contact_subsamples: int = 1


class TorchCudaCosseratRod:
    """Explicit CUDA Cosserat solver with GPU penalty-contact evaluation.

    State tensors use PyElastica-compatible conventions and shapes:
    node vectors are ``(3, n_elem + 1)``, element frames ``(3, 3, n_elem)``.
    Contact mesh assets are uploaded once with :meth:`register_contact_mesh`;
    per-step rigid poses are then supplied as device tensors or small host poses.
    """

    backend = "torch_cuda"
    approximation = "torch_cuda_cosserat_shear_stretch_bend_twist_penalty_contact"

    def __init__(self, config: GpuRodConfig | None = None, *, device: str = "cuda") -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for the torch_cuda backend") from exc
        if not torch.cuda.is_available() or not str(device).startswith("cuda"):
            raise RuntimeError("torch_cuda backend requires an available CUDA device")
        self.torch = torch
        self.cfg = config or GpuRodConfig()
        self._validate_config(self.cfg)
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.n_elem = self.cfg.n_elem
        self._eye = torch.eye(3, dtype=self.dtype, device=self.device)
        self._zero3 = torch.zeros(3, dtype=self.dtype, device=self.device)
        self._material_z = self._tensor([0., 0., 1.])[:, None]
        self._gravity = self._tensor(self.cfg.gravity).reshape(3, 1)
        self._table_center = (None if self.cfg.table_center is None else
                              self._tensor(self.cfg.table_center).reshape(3, 1))
        self._table_half = (None if self.cfg.table_size is None else
                            self._tensor(self.cfg.table_size).reshape(3, 1) * .5)
        self._mesh_assets: dict[str, tuple[Any, Any]] = {}
        self._mesh_motions: dict[str, dict[str, Any]] = {}
        self.mesh_bodies: dict[str, _GpuMeshBody] = {}
        self._external_forces = torch.zeros((3, self.n_elem + 1), dtype=self.dtype, device=self.device)
        self._external_torques = torch.zeros((3, self.n_elem), dtype=self.dtype, device=self.device)
        self._initialize_rest_model()
        self.rod = self
        self.radius = torch.full((self.n_elem,), self.cfg.base_radius,
                                 dtype=self.dtype, device=self.device)
        self.contact_k = self.cfg.contact_stiffness
        self.contact_nu = self.cfg.contact_damping
        self.friction_coefficient = self.cfg.friction_coefficient
        self.velocity_damping_coefficient = self.cfg.tangential_damping
        self.contact_subsamples = self.cfg.contact_subsamples
        self.contact_broad_phase = True  # conservative CUDA mesh-AABB culling
        self.time = torch.zeros((), dtype=self.dtype, device=self.device)
        self.last_contact_diagnostics: dict[str, Any] = {}
        self._physical_model_sha256 = self._model_digest()

    @staticmethod
    def _validate_config(cfg: GpuRodConfig) -> None:
        if cfg.n_elem < 2 or cfg.base_length <= 0 or cfg.base_radius <= 0 or cfg.density <= 0:
            raise ValueError("invalid GPU rod geometry or density")
        if cfg.dt <= 0 or cfg.damping < 0 or cfg.youngs_modulus <= 0:
            raise ValueError("invalid GPU time step, damping, or Young's modulus")
        if cfg.shear_modulus_ratio <= 0 or cfg.axial_stretch_stiffening <= 0:
            raise ValueError("invalid GPU shear/stretch material parameters")
        if cfg.contact_stiffness <= 0 or min(cfg.contact_damping, cfg.friction_coefficient,
                                             cfg.tangential_damping) < 0:
            raise ValueError("invalid GPU contact parameters")
        if type(cfg.contact_subsamples) is not int or not 1 <= cfg.contact_subsamples <= 128:
            raise ValueError("contact_subsamples must be an integer in [1,128]")
        if (cfg.table_center is None) != (cfg.table_size is None):
            raise ValueError("table_center and table_size must be supplied together")
        if cfg.table_size is not None and np.any(np.asarray(cfg.table_size) <= 0):
            raise ValueError("table_size must be positive")

    def _tensor(self, value, *, dtype=None):
        if isinstance(value, np.ndarray):
            value = np.array(value, copy=True)
        return self.torch.as_tensor(value, dtype=dtype or self.dtype, device=self.device)

    def _initialize_rest_model(self) -> None:
        torch = self.torch
        cfg = self.cfg
        n = cfg.n_elem
        start = self._tensor(cfg.rod_start)
        direction = self._tensor(cfg.rod_direction)
        normal = self._tensor(cfg.rod_normal)
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-15)
        normal = normal - torch.dot(normal, direction) * direction
        normal = normal / torch.linalg.vector_norm(normal).clamp_min(1e-15)
        binormal = torch.linalg.cross(direction, normal)
        frames = torch.stack((normal, binormal, direction), dim=0)
        s = torch.linspace(0.0, cfg.base_length, n + 1, dtype=self.dtype, device=self.device)
        self.rod_position = start[:, None] + direction[:, None] * s[None, :]
        self.rod_velocity = torch.zeros_like(self.rod_position)
        self.rod_director = frames[:, :, None].repeat(1, 1, n)
        self.rod_omega = torch.zeros((3, n), dtype=self.dtype, device=self.device)
        self.rest_lengths = torch.full((n,), cfg.base_length / n, dtype=self.dtype, device=self.device)
        self.rest_voronoi_lengths = torch.full((n - 1,), cfg.base_length / n, dtype=self.dtype, device=self.device)
        self.rest_sigma = torch.zeros((3, n), dtype=self.dtype, device=self.device)
        self.rest_kappa = torch.zeros((3, n - 1), dtype=self.dtype, device=self.device)
        area = torch.pi * cfg.base_radius**2
        i1 = area**2 / (4.0 * torch.pi)
        i3 = 2.0 * i1
        shear_modulus = cfg.youngs_modulus / (2.0 * cfg.shear_modulus_ratio)
        shear_diag = self._tensor([
            (27.0 / 28.0) * shear_modulus * area,
            (27.0 / 28.0) * shear_modulus * area,
            cfg.youngs_modulus * area * cfg.axial_stretch_stiffening,
        ])
        bend_diag = self._tensor([
            cfg.youngs_modulus * i1,
            cfg.youngs_modulus * i1,
            shear_modulus * i3,
        ])
        self._shear_diag = shear_diag[:, None]
        self._bend_diag = bend_diag[:, None]
        element_mass = cfg.density * area * self.rest_lengths
        self.node_mass = torch.cat((0.5 * element_mass[:1],
                                    0.5 * (element_mass[:-1] + element_mass[1:]),
                                    0.5 * element_mass[-1:]))
        inertia_diag = self._tensor([cfg.density * cfg.base_length * i1,
                                     cfg.density * cfg.base_length * i1,
                                     cfg.density * cfg.base_length * i3]) / n
        self._inverse_inertia_diag = inertia_diag.reciprocal()[:, None]
        self._rotational_damping_rate = (cfg.damping * element_mass[None, :] *
                                         self._inverse_inertia_diag)
        self._linear_damping_rate = self._tensor(cfg.damping)

    def _geometry(self):
        delta = self.rod_position[:, 1:] - self.rod_position[:, :-1]
        lengths = torch_norm(delta, self.torch)
        tangents = delta / lengths.clamp_min(1.0e-12)[None, :]
        self.radius = self.torch.sqrt(self.cfg.base_radius ** 2 * self.rest_lengths /
                                      lengths.clamp_min(1.0e-12))
        dilatation = lengths / self.rest_lengths
        voronoi = 0.5 * (lengths[1:] + lengths[:-1]) / self.rest_voronoi_lengths
        return lengths, tangents, dilatation, voronoi

    def compute_internal_forces_and_torques(self):
        """Evaluate CUDA shear/stretch forces and bend/twist torques."""
        torch = self.torch
        lengths, tangents, dilatation, voronoi = self._geometry()
        local_tangent = torch.einsum("ijn,jn->in", self.rod_director, tangents)
        sigma = dilatation[None, :] * local_tangent - self._material_z
        stress = self._shear_diag * (sigma - self.rest_sigma)
        world_stress = torch.einsum("jin,jn->in", self.rod_director, stress) / dilatation.clamp_min(1e-12)[None, :]
        internal_forces = torch.zeros_like(self.rod_position)
        internal_forces[:, 0] = world_stress[:, 0]
        internal_forces[:, 1:-1] = world_stress[:, 1:] - world_stress[:, :-1]
        internal_forces[:, -1] = -world_stress[:, -1]

        if self.n_elem > 1:
            relative = torch.einsum("ijn,kjn->ikn", self.rod_director[:, :, 1:],
                                    self.rod_director[:, :, :-1])
            vee = torch.stack((relative[2, 1] - relative[1, 2],
                               relative[0, 2] - relative[2, 0],
                               relative[1, 0] - relative[0, 1]))
            cosine = ((relative[0, 0] + relative[1, 1] + relative[2, 2] - 1.0) * 0.5).clamp(-1.0, 1.0)
            angle = torch.acos(cosine)
            sin_angle = torch.sin(angle)
            scale = -0.5 * angle / torch.where(sin_angle.abs() < 1e-8,
                                                torch.full_like(sin_angle, 1e-8), sin_angle)
            kappa = vee * scale[None, :] / self.rest_voronoi_lengths[None, :]
            couple = self._bend_diag * (kappa - self.rest_kappa)
            inv_v3 = voronoi.clamp_min(1e-9).pow(-3)
            diff = couple * inv_v3[None, :]
            torque = torch.zeros((3, self.n_elem), dtype=self.dtype, device=self.device)
            torque[:, 0] = diff[:, 0]
            torque[:, 1:-1] = diff[:, 1:] - diff[:, :-1]
            torque[:, -1] = -diff[:, -1]
            bend_cross = torch.linalg.cross(kappa.T, couple.T, dim=1).T
            quad = torch.zeros_like(torque)
            weighted = bend_cross * (self.rest_voronoi_lengths * inv_v3)[None, :]
            quad[:, 0] = 0.5 * weighted[:, 0]
            quad[:, -1] = 0.5 * weighted[:, -1]
            if self.n_elem > 2:
                quad[:, 1:-1] = 0.5 * (weighted[:, 1:] + weighted[:, :-1])
            shear_couple = torch.linalg.cross(local_tangent.T, stress.T, dim=1).T * self.rest_lengths[None, :]
            jomega = self.rod_omega * (1.0 / self._inverse_inertia_diag)
            lagrangian = torch.linalg.cross((jomega / dilatation[None, :]).T,
                                            self.rod_omega.T, dim=1).T
            velocity_delta = self.rod_velocity[:, 1:] - self.rod_velocity[:, :-1]
            dilatation_rate = torch.sum(tangents * velocity_delta, dim=0) / self.rest_lengths
            unsteady = (jomega / dilatation[None, :]) * (dilatation_rate / dilatation.clamp_min(1e-12))[None, :]
            internal_torques = torque + quad + shear_couple + lagrangian + unsteady
        else:
            internal_torques = torch.zeros((3, 1), dtype=self.dtype, device=self.device)
        return internal_forces, internal_torques

    def compute_table_contact(self):
        """GPU capsule-vs-box contact; returns rod loads and equal/opposite table reaction."""
        torch = self.torch
        if self.cfg.table_center is None:
            zero = torch.zeros_like(self.rod_position)
            return dict(rod_forces=zero, rod_torques=torch.zeros_like(self.rod_omega),
                        reaction_force=torch.zeros(3, dtype=self.dtype, device=self.device),
                        reaction_torque=torch.zeros(3, dtype=self.dtype, device=self.device),
                        contact_count=torch.zeros((), dtype=torch.int64, device=self.device))
        center, half = self._table_center, self._table_half
        points, element_ids, weights, fractions = self._contact_samples()
        local = points - center
        face_distances = local.abs() - half
        nearest_distance, nearest_axis = face_distances.max(dim=0)
        outside_distance = torch.linalg.vector_norm(face_distances.clamp_min(0), dim=0)
        signed_distance = outside_distance + nearest_distance.clamp_max(0)
        inside = nearest_distance <= 0
        closest_outside = center + torch.maximum(-half, torch.minimum(local, half))
        inside_normals = torch.nn.functional.one_hot(nearest_axis, 3).T.to(self.dtype)
        inside_normals *= torch.where(local >= 0, 1., -1.)
        outside_normals = (points - closest_outside) / outside_distance.clamp_min(1e-15)[None, :]
        normals = torch.where(inside[None, :], inside_normals, outside_normals)
        closest = torch.where(inside[None, :],
                              points - signed_distance[None, :] * normals, closest_outside)
        penetration = self.radius[element_ids] - signed_distance
        active = penetration > 0
        velocity = self._sample_velocities(element_ids, fractions)
        angular_world = torch.einsum("jin,jn->in", self.rod_director, self.rod_omega)
        surface_offset = closest - points
        spin_velocity = torch.linalg.cross(angular_world[:, element_ids].T,
                                           surface_offset.T, dim=1).T
        velocity = velocity + spin_velocity
        normal_speed = torch.sum(velocity * normals, dim=0)
        normal_force = (self.cfg.contact_stiffness * penetration -
                        self.cfg.contact_damping * normal_speed)
        normal_force = torch.where(active, normal_force.clamp_min(0), torch.zeros_like(normal_force))
        tangential = velocity - normal_speed[None, :] * normals
        slip_speed = torch.linalg.vector_norm(tangential, dim=0)
        friction_mag = torch.minimum(self.cfg.tangential_damping * slip_speed,
                                     self.cfg.friction_coefficient * normal_force)
        friction = -friction_mag[None, :] * tangential / (slip_speed[None, :] + 1e-14)
        force = friction + normal_force[None, :] * normals
        node_forces = torch.zeros_like(self.rod_position)
        node_torques = torch.zeros_like(self.rod_omega)
        self._scatter_contact(force, element_ids, weights, fractions, node_forces)
        torque_world = torch.zeros_like(force)
        torque_world = torch.linalg.cross(surface_offset.T, force.T, dim=1).T
        torque_material = torch.einsum("ijn,jn->in", self.rod_director[:, :, element_ids], torque_world)
        node_torques.index_add_(1, element_ids, torque_material * weights[None, :])
        weighted_force = force * weights[None, :]
        reaction = -weighted_force.sum(dim=1)
        reaction_torque = torch.linalg.cross(closest.T - center.T,
                                             -weighted_force.T, dim=1).sum(dim=0)
        return dict(rod_forces=node_forces, rod_torques=node_torques,
                    reaction_force=reaction, reaction_torque=reaction_torque,
                    contact_count=active.sum(), contact_force=force)

    def register_contact_mesh(self, name: str, vertices, triangles) -> None:
        """Upload one closed triangle collider once; contact evaluation is CUDA-only."""
        if not isinstance(name, str) or not name:
            raise ValueError("contact mesh name must be non-empty")
        v = self._tensor(vertices)
        f = self._tensor(triangles, dtype=self.torch.long)
        if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
            raise ValueError("contact mesh must have vertices (V,3) and triangles (F,3)")
        if not torch_all_finite(v) or f.numel() == 0 or int(f.min()) < 0 or int(f.max()) >= len(v):
            raise ValueError("contact mesh contains invalid vertices or triangle indices")
        self._mesh_assets[name] = (v.contiguous(), f.contiguous())
        self.mesh_bodies[name] = _GpuMeshBody(self, name, v.contiguous(), f.contiguous())
        self._physical_model_sha256 = self._model_digest()

    def closest_contact_point(self, name, *, position, rotation):
        """Return the actual closest pad-surface point for a control observation.

        This CUDA query does not advance or constrain the rod. The supplied
        pose is the current mesh COM/world rotation, as at the Isaac boundary.
        """
        torch=self.torch
        local,triangles=self._mesh_assets[name]
        vertices=local @ self._tensor(rotation).T + self._tensor(position)[None,:]
        points,ids,_,_=self._contact_samples()
        closest,distance,winding=self._closest_mesh(points.T,vertices[triangles])
        signed=torch.where(winding.abs()>.5,-distance,distance)
        gaps=signed-self.radius[ids]
        index=gaps.argmin()
        return dict(point_w=closest[index],rod_point_w=points.T[index],
                    gap_m=gaps[index],element_id=ids[index])

    def _model_digest(self):
        payload = {"config": self.cfg.__dict__, "engine": type(self).__name__}
        payload["meshes"] = {
            name: hashlib.sha256(vertices.detach().cpu().numpy().tobytes() +
                                 faces.detach().cpu().numpy().tobytes()).hexdigest()
            for name, (vertices, faces) in self._mesh_assets.items()
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
            default=lambda value: np.asarray(value).tolist()).encode()).hexdigest()

    def compute_mesh_contact(self, name: str, vertices=None, triangles=None, *,
                             body_origin=None, body_velocity=None, body_omega=None,
                             broad_phase=True):
        """GPU signed-distance contact against a closed moving triangle mesh.

        If ``vertices`` is omitted, the uploaded mesh is used as world geometry.
        Signed distance uses a triangle solid-angle sum, so input meshes must be
        closed and consistently oriented (as required by the existing contact path).
        """
        torch = self.torch
        if vertices is None:
            if name not in self._mesh_assets:
                raise KeyError(f"unknown contact mesh {name!r}")
            vertices, triangles = self._mesh_assets[name]
        else:
            vertices = self._tensor(vertices)
            triangles = self._tensor(triangles, dtype=torch.long)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("mesh geometry must have vertices (V,3) and triangles (F,3)")
        points, element_ids, weights, fractions = self._contact_samples()
        faces = vertices[triangles]
        bounds_min = vertices.amin(dim=0) - self.radius.max()
        bounds_max = vertices.amax(dim=0) + self.radius.max()
        candidate_mask = ((points.T >= bounds_min[None, :]) & (points.T <= bounds_max[None, :])).all(dim=1)
        candidate_ids = (torch.nonzero(candidate_mask, as_tuple=False).flatten() if broad_phase else
                         torch.arange(points.shape[1], device=self.device))
        rod_forces = torch.zeros_like(self.rod_position)
        rod_torques = torch.zeros_like(self.rod_omega)
        signed_gaps = torch.full((points.shape[1],), 1.0e6, dtype=self.dtype, device=self.device)
        relative_normal_speeds = torch.zeros_like(signed_gaps)
        if candidate_ids.numel() == 0:
            zeros = torch.zeros(3, dtype=self.dtype, device=self.device)
            return dict(rod_forces=rod_forces, rod_torques=rod_torques,
                        body_force=zeros, body_torque=zeros, body_angular_origin=zeros,
                        rod_linear=zeros, rod_angular_origin=zeros,
                        contact_count=torch.zeros((), dtype=torch.int64, device=self.device),
                        signed_gaps=signed_gaps, sample_points=points.T,
                        relative_normal_speeds=relative_normal_speeds,
                        element_ids=element_ids, candidate_ids=candidate_ids)
        candidate_points = points[:, candidate_ids].T
        closest, distance, winding = self._closest_mesh(candidate_points, faces)
        signed_distance = torch.where(winding.abs() > 0.5, -distance, distance)
        candidate_elements = element_ids[candidate_ids]
        candidate_weights = weights[candidate_ids]
        contact_radius = self.radius[candidate_elements]
        penetration = contact_radius - signed_distance
        active = penetration > 0
        normals = candidate_points - closest
        inside = winding.abs() > 0.5
        normals = torch.where(inside[:, None], -normals, normals)
        normals = normals / torch.linalg.vector_norm(normals, dim=1, keepdim=True).clamp_min(1e-12)
        candidate_fractions = fractions[candidate_ids]
        velocity = self._sample_velocities(element_ids, fractions)[:, candidate_ids].T
        angular_world = torch.einsum("jin,jn->in", self.rod_director, self.rod_omega)
        centerline = candidate_points
        velocity += torch.linalg.cross(angular_world[:, candidate_elements].T,
                                       (closest - centerline), dim=1)
        origin = self._tensor(body_origin if body_origin is not None else vertices.mean(dim=0))
        linear_velocity = self._tensor(body_velocity if body_velocity is not None else self._zero3)
        angular_velocity = self._tensor(body_omega if body_omega is not None else self._zero3)
        velocity -= linear_velocity[None, :] + torch.linalg.cross(
            angular_velocity.expand_as(closest - origin[None, :]), closest - origin[None, :], dim=1)
        normal_speed = torch.sum(velocity * normals, dim=1)
        relative_normal_speeds[candidate_ids] = normal_speed
        signed_gaps[candidate_ids] = signed_distance - contact_radius
        normal_force = (self.cfg.contact_stiffness * penetration -
                        self.cfg.contact_damping * normal_speed)
        normal_force = torch.where(active, normal_force.clamp_min(0), torch.zeros_like(normal_force))
        tangent_velocity = velocity - normal_speed[:, None] * normals
        slip_speed = torch.linalg.vector_norm(tangent_velocity, dim=1)
        friction_mag = torch.minimum(self.cfg.tangential_damping * slip_speed,
                                     self.cfg.friction_coefficient * normal_force)
        friction = -friction_mag[:, None] * tangent_velocity / (slip_speed[:, None] + 1e-14)
        force_world = normal_force[:, None] * normals + friction
        self._scatter_contact(force_world.T, candidate_elements, candidate_weights,
                              candidate_fractions, rod_forces)
        # Contact offset from centerline produces the appropriate rod couple.
        # Apply equal/opposite loads at the same world contact point so the
        # integrated angular impulse balances even with tangential friction.
        offset = closest - centerline
        torque_world = torch.linalg.cross(offset, force_world, dim=1)
        material_torque = torch.einsum("ijn,nj->in", self.rod_director[:, :, candidate_elements], torque_world)
        rod_torques.index_add_(1, candidate_elements, material_torque * candidate_weights[None, :])
        weighted_force = force_world * candidate_weights[:, None]
        body_force = -weighted_force.sum(dim=0)
        body_torque = torch.linalg.cross(closest - origin[None, :], -weighted_force, dim=1).sum(dim=0)
        body_angular_origin = body_torque + torch.linalg.cross(origin, body_force, dim=0)
        rod_linear = rod_forces.sum(dim=1)
        rod_angular_origin = (torch.linalg.cross(self.rod_position.T, rod_forces.T, dim=1).sum(0)
            + torch.einsum("jin,jn->in", self.rod_director, rod_torques).sum(dim=1))
        return dict(rod_forces=rod_forces, rod_torques=rod_torques,
                    body_force=body_force, body_torque=body_torque,
                    body_angular_origin=body_angular_origin,
                    rod_linear=rod_linear, rod_angular_origin=rod_angular_origin,
                    contact_count=active.sum(), signed_gaps=signed_gaps,
                    sample_points=points.T, relative_normal_speeds=relative_normal_speeds,
                    element_ids=element_ids, candidate_ids=candidate_ids)

    def _closest_mesh(self, points, faces, *, face_chunk: int = 2048):
        """Tiled CUDA closest-point and generalized winding-number query."""
        torch = self.torch
        best_d2 = torch.full((len(points),), float("inf"), dtype=self.dtype, device=self.device)
        best_q = torch.zeros_like(points)
        winding = torch.zeros((len(points),), dtype=self.dtype, device=self.device)
        p = points[:, None, :]
        for begin in range(0, len(faces), face_chunk):
            tri = faces[begin:begin + face_chunk]
            a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
            ab = b - a; ac = c - a
            ap = p - a[None, :, :]
            d00 = torch.sum(ab * ab, dim=1); d01 = torch.sum(ab * ac, dim=1)
            d11 = torch.sum(ac * ac, dim=1)
            d20 = torch.sum(ap * ab[None, :, :], dim=2)
            d21 = torch.sum(ap * ac[None, :, :], dim=2)
            denom = (d00 * d11 - d01 * d01).clamp_min(1e-20)
            v = (d11[None, :] * d20 - d01[None, :] * d21) / denom[None, :]
            w = (d00[None, :] * d21 - d01[None, :] * d20) / denom[None, :]
            projection = a[None, :, :] + v[:, :, None] * ab[None, :, :] + w[:, :, None] * ac[None, :, :]
            inside = (v >= 0) & (w >= 0) & (v + w <= 1)
            q_ab = _closest_segment(points, a, b, torch)
            q_bc = _closest_segment(points, b, c, torch)
            q_ca = _closest_segment(points, c, a, torch)
            candidates = torch.stack((projection, q_ab, q_bc, q_ca), dim=2)
            d2 = torch.sum((candidates - p[:, :, None, :]) ** 2, dim=3)
            d2[:, :, 0] = torch.where(inside, d2[:, :, 0], torch.full_like(d2[:, :, 0], float("inf")))
            local_d2, local_id = d2.min(dim=2)
            local_q = torch.gather(candidates, 2, local_id[:, :, None, None].expand(-1, -1, 1, 3)).squeeze(2)
            chunk_d2, ids = local_d2.min(dim=1)
            update = chunk_d2 < best_d2
            best_d2 = torch.where(update, chunk_d2, best_d2)
            best_q = torch.where(update[:, None], local_q[torch.arange(len(points), device=self.device), ids], best_q)
            # Oriented solid angle; winding magnitude > 1/2 denotes interior.
            va = a[None, :, :] - p; vb = b[None, :, :] - p; vc = c[None, :, :] - p
            la = torch.linalg.vector_norm(va, dim=2).clamp_min(1e-20)
            lb = torch.linalg.vector_norm(vb, dim=2).clamp_min(1e-20)
            lc = torch.linalg.vector_norm(vc, dim=2).clamp_min(1e-20)
            numerator = torch.sum(va * torch.linalg.cross(vb, vc, dim=2), dim=2)
            denominator = (la * lb * lc + torch.sum(va * vb, dim=2) * lc +
                           torch.sum(vb * vc, dim=2) * la + torch.sum(vc * va, dim=2) * lb)
            winding += (2.0 * torch.atan2(numerator, denominator)).sum(dim=1)
        return best_q, torch.sqrt(best_d2.clamp_min(0)), winding / (4.0 * torch.pi)

    def _contact_samples(self):
        """Element-local samples matching the CPU backend's contact quadrature."""
        torch = self.torch
        count = self.cfg.contact_subsamples
        if count == 1:
            fractions = torch.full((self.n_elem,), 0.5, dtype=self.dtype, device=self.device)
            fractions[0] = 2.0 / 3.0
            fractions[-1] = 1.0 / 3.0
        else:
            unit = (torch.arange(count, dtype=self.dtype, device=self.device) + 0.5) / count
            fractions = unit.repeat(self.n_elem)
        element_ids = torch.arange(self.n_elem, device=self.device).repeat_interleave(count)
        weights = torch.full((self.n_elem * count,), 1.0 / count,
                             dtype=self.dtype, device=self.device)
        points = (self.rod_position[:, element_ids] * (1.0 - fractions)[None, :] +
                  self.rod_position[:, element_ids + 1] * fractions[None, :])
        return points, element_ids, weights, fractions

    def _sample_velocities(self, element_ids, fractions):
        return (self.rod_velocity[:, element_ids] * (1.0 - fractions)[None, :] +
                self.rod_velocity[:, element_ids + 1] * fractions[None, :])

    def _scatter_contact(self, forces, element_ids, weights, fractions, output):
        weighted = forces * weights[None, :]
        output.index_add_(1, element_ids, weighted * (1.0 - fractions)[None, :])
        output.index_add_(1, element_ids + 1, weighted * fractions[None, :])

    def _rotate_directors(self, dt):
        torch = self.torch
        omega = self.rod_omega
        theta = torch.linalg.vector_norm(omega, dim=0) * dt
        axis = omega / torch.linalg.vector_norm(omega, dim=0).clamp_min(1e-15)[None, :]
        x, y, z = axis
        c = torch.cos(theta); s = torch.sin(theta); one = 1.0 - c
        # PyElastica's director convention uses the passive Rodrigues matrix.
        rot = torch.zeros((3, 3, self.n_elem), dtype=self.dtype, device=self.device)
        rot[0, 0] = c + x*x*one; rot[1, 1] = c + y*y*one; rot[2, 2] = c + z*z*one
        rot[0, 1] = x*y*one + z*s; rot[1, 0] = x*y*one - z*s
        rot[0, 2] = x*z*one - y*s; rot[2, 0] = x*z*one + y*s
        rot[1, 2] = y*z*one + x*s; rot[2, 1] = y*z*one - x*s
        self.rod_director = torch.einsum("ijn,jkn->ikn", rot, self.rod_director)

    def step(self, steps: int = 1, *, dt: float | None = None,
             contact_meshes: Mapping[str, Mapping[str, Any]] | None = None,
             validate_finite: bool = True, return_snapshot: bool = True):
        if int(steps) < 1:
            raise ValueError("steps must be positive")
        torch = self.torch
        diagnostics = {}
        for _ in range(int(steps)):
            h = self.cfg.dt if dt is None else float(dt)
            if not np.isfinite(h) or h <= 0:
                raise ValueError("step dt must be finite and positive")
            old_position = self.rod_position.clone()
            self.rod_position += 0.5 * h * self.rod_velocity
            midpoint_position = self.rod_position.clone()
            self._rotate_directors(0.5 * h)
            # Refresh dilatation/radius before contact, matching PyElastica's
            # internal-force then external-force ordering at the midpoint.
            internal_force, internal_torque = self.compute_internal_forces_and_torques()
            self._external_forces.zero_(); self._external_torques.zero_()
            self._external_forces += self.node_mass[None, :] * self._gravity
            table = self.compute_table_contact()
            self._external_forces += table["rod_forces"]
            self._external_torques += table["rod_torques"]
            diagnostics = {"table": table}
            if contact_meshes:
                for name, motion in contact_meshes.items():
                    vertices = motion.get("vertices")
                    triangles = motion.get("triangles")
                    if vertices is None and name not in self._mesh_assets:
                        raise KeyError(f"contact mesh {name!r} is not registered")
                    if vertices is None:
                        local, triangles = self._mesh_assets[name]
                        rotation = self._tensor(motion.get("rotation", self._eye))
                        translation = self._tensor(motion.get("position", self._zero3))
                        vertices = local @ rotation.T + translation[None, :]
                    contact = self.compute_mesh_contact(name, vertices, triangles,
                        body_origin=motion.get("body_origin", motion.get("position")),
                        body_velocity=motion.get("velocity"), body_omega=motion.get("omega"))
                    self._external_forces += contact["rod_forces"]
                    self._external_torques += contact["rod_torques"]
                    diagnostics[name] = contact
            self.rod_velocity += h * (internal_force + self._external_forces) / self.node_mass[None, :]
            _, _, dilatation, _ = self._geometry()
            alpha = (internal_torque + self._external_torques) * self._inverse_inertia_diag * dilatation[None, :]
            self.rod_omega += h * alpha
            self.rod_velocity *= torch.exp(-self._linear_damping_rate * h)
            self.rod_omega *= torch.exp(-self._rotational_damping_rate * h * dilatation[None, :])
            self.rod_position += 0.5 * h * self.rod_velocity
            self._rotate_directors(0.5 * h)
            first_segment = torch.linalg.vector_norm(midpoint_position - old_position, dim=0).max()
            second_segment = torch.linalg.vector_norm(self.rod_position - midpoint_position, dim=0).max()
            diagnostics["rod_displacement_bound_m"] = first_segment + second_segment
            self.time += h
        self.last_contact_diagnostics = diagnostics
        if validate_finite:
            self._require_finite()
        return self.snapshot(device=True) if return_snapshot else None

    advance = step

    def contact_geometry(self, channel):
        if channel not in self._mesh_assets or channel not in self._mesh_motions:
            raise KeyError(f"no CUDA contact sample for {channel}")
        motion = self._mesh_motions[channel]
        local, triangles = self._mesh_assets[channel]
        vertices = local @ motion["rotation"].T + motion["position"][None, :]
        # Contact broad-phase culling never replaces controller observations
        # with sentinel distances. Query the current outer-boundary state.
        self._geometry()
        detail = self.compute_mesh_contact(channel, vertices, triangles,
            body_origin=motion["position"], body_velocity=motion["velocity"],
            body_omega=motion["omega"], broad_phase=False)
        def host(value):
            return value.detach().cpu().numpy().copy() if hasattr(value, "detach") else value
        return {"gaps": host(detail["signed_gaps"]),
                "relative_normal_speeds": host(detail["relative_normal_speeds"]),
                "element_ids": host(detail["element_ids"]),
                "positions": host(detail["sample_points"])}

    def advance(self, duration: float | None = None):
        interval = self.cfg.dt if duration is None else float(duration)
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("duration must be finite and positive")
        count = max(1, int(np.ceil(interval / self.cfg.dt)))
        h = interval / count
        result = self.step(count, dt=h)
        return {k: (v.detach().cpu().numpy().copy() if hasattr(v, "detach") else v)
                for k, v in result.items()}

    def advance_with_meshes(self, motions: Mapping[str, Any], duration: float):
        """Advance the rod and GPU contacts; export integrated reactions at Isaac boundary."""
        from .moving_contact_models import ContactImpulse, validate_motion
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        if tuple(motions) != tuple(self._mesh_assets):
            raise ValueError("motion channels must match the uploaded collider order")
        n_steps = max(1, int(np.ceil(float(duration) / self.cfg.dt)))
        h = float(duration) / n_steps
        torch = self.torch
        force_impulses = {name: torch.zeros(3, dtype=self.dtype, device=self.device) for name in motions}
        torque_impulses = {name: torch.zeros(3, dtype=self.dtype, device=self.device) for name in motions}
        origin_torque_impulses = {name: torch.zeros(3, dtype=self.dtype, device=self.device) for name in motions}
        rod_linear_impulses = {name: torch.zeros(3, dtype=self.dtype, device=self.device) for name in motions}
        rod_angular_impulses = {name: torch.zeros(3, dtype=self.dtype, device=self.device) for name in motions}
        peaks = {name: torch.zeros((), dtype=self.dtype, device=self.device) for name in motions}
        duration_steps = {name: torch.zeros((), dtype=self.dtype, device=self.device) for name in motions}
        displacement_bound = torch.zeros((), dtype=self.dtype, device=self.device)
        max_radius = self.radius.max()
        validated_motions = {name: validate_motion(motion) for name, motion in motions.items()}
        times = torch.cat(((torch.arange(n_steps, dtype=self.dtype, device=self.device) + .5) * h,
                           self._tensor([duration])))
        predicted = self._predict_mesh_motions(validated_motions, times)
        for index in range(n_steps):
            prepared = {name: {key: value[index, channel] for key, value in predicted.items()}
                        for channel, name in enumerate(motions)}
            self.step(1, dt=h, contact_meshes=prepared, validate_finite=False,
                      return_snapshot=False)
            displacement_bound += self.last_contact_diagnostics["rod_displacement_bound_m"]
            max_radius = torch.maximum(max_radius, self.radius.max())
            for name in motions:
                detail = self.last_contact_diagnostics.get(name)
                if detail is None:
                    continue
                force = detail["body_force"]
                torque = detail["body_torque"]
                force_impulses[name] += force * h
                torque_impulses[name] += torque * h
                origin_torque_impulses[name] += detail["body_angular_origin"] * h
                rod_linear_impulses[name] += detail["rod_linear"] * h
                rod_angular_impulses[name] += detail["rod_angular_origin"] * h
                magnitude = torch.linalg.vector_norm(force)
                peaks[name] = torch.maximum(peaks[name], magnitude)
                duration_steps[name] += (magnitude > 0).to(self.dtype)
        self._require_finite()
        self._mesh_motions = {name: {key: value[-1, channel] for key, value in predicted.items()}
                              for channel, name in enumerate(motions)}
        final = torch.cat((predicted["position"][-1], predicted["rotation"][-1].flatten(1),
                           predicted["velocity"][-1], predicted["omega"][-1]), dim=1).detach().cpu().numpy()
        for name, row in zip(motions, final, strict=True):
            body = self.mesh_bodies[name]
            body.position_collection[:, 0] = row[:3]
            body.director_collection[:, :, 0] = row[3:12].reshape(3, 3).T
            body.velocity_collection[:, 0] = row[12:15]
            body.omega_collection[:, 0] = row[15:18]
        stacked = torch.stack([torch.cat((force_impulses[channel], torque_impulses[channel],
            origin_torque_impulses[channel], rod_linear_impulses[channel], rod_angular_impulses[channel],
            peaks[channel].reshape(1),
            duration_steps[channel].reshape(1))) for channel in motions]).detach().cpu().numpy()
        host_data = {channel:(row[:3],row[3:6],row[6:9],row[9:12],row[12:15],float(row[15]),float(row[16])*h)
                     for channel,row in zip(motions,stacked,strict=True)}
        impulses = {name: ContactImpulse(host_data[name][0], host_data[name][1],
                                         host_data[name][5], host_data[name][6]) for name in motions}
        channels = {}
        evaluations = n_steps
        for name in motions:
            # The rod-side wrench is evaluated independently from the body-side
            # wrench; both use the shared GPU contact point and opposite force.
            mesh_linear = host_data[name][0]
            rod_linear = host_data[name][3]
            mesh_angular = host_data[name][2]
            rod_angular = host_data[name][4]
            channels[name] = {
                "rod_linear_w": rod_linear.copy(), "mesh_linear_w": mesh_linear.copy(),
                "linear_balance_w": (rod_linear + mesh_linear).copy(),
                "rod_angular_origin_w": rod_angular.copy(),
                "mesh_angular_origin_w": mesh_angular.copy(),
                "angular_origin_balance_w": (rod_angular + mesh_angular).copy(),
                "evaluations": evaluations,
            }
        self.last_contact_diagnostics["channels"] = channels
        self.last_contact_diagnostics["evaluation_count"] = evaluations * len(motions)
        self.last_contact_diagnostics["integrated_duration_s"] = float(duration)
        self.last_contact_diagnostics["table_label"] = "table"
        self.last_contact_diagnostics["rod_path_bound_m"] = float(displacement_bound.item())
        self.last_contact_diagnostics["max_rod_radius_m"] = float(max_radius.item())
        return self.snapshot(device=False), impulses

    def _predict_mesh_motions(self, motions, times):
        """Batch prescribed symplectic-Euler pose prediction entirely on CUDA."""
        torch = self.torch
        uploaded = {key: self._tensor(np.stack([getattr(motion, key) for motion in motions.values()]))
                    for key in ("position_w", "rotation_wb", "velocity_w", "omega_w",
                                "acceleration_w", "alpha_w")}
        t = times[:, None, None]
        velocity = uploaded["velocity_w"][None] + uploaded["acceleration_w"][None] * t
        position = uploaded["position_w"][None] + velocity * t
        omega = uploaded["omega_w"][None] + uploaded["alpha_w"][None] * t
        rotation_vector = omega * t
        x, y, z = rotation_vector.unbind(dim=-1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
            len(times), len(motions), 3, 3)
        angle = torch.linalg.vector_norm(rotation_vector, dim=-1)
        a = torch.sinc(angle / torch.pi)
        b = .5 * torch.sinc(angle / (2. * torch.pi)).square()
        increment = self._eye + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)
        rotation = increment @ uploaded["rotation_wb"][None]
        return dict(position=position, rotation=rotation, velocity=velocity, omega=omega,
                    body_origin=position)

    def _require_finite(self):
        for name in ("rod_position", "rod_velocity", "rod_director", "rod_omega"):
            if not bool(self.torch.isfinite(getattr(self, name)).all()):
                raise RuntimeError(f"CUDA Cosserat state became non-finite in {name}")

    def snapshot(self, *, device: bool = True) -> dict[str, Any]:
        values = {"mode": "FREE_TABLE_MOVING_CONTACT", "active_constraints": (),
                  "sim_time": self.time, "rod_position": self.rod_position,
                  "rod_velocity": self.rod_velocity, "rod_director": self.rod_director,
                  "rod_omega": self.rod_omega, "backend": self.backend,
                  "approximation": self.approximation}
        if device:
            return {k: (v.clone() if hasattr(v, "clone") else v) for k, v in values.items()}
        return {k: (v.detach().cpu().numpy().copy() if hasattr(v, "detach") else v) for k, v in values.items()}

    def export_physical_state(self):
        return {"model_sha256": self._physical_model_sha256,
                "dynamic": {key: self.snapshot(device=False)[key].tolist() for key in
                            ("rod_position", "rod_velocity", "rod_director", "rod_omega")},
                "reference": {key: value.detach().cpu().numpy().tolist() for key, value in
                              (("rest_lengths", self.rest_lengths),
                               ("rest_voronoi_lengths", self.rest_voronoi_lengths),
                               ("rest_sigma", self.rest_sigma),
                               ("rest_kappa", self.rest_kappa))}}

    def restore_physical_state(self, state):
        if state.get("model_sha256") != self._physical_model_sha256:
            raise ValueError("CUDA rod physical model identity mismatch")
        dynamic = state["dynamic"]
        for key in ("rod_position", "rod_velocity", "rod_director", "rod_omega"):
            value = self._tensor(dynamic[key])
            if value.shape != getattr(self, key).shape or not bool(self.torch.isfinite(value).all()):
                raise ValueError(f"invalid restored CUDA rod state: {key}")
            setattr(self, key, value.clone())
        expected = {"rest_lengths": self.rest_lengths,
                    "rest_voronoi_lengths": self.rest_voronoi_lengths,
                    "rest_sigma": self.rest_sigma, "rest_kappa": self.rest_kappa}
        for key, reference in expected.items():
            saved = self._tensor(state["reference"][key])
            if saved.shape != reference.shape or not bool(self.torch.equal(saved, reference)):
                raise ValueError(f"CUDA rod reference structure mismatch: {key}")
        frames = self.rod_director.permute(2, 0, 1)
        if (not bool(self.torch.allclose(frames @ frames.transpose(1, 2),
                self._eye.expand_as(frames), atol=1e-6, rtol=0)) or
                bool((self.torch.linalg.det(frames) <= 0).any()) or
                bool((self.torch.linalg.vector_norm(self.rod_position[:, 1:] -
                     self.rod_position[:, :-1], dim=0) < 1e-10).any())):
            raise ValueError("invalid restored CUDA rod geometry or material frames")


def torch_norm(value, torch):
    return torch.linalg.vector_norm(value, dim=0)


def torch_all_finite(value):
    return bool(value.isfinite().all())


def _closest_segment(points, a, b, torch):
    ab = b - a
    t = torch.sum((points[:, None, :] - a[None, :, :]) * ab[None, :, :], dim=2)
    t = t / torch.sum(ab * ab, dim=1).clamp_min(1e-20)[None, :]
    return a[None, :, :] + t.clamp(0.0, 1.0)[:, :, None] * ab[None, :, :]


def create_rod_backend(backend: str, **kwargs: Any) -> Any:
    if backend != "torch_cuda":
        raise ValueError("GPU factory only accepts backend='torch_cuda'; use FreeRodEngine for CPU")
    return TorchCudaCosseratRod(**kwargs)


def resolve_rod_backend(requested: str, *, table_wire: bool, cuda_available: bool) -> str:
    """Resolve the integrated runner's default without silently falling back.

    Table-wire simulation uses CUDA by default.  CPU remains an explicit
    diagnostic/fallback selection; a requested CUDA run without CUDA support
    fails before Isaac is launched.
    """
    if requested not in {"auto", "cpu", "torch_cuda"}:
        raise ValueError("rod backend must be auto, cpu, or torch_cuda")
    if requested == "auto":
        if table_wire and not cuda_available:
            raise RuntimeError("table-wire defaults to torch_cuda, but CUDA is unavailable; select --rod-backend cpu explicitly to opt out")
        return "torch_cuda" if table_wire else "cpu"
    if requested == "torch_cuda":
        if not table_wire:
            raise ValueError("torch_cuda rod backend requires table-wire simulation")
        if not cuda_available:
            raise RuntimeError("torch_cuda rod backend requested but CUDA is unavailable")
    return requested


__all__ = ["GpuRodConfig", "TorchCudaCosseratRod", "create_rod_backend",
           "resolve_rod_backend"]
