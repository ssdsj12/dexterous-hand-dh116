"""Prescribed closed-mesh contact for the free table rod backend."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile
from typing import Mapping

import elastica as ea
import numpy as np
import open3d as o3d

from .free_rod import _finite_number
from .moving_contact_models import ContactImpulse, MeshMotion, predict_motion, validate_motion
from .table_rod import TableRodEngine


CHANNELS = ("left_thumb", "left_index", "right_thumb", "right_index")


@dataclass(frozen=True)
class MeshAsset:
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        path = Path(self.path).expanduser().resolve()
        digest = str(self.sha256).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters.")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "sha256", digest)


class _PrescribedMeshBC(ea.ConstraintBase):
    def __init__(self, engine, channel, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine
        self.channel = channel

    def _motion(self, time):
        return self.engine._motion_at(self.channel, float(time))

    def constrain_values(self, system, time):
        motion = self._motion(time)
        if motion is None:
            return
        system.position_collection[:, 0] = motion.position_w
        system.director_collection[:, :, 0] = motion.rotation_wb.T

    def constrain_rates(self, system, time):
        motion = self._motion(time)
        if motion is None:
            return
        system.velocity_collection[:, 0] = motion.velocity_w
        system.omega_collection[:, 0] = motion.rotation_wb.T @ motion.omega_w


class _ContactSampler(ea.NoForces):
    def __init__(self, engine, channel):
        super().__init__()
        self.engine = engine
        self.channel = channel

    def apply_torques(self, system, time=np.float64(0.0)):
        self.engine._sample_contact(self.channel, float(time))


class _BaselineSampler(ea.NoForces):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def apply_torques(self, system, time=np.float64(0.0)):
        self.engine._sample_baseline(float(time))


class MovingContactRodEngine(TableRodEngine):
    """A free table rod coupled to four externally prescribed pad meshes."""

    def __init__(self, config, *, table_center, table_size, meshes, **contact_parameters):
        prepared = self._prepare_assets(meshes)
        self._prepared_meshes = prepared
        self.mesh_bodies = {}
        self._active_motions = None
        self._motion_cache = {}
        self._call_start = None
        self._active_step = None
        self._accumulators = None
        self._prior_rod_force = None
        self._prior_rod_torque = None
        self._step_count = 0
        self._step_generation = 0
        self._baseline_generation = -1
        self._expected_channel_index = 0
        self._expected_sample_time = None
        self.last_contact_diagnostics = None
        super().__init__(
            config,
            table_center=table_center,
            table_size=table_size,
            **contact_parameters,
        )

    @staticmethod
    def _prepare_assets(meshes):
        if not isinstance(meshes, Mapping) or set(meshes) != set(CHANNELS):
            raise ValueError(f"meshes must contain exactly the channels {CHANNELS}.")
        prepared = {}
        for channel in CHANNELS:
            raw = meshes[channel]
            asset = raw if isinstance(raw, MeshAsset) else MeshAsset(*raw)
            try:
                payload = asset.path.read_bytes()
            except OSError as exc:
                raise ValueError(f"Unable to read mesh asset for {channel}.") from exc
            actual = hashlib.sha256(payload).hexdigest()
            if actual != asset.sha256:
                raise ValueError(f"Mesh asset SHA-256 mismatch for {channel}.")
            # Parse the exact bytes that were hashed, not a mutable source path
            # that could change between verification and geometry loading.
            with tempfile.NamedTemporaryFile(suffix=asset.path.suffix) as snapshot:
                snapshot.write(payload)
                snapshot.flush()
                geometry = o3d.io.read_triangle_mesh(snapshot.name)
            if geometry.is_empty() or not geometry.has_triangles():
                raise ValueError(f"Mesh asset {channel} has no triangle geometry.")
            if not geometry.is_watertight() or not geometry.is_orientable():
                raise ValueError(f"Mesh asset {channel} must be closed and orientable.")
            if geometry.is_self_intersecting():
                raise ValueError(f"Mesh asset {channel} must not self-intersect.")
            geometry.compute_triangle_normals()
            mesh = ea.Mesh(geometry, warn_if_not_watertight=False)
            volume = float(mesh.compute_volume())
            inertia = mesh.compute_inertia_tensor(density=1.0, com=np.zeros(3))
            com = mesh.compute_center_of_mass()
            if (
                not np.isfinite(volume)
                or volume <= 0.0
                or not np.isfinite(inertia).all()
                or np.min(np.linalg.eigvalsh(0.5 * (inertia + inertia.T))) <= 0.0
                or not np.isfinite(com).all()
                or np.linalg.norm(com) > 1.0e-6
            ):
                raise ValueError(f"Mesh asset {channel} must have finite positive COM-local mass geometry.")
            prepared[channel] = (asset, mesh, volume, inertia)
        return prepared

    def _setup_simulator(self):
        super()._setup_simulator()
        # Registered after table contact and before every pad contact. This
        # excludes gravity/table loads from the first pad's measured delta.
        self.sim.add_forcing_to(self.rod).using(_BaselineSampler, engine=self)
        for channel in CHANNELS:
            _, mesh, volume, inertia = self._prepared_meshes[channel]
            body = ea.MeshRigidBody(
                mesh=mesh,
                center_of_mass=np.zeros(3),
                density=1.0,
                volume=volume,
                mass_second_moment_of_inertia=inertia,
            )
            self.mesh_bodies[channel] = body
            self.sim.append(body)
            self.sim.constrain(body).using(_PrescribedMeshBC, engine=self, channel=channel)
            self.sim.detect_contact_between(self.rod, body).using(
                ea.RodMeshContact,
                k=self.contact_k,
                nu=self.contact_nu,
                friction_coefficient=self.friction_coefficient,
                velocity_damping_coefficient=self.velocity_damping_coefficient,
                mesh_frozen=False,
                subsamples_per_element=self.contact_subsamples,
                broad_phase=self.contact_broad_phase,
            )
            self.sim.add_forcing_to(self.rod).using(_ContactSampler, engine=self, channel=channel)

    def contact_geometry(self, channel):
        """Instantaneous contact observations at the kernel's quadrature points.

        Includes rod spin and current mesh COM twist. Integrated reaction
        impulses remain the source of contact load/duration evidence.
        """
        body = self.mesh_bodies[channel]
        n = self.contact_subsamples
        contact = ea.RodMeshContact(k=self.contact_k, nu=self.contact_nu,
                                    subsamples_per_element=n)
        positions = self.rod.position_collection
        samples = contact.sample_positions(positions)
        cp, distances, normals = body.query_closest_points(samples)
        element_ids = np.repeat(np.arange(len(self.rod.radius)), n)
        fraction = np.tile((np.arange(n) + .5) / n, len(self.rod.radius))
        if n == 1:  # exactly match the legacy endpoint load/velocity weights
            fraction[0] = 2. / 3.
            if len(fraction) > 1:
                fraction[-1] = 1. / 3.
        application = (positions[:, element_ids] * (1-fraction)
                       + positions[:, element_ids+1] * fraction).T
        velocity = self.rod.velocity_collection
        point_velocity = (velocity[:, element_ids] * (1-fraction)
                          + velocity[:, element_ids+1] * fraction).T
        omega_world = np.einsum('ijk,ik->jk', self.rod.director_collection,
                                self.rod.omega_collection).T
        point_velocity += np.cross(omega_world[element_ids], cp-application)
        mesh_omega = body.director_collection[:, :, 0].T @ body.omega_collection[:, 0]
        relative = point_velocity - body.velocity_collection[:, 0] - np.cross(
            mesh_omega, cp-body.position_collection[:, 0])
        return dict(gaps=np.asarray(distances)-self.rod.radius[element_ids],
                    relative_normal_speeds=np.sum(relative*normals,axis=1),
                    element_ids=element_ids, positions=samples)

    def _motion_at(self, channel, time):
        if self._active_motions is None:
            return None
        elapsed = max(0.0, time - self._call_start)
        if elapsed == 0.0:
            return self._active_motions[channel]
        current = self._active_motions[channel]
        # Inputs were defensively validated and frozen at the outer boundary.
        # Values, rates and contact sampling revisit the same fine-step time.
        # Reuse only the exact same time AND input object; retain one entry per
        # channel so cache size is independent of fine-step count.
        cached = self._motion_cache.get(channel)
        if cached is not None and cached[0] is current and cached[1] == elapsed:
            return cached[2]
        if not any(np.any(v) for v in (current.velocity_w, current.omega_w,
                                      current.acceleration_w, current.alpha_w)):
            predicted = current
        else:
            predicted = predict_motion(current, elapsed)
        self._motion_cache[channel] = (current, elapsed, predicted)
        return predicted

    def _sample_rod_motion_bound(self):
        # PositionVerlet performs two linear position drifts per fine step.
        # Sample at the force midpoint and every step boundary; summing each
        # segment's largest node displacement bounds the entire centerline
        # path, not just the outer-step endpoint displacement.
        position=self.rod.position_collection
        self._rod_path_bound += float(np.linalg.norm(position-self._rod_previous_position,axis=0).max())
        self._rod_previous_position=position.copy()
        self._max_rod_radius=max(self._max_rod_radius,float(self.rod.radius.max()))
        self._position_segment_samples+=1

    def _before_step(self, time, step):
        if self._active_motions is None:
            raise RuntimeError("Moving-contact step has no active prescribed motions.")
        self._sample_rod_motion_bound()
        if self._step_generation and self._expected_channel_index != len(CHANNELS):
            raise RuntimeError("Missed moving-contact callback in the prior fine step.")
        self._active_step = float(step)
        sample_time = float(time) + 0.5 * float(step)
        self._step_generation += 1
        self._baseline_generation = -1
        self._expected_channel_index = 0
        self._expected_sample_time = sample_time
        sample_motions = {
            channel: self._motion_at(channel, sample_time) for channel in CHANNELS
        }
        for channel, body in self.mesh_bodies.items():
            motion = sample_motions[channel]
            body.position_collection[:, 0] = motion.position_w
            body.director_collection[:, :, 0] = motion.rotation_wb.T
            body.velocity_collection[:, 0] = motion.velocity_w
            body.omega_collection[:, 0] = motion.rotation_wb.T @ motion.omega_w
        self._prior_rod_force = self.rod.external_forces.copy()
        self._prior_rod_torque = self.rod.external_torques.copy()
        self._step_count += 1

    def _validate_sample_time(self, time):
        tolerance = 8.0 * np.spacing(np.float64(max(1.0, abs(self._expected_sample_time))))
        if not np.isfinite(time) or abs(time - self._expected_sample_time) > tolerance:
            raise RuntimeError("Stale moving-contact callback time.")

    def _sample_baseline(self, time):
        if self._accumulators is None or self._active_step is None:
            raise RuntimeError("Stale moving-contact baseline evaluation.")
        self._validate_sample_time(time)
        if self._baseline_generation == self._step_generation:
            raise RuntimeError("Repeated moving-contact baseline callback.")
        if self._expected_channel_index != 0:
            raise RuntimeError("Out-of-order moving-contact baseline callback.")
        self._sample_rod_motion_bound()
        self._baseline_generation = self._step_generation
        self._prior_rod_force = self.rod.external_forces.copy()
        self._prior_rod_torque = self.rod.external_torques.copy()

    def _sample_contact(self, channel, time):
        if self._accumulators is None or self._active_step is None:
            raise RuntimeError("Stale moving-contact force evaluation.")
        self._validate_sample_time(time)
        if self._baseline_generation != self._step_generation:
            raise RuntimeError("Moving-contact callback arrived before its baseline.")
        if self._expected_channel_index >= len(CHANNELS):
            raise RuntimeError("Repeated moving-contact channel callback.")
        expected_channel = CHANNELS[self._expected_channel_index]
        if channel != expected_channel:
            raise RuntimeError(
                f"Out-of-order moving-contact callback: expected {expected_channel}, got {channel}."
            )
        body = self.mesh_bodies[channel]
        rod_force = self.rod.external_forces - self._prior_rod_force
        rod_torque = self.rod.external_torques - self._prior_rod_torque
        mesh_force = body.external_forces[:, 0].copy()
        mesh_torque_material = body.external_torques[:, 0].copy()
        mesh_torque_world = body.director_collection[:, :, 0].T @ mesh_torque_material
        rod_torque_world = np.einsum(
            "jik,jk->ik", self.rod.director_collection, rod_torque
        )
        rod_linear = np.sum(rod_force, axis=1)
        rod_angular_origin = np.sum(
            np.cross(self.rod.position_collection.T, rod_force.T), axis=0
        ) + np.sum(rod_torque_world, axis=1)
        mesh_angular_origin = mesh_torque_world + np.cross(
            body.position_collection[:, 0], mesh_force
        )
        weight = self._active_step
        acc = self._accumulators[channel]
        acc["mesh_linear"] += mesh_force * weight
        acc["mesh_angular_com"] += mesh_torque_world * weight
        acc["rod_linear"] += rod_linear * weight
        acc["rod_angular_origin"] += rod_angular_origin * weight
        acc["mesh_angular_origin"] += mesh_angular_origin * weight
        magnitude = float(np.linalg.norm(mesh_force))
        acc["peak"] = max(acc["peak"], magnitude)
        if magnitude > 0.0:
            acc["duration"] += weight
        acc["evaluations"] += 1
        self._expected_channel_index += 1
        self._prior_rod_force = self.rod.external_forces.copy()
        self._prior_rod_torque = self.rod.external_torques.copy()
        body.external_forces.fill(0.0)
        body.external_torques.fill(0.0)

    def advance(self, duration=None):
        raise RuntimeError("MovingContactRodEngine requires advance_with_meshes().")

    def advance_with_meshes(self, motions, duration):
        interval = _finite_number(duration, "duration", positive=True)
        target = float(self.time) + interval
        if not np.isfinite(target) or target <= float(self.time):
            raise ValueError("duration is not representable as a target-time increment.")
        if not isinstance(motions, Mapping) or set(motions) != set(CHANNELS):
            raise ValueError(f"motions must contain exactly the channels {CHANNELS}.")
        validated = {name: validate_motion(motions[name]) for name in CHANNELS}
        accumulators = {
            name: {
                "mesh_linear": np.zeros(3),
                "mesh_angular_com": np.zeros(3),
                "rod_linear": np.zeros(3),
                "rod_angular_origin": np.zeros(3),
                "mesh_angular_origin": np.zeros(3),
                "peak": 0.0,
                "duration": 0.0,
                "evaluations": 0,
            }
            for name in CHANNELS
        }
        old_diagnostics = self.last_contact_diagnostics
        self._active_motions = validated
        self._motion_cache.clear()
        self._call_start = float(self.time)
        self._accumulators = accumulators
        self._step_count = 0
        self._step_generation = 0
        self._baseline_generation = -1
        self._expected_channel_index = 0
        self._expected_sample_time = None
        self._rod_previous_position=self.rod.position_collection.copy()
        self._rod_path_bound=0.
        self._max_rod_radius=float(self.rod.radius.max())
        self._position_segment_samples=0
        try:
            snapshot = super().advance(interval)
            self._sample_rod_motion_bound()
            if self._expected_channel_index != len(CHANNELS):
                raise RuntimeError("Missed moving-contact callback in the final fine step.")
            expected = self._step_count
            if any(acc["evaluations"] != expected for acc in accumulators.values()):
                raise RuntimeError("Missed or duplicate moving-contact force evaluation.")
            impulses = {
                name: ContactImpulse(
                    acc["mesh_linear"],
                    acc["mesh_angular_com"],
                    acc["peak"],
                    acc["duration"],
                )
                for name, acc in accumulators.items()
            }
            diagnostics = {
                "rod_path_bound_m": self._rod_path_bound,
                "max_rod_radius_m": self._max_rod_radius,
                "position_segment_samples": self._position_segment_samples,
                "evaluation_count": sum(acc["evaluations"] for acc in accumulators.values()),
                "integrated_duration_s": interval,
                "channels": {
                    name: {
                        "rod_linear_w": acc["rod_linear"].copy(),
                        "mesh_linear_w": acc["mesh_linear"].copy(),
                        "linear_balance_w": (acc["rod_linear"] + acc["mesh_linear"]).copy(),
                        "rod_angular_origin_w": acc["rod_angular_origin"].copy(),
                        "mesh_angular_origin_w": acc["mesh_angular_origin"].copy(),
                        "angular_origin_balance_w": (
                            acc["rod_angular_origin"] + acc["mesh_angular_origin"]
                        ).copy(),
                        "evaluations": acc["evaluations"],
                    }
                    for name, acc in accumulators.items()
                },
                "table_label": "table",
            }
            self.last_contact_diagnostics = diagnostics
            return snapshot, impulses
        except Exception:
            self._unusable = True
            self.last_contact_diagnostics = old_diagnostics
            raise
        finally:
            self._active_motions = None
            self._motion_cache.clear()
            self._call_start = None
            self._active_step = None
            self._accumulators = None
            self._expected_sample_time = None

    def snapshot(self):
        result = super().snapshot()
        result["mode"] = "FREE_TABLE_MOVING_CONTACT"
        return result


__all__ = ["MeshAsset", "MovingContactRodEngine"]
