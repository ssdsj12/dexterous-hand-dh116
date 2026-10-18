"""Opt-in unconstrained Cosserat-rod propagation.

This module models gravity and damping only.  It does not qualify table, finger,
ground, or other contact behavior and deliberately exposes no binding/release API.
"""

from __future__ import annotations

from numbers import Real
from dataclasses import asdict
import hashlib
import json

import elastica as ea
import numpy as np

from .models import CoSimConfig, RodInitialState


def _finite_number(value, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric scalar, not bool.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result}.")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be > 0, got {result}.")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be >= 0, got {result}.")
    return result


def _vec3(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


class FreeRodEngine:
    """Propagate one real PyElastica rod with no constraints or contact."""

    class _Simulator(ea.BaseSystemCollection, ea.Forcing, ea.Damping):
        pass

    def __init__(
        self,
        config: CoSimConfig,
        rod_initial_state: RodInitialState | None = None,
    ) -> None:
        if config.use_ground_contact is not False:
            raise ValueError("use_ground_contact is unsupported by the FREE engine.")
        settle_time = _finite_number(config.settle_time, "settle_time", nonnegative=True)
        if settle_time != 0.0:
            raise ValueError("nonzero settle_time is unsupported by the FREE engine.")
        if config.initial_wire_theta is not None:
            raise ValueError("initial_wire_theta is unsupported by the FREE engine.")

        if isinstance(config.n_elem, (bool, np.bool_)) or not isinstance(
            config.n_elem, (int, np.integer)
        ):
            raise TypeError("n_elem must be an integer >= 2.")
        self.n_elem = int(config.n_elem)
        if self.n_elem < 2:
            raise ValueError("n_elem must be an integer >= 2.")

        self.py_dt = _finite_number(config.py_dt, "py_dt", positive=True)
        self.isaac_dt = _finite_number(config.isaac_dt, "isaac_dt", positive=True)
        base_length = _finite_number(config.base_length, "base_length", positive=True)
        base_radius = _finite_number(config.base_radius, "base_radius", positive=True)
        density = _finite_number(config.density, "density", positive=True)
        youngs_modulus = _finite_number(config.youngs_modulus, "youngs_modulus", positive=True)
        shear_ratio = _finite_number(
            config.shear_modulus_ratio, "shear_modulus_ratio", positive=True
        )
        stretch = _finite_number(
            config.axial_stretch_stiffening,
            "axial_stretch_stiffening",
            positive=True,
        )
        damping = _finite_number(
            config.damping_constant, "damping_constant", nonnegative=True
        )
        self._damping_constant = damping

        source = rod_initial_state
        if source is None:
            start = _vec3(config.rod_start, "rod_start")
            direction = _vec3(config.rod_direction, "rod_direction")
            normal = _vec3(config.rod_normal, "rod_normal")
        else:
            start = _vec3(source.start, "rod.start")
            direction = _vec3(source.direction, "rod.direction")
            normal = _vec3(source.normal, "rod.normal")
        if not np.isclose(np.linalg.norm(direction), 1.0, atol=1.0e-7, rtol=0.0):
            raise ValueError("rod direction must be unit length within 1e-7.")
        if not np.isclose(np.linalg.norm(normal), 1.0, atol=1.0e-7, rtol=0.0):
            raise ValueError("rod normal must be unit length within 1e-7.")
        if not np.isclose(np.dot(direction, normal), 0.0, atol=1.0e-7, rtol=0.0):
            raise ValueError("rod direction and normal must be perpendicular within 1e-7.")

        self.sim = self._Simulator()
        self.rod = ea.CosseratRod.straight_rod(
            n_elements=self.n_elem,
            start=start,
            direction=direction,
            normal=normal,
            base_length=base_length,
            base_radius=base_radius,
            density=density,
            youngs_modulus=youngs_modulus,
            shear_modulus=youngs_modulus / (2.0 * shear_ratio),
        )
        if not np.isclose(stretch, 1.0):
            self.rod.shear_matrix[2, 2, :] *= stretch
        self.sim.append(self.rod)
        self.sim.add_forcing_to(self.rod).using(
            ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -9.81])
        )
        self.sim.dampen(self.rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=damping,
            time_step=self.py_dt,
        )
        self._setup_simulator()
        self.sim.finalize()
        self._rod_damper = self._resolve_rod_damper()
        self.stepper = ea.PositionVerlet()
        self.time = np.float64(0.0)
        self._unusable = False
        self._state_restored = False
        identity = {'config': asdict(config), 'engine': type(self).__name__,
                    'start': start, 'direction': direction, 'normal': normal}
        for name in ('table_center', 'table_size', 'contact_k', 'contact_nu',
                     'friction_coefficient', 'velocity_damping_coefficient',
                     'contact_subsamples', 'contact_broad_phase'):
            if hasattr(self, name): identity[name] = getattr(self, name)
        if hasattr(self, '_prepared_meshes'):
            identity['meshes'] = {k: v[0].sha256 for k, v in self._prepared_meshes.items()}
        self._physical_model_sha256 = hashlib.sha256(json.dumps(
            identity, sort_keys=True, allow_nan=False,
            default=lambda x: np.asarray(x).tolist()).encode()).hexdigest()

    def _setup_simulator(self) -> None:
        """Protected extension hook for opt-in systems registered before finalize."""

    def _resolve_rod_damper(self) -> ea.AnalyticalLinearDamper:
        """Find the pinned PyElastica damper used for partial-step retuning."""
        for group in self.sim._feature_group_damping._operator_collection:
            for operator in group:
                owner = getattr(getattr(operator, "func", None), "__self__", None)
                if isinstance(owner, ea.AnalyticalLinearDamper):
                    return owner
        raise RuntimeError("Unable to locate FREE rod analytical damper.")

    def _set_damping_step(self, step: float) -> None:
        # PyElastica 0.3.x bakes time_step into this private legacy protocol.
        # Keep this narrow compatibility helper until a public retuning API exists.
        self._rod_damper._dampen_rates_protocol = (
            self._rod_damper._deprecated_damping_protocol(
                damping_constant=np.float64(self._damping_constant),
                time_step=np.float64(step),
            )
        )

    def _before_step(self, time: float, step: float) -> None:
        """Protected default-noop hook immediately before each fine step."""

    def _state_arrays(self) -> dict[str, np.ndarray]:
        return {
            "rod_position": self.rod.position_collection,
            "rod_velocity": self.rod.velocity_collection,
            "rod_director": self.rod.director_collection,
            "rod_omega": self.rod.omega_collection,
        }

    def _require_finite_state(self) -> None:
        if not np.isfinite(float(self.time)):
            raise RuntimeError("FREE rod has non-finite simulation time.")
        for name, array in self._state_arrays().items():
            if not np.isfinite(array).all():
                raise RuntimeError(f"FREE rod has non-finite physical state in {name}.")

    def snapshot(self) -> dict[str, object]:
        """Return independent copies of the current measured dynamical state."""
        self._require_finite_state()
        return {
            "mode": "FREE",
            "active_constraints": (),
            "sim_time": float(self.time),
            **{name: array.copy() for name, array in self._state_arrays().items()},
        }

    def export_physical_state(self) -> dict:
        """Portable physical initialization, never a solver/contact checkpoint."""
        self._require_finite_state()
        return {
            'model_sha256': self._physical_model_sha256,
            'dynamic': {k: v.tolist() for k, v in self._state_arrays().items()},
            'reference': {k: getattr(self.rod, k).tolist() for k in (
                'rest_lengths', 'rest_voronoi_lengths', 'rest_sigma', 'rest_kappa')},
        }

    def restore_physical_state(self, state: dict) -> None:
        """Inject into a fresh engine at time zero; original rest model must match.

        Contact forces are rebuilt on the next step, not replayed. Callers must
        revalidate contacts and robot response before granting a stage result.
        """
        if self._unusable or self._state_restored or float(self.time) != 0.:
            raise RuntimeError('physical restore requires a fresh unused engine')
        if state['model_sha256'] != self._physical_model_sha256:
            raise ValueError('physical model identity mismatch')
        values = {}
        for name, target in self._state_arrays().items():
            raw = np.asarray(state['dynamic'][name])
            if raw.dtype.kind not in 'fiu' or raw.shape != target.shape or not np.isfinite(raw).all():
                raise ValueError('invalid rod dynamic state')
            values[name] = raw.astype(float)
        for name in ('rest_lengths', 'rest_voronoi_lengths', 'rest_sigma', 'rest_kappa'):
            raw = np.asarray(state['reference'][name])
            if raw.dtype.kind not in 'fiu' or not np.array_equal(raw, getattr(self.rod, name)):
                raise ValueError('original rod reference structure mismatch')
        frames = np.moveaxis(values['rod_director'], -1, 0)
        if (not np.allclose(frames @ frames.transpose(0,2,1), np.eye(3), atol=1e-6, rtol=0)
                or np.any(np.linalg.det(frames) <= 0)
                or np.any(np.linalg.norm(np.diff(values['rod_position'], axis=1), axis=0) < 1e-10)):
            raise ValueError('invalid rod geometry or material frame')
        self._state_restored = True
        try:
            for name, target in self._state_arrays().items(): target[:] = values[name]
            self.rod.zeroed_out_external_forces_and_torques(np.float64(0.))
            self.rod.compute_internal_forces_and_torques(np.float64(0.))
            self.rod.update_accelerations(np.float64(0.))
            self._require_finite_state()
        except Exception:
            self._unusable = True
            raise

    def advance(self, duration: float | None = None) -> dict[str, object]:
        """Advance the same rod, using a final partial step when necessary."""
        if self._unusable:
            raise RuntimeError("FREE rod engine is unusable after a prior stepping failure.")
        interval = self.isaac_dt if duration is None else _finite_number(
            duration, "duration", positive=True
        )
        start = float(self.time)
        target = start + interval
        if not np.isfinite(target) or target <= start:
            raise ValueError("duration is not representable as a target-time increment.")

        try:
            steps_taken = 0
            while float(self.time) < target:
                current = float(self.time)
                remaining = target - current
                roundoff_tail = float(np.spacing(np.float64(abs(target)))) * max(
                    1, 2 * steps_taken
                )
                if remaining <= roundoff_tail:
                    self.time = np.float64(target)
                    break
                step = min(self.py_dt, remaining)
                if not np.isfinite(step) or step <= 0.0:
                    raise RuntimeError("step calculation made no progress.")
                self._set_damping_step(step)
                self._before_step(current, step)
                new_time = float(self.stepper.step(self.sim, self.time, np.float64(step)))
                if not np.isfinite(new_time) or new_time <= current:
                    raise RuntimeError("stepper failed to make finite forward progress.")
                self.time = np.float64(new_time)
                steps_taken += 1
                self._require_finite_state()
        except Exception:
            self._unusable = True
            raise
        return self.snapshot()
