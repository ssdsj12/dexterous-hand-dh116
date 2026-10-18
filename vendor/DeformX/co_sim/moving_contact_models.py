"""Validated value objects for prescribed moving-mesh contact."""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Real

import numpy as np


_VECTOR_FIELDS = (
    "position_w",
    "velocity_w",
    "omega_w",
    "acceleration_w",
    "alpha_w",
)


def _readonly_real_array(value, name: str, shape: tuple[int, ...]) -> np.ndarray:
    source = np.asarray(value, dtype=object)
    if source.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {source.shape}.")
    if any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
        for item in source.flat
    ):
        raise TypeError(f"{name} must contain real numeric values, not bool.")
    result = np.array(source, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return np.frombuffer(result.tobytes(), dtype=float).reshape(shape)


def _finite_scalar(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric scalar, not bool.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return result


def _rotation(value, name: str = "rotation_wb") -> np.ndarray:
    result = _readonly_real_array(value, name, (3, 3))
    if not np.allclose(result.T @ result, np.eye(3), atol=1.0e-7, rtol=0.0):
        raise ValueError(f"{name} must be orthonormal within 1e-7.")
    determinant = float(np.linalg.det(result))
    if determinant <= 0.0 or not np.isclose(
        determinant, 1.0, atol=1.0e-7, rtol=0.0
    ):
        raise ValueError(f"{name} must be a proper rotation with determinant +1.")
    return result


@dataclass(frozen=True)
class MeshMotion:
    """World-frame pose, twist, and acceleration of one prescribed mesh."""

    position_w: np.ndarray
    rotation_wb: np.ndarray
    velocity_w: np.ndarray
    omega_w: np.ndarray
    acceleration_w: np.ndarray
    alpha_w: np.ndarray

    def __post_init__(self) -> None:
        for name in _VECTOR_FIELDS:
            object.__setattr__(self, name, _readonly_real_array(getattr(self, name), name, (3,)))
        object.__setattr__(self, "rotation_wb", _rotation(self.rotation_wb))


@dataclass(frozen=True)
class ContactImpulse:
    """Time-integrated reaction wrench on one mesh, expressed in world axes."""

    linear_w: np.ndarray
    angular_com_w: np.ndarray
    peak_force_n: float
    contact_duration_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "linear_w", _readonly_real_array(self.linear_w, "linear_w", (3,))
        )
        object.__setattr__(
            self,
            "angular_com_w",
            _readonly_real_array(self.angular_com_w, "angular_com_w", (3,)),
        )
        peak_force = _finite_scalar(self.peak_force_n, "peak_force_n")
        duration = _finite_scalar(self.contact_duration_s, "contact_duration_s")
        if peak_force < 0.0:
            raise ValueError("peak_force_n must be >= 0.")
        if duration < 0.0:
            raise ValueError("contact_duration_s must be >= 0.")
        object.__setattr__(self, "peak_force_n", peak_force)
        object.__setattr__(self, "contact_duration_s", duration)


def validate_motion(value: MeshMotion) -> MeshMotion:
    """Return a fully revalidated, independent immutable motion value."""
    if not isinstance(value, MeshMotion):
        raise TypeError("value must be a MeshMotion.")
    return MeshMotion(**{field.name: getattr(value, field.name) for field in fields(MeshMotion)})


def _so3_exp(rotation_vector: np.ndarray) -> np.ndarray:
    """Return a stable SO(3) exponential for a world-axis rotation vector."""
    scale = float(np.max(np.abs(rotation_vector)))
    theta = 0.0 if scale == 0.0 else scale * float(np.linalg.norm(rotation_vector / scale))
    if not np.isfinite(theta):
        raise ValueError("angular update is not finite.")
    x, y, z = rotation_vector
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    theta_squared = theta * theta
    if theta < 1.0e-8:
        a = 1.0 - theta_squared / 6.0 + theta_squared * theta_squared / 120.0
        b = 0.5 - theta_squared / 24.0 + theta_squared * theta_squared / 720.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / theta_squared
    return np.eye(3) + a * skew + b * (skew @ skew)


def predict_motion(value: MeshMotion, dt: float) -> MeshMotion:
    """Advance prescribed motion by one symplectic-Euler interval."""
    current = validate_motion(value)
    step = _finite_scalar(dt, "dt", positive=True)
    with np.errstate(over="ignore", invalid="ignore"):
        velocity = current.velocity_w + current.acceleration_w * step
        position = current.position_w + velocity * step
        omega = current.omega_w + current.alpha_w * step
        rotation_vector = omega * step
    increment = _so3_exp(rotation_vector)
    return MeshMotion(
        position_w=position,
        rotation_wb=increment @ current.rotation_wb,
        velocity_w=velocity,
        omega_w=omega,
        acceleration_w=current.acceleration_w,
        alpha_w=current.alpha_w,
    )


__all__ = ["ContactImpulse", "MeshMotion", "predict_motion", "validate_motion"]
