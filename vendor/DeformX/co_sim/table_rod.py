"""Unconstrained Cosserat rod interacting with one finite static table mesh.

The table is a frozen rigid mesh and therefore absorbs contact load in this
bounded backend. Moving meshes, finger contacts, and wrench transfer to an
external rigid-body solver remain unimplemented.
"""

from __future__ import annotations

from numbers import Real

import elastica as ea
import numpy as np
import open3d as o3d

from .free_rod import FreeRodEngine
from .models import CoSimConfig


def _finite_vec3(value, name: str, *, positive: bool = False) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {raw.shape}.")
    if any(isinstance(item, (bool, np.bool_)) for item in raw.flat):
        raise TypeError(f"{name} must contain real numeric values, not bool.")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values.") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} must contain only values > 0.")
    return result.copy()


def _finite_scalar(
    value, name: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
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


class TableRodEngine(FreeRodEngine):
    """Propagate a free rod against a closed finite static table box."""

    class _Simulator(
        ea.BaseSystemCollection,
        ea.Constraints,
        ea.Forcing,
        ea.Contact,
        ea.Damping,
    ):
        pass

    def __init__(
        self,
        config: CoSimConfig,
        *,
        table_center,
        table_size,
        contact_k=1e4,
        contact_nu=10.0,
        friction_coefficient=0.5,
        velocity_damping_coefficient=10.0,
        contact_subsamples=1,
        contact_broad_phase=False,
    ) -> None:
        if type(contact_subsamples) is not int or not 1 <= contact_subsamples <= 128:
            raise ValueError('contact_subsamples must be an integer in [1,128]')
        self.contact_subsamples = contact_subsamples
        if type(contact_broad_phase) is not bool:
            raise ValueError('contact_broad_phase must be bool')
        self.contact_broad_phase = contact_broad_phase
        self.table_center = _finite_vec3(table_center, "table_center")
        self.table_size = _finite_vec3(table_size, "table_size", positive=True)
        self.contact_k = _finite_scalar(contact_k, "contact_k", positive=True)
        self.contact_nu = _finite_scalar(contact_nu, "contact_nu", nonnegative=True)
        self.friction_coefficient = _finite_scalar(
            friction_coefficient, "friction_coefficient", nonnegative=True
        )
        self.velocity_damping_coefficient = _finite_scalar(
            velocity_damping_coefficient,
            "velocity_damping_coefficient",
            nonnegative=True,
        )
        super().__init__(config)

    def _setup_simulator(self) -> None:
        size = self.table_size
        box = o3d.geometry.TriangleMesh.create_box(
            width=float(size[0]), height=float(size[1]), depth=float(size[2])
        )
        box.translate(-0.5 * size)
        mesh = ea.Mesh(box)
        self.table_body = ea.MeshRigidBody(
            mesh=mesh,
            center_of_mass=self.table_center,
            density=1.0,
            volume=mesh.compute_volume(),
        )
        self.sim.append(self.table_body)
        self.sim.constrain(self.table_body).using(
            ea.OneEndFixedBC,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,),
        )
        self.sim.detect_contact_between(self.rod, self.table_body).using(
            ea.RodMeshContact,
            k=self.contact_k,
            nu=self.contact_nu,
            friction_coefficient=self.friction_coefficient,
            velocity_damping_coefficient=self.velocity_damping_coefficient,
            mesh_frozen=True,
            subsamples_per_element=self.contact_subsamples,
            broad_phase=self.contact_broad_phase,
        )

    def snapshot(self) -> dict[str, object]:
        result = super().snapshot()
        result.update(
            mode="FREE_TABLE",
            table_center=self.table_center.copy(),
            table_size=self.table_size.copy(),
        )
        return result
