"""Canonical force reduction for a separate successor to the frozen Pinch ledger."""
from copy import deepcopy

import numpy as np

from .deformx_dh116_colliders import DISTAL_PAD_LINKS
from .tabletop_deformable_grasp_runtime import validate_pinch_ledger


def validate_ledger(rows):
    """Check original producer semantics and the independent unchanged 1 mN limit."""
    canonical = deepcopy(rows)
    names = (*DISTAL_PAD_LINKS, "table")
    independent, mismatches = [], 0
    for row in canonical:
        forces = np.asarray(row["rod_node_contact_forces_w"], dtype=np.float64)
        wrenches = row["rod_mesh_wrenches_w"]
        if set(wrenches) != set(names):
            raise ValueError("exact named mesh force sources required")
        mesh = np.asarray([wrenches[name] for name in names], dtype=np.float64)
        reported = row["pair_force_residual_n"]
        if (forces.shape != (41, 3) or mesh.shape != (6, 6)
                or not np.isfinite(forces).all() or not np.isfinite(mesh).all()
                or type(reported) not in (int, float) or not np.isfinite(reported)):
            raise ValueError("complete finite raw force vectors required")
        mesh_force = mesh[:, :3].sum(0)
        producer = float(np.linalg.norm(forces.astype(np.float32).sum(0) + mesh_force))
        residual = float(np.linalg.norm(forces.sum(0) + mesh_force))
        if not np.isclose(reported, producer, atol=1e-9, rtol=1e-6):
            raise ValueError("reported force residual differs from the original FP32 producer")
        if residual > 1e-3:
            raise ValueError("independent FP64 force residual exceeds the original limit")
        mismatches += int(not np.isclose(reported, residual, atol=1e-9, rtol=1e-6))
        independent.append(residual)
        row["pair_force_residual_n"] = residual
    result = validate_pinch_ledger(canonical)
    return dict(result, force_reduction="original_fp32_producer_replayed_then_independent_fp64_balance",
                force_precision_mismatch_rows=mismatches,
                max_independent_force_residual_n=max(independent))
