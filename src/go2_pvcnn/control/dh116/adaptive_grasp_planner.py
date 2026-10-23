"""Candidate ranking for privileged DH116 grasp selection.

The planner only filters and ranks candidates.  It does not move Panda/DH116,
attach the wire, or report a physical grasp.  A candidate must pass the
reachability/collision callbacks and the conservative quasistatic support LP;
the selected result is then consumed by the existing motion/contact loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .adaptive_force_controller import allocate_preload


@dataclass(frozen=True)
class RankedGraspCandidate:
    candidate_id: str
    material_ids: tuple[int, int, int, int]
    points: np.ndarray
    normals: np.ndarray
    support: Mapping[str, Any]
    score: float
    risk: float
    alternates: tuple[str, ...]


def _candidate_geometry(candidate: Mapping[str, Any]):
    if not isinstance(candidate, Mapping):
        raise TypeError("grasp candidates must be mappings")
    cid = candidate.get("candidate_id")
    if not isinstance(cid, str) or not cid:
        raise ValueError("candidate_id must be a non-empty string")
    points = np.asarray(candidate.get("points"), dtype=float)
    normals = np.asarray(candidate.get("normals"), dtype=float)
    ids = tuple(candidate.get("material_ids", ()))
    if (points.shape != (4, 3) or normals.shape != (4, 3)
            or len(ids) != 4 or any(type(i) is not int or i < 0 for i in ids)
            or not np.isfinite(points).all() or not np.isfinite(normals).all()
            or not np.allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-6, rtol=0)):
        raise ValueError("grasp candidates require four finite points, unit normals and material IDs")
    return cid, points, normals, ids


def rank_grasp_candidates(
    wire_nodes: Any,
    candidates: Sequence[Mapping[str, Any]],
    *,
    mass_kg: float = 0.1,
    mu: float = 0.5,
    max_load_n: float | Sequence[float] = 3.0,
    uncertainty: float = 0.1,
    collision_check: Callable[[Mapping[str, Any]], Any] | None = None,
    reachability_check: Callable[[Mapping[str, Any]], Any] | None = None,
    score_weights: Mapping[str, float] | None = None,
) -> list[RankedGraspCandidate]:
    """Return feasible candidates in ascending cost order.

    ``wire_nodes`` is used to reject empty/non-finite privileged geometry and
    to compute no hidden contact force.  The callbacks are intentionally
    external so USD/IK state remains outside this pure planner boundary.
    """
    wire = np.asarray(wire_nodes, dtype=float)
    if wire.ndim != 2 or wire.shape[1] != 3 or len(wire) < 2 or not np.isfinite(wire).all():
        raise ValueError("wire_nodes must contain at least two finite XYZ samples")
    weights = dict(reach=1., collision=1., load=1., bending=1., visibility=1.)
    if score_weights is not None:
        for key, value in score_weights.items():
            if key not in weights or not np.isfinite(value) or value < 0:
                raise ValueError("score weights must be finite non-negative known terms")
            weights[key] = float(value)
    accepted: list[RankedGraspCandidate] = []
    for candidate in candidates:
        cid, points, normals, ids = _candidate_geometry(candidate)
        if reachability_check is not None and not bool(reachability_check(candidate)):
            continue
        if collision_check is not None:
            decision = collision_check(candidate)
            safe = bool(decision.get("safe", False)) if isinstance(decision, Mapping) else bool(decision)
            if not safe:
                continue
        support = allocate_preload(
            points, normals, np.asarray(candidate.get("com", wire.mean(axis=0)), dtype=float),
            mass_kg=float(candidate.get("mass_kg", mass_kg)),
            mu=float(candidate.get("mu", mu)), max_load_n=max_load_n,
            uncertainty=uncertainty,
        )
        if not support["feasible"]:
            continue
        target = np.asarray(support["target_loads_n"], dtype=float)
        cap = np.asarray(max_load_n, dtype=float)
        cap_sum = float(np.sum(cap)) if cap.ndim else 4. * float(cap)
        load_cost = float(np.sum(target) / max(cap_sum, 1e-12))
        terms = {
            "reach": float(candidate.get("reach_cost", 0.)),
            "collision": float(candidate.get("collision_risk", 0.)),
            "load": load_cost,
            "bending": float(candidate.get("rod_bending_cost", 0.)),
            "visibility": float(candidate.get("visibility_uncertainty", 0.)),
        }
        if any(not np.isfinite(value) or value < 0 for value in terms.values()):
            raise ValueError("candidate costs must be finite and non-negative")
        score = float(sum(weights[key] * value for key, value in terms.items()))
        accepted.append(RankedGraspCandidate(
            candidate_id=cid, material_ids=ids, points=points.copy(), normals=normals.copy(),
            support=support, score=score,
            risk=float(terms["collision"] + terms["visibility"]), alternates=(),
        ))
    accepted.sort(key=lambda item: (item.score, item.candidate_id))
    alternates = tuple(item.candidate_id for item in accepted)
    return [RankedGraspCandidate(
        candidate_id=item.candidate_id, material_ids=item.material_ids,
        points=item.points, normals=item.normals, support=item.support,
        score=item.score, risk=item.risk, alternates=alternates,
    ) for item in accepted]


__all__ = ["RankedGraspCandidate", "rank_grasp_candidates"]
