"""Fit a guarded 902 Pinch endpoint with coupled wrist and finger corrections."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from scripts.audit_dh116_certified_segment_distance import _group_envelope_bounds, segment_hull_bounds
from scripts.fit_dh116_902_pinch_alignment import (
    MOVE_SCALE, PADS, PAD_LIMIT_M, ROD_RADIUS_M, _boundary_context, _side_fit,
)
from scripts.verify_dh116_902_flexible_reset import sha


WARM_SHA = {
    "left": "2328a0643f260c54517c76540d196e3f2d5d616162c3454cc1c4f40eeaf58949",
    "right": "14d431563ee931463952fb27773b84a85f749c387711d025fa1279d8afc405d3",
}
# Search span around the sealed endpoint. Workspace limits still come from the
# live controller contract and are intersected before every proposed step.
LOCAL_SPAN = np.array([1.25] * 3 + [2.] * 3 + [1.65] * 3)
FINITE_STEP = .001
GUARD_TOLERANCE = 1e-7


def minimax_step(gaps, gap_derivatives, reserves, reserve_derivatives, basis, low, high, radius):
    """Minimize the larger pad gap while retaining every linearized reserve."""
    n = basis.shape[0]
    planes = np.vstack((
        np.column_stack((gap_derivatives @ basis, -np.ones(len(gaps)))),
        np.column_stack((-reserve_derivatives @ basis, np.zeros(len(reserves)))),
        np.column_stack((basis, np.zeros(n))),
        np.column_stack((-basis, np.zeros(n))),
    ))
    bounds = np.r_[-gaps, reserves, high, -low]
    result = linprog(np.r_[np.zeros(n), 1.], A_ub=planes, b_ub=bounds,
                     bounds=[(-radius, radius)] * n + [(None, None)], method="highs")
    if not result.success:
        raise ValueError(f"guarded local linear solve failed: {result.message}")
    step = basis @ result.x[:n]
    active = np.flatnonzero(reserves + reserve_derivatives @ step <= .0001).tolist()
    return step, active


def _node_basis(nodes, station, origin, rotation):
    """World translation and wrist-local rotation in one measured wire frame."""
    tangent = nodes[station + 1] - nodes[station - 1]
    tangent /= np.linalg.norm(tangent)
    radial = origin - nodes[station]
    radial -= tangent * (tangent @ radial)
    radial /= np.linalg.norm(radial)
    world_axes = np.column_stack((tangent, radial, np.cross(tangent, radial)))
    basis = np.eye(9)
    basis[:3, :3] = world_axes
    basis[3:6, 3:6] = rotation.inv().apply(world_axes.T).T
    return basis


def add_support_directions(tube, nodes, directions):
    """Augment the original tube with unit directions evaluated on every archived node."""
    extra = np.asarray(directions, dtype=float)
    lengths = np.linalg.norm(extra, axis=1)
    if not np.isfinite(extra).all() or np.any(lengths <= 0.):
        raise ValueError("finite, nonzero rod envelope witness directions required")
    extra = extra / (lengths[:, None] * (1. + 1e-12))
    upper = np.stack([(nodes[:, start:start + 3].reshape(-1, 3) @ extra.T).max(axis=0) + ROD_RADIUS_M
                      for start in range(0, 40, 2)])
    return np.vstack((tube[0], extra)), np.column_stack((tube[1], upper))


def _refine_nearest(world, nodes, tube):
    """Refine selected-pad witnesses without changing the original non-pad proof."""
    normals, upper = tube
    augmented, evidence = {}, {}
    for label, vertices in world.items():
        if label.split(":")[1] not in PADS:
            continue
        plane_bounds = ((vertices @ normals.T).min(axis=0)[None, :] - upper).max(axis=1)
        witnesses = []
        for group in np.argsort(plane_bounds)[:2]:
            if plane_bounds[group] > -PAD_LIMIT_M + .00015:
                break
            start = int(group) * 2
            fit = _group_envelope_bounds(vertices, nodes[:, start:start + 3].reshape(-1, 3))
            if fit is not None:
                witnesses.append(dict(group=int(group), kind="full_group", normal=fit["normal"]))
                continue
            for segment in (start, start + 1):
                try:
                    bounds = segment_hull_bounds(vertices, nodes[-1, segment], nodes[-1, segment + 1])
                except ValueError:
                    continue
                axis = np.asarray(bounds["hull_point_m"]) - bounds["segment_point_m"]
                if np.linalg.norm(axis) > 0.:
                    witnesses.append(dict(group=int(group), kind="segment", normal=axis.tolist()))
        if witnesses:
            augmented[label] = add_support_directions(tube, nodes, [w["normal"] for w in witnesses])
            evidence[label] = witnesses
    return augmented, evidence


def _solve(coordinates, low, high, basis, evaluate, labels, iterations):
    state = evaluate(coordinates)
    if min(state[2:]) < -GUARD_TOLERANCE:
        raise ValueError("sealed warm endpoint requires original conservative guard reserves")
    radius = .16
    records = []
    for ordinal in range(1, iterations + 1):
        # Use central differences except when a joint or wrist is at its original bound.
        jacobian = np.empty((len(state), len(coordinates)))
        for i, direction in enumerate(np.eye(len(coordinates))):
            p = min(FINITE_STEP, max(0., high[i] - coordinates[i]))
            n = min(FINITE_STEP, max(0., coordinates[i] - low[i]))
            plus = evaluate(coordinates + direction * p) if p else state
            minus = evaluate(coordinates - direction * n) if n else state
            jacobian[:, i] = (plus - minus) / (p + n)
        # Keep a small solver cushion where one already exists, then use
        # direct nonlinear readbacks to accept or reject the actual step.
        reserves = state[2:]
        cushion = np.minimum(.003, np.maximum(0., reserves * .5))
        step, active = minimax_step(state[:2], jacobian[:2], reserves - cushion, jacobian[2:],
                                    basis, low - coordinates, high - coordinates, radius)
        best_trial = None
        attempted = []
        for fraction in (1., .5, .25, .125, .0625):
            candidate = np.clip(coordinates + step * fraction, low, high)
            trial = evaluate(candidate)
            blocker = labels[int(np.argmin(trial[2:]))] if min(trial[2:]) < -GUARD_TOLERANCE else None
            gap = max(trial[:2])
            attempted.append(dict(fraction=fraction, pad_gaps_mm=trial[:2].tolist(), guard=blocker))
            if blocker is None and gap < max(state[:2]) - .00005:
                if best_trial is None or gap < max(best_trial[1][:2]):
                    best_trial = (candidate, trial, fraction)
        records.append(dict(ordinal=ordinal, radius=radius, step_norm=float(np.linalg.norm(step)),
                            trials=attempted, accepted=best_trial is not None,
                            linear_active_guards=[labels[i] for i in active]))
        if best_trial is not None:
            coordinates, state, fraction = best_trial
            radius = min(.4, radius * (1.5 if fraction >= .5 else .75))
        else:
            radius *= .5
        if max(state[:2]) < -.005 or radius < .001:
            break
    return coordinates, records


def _warm_seed(path, side, source):
    if sha(path) != WARM_SHA[side]:
        raise ValueError("sealed side-specific earlier Pinch fit required")
    report = json.loads(path.read_bytes())
    if any(report[key] != source[key] for key in ("outer_sha256", "last_row_sha256", "archive_sha256")):
        raise ValueError("coupled correction must use the same measured approach")
    hand = report["hands"][side] if side == "left" else report
    measured, best = hand["measured"], hand["best"]
    position = np.asarray(best["pose_wxyz"][:3]) - measured["pose_wxyz"][:3]
    q = np.asarray(best["active_q_rad"][:3]) - measured["active_q_rad"][:3]
    rotations = np.asarray([measured["pose_wxyz"], best["pose_wxyz"]])[:, [4, 5, 6, 3]]
    start, end = Rotation.from_quat(rotations)
    return np.r_[position, (start.inv() * end).as_rotvec(), q] / MOVE_SCALE, best


def fit(receipt, warm_path, *, side, iterations=32):
    if side not in WARM_SHA or type(iterations) is not int or not 1 <= iterations <= 64:
        raise ValueError("explicit side and bounded CPU iteration count required")
    context, models, source = _boundary_context(receipt, sides=(side,))
    initial, expected = _warm_seed(warm_path, side, source)

    def optimize(*, station, rotation, origin, state, measure, lower, upper, labels, world_shapes):
        if expected["pad_stations"] != [station, station] or not 1 <= station < 40:
            raise ValueError("warm endpoint must retain the original material station")
        lower = np.maximum(lower, initial - LOCAL_SPAN)
        upper = np.minimum(upper, initial + LOCAL_SPAN)
        if np.any(lower >= upper) or np.any(initial < lower) or np.any(initial > upper):
            raise ValueError("warm endpoint lies outside the original workspace")
        warm = measure(initial)
        if not np.allclose(warm["pad_gaps_m"], expected["pad_gaps_m"], atol=1e-10, rtol=0.):
            raise ValueError("sealed warm pad gaps failed direct replay")
        basis = _node_basis(context["nodes"][-1], station, origin, rotation)
        coordinates, steps = _solve(initial, lower, upper, basis, lambda x: state(x, station),
                                    labels, iterations)
        sampled = measure(coordinates)
        augmented, witnesses = _refine_nearest(world_shapes(coordinates), context["nodes"], context["tube"])
        refined_steps = []
        if augmented and max(sampled["pad_gaps_m"]) >= -.000005:
            coordinates, refined_steps = _solve(coordinates, lower, upper, basis,
                                                lambda x: state(x, station, augmented), labels, iterations)
        best = measure(coordinates)
        if best["pad_stations"] != [station, station]:
            best["endpoint_passed"] = False
            best["failures"].append("requested_station")
        return dict(warm=warm, sampled_axis_endpoint=sampled, best=best, steps=steps,
                    witness_steps=refined_steps, additional_rod_witnesses=witnesses, station=station,
                    basis_world=basis[:3, :3].tolist(), correction=((coordinates - initial) * MOVE_SCALE).tolist())

    outcome = _side_fit(side, **context, model=models[side], max_evaluations=1, starts=1, optimize=optimize)
    return dict(version="dh116-902-pinch-coupled-alignment-v1", **source, **outcome, side=side,
                coupling_source_sha256=sha(Path(__file__)), warm_sha256=WARM_SHA[side],
                iteration_limit_per_phase=iterations, scope="cpu_geometric_endpoint_only",
                command_path_qualified=False, native_contact_verified=False, teacher_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--warm-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--side", required=True, choices=tuple(WARM_SHA))
    parser.add_argument("--iterations", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh coupled alignment evidence output required")
    result = fit(args.receipt, args.warm_result, side=args.side, iterations=args.iterations)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0 if result["best"]["endpoint_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
