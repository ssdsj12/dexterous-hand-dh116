#!/usr/bin/env python3
"""Three bounded nonlinear trust directions per Pinch hand; CPU geometry only."""
import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
from scipy.optimize import linprog

from scripts import audit_dh116_pinch_spread_axial_direction as previous
from scripts.audit_dh116_pinch_initial_pose import finite_array, digest, load_geometry, path_displacement_bound
from scripts.screen_dh116_right_pinch_closure import BudgetExpired, _commit_row_without_alarm

VERSION = "dh116-pinch-bounded-nonlinear-trust-v1"
PARENT = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/"
              "pinch-spread-axial-20261113-q3um_jla/run01/result.json")
PARENT_SHA = "de7a80babae315a71398d203cbed0cf7ec97a331aaf3b8264240ef0b6fa52c3d"
PREVIOUS_SOURCE_SHA = "fa1ac1aebea269bfe6e319ec2841ea1c42ef26f3e1d2c6c5182f72b04ac90dcb"
MAX_DIRECTIONS = 3
FD_QUERY_POINTS = 21
MAX_STATE_ATTEMPTS = 352
WORST_CASE_STATES = 2 * (MAX_DIRECTIONS * (FD_QUERY_POINTS * 2 + 2) + 33)
INITIAL_TRUST_RADIUS = .2
MODEL_RELATIVE_LIMIT = .2
ACCEPT_RATIO = .1
MERIT_IMPROVEMENT_MM = 1e-6
HARD_NUMERIC_TOL_MM = 1e-6  # 1 nm numerical comparison; never replaces physical final gates.


class BudgetLimit(RuntimeError):
    pass


class StateBudget:
    """Reserve and retain every attempted complete FK/SDF state, including errors."""
    def __init__(self, records, maximum=MAX_STATE_ATTEMPTS):
        if type(maximum) is not int or not 1 <= maximum <= MAX_STATE_ATTEMPTS:
            raise ValueError("1..352 shared complete geometry states required")
        self.records, self.maximum, self.count = records, maximum, 0

    def call(self, function, argument, **metadata):
        if self.count >= self.maximum:
            raise BudgetLimit("shared complete geometry-state cap")
        self.count += 1
        entry = dict(attempt=self.count, status="started", **metadata)
        with _commit_row_without_alarm():
            self.records.append(entry)
        try:
            row = function(argument)
            with _commit_row_without_alarm():
                entry.update(status="complete", measurement=row)
            return row
        except BaseException as error:
            with _commit_row_without_alarm():
                entry.update(status="interrupted" if isinstance(error, BudgetExpired) else "error",
                             error=type(error).__name__ + ": " + str(error))
            raise


def values(row, labels):
    if row["labels"] != labels:
        raise ValueError("constraint order changed")
    return finite_array(row["values"], (len(labels),))


def stencil(x, index, lower, upper, requested):
    """Central when possible; otherwise second-order one-sided, entirely in bounds."""
    room_left, room_right = x[index] - lower[index], upper[index] - x[index]
    if room_left >= requested and room_right >= requested:
        h, mode, signs = requested, "central", (1., -1.)
    else:
        direction = 1. if room_right >= room_left else -1.
        room = room_right if direction > 0 else room_left
        h = min(requested, room / 2)
        if h <= 1e-12:
            raise ValueError("no numerically resolvable bounded FD stencil")
        mode, signs = ("forward" if direction > 0 else "backward"), (direction, 2*direction)
    probes = []
    for sign in signs:
        p = x.copy()
        p[index] = np.clip(x[index] + sign*h, lower[index], upper[index])
        if not np.all((p >= lower) & (p <= upper)):
            raise ValueError("FD probe outside original bounds")
        probes.append(p)
    return dict(variable=index, mode=mode, step=float(h), points=[p.tolist() for p in probes])


def local_model(query, center, lower, upper, radius):
    center, lower, upper = [finite_array(a, (5,)) for a in (center, lower, upper)]
    if (not np.all(lower <= center) or not np.all(center <= upper)
            or not np.all(lower < upper) or not 0 < radius <= INITIAL_TRUST_RADIUS):
        raise ValueError("bounded current point and trust radius required")
    base = query(center.copy()); labels = base["labels"]; c = values(base, labels)
    jacobians, all_stencils = [], []
    # Shrinking the trust radius also shrinks the diagnostic probe scale.
    for requested in (radius/20, radius/40):
        columns, stencils = [], []
        for j in range(5):
            s = stencil(center, j, lower, upper, requested)
            first, second = [values(query(np.array(p)), labels) for p in s["points"]]
            if s["mode"] == "central":
                column = (first-second)/(2*s["step"])
            else:
                sign = 1 if s["mode"] == "forward" else -1
                column = sign*(-3*c+4*first-second)/(2*s["step"])
            columns.append(column); stencils.append(s)
        jacobians.append(np.stack(columns, axis=1)); all_stencils.append(stencils)
    relative = float(np.linalg.norm(jacobians[1]-jacobians[0])
                     / max(np.linalg.norm(jacobians[1]), 1e-12))
    return dict(baseline=base, jacobians=[j.tolist() for j in jacobians],
        stencils=all_stencils, jacobian_relative_difference=relative,
        supported=relative <= MODEL_RELATIVE_LIMIT)


def contact_mask(labels):
    return np.array([s.endswith(":contact_minus0p1mm") for s in labels], dtype=bool)


def violation(c):
    return max(0., -float(np.min(c)))


def assess(labels, before, predicted, actual):
    before, predicted, actual = [finite_array(c, (len(labels),)) for c in (before, predicted, actual)]
    hard = ~contact_mask(labels)
    hard_ok = bool(np.all(actual[hard] >= -HARD_NUMERIC_TOL_MM))
    old, forecast, measured = map(violation, (before, predicted, actual))
    predicted_drop, actual_drop = old-forecast, old-measured
    ratio = actual_drop/predicted_drop if predicted_drop > MERIT_IMPROVEMENT_MM else None
    accepted = bool(hard_ok and ratio is not None and ratio >= ACCEPT_RATIO
                    and actual_drop > MERIT_IMPROVEMENT_MM)
    return dict(accepted=accepted, hard_constraints_preserved=hard_ok,
        before_max_violation_mm=old, predicted_max_violation_mm=forecast,
        actual_max_violation_mm=measured, predicted_improvement_mm=predicted_drop,
        actual_improvement_mm=actual_drop, ratio=ratio,
        hard_numeric_tolerance_mm=HARD_NUMERIC_TOL_MM)


def refine(query, initial, lower, upper, trace):
    """At most one LP and one real candidate per direction; no line-search retries."""
    center = finite_array(initial, (5,)).copy()
    lower, upper = [finite_array(a, (5,)) for a in (lower, upper)]
    radius = INITIAL_TRUST_RADIUS; accepted_count = 0
    for ordinal in range(MAX_DIRECTIONS):
        item = dict(direction=ordinal+1, center_x=center.tolist(), trust_radius_before=radius,
                    solver_calls=0, accepted=False, status="evaluating")
        with _commit_row_without_alarm():
            trace.append(item)
        try:
            model = local_model(query, center, lower, upper, radius)
            item["model"] = model
            if not model["supported"]:
                item["reason"] = "two_scale_model_disagreement"
            else:
                labels = model["baseline"]["labels"]; c = values(model["baseline"], labels)
                soft = contact_mask(labels)
                if not soft.any():
                    raise ValueError("explicit target-contact constraints required")
                jac = finite_array(model["jacobians"][1], (len(c), 5))
                step_lower, step_upper = np.maximum(lower-center, -radius), np.minimum(upper-center, radius)
                # Only target-contact deficit receives slack. All initial/endpoint
                # table, non-target and bounded-penetration design constraints stay hard.
                fit = linprog([0., 0., 0., 0., 0., 1.],
                    A_ub=-np.column_stack((jac, soft.astype(float))), b_ub=c,
                    bounds=[*zip(step_lower, step_upper), (0, None)], method="highs",
                    options=dict(maxiter=100, time_limit=1., threads=2))
                item.update(solver_calls=1, solver_status=int(fit.status), solver_message=fit.message)
                if fit.success:
                    proposed = center+fit.x[:5]
                    if np.any(proposed < lower-1e-9) or np.any(proposed > upper+1e-9):
                        raise ValueError("LP proposal outside original bounds")
                    proposed = np.clip(proposed, lower, upper)
                    predicted = c+jac@(proposed-center)
                    item.update(proposed_x=proposed.tolist(), predicted_values=predicted.tolist(),
                                predicted_common_contact_slack_mm=float(fit.x[-1]))
                    # No model-predicted candidate can advance the accepted center.
                    actual = query(proposed)
                    item["actual"] = actual
                    decision = assess(labels, c, predicted, values(actual, labels))
                    item["assessment"] = decision; item["accepted"] = decision["accepted"]
                    item["reason"] = "actual_predicted_improvement" if item["accepted"] else "nonlinear_trial_rejected"
                    if item["accepted"]:
                        center = proposed.copy(); accepted_count += 1
                else:
                    item["reason"] = "bounded_lp_no_solution"
        except ValueError as error:
            item.update(reason="source_or_constraint_unavailable", error=str(error))
        if not item["accepted"]:
            radius *= .5
        item.update(status="complete", trust_radius_after=radius, selected_x=center.tolist())
    return dict(selected_x=center.tolist(), accepted_directions=accepted_count,
                final_trust_radius=radius, global_infeasibility_proven=False)


def run(project, result):
    if digest(project/"scripts/audit_dh116_pinch_spread_axial_direction.py") != PREVIOUS_SOURCE_SHA:
        raise ValueError("unchanged original spread/axial source required")
    seed = previous.checked_json(PARENT, PARENT_SHA)
    pair = previous.load_pair_proposal(); nodes, identity = previous.native_nodes()
    if (seed["status"] != "complete" or seed["geometry_state_attempts"] != 150
            or seed["pair_sha256"] != previous.PAIR_SHA or seed["reset_seed_sha256"] != previous.SEED_SHA
            or seed["parent_identity"] != identity or not np.array_equal(nodes, seed["actual_parent_nodes_m"])):
        raise ValueError("same two nonlinear seed candidates and actual original rod nodes required")
    result.update(parent_result_path=str(PARENT), parent_result_sha256=PARENT_SHA,
        pair_sha256=previous.PAIR_SHA, reset_seed_sha256=previous.SEED_SHA,
        parent_receipt_sha256=previous.RECEIPT_SHA, parent_ledger_sha256=previous.LEDGER_SHA,
        parent_identity=identity, actual_parent_nodes_m=nodes.tolist(), sources={}, sides={},
        state_records=[], original_runtime_floor_m=previous.RUNTIME_FLOOR,
        design_targets_are_not_changed_physics_thresholds=True,
        variable_scales=previous.SCALES.tolist(), theoretical_maximum_states=WORST_CASE_STATES,
        shared_hard_cap=MAX_STATE_ATTEMPTS, maximum_directions_per_side=MAX_DIRECTIONS)
    files = ("scripts/solve_dh116_pinch_trust_region.py",
        "scripts/audit_dh116_pinch_spread_axial_direction.py",
        "scripts/audit_dh116_pinch_raised_reset_direction.py",
        "scripts/prepare_dh116_dual_pinch_readback_profile.py",
        "scripts/prepare_dh116_pinch_closure_margin.py",
        "scripts/solve_dh116_tripod_sdf_constrained_direction.py",
        "scripts/audit_dh116_pinch_initial_pose.py",
        "src/go2_pvcnn/control/dh116/tabletop_measured_closure_geometry.py",
        "src/go2_pvcnn/control/dh116/linear_coupling.py")
    result["implementation_sha256"] = {f: digest(project/f) for f in files}
    budget = StateBudget(result["state_records"])
    try:
        for side in ("left", "right"):
            g = load_geometry(project, side=side); parent = dict(pair[side]["candidate"], side=side)
            expected = parent["source_sha256"] if side == "left" else pair[side]["source_sha256"]
            if g["source_sha256"] != expected:
                raise ValueError("independent original side geometry differs")
            grids, binding = previous.grids_for(g, previous.checked_json(previous.SDF_DIR/"result.json", previous.SDF_SHA))
            result["sources"][side] = dict(geometry_sha256=g["source_sha256"], sdf_bindings=binding)
            x0 = finite_array(seed["sides"][side]["candidate"]["normalized_direction"], (5,))
            if previous.decode(parent, x0) != seed["sides"][side]["candidate"]:
                raise ValueError("seed was rebased instead of using original absolute variable bounds")
            lower, upper = previous.bounds(parent)
            side_result = dict(initial_x=x0.tolist(), original_lower=lower.tolist(), original_upper=upper.tolist(),
                               directions=[], static_candidate_verified=False)
            result["sides"][side] = side_result; start_count = budget.count

            def measure(argument):
                x, fraction = argument; selected = previous.decode(parent, x)
                row = previous.previous.measure(g, selected, grids, nodes, np.zeros(3), fraction)
                row["static_reset_delta"] = selected["static_reset_delta"]
                return row

            def state(x, fraction, phase):
                return budget.call(measure, (x, fraction), side=side, phase=phase,
                                   parameter_x=np.asarray(x).tolist(), path_fraction=float(fraction))

            def query(x):
                first = state(x, 0., "direction_initial")
                last = state(x, 1., "direction_endpoint")
                labels, c = previous.initial_constraints(first)
                return dict(labels=labels+["endpoint:"+n for n in last["labels"]],
                            values=c+last["values"], initial=first, endpoint=last)

            outcome = refine(query, x0, lower, upper, side_result["directions"])
            side_result.update(outcome)
            x = np.array(outcome["selected_x"]); path = []
            side_result["path"] = path
            for fraction in np.linspace(0, 1, 33):
                row = state(x, float(fraction), "final_path")
                with _commit_row_without_alarm():
                    path.append(row)
            candidate = previous.decode(parent, x)
            q, delta = np.array(candidate["active_q_rad"]), np.array(candidate["active_delta_rad"])
            travel = {**{"source:"+n: v for n, v in path_displacement_bound(g, q, delta).items()},
                      **{"cooked:"+n: v for n, v in path_displacement_bound(g, q, delta, cooked=True).items()}}
            table = {k: previous.path_lower([r["table_m"][k] for r in path], b) for k, b in travel.items()}
            others = {k: previous.path_lower([r["all_shape_rod_lower_m"][k] for r in path], b)
                      for k, b in travel.items() if k.split(":")[1] not in previous.previous.PADS}
            initial_clear = min(path[0]["all_shape_rod_lower_m"].values()) >= 0
            contact = all(-.0003 <= path[-1]["node_sdf"][n]["minimum_gap_m"] <= -.0001
                          and bool(path[-1]["node_sdf"][n]["contact_predicate_nodes"]) for n in previous.previous.PADS)
            labels, c = previous.initial_constraints(path[0])
            labels += ["endpoint:"+n for n in path[-1]["labels"]]; c += path[-1]["values"]
            design = bool(np.all(np.array(c)[~contact_mask(labels)] >= -HARD_NUMERIC_TOL_MM))
            side_result.update(candidate=candidate, point_travel_m=travel,
                continuous_table_lower_m=table, continuous_non_target_lower_m=others,
                initial_all_shape_rod_clear=bool(initial_clear), endpoint_both_pad_target_contact=bool(contact),
                noncontact_design_constraints_preserved=design, geometry_states=budget.count-start_count,
                **previous.gate(min(table.values()), min(others.values()), initial_clear and design, contact))
        result["status"] = "complete"
    finally:
        result["geometry_state_attempts"] = budget.count
        result["completed_geometry_states"] = sum(r["status"] == "complete" for r in result["state_records"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv); a.output.mkdir(parents=True, exist_ok=False); start = time.monotonic()
    result = dict(version=VERSION, status="partial_setup", physics_steps=0, training_updates=0,
        reset_proposal_created=False, native_qualification=False, teacher_admission=False,
        global_infeasibility_proven=False)
    def timeout(*unused):
        raise BudgetExpired()
    old = signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, 50)
    try:
        run(Path(__file__).resolve().parents[1], result)
    except BudgetExpired:
        result["status"] = "partial_timeout"
    except (ValueError, BudgetLimit) as error:
        result.update(status="diagnostic_error", error=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)
    result["elapsed_s"] = time.monotonic()-start
    with (a.output/"result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({k: result.get(k) for k in ("status", "elapsed_s", "geometry_state_attempts", "error")}))
    return int(result["status"] == "diagnostic_error")


if __name__ == "__main__":
    raise SystemExit(main())
