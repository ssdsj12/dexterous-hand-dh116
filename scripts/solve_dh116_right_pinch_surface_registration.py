#!/usr/bin/env python3
"""Independent right-hand source-surface registration. CPU proposals, never contact."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.spatial.transform import Rotation
from scripts import audit_dh116_right_pinch_measured_direction as base
from go2_pvcnn.control.dh116.linear_coupling import active_limits, expand
from go2_pvcnn.control.dh116.tabletop_effective_pad import _patch

VERSION = "dh116-right-pinch-source-surface-registration-v1"
PADS = ("finger14_link", "finger23_link")


def _array(value, shape):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError("finite geometry with exact shape required")
    return a


def _unit(value, shape):
    a = _array(value, shape)
    if not np.allclose(np.linalg.norm(a, axis=-1), 1, atol=1e-5, rtol=0):
        raise ValueError("unit normals required")
    return a


class EvaluationBudget:
    def __init__(self, maximum):
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("positive integer budget required")
        self.maximum, self.count = maximum, 0

    def consume(self):
        if self.count >= self.maximum:
            raise ValueError("FK budget exhausted")
        self.count += 1
        return self.count


def proper_registration(points, normals, targets, target_normals):
    """Kabsch of two surface patches plus unit-normal anchors; never reflect root."""
    p, t = _array(points, (2, 3)), _array(targets, (2, 3))
    n, nt = _unit(normals, (2, 3)), _unit(target_normals, (2, 3))
    a, b = np.vstack([p, p + .02*n, p - .02*n]), np.vstack([t, t + .02*nt, t - .02*nt])
    ac, bc = a.mean(0), b.mean(0)
    if min(np.linalg.matrix_rank(a-ac, tol=1e-10), np.linalg.matrix_rank(b-bc, tol=1e-10)) < 2:
        raise ValueError("surface geometry cannot identify a proper root")
    u, _, vt = np.linalg.svd((a-ac).T @ (b-bc))
    correction = np.diag([1., 1., 1. if np.linalg.det(vt.T @ u.T) > 0 else -1.])
    r = vt.T @ correction @ u.T
    m = np.eye(4); m[:3, :3] = r; m[:3, 3] = bc-r@ac
    return dict(matrix=m, rms_m=float(np.sqrt(np.mean((a@r.T+m[:3, 3]-b)**2))),
                point_residual_m=p@r.T+m[:3, 3]-t, normal_residual=n@r.T-nt,
                proper_rotation=True, native_contact_observed=False)


def surface_seed(grid, center, normal):
    """Resolve the locked trilinear SDF zero within 2 mm of an audited patch."""
    c, n = _array(center, (3,)), _unit(normal, (3,))
    offsets = np.linspace(-.002, .002, 65)
    phi, _, valid = base.sample_grid(grid, c[None] + offsets[:, None]*n)
    brackets = [i for i in range(64) if valid[i] and valid[i+1] and phi[i]*phi[i+1] <= 0]
    if not brackets:
        raise ValueError("no locked SDF zero in audited surface neighborhood")
    i = min(brackets, key=lambda i: abs(offsets[i]+offsets[i+1]))
    def f(x):
        v, _, ok = base.sample_grid(grid, [c+x*n])
        if not ok[0]: raise ValueError("surface bracket left locked grid")
        return float(v[0])
    distance = brentq(f, offsets[i], offsets[i+1], xtol=1e-10)
    p = c+distance*n
    value, gradient, valid = base.sample_grid(grid, [p])
    length = np.linalg.norm(gradient[0])
    if not valid[0] or length < 1e-8 or abs(value[0]) > 1e-7:
        raise ValueError("locked surface zero or gradient invalid")
    outward = gradient[0]/length
    if np.dot(outward, n) < .8:
        raise ValueError("locked surface orientation differs from audited region")
    return dict(point_local_m=p, normal_local=outward.astype(float), sdf_value_m=float(value[0]),
                patch_offset_m=float(distance), native_contact_observed=False)


def reflected_targets(pairs, nodes):
    """Reference directions only: left measured pair to right Reset material station."""
    nodes = _array(nodes, (41, 3))
    expected = ["left."+n for n in PADS]
    if len(pairs) != 2 or sorted(p["mesh_name"] for p in pairs) != sorted(expected):
        raise ValueError("one left thumb and index pair required")
    selected = [next(p for p in pairs if p["mesh_name"] == n) for n in expected]
    if any(p.get("substep") != 11 for p in selected):
        raise ValueError("same final solver substep required")
    ids = [p["node_index"] for p in selected]
    if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < 41 for i in ids):
        raise ValueError("actual material node identity required")
    normals = _unit([p["outward_normal_world"] for p in selected], (2, 3))
    normals = normals @ np.diag([1., -1., 1.])
    right_ids = [40-i for i in ids]
    return dict(node_indices=right_ids, points_world_m=nodes[right_ids]-.00485*normals,
                normals_world=normals, source_pairs=selected, reference_only=True,
                teacher_success=False, native_contact_observed=False)


def _pose_vector(matrix):
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.r_[matrix[:3, 3], q[3], q[:3]]


def _separation(vertices, nodes):
    lower = base.rod_support_lower(vertices, nodes)
    if lower < 0:
        certificate = base.certify(vertices, nodes)
        base.validate(certificate, vertices, nodes)
        lower = max(lower, certificate["lower_m"])
    return float(lower)


def geometry_record(g, grids, root, q, nodes):
    expand(q)
    poses = base.fk_poses(g, _pose_vector(root), q)
    table, other = {}, {}
    for kind, key in (("source", "source_colliders"), ("cooked", "cooked_pads")):
        for name, item in g[key].items():
            vertices = base.transform(item["vertices"], poses[name])
            table[kind+":"+name] = float(vertices[:, 2].min()-.3)
            if name not in PADS:
                other[kind+":"+name] = _separation(vertices, nodes)
    pads = {name: base.sample_contact(grids[name], poses[name], nodes) for name in PADS}
    normals = []
    for name, row in pads.items():
        if row["minimum_gap_m"] is None or row["gradient_norm_at_minimum"] <= 0:
            return dict(table_m=table, non_target_rod_gap_m=other, pads=pads,
                        endpoint_passed=False, reason="required_pad_outside_locked_SDF")
        _, grad, _ = base.sample_grid(grids[name], [row["local_node_at_minimum"]])
        normal = poses[name][:3, :3]@grad[0]; normals.append(normal/np.linalg.norm(normal))
    gaps = [r["minimum_gap_m"] for r in pads.values()]
    flags = base.gates(min(table.values()), min(other.values()), min(gaps), max(gaps), float(np.dot(*normals)))
    return dict(table_m=table, non_target_rod_gap_m=other, pads=pads,
                normal_dot=float(np.dot(*normals)), endpoint_passed=flags["static_path_passed"],
                geometric_gates=flags)


def run(result, emit):
    anchor = base.load_anchor()
    g = base.load_geometry(base.ROOT, side="right")
    grids, binding = base.grids_for(g, base.checked_json(base.SDF_DIR/"result.json", base.SDF_SHA))
    last = None
    with (base.evidence.NATIVE/"physical.jsonl").open() as f:
        for line in f:
            event = json.loads(line)
            if event.get("operation") == "physical":
                last = base.decode_owned(event["payload"])["exchange"]
    if last["rod"]["physics_step"] != 234:
        raise ValueError("frozen source tick mismatch")
    pairs = [p for p in last["rod"]["rod_mesh_contact_pairs"]["pairs"]
             if p["substep"] == 11 and p["mesh_name"] in ["left."+n for n in PADS]]
    nodes = np.stack([np.full(41, .43), np.linspace(-.25, .25, 41), np.full(41, .305)], axis=1)
    targets = reflected_targets(pairs, nodes)
    patches, seeds = {}, {}
    for name in PADS:
        patches[name] = _patch(g["source_colliders"][name]["vertices"], "right", name)
        seeds[name] = surface_seed(grids[name], patches[name]["center_local_m"],
                                   patches[name]["reference_outward_normal_local"])
    result.update(source_ledger_sha256=anchor["ledger_sha256"], source_identity=anchor["identity"],
                  geometry_sources=g["source_sha256"], sdf_bindings=binding,
                  source_patches=patches, locked_surface_seeds=seeds, targets=targets,
                  reset_nodes_m=nodes, source_root=g["pose"], candidates=[])
    budget = EvaluationBudget(768)
    lower, upper = active_limits()
    lower = lower[:3]+1e-7; upper = upper[:3]-1e-7
    starting = [anchor["q"][:3], np.asarray(g["active_q"])[:3], (lower+upper)/2]
    def evaluate(q3):
        call = budget.consume()
        q = np.r_[q3, 1.15, 1.15, 1.15]; expand(q)
        poses = base.fk_poses(g, g["pose"], q)
        p = np.array([base.transform(np.array([seeds[n]["point_local_m"]]), poses[n])[0] for n in PADS])
        normals = np.array([poses[n][:3, :3]@seeds[n]["normal_local"] for n in PADS])
        fit = proper_registration(p, normals, targets["points_world_m"], targets["normals_world"])
        root = fit["matrix"]@base.pose_matrix(g["pose"])
        residual = np.r_[fit["point_residual_m"].ravel()/.001, fit["normal_residual"].ravel()/.05]
        emit(dict(phase="fit", evaluation=call, q_rad=q, root_wxyz=_pose_vector(root),
                  residual=residual, rms_m=fit["rms_m"]))
        result["fk_evaluations"] = budget.count
        return residual, q, root, fit
    for index, start in enumerate(starting):
        fit = least_squares(lambda q: evaluate(q)[0], np.clip(start, lower, upper),
                            bounds=(lower, upper), max_nfev=60, diff_step=1e-4,
                            ftol=1e-9, xtol=1e-9, gtol=1e-9)
        residual, q, root, registered = evaluate(fit.x)
        audit = geometry_record(g, grids, root, q, nodes)
        row = dict(seed_index=index, solver_status=int(fit.status), solver_nfev=int(fit.nfev),
                   final_q_rad=q, base_pose_wxyz=_pose_vector(root), registration=registered,
                   residual_norm=float(np.linalg.norm(residual)), geometry=audit)
        result["candidates"].append(row); emit(dict(phase="candidate", **row))
    result.update(status="complete", fk_evaluations=budget.count,
                  endpoint_candidates_passed=sum(r["geometry"]["endpoint_passed"] for r in result["candidates"]),
                  static_path_passed=False, native_ready=False, teacher_admission=False,
                  native_recipe_created=False,
                  limitations=["Proper-root surface fitting is only a CPU proposal.",
                    "All seventeen source envelopes and five locked cooked pads checked independently.",
                    "Full closure path, self collision, fresh Reset tracking and same-rod dynamics remain unqualified.",
                    "Left measured directions are reference seeds, not mirrored right joint commands."])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    result = dict(version=VERSION, status="partial", new_physics_steps=0, ppo_updates=0,
                  native_qualified=False, teacher_admission=False,
                  budget=dict(fk_evaluations=768, seeds=3, nfev_per_seed=60, wall_s=120))
    start = time.monotonic()
    old = signal.signal(signal.SIGALRM, lambda *unused: (_ for _ in ()).throw(TimeoutError("120s CPU budget exhausted")))
    signal.setitimer(signal.ITIMER_REAL, 120)
    try:
        with (args.output/"progress.jsonl").open("x") as f:
            def emit(row):
                f.write(json.dumps(base.encode_owned(row), allow_nan=False)+"\n"); f.flush()
            try: run(result, emit)
            finally: f.flush(); os.fsync(f.fileno())
    except Exception as exc:
        result.update(status="blocked", reason=type(exc).__name__+": "+str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)
    result.update(elapsed_s=time.monotonic()-start, source_sha256=base.digest(__file__))
    with (args.output/"result.json").open("x") as f:
        json.dump(base.encode_owned(result), f, indent=2, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    print(json.dumps({k:result.get(k) for k in ("status", "reason", "fk_evaluations", "endpoint_candidates_passed", "elapsed_s")}))
    return int(result["status"] != "complete")

if __name__ == "__main__":
    raise SystemExit(main())
