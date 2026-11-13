#!/usr/bin/env python3
"""Right-owned material-station proposals, independent of left-hand mirroring."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
from scipy.optimize import least_squares
from scripts import solve_dh116_right_pinch_surface_registration as surface

VERSION = "dh116-right-pinch-owned-stations-v1"


def owned_targets(points, nodes, normal_z):
    p = surface._array(points, (2, 3))
    nodes = surface._array(nodes, (41, 3))
    if not np.allclose(nodes[:, [0, 2]], [.43, .305], atol=1e-12, rtol=0) or not np.allclose(
            nodes[:, 1], np.linspace(-.25, .25, 41), atol=1e-12, rtol=0):
        raise ValueError("unchanged ordered straight Reset nodes required")
    if not np.isfinite(normal_z) or not -.42 <= normal_z <= 0:
        raise ValueError("explicit bounded radial direction reference required")
    ids = np.argmin(np.linalg.norm(p[:, None]-nodes[None], axis=-1), axis=1)
    if ids[0] == ids[1] or np.any(abs(p[:, 0]-.43)<1e-5) or np.prod(p[:, 0]-.43) >= 0:
        raise ValueError("distinct opposed right-owned stations required")
    normals = np.stack([-np.sign(p[:, 0]-.43)*np.sqrt(1-normal_z**2),
                        np.zeros(2), np.full(2, normal_z)], axis=1)
    return dict(node_indices=ids.tolist(), normals_world=normals,
                points_world_m=nodes[ids]-.00485*normals, reference_only=True,
                native_contact_observed=False, teacher_success=False,
                station_source="right_source_FK_at_actual_root_and_q__nearest_Reset_material_node",
                normal_source="explicit_radial_candidate_not_a_measured_force")


def run(result, emit):
    base = surface.base
    a = base.load_anchor(); g = base.load_geometry(base.ROOT, side="right")
    grids, bindings = base.grids_for(g, base.checked_json(base.SDF_DIR/"result.json", base.SDF_SHA))
    initial_root = a["actual"]["base_link"]
    initial_q = a["q"].copy(); initial_q[3:] = 1.15
    actual_source_poses = base.fk_poses(g, surface._pose_vector(initial_root), initial_q)
    seeds, patches = {}, {}
    for n in surface.PADS:
        patches[n] = surface._patch(g["source_colliders"][n]["vertices"], "right", n)
        seeds[n] = surface.surface_seed(grids[n], patches[n]["center_local_m"],
                                       patches[n]["reference_outward_normal_local"])
    right_points = np.array([base.transform(np.array([seeds[n]["point_local_m"]]), actual_source_poses[n])[0]
                             for n in surface.PADS])
    nodes = np.stack([np.full(41, .43), np.linspace(-.25, .25, 41), np.full(41, .305)], axis=1)
    budget = surface.EvaluationBudget(768)
    low, high = surface.active_limits()
    low, high = low[:3]+1e-7, high[:3]-1e-7
    result.update(source_identity=a["identity"], source_ledger_sha256=a["ledger_sha256"],
                  geometry_sources=g["source_sha256"], sdf_bindings=bindings, patches=patches,
                  surface_seeds=seeds, actual_root=initial_root, initial_q_rad=initial_q,
                  right_source_points_world_m=right_points, reset_nodes_m=nodes, candidates=[])
    for case, nz in enumerate((0., -.2, -.42)):
        target = owned_targets(right_points, nodes, nz)
        def evaluate(q3):
            count = budget.consume()
            q = np.r_[q3, 1.15, 1.15, 1.15]; surface.expand(q)
            poses = base.fk_poses(g, surface._pose_vector(initial_root), q)
            p = np.array([base.transform(np.array([seeds[n]["point_local_m"]]), poses[n])[0] for n in surface.PADS])
            normals = np.array([poses[n][:3, :3]@seeds[n]["normal_local"] for n in surface.PADS])
            fit = surface.proper_registration(p, normals, target["points_world_m"], target["normals_world"])
            root = fit["matrix"]@initial_root
            residual = np.r_[fit["point_residual_m"].ravel()/.001, fit["normal_residual"].ravel()/.05]
            emit(dict(phase="fit", case=case, evaluation=count, q_rad=q,
                      root_wxyz=surface._pose_vector(root), residual=residual))
            result["fk_evaluations"] = count
            return residual, q, root, fit
        fitted = least_squares(lambda q: evaluate(q)[0], np.clip(initial_q[:3], low, high),
                               bounds=(low, high), max_nfev=60, diff_step=1e-4,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        residual, q, root, fit = evaluate(fitted.x)
        audit = surface.geometry_record(g, grids, root, q, nodes)
        row = dict(case=case, normal_z=nz, target=target, q_rad=q,
                   root_wxyz=surface._pose_vector(root), registration=fit,
                   residual_norm=float(np.linalg.norm(residual)),
                   solver_nfev=int(fitted.nfev), solver_status=int(fitted.status), geometry=audit)
        result["candidates"].append(row); emit(dict(phase="candidate", **row))
    result.update(status="complete", endpoint_candidates_passed=sum(r["geometry"]["endpoint_passed"] for r in result["candidates"]),
                  static_path_passed=False, native_ready=False, native_recipe_created=False,
                  teacher_admission=False,
                  limitations=["Right-owned station selection and radial target directions are geometric proposals only.",
                    "No actual contact, reset tracking, full closure path, self collision or Teacher qualification."])


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    out = dict(version=VERSION, status="partial", new_physics_steps=0, ppo_updates=0,
               native_qualified=False, teacher_admission=False,
               budget=dict(fk_evaluations=768, radial_cases=3, nfev_per_case=60, wall_s=120))
    start = time.monotonic()
    old = signal.signal(signal.SIGALRM, lambda *unused: (_ for _ in ()).throw(TimeoutError("120s CPU budget exhausted")))
    signal.setitimer(signal.ITIMER_REAL, 120)
    try:
        with (a.output/"progress.jsonl").open("x") as f:
            def emit(row):
                f.write(json.dumps(surface.base.encode_owned(row), allow_nan=False)+"\n"); f.flush()
            try: run(out, emit)
            finally: f.flush(); os.fsync(f.fileno())
    except Exception as exc:
        out.update(status="blocked", reason=type(exc).__name__+": "+str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)
    out.update(elapsed_s=time.monotonic()-start, source_sha256=surface.base.digest(__file__),
               registration_source_sha256=surface.base.digest(surface.__file__))
    with (a.output/"result.json").open("x") as f:
        json.dump(surface.base.encode_owned(out), f, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    print(json.dumps({k:out.get(k) for k in ("status", "reason", "fk_evaluations", "endpoint_candidates_passed", "elapsed_s")}))
    return int(out["status"] != "complete")

if __name__ == "__main__":
    raise SystemExit(main())
