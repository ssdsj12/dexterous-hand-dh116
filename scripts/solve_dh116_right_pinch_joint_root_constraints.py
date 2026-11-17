#!/usr/bin/env python3
"""Bounded right joint/root proposals with all collision constraints in residual."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scripts import solve_dh116_right_pinch_surface_registration as surface

VERSION = "dh116-right-pinch-joint-root-constraints-v1"
PARENT = Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-owned-surfaces-20261113-ArN16oy2/run01/result.json")
PARENT_SHA = "b702f1a267b6ace0a5db2e5300351a2b19ce365b24726f35ac948f6bfd417f2a"


def root_candidate(root, q, x):
    root = surface._array(root, (4, 4)); q = surface._array(q, (6,)); x = surface._array(x, (9,))
    if not np.allclose(root[3], [0, 0, 0, 1]) or not np.allclose(root[:3, :3].T@root[:3, :3], np.eye(3), atol=1e-6):
        raise ValueError("proper homogeneous source root required")
    if np.linalg.det(root[:3, :3]) < .999999:
        raise ValueError("reflected source root forbidden")
    if np.any(abs(x)>2+1e-12):
        raise ValueError("fixed root/joint proposal range exceeded")
    changed = q.copy(); changed[:3] += .2*x[:3]; surface.expand(changed)
    result = root.copy()
    result[:3, :3] = Rotation.from_rotvec(.3*x[6:]).as_matrix()@root[:3, :3]
    result[:3, 3] += .01*x[3:6]
    return result, changed


def constraint_residual(audit, x):
    gaps = [audit["pads"][name]["minimum_gap_m"] for name in surface.PADS]
    if any(v is None or not np.isfinite(v) for v in gaps) or "normal_dot" not in audit:
        raise ValueError("actual locked-grid values required for both pad proposals")
    # These residuals propose, never waive the independently recomputed gates.
    r = [(v+.00015)/.001 for v in gaps]
    r.extend(max(0., .0022-v)/.001 for v in audit["table_m"].values())
    r.extend(max(0., .0001-v)/.001 for v in audit["non_target_rod_gap_m"].values())
    r.append(max(0., audit["normal_dot"]+.25))
    r.extend(.002*np.asarray(x))
    r = np.asarray(r, float)
    if not np.isfinite(r).all(): raise ValueError("finite complete geometry residual required")
    return r


def run(out, emit):
    base = surface.base
    parent = base.checked_json(PARENT, PARENT_SHA)
    parent = base.decode_owned(parent)
    if parent["endpoint_candidates_passed"] != 0: raise ValueError("exact failed owned-reference evidence required")
    a = base.load_anchor(); g = base.load_geometry(base.ROOT, side="right")
    grids, bindings = base.grids_for(g, base.checked_json(base.SDF_DIR/"result.json", base.SDF_SHA))
    nodes = parent["reset_nodes_m"]
    q0 = a["q"].copy(); q0[3:] = 1.15
    first = (a["actual"]["base_link"], q0)
    c = parent["candidates"][0]
    second = (base.pose_matrix(c["root_wxyz"]), np.asarray(c["q_rad"]))
    out.update(parent_sha256=PARENT_SHA, source_identity=a["identity"], source_ledger_sha256=a["ledger_sha256"],
               geometry_sources=g["source_sha256"], sdf_bindings=bindings, candidates=[], reset_nodes_m=nodes)
    budget = surface.EvaluationBudget(1200); lowq, highq = surface.active_limits()
    for case, (root, q) in enumerate((first, second)):
        lower = np.r_[np.maximum(-2., (lowq[:3]+1e-6-q[:3])/.2), [-2.]*6]
        upper = np.r_[np.minimum(2., (highq[:3]-1e-6-q[:3])/.2), [2.]*6]
        best = None
        def query(x):
            nonlocal best
            count = budget.consume()
            changed, joints = root_candidate(root, q, x)
            audit = surface.geometry_record(g, grids, changed, joints, nodes)
            residual = constraint_residual(audit, x)
            row = dict(case=case, evaluation=count, x=np.asarray(x), root_wxyz=surface._pose_vector(changed),
                       q_rad=joints, geometry=audit, residual_norm=float(np.linalg.norm(residual)))
            if best is None or row["residual_norm"] < best["residual_norm"]: best=row
            out["fk_evaluations"] = count
            emit(dict(phase="nonlinear_proposal", **row))
            return residual
        fit = least_squares(query, np.clip(np.zeros(9), lower+1e-8, upper-1e-8),
                            bounds=(lower, upper), max_nfev=55, diff_step=1e-3,
                            ftol=1e-8, xtol=1e-8, gtol=1e-8)
        query(fit.x)
        row = dict(case=case, solver_status=int(fit.status), nfev=int(fit.nfev), selected=best)
        out["candidates"].append(row); emit(dict(phase="selected", **row))
    out.update(status="complete", endpoint_candidates_passed=sum(r["selected"]["geometry"]["endpoint_passed"] for r in out["candidates"]),
               static_path_passed=False, native_ready=False, native_recipe_created=False,
               teacher_admission=False, actual_contact_observed=False,
               limitations=["All-shape static endpoint proposals only; full closure/Reset/self-collision/tracking not qualified.",
                            "Original table2.1mm,depth-0.3mm,end-0.1mm,opposition-0.2 gates unchanged."])


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    out=dict(version=VERSION,status="partial",new_physics_steps=0,ppo_updates=0,native_qualified=False,teacher_admission=False,
             budget=dict(fk_evaluations=1200,cases=2,nfev_per_case=55,wall_s=120))
    old=signal.signal(signal.SIGALRM,lambda *unused:(_ for _ in ()).throw(TimeoutError("120s CPU budget exhausted")))
    signal.setitimer(signal.ITIMER_REAL,120)
    try:
        with (args.output/"progress.jsonl").open("x") as f:
            def emit(row):
                f.write(json.dumps(surface.base.encode_owned(row),allow_nan=False)+"\n");f.flush()
            try:run(out,emit)
            finally:f.flush();os.fsync(f.fileno())
    except Exception as exc:out.update(status="blocked",reason=type(exc).__name__+": "+str(exc))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    out.update(elapsed_s=time.monotonic()-start,source_sha256=surface.base.digest(__file__),
               geometry_source_sha256=surface.base.digest(surface.__file__))
    with (args.output/"result.json").open("x") as f:
        json.dump(surface.base.encode_owned(out),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:out.get(k) for k in ("status","reason","fk_evaluations","endpoint_candidates_passed","elapsed_s")}))
    return int(out["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
