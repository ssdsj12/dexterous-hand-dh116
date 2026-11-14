#!/usr/bin/env python3
"""Refine the existing Tripod left quiet-start SDF uncertainty, without Isaac."""
import argparse
from collections import Counter
import json
from pathlib import Path
import signal
import time

import numpy as np

from go2_pvcnn.control.dh116.tabletop_sdf_segment_extrema import segment_extrema
from scripts.audit_dh116_pinch_initial_pose import checked_json, digest, load_geometry, world_vertices
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_grid
from scripts.solve_dh116_tripod_sdf_constrained_direction import (
    grids_for, SDF_DIR, SDF_SHA, load_parent)
from scripts.solve_dh116_tripod_joint_root_direction import envelope_support
from scripts.screen_dh116_right_pinch_closure import BudgetExpired
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.linear_coupling import expand

VERSION = "dh116-tripod-vertical-sdf-reference-v1"
PARENT = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/tripod-two-phase-sdf-20261113-4078wfvx/run01/result.json")
PARENT_SHA = "68b244ea5adee6dada5394b5e46e7e96270b12d3f9d6d1cfb24e8d2a9fad51a4"
CONTACT = Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-pair-build-zQEenV/source/third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh")
CONTACT_SHA = "8d90d9d18c853fbc3931a69435000726f49d36c887e62debe10225e8aa77d2bb"
GUARD_M = 2e-6  # Existing CPU geometric guard; not a proven CUDA rounding bound.


def audit_paths(grids, poses, nodes_start, nodes_end, radius):
    a, b = np.asarray(nodes_start, dtype=float), np.asarray(nodes_end, dtype=float)
    if (not grids or set(grids) != set(poses) or a.ndim != 2 or a.shape[1] != 3
            or a.shape != b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("same pad identities and finite paired node endpoints required")
    result = dict(nodes=[], pads={}, reference_no_contact=False, physics_steps=0,
                  teacher_admitted=False, native_ready=False,
                  scope="fixed_pad_linear_node_vertical_envelope_not_actual_free_fall")
    for pad, grid in grids.items():
        pose = np.asarray(poses[pad], dtype=float)
        if (pose.shape != (4,4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0,0,0,1], atol=1e-12, rtol=0)
                or not np.allclose(pose[:3,:3].T@pose[:3,:3], np.eye(3), atol=1e-10, rtol=0)
                or abs(np.linalg.det(pose[:3,:3])-1)>1e-10):
            raise ValueError("proper rigid local-to-world pad transform required")
        starts, ends = (a-pose[:3,3])@pose[:3,:3], (b-pose[:3,3])@pose[:3,:3]
        pad_rows = []
        for node, (start,end) in enumerate(zip(starts,ends)):
            row = segment_extrema(grid,start,end,radius=radius)
            gap, fraction = row["minimum_gap_m"], row["minimum_at_fraction"]
            witness = None
            if gap is None:
                label = "outside_sampler_lattice"
            else:
                local = start+(end-start)*fraction
                phi,gradient,valid = sample_grid(grid,local[None],dtype=np.float32)
                witness_gap = float(phi[0]-np.float32(radius)) if valid[0] else None
                grad_norm = float(np.linalg.norm(gradient[0])) if valid[0] else None
                witness = dict(local_point_m=local.tolist(),fraction=fraction,
                               valid=bool(valid[0]),gap_m=witness_gap,gradient_norm=grad_norm,
                               contact_predicate=bool(valid[0] and witness_gap<0 and grad_norm>0),
                               native_contact_observed=False)
                if witness["contact_predicate"]:
                    label = "cpu_contact_witness"
                elif gap > GUARD_M:
                    label = "reference_clear_with_guard"
                else:
                    label = "reference_margin_unresolved"
            row.update(pad=pad,node_id=node,classification=label,float32_witness=witness,
                       world_start_m=a[node].tolist(),world_end_m=b[node].tolist(),
                       local_start_m=start.tolist(),local_end_m=end.tolist())
            pad_rows.append(row); result["nodes"].append(row)
        measured = [r for r in pad_rows if r["minimum_gap_m"] is not None]
        minimum = min(measured,key=lambda r:r["minimum_gap_m"]) if measured else None
        result["pads"][pad] = dict(counts=dict(Counter(r["classification"] for r in pad_rows)),
            minimum_reference_gap_m=None if minimum is None else minimum["minimum_gap_m"],
            minimum_node_id=None if minimum is None else minimum["node_id"],
            minimum_fraction=None if minimum is None else minimum["minimum_at_fraction"],
            segments=sum(len(r["segments"]) for r in pad_rows))
    measured = [r["minimum_gap_m"] for r in result["nodes"] if r["minimum_gap_m"] is not None]
    result["minimum_reference_gap_m"] = min(measured) if measured else None
    result["counts"] = dict(Counter(r["classification"] for r in result["nodes"]))
    result["reference_no_contact"] = all(r["classification"] in (
        "reference_clear_with_guard", "outside_sampler_lattice") for r in result["nodes"])
    result.update(cpu_reference_guard_m=GUARD_M,cuda_roundoff_bound_verified=False,
                  gpu_bitwise_verified=False,free_rod_trajectory_verified=False)
    return result


def run(project, result):
    prior = checked_json(PARENT,PARENT_SHA)
    if prior["status"] != "complete" or digest(CONTACT) != CONTACT_SHA:
        raise ValueError("immutable complete Tripod parent and actual sampler required")
    parent = load_parent()
    g = load_geometry(project,side="left")
    if g["source_sha256"] != prior["geometry_sources"]["left"]:
        raise ValueError("left Tripod geometry provenance changed")
    grids, bindings = grids_for(g,checked_json(SDF_DIR/"result.json",SDF_SHA))
    quiet = prior["left_quiet_start"]
    pose, q = quiet["base_pose_wxyz"], quiet["initial_q_rad"]
    expected = np.array(parent["sides"]["left"]["best"]["candidate"]["active_q_rad"])
    expected[1:4] -= .03
    if not np.allclose(q,expected,atol=1e-14,rtol=0):
        raise ValueError("independent Tripod three-digit retreat changed")
    expand(q)
    fk = fk_poses(g,pose,q)
    layout = g["reference"]["reference_layout"]
    nodes = np.array(layout["wire_points"])
    top = layout["table_center"][2]+layout["table_size"][2]/2
    radius = g["old_candidate"]["rod"]["radius_m"]
    if nodes.shape != (41,3) or radius != .005 or not np.all(nodes[:,2] == .308) or top != .3:
        raise ValueError("original 41-node/table/rod source changed")
    end = nodes.copy();end[:,2] = top+radius
    result.update(parent_path=str(PARENT),parent_sha256=PARENT_SHA,
                  contact_source_sha256=CONTACT_SHA,sdf_summary_sha256=SDF_SHA,
                  geometry_sources=g["source_sha256"],sdf_bindings=bindings,
                  independent_style="tripod",side="left",base_pose_wxyz=pose,
                  initial_q_rad=q,registered_rod_axis_z_m=.308,
                  hypothetical_settled_axis_z_m=.305,table_top_m=top,rod_radius_m=radius)
    result["continuous_reference"] = audit_paths(grids,{k:fk[k] for k in grids},nodes,end,radius)
    shapes = {**{"source:"+k:v for k,v in world_vertices(g,pose,q).items()},
              **{"cooked:"+k:v for k,v in world_vertices(g,pose,q,cooked=True).items()}}
    table = {k:float(v[:,2].min()-top) for k,v in shapes.items()}
    envelope = {k:envelope_support(v,layout,radius)["surface_gap_lower_bound_m"] for k,v in shapes.items()}
    for key,current in (("table_gap_by_shape_m",table),("all_shape_rod_envelope_lower_m",envelope)):
        if set(current)!=set(quiet[key]) or any(abs(v-quiet[key][k])>1e-10 for k,v in current.items()):
            raise ValueError("unchanged Tripod geometry no longer reproduces prior evidence")
    result["geometry_recheck"] = dict(table_gap_by_shape_m=table,
        all_shape_rod_envelope_lower_m=envelope,original_clearance_m=CLEARANCE_M,
        table_passed=min(table.values())>=CLEARANCE_M,envelope_passed=min(envelope.values())>=0,
        old_sdf_superset_lower_m=quiet["sdf_contact_region_envelope_lower_m"])
    files = ["scripts/audit_dh116_tripod_vertical_sdf.py",
        "src/go2_pvcnn/control/dh116/tabletop_sdf_segment_extrema.py",
        "scripts/prepare_dh116_pinch_closure_margin.py","scripts/audit_dh116_pinch_initial_pose.py",
        "scripts/solve_dh116_tripod_sdf_constrained_direction.py",
        "scripts/solve_dh116_tripod_joint_root_direction.py",
        "src/go2_pvcnn/control/dh116/linear_coupling.py"]
    result["source_sha256"] = {f:digest(project/f) for f in files}
    result["status"] = "complete"


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--wall-seconds",type=float,default=45)
    args=p.parse_args(argv)
    if not np.isfinite(args.wall_seconds) or not 1<=args.wall_seconds<=45:
        p.error("bounded CPU wall limit 1..45 seconds")
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    result=dict(version=VERSION,status="partial_setup",physics_steps=0,ppo_updates=0,
                native_ready=False,teacher_admitted=False,reset_verified=False)
    def expire(*unused):raise BudgetExpired()
    handler=signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,args.wall_seconds)
    try:run(Path(__file__).resolve().parents[1],result)
    except BudgetExpired:result["status"]="partial_timeout"
    except (ValueError,KeyError) as e:result.update(status="error",error=str(e))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
    result["elapsed_s"]=time.monotonic()-start
    with (args.output/"result.json").open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write("\n");f.flush()
    print(json.dumps({k:result[k] for k in ("status","elapsed_s","physics_steps","native_ready")}))
    return 0 if result["status"]=="complete" else 1

if __name__=="__main__":raise SystemExit(main())
