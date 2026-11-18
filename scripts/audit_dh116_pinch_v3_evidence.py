#!/usr/bin/env python3
"""Read-only replay of the completed left Pinch v3 measurement and failed exit."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from scripts.audit_dh116_pinch_initial_pose import (
    capsule_gap, checked_json, digest, load_geometry, pose_matrix, transform,
)
from scripts.dh116_pinch_evidence_v2 import require_quiet, validate_action_targets
from scripts.run_dh116_work_packages import debugger_status
from scripts.validate_dh116_physx_lifetime import capture_evidence
from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS
from go2_pvcnn.control.dh116.tabletop_pinch_initial_geometry_v2 import validate_pinch_initial_geometry


RECEIPT_SHA = "fe0aa842d11ba118e3151e3a80ba58f7d60507548fade93b0dab88f6a300681f"
LEDGER_SHA = {
    "physics.jsonl": "d002b0f8cab4108b91bd92976acbb0e95629aab856e949bffb279c240a7c580a",
    "pairs.jsonl": "8415ef2f62d790d3a4f86b4cbdc91e89700b3d98a25050d4a0d62c7274479762",
    "table_contacts.jsonl": "729ca5be095976d7ae078d7235706cf497baa0480f03c6ba032d7ad48f2aa7fe",
    "result.json": "449921f1c3bba4ddb1969f48cecca220724088d0c37fe891b461e04c07454b30",
    "app_close_returned.json": "37b1ba15219bb2264687d2883ed6129cc5cd54fe64d0d7afbb1e5ca7d9308ae4",
}
NAMES = (*DISTAL_PAD_LINKS, "table")
REQUIRED = ("finger14_link", "finger23_link")


def residual_precision(nodes, meshes, reported):
    node = np.asarray(nodes,dtype=np.float64)
    mesh = np.asarray(meshes,dtype=np.float64)
    if (node.shape!=(41,3) or mesh.shape!=(6,6)
            or not np.isfinite(node).all() or not np.isfinite(mesh).all() or not np.isfinite(reported)):
        raise ValueError("finite original node and named mesh forces required")
    mesh_total = mesh[:,:3].sum(0)
    fp32 = float(np.linalg.norm(node.astype(np.float32).sum(0) + mesh_total))
    fp64 = float(np.linalg.norm(node.sum(0) + mesh_total))
    return dict(producer_fp32_n=fp32, independent_fp64_n=fp64,
        matches_producer=bool(np.isclose(fp32,reported,atol=1e-9,rtol=1e-6)),
        matches_legacy_fp64=bool(np.isclose(fp64,reported,atol=1e-9,rtol=1e-6)),
        independent_balance_passed=fp64 <= .001)


def ledger(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("incomplete raw physical ledger")
            yield json.loads(line)


def replay(project, receipt_path):
    receipt = checked_json(receipt_path,RECEIPT_SHA)
    output = Path(receipt["output"])
    for name, expected in LEDGER_SHA.items():
        if digest(output/name) != expected:
            raise ValueError(f"raw evidence changed: {name}")
    outer_path = receipt_path.parent/"outer_exit.json"
    outer = checked_json(outer_path,"1e1caf1e0ada1dbf71ccdb6c5845280f39ba15adf12497c119a09c16d90d7b40")
    log = receipt_path.parent/"native.log"
    if digest(log)!=outer["log_sha256"] or debugger_status(log)["debugger_inferior_exit_code"]!=1:
        raise ValueError("original outer or inferior status differs")
    result = json.loads((output/"result.json").read_text())
    if (result["physics_steps"]!=240 or result["lifetime_released_before_close"] is not True
            or "candidate hand/table/rod force accounting differs" not in result["failure"]
            or outer["outer_exit_code"]!=1 or outer["debugger_inferior_signal"] is not None):
        raise ValueError("v3 failed receipt scope differs")
    capture = capture_evidence(output,"finalize")
    plan = json.loads((output/"plan.json").read_text())
    g = load_geometry(project,side="left")
    layout = g["reference"]["reference_layout"]
    geom = json.loads((output/"initial_geometry.json").read_text())
    init = validate_pinch_initial_geometry(geom,g,table_center_m=layout["table_center"],
        table_size_m=layout["table_size"],rod_nodes_w=layout["wire_points"],rod_radius_m=.005,
        minimum_table_gap_m=0.,minimum_rod_gap_m=0.)
    if not (init["valid"] and init["initial_geometry_passed"]):
        raise ValueError("independent initial geometry replay failed")
    rows = list(ledger(output/"physics.jsonl"))
    cfg = json.loads((project/"configs/dh116_tasks/flexible_pinch_candidate_v1.json").read_text())
    cfg["initial_pose"] = json.loads((project/"configs/dh116_tasks/flexible_pinch_pose_candidate_v2.json").read_text())
    validate_action_targets(rows,plan,cfg)
    pair_counts, precision = Counter(), []
    hand_rows = both = table_count = 0
    selected = {}
    for i,(row,pair,contact) in enumerate(zip(rows,ledger(output/"pairs.jsonl"),
            ledger(output/"table_contacts.jsonl"),strict=True),1):
        if (row["phase_step"],row["sdk_step"],row["rod_tick"]) != (i,i+2,i*12):
            raise ValueError("wrong physical clocks")
        require_quiet(pair,contact,row)
        balance = residual_precision(row["rod_node_contact_forces_w"],
            [row["rod_mesh_wrenches_w"][n] for n in NAMES],row["pair_force_residual_n"])
        if not balance["matches_producer"] or not balance["independent_balance_passed"]:
            raise ValueError(f"source residual or original balance threshold failed at row {i}")
        precision.append(balance)
        pair_counts.update(s["mesh_name"] for s in pair["pairs"])
        loaded = {s["mesh_name"] for s in pair["pairs"] if s["mesh_index"]<5
                  and s["normal_force_on_rod_n"]>1e-5}
        hand_rows += bool(loaded)
        both += set(REQUIRED)<=loaded
        table_count += contact["total_contacts"]
        if i in (1,48,49,128,208,238):
            state = row["post_state"]
            selected[i] = {}
            for name in REQUIRED:
                j = DISTAL_PAD_LINKS.index(name)
                xform = pose_matrix(np.r_[state["link_pos"][j],state["link_quat"][j]])
                vertices = transform(g["cooked_pads"][name]["vertices"],xform)
                selected[i][name] = capsule_gap(vertices,row["rod_nodes_w"],.005)["capsule_surface_gap_m"]
    if len(rows)!=238:
        raise ValueError("238 complete v3 rows required")
    return dict(version="dh116-pinch-v3-evidence-replay-cpu-v1",
        receipt_sha256=RECEIPT_SHA,raw_ledger_sha256=LEDGER_SHA,
        outer_exit=outer,finalization=capture,observed_physics_steps=240,
        observed_measurement_rows=len(rows),positive_hand_pair_rows=hand_rows,
        both_required_loaded_rows=both,pair_counts_by_mesh=dict(pair_counts),
        physx_hand_table_contacts=table_count,measured_pose_locked_hull_gap_m=selected,
        initial_geometry=init,
        legacy_precision_mismatch_rows=sum(not r["matches_legacy_fp64"] for r in precision),
        max_fp64_force_residual_n=max(r["independent_fp64_n"] for r in precision),
        all_source_fp32_reproduced=True,new_native_steps=0,new_training_updates=0,
        native_qualification=False,teacher_admission=False,contact_qualification=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=replay(Path(__file__).resolve().parents[1],args.receipt)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:report[k] for k in ("observed_measurement_rows","positive_hand_pair_rows",
        "legacy_precision_mismatch_rows","max_fp64_force_residual_n","measured_pose_locked_hull_gap_m")}))


if __name__=="__main__":
    main()
