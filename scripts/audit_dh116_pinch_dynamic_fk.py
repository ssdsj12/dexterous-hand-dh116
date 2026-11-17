#!/usr/bin/env python3
"""Read-only nonzero-action FK/tracking separation on a pinned native ledger."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts import replay_dh116_native_parameter_r0 as evidence
from scripts.audit_dh116_pinch_initial_pose import load_geometry,pose_matrix,digest
from go2_pvcnn.control.dh116.linear_coupling import JOINTS,ACTIVE,MIMIC,expand
from go2_pvcnn.control.dh116.tabletop_measured_closure_geometry import MeasuredClosureGeometry,REQUIRED
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,encode_owned

VERSION="dh116-pinch-dynamic-fk-readback-v1"

def rigid(value):
    t=np.array(value,dtype=float,copy=True)
    if (t.shape!=(4,4) or not np.isfinite(t).all() or
            not np.allclose(t[3],[0,0,0,1],atol=1e-12,rtol=0) or
            not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-10,rtol=0) or
            not np.isclose(np.linalg.det(t[:3,:3]),1.,atol=1e-10,rtol=0)):
        raise ValueError("finite proper column-vector rigid transform required")
    return t

def full_named_poses(fk,root,angles):
    """Reuse original USD joint frames, with ALL measured joints verbatim.

    No active expansion, clamps, reauthoring or model fit. The explicit root is
    the actual base_link pose; the six mount DOFs are outside this hand tree.
    """
    if set(angles)!=set(JOINTS) or not np.isfinite(list(angles.values())).all():
        raise ValueError("eleven named finite actual joint angles required")
    poses={fk.root:rigid(root)}
    for name,parent,child,t0,t1,axis in fk.edges:
        if parent not in poses or child in poses:raise ValueError("invalid source hand tree")
        motion=np.eye(4)
        if axis is not None:
            axis=np.asarray(axis,dtype=float)
            if name not in angles or axis.shape!=(3,) or not np.isclose(np.linalg.norm(axis),1.,atol=1e-12):
                raise ValueError("named source joint with unit axis required")
            motion[:3,:3]=Rotation.from_rotvec(axis*float(angles[name])).as_matrix()
        poses[child]=poses[parent]@rigid(t0)@motion@rigid(t1)
    named={k.rsplit("/",1)[-1]:v for k,v in poses.items()}
    if len(named)!=len(poses):raise ValueError("ambiguous source link names")
    return named

def pose_error(predicted,actual):
    p,a=rigid(predicted),rigid(actual)
    delta=p[:3,3]-a[:3,3]
    return dict(position_delta_m=delta.tolist(),position_m=float(np.linalg.norm(delta)),
                rotation_rad=float(Rotation.from_matrix(p[:3,:3]@a[:3,:3].T).magnitude()))

def statistics(values):
    a=np.asarray(values,dtype=float)
    if a.ndim!=1 or not np.isfinite(a).all():raise ValueError("finite scalar samples required")
    if not len(a):return None
    return dict(count=len(a),p50=float(np.percentile(a,50)),p90=float(np.percentile(a,90)),
                p95=float(np.percentile(a,95)),max=float(a.max()))

def evaluate_exchange(exchange,declared_action,geometry,models):
    packet=geometry.validate(exchange["closure_geometry"],exchange)
    radius=evidence.check_radius_packet(exchange["material_director"],exchange["rod"])
    raw=packet["raw"];identity=raw["identity"];a=exchange["action_readback"]
    declared=np.asarray(declared_action)
    actual_action=np.asarray(a["request"]["action24"])
    if (declared.shape!=(1,24) or not np.isfinite(declared).all() or
            np.any(abs(declared)>1) or not np.array_equal(declared,actual_action)):
        raise ValueError("canonical request differs from same control's recorded action")
    hands={}
    for side in ("left","right"):
        g=models[side]
        if g["source_sha256"]!=geometry.sources[side]["source_sha256"]:
            raise ValueError("independent original side-specific geometry source differs")
        h=a["post"]["hands"][side];names=h["joint_names"];q=np.asarray(h["q"])[0];qd=np.asarray(h["qd"])[0]
        targets=np.asarray(h["backend_targets"])[0]
        submitted=a["submission"]["expected_backend_targets"][side]
        ids=submitted["joint_indices"]
        if (len(ids)!=12 or len(set(ids))!=12 or any(type(i) is not int or not 0<=i<17 for i in ids)
                or not np.array_equal(targets[ids],np.asarray(submitted["values"])[0])):
            raise ValueError("SDK backend targets differ from actual submission")
        body=raw["hands"][side];poses=np.asarray(body["link_poses_xyzw"])[0]
        actual={n:pose_matrix(np.r_[p[:3],p[6],p[3:6]]) for n,p in zip(body["body_names"],poses)}
        base=actual["base_link"]
        full=dict(zip(JOINTS,[float(q[names.index(n)]) for n in JOINTS]))
        active=np.array([q[names.index(n)] for n in ACTIVE])
        command=np.array([targets[names.index(n)] for n in ACTIVE])
        variants={
            "actual_full11":full_named_poses(g["fk"],base,full),
            "actual_active_mimic":full_named_poses(g["fk"],base,dict(zip(JOINTS,expand(active)))),
            "sdk_active_target_mimic":full_named_poses(g["fk"],base,dict(zip(JOINTS,expand(command))))}
        root_delta=base@np.linalg.inv(g["fk"].root_pose)
        original={k.rsplit("/",1)[-1]:root_delta@v for k,v in g["fk"].poses(active).items()}
        if set(original)!=set(variants["actual_active_mimic"]) or any(
                not np.allclose(v,variants["actual_active_mimic"][n],rtol=0,atol=1e-12) for n,v in original.items()):
            raise ValueError("diagnostic full tree does not reproduce original coupled FK")
        output={}
        for name,predicted in variants.items():
            if set(predicted)!=set(g["source_colliders"]):
                raise ValueError("source FK and original 17-link geometry coverage differ")
            output[name]=dict(
                link_errors={n:pose_error(p,actual[n]) for n,p in predicted.items()},
                required_pad_sdf={n:geometry._sdf(geometry._grids[side][n],predicted[n],raw["rod_nodes_m"],
                                                radius=radius["value"]) for n in REQUIRED})
        hands[side]=dict(joint_names=list(names),q_actual_si=q.tolist(),qd_actual_si_per_s=qd.tolist(),
            sdk_backend_targets=targets.tolist(),sdk_targets_are_measurements=False,
            joint_units_by_name=deepcopy(h["units_by_name"]),active_order=list(ACTIVE),
            active_target_minus_actual_rad=(command-active).tolist(),
            mimic_residual_rad={f:float(q[names.index(f)]-ratio*q[names.index(m)]-offset)
                                for f,m,ratio,offset in MIMIC},
            actual_base_pose_xyzw=poses[body["body_names"].index("base_link")].tolist(),
            models=output,actual_required_pad_sdf={n:geometry._sdf(geometry._grids[side][n],actual[n],
                raw["rod_nodes_m"],radius=radius["value"]) for n in REQUIRED},
            contact_inferred_from_model=False)
    pairs=exchange["rod"]["rod_mesh_contact_pairs"]["pairs"]
    return dict(identity=deepcopy(identity),action24=declared[0].tolist(),
        issued_action24=actual_action[0].tolist(),
        action_semantics="accepted canonical intent; actual SDK targets and measured q kept separately",
        hands=hands,raw_hand_rod_pair_samples=sum(p["mesh_name"].startswith(("left.","right.")) for p in pairs))

def run(project=ROOT):
    project=Path(project).resolve()
    receipt=json.loads(evidence.read_pinned(evidence.RECEIPT,evidence.RECEIPT_SHA))
    if (receipt["outer_exit_code"]!=0 or receipt["debugger_inferior_signal"] is not None
            or receipt["physics_clock"]["total_steps"]!=18):
        raise ValueError("exact completed historical parameter-task receipt required")
    raw=evidence.read_pinned(Path(receipt["output"])/"physical.jsonl",receipt["artifact_sha256"]["physical.jsonl"])
    ledger=[decode_owned(json.loads(l)) for l in raw.splitlines()]
    geometry=MeasuredClosureGeometry(project)
    models={s:load_geometry(project,side=s) for s in ("left","right")}
    rows=[];declared=None;current_policy=None
    for item in ledger:
        operation=item.get("operation");payload=item.get("payload")
        if operation=="begin":
            if declared is not None:raise ValueError("nested control boundary")
            declared=deepcopy(payload["action"])
            current_policy=payload["pre_action_state"]["common"]["identity"]["policy_step"]
        elif operation=="physical":
            if declared is None:raise ValueError("physical record outside declared control")
            exchange=payload["exchange"]
            if exchange["policy_step"]!=current_policy:raise ValueError("foreign policy action identity")
            rows.append(evaluate_exchange(exchange,declared,geometry,models))
        elif operation=="end":
            if declared is None:raise ValueError("end without begin")
            declared=None
    if declared is not None or len(rows)!=16 or [r["identity"]["physics_step"] for r in rows]!=list(range(3,19)):
        raise ValueError("exact complete two-control archived trajectory required")
    summary={}
    for side in ("left","right"):
        summary[side]={}
        for model in rows[0]["hands"][side]["models"]:
            samples=[(r,n,v) for r in rows for n,v in r["hands"][side]["models"][model]["link_errors"].items()]
            stats={k:statistics([v[k] for _,_,v in samples]) for k in ("position_m","rotation_rad")}
            for k in stats:
                r,n,v=max(samples,key=lambda t:t[2][k])
                stats[k]["peak_identity"]=deepcopy(r["identity"]);stats[k]["peak_link"]=n
            summary[side][model]=stats
    return dict(version=VERSION,scope="offline_archived_nonzero_action_kinematics_not_new_native",
        receipt_sha256=evidence.RECEIPT_SHA,ledger_sha256=receipt["artifact_sha256"]["physical.jsonl"],
        source_task_scope=receipt["qualification_scope"],
        geometry_sources=deepcopy(geometry.sources),
        implementation_sha256=digest(Path(__file__)),
        radius_source="actual native parameter snapshot, checked at every endpoint",
        rows=rows,summary=summary,model_is_not_actual_contact=True,
        new_physics_steps=0,optimizer_updates=0,native_qualification=False,teacher_admission=False)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True,type=Path)
    a=p.parse_args(argv)
    if a.output.exists() or a.output.is_symlink():p.error("preserve existing output")
    result=run()
    forbidden=[m for m in sys.modules if m.split(".")[0] in ("omni","isaacsim","isaaclab","ropesim","warp")]
    if forbidden:raise RuntimeError("offline import boundary crossed")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x") as stream:
        json.dump(encode_owned(dict(result,native_imports=forbidden)),stream,indent=2,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(a.output),rows=len(result["rows"]),summary=result["summary"],
                          new_physics_steps=0,teacher_admission=False)))
    return 0

if __name__=="__main__":raise SystemExit(main())
