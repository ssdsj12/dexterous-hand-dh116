"""Bounded CPU search near the original 902 Tripod endpoint; no native actions."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scripts.screen_dh116_902_fixed_wrist_open import (
    ORIGINAL_REQUEST, ORIGINAL_REQUEST_SHA256, endpoint_contact_candidate)
from go2_pvcnn.control.dh116.linear_coupling import expand

ROOT=Path(__file__).resolve().parents[1]
VERSION="dh116-902-joint-contact-cpu-v1"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def decode_candidate(x,target):
    x=np.asarray(x,dtype=float)
    if x.shape!=(9,) or not np.isfinite(x).all() or np.max(abs(x))>1:
        raise ValueError("finite bounded search coordinates required")
    pose=np.asarray(target["pose_wxyz"],dtype=float).copy()
    q=np.asarray(target["active_q_rad"],dtype=float).copy()
    pose[:3]+=.0005*x[:3]/np.sqrt(1+float(x[:3]@x[:3]))
    delta=.015*x[3:6]/np.sqrt(1+float(x[3:6]@x[3:6]))
    rot=Rotation.from_rotvec(delta)*Rotation.from_quat(pose[[4,5,6,3]])
    pose[3:]=rot.as_quat()[[3,0,1,2]]
    q[[0,2,3]]+=.02*x[6:];expand(q)
    if pose[1]>=0:raise ValueError("right-hand side identity changed")
    return pose,q


def candidate_passed(row):
    try:
        metrics=[row[k] for k in ("noncontact_rod_clearance_m","support_clearance_m",
            "table_clearance_m","contact_rod_clearance_m")]
        return bool(row["shape_count"]==22 and np.isfinite(metrics).all()
            and metrics[0]>.00025 and metrics[1]>=.001 and metrics[2]>=.0021
            and metrics[3]>=-.00015 and row["all_envelopes_certified"] is True
            and endpoint_contact_candidate(row))
    except (KeyError,ValueError,TypeError):return False


def search(receipt,expected_sha,*,starts=8,nfev=48,wall_seconds=180):
    if (type(starts) is not int or not 1<=starts<=8 or type(nfev) is not int or not 1<=nfev<=48
            or not 1<=wall_seconds<=180):
        raise ValueError("explicit bounded CPU search budget required")
    from go2_pvcnn.control.dh116.tabletop_supported_rod_reference import load_parked_reference
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from scripts.audit_dh116_pinch_initial_pose import load_geometry
    from scripts.prepare_dh116_pinch_closure_margin import fk_poses,sample_grid
    from scripts.screen_dh116_902_command_contact import _measure_hand,PADS
    from scripts.screen_dh116_tripod_supported_reset import archived_tube
    from scripts.verify_dh116_902_flexible_reset import REFERENCE
    source=ROOT/ORIGINAL_REQUEST
    if sha(source)!=ORIGINAL_REQUEST_SHA256:raise ValueError("original 902 endpoint source changed")
    request=json.loads(source.read_bytes())
    receipt=Path(receipt).resolve(strict=True)
    references=[load_parked_reference(receipt,expected_sha,sdk,task="tripod") for sdk in range(232,241)]
    nodes=np.stack([r.node_positions for r in references])
    model=load_geometry(ROOT,side="right");prepared=scene.prepare_scene(ROOT);grids=scene.load_sdf_arrays(prepared)
    boxes=[(np.asarray(b["center_m"])-np.asarray(b["size_m"])/2,
            np.asarray(b["center_m"])+np.asarray(b["size_m"])/2) for b in references[0].support_boxes]
    table=json.loads(REFERENCE.read_bytes())["table"];top=table["center_m"][2]+table["size_m"][2]/2
    planes={pad:ConvexHull(model["source_colliders"][pad]["vertices"]).equations for pad in PADS}
    began=time.monotonic();calls=0;rows=[];budget_exhausted=False
    def residual(x,source_weight):
        nonlocal calls
        calls+=1
        if time.monotonic()-began>wall_seconds:raise TimeoutError("bounded CPU search expired")
        pose,q=decode_candidate(x,request["target"]);fk=fk_poses(model,pose,q);gaps=[];source_gaps=[]
        for pad in PADS:
            t=fk[pad];local=(nodes[-1]-t[:3,3])@t[:3,:3]
            phi,_,valid=sample_grid(grids["right."+pad],local,dtype=np.float64)
            if not valid.any():return np.full(15,1e3)
            gaps.append(float(np.nanmin(phi)-.005))
            # Supporting-plane lower estimate guides candidates only; exact continuous
            # source/cooked guard is mandatory below and cannot be bypassed by this cost.
            eq=planes[pad];plane_distance=(local@eq[:,:3].T+eq[:,3]).max(axis=1)
            source_gaps.append(float(plane_distance.min()-.005))
        return np.r_[(np.asarray(gaps)+.00002)/.0001,
            source_weight*np.minimum(np.asarray(source_gaps)+.00012,0)/.0001,.01*x]
    seeds=np.random.default_rng(902).uniform(-.8,.8,(starts,9));seeds[0]=0.
    try:
        for i,seed in enumerate(seeds):
            fit=least_squares(lambda x:residual(x,0. if i%2==0 else 3.),seed,
                bounds=(-1,1),max_nfev=nfev,diff_step=1e-4,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            pose,q=decode_candidate(fit.x,request["target"])
            measured=_measure_hand("right",(model,{}),pose,q,nodes,archived_tube(nodes),boxes,grids,top,request)
            rows.append(dict(ordinal=i,x=fit.x.tolist(),pose_wxyz=pose.tolist(),active_q_rad=q.tolist(),
                optimizer_cost=float(fit.cost),optimizer_nfev=fit.nfev,measurement=measured,
                endpoint_candidate_passed=candidate_passed(measured)))
    except TimeoutError:budget_exhausted=True
    accepted=[row for row in rows if row["endpoint_candidate_passed"]]
    return dict(version=VERSION,scope="CPU_endpoint_only_not_command_path_or_measured_contact",
        source_request_sha256=sha(source),receipt_sha256=expected_sha,
        rod_frame_sha256=[r.sample_sha256 for r in references],sdk_range=[232,240],
        geometry_sources=model["source_sha256"],source_sha256=sha(__file__),
        bounds=dict(translation_norm_m=.0005,rotation_norm_rad=.015,grasp_joint_delta_rad=.02),
        budget=dict(starts=starts,max_nfev_per_start=nfev,wall_seconds=wall_seconds),
        completed_starts=len(rows),residual_calls=calls,budget_exhausted=budget_exhausted,
        elapsed_seconds=time.monotonic()-began,rows=rows,accepted_count=len(accepted),
        selected=None if not accepted else min(accepted,key=lambda row:np.linalg.norm(row["x"])),
        command_path_qualified=False,native_contact_verified=False,native_runs=0,ppo_updates=0,
        teacher_admission=False,limitations=["SDF and plane costs only guide candidates",
            "Exact unchanged 22-shape endpoint gates are required",
            "Neither geometric contact nor optimizer convergence proves reaction force"])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt",type=Path,required=True);p.add_argument("--receipt-sha256",required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--starts",type=int,default=8)
    a=p.parse_args(argv)
    if a.output.exists():raise ValueError("fresh search evidence required")
    result=search(a.receipt,a.receipt_sha256,starts=a.starts)
    with a.output.open("x") as f:json.dump(result,f,indent=2,allow_nan=False)
    return int(not result["accepted_count"])


if __name__=="__main__":raise SystemExit(main())
