#!/usr/bin/env python3
"""Recheck sixteen archived yaw endpoints with feasible convex-envelope witnesses."""
import argparse,json,signal,time
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scripts import audit_dh116_pinch_static_yaw as yaw
from scripts.audit_dh116_pinch_initial_pose import finite_array,digest,load_geometry,world_vertices,convex_patch_closest
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm

VERSION="dh116-pinch-archived-yaw-convex-witness-v1"
PARENT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-static-yaw-20261113-sqwO6J/run01/result.json")
PARENT_SHA="d434fbd069325dbaf71195a0452717667a43be9c043e7e57601b77ffc1407890"
MAX_WITNESSES=16

def witness(vertices,nodes):
    v=finite_array(vertices);p=finite_array(nodes)
    if v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or p.ndim!=2 or p.shape[1]!=3 or len(p)<2:
        raise ValueError("source convex vertices and actual nondegenerate polyline required")
    edges=np.diff(p,axis=0);length2=np.sum(edges*edges,axis=1)
    if np.any(length2<=1e-20):raise ValueError("degenerate actual rod segment")
    center=v.mean(0)
    fractions=np.clip(np.sum((center-p[:-1])*edges,axis=1)/length2,0,1)
    nearest=p[:-1]+fractions[:,None]*edges
    i=int(np.argmin(np.sum((nearest-center)**2,axis=1)))
    raw=convex_patch_closest(v,p[i],[edges[i]])
    # Convert a numerical QP candidate into a strictly feasible hull/segment pair.
    # Radial contraction toward the interior vertex centroid avoids declaring
    # a slightly outside solver iterate to be a collision witness.
    planes=ConvexHull(v-center).equations
    eps=min(1e-9,float(np.min(-planes[:,3]))/4)
    if eps<=0:raise ValueError("strict convex-envelope interior required")
    point=finite_array(raw["hull_point_w"],(3,))-center
    directions=planes[:,:3]@point
    positive=directions>0
    scale=min(1.,float(np.min((-eps-planes[positive,3])/directions[positive]))) if positive.any() else 1.
    scale=float(np.clip(scale,0,1))
    hull=center+scale*point
    residual=planes[:,:3]@(hull-center)+planes[:,3]
    if float(residual.max())>-eps/2:raise ValueError("strictly feasible hull witness failed")
    axis=finite_array(raw["axis_point_w"],(3,))
    t=float(np.clip(np.dot(axis-p[i],edges[i])/length2[i],0,1))
    axis=p[i]+t*edges[i]
    distance=float(np.linalg.norm(hull-axis))
    gap=distance-.005+2e-6
    return dict(segment_index=i,segment_fraction=t,segment_nodes_m=p[i:i+2].tolist(),
        closest=dict(distance_m=distance,hull_point_w=hull.tolist(),axis_point_w=axis.tolist()),
        qp_candidate=raw,hull_centroid_contraction=scale,max_hull_plane_residual_m=float(residual.max()),
        radius_m=.005,numerical_upper_guard_m=2e-6,gap_upper_m=gap,
        convex_envelope_overlap_witness=bool(gap<0),full_polyline_closest_distance_proven=False,
        geometry_clearance_proven=False,native_contact_proven=False,hardware_geometry_calibrated=False,
        source="convex envelope of supplied source vertices; one actual polyline segment")

def evaluate(project,result):
    old=yaw.previous.checked_json(PARENT,PARENT_SHA)
    if old["status"]!="complete" or old["geometry_state_attempts"]!=102:
        raise ValueError("exact complete archived yaw cohort required")
    result.update(parent_result_path=str(PARENT),parent_result_sha256=PARENT_SHA,
        original_parent_identity=old["original_parent_identity"],records=[],
        new_candidate_states=0,maximum_witness_reconstructions=MAX_WITNESSES,
        source_sha256={},successful_witnesses=0)
    nodes=np.asarray(old["actual_parent_nodes_m"])
    for side in ("left","right"):
        g=load_geometry(project,side=side)
        if g["source_sha256"]!=old["sources"][side]["geometry_sha256"]:
            raise ValueError("independent unchanged side geometry required")
        result["source_sha256"][side]=g["source_sha256"]
        for saved in old["sides"][side]["candidates"]:
            c=saved["candidate"]
            if c["static_reset_yaw_world_rad"]==0:continue
            if len(result["records"])>=MAX_WITNESSES:raise ValueError("sixteen witness budget exhausted")
            hard=[(k,v) for k,v in zip(saved["labels"],saved["values"],strict=True)
                if not k.endswith(":contact_minus0p1mm")]
            label,margin=min(hard,key=lambda x:x[1])
            parts=label.split(":")
            if (len(parts)!=4 or parts[0]!="endpoint" or parts[1] not in ("source","cooked")
                or parts[3]!="non_target_design_0p2mm" or margin>=0):
                raise ValueError("archived worst constraint is not the expected failed shape envelope")
            q=np.asarray(c["active_q_rad"])+np.asarray(c["active_delta_rad"])
            if not np.array_equal(q,np.asarray(saved["endpoint"]["actual_q_rad"])):
                raise ValueError("archived endpoint active q differs")
            row=dict(side=side,static_yaw_world_rad=c["static_reset_yaw_world_rad"],
                shape=parts[1]+":"+parts[2],original_constraint=label,original_margin_mm=margin,
                source_lower_m=saved["endpoint"]["all_shape_rod_lower_m"][parts[1]+":"+parts[2]],
                status="started")
            with _commit_row_without_alarm():result["records"].append(row)
            try:
                shapes=world_vertices(g,c["base_pose_wxyz"],q,cooked=parts[1]=="cooked")
                value=witness(shapes[parts[2]],nodes)
                with _commit_row_without_alarm():
                    row.update(status="complete",witness=value)
                    result["successful_witnesses"]+=int(value["convex_envelope_overlap_witness"])
            except BaseException as error:
                with _commit_row_without_alarm():
                    row.update(status="interrupted" if isinstance(error,BudgetExpired) else "error",error=type(error).__name__+": "+str(error))
                raise
    if len(result["records"])!=16:raise ValueError("exact sixteen archived nonzero yaw endpoints required")
    result["status"]="complete"

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result=dict(version=VERSION,status="partial_setup",physics_steps=0,training_updates=0,
        native_qualification=False,teacher_admission=False,reset_proposal_created=False)
    def timeout(*unused):raise BudgetExpired()
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,30)
    try:evaluate(Path(__file__).resolve().parents[1],result)
    except BudgetExpired:result["status"]="partial_timeout"
    except Exception as error:result.update(status="diagnostic_error",error=type(error).__name__+": "+str(error))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    result["elapsed_s"]=time.monotonic()-start
    with (a.output/"result.json").open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({k:result.get(k) for k in ("status","elapsed_s","successful_witnesses","error")}))
    return int(result["status"]=="diagnostic_error")

if __name__=="__main__":raise SystemExit(main())
