"""CPU screening of an opt-in open-hand Reset at the original 902 final wrist.

This is neither a native entry point nor a force/contact qualification.
The original scene, cameras, target, physical guards and action rates stay fixed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from go2_pvcnn.control.dh116.linear_coupling import expand
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from scripts.screen_dh116_tripod_supported_reset import archived_tube, separation
from scripts.certify_dh116_tripod_distal_route import support_clearance_lower

SOURCE_LINKS = ("base_link", *(f"finger{i}{j}_link" for i in range(1, 6)
                             for j in range(1, 5 if i == 1 else 4)))
PADS = ("finger14_link", "finger23_link", "finger33_link", "finger43_link", "finger53_link")
REQUIRED_SHAPES = {"source:" + n for n in SOURCE_LINKS} | {"cooked:" + n for n in PADS}
VERSION = "dh116-902-fixed-wrist-open-cpu-v1"


def compile_closure(initial, target, *, windows):
    q=np.asarray(initial,dtype=float);goal=np.asarray(target,dtype=float)
    expand(q);expand(goal)
    if type(windows) is not int or not 1<=windows<=120:
        raise ValueError("explicit bounded closure windows required")
    moving=goal!=q
    q=q.astype(np.float32).astype(float);expand(q)
    # Unchanged source joints retain their actual float32 target exactly.
    goal=np.where(moving,goal,q)
    start=q.copy();scale=ACTION_CONTRACT["joint_rate_rad_s"]*8/480
    nominal=(goal-start)/windows/scale
    if np.max(abs(nominal))>1.:
        raise ValueError("closure exceeds original joint rate; clipping forbidden")
    actions=np.zeros((windows,24));states=[]
    for i in range(windows):
        desired=start+(goal-start)*(i+1)/windows;expand(desired)
        a=np.zeros(6);a[moving]=(desired[moving]-q[moving])/scale
        if np.max(abs(a))>1. or np.max(abs(a-nominal))*scale>1e-6:
            raise ValueError("closure exceeds rate or float32 correction budget")
        actions[i,6:12]=a
        for _ in range(8):
            q=(q+a*scale/8).astype(np.float32).astype(float);expand(q)
            states.append(q.copy())
    if np.max(abs(q-goal))>1e-6:
        raise ValueError("original closure endpoint not reproduced")
    return actions,np.asarray(states)


def reset_geometry(shapes, nodes, boxes, *, table_top):
    nodes=np.asarray(nodes,dtype=float)
    if (set(shapes)!=REQUIRED_SHAPES or nodes.shape!=(239,41,3)
            or not np.isfinite(nodes).all() or len(boxes)!=2 or not np.isfinite(table_top)):
        raise ValueError("all 22 shapes, initial plus SDK3..240 rod frames and two supports required")
    for vertices in shapes.values():
        v=np.asarray(vertices,dtype=float)
        if v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all():
            raise ValueError("complete finite convex geometry required")
    for lo,hi in boxes:
        if (np.asarray(lo).shape!=(3,) or np.asarray(hi).shape!=(3,)
                or not np.isfinite([lo,hi]).all() or not (np.asarray(lo)<hi).all()):
            raise ValueError("finite ordered support box required")
    tube=archived_tube(nodes)
    gaps={name:separation(v,*tube) for name,v in shapes.items()}
    support=min(support_clearance_lower(v,lo,hi) for v in shapes.values() for lo,hi in boxes)
    table=min(float(np.asarray(v)[:,2].min()-table_top) for v in shapes.values())
    limiting=min(gaps,key=gaps.get)
    return dict(rod_gap_lower_m=gaps[limiting],limiting_shape=limiting,
        support_gap_lower_m=support,table_gap_m=table,gaps=gaps,shape_count=len(shapes),frame_count=len(nodes),
        passed=bool(gaps[limiting]>.001 and support>.001 and table>=.0021),
        native_reset_verified=False,teacher_admission=False)



ORIGINAL_REQUEST = "configs/dh116_tasks/flexible_902_tripod_right_command_contact_v2.json"
ORIGINAL_REQUEST_SHA256 = "871e37907badbd4a7b534ac4e458fe42a43e80dbf15216e87e1746c7873827b0"


def checked_source_request(candidate, root):
    if (candidate.get("source_request")!=ORIGINAL_REQUEST
            or candidate.get("source_request_sha256")!=ORIGINAL_REQUEST_SHA256):
        raise ValueError("candidate cannot redefine the original 902 endpoint source")
    source=Path(root)/ORIGINAL_REQUEST
    if hashlib.sha256(source.read_bytes()).hexdigest()!=ORIGINAL_REQUEST_SHA256:
        raise ValueError("original 902 endpoint source bytes differ")
    return source,json.loads(source.read_bytes())


def endpoint_contact_candidate(row):
    """Retain the old geometric endpoint gate; never certify measured force."""
    try:
        gaps=row["pad_gaps_m"];ids=row["node_ids"];opposition=row["opposition"]
        return bool(len(gaps)==3 and len(ids)==3 and len(opposition)==2
            and np.isfinite(gaps).all() and np.isfinite(opposition).all()
            and min(gaps)>=-.00015 and max(gaps)<0. and max(opposition)<-.2
            and all(type(k) is int and 0<=k<20 for k in ids))
    except (KeyError,TypeError,ValueError):
        return False


def screen(candidate_path, receipt):
    from scripts.verify_dh116_902_flexible_reset import load_candidate,ROOT
    from scripts.audit_dh116_pinch_initial_pose import load_geometry
    from scripts.audit_dh116_wire_contact_witness import shapes_world
    from scripts.screen_dh116_902_command_contact import _measure_hand
    from go2_pvcnn.control.dh116.tabletop_supported_rod_reference import load_parked_reference
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    start=time.monotonic()
    candidate_path=candidate_path.resolve(strict=True)
    candidate_path.relative_to(ROOT)
    candidate=json.loads(candidate_path.read_bytes())
    if (candidate.get("version")!=VERSION or candidate.get("grasp")!="tripod"
            or candidate.get("side")!="right" or candidate.get("native_admissible") is not False):
        raise ValueError("independent CPU-only Tripod/right request required")
    source,request=checked_source_request(candidate,ROOT)
    if (request.get("grasp")!="tripod" or request.get("contact_side")!="right"
            or request.get("max_penetration_m")!=.00015 or request.get("support_reserve_m")!=.001):
        raise ValueError("original Tripod side and physical geometry thresholds required")
    ref,config,files=load_candidate("tripod")
    archive=[load_parked_reference(receipt,candidate["settling_outer_sha256"],tick) for tick in range(3,241)]
    if any(list(row.support_boxes)!=config["support_scene"]["supports"] for row in archive):
        raise ValueError("archive and current supports differ")
    nodes=np.stack([config["rod_nodes_w_m"],*[row.node_positions for row in archive]])
    boxes=[(np.asarray(b["center_m"])-np.asarray(b["size_m"])/2,
            np.asarray(b["center_m"])+np.asarray(b["size_m"])/2) for b in archive[0].support_boxes]
    table=ref["table"];top=table["center_m"][2]+table["size_m"][2]/2
    right_pose=np.asarray(request["target"]["pose_wxyz"])
    if right_pose[1]>=0 or config["hands"]["left"]["pose_wxyz"][1]<=0:
        raise ValueError("original side identities required")
    models={s:load_geometry(ROOT,side=s) for s in ("left","right")}
    if any(models[s]["usd"]["sha256"]!=ref["hand_usd_sha256"][s] for s in models):
        raise ValueError("original side-specific USD required")
    goal=np.asarray(request["target"]["active_q_rad"])
    opening=np.asarray(candidate["opening_active_q_rad"])
    expand(opening)
    if not np.array_equal(opening[4:],goal[4:]):
        raise ValueError("non-grasp fingers must preserve original 902 targets")
    poses={"left":config["hands"]["left"]["pose_wxyz"],"right":right_pose}
    active={"left":config["hands"]["left"]["q_rad"],"right":opening.astype(np.float32).astype(float)}
    initial={s:reset_geometry(shapes_world(models[s],poses[s],active[s]),nodes,boxes,table_top=top) for s in models}
    actions,states=compile_closure(opening,goal,windows=candidate["closure_windows"])
    grids=scene.load_sdf_arrays(scene.prepare_scene(ROOT))
    tail=nodes[-9:];tube=archived_tube(tail);samples=[]
    for i,q in enumerate(states):
        if time.monotonic()-start>180:
            raise RuntimeError("bounded CPU screening exceeded 180 seconds")
        row=_measure_hand("right",(models["right"],{}),right_pose,q,tail,tube,boxes,grids,top,request)
        row["ordinal"]=i+1
        row["passed"]=bool(row["noncontact_rod_clearance_m"]>.00025
            and row["support_clearance_m"]>=.001 and row["table_clearance_m"]>=.0021
            and row["contact_rod_clearance_m"]>=-.00015 and row["all_envelopes_certified"]
            and min(row["pad_gaps_m"])>=-.00015)
        samples.append(row)
    from scripts.screen_dh116_902_command_contact import dependency_files
    dependencies=[Path(__file__),source,candidate_path,*files,*dependency_files(),
                  Path(__file__).with_name("screen_dh116_tripod_supported_reset.py"),
                  Path(__file__).with_name("audit_dh116_wire_contact_witness.py"),
                  Path(__file__).with_name("certify_dh116_tripod_distal_route.py")]
    dependencies=list(dict.fromkeys(path.resolve(strict=True) for path in dependencies))
    return dict(version=VERSION,scope="ideal_fixed_wrist_reset_and_closure_only",initial=initial,
        initial_passed=all(row["passed"] for row in initial.values()),
        closure_geometry_passed=all(row["passed"] for row in samples),
        endpoint_contact_candidate_passed=endpoint_contact_candidate(samples[-1]),
        closure_passed=all(row["passed"] for row in samples) and endpoint_contact_candidate(samples[-1]),
        samples=samples,
        physical_substeps=len(states),control_windows=len(actions),
        action_sha256=hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(),
        endpoint_q_error_rad=float(np.max(abs(states[-1]-goal))),
        right_wrist_pose_wxyz=right_pose.tolist(),opening_active_q_rad=active["right"].tolist(),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in dependencies},
        geometry_sources={s:models[s]["source_sha256"] for s in models},
        archive_sha256=[row.sample_sha256 for row in archive],settling_outer_sha256=sha(receipt),
        elapsed_seconds=time.monotonic()-start,native_runs=0,ppo_updates=0,
        native_admissible=False,teacher_admission=False,
        limitations=["archived rod does not predict contact deformation",
                     "no native Reset, wrist settling, opposed force or reaction qualification",
                     "inter-hand and self-collision are not certified by this screen"])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--settling-receipt",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    if a.output.exists():raise ValueError("fresh evidence output required")
    result=screen(a.candidate,a.settling_receipt)
    with a.output.open("x") as f:json.dump(result,f,indent=2,allow_nan=False)
    return int(not(result["initial_passed"] and result["closure_passed"]))


if __name__=="__main__":
    raise SystemExit(main())
