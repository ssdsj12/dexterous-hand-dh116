"""Read-only diagnosis of a sealed 902 native contact guard; never admits a grasp."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.audit_dh116_certified_segment_distance import segment_hull_bounds
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def geometry_bounds(vertices,nodes):
    v=np.asarray(vertices,dtype=float);n=np.asarray(nodes,dtype=float)
    if (v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or n.shape!=(41,3)
            or not np.isfinite(v).all() or not np.isfinite(n).all()):
        raise ValueError("finite full source hull and all 41 rod nodes required")
    def nearest(pairs):
        # AABB separation certifies that distant nodes/segments cannot win.
        # Do not feed numerically ill-scaled far points to the local QP.
        low,high=v.min(0),v.max(0)
        ranked=sorted((max(0.,float(np.linalg.norm(np.maximum(
            np.maximum(low-np.maximum(a,b),np.minimum(a,b)-high),0)))-1e-12),i,a,b)
            for i,(a,b) in enumerate(pairs))
        lowers=[];winner=None;calls=0
        for lower,i,a,b in ranked:
            if winner is not None and lower>=winner["upper_m"]:
                lowers.append(lower);continue
            result=dict(index=i,**segment_hull_bounds(v,a,b));calls+=1
            lowers.append(result["lower_m"])
            if winner is None or result["upper_m"]<winner["upper_m"]:winner=result
        return min(lowers),winner,calls
    sl,sw,sc=nearest(list(zip(n[:-1],n[1:])))
    nl,pw,nc=nearest([(a,a) for a in n])
    sl-=.005;su=sw["upper_m"]-.005;nl-=.005;nu=pw["upper_m"]-.005
    return dict(segment_qp_calls=sc,node_qp_calls=nc,segment_min_lower_m=sl,segment_min_upper_m=su,segment_witness=sw,
        node_min_lower_m=nl,node_min_upper_m=nu,node_witness=pw,
        inter_node_collision_without_node_contact=bool(su<-.00015 and nl>0.),
        certified_contact_limit_violation=bool(su<-.00015),
        original_contact_limit_m=.00015,original_radius_m=.005)

def joint_tracking(hand):
    """Separate commanded active joints from passive mimic joints; no zero target fiction."""
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,MIMIC
    names=hand["joint_names"];indices=hand["active_indices"]
    q=np.asarray(hand["q"],dtype=float);qd=np.asarray(hand["qd"],dtype=float)
    target=np.asarray(hand["backend_targets"],dtype=float)
    if (len(set(names))!=len(names) or not set(JOINTS)<=set(names)
            or [names[i] for i in indices]!=list(ACTIVE)
            or q.shape!=(1,len(names)) or qd.shape!=q.shape or target.shape!=q.shape
            or not all(np.isfinite(a).all() for a in (q,qd,target))):
        raise ValueError("finite named real joint q/qd/target readback required")
    rows={}
    for i,name in enumerate(names):
        if name in ACTIVE or name.startswith("wrist_"):
            rows[name]=dict(position=float(q[0,i]),target=float(target[0,i]),
                tracking_error=float(q[0,i]-target[0,i]),velocity=float(qd[0,i]),
                position_unit="m" if name in ("wrist_x","wrist_y","wrist_z") else "rad")
    mimic={f:float(q[0,names.index(f)]-ratio*q[0,names.index(master)]-offset)
           for f,master,ratio,offset in MIMIC}
    return dict(commanded_joint_rows=rows,passive_mimic_residual_rad=mimic,
                passive_backend_zero_not_a_command=True)

def audit(receipt,guard_name):
    receipt=Path(receipt)
    if receipt.is_symlink():raise ValueError("sealed original receipt required")
    outer=json.loads(receipt.read_bytes())
    if (type(outer.get("exit_code")) is not int or outer["exit_code"]!=1
            or outer.get("forced_exit") is not False or outer.get("exceeded_budget") is not False
            or outer.get("debugger_status_verified") is not True
            or outer.get("debugger_inferior_signal") is not None
            or outer.get("debugger_inferior_exit_code")!=1):
        raise ValueError("naturally exited, unforced failed diagnostic receipt required")
    root=receipt.parent/"native"
    def read(name):
        if Path(name).name!=name:raise ValueError("single artifact filename required")
        f=root/name
        if f.is_symlink() or not f.is_file() or sha(f)!=outer["artifact_sha256"].get(name):
            raise ValueError("sealed native artifact SHA differs: "+name)
        return decode_owned(json.loads(f.read_bytes()))
    guard=read(guard_name);sdk=guard["observed_sdk_step"];rod_sdk=guard["rod_source_physics_step"]
    if type(sdk) is not int or rod_sdk!=sdk-1 or guard["current_sdk_clock"]["sdk_step"]!=sdk:
        raise ValueError("post-SDK / preceding-rod clock identity required")
    task=guard_name.split("_")[0]
    if task not in ("pinch","tripod"):raise ValueError("known 902 task required")
    previous=read(f"{task}_control_{rod_sdk}.json")
    if (previous["physics_step"]!=rod_sdk or previous["rod"]["physics_step"]!=rod_sdk
            or previous["rod"]["rod_tick"]!=(rod_sdk-2)*12 or previous["guard_failure"] is not None):
        raise ValueError("complete preceding physical exchange required")
    nodes=np.asarray(guard["rod_node_positions_world_m"],dtype=float)
    if nodes.shape!=(1,41,3) or not np.array_equal(nodes,previous["rod"]["node_positions"]):
        raise ValueError("same original preceding rod buffer required")
    plan=read("frozen_plan.json")
    contact=plan["distal_contact"]
    cooked=contact["continuous_cooked_hulls"]
    if cooked != contact["command_proof"]["continuous_cooked_hulls"]:
        raise ValueError("identical frozen proof and runtime cooked hulls required")
    if not isinstance(guard["failures"],list) or not guard["failures"]:
        raise ValueError("nonempty original guard failures required")
    reports=[]
    for item in guard["failures"]:
        side=item["side"];body=item["body_name"]
        if side not in ("left","right") or item["link_prim"]!=f"/World/DeformXDualHand/{side}Hand/{body}":
            raise ValueError("actual side-specific link identity required")
        pose=np.asarray(item["pose_world_wxyz"],dtype=float)
        if pose.shape!=(7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1)>1e-5:
            raise ValueError("finite original SDK pose required")
        local=np.asarray(item["source_vertices_local_m"],dtype=float)
        world=local@Rotation.from_quat(pose[[4,5,6,3]]).as_matrix().T+pose[:3]
        source_bounds=geometry_bounds(world,nodes[0])
        key=side+"."+body
        if key not in cooked:raise ValueError("failed pad requires its own frozen cooked geometry")
        cooked_world=np.asarray(cooked[key],dtype=float)@Rotation.from_quat(pose[[4,5,6,3]]).as_matrix().T+pose[:3]
        cooked_bounds=geometry_bounds(cooked_world,nodes[0])
        reports.append(dict(side=side,body=body,geometry=source_bounds,cooked_geometry=cooked_bounds,
            source_representation="source_USD_convex_envelope_not_live_cooked",
            cooked_representation="frozen_original_F4_cooked_vertices",
            source_only_rejection_not_cooked_contact=bool(
                source_bounds["certified_contact_limit_violation"] and cooked_bounds["segment_min_lower_m"]>0),
            actual_physical_contact_proven=False,original_guard_conditions=item["conditions"]))
    forces={}
    for side,record in previous["rod"]["hand_wrenches_com_world"].items():
        f=np.asarray(record["forces"]);t=np.asarray(record["torques"])
        if not np.isfinite(f).all() or not np.isfinite(t).all():raise ValueError("finite raw hand wrench required")
        forces[side]=dict(maximum_force_n=float(np.linalg.norm(f,axis=-1).max()),
                         maximum_torque_nm=float(np.linalg.norm(t,axis=-1).max()))
    tracking={side:joint_tracking(hand) for side,hand in previous["action_readback"]["post"]["hands"].items()}
    if set(tracking)!={"left","right"}:raise ValueError("both hands must retain real joint readback")
    names=sorted({p["mesh_name"] for p in previous["rod"]["rod_mesh_contact_pairs"]["pairs"]})
    return dict(version="dh116-902-contact-guard-audit-v1",outer_sha256=sha(receipt),
        guard_sha256=sha(root/guard_name),observed_sdk_step=sdk,rod_source_physics_step=rod_sdk,
        source_sha256={str(Path(__file__).name):sha(Path(__file__)),
            "audit_dh116_certified_segment_distance.py":sha(Path(__file__).with_name("audit_dh116_certified_segment_distance.py"))},
        failures=reports,preceding_hand_wrenches=forces,preceding_pair_mesh_names=names,
        tracking_source_sdk=rod_sdk,preceding_joint_tracking=tracking,
        frozen_plan_sha256=sha(root/"frozen_plan.json"),
        native_runs=0,ppo_updates=0,native_grasp_verified=False,teacher_admission=False)

def main(argv=None):
    import argparse
    from scripts.diagnose_dh116_teacher_initialization import write_new
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--guard",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv);write_new(a.output,audit(a.receipt,a.guard))
    return 0
if __name__=="__main__":
    raise SystemExit(main())
