#!/usr/bin/env python3
"""Bounded CPU Pinch proposals targeting original inner-pad contact regions.

Reuse full source/cooked geometry, locked SDF and the existing finite minimax
search. No simulator, modified physical model, force application or native claim.
"""
import argparse,json,os,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts import solve_dh116_right_pinch_surface_registration as surface
from scripts.solve_dh116_tripod_contact_closure import bounded_search,candidate_admission
from scripts import prepare_dh116_bilateral_owned_pinch as parent
from scripts.replay_dh116_bilateral_contact_r0 import load_source
from scripts.replay_dh116_segment_contact_r0 import stream_events
from go2_pvcnn.control.dh116.tabletop_deformable_pad_region import classify_pair,rigid
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned,decode_owned
from go2_pvcnn.control.dh116.linear_coupling import active_limits,expand,ACTIVE

ROOT=Path(__file__).resolve().parents[1]
VERSION="dh116-pinch-effective-target-cpu-v3"
CONFIG=dict(version=VERSION,mode="cpu_only",translation_scales_m=[.002]*3,
    reset_rotation_scales_rad=[.03]*3,joint_scales_rad=[.04]*3,
    table_clearance_m=.0021,native_required_gap_m=-.0003,
    reserve_m=.00001,contact_target_gap_m=-.0001,ellipse_planning_max=.95,
    maximum_evaluations_per_side=600,maximum_seconds_per_side=60.,
    native_execution=False,teacher_admission=False)

def validate_config(c):
    if c!=CONFIG:raise ValueError("independent frozen CPU bounds and original physical guards required")
def admit(c):return candidate_admission(c)
def rod_guard_applies(label):
    return label.split(":")[-1] not in surface.PADS
def decode(anchor,x):
    x=np.asarray(x,dtype=float)
    if x.shape!=(9,) or not np.isfinite(x).all() or np.any(abs(x)>1+1e-12):
        raise ValueError("bounded translation/rotation/active-joint variables required")
    p=np.array(anchor["root_pose_wxyz"],float);q=np.array(anchor["q_rad"],float)
    if p.shape!=(7,) or q.shape!=(6,) or not np.isfinite(p).all():raise ValueError("original own-side endpoint")
    p[:3]+=x[:3]*CONFIG["translation_scales_m"]
    rot=Rotation.from_rotvec(x[3:6]*CONFIG["reset_rotation_scales_rad"])*Rotation.from_quat(p[[4,5,6,3]])
    p[3:]=np.roll(rot.as_quat(),1)
    lo,hi=active_limits();expand(q)
    scales=np.minimum(CONFIG["joint_scales_rad"],.99*np.minimum(q[:3]-lo[:3],hi[:3]-q[:3]))
    q[:3]+=x[6:]*scales;expand(q)
    return p,q

def predict_region(grid,pose,nodes,atlas):
    T=rigid(pose);points=np.asarray(nodes,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError("finite explicit reference rod nodes")
    sample=surface.base.sample_contact(grid,T,points)
    i=sample["minimum_node_id"]
    if i is None:raise ValueError("required pad outside locked SDF")
    _,grad,ok=surface.base.sample_grid(grid,[sample["local_node_at_minimum"]])
    n=T[:3,:3]@grad[0];length=np.linalg.norm(n)
    if not ok[0] or length<1e-8:raise ValueError("valid SDF normal prediction")
    n=n/length
    # Pure geometry query: 1 is a unit selection weight, never a measured or
    # applied force. Do not export it as contact evidence or a physical sample.
    geometry=classify_pair(dict(contact_point_world_m=(points[i]-.005*n).tolist(),
        outward_normal_world=n.tolist(),normal_force_on_rod_n=1.),T,atlas)
    return dict(node_index=i,sdf_gap_m=sample["minimum_gap_m"],normal_world=n.tolist(),
        point_world_m=(points[i]-.005*n).tolist(),point_link_m=geometry["point_link_m"],
        ellipse_value=geometry["ellipse_value"],uv_m=geometry["uv_m"],
        plane_signed_distance_m=geometry["plane_signed_distance_m"],
        normal_cosine_on_hand=geometry["normal_cosine_on_hand"],
        original_region_passed=geometry["accepted_by_original_patch"],
        first_rejection=geometry["first_rejection"],actual_contact_observed=False)

def evaluate(g,grids,atlas,anchor,x,nodes):
    p,q=decode(anchor,x);poses=surface.base.fk_poses(g,p,q);constraints={};table={};rod={}
    for kind,key in (("source","source_colliders"),("cooked","cooked_pads")):
        for name,shape in g[key].items():
            v=surface.base.transform(shape["vertices"],poses[name]);label=kind+":"+name
            table[label]=float(v[:,2].min()-.3)
            constraints["table:"+label]=table[label]-CONFIG["table_clearance_m"]-CONFIG["reserve_m"]
            # Actual native guard: all22 table bounds,18 non-target capsule
            # bounds, and required-pad depth solely from the locked SDF.
            if rod_guard_applies(label):
                rod[label]=surface._separation(v,nodes)
                constraints["rod:"+label]=rod[label]-CONFIG["reserve_m"]
    pads={n:predict_region(grids[n],poses[n],nodes,atlas[n]) for n in surface.PADS}
    for name,r in pads.items():
        constraints["sdf_target:"+name]=CONFIG["contact_target_gap_m"]-r["sdf_gap_m"]
        constraints["sdf_depth:"+name]=r["sdf_gap_m"]-CONFIG["native_required_gap_m"]-CONFIG["reserve_m"]
        constraints["region:"+name]=(CONFIG["ellipse_planning_max"]-r["ellipse_value"])*.001
        constraints["surface:"+name]=.001-abs(r["plane_signed_distance_m"])-CONFIG["reserve_m"]
        constraints["normal:"+name]=(-.8-r["normal_cosine_on_hand"])*.001
    constraints["opposition"]=(-.25-float(np.dot(*[r["normal_world"] for r in pads.values()])))*.001
    return dict(x=np.asarray(x).tolist(),root_pose_wxyz=p.tolist(),q_rad=q.tolist(),
        constraints=constraints,table_m=table,rod_lower_m=rod,pads=pads,
        native_contact_observed=False,teacher_admission=False)


def original_path(endpoint,old_candidate,fraction):
    if type(fraction) not in (int,float) or not np.isfinite(fraction) or not 0<=fraction<=1:
        raise ValueError("bounded path fraction")
    delta=np.asarray(old_candidate["active_delta_rad"],dtype=float)
    motion=np.asarray(old_candidate.get("root_translation_delta_m",[0,0,0]),dtype=float)
    if delta.shape!=(6,) or motion.shape!=(3,) or not np.isfinite(delta).all() or not np.isfinite(motion).all():
        raise ValueError("original finite action schedule")
    # Same velocity and duration as the archived schedule. Only independent
    # reset/endpoint offsets change; no command amplitude is increased.
    q=np.array(endpoint["q_rad"],float)-(1-fraction)*delta
    root=np.array(endpoint["root_pose_wxyz"],float);root[:3]-=(1-fraction)*motion
    expand(q)
    return dict(root_pose_wxyz=root.tolist(),q_rad=q.tolist()),delta.copy()

def isolated_sides(solve):
    rows={}
    for i,side in enumerate(("left","right")):
        try:rows[side]=solve(i,side)
        except (ValueError,RuntimeError) as e:
            rows[side]=dict(status="blocked",reason=type(e).__name__+": "+str(e),
                native_ready=False,teacher_admission=False)
    return rows

def native_anchor():
    native,r=load_source();last=None
    for e in stream_events(native/"physical.jsonl",r["artifact_sha256"]["physical.jsonl"]):
        if e.get("operation")=="physical":last=decode_owned(e["payload"])["exchange"]
    if last is None or last["closure_geometry"]["raw"]["identity"]["physics_step"]!=234:
        raise ValueError("actual complete bilateral endpoint required")
    return last,r

def run(output,config):
    validate_config(config);pair=parent.load_pair(ROOT);target=parent.expected_boundary(pair,232)
    last,receipt=native_anchor();raw=last["closure_geometry"]["raw"]
    nodes=np.stack([np.full(41,.43),np.linspace(-.25,.25,41),np.full(41,.305)],axis=1)
    result=dict(version=VERSION,status="complete",config=config,
        source_ledger_sha256=receipt["artifact_sha256"]["physical.jsonl"],source_identity=raw["identity"],
        nodes_scope="unchanged straight free-rod Reset seed; table present",
        sides={},new_physics_steps=0,ppo_updates=0,teacher_admission=False,native_recipe_created=False)
    with (output/"progress.jsonl").open("x") as f:
        def solve(side_index,side):
            g=surface.base.load_geometry(ROOT,side=side)
            grids,bindings=surface.base.grids_for(g,surface.base.checked_json(surface.base.SDF_DIR/"result.json",surface.base.SDF_SHA))
            atlas={n:surface._patch(g["source_colliders"][n]["vertices"],side,n) for n in surface.PADS}
            anchor=dict(root_pose_wxyz=target["root_poses_w"][side_index].tolist(),q_rad=target[side+"_q_rad"].tolist())
            baseline=evaluate(g,grids,atlas,anchor,np.zeros(9),nodes)
            body=raw["hands"][side];j=raw["joint_readback"]["hands"][side]
            q=np.array([j["q"][0,j["joint_names"].index(n)] for n in ACTIVE],float)
            po=body["link_poses_xyzw"][0,body["body_names"].index("base_link")]
            measured_anchor=dict(root_pose_wxyz=np.r_[po[:3],po[6],po[3:6]].tolist(),q_rad=q.tolist())
            measured=evaluate(g,grids,atlas,measured_anchor,np.zeros(9),raw["rod_nodes_m"])
            def query(x):
                row=evaluate(g,grids,atlas,anchor,x,nodes)
                f.write(json.dumps(dict(side=side,**row),allow_nan=False)+"\n");f.flush()
                return row
            search=bounded_search(query,np.zeros(9),maximum_evaluations=config["maximum_evaluations_per_side"],
                maximum_seconds=config["maximum_seconds_per_side"])
            return dict(source_geometry_sha256=g["source_sha256"],sdf_bindings=bindings,
                baseline=baseline,measured_endpoint_fk_diagnostic=measured,search=search,
                reference_state_not_optimized_actual_trajectory=True)
        result["sides"]=isolated_sides(solve)
        f.flush();os.fsync(f.fileno())
    result["all_static_candidates_passed"]=all(r.get("search",{}).get("best",{}).get("admission",{}).get("static_constraints_passed") is True for r in result["sides"].values())
    result.update(continuous_path_proved=False,tracking_qualified=False,native_ready=False)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--config",type=Path,default=ROOT/"configs/dh116_tasks/flexible_pinch_effective_target_cpu_v3.json")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:r=run(a.output,json.loads(a.config.read_bytes()))
    except Exception as e:r=dict(status="blocked",reason=type(e).__name__+": "+str(e),new_physics_steps=0,ppo_updates=0,teacher_admission=False)
    r.update(elapsed_s=time.monotonic()-start,script_sha256=surface.base.digest(__file__),config_sha256=surface.base.digest(a.config))
    with (a.output/"result.json").open("x") as f:
        json.dump(encode_owned(r),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:r.get(k) for k in ("status","reason","elapsed_s","all_static_candidates_passed","native_ready")}))
    return int(r["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
