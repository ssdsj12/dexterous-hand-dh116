#!/usr/bin/env python3
"""CPU-only measured-frame SDF closure direction; never writes a native recipe."""
import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np

from scripts.audit_dh116_pinch_initial_pose import (
    checked_json,digest,finite_array,load_geometry,pose_matrix,transform,
    world_vertices,capsule_gap,path_displacement_bound,convex_patch_closest)
from scripts.audit_dh116_pinch_v3_evidence import LEDGER_SHA,RECEIPT_SHA,ledger
from scripts.solve_dh116_tripod_joint_root_direction import envelope_support
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm
from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,active_limits,expand

VERSION="dh116-pinch-closure-margin-cpu-v1"
REQUIRED=("finger14_link","finger23_link")
EVIDENCE=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence")
RECEIPT=EVIDENCE/"native-attempts/flex.f4-pinch-left-v3-native/113882bbab640846e6b3d3d5042bdfa6487ddbffbd497869d71acd8e11eb8362/attempt-dsfwx5kr/receipt.json"
MARGIN=.0001
SDF_CONTACT=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-pair-build-zQEenV/source/third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh")


def sample_grid(grid,points,*,dtype=np.float32):
    """contact.cuh sdfSample arithmetic order; CPU estimate, no GPU bitwise claim."""
    v=np.asarray(grid["values"],dtype=dtype);o=np.asarray(grid["origin"],dtype=dtype)
    p=np.asarray(points,dtype=dtype);h=dtype(grid["spacing"]);inv=dtype(1)/h
    if (v.ndim!=3 or min(v.shape)<2 or o.shape!=(3,) or p.ndim!=2 or p.shape[1]!=3
            or not np.isfinite(v).all() or not np.isfinite(o).all()
            or not np.isfinite(p).all() or not np.isfinite(h) or h<=0):
        raise ValueError("finite three-dimensional locked SDF and point array required")
    u=(p-o)*inv;inside=np.all((u>=0)&(u<np.array(v.shape)-1),axis=1)
    phi=np.full(len(p),np.nan,dtype=dtype);grad=np.full((len(p),3),np.nan,dtype=dtype)
    if not inside.any():return phi,grad,inside
    i=u[inside].astype(np.int64)
    f=np.clip((p[inside]-(o+i.astype(dtype)*h))*inv,dtype(0),dtype(1))
    fx,fy,fz=f.T
    def c(a,b,c):return v[i[:,0]+a,i[:,1]+b,i[:,2]+c]
    c000,c001,c010,c011,c100,c101,c110,c111=[c(*x) for x in (
        (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1))]
    dx00,dx01,dx10,dx11=c100-c000,c101-c001,c110-c010,c111-c011
    c00,c01,c10,c11=c000+fx*dx00,c001+fx*dx01,c010+fx*dx10,c011+fx*dx11
    dy0,dy1=c10-c00,c11-c01;c0,c1=c00+fy*dy0,c01+fy*dy1
    phi[inside]=c0+fz*(c1-c0)
    gx0,gx1=dx00+fy*(dx10-dx00),dx01+fy*(dx11-dx01)
    grad[inside]=np.stack((gx0+fz*(gx1-gx0),dy0+fz*(dy1-dy0),c1-c0),axis=-1)*inv
    return phi,grad,inside


def sample_contact(grid,pose,nodes,*,radius=.005,dtype=np.float32):
    t=finite_array(pose,(4,4));nodes=finite_array(nodes)
    local=(nodes.astype(dtype)-t[:3,3].astype(dtype))@t[:3,:3].astype(dtype)
    phi,gradient,valid=sample_grid(grid,local,dtype=dtype)
    gap=phi-dtype(radius);norm=np.linalg.norm(gradient,axis=1)
    candidates=np.where(valid)[0];i=int(candidates[np.argmin(gap[valid])]) if len(candidates) else None
    return dict(minimum_gap_m=None if i is None else float(gap[i]),minimum_node_id=i,
        gradient_norm_at_minimum=None if i is None else float(norm[i]),
        in_lattice_nodes=np.flatnonzero(valid).tolist(),
        negative_gap_nodes=np.flatnonzero(valid&(gap<0)).tolist(),
        contact_predicate_nodes=np.flatnonzero(valid&(gap<0)&(norm>0)).tolist(),
        node_gaps_m=[float(x) if ok else None for x,ok in zip(gap,valid)],
        local_node_at_minimum=None if i is None else local[i].astype(float).tolist(),
        cpu_dtype=np.dtype(dtype).name,native_contact_observed=False)


def relative_pose(actual,old_fk,new_fk):
    return finite_array(actual,(4,4))@np.linalg.inv(finite_array(old_fk,(4,4)))@finite_array(new_fk,(4,4))


def command_path(q,delta):
    q=finite_array(q,(6,));delta=finite_array(delta,(2,))
    if np.any(delta<0) or np.any(delta>.3*160/480+1e-12):
        raise ValueError("original 160-step 0.3 rad/s closure cap required")
    full=np.zeros(6);full[1:3]=delta
    expand(q);expand(q+full)
    fractions=np.clip((np.arange(239)-48)/160,0,1)
    return q+fractions[:,None]*full


def margin_gate(pads):
    return set(pads)==set(REQUIRED) and all(
        r["minimum_gap_m"] is not None and np.isfinite(r["minimum_gap_m"])
        and r["minimum_gap_m"]<=-MARGIN
        and r["gradient_norm_at_minimum"] is not None and r["gradient_norm_at_minimum"]>0
        for r in pads.values())


def fk_poses(geometry,pose,q):
    delta=pose_matrix(pose)@np.linalg.inv(pose_matrix(geometry["pose"]))
    return {k.rsplit("/",1)[-1]:delta@v for k,v in geometry["fk"].poses(q).items()}


def load_inputs(project):
    receipt=checked_json(RECEIPT,RECEIPT_SHA);output=Path(receipt["output"])
    for name,sha in LEDGER_SHA.items():
        if digest(output/name)!=sha:raise ValueError("original v3 ledger changed")
    plan=checked_json(output/"plan.json","bfdb684a29fc117925fa1bbad35b86f0da9088de3978c68c29cee8471809aa1f")
    if digest(SDF_CONTACT)!="8d90d9d18c853fbc3931a69435000726f49d36c887e62debe10225e8aa77d2bb":
        raise ValueError("actual contact implementation changed")
    cfg=checked_json(project/"configs/dh116_tasks/flexible_pinch_pose_candidate_v2.json",
        "a2eefb1a8f487fdba7e567c51aab005211dec3f658d2aa939c0ff7289de0dc44")
    rows=list(ledger(output/"physics.jsonl"))
    if len(rows)!=238 or [r["phase_step"] for r in rows]!=list(range(1,239)):
        raise ValueError("complete ordered original v3 frames required")
    if (set(plan["joint_names"])!=set(JOINTS) or len(plan["joint_names"])!=11
            or set(plan["pad_bindings"])!=set(DISTAL_PAD_LINKS)):
        raise ValueError("exact original joint and five SDF identities required")
    grids={}
    for n,r in plan["pad_bindings"].items():
        if digest(r["sdf_path"])!=r["sdf_sha256"]:raise ValueError("locked SDF changed")
        with np.load(r["sdf_path"],allow_pickle=False) as d:
            grids[n]={k:d[k].copy() for k in d.files}
        if float(grids[n]["spacing"])!=.001:raise ValueError("original 1mm grid required")
    geometry=load_geometry(project,side="left")
    for n,r in plan["pad_bindings"].items():
        if (r["capture_sha256"]!=geometry["cooked_pads"][n]["capture_sha256"]
                or r["geometry_sha256"]!=geometry["cooked_pads"][n]["geometry_sha256"]):
            raise ValueError("SDF/cooked original link identity differs")
    return geometry,cfg,plan,rows,grids,output


def measured_anchors(geometry,cfg,plan,rows):
    active=[plan["joint_names"].index(n) for n in ACTIVE]
    anchors={}
    for step in (208,238):
        row=rows[step-1];state=row["post_state"];q=np.asarray(state["q"])[active]
        fk=fk_poses(geometry,cfg["base_pose_wxyz"],q)
        actual={n:pose_matrix(np.r_[state["link_pos"][i],state["link_quat"][i]])
                for i,n in enumerate(DISTAL_PAD_LINKS)}
        anchors[step]=dict(q=q,fk=fk,actual=actual,nodes=np.array(row["rod_nodes_w"]),
            pose_residual_m={n:float(np.linalg.norm(actual[n][:3,3]-fk[n][:3,3])) for n in actual})
    return anchors


def endpoint(geometry,cfg,grids,anchors,extra,spread=0.):
    extra=finite_array(extra,(2,))
    if np.any(extra<-.0021) or np.any(extra>.075+1e-12):
        raise ValueError("bounded closure direction outside original total speed cap")
    frames={}
    for step,a in anchors.items():
        q=a["q"].copy();q[0]+=spread;q[1:3]+=extra;expand(q)
        fk=fk_poses(geometry,cfg["base_pose_wxyz"],q)
        frames[str(step)]={}
        for name in REQUIRED:
            t=relative_pose(a["actual"][name],a["fk"][name],fk[name])
            frames[str(step)][name]=sample_contact(grids[name],t,a["nodes"])
    gaps=[frames[str(s)][n]["minimum_gap_m"] for s in anchors for n in REQUIRED]
    if any(x is None for x in gaps):raise ValueError("required pad lost actual SDF lattice support")
    return dict(initial_spread_change_rad=float(spread),extra_delta_rad=extra.tolist(),total_delta_rad=(extra+.025).tolist(),
        frames=frames,worst_required_gap_m=float(max(gaps)),
        sdf_margin_passed=all(margin_gate(r) for r in frames.values()))


def prepare_metadata(project,geometry,plan,anchors,output):
    return dict(measured_output=str(output),ledger_sha256=LEDGER_SHA,
        source_sha256={str(Path(__file__).resolve()):digest(__file__),str(SDF_CONTACT):digest(SDF_CONTACT),
            **{str(project/n):digest(project/n) for n in (
                "scripts/audit_dh116_pinch_initial_pose.py","scripts/audit_dh116_pinch_v3_evidence.py",
                "scripts/audit_dh116_power_closure_constraints.py","scripts/solve_dh116_tripod_joint_root_direction.py",
                "configs/dh116_tasks/flexible_pinch_candidate_v1.json",
                "configs/dh116_tasks/flexible_pinch_pose_candidate_v2.json")}},
        geometry_sources=geometry["source_sha256"],sdf_bindings=plan["pad_bindings"],
        measured_fk_position_residual_m={str(s):a["pose_residual_m"] for s,a in anchors.items()})


def measured_grid_audit(geometry,rows,grids):
    if set(grids)!=set(DISTAL_PAD_LINKS):raise ValueError("all original five grids required")
    result=dict(quiet_rows=[],selected_frames={})
    for row in rows:
        step=row["phase_step"]
        if step>48 and step not in (208,238):continue
        state=row["post_state"];pads={}
        for i,name in enumerate(DISTAL_PAD_LINKS):
            t=pose_matrix(np.r_[state["link_pos"][i],state["link_quat"][i]])
            sampled=sample_contact(grids[name],t,row["rod_nodes_w"])
            if step<=48:
                if sampled["contact_predicate_nodes"]:raise ValueError("quiet replay predicts initial hand contact")
                pads[name]=sampled
            elif name in REQUIRED:
                sampled64=sample_contact(grids[name],t,row["rod_nodes_w"],dtype=np.float64)
                v=transform(geometry["cooked_pads"][name]["vertices"],t)
                node=np.asarray(row["rod_nodes_w"])[sampled["minimum_node_id"]]
                feasible=convex_patch_closest(v,node,[np.zeros(3)])
                sampled.update(fp64_sample=sampled64,
                    continuous_capsule_gap=capsule_gap(v,row["rod_nodes_w"],.005),
                    selected_node_convex_feasible_gap_m=feasible["distance_m"]-.005)
                pads[name]=sampled
        if step<=48:result["quiet_rows"].append(dict(phase_step=step,pads=pads))
        else:result["selected_frames"][str(step)]=pads
    result["zero_contact_quiet_frames_replayed"]=len(result["quiet_rows"])
    return result


def path_gate(table_lower,other_lower,initial_lower):
    return dict(static_geometry_passed=bool(table_lower>=.0019 and other_lower>=0 and initial_lower>=0),
        runtime_2p1mm_passed=bool(table_lower>=.0021),
        minimum_table_lower_m=float(table_lower),minimum_nonrequired_lower_m=float(other_lower),
        minimum_initial_all_shape_lower_m=float(initial_lower),
        native_ready=False,native_recipe_created=False)


def path_protection(geometry,cfg,total_delta,records):
    trajectory=command_path(cfg["active_q_rad"],total_delta)
    q0=trajectory[0];delta=trajectory[208]-q0;pose=cfg["base_pose_wxyz"]
    layout=geometry["reference"]["reference_layout"];top=layout["table_center"][2]+layout["table_size"][2]/2
    bounds={kind:path_displacement_bound(geometry,q0,delta,cooked=(kind=="cooked"))
            for kind in ("source","cooked")}
    table_min={};other_min={};initial_min=[]
    for k,q in enumerate(trajectory[48:209]):
        row=dict(path_index=k,table_lower_m={},nonrequired_lower_m={})
        for kind in ("source","cooked"):
            shapes=world_vertices(geometry,pose,q,cooked=(kind=="cooked"))
            for name,v in shapes.items():
                identity=kind+":"+name;travel=bounds[kind][name]/320
                t=float(v[:,2].min()-top)-travel
                table_min[identity]=min(table_min.get(identity,float("inf")),t)
                row["table_lower_m"][identity]=t
                if name not in REQUIRED or k==0:
                    witness=envelope_support(v,layout,.005)
                    g=witness["surface_gap_lower_bound_m"]
                    if k==0:initial_min.append(g)
                    if name not in REQUIRED:
                        row["nonrequired_lower_m"][identity]=g-travel
                        other_min[identity]=min(other_min.get(identity,float("inf")),g-travel)
        with _commit_row_without_alarm():records.append(row)
    return dict(path_gate(min(table_min.values()),min(other_min.values()),min(initial_min)),
        sample_count=len(records),all_shape_table_lower_m=table_min,
        nonrequired_envelope_lower_m=other_min,commanded_target_end=trajectory[208].tolist(),
        commanded_velocity_rad_s=(delta*3).tolist(),normalized_action24=[0.,total_delta[0]*10,total_delta[1]*10]+[0.]*21,
        candidate_initial_q=q0.tolist(),unchanged_base_pose_wxyz=pose,
        quiet_steps=48,move_steps=160,hold_steps=30,
        proof_scope="commanded_fixed_base_FK_only_no_dynamic_tracking_or_force_proof")


def candidate_geometry(geometry,cfg,grids,anchors,extra,spread=0.):
    result={}
    for step,a in anchors.items():
        q=a["q"].copy();q[0]+=spread;q[1:3]+=extra
        fk=fk_poses(geometry,cfg["base_pose_wxyz"],q);result[str(step)]={}
        for name in REQUIRED:
            t=relative_pose(a["actual"][name],a["fk"][name],fk[name])
            v=transform(geometry["cooked_pads"][name]["vertices"],t)
            cpu=sample_contact(grids[name],t,a["nodes"])
            fp64=sample_contact(grids[name],t,a["nodes"],dtype=np.float64)
            result[str(step)][name]=dict(sdf_fp32=cpu,sdf_fp64=fp64,
                continuous_capsule_gap=capsule_gap(v,a["nodes"],.005),
                predicted_pad_pose_matrix=t.tolist(),anchor_kind="measured_pose_times_original_FK_relative_change")
    return result


def load_refinement_parent(path):
    parent=checked_json(path,"c861cfeaccabd242d11af1e4179290ef1082fc87f93abd8bb71670d65f6b4aea")
    if (parent["status"]!="complete" or parent["offline_candidate_passed"] is not False
            or len(parent["endpoint_queries"])!=12 or len(parent["path_records"])!=161):
        raise ValueError("exact rejected first direction required")
    return parent


def refinement_parameters(cfg,base_extra,x):
    x=finite_array(x,(2,));extra=finite_array(base_extra,(2,)).copy()
    if abs(x[0])>.05 or abs(x[1])>.02:raise ValueError("initial-spread refinement trust box exceeded")
    new=dict(cfg);new["active_q_rad"]=list(cfg["active_q_rad"])
    new["active_q_rad"][0]+=float(x[0]);extra[0]+=x[1]
    command_path(new["active_q_rad"],extra+.025)
    return new,extra


def quiet_joint_hypothesis(geometry,cfg,rows,grids,initial_delta):
    initial_delta=finite_array(initial_delta,(6,))
    minimum=float("inf");total=0
    for row in rows[:48]:
        # Fixed old canonical joint order is validated against the locked plan.
        q=np.array([row["post_state"]["q"][i] for i in (0,5,1,2,3,4)])
        old=fk_poses(geometry,cfg["base_pose_wxyz"],q);q+=initial_delta
        new=fk_poses(geometry,cfg["base_pose_wxyz"],q)
        for i,name in enumerate(DISTAL_PAD_LINKS):
            state=row["post_state"]
            actual=pose_matrix(np.r_[state["link_pos"][i],state["link_quat"][i]])
            pose=relative_pose(actual,old[name],new[name])
            sampled=sample_contact(grids[name],pose,row["rod_nodes_w"])
            total+=len(sampled["contact_predicate_nodes"])
            if sampled["minimum_gap_m"] is not None:minimum=min(minimum,sampled["minimum_gap_m"])
    return dict(predicted_contact_predicates=total,minimum_in_lattice_gap_m=minimum,
        frames=48,scope="changed_initial_q_counterfactual_at_historical_rod_nodes_not_actual_reset")


def refine_initial(geometry,cfg,plan,rows,grids,anchors,parent,result):
    base_extra=np.array(parent["selected_endpoint"]["extra_delta_rad"])
    records=result["endpoint_queries"]
    def query(x):
        if len(records)>=12:raise ValueError("12 refinement endpoint limit")
        new,extra=refinement_parameters(cfg,base_extra,x)
        row=endpoint(geometry,cfg,grids,anchors,extra,float(x[0]))
        q=command_path(new["active_q_rad"],extra+.025)[208]
        shapes=world_vertices(geometry,cfg["base_pose_wxyz"],q)
        support=envelope_support(shapes["finger13_link"],geometry["reference"]["reference_layout"],.005)
        row.update(refinement_x=np.asarray(x).tolist(),finger13_endpoint_lower_m=support["surface_gap_lower_bound_m"])
        with _commit_row_without_alarm():records.append(row)
        return row
    def residual(row):
        return np.array([max(row["frames"][str(s)][REQUIRED[0]]["minimum_gap_m"] for s in anchors),
            row["finger13_endpoint_lower_m"]])
    base=query(np.zeros(2));jac=[]
    for h in (.001,.0005):
        jac.append(np.stack([(residual(query(np.eye(2)[j]*h))-residual(query(-np.eye(2)[j]*h)))/(2*h)
                             for j in range(2)],axis=1))
    relative=float(np.linalg.norm(jac[1]-jac[0])/max(np.linalg.norm(jac[1]),1e-12))
    result.update(jacobians=[j.tolist() for j in jac],jacobian_relative_difference=relative)
    if relative>.2:raise ValueError("initial spread two-scale direction unstable")
    direction=np.linalg.solve(jac[-1],np.array([-.00015,.0001])-residual(base))
    result["predicted_spread_thumb_adjustment_rad"]=direction.tolist()
    refinement_parameters(cfg,base_extra,direction)
    for scale in (1.,.75,.5):query(direction*scale)
    acceptable=[r for r in records[9:] if r["sdf_margin_passed"] and r["finger13_endpoint_lower_m"]>0]
    if not acceptable:
        result.update(offline_candidate_passed=False,rejection="no refined endpoint passes SDF margin and proximal lower bound")
        return
    selected=acceptable[0];x=np.array(selected["refinement_x"])
    new,extra=refinement_parameters(cfg,base_extra,x)
    result["selected_endpoint"]=selected
    result["selected_geometry"]=candidate_geometry(geometry,cfg,grids,anchors,extra,float(x[0]))
    result["quiet_hypothesis"]=quiet_joint_hypothesis(geometry,cfg,rows,grids,[float(x[0]),0,0,0,0,0])
    result["path_records"]=[]
    result["path_protection"]=path_protection(geometry,new,extra+.025,result["path_records"])
    result["offline_candidate_passed"]=bool(result["path_protection"]["static_geometry_passed"]
        and result["quiet_hypothesis"]["predicted_contact_predicates"]==0)


def safe_suffix_start(records,selection_clearance=.00195):
    if len(records)!=161 or [r["path_index"] for r in records]!=list(range(161)):
        raise ValueError("complete ordered 161-point parent protection required")
    if not np.isfinite(selection_clearance) or selection_clearance<.0019:
        raise ValueError("selection buffer cannot weaken original 1.9mm gate")
    last_failure=-1
    for i,row in enumerate(records):
        t=np.asarray(list(row["table_lower_m"].values()))
        g=np.asarray(list(row["nonrequired_lower_m"].values()))
        if not len(t) or not len(g) or not np.isfinite(t).all() or not np.isfinite(g).all():
            raise ValueError("finite complete protection rows required")
        if min(t)<selection_clearance or min(g)<0:last_failure=i
    if last_failure>=160:raise ValueError("no safe suffix exists")
    return last_failure+1


def suffix_parameters(q,delta,start):
    if type(start) is not int or not 0<=start<160:raise ValueError("nonempty original path suffix required")
    old=command_path(q,delta)
    initial=old[48+start];remaining=np.asarray(delta)*(1-start/160)
    new=command_path(initial,remaining)
    if not np.allclose(new[208],old[208],atol=1e-14,rtol=0):
        raise ValueError("suffix changed the original endpoint")
    return initial,remaining


def crop_proven_path(geometry,cfg,rows,grids,parent_path,result):
    parent=checked_json(parent_path,"6778504d7485b8ba99f3629ad134f8f79840e2524612f9cc6659b0e6c2ae515f")
    records=parent["path_records"];start=safe_suffix_start(records)
    old_initial=parent["path_protection"]["unchanged_initial_q"]
    total=parent["selected_endpoint"]["total_delta_rad"]
    initial,remaining=suffix_parameters(old_initial,total,start)
    layout=geometry["reference"]["reference_layout"];initial_gaps={}
    for kind in ("source","cooked"):
        for name,v in world_vertices(geometry,cfg["base_pose_wxyz"],initial,cooked=(kind=="cooked")).items():
            initial_gaps[kind+":"+name]=envelope_support(v,layout,.005)["surface_gap_lower_bound_m"]
    quiet=quiet_joint_hypothesis(geometry,cfg,rows,grids,initial-np.asarray(cfg["active_q_rad"]))
    t=min(v for r in records[start:] for v in r["table_lower_m"].values())
    g=min(v for r in records[start:] for v in r["nonrequired_lower_m"].values())
    gate=path_gate(t,g,min(initial_gaps.values()))
    result.update(scope="exact_safe_suffix_reparameterization_of_saved_geometry_path",
        parent_result_sha256=digest(parent_path),parent_source_sha256=parent["source_sha256"],
        path_suffix_start=start,reused_path_samples=len(records)-start,new_endpoint_queries=0,
        suffix_selection_table_buffer_m=.00195,original_table_gate_m=.0019,
        new_initial_shape_checks=len(initial_gaps),initial_shape_lower_m=initial_gaps,
        original_initial_q=cfg["active_q_rad"],candidate_initial_q=initial.tolist(),
        unchanged_base_pose_wxyz=cfg["base_pose_wxyz"],candidate_closure_delta_rad=remaining.tolist(),
        candidate_velocity_rad_s=(remaining*3).tolist(),
        candidate_normalized_thumb_index=(remaining*10).tolist(),
        unchanged_target_end=parent["path_protection"]["commanded_target_end"],
        quiet_steps=48,move_steps=160,hold_steps=30,
        selected_endpoint=parent["selected_endpoint"],selected_geometry=parent["selected_geometry"],
        quiet_hypothesis=quiet,path_protection=gate,
        offline_candidate_passed=bool(gate["static_geometry_passed"] and
            quiet["predicted_contact_predicates"]==0 and parent["selected_endpoint"]["sdf_margin_passed"]),
        limitations=["CPU SDF interpolant predicate is not native contact/force",
            "Endpoint uses historical tracking residual as a hypothesis, not a new dynamics prediction",
            "12 nonpad shapes have source envelopes only; actual cooked geometry unknown",
            "2.1mm bimanual runtime floor remains independent and is not granted",
            "No self-collision, grasp, full371D, material, lifetime or native qualification"])


def export_candidate(path,output):
    r=checked_json(path,"f96b29a82e6b532e153f3b63e91a34a6303c7d3be9bff7b686eb3508b65ed6d0")
    if (r["status"]!="complete" or r["offline_candidate_passed"] is not True
            or r["native_ready"] is not False or r["native_recipe_created"] is not False):
        raise ValueError("exact independently scoped CPU candidate required")
    low,high=active_limits()
    packet=dict(version="dh116-single-left-fixed-base-closure-candidate-cpu-v1",
        scope="single_left_fixed_base_offline_contact_discovery_only",
        producer_sha256=digest(__file__),candidate_result_path=str(path),
        candidate_result_sha256=digest(path),original_native_receipt_sha256=RECEIPT_SHA,
        original_native_ledger_sha256=LEDGER_SHA,
        original_path_result_path=str(path.parent.parent/"run02/result.json"),
        original_path_result_sha256=r["parent_result_sha256"],
        inherited_geometry_samples=dict(first_index=r["path_suffix_start"],last_index=160,
            count=r["reused_path_samples"],proof="same_geometric_path_suffix_with_unchanged_conservative_travel_bounds"),
        base_pose_world_wxyz=r["unchanged_base_pose_wxyz"],base_fixed=True,side="left",
        active_joint_names=list(ACTIVE),original_initial_q_rad=r["original_initial_q"],
        candidate_initial_q_rad=r["candidate_initial_q"],target_end_q_rad=r["unchanged_target_end"],
        active_joint_lower_rad=low.tolist(),active_joint_upper_rad=high.tolist(),
        action=dict(physics_hz=480,quiet_steps=48,move_steps=160,hold_steps=30,
            active_velocity_limit_rad_s=.3,thumb_index_delta_rad=r["candidate_closure_delta_rad"],
            thumb_index_velocity_rad_s=r["candidate_velocity_rad_s"],
            canonical_action24=[0.,*r["candidate_normalized_thumb_index"],*([0.]*21)],
            all_other_action_channels_zero=True),
        gates=dict(single_hand_static_table_minimum_m=.0019,suffix_selection_buffer_m=.00195,
            policy_six_axis_projection_minimum_m=.0021,policy_projection_qualified=False,
            actual_table_lower_m=r["path_protection"]["minimum_table_lower_m"],
            non_target_geometry_lower_m=r["path_protection"]["minimum_nonrequired_lower_m"],
            initial_all_shape_rod_lower_m=r["path_protection"]["minimum_initial_all_shape_lower_m"],
            required_sdf_negative_margin_m=MARGIN,old_abs_50um_reach_gate_applied=False),
        quiet_counterfactual=r["quiet_hypothesis"],expected_measured_anchor_pose_and_gaps=r["selected_geometry"],
        expected_gap_scope="historical_208_238_frame_plus_relative_source_FK_change_not_new_native_readback",
        source_sha256=r["source_sha256"],sdf_bindings=r["sdf_bindings"],
        original_geometry_source_sha256=r["geometry_sources"],
        material_configuration_changed=False,rod_seed_changed=False,
        old_v2_preflight_compatible=False,old_geometry_candidate_v2_compatible=False,
        incompatibility_reasons=["old_v2_locks_original_pose_and_action_source_identity",
            "old_CPU_geometry_candidate_requires_equal_two_digit_velocity_and_abs_gap_lt_50um",
            "actual_initial_geometry_v2_still_requires_new_same_clock_native_body_pose_capture"],
        native_ready=False,native_recipe_created=False,contact_qualified=False,teacher_admitted=False,
        physics_steps=0,training_updates=0,limitations=r["limitations"])
    with Path(output).open("x") as f:json.dump(packet,f,indent=2,allow_nan=False);f.write("\n")
    return packet


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--wall-seconds",type=float,default=50.)
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--refine-from",type=Path)
    group.add_argument("--crop-from",type=Path)
    group.add_argument("--export-from",type=Path)
    args=parser.parse_args(argv)
    if not np.isfinite(args.wall_seconds) or not 1<=args.wall_seconds<=50:
        parser.error("soft CPU budget must be 1..50 seconds")
    args.output.mkdir(parents=True,exist_ok=False)
    if args.export_from:
        export_candidate(args.export_from,args.output/"candidate.json")
        return 0
    result=dict(version=VERSION,status="partial_setup",endpoint_queries=[],native_ready=False,
        native_recipe_created=False,physics_steps=0,training_updates=0)
    started=time.monotonic()
    def expire(*unused):raise BudgetExpired()
    old=signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,args.wall_seconds)
    try:
        project=Path(__file__).resolve().parents[1]
        geometry,cfg,plan,rows,grids,output=load_inputs(project)
        anchors=measured_anchors(geometry,cfg,plan,rows)
        result.update(prepare_metadata(project,geometry,plan,anchors,output))
        if args.crop_from:
            crop_proven_path(geometry,cfg,rows,grids,args.crop_from,result)
            result["status"]="complete"
        elif args.refine_from:
            parent=load_refinement_parent(args.refine_from)
            result["parent_result_sha256"]=digest(args.refine_from)
            result["scope"]="bounded_initial_spread_refinement_after_proximal_rejection"
            refine_initial(geometry,cfg,plan,rows,grids,anchors,parent,result)
            result["status"]="complete"
        else:
            result["measured_grid_replay"]=measured_grid_audit(geometry,rows,grids)
        if not args.refine_from and not args.crop_from:
            def query(extra):
                if len(result["endpoint_queries"])>=12:raise ValueError("12 endpoint limit")
                row=endpoint(geometry,cfg,grids,anchors,extra)
                with _commit_row_without_alarm():result["endpoint_queries"].append(row)
                return row
            base=query(np.zeros(2));jac=[]
            for h in (.001,.0005):
                columns=[]
                for j in range(2):
                    e=np.eye(2)[j]*h
                    plus,minus=query(e),query(-e)
                    def gap(r):return np.array([max(r["frames"][str(s)][n]["minimum_gap_m"] for s in anchors) for n in REQUIRED])
                    columns.append((gap(plus)-gap(minus))/(2*h))
                jac.append(np.stack(columns,axis=1))
            relative=float(np.linalg.norm(jac[1]-jac[0])/max(np.linalg.norm(jac[1]),1e-12))
            result.update(jacobians= [j.tolist() for j in jac],jacobian_relative_difference=relative)
            if relative>.2:raise ValueError("two-scale SDF direction unstable")
            worst=np.array([max(base["frames"][str(s)][n]["minimum_gap_m"] for s in anchors) for n in REQUIRED])
            direction=np.linalg.solve(jac[-1],-.00015-worst)
            if np.any(direction<0) or np.any(direction>.075):raise ValueError("required closure direction outside speed budget")
            result["predicted_extra_delta_rad"]=direction.tolist()
            for scale in (1.,1.25,1.5):
                if np.max(direction*scale)>.075:break
                query(direction*scale)
            acceptable=[r for r in result["endpoint_queries"][9:] if r["sdf_margin_passed"]]
            if acceptable:
                selected=acceptable[0]
                result["selected_endpoint"]=selected
                result["selected_geometry"]=candidate_geometry(geometry,cfg,grids,anchors,np.array(selected["extra_delta_rad"]))
                result["path_records"]=[]
                result["path_protection"]=path_protection(geometry,cfg,np.array(selected["total_delta_rad"]),result["path_records"])
                result["offline_candidate_passed"]=result["path_protection"]["static_geometry_passed"]
            else:
                result["offline_candidate_passed"]=False
                result["rejection"]="no tested endpoint reaches both measured-frame SDF margins"
            result["status"]="complete"
    except BudgetExpired:result["status"]="partial_timeout"
    except ValueError as exc:result.update(status="diagnostic_error",error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    result["elapsed_s"]=time.monotonic()-started
    with (args.output/"result.json").open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({k:result[k] for k in ("status","elapsed_s")}))
    return 1 if result["status"]=="diagnostic_error" else 0


if __name__=="__main__":raise SystemExit(main())
