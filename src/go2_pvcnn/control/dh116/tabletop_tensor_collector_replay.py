"""CPU replay of a native N2 Pinch probe with the original control and reward readers.

Run with python -m go2_pvcnn.control.dh116.tabletop_tensor_collector_replay NATIVE --output FILE.
Requires the unmodified locked meshes/SDF inputs and all sixteen captured SDK rows.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.audit_dh116_pinch_initial_pose import CAPTURE, pose_matrix, transform
from scripts.bake_dh116_deformx_pad_sdfs import input_packets
from scripts.prepare_dh116_pinch_closure_margin import sample_contact
from scripts.solve_dh116_tripod_sdf_constrained_direction import SDF_DIR,SDF_SHA,checked_json,grids_for
from scripts.verify_dh116_deformx_rigid_bridge import rotation_wxyz
from .tabletop_batched_control import BatchedControl, batch_projection
from .tabletop_batch_geometry import collision_clearance
from .tabletop_deformable_physical_exchange import FlexiblePhysicalExchange
from .tabletop_deformable_wrist_projection import CLEARANCE_M
from .tabletop_dual_hand_rod_runtime import route_world_wrenches
from .tabletop_joint_target_readback import WRIST_ANGULAR
from .tabletop_measured_closure_geometry import REQUIRED, rod_support_lower, required_sdf_valid
from .tabletop_native_teacher_profile import ROOT
from .tabletop_palm_mount import local_translation, CONTRACT as WRIST
from .tabletop_polyline_support_certificate import RADIUS, certify, validate
from .tabletop_shared_teacher_reward import endpoint
from .tabletop_stereo_contract import file_sha256
from .tabletop_training_reward import TrainingReward,SubmittedRequest,load_config

SIDES=("left","right")


def compare(stats,name,actual,expected,*,atol=0.,rtol=0.):
    a,b=np.asarray(actual),np.asarray(expected)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"incomplete original replay {name}")
    residual=float(np.max(np.abs(a.astype(float)-b.astype(float)),initial=0))
    stats[name]=max(stats.get(name,0.),residual)
    np.testing.assert_allclose(a,b,atol=atol,rtol=rtol,err_msg=name)


def hands(meta,state,lane):
    return {s:dict(joint_names=meta["joint_names"][i],active_indices=meta["active_ids"][i],
        q=state["q"][lane:lane+1,i].numpy(),backend_targets=state["backend_targets"][lane:lane+1,i].numpy())
        for i,s in enumerate(SIDES)}


def original_targets(meta,row,stats):
    pre=row["state_pre"]
    ctrl=BatchedControl.__new__(BatchedControl)
    ctrl.device=torch.device("cpu")
    ctrl.plans=[SimpleNamespace(lane=i,pre=lambda i=i:dict(hands=hands(meta,pre,i))) for i in range(2)]
    req=ctrl.prepare(row["action"],row["palms"],row["angles"])
    projected=batch_projection(req,meta["roots"],pre["base_link_pose_w"],row["pre_clearance"],
        row["palms"],row["angles"],device="cpu")
    for lane in (0,1):
        for k in ("palm_offsets_m","projected_angles","projected_root_poses","safe_z_offset_m","safe_z_clipped","projection_reasons"):
            compare(stats,"projection."+k,row["projection"][k][lane],projected[lane][k][0],atol=2e-12)
        for hi in (0,1):
            roots=projected[lane]["projected_root_poses"][:,hi]
            local=local_translation(roots,pre["root_pose_wxyz"][lane:lane+1,hi].double())
            q=req[lane]._request["requested_finger_targets_rad"][:,hi*6:hi*6+6]
            angles=projected[lane]["projected_angles"][:,hi]
            expected=torch.cat((q,local,angles),dim=-1).float()[0]
            compare(stats,"issued_24d",row["command"][hi][lane],expected,atol=1e-7)
            compare(stats,"actual_targets",row["state_post"]["backend_targets"][lane,hi,meta["write_ids"][hi]],expected,atol=1e-7)


def original_routes(meta,state,wrenches,lane):
    poses,centers={},{}
    names=deepcopy(meta["bindings"])
    for key,bind in names.items():
        bind["prim_path"]=bind["prim_paths"][lane]
        side=SIDES.index(bind["side"])
        link=bind["body_index"]
        pose=state["link_pose_xyzw"][lane,side,link].numpy()
        xyz,quat=pose[:3],pose[3:]
        poses[key]=(xyz,quat)
        centers[key]=xyz+rotation_wxyz(quat[[3,0,1,2]])@state["body_com_pose_b_xyzw"][lane,side,link,:3].numpy()
    named={key:wrenches[lane,i].numpy() for i,key in enumerate(meta["mesh_names"])}
    return route_world_wrenches(named,names,poses,centers)[0]


def original_composer(meta,row,stats):
    for i in (0,1):
        state=row["state_pre"]
        routes=original_routes(meta,state,row["reaction"]["mesh_wrenches"],i)
        for hi,s in enumerate(SIDES):
            ids=routes[s]["body_ids"]
            q=state["link_pose_wxyz"][i,hi,ids,3:].numpy()
            for name in ("force","torque"):
                world=routes[s][name+"s"][0].astype(np.float32)
                local=np.stack([rotation_wxyz(quat).T@w for quat,w in zip(q,world,strict=True)])
                # The physical force cap is checked separately on the original world
                # forces in FP64. This bound concerns the Warp FP32 frame conversion.
                scale=np.max(np.abs(world),initial=0.)
                compare(stats,"actual_local_"+name,row["composer"][s][name][i,ids],local,
                    atol=12*np.finfo(np.float32).eps*scale,rtol=0)


class PinchOracle:
    def __init__(self,meta):
        self.meta=meta
        lock=json.loads((ROOT/"docs/contracts/dh116_deformx_pad_sdf_evidence_lock.json").read_text())
        packets=input_packets(CAPTURE,lock)
        summary=checked_json(SDF_DIR/"result.json",SDF_SHA)
        self.shapes,self.grids=[],[]
        for i,s in enumerate(SIDES):
            source={meta["body_names"][i][body]:np.asarray(v) for side,body,v in meta["shapes"] if side==i}
            cooked={}
            for key in meta["bindings"]:
                side,name=key.split(".")
                if side!=s:
                    continue
                _,hull,_,sha=packets[key]
                if meta["bindings"][key]["capture_sha256"]!=sha:
                    raise ValueError("snapshot and original F4 hull source differ")
                cooked[name]=dict(vertices=np.asarray(hull["vertices"]),capture_sha256=sha)
            grids,binding=grids_for(dict(side=s,cooked_pads=cooked),summary)
            if len(source)!=17 or len(cooked)!=5 or any(meta["bindings"][s+"."+key]["sdf_sha256"]!=b["sha256"] for key,b in binding.items()):
                raise ValueError("incomplete original 44 hull and locked SDF geometry")
            self.shapes.append([*source.items(),*((k,v["vertices"]) for k,v in cooked.items())])
            self.grids.append(grids)

    def distance(self,state,nodes,lane,*,post,geometry=None):
        positions=np.asarray(nodes[lane])
        radius=float(self.meta["radius"])
        gaps=[]
        clouds=[]
        for i in (0,1):
            names=self.meta["body_names"][i]
            matrices={name:pose_matrix(state["link_pose_wxyz"][lane,i,link].numpy()) for link,name in enumerate(names)}
            cloud=[]
            if post:
                for name,vertices in self.shapes[i]:
                    world=transform(vertices,matrices[name])
                    cloud.append(world)
                    if float(world[:,2].min()-.3)<CLEARANCE_M:
                        raise ValueError("original scalar source/cooked table guard")
                    if name not in REQUIRED:
                        lower=rod_support_lower(world,positions)+RADIUS-radius
                        if lower<0:
                            cert=certify(world,positions)
                            validate(cert,world,positions)
                            lower=cert["lower_m"]+RADIUS-radius
                        if lower<0:
                            raise ValueError("original scalar non-target support guard")
                clouds.append(np.concatenate(cloud))
            for name in REQUIRED:
                result=sample_contact(self.grids[i][name],matrices[name],positions,radius=radius)
                if not required_sdf_valid(result):
                    raise ValueError("original scalar locked SDF coverage guard")
                if geometry is not None:
                    pad=2*i+REQUIRED.index(name)
                    for k,key in (("in_lattice_nodes","sdf_node_valid"),("negative_gap_nodes","sdf_negative_nodes"),
                            ("contact_predicate_nodes","sdf_contact_nodes")):
                        expected=np.zeros(41,dtype=bool)
                        expected[result[k]]=True
                        np.testing.assert_array_equal(geometry[key][lane,pad].numpy(),expected,err_msg=key)
                gaps.append(max(0.,result["minimum_gap_m"]))
        if post:
            l,r=clouds
            if max((r.min(0)-l.max(0)).max(),(l.min(0)-r.max(0)).max())<0:
                raise ValueError("original scalar hand separation guard")
        return math.fsum(gaps)/len(gaps)


def clock(src,i):
    return dict(episode_uid=src["episode_uids"][i],physics_step=src["sdk_step"],
        solver_time_s=src["solver_time_s"],env_id=i,reset_id=src["reset_ids"][i],rod_uid=src["rod_uids"][i])


def replay(directory):
    files={}
    def load(name):
        path=directory/("tensor_n2_"+name+".pt")
        files[path.name]=file_sha256(path)
        return torch.load(path,map_location="cpu",weights_only=True)
    meta=load("startup")
    cfg=load_config(ROOT/"configs/dh116_training/pinch_finger_goal_reward_v1.json",ppo_gamma=.995)
    oracle=PinchOracle(meta)
    stats={}
    contacts=dict(flat_samples=0,hand_pair_samples=0,nonzero_submitted_hand_wrenches=0,
        maximum_submitted_hand_force_n=0.,maximum_submitted_hand_torque_nm=0.)
    hand_slots=[i for i,name in enumerate(meta["mesh_names"]) if name in meta["bindings"]]
    engines=[TrainingReward(cfg),TrainingReward(cfg)]
    original_engines=[TrainingReward(cfg),TrainingReward(cfg)]
    last=[None,None]
    last_original=[None,None]
    def sample(state,frame,identity,i,distance):
        return endpoint("pinch",clock(identity,i),distance_m=float(distance),
            health=0,guard=None,hands=hands(meta,state,i),finger_goal=cfg.finger_goal,scope="owner_source_unqualified")
    def reset(ids,state,nodes,frame,identity):
        for i in ids:
            d=oracle.distance(state,nodes,i,post=False)
            actual=float(frame["fields"]["distance"][i])
            compare(stats,"distance_m",actual,d,atol=2e-6)
            for engine,history,dist in ((engines,last,actual),(original_engines,last_original,d)):
                h=sample(state,frame,identity,i,dist)
                engine[i].reset_from_verified_state(h,previous_request=SubmittedRequest(
                    "reset:"+h.stamp.episode_uid,"controller_reset_zero",h.stamp,(0.,)*24,"normalized_submitted_command24_preprojection"))
                history[i]=h
    src=meta["source"]
    reset([0,1],meta["state"],meta["nodes"],meta["reset"],src)
    start=src["sdk_step"]
    for window in range(2):
        record=load("window_"+str(window+1))
        settlement=record["packet"]
        action=record["actions"]
        if torch.equal(action[0],action[1]) or action.shape!=(2,24):
            raise ValueError("two distinct real 24D requests required")
        for i in range(2):
            for engine,history in ((engines,last),(original_engines,last_original)):
                engine[i].begin(SubmittedRequest(settlement["tokens"][i],settlement["policy_uids"][i],
                    history[i].stamp,tuple(action[i].tolist()),"normalized_submitted_command24_preprojection"),remaining_substeps=8)
        for j in range(8):
            sdk=start+8*window+j+1
            row=load("sdk_"+str(sdk))
            if src!=row["pre"] or row["post"]["sdk_step"]!=sdk:
                raise ValueError("nonconsecutive native source/Reset identity")
            compare(stats,"sampled_action",row["action"],action)
            mesh_index=row["capture"]["indices"][:,2]
            contacts["flat_samples"]+=len(mesh_index)
            contacts["hand_pair_samples"]+=int(torch.isin(mesh_index,torch.tensor(hand_slots)).sum())
            w=row["reaction"]["mesh_wrenches"][:,hand_slots]
            contacts["nonzero_submitted_hand_wrenches"]+=int(w.ne(0).any(-1).sum())
            contacts["maximum_submitted_hand_force_n"]=max(contacts["maximum_submitted_hand_force_n"],float(w[...,:3].norm(dim=-1).max()))
            contacts["maximum_submitted_hand_torque_nm"]=max(contacts["maximum_submitted_hand_torque_nm"],float(w[...,3:].norm(dim=-1).max()))
            residual=row["capture"]["node_forces"].sum(1)+row["capture"]["mesh_wrenches"][...,:3].sum(1)
            if (residual.norm(dim=-1)>1e-3).any():
                raise ValueError("original rod/mesh force balance guard")
            original_targets(meta,row,stats)
            original_composer(meta,row,stats)
            state=row["state_post"]
            clearance,_=collision_clearance(state["link_pose_wxyz"].numpy(),meta["shapes"])
            compare(stats,"raw_source_clearance_m",row["post_clearance"],clearance,atol=1e-12)
            if np.any(clearance<CLEARANCE_M):
                raise ValueError("original scalar raw table clearance guard")
            for i in range(2):
                routes=original_routes(meta,state,row["capture"]["mesh_wrenches"],i)
                guard=FlexiblePhysicalExchange._guard_failure(None,hands(meta,state,i),routes)
                if guard:
                    raise ValueError("original scalar joint/force guard: "+guard)
                bases=state["base_link_pose_w"][i,:,:3].double()
                track=(bases-row["projection"]["projected_root_poses"][i,:,:3]).norm(dim=-1)
                speed=state["base_link_com_velocity_w"][i,:,:3].double().norm(dim=-1)
                if (track>WRIST["tracking_guard_m"]).any() or (speed>WRIST["linear_speed_guard_m_s"]).any():
                    raise ValueError("original scalar wrist transport guard")
                for side,names in enumerate(meta["joint_names"]):
                    if any(abs(float(state["q"][i,side,names.index(n)]))>math.pi/2 for n in WRIST_ANGULAR):
                        raise ValueError("original scalar wrist angular limit")
                d=oracle.distance(state,row["nodes_post"],i,post=True,geometry=row["geometry"])
                actual=float(row["post_frame"]["fields"]["distance"][i])
                compare(stats,"distance_m",actual,d,atol=2e-6)
                for engine,history,dist in ((engines,last,actual),(original_engines,last_original,d)):
                    h=sample(state,row["post_frame"],row["post"],i,dist)
                    engine[i].observe(h)
                    history[i]=h
            src=row["post"]
        for i in range(2):
            def settle(engine):
                return engine[i].settle_once(settlement["tokens"][i],truncated=bool(settlement["truncated"][i]),
                    buffer_full=bool(settlement["buffer_full"][i]))
            same,original=settle(engines),settle(original_engines)
            compare(stats,"reward_same_measured_distance",settlement["reward"][i],same.reward,atol=2e-12)
            # Bound only the propagated measured-distance differences in the exact
            # clipped scalar potential. Independently compare the identical-input
            # reward above, so a difference in reward logic cannot hide in this bound.
            phi_diff=abs(same.phi_pre-original.phi_pre)+cfg.gamma*abs(same.endpoint_phi-original.endpoint_phi)
            compare(stats,"reward_original_sdf_distance",settlement["reward"][i],original.reward,atol=phi_diff+2e-12)
            if bool(settlement["done"][i])!=original.done or bool(settlement["terminated"][i])!=original.terminated:
                raise ValueError("original terminal semantics differ")
        if window==0:
            r=load("reset_"+str(sdk))
            if tuple(r["ids"])!=(1,) or r["source"]["rod_ticks"][1]!=0:
                raise ValueError("missing selected lane-1 Reset before continuation")
            src=r["source"]
            reset(r["ids"],r["state"],r["nodes"],r["references"],src)
    return dict(scope="N2 Pinch numeric replay on identical measured native frames",sdk_steps=16,lanes=2,
        updates=0,original_guard_passed=True,maximum_absolute_errors=stats,actual_contact_coverage=contacts,input_sha256=files,
        native_qualification=False,teacher_admission=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=replay(args.native)
    from scripts.diagnose_dh116_teacher_initialization import write_new
    write_new(args.output,result)


if __name__=="__main__":
    main()

