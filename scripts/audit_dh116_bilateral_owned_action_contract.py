#!/usr/bin/env python3
"""CPU ideal-tracking replay of original control kernels; no native evidence."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scripts import prepare_dh116_bilateral_owned_pinch as operation
from scripts import solve_dh116_right_pinch_surface_registration as surface
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,active_limits
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from go2_pvcnn.control.dh116.tabletop_command_kernels import finger_pre_guard,surface_final_guard,wrist_requested_targets
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import project_policy_wrist_targets
from go2_pvcnn.control.dh116.tabletop_palm_mount import local_translation

from go2_pvcnn.control.dh116.tabletop_bilateral_owned_geometry import separation_lower

def target_errors(pair,q,root,ordinal):
    expected=operation.expected_boundary(pair,ordinal)
    wanted=np.r_[expected["left_q_rad"],expected["right_q_rad"]]
    return float(np.max(abs(np.asarray(q)-wanted))),float(np.max(abs(np.asarray(root)[:,:3]-expected["root_poses_w"][:,:3])))

def run(out,emit):
    pair=operation.load_pair();actions=operation.action_schedule(pair)
    seed=torch.tensor(pair["reset_seed"]["reset_root_pose_w"],dtype=torch.float64)[None]
    ids=[JOINTS.index(n) for n in ACTIVE]
    q=torch.tensor(pair["reset_seed"]["joint_positions"],dtype=torch.float64)[:,ids].reshape(1,12)
    command=q.clone();palms=torch.zeros((1,2,3),dtype=torch.float64);angles=torch.zeros_like(palms)
    low,high=[torch.tensor(np.tile(v,2),dtype=torch.float64) for v in active_limits()]
    rates=dict(joint_rate_rad_s=ACTION_CONTRACT["joint_rate_rad_s"],wrist_rate_m_s=ACTION_CONTRACT["translation_rate_m_s"],
               rotation_rate_rad_s=ACTION_CONTRACT["rotation_rate_rad_s"])
    geometries={s:surface.base.load_geometry(operation.ROOT,side=s) for s in ("left","right")}
    measured=seed.clone();maximum=dict(q_error_rad=0.,root_error_m=0.,local_roundtrip_error_m=0.,clipped_rows=0)
    min_separation=float("inf");min_clearance=float("inf")
    for ordinal in range(1,233):
        clear=[];whole=[]
        for hi,side in enumerate(("left","right")):
            g=geometries[side]
            poses=surface.base.fk_poses(g,measured[0,hi].numpy(),q[0,6*hi:6*hi+6].numpy())
            vertices=np.concatenate([surface.base.transform(v["vertices"],poses[n])
                for key in ("source_colliders","cooked_pads") for n,v in g[key].items()])
            clear.append(float(vertices[:,2].min()-.3));whole.append(vertices)
        gap=separation_lower(*whole)
        min_separation=min(min_separation,gap);min_clearance=min(min_clearance,min(clear))
        if gap<0 or min(clear)<.0021:raise ValueError("combined original whole-hand geometry not proven clear")

        action=actions[(ordinal-1)//8][None]
        requested=finger_pre_guard(command,action,q,torch.zeros((1,10),dtype=torch.float64),low,high,1/480,rates,policy_owned=True)
        _,target=surface_final_guard(command,requested,low,high,1/480)
        palm,angle=wrist_requested_targets(palms,angles,action,rates,1/480)
        projection=project_policy_wrist_targets(root_seed_w=seed,measured_base_w=measured,
            common_lift_offset_m=torch.zeros(1,dtype=torch.float64),measured_clearance_m=torch.tensor([clear],dtype=torch.float64),
            palm_offsets_m=palms,requested_palm_offsets_m=palm,angles_rad=angles,requested_angles_rad=angle,
            allowed=torch.ones(1,dtype=torch.bool),live=torch.ones(1,dtype=torch.bool))
        qe,re=target_errors(pair,target[0].numpy(),projection["projected_root_poses"][0].numpy(),ordinal)
        clipped=bool(projection["safe_z_clipped"].any()) or bool(projection["projection_reasons"].any())
        local=local_translation(projection["projected_root_poses"].reshape(2,7),seed.reshape(2,7)).numpy()
        roots=seed[0].numpy();roundtrip=[]
        for hi in range(2):
            rotation=Rotation.from_quat(roots[hi,[4,5,6,3]]).as_matrix()
            roundtrip.append(roots[hi,:3]+rotation@local[hi])
        local_error=float(np.max(abs(np.asarray(roundtrip)-projection["projected_root_poses"][0,:,:3].numpy())))
        maximum["q_error_rad"]=max(maximum["q_error_rad"],qe);maximum["root_error_m"]=max(maximum["root_error_m"],re)
        maximum["local_roundtrip_error_m"]=max(maximum["local_roundtrip_error_m"],local_error);maximum["clipped_rows"]+=int(clipped)
        emit(dict(ordinal=ordinal,policy_step=(ordinal-1)//8,substep=(ordinal-1)%8,source="CPU_ideal_tracking_not_native",
                  action24=action.numpy(),requested_fingers=requested.numpy(),projected_fingers=target.numpy(),
                  projected_roots=projection["projected_root_poses"].numpy(),mount_local_translation_m=local,
                  clearance_m=clear,inter_hand_separation_lower_m=gap,projection_clipped=clipped,q_error_rad=qe,root_error_m=re,local_error_m=local_error))
        # Explicit ideal feedback for arithmetic parity only. Never called SDK readback.
        q=target.clone();command=target.clone();measured=projection["projected_root_poses"].clone()
        palms=projection["palm_offsets_m"].clone();angles=projection["projected_angles"].clone()
    if maximum["q_error_rad"]>1e-7 or maximum["root_error_m"]>1e-8 or maximum["local_roundtrip_error_m"]>1e-12 or maximum["clipped_rows"]:
        raise ValueError("original action/projection parity failed: "+str(maximum))
    out.update(status="complete",cpu_rows=232,maximum=maximum,reset_seed_sha256=pair["reset_seed"]["sha256"],
               config_sha256=surface.base.digest(operation.ROOT/operation.CONFIG),
               left_proof_sha256=operation.left.PROOF_SHA,right_proof_sha256=operation.right.PROOF_SHA,
               minimum_inter_hand_lower_m=min_separation,minimum_full_table_clearance_m=min_clearance,action_contract=ACTION_CONTRACT,
               pending=["same-episode bilateral SDK tracking and contact","new native outer-exit evidence","full R0 and Teacher admission"])


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    out=dict(status="partial",new_physics_steps=0,ppo_updates=0,native_qualified=False,teacher_admission=False,
             source="CPU_ideal_tracking_not_native")
    with (a.output/"cpu_commands.jsonl").open("x") as f:
        def emit(row):f.write(json.dumps(surface.base.encode_owned(row),allow_nan=False)+"\n");f.flush()
        try:run(out,emit)
        except Exception as exc:out.update(status="blocked",reason=type(exc).__name__+": "+str(exc))
        finally:f.flush();os.fsync(f.fileno())
    out.update(elapsed_s=time.monotonic()-start,script_sha256=surface.base.digest(__file__))
    with (a.output/"result.json").open("x") as f:
        json.dump(out,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps(out));return int(out["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
