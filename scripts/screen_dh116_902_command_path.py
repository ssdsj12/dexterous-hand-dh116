"""CPU replay of the existing 480 Hz command kernels, not a dynamics simulator."""
import numpy as np
import torch
from go2_pvcnn.control.dh116.linear_coupling import active_limits
from go2_pvcnn.control.dh116.tabletop_palm_mount import local_translation, CONTRACT as MOUNT_CONTRACT
from go2_pvcnn.control.dh116.tabletop_command_kernels import (
    finger_pre_guard, surface_final_guard, wrist_requested_targets)
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import project_policy_wrist_targets
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT, validate_actions

DT = 1 / 480


def command_step(*, action, backend_targets, measured_q, seed, measured_roots,
                 clearance, palms, angles, mount_anchor_poses):
    """Require explicit measured inputs; never synthesize future q or clearance."""
    shapes = dict(action=(1,24),backend_targets=(1,12),measured_q=(1,12),
                  seed=(1,2,7),measured_roots=(1,2,7),clearance=(1,2),
                  palms=(1,2,3),angles=(1,2,3),mount_anchor_poses=(1,2,7))
    values = locals().copy()
    tensors = {}
    for key,shape in shapes.items():
        a=np.asarray(values[key],dtype=np.float64)
        if a.shape!=shape or not np.isfinite(a).all():
            raise ValueError(f"finite measured {key} {shape} required")
        tensors[key]=torch.from_numpy(a.copy())
    anchors=tensors["mount_anchor_poses"]
    if not torch.allclose(torch.linalg.vector_norm(anchors[...,3:],dim=-1),
                          torch.ones((1,2),dtype=torch.float64),atol=1e-5,rtol=0):
        raise ValueError("unit mount anchor quaternion required")
    a=tensors["action"];command=tensors["backend_targets"]
    validate_actions(a)
    low,high=(torch.from_numpy(np.tile(v,2)) for v in active_limits())
    rates=dict(joint_rate_rad_s=ACTION_CONTRACT["joint_rate_rad_s"],
        wrist_rate_m_s=ACTION_CONTRACT["translation_rate_m_s"],
        rotation_rate_rad_s=ACTION_CONTRACT["rotation_rate_rad_s"])
    # policy_owned ignores mimic compensation; the native safety guard still checks it.
    target=finger_pre_guard(command,a,tensors["measured_q"],torch.zeros((1,10),dtype=torch.float64),
        low,high,DT,rates,policy_owned=True)
    _,target=surface_final_guard(command,target,low,high,DT)
    requested_palms,requested_angles=wrist_requested_targets(tensors["palms"],tensors["angles"],a,rates,DT)
    projection=project_policy_wrist_targets(root_seed_w=tensors["seed"],
        measured_base_w=tensors["measured_roots"],common_lift_offset_m=torch.zeros(1,dtype=torch.float64),
        measured_clearance_m=tensors["clearance"],palm_offsets_m=tensors["palms"],
        requested_palm_offsets_m=requested_palms,angles_rad=tensors["angles"],
        requested_angles_rad=requested_angles,allowed=torch.ones(1,dtype=torch.bool),
        live=torch.ones(1,dtype=torch.bool))
    local_targets=torch.stack([local_translation(projection["projected_root_poses"][:,i],
                             anchors[:,i]) for i in range(2)],dim=1)
    if bool((local_targets.abs()>MOUNT_CONTRACT["limit_m"]).any()):
        raise ValueError("projected wrist translation exceeds original carriage limits")
    return dict(finger_targets=target.numpy().copy(),
        submitted_finger_targets=target.float().numpy().copy(),
        requested_palms=requested_palms.numpy().copy(),requested_angles=requested_angles.numpy().copy(),
        projection={k:v.numpy().copy() if isinstance(v,torch.Tensor) else v for k,v in projection.items()},
        source="original_command_kernels_with_explicit_measured_inputs",
        local_carriage_targets=local_targets.numpy().copy(),
        submitted_wrist_targets=torch.cat((local_targets,projection["projected_angles"]),dim=-1).float().numpy().copy(),
        context="single_live_policy_owned_zero_common_lift",
        evaluates_native_safety=False,predicts_actual_tracking=False,
        native_qualification=False,teacher_admission=False)



def replay_errors(rebuilt, request, projection, submitted, submitted_wrist):
    pairs={
        "submitted_fingers":(rebuilt["submitted_finger_targets"],submitted),
        "submitted_wrist":(rebuilt["submitted_wrist_targets"],submitted_wrist),
        "finger":(rebuilt["finger_targets"],request["requested_finger_targets_rad"]),
        "palms":(rebuilt["requested_palms"],request["requested_palm_offsets_m"]),
        "angles":(rebuilt["requested_angles"],request["requested_angles_rad"]),
        "projected_root":(rebuilt["projection"]["projected_root_poses"],projection["projected_root_poses"]),
        "projected_angles":(rebuilt["projection"]["projected_angles"],projection["projected_angles"]),
        "projected_palms":(rebuilt["projection"]["palm_offsets_m"],projection["palm_offsets_m"]),
    }
    errors={}
    for key,(a,b) in pairs.items():
        a,b=np.asarray(a),np.asarray(b)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("matching finite replay fields required")
        errors[key]=float(np.max(abs(a-b)))
    return errors

def audit_boundary(receipt, grasp):
    """Replay the eight saved physical transitions at a sealed natural boundary."""
    from pathlib import Path
    from scripts.screen_dh116_902_flexible_contact import boundary
    from scripts.verify_dh116_902_flexible_reset import load_candidate,sha
    from scripts.diagnose_dh116_teacher_initialization import digest
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
    from go2_pvcnn.control.dh116.tabletop_joint_target_readback import WRIST_LINEAR,WRIST_ANGULAR
    raw,_,_,sdk=boundary(receipt,grasp)
    rows=[decode_owned(v) for v in raw]
    _,config,_=load_candidate(grasp)
    seed=np.array([[config["hands"][s]["pose_wxyz"] for s in ("left","right")]])
    checks=[]
    for old,row in zip(rows[:-1],rows[1:]):
        if row["physics_step"]!=old["physics_step"]+1:
            raise ValueError("contiguous measured physical rows required")
        pre=row["action_readback"]["pre"]["hands"]
        command=[];q=[]
        for side in ("left","right"):
            h=pre[side];ids=h["active_indices"]
            if [h["joint_names"][i] for i in ids]!=list(ACTIVE):
                raise ValueError("original active joint order required")
            command.append(h["backend_targets"][:,ids]);q.append(h["q"][:,ids])
        poses=old["sdk_hand_geometry"]["poses"]
        roots=np.array([[np.r_[poses[s+".base_link"][0],poses[s+".base_link"][1][[3,0,1,2]]]
                         for s in ("left","right")]])
        request=row["action_readback"]["request"]
        rebuilt=command_step(action=request["action24"],backend_targets=np.concatenate(command,axis=1),
            measured_q=np.concatenate(q,axis=1),seed=seed,measured_roots=roots,
            clearance=np.asarray(old["sdk_hand_geometry"]["table_clearance_m"]).reshape(1,2),
            mount_anchor_poses=np.stack([pre[s]["mount_anchor_pose_w"] for s in ("left","right")],axis=1),
            palms=old["wrist_projection"]["palm_offsets_m"],angles=old["wrist_projection"]["projected_angles"])
        submitted=np.concatenate([row["action_readback"]["post"]["hands"][side]["backend_targets"][:,
            row["action_readback"]["post"]["hands"][side]["active_indices"]] for side in ("left","right")],axis=1)
        post=row["action_readback"]["post"]["hands"]
        submitted_wrist=np.stack([post[s]["backend_targets"][:,
            [post[s]["joint_names"].index(n) for n in WRIST_LINEAR+WRIST_ANGULAR]]
            for s in ("left","right")],axis=1)
        errors=replay_errors(rebuilt,request,row["wrist_projection"],submitted,submitted_wrist)
        checks.append(dict(physics_step=row["physics_step"],maximum_errors=errors,
            passed=max(errors.values())<=1e-12))
    return dict(version="dh116-902-command-replay-v1",grasp=grasp,boundary_sdk=sdk,
        source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])):sha(p) for p in (
            Path(__file__),
            Path(__file__).resolve().parents[1]/"src/go2_pvcnn/control/dh116/tabletop_command_kernels.py",
            Path(__file__).resolve().parents[1]/"src/go2_pvcnn/control/dh116/tabletop_palm_mount.py",
            Path(__file__).resolve().parents[1]/"src/go2_pvcnn/control/dh116/tabletop_deformable_wrist_projection.py")},
        outer_sha256=sha(receipt),last_row_sha256=sha(receipt.parent/"native"/f"{grasp}_control_{sdk}.json"),
        row_content_sha256=digest(raw[-1]),checks=checks,
        all_passed=len(checks)==8 and all(c["passed"] for c in checks),
        native_runs=0,predicts_actual_tracking=False,teacher_admission=False)


def main(argv=None):
    import argparse
    from pathlib import Path
    from scripts.diagnose_dh116_teacher_initialization import write_new
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--grasp",choices=("pinch","tripod"),required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv);result=audit_boundary(a.receipt,a.grasp);write_new(a.output,result)
    return 0 if result["all_passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
