"""New right-only original-pad approach. Existing plans remain immutable."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts import prepare_dh116_right_pinch_owned_contact as previous
from scripts import audit_dh116_effective_pinch_target_path as path
from scripts import solve_dh116_pinch_effective_targets as target
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.linear_coupling import expand
from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import prepare_reset_state
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT,validate_actions
ROOT=Path(__file__).resolve().parents[1]
VERSION="dh116-effective-right-pinch-plan-v1"
CONFIG="configs/dh116_tasks/flexible_effective_right_pinch_v1.json"
PROOF=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/pinch-effective-path-20261114-run03/result.json")
PROOF_SHA="03b70ba97423665eac710ddd0fe3e868282885428efe1e0775bf304ccf9aa280"

def derive_pair():
    old=previous.load_pair(ROOT)
    proof=decode_owned(previous.previous.read_locked(PROOF,PROOF_SHA))
    parent=decode_owned(previous.previous.read_locked(path.PARENT,path.PARENT_SHA))
    r=proof["sides"]["right"]
    if (proof["version"]!=path.VERSION or proof["parent_sha256"]!=path.PARENT_SHA
        or not r["static_path_passed"] or r["path_samples"]!=1025
        or not r["original_wrist_limits_preserved"] or r["bow_world_x_m"]!=.0001
        or r["worst"]["table_m"]<.0021 or r["worst"]["other_m"]<0
        or r["worst"]["depth_m"]<-.0003 or r["initial"]["minimum_m"]<.00005
        or proof["native_ready"] or proof["teacher_admission"]
        or parent["sides"]["right"]["search"]["best"]!=r["end"]):
        raise ValueError("exact independently passing static path and explicit native-unqualified scope required")
    candidate=deepcopy(old["right"]["candidate"])
    candidate.update(candidate_id="right-effective-pad-bowed-approach-v1",
        base_pose_wxyz=list(r["start"]["root_pose_wxyz"]),end_base_pose_wxyz=list(r["end"]["root_pose_wxyz"]),
        active_q_rad=list(r["start"]["q_rad"]),active_delta_rad=(np.array(r["end"]["q_rad"])-r["start"]["q_rad"]).tolist(),
        active_velocity_rad_s=list(r["physical_joint_velocity_rad_s"]),
        root_translation_delta_m=[0.,0.,-.0005],bow_world_x_m=.0001,
        root_velocity_knots_world_m_s=r["root_velocity_world_m_s"],
        static_path_passed=True,native_ready=False,teacher_admitted=False)
    seed=old["reset_seed"];joints=deepcopy(seed["joint_positions"])
    joints[1]=expand(candidate["active_q_rad"]).tolist()
    roots=dict(zip(("left","right"),deepcopy(seed["hand_roots"])))
    roots["right"]=candidate["base_pose_wxyz"]
    reset=prepare_reset_state(hand_roots=roots,origin=seed["origin"],root_xy=seed["root_xy"],
        initial_offset_m=seed["initial_offset_m"],joint_positions=joints,joint_velocities=seed["joint_velocities"],
        source_description="independent right original-pad bounded path; original left and free rod; native unqualified")
    value=deepcopy(old)
    value.update(version=VERSION,scope="right_original_effective_pad_approach_candidate_not_teacher",
        reset_seed=reset,native_qualified=False,teacher_admission=False,
        sources={**old["sources"],str(PROOF):PROOF_SHA,str(path.PARENT):path.PARENT_SHA},
        limitations=["Right-only contact diagnostic. Original left requests zero.",
            "Static bounds do not qualify actual tracking, moving rod, contact, Hold or Teacher.",
            "New100um World-X piecewise bow, unchanged finger rates and original wrist limits.",
            "Original234-step schedule is not5second Hold; no long training or retry."])
    value["right"]["candidate"]=candidate
    value["right"]["path"]=dict(source=str(PROOF),sha256=PROOF_SHA,static_path_passed=True)
    return value

def load_pair(project=ROOT):
    value=json.loads((Path(project)/CONFIG).read_bytes())
    if value!=derive_pair():raise ValueError("frozen effective-right reset/path differs")
    return value

def action_schedule(packet=None):
    expected=derive_pair();p=expected if packet is None else packet
    if p!=expected:raise ValueError("frozen independent right plan")
    c=p["right"]["candidate"];out=torch.zeros((29,24),dtype=torch.float64)
    out[6:26,6:12]=torch.tensor(c["active_delta_rad"],dtype=out.dtype)/(160/480)/ACTION_CONTRACT["joint_rate_rad_s"]
    out[6:26,15:18]=torch.tensor(c["root_velocity_knots_world_m_s"],dtype=out.dtype)/ACTION_CONTRACT["translation_rate_m_s"]
    validate_actions(out)
    return out

def expected_boundary(packet,ordinal):
    if type(ordinal) is not int or not 0<=ordinal<=232:raise ValueError("exact bounded physical ordinal")
    f=float(np.clip((ordinal-48)/160,0,1));c=packet["right"]["candidate"]
    offsets=np.array([path.bow(f,c["bow_world_x_m"]),0.,-.0005*f])
    roots=np.array(packet["reset_seed"]["reset_root_pose_w"],float);roots[1,:3]+=offsets
    return dict(fraction=f,right_offset_m=offsets,root_poses_w=roots,
        right_q_rad=np.asarray(c["active_q_rad"])+f*np.asarray(c["active_delta_rad"]))
