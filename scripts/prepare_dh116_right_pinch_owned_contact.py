"""Compile right-owned geometry into original Reset/24D contracts; never launches."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from scripts import prepare_dh116_pinch_margin_contact as previous
from scripts import solve_dh116_right_pinch_surface_registration as surface
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, expand
from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import prepare_reset_state

VERSION="dh116-right-pinch-owned-contact-plan-v1"
PROOF=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-owned-approach-v2-20261113-sQz6EbUf/run01/result.json")
PROOF_SHA="28f4fdfc138daf800d372d7f3882c60e3953b16779c4fcb3032801cfff5ed8da"
CONFIG="configs/dh116_tasks/flexible_right_pinch_owned_approach_v1.json"
ROOT=Path(__file__).resolve().parents[1]
SETUP_STEPS,MEASURED_STEPS,TOTAL_STEPS=2,232,234


def derive_pair():
    old=previous.load_pair()
    proof=surface.base.decode_owned(previous.read_locked(PROOF,PROOF_SHA))
    if (proof["status"]!="complete" or proof["static_path_passed"] is not True or proof["path_samples"]!=161
        or proof["native_ready"] is not False or proof["teacher_admission"] is not False
        or proof["minimum_table_lower_m"]<.0021 or proof["minimum_other_lower_m"]<0
        or proof["minimum_depth_lower_m"]<-.0003 or proof["start_root_lift_m"]!=.0005):
        raise ValueError("complete frozen CPU path required, without native qualification")
    candidate=dict(candidate_id="right-owned-station-approach-v1",side="right",independent_style="pinch",
        active_joint_names=list(ACTIVE),active_q_rad=proof["selected_start_q_rad"].tolist(),
        active_delta_rad=proof["delta_rad"].tolist(),active_velocity_rad_s=proof["physical_velocity_rad_s"].tolist(),
        root_translation_delta_m=[0.,0.,-.0005],command_duration_s=160/480,
        base_pose_wxyz=list(proof["root_wxyz"]),end_base_pose_wxyz=list(proof["root_wxyz"]),
        required_pads=list(surface.PADS),native_ready=False,teacher_admitted=False,static_path_passed=True)
    candidate["base_pose_wxyz"][2]+=.0005
    seed=old["reset_seed"];joints=deepcopy(seed["joint_positions"]);joints[1]=expand(candidate["active_q_rad"]).tolist()
    roots=dict(zip(("left","right"),deepcopy(seed["hand_roots"])));roots["right"]=candidate["base_pose_wxyz"]
    reset=prepare_reset_state(hand_roots=roots,origin=seed["origin"],root_xy=seed["root_xy"],
        initial_offset_m=seed["initial_offset_m"],joint_positions=joints,joint_velocities=seed["joint_velocities"],
        source_description="right-owned static approach; original left initial pose and free rod; native unqualified")
    return dict(version=VERSION,scope="independent_right_contact_candidate_not_teacher",left_pose_sha256=old["left_pose_sha256"],
        left=deepcopy(old["left"]),right=dict(candidate=candidate,source_sha256=deepcopy(old["right"]["source_sha256"]),
        path=dict(source=str(PROOF),sha256=PROOF_SHA,static_path_passed=True)),rod_seed=deepcopy(old["rod_seed"]),
        reset_seed=reset,wrist_clearance_floor_m=.0021,required_pad_min_gap_m=-.0003,
        native_qualified=False,teacher_admission=False,sources={**old["sources"],str(PROOF):PROOF_SHA},
        schedule=deepcopy(old["schedule"]),limitations=[
            "Right contact diagnostic only; left commands remain zero.",
            "CPU proof is not actual SDK/Reset/tracking/contact evidence.",
            "Existing zero-wrist validator needs a separately tested scoped target validator before native recipe."])


def load_pair(project=ROOT):
    import json
    value=json.loads((Path(project)/CONFIG).read_text())
    if value!=derive_pair():raise ValueError("frozen right-owned Reset differs")
    return value


def action_schedule(packet=None):
    import torch
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT,validate_actions
    p=derive_pair() if packet is None else packet
    c=p["right"]["candidate"]
    action=torch.zeros((29,24),dtype=torch.float64)
    action[6:26,6:12]=torch.tensor(c["active_delta_rad"],dtype=action.dtype)/(160/480)/ACTION_CONTRACT["joint_rate_rad_s"]
    action[6:26,15:18]=torch.tensor(c["root_translation_delta_m"],dtype=action.dtype)/(160/480)/ACTION_CONTRACT["translation_rate_m_s"]
    validate_actions(action)
    return action


def expected_boundary(packet,ordinal):
    if type(ordinal) is not int or not 0<=ordinal<=232:raise ValueError("bounded physical boundary required")
    f=float(np.clip((ordinal-48)/160,0,1));c=packet["right"]["candidate"]
    root=np.array(packet["reset_seed"]["reset_root_pose_w"],float)
    root[1,:3]+=f*np.asarray(c["root_translation_delta_m"])
    return dict(fraction=f,right_q_rad=np.asarray(c["active_q_rad"])+f*np.asarray(c["active_delta_rad"]),
                root_poses_w=root)
