"""Independent left Tripod reset and original24D schedule; never starts physics."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts import prepare_dh116_right_pinch_owned_contact as previous
from scripts.prepare_dh116_pinch_margin_contact import read_locked
from scripts.audit_dh116_tripod_sdk_path import target_at
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,expand
from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import prepare_reset_state
from go2_pvcnn.control.dh116.tabletop_surface_contract import validate_actions,ACTION_CONTRACT

VERSION="dh116-left-tripod-owned-approach-plan-v1"
ROOT=Path(__file__).resolve().parents[1]
CONFIG="configs/dh116_tasks/flexible_left_tripod_owned_approach_v1.json"
BASE=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166")
PROOF=BASE/"tripod-sdk-path-20261114-intrinsic_linear-run04/result.json"
PROOF_SHA="f0726bb65f7ed3d2e645fb9d93226dd0e83f2100ee1cc928c95fa60fc74156ae"
NEAR=BASE/"tripod-hard-constraints-20261114-run01/result.json"
NEAR_SHA="a484586dd4e08cbdfedd740fba3df53b4bcf9d636b16a7162e624422665fb1df"
QUIET=BASE/"tripod-vertical-sdf-20261114-run01/result.json"
QUIET_SHA="d0040fc6f282687b16bbc1336f7cbfdabc54bac52fc6b696facdf720e6ec4ba1"
SETUP_STEPS,MEASURED_STEPS,TOTAL_STEPS=2,232,234
REQUIRED_LEFT=("finger14_link","finger23_link","finger33_link")
QUIET_LIMITS=dict(samples=8,max_node_speed_m_s=.0001,max_node_drift_m=.00001,
                  max_joint_angular_speed_rad_s=.001,max_mount_linear_speed_m_s=.0001,
                  require_table_loaded_every_solver_substep=True,require_no_hand_contact=True)

def derive_pair():
    old=previous.load_pair()
    proof=read_locked(PROOF,PROOF_SHA);near=read_locked(NEAR,NEAR_SHA);quiet=read_locked(QUIET,QUIET_SHA)
    p=proof["plan"];start=near["parent_candidate"];end=near["final_packet"]["measurement"]["candidate"]
    if (proof["status"]!="complete" or not proof["target_path"]["proved"]
            or proof["target_path"]["required_rod_minimum_threshold_m"]!=-.00005
            or proof["summary"]["projection_changed_steps"]!=0 or p["physics_steps"]!=200
            or p["policy_steps"]!=25 or p["mode"]!="intrinsic_linear" or p["side"]!="left"
            or p["action_contract"]!=ACTION_CONTRACT or start["independent_style"]!="tripod"
            or proof["native_ready"] is not False or proof["actual_tracking_validated"] is not False
            or p["seed_pose_wxyz"]!=start["base_pose_wxyz"] or p["q0_rad"]!=start["initial_q_rad"]
            or p["end_pose_wxyz"]!=end["base_pose_wxyz"] or p["q1_rad"]!=end["active_q_rad"]):
        raise ValueError("frozen independent Tripod reference path required, never native qualification")
    # The full private vertical proof is checked by SHA; the current source keys
    # remain independently checked by the original scene/builder validators.
    candidate=dict(candidate_id="left-tripod-owned-approach-v1",side="left",independent_style="tripod",
        independent_seed_sha256=start["independent_seed_sha256"],active_joint_names=list(ACTIVE),
        active_q_rad=p["q0_rad"],active_delta_rad=(np.array(p["q1_rad"])-p["q0_rad"]).tolist(),
        base_pose_wxyz=p["seed_pose_wxyz"],end_base_pose_wxyz=p["end_pose_wxyz"],
        source_sha256=deepcopy(near["geometry_sources"]),required_pads=list(REQUIRED_LEFT),
        command_duration_s=200/480,native_ready=False,teacher_admitted=False)
    oldseed=old["reset_seed"];joints=deepcopy(oldseed["joint_positions"]);joints[0]=expand(p["q0_rad"]).tolist()
    roots=dict(zip(("left","right"),deepcopy(oldseed["hand_roots"])));roots["left"]=p["seed_pose_wxyz"]
    seed=prepare_reset_state(hand_roots=roots,origin=oldseed["origin"],root_xy=oldseed["root_xy"],
        initial_offset_m=oldseed["initial_offset_m"],joint_positions=joints,
        joint_velocities=deepcopy(oldseed["joint_velocities"]),
        source_description="independent left Tripod reference; original right Pinch parked and same free rod; native unqualified")
    return dict(version=VERSION,scope="left_tripod_moving_wrist_contact_diagnostic_not_teacher",
        left_pose_sha256=old["left_pose_sha256"],left=dict(candidate=candidate),right=deepcopy(old["right"]),
        rod_seed=deepcopy(old["rod_seed"]),reset_seed=seed,wrist_clearance_floor_m=.0021,
        required_pad_min_gap_m=-.0003,required_pads=dict(left=list(REQUIRED_LEFT),right=["finger14_link","finger23_link"]),
        inactive_side_role="right_independent_pinch_zero_action_not_tripod_validation",
        native_qualified=False,teacher_admission=False,
        sources={**old["sources"],str(PROOF):PROOF_SHA,str(NEAR):NEAR_SHA,str(QUIET):QUIET_SHA},
        original_24d_plan=deepcopy(p),quiet_admission_limits=deepcopy(QUIET_LIMITS),
        schedule=dict(setup_steps=2,quiet_steps=8,closure_steps=200,hold_steps=24,total_steps=234,
            policy_steps=29,physics_hz=480,control_hz=60,ppo_updates=0,environments=1),
        limitations=["8-step quiet admission must be measured in THIS new instance before any approach.",
            "Positive final SDF gaps are not actual contact; no contact/grasp/Teacher qualification.",
            "CPU path assumes exact target tracking; live original safety and source/cooked guards remain.",
            "Source17 plus locked5 is not a claim that all non-pad cooked shapes are available.",
            "Right USD/link/SDF identity remains independent; zero right action does not validate right Tripod.",
            "24 hold steps are50ms observation, not the native5second success criterion."])

def load_pair(project=ROOT):
    value=json.loads((Path(project)/CONFIG).read_text())
    if value!=derive_pair():raise ValueError("frozen Tripod reset/schedule differs")
    return value

def action_schedule(packet=None):
    p=derive_pair() if packet is None else packet
    if p!=derive_pair():raise ValueError("frozen Tripod schedule input differs")
    out=torch.zeros((29,24),dtype=torch.float64)
    out[1:26]=torch.tensor(p["original_24d_plan"]["actions"],dtype=out.dtype)
    validate_actions(out)
    return out

def expected_boundary(packet,ordinal):
    if type(ordinal) is not int or not 0<=ordinal<=232:raise ValueError("bounded integer physical ordinal0..232")
    f=float(np.clip((ordinal-8)/200,0,1));p=packet["original_24d_plan"]
    pose,q=target_at(p,f)
    roots=np.array(packet["reset_seed"]["reset_root_pose_w"],dtype=float);roots[0]=pose
    angle=f*np.asarray(p["intrinsic_knots_rad"][-1])
    offsets=np.zeros((2,3));offsets[0]=pose[:3]-np.array(p["seed_pose_wxyz"][:3])
    angles=np.zeros((2,3));angles[0]=angle
    return dict(fraction=f,phase="quiet" if ordinal<=8 else "approach" if ordinal<=208 else "observe",
        left_q_rad=q,root_poses_w=roots,palm_offsets_m=offsets,intrinsic_angles_rad=angles)
