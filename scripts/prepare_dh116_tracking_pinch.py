"""Frozen tracking-aware right Pinch reset and same-amplitude approach."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts import prepare_dh116_effective_right_pinch as previous
from scripts import audit_dh116_pinch_tracking_target_path as proof_module
from go2_pvcnn.control.dh116.linear_coupling import expand
from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import prepare_reset_state
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT,validate_actions
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

ROOT=Path(__file__).resolve().parents[1]
VERSION="dh116-tracking-aware-pinch-plan-v1"
CONFIG="configs/dh116_tasks/flexible_tracking_pinch_v1.json"
PROOF=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-tracking-target-20261114/path-04/result.json")
PROOF_SHA="b4a4d2a95ae97437c182a800f01f0862180e746f90149adc924e3b4fa71da600"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def derive_pair():
    old=previous.load_pair(ROOT)
    if sha(PROOF)!=PROOF_SHA:raise ValueError("frozen complete path proof required")
    proof=decode_owned(json.loads(PROOF.read_bytes()))
    right=proof["continuous_static_path"]["sides"]["right"]
    observed=proof["observed_path"];interhand=proof["continuous_interhand"]
    if (proof["status"]!="complete" or proof["version"]!="dh116-pinch-tracking-target-path-cpu-v1"
        or proof["profile"]!=proof_module.PROFILE or proof["script_sha256"]!=sha(proof_module.__file__)
        or not right["static_path_passed"] or right["path_samples"]!=1025
        or right["initial"]["minimum_m"]<.00005 or right["worst"]["table_m"]<.0021
        or right["worst"]["other_m"]<0 or right["worst"]["depth_m"]<-.0003
        or observed["samples"]!=232 or not observed["passed"] or observed["first_failure"] is not None
        or interhand["samples"]!=1025 or not interhand["passed"]
        or proof["native_ready"] or proof["teacher_admission"] or proof["new_physics_steps"] or proof["ppo_updates"]):
        raise ValueError("complete original-guard CPU proof, never native qualification")
    c=deepcopy(old["right"]["candidate"])
    delta=np.asarray(c["active_delta_rad"])
    if not np.allclose(np.asarray(right["end"]["q_rad"])-right["start"]["q_rad"],delta,rtol=0,atol=1e-14):
        raise ValueError("original finger action delta changed")
    c.update(candidate_id="tracking-aware-right-pinch-plateau-v1",
        base_pose_wxyz=list(right["start"]["root_pose_wxyz"]),
        end_base_pose_wxyz=list(right["end"]["root_pose_wxyz"]),
        active_q_rad=list(right["start"]["q_rad"]),
        root_velocity_knots_world_m_s=right["root_velocity_world_m_s"],
        bow_world_x_m=.0001,bow_profile=proof_module.PROFILE["version"],
        static_path_passed=True,native_ready=False,teacher_admitted=False)
    rates=np.asarray(c["root_velocity_knots_world_m_s"])
    if rates.shape!=(20,3) or abs(rates[:,0]).max()>.00114+1e-15:
        raise ValueError("same-amplitude and maximum-speed wrist path required")
    seed=old["reset_seed"];q=deepcopy(seed["joint_positions"])
    q[1]=expand(c["active_q_rad"]).tolist()
    roots=dict(zip(("left","right"),deepcopy(seed["hand_roots"])));roots["right"]=c["base_pose_wxyz"]
    reset=prepare_reset_state(hand_roots=roots,origin=seed["origin"],root_xy=seed["root_xy"],
        initial_offset_m=seed["initial_offset_m"],joint_positions=q,joint_velocities=seed["joint_velocities"],
        source_description="independent tracking-aware right Pinch; original free rod and left; native unqualified")
    p=deepcopy(old)
    p.update(version=VERSION,scope="tracking_aware_right_original_pad_candidate_not_teacher",
        reset_seed=reset,native_qualified=False,teacher_admission=False,
        sources={**old["sources"],str(PROOF):PROOF_SHA,str(proof_module.SOURCE):proof_module.SOURCE_SHA},
        limitations=["CPU empirical scenarios do not qualify new native dynamics or contact.",
            "Original finger deltas; World-X<=100um and max1.14mm/s; unchanged original Z motion.",
            "234 SDK steps include2 setup,232 measured; not5s Hold; no Teacher qualification."])
    p["right"]["candidate"]=c
    p["right"]["path"]=dict(source=str(PROOF),sha256=PROOF_SHA,static_path_passed=True,
        observed_scenarios_passed=True,unobserved_dynamics_qualified=False)
    return p

def load_pair(project=ROOT):
    p=json.loads((Path(project)/CONFIG).read_bytes())
    if p!=derive_pair():raise ValueError("frozen tracking-aware plan differs")
    return p

def action_schedule(packet=None):
    expected=derive_pair();p=expected if packet is None else packet
    if p!=expected:raise ValueError("frozen independent tracking plan")
    c=p["right"]["candidate"];out=torch.zeros((29,24),dtype=torch.float64)
    out[6:26,6:12]=torch.tensor(c["active_delta_rad"],dtype=out.dtype)/(160/480)/ACTION_CONTRACT["joint_rate_rad_s"]
    out[6:26,15:18]=torch.tensor(c["root_velocity_knots_world_m_s"],dtype=out.dtype)/ACTION_CONTRACT["translation_rate_m_s"]
    validate_actions(out)
    return out

def expected_boundary(packet,ordinal):
    if type(ordinal) is not int or not 0<=ordinal<=232:raise ValueError("bounded234-step physical clock")
    f=float(np.clip((ordinal-48)/160,0,1));c=packet["right"]["candidate"]
    offsets=np.array([proof_module.plateau_bow(f,.0001),0.,-.0005*f])
    roots=np.array(packet["reset_seed"]["reset_root_pose_w"],float);roots[1,:3]+=offsets
    return dict(fraction=f,right_offset_m=offsets,root_poses_w=roots,
        right_q_rad=np.asarray(c["active_q_rad"])+f*np.asarray(c["active_delta_rad"]))
