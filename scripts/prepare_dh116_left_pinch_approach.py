"""Independent bilateral closure: certified left approach, unchanged right path."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from scripts import prepare_dh116_tracking_pinch as previous
from scripts import audit_dh116_left_pinch_tracking_path as proof_module
from go2_pvcnn.control.dh116.linear_coupling import expand
from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import prepare_reset_state
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT,validate_actions
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

ROOT=Path(__file__).resolve().parents[1]
VERSION="dh116-left-pinch-raised-approach-plan-v1"
CONFIG="configs/dh116_tasks/flexible_left_pinch_approach_v1.json"
PROOF=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/left-pinch-tracking-20261114/path-02/result.json")
PROOF_SHA="a5ca10a1f612a20e216b545ccbd71a7aeeceb56117ddddb1101f4e7dfa1b4d74"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def derive_pair():
    old=previous.load_pair(ROOT)
    if sha(PROOF)!=PROOF_SHA:raise ValueError("frozen left complete proof SHA")
    proof=decode_owned(json.loads(PROOF.read_bytes()))
    checked=proof_module.qualification(proof)
    if (proof["status"]!="complete" or proof["profile"]!=proof_module.APPROACH_CONFIG
        or proof["script_sha256"]!=sha(proof_module.__file__)
        or not checked["conditional_path_passed"] or proof["qualification"]!=checked
        or proof["new_physics_steps"] or proof["ppo_updates"] or proof["teacher_admission"]):
        raise ValueError("complete independent CPU proof required, never native success")
    row=proof["continuous_static_path"]["sides"]["left"]
    if (row["initial"]["minimum_m"]<.00005 or row["worst"]["table_m"]<.0021
        or row["worst"]["other_m"]<0 or row["worst"]["depth_m"]<-.0003):
        raise ValueError("original geometry guard")
    c=deepcopy(old["left"]["candidate"]);delta=np.asarray(c["active_delta_rad"])
    if not np.allclose(np.asarray(row["end"]["q_rad"])-row["start"]["q_rad"],delta,atol=1e-14,rtol=0):
        raise ValueError("original left finger increment changed")
    c.update(candidate_id="left-pinch-raised-approach-v1",static_reset_only=False,
        base_pose_wxyz=list(row["start"]["root_pose_wxyz"]),
        end_base_pose_wxyz=list(row["end"]["root_pose_wxyz"]),
        active_q_rad=list(row["start"]["q_rad"]),root_translation_delta_m=[0.,0.,-.0015],
        root_velocity_knots_world_m_s=row["root_velocity_world_m_s"],
        static_path_passed=True,native_ready=False,teacher_admitted=False)
    rates=np.asarray(c["root_velocity_knots_world_m_s"])
    if rates.shape!=(20,3) or np.max(abs(rates))>ACTION_CONTRACT["translation_rate_m_s"]:
        raise ValueError("unchanged original wrist speed cap")
    np.testing.assert_allclose(rates,np.tile([0.,0.,-.0045],(20,1)),atol=1e-12,rtol=0)
    seed=old["reset_seed"];q=deepcopy(seed["joint_positions"]);q[0]=expand(c["active_q_rad"]).tolist()
    roots=dict(zip(("left","right"),deepcopy(seed["hand_roots"])));roots["left"]=c["base_pose_wxyz"]
    reset=prepare_reset_state(hand_roots=roots,origin=seed["origin"],root_xy=seed["root_xy"],
        initial_offset_m=seed["initial_offset_m"],joint_positions=q,joint_velocities=seed["joint_velocities"],
        source_description="independent left raised approach and original right; unqualified native candidate")
    p=deepcopy(old);p["left"]["candidate"]=c
    p["left"]["path"]=dict(source=str(PROOF),sha256=PROOF_SHA,static_path_passed=True,observed_scenarios_passed=True,
        unobserved_dynamics_qualified=False)
    p.update(version=VERSION,scope="bilateral_pinch_own_left_approach_not_teacher",reset_seed=reset,
        native_qualified=False,teacher_admission=False,
        sources={**old["sources"],str(PROOF):PROOF_SHA,str(proof_module.SOURCE):proof_module.SOURCE_SHA},
        limitations=["CPU held-left residual transfer is not dynamic qualification.",
            "New left World-Z start+1.5mm and descent4.5mm/s; original finger delta and unchanged right path.",
            "234 SDK steps include2 setup; no5s Hold or Teacher qualification."])
    return p

def load_pair(project=ROOT):
    p=json.loads((Path(project)/CONFIG).read_bytes())
    if p!=derive_pair():raise ValueError("frozen independent bilateral approach")
    return p

def action_schedule(packet=None):
    expected=derive_pair();p=expected if packet is None else packet
    if p!=expected:raise ValueError("foreign bilateral approach")
    out=previous.action_schedule().clone();c=p["left"]["candidate"]
    out[6:26,:6]=torch.tensor(c["active_delta_rad"],dtype=out.dtype)/(160/480)/ACTION_CONTRACT["joint_rate_rad_s"]
    out[6:26,12:15]=torch.tensor(c["root_velocity_knots_world_m_s"],dtype=out.dtype)/ACTION_CONTRACT["translation_rate_m_s"]
    validate_actions(out);return out

def expected_boundary(packet,ordinal):
    if type(ordinal) is not int or not 0<=ordinal<=232:raise ValueError("bounded234-step clock")
    # Previous function uses the packet's right candidate; retain exactly.
    e=previous.expected_boundary(packet,ordinal);f=e["fraction"];c=packet["left"]["candidate"]
    offset=np.array([0.,0.,-.0015*f]);e["root_poses_w"][0,:3]+=offset
    e.update(left_offset_m=offset,left_q_rad=np.asarray(c["active_q_rad"])+f*np.asarray(c["active_delta_rad"]))
    return e
