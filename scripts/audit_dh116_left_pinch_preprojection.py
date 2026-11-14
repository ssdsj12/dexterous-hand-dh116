"""Reconstruct the first safe-z clip from the preceding native link buffers."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from scripts import audit_dh116_left_pinch_projection_failure as source
from scripts import prepare_dh116_left_pinch_approach as plan
from go2_pvcnn.control.dh116.tabletop_left_pinch_approach_geometry import LeftPinchApproachGeometry
from go2_pvcnn.control.dh116.tabletop_batch_geometry import rotated_heights
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import project_policy_wrist_targets
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned

VERSION="dh116-left-pinch-preprojection-native-replay-v1"
def load_case():
    rows=source.validate_prefix(source.parse_jsonl(source.load_frozen()["physical_bytes"]))
    return rows[-2],rows[-1],LeftPinchApproachGeometry(plan.ROOT),plan.load_pair()

def evaluate(previous,current,geometry,pair):
    raw=previous["closure_geometry"]["raw"];pre=current["action_readback"]["pre"]["identity"][0]
    post=current["action_readback"]["post"]["identity"][0];c=raw["identity"]
    fixed=("run_uuid","env_id","episode_uid","reset_id","rod_uid","physics_step","rod_tick","timestamp_s","solver_time_s","clock_valid")
    if (previous["physics_step"]+1!=current["physics_step"] or c["physics_step"]!=previous["physics_step"]
        or any(c[k]!=pre[k] for k in fixed) or post["physics_step"]!=pre["physics_step"]+1):
        raise ValueError("preceding same-episode native boundary required")
    roots=[];clearances=[];details={}
    seed=np.array(pair["reset_seed"]["reset_root_pose_w"])[None]
    for hi,side in enumerate(("left","right")):
        hand=raw["hands"][side]
        poses={n:np.r_[v[:3],v[6],v[3:6]] for n,v in zip(hand["body_names"],hand["link_poses_xyzw"][0])}
        heights={n:float(rotated_heights(poses[n],vertices).min()-.3)
            for n,vertices in geometry._geometry[side]["source"].items()}
        limiting=min(heights,key=heights.get);clear=heights[limiting]
        logged=previous["post_step_clearance"]["clearance_m"][0][hi]
        if not np.isclose(clear,logged,atol=1e-12,rtol=0):raise ValueError("raw source-envelope readback mismatch")
        roots.append(poses["base_link"]);clearances.append(clear)
        details[side]=dict(limiting_source_shape=limiting,source_clearance_mm=clear*1000,
            measured_base_z_m=float(poses["base_link"][2]),authored_seed_z_m=float(seed[0,hi,2]),
            table_floor_mm=2.1,post_clearance_mm=float(current["post_step_clearance"]["clearance_m"][0][hi])*1000)
    request=current["action_readback"]["request"];old=previous["wrist_projection"]
    # This consumed owner uses zero common lift and true allowed/live on every
    # step. Those are frozen source-code constants, not measured sensor fields.
    values=dict(root_seed_w=torch.tensor(seed),measured_base_w=torch.tensor(np.array([roots])),
        common_lift_offset_m=torch.zeros(1,dtype=torch.float64),measured_clearance_m=torch.tensor([clearances],dtype=torch.float64),
        palm_offsets_m=torch.tensor(old["palm_offsets_m"]),requested_palm_offsets_m=torch.tensor(request["requested_palm_offsets_m"]),
        angles_rad=torch.tensor(old["projected_angles"]),requested_angles_rad=torch.tensor(request["requested_angles_rad"]),
        allowed=torch.ones(1,dtype=torch.bool),live=torch.ones(1,dtype=torch.bool))
    # Source clearances and other control values must retain original float64.
    for k,v in values.items():
        if v.dtype!=torch.bool:values[k]=v.double()
    replay=project_policy_wrist_targets(**values);actual=current["wrist_projection"];errors={}
    for key,value in replay.items():
        if isinstance(value,torch.Tensor):
            a=value.numpy();b=np.asarray(actual[key])
            if a.shape!=b.shape or not np.isfinite(b).all():raise ValueError("actual projection output shape")
            if a.dtype==bool:
                if not np.array_equal(a,b):raise ValueError("projection flags differ")
                errors[key]=0.
            else:
                errors[key]=float(np.max(abs(a-b)))
                if errors[key]>1e-12:raise ValueError("source replay differs: "+key)
        elif actual[key]!=value:raise ValueError("projection metadata differs")
    for hi,side in enumerate(("left","right")):
        p=float(actual["palm_offsets_m"][0,hi,2]);r=float(request["requested_palm_offsets_m"][0,hi,2])
        details[side].update(requested_z_offset_mm=r*1000,projected_z_offset_mm=p*1000,
            safe_z_offset_mm=float(actual["safe_z_offset_m"][0,hi])*1000,projection_delta_mm=(p-r)*1000)
    return dict(version=VERSION,receipt_sha256=source.RECEIPT_SHA,
        physical_sha256=source.LOCKS["physical.jsonl"],pre_identity=c,action_identity=post,
        sides=details,replay_errors=errors,replay_max_error_m=max(errors.values()),
        measured_source="preceding actual SDK link transforms plus locked source collision vertices",
        inferred_from_frozen_owner_code={"common_lift_offset_m":0.,"allowed":True,"live":True},
        source_geometry_sha256=geometry.sources,teacher_admission=False,native_qualified=False,
        new_physics_steps=0,ppo_updates=0)

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    r=evaluate(*load_case())
    with a.output.open("x") as f:json.dump(encode_owned(r),f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(dict(sides=r["sides"],replay_max_error=r["replay_max_error_m"],sha256=source.sha(a.output))))
if __name__=="__main__":main()
