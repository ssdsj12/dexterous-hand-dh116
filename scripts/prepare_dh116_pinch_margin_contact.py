"""Opt-in 234-step left contact candidate; old paired owner/defaults preserved."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from scripts import prepare_dh116_native_parameter_task as previous
from scripts import prepare_dh116_dual_pinch_closure_interval as closure

VERSION="dh116-pinch-margin-contact-v1"
SETUP_STEPS,MEASURED_STEPS,TOTAL_STEPS=2,232,234
PROFILE_SHA=None
ROOT=Path(__file__).resolve().parents[1]
CONFIG="configs/dh116_tasks/flexible_pinch_margin_contact_v1.json"
OLD_PAIR=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/paired-pinch-reset-BaP0hx/result.json")
OLD_PAIR_SHA="996bdb33ed2d6f9abddb9171245142b5147af222b976fc799b5f89711c4f6a37"
PROOF=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/evidence-rod-axis-margin-20261113-Pz4lNR/run01/result.json")
PROOF_SHA="d1fc5b21cd2ed0d3e9d4c17e5cc56eff1259a13dc6558942c052c668f9d1c03b"
sdk_clock_sample=closure.sdk_clock_sample
make_geometry=closure.make_geometry
run_previous=previous.run_existing_app

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def read_locked(path,sha):
    p=Path(path);raw=p.read_bytes()
    if p.resolve(strict=True)!=p or hashlib.sha256(raw).hexdigest()!=sha:
        raise ValueError("locked evidence changed: "+str(p))
    return json.loads(raw)

def derive_pair():
    from go2_pvcnn.control.dh116.linear_coupling import expand
    old=read_locked(OLD_PAIR,OLD_PAIR_SHA)
    proof=read_locked(PROOF,PROOF_SHA);left=proof["sides"]["left"]
    if (proof["status"]!="complete" or proof["physics_steps"]!=0
        or not all(left[k] is True for k in ("static_geometry_passed","table_clear","non_target_clear",
            "initial_clear","both_end_sdf_margin","opposed","required_depth_161_targets_passed"))
        or left["native_ready"] is not False or min(left["continuous_table_lower_m"].values())<.0021
        or min(left["continuous_non_target_lower_m"].values())<0):
        raise ValueError("complete static left proof required; not native qualification")
    candidate=deepcopy(left["candidate"])
    seed=deepcopy(old["reset_seed"]);seed.pop("sha256")
    for key in ("hand_roots","authored_root_seed_w","reset_root_pose_w"):
        seed[key][0]=deepcopy(candidate["base_pose_wxyz"])
    seed["joint_positions"][0]=expand(candidate["active_q_rad"]).tolist()
    seed["source_description"]="left max-table-margin static candidate; original right and free rod unchanged"
    seed["sha256"]=digest(seed)
    return dict(version=VERSION,scope="independent_left_contact_candidate_not_teacher",
        left_pose_sha256=old["left_pose_sha256"],left=dict(candidate=candidate),
        right=deepcopy(old["right"]),rod_seed=deepcopy(old["rod_seed"]),reset_seed=seed,
        wrist_clearance_floor_m=.0021,native_qualified=False,teacher_admission=False,
        sources={str(OLD_PAIR):OLD_PAIR_SHA,str(PROOF):PROOF_SHA},
        schedule=dict(setup_steps=2,quiet_steps=48,closure_steps=160,hold_steps=24,total_steps=234,
            policy_steps=29,physics_hz=480,control_hz=60,ppo_updates=0,environments=1),
        required_pad_min_gap_m=-.0003,
        limitations=["left contact diagnostic only; right has zero commands",
            "static proof does not establish mounted tracking or real contact",
            "static hand-table contact remains NA","no grasp/Teacher/slip qualification"])

def load_pair(project=ROOT):
    value=json.loads((Path(project)/CONFIG).read_text())
    if value!=derive_pair():raise ValueError("frozen derived left reset differs")
    return value

def action_schedule():
    import torch
    delta=load_pair()["left"]["candidate"]["active_delta_rad"]
    out=torch.zeros((29,24),dtype=torch.float64)
    out[6:26,:6]=torch.tensor(delta,dtype=out.dtype)*10
    return out

def required_sdf_valid(row):
    from go2_pvcnn.control.dh116 import tabletop_measured_closure_geometry as geometry
    # Original validity is checked independently, plus a strict additional bound.
    value=row["minimum_gap_m"];gradient=row["gradient_norm_at_minimum"]
    import math
    return bool(value is not None and math.isfinite(value) and value>=-.0003
        and gradient is not None and math.isfinite(gradient) and gradient>0
        and len(row["in_lattice_nodes"])>0)

@contextmanager
def scenario_scope(project=ROOT):
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116 import tabletop_measured_closure_geometry as geometry
    from scripts import dh116_pinch_margin_contact_audit as audit
    from scripts import validate_dh116_dual_pinch_director_interval as director_audit
    from scripts import validate_dh116_dual_pinch_closure_interval as closure_audit
    packet=load_pair(project);path=Path(project)/CONFIG
    with bindings((builder,"PAIR",path),(builder,"PAIR_SHA",hashlib.sha256(path.read_bytes()).hexdigest()),
            (builder,"SEED_SHA",packet["reset_seed"]["sha256"]),(builder,"load_pair_proposal",lambda:load_pair(project)),
            (closure,"MEASURED_STEPS",MEASURED_STEPS),(closure,"TOTAL_STEPS",TOTAL_STEPS),
            (closure,"action_schedule",action_schedule),
            (previous.director,"inspect_closure",audit.inspect_closure),
            (director_audit,"inspect_directors",audit.inspect_directors),
            (closure_audit,"validate_artifacts",audit.validate_artifacts),
            (geometry,"required_sdf_valid",required_sdf_valid),
            (previous,"VERSION",VERSION),(previous,"PROFILE_SHA",PROFILE_SHA)):
        yield

def inspect_interval(path,*,complete=True):
    with scenario_scope():
        return previous.inspect_interval(path,complete=complete)

def run_existing_app(*args,**kwargs):
    try:
        with scenario_scope(kwargs.get("project",ROOT)):
            return run_previous(*args,**kwargs)
    finally:
        args=kwargs=None
