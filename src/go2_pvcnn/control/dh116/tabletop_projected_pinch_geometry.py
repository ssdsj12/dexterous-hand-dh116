"""Independent experimental target contract with the original safety projection.

This does not relax any physical geometry, joint, depth or velocity guard.
Unlike the old exact unprojected-path diagnostic, valid safety projections are
recomputed from owned pre-action source geometry and retained as real control.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
import torch
from scripts import prepare_dh116_left_pinch_approach as plan
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from . import tabletop_effective_right_pinch_geometry as original
from .tabletop_wrist_projection_evidence import validate_record,equal,RetainedGeometryMixin
from .tabletop_command_kernels import wrist_requested_targets
from .tabletop_surface_contract import ACTION_CONTRACT
from .tabletop_deformable_wrist_projection import SOURCE

VERSION="dh116-source-recomputed-projected-pinch-v1"

def validate_projected_target(pair,exchange,expected_actions=None):
    a=exchange["action_readback"];p=exchange["wrist_projection"]
    if len(a["pre"]["identity"])!=1 or len(a["post"]["identity"])!=1:raise ValueError("one actual identity")
    pre,post=a["pre"]["identity"][0],a["post"]["identity"][0]
    ordinal=post["physics_step"]-2
    if (type(post["physics_step"]) is not int or not 1<=ordinal<=232
        or pre["physics_step"]+1!=post["physics_step"]
        or post["policy_step"]!=(ordinal-1)//8 or post["substep"]!=(ordinal-1)%8
        or any(pre[k]!=post[k] for k in ("run_uuid","env_id","episode_uid","reset_id","rod_uid","policy_step","substep"))
        or abs(post["timestamp_s"]-post["physics_step"]/480)>1e-9):
        raise ValueError("same native 480Hz boundary")
    checked=validate_record(p,pre);inputs=p["input_evidence"]["inputs"]
    expected=plan.action_schedule(pair).numpy() if expected_actions is None else expected_actions
    equal(a["request"]["action24"],expected[(ordinal-1)//8][None],"unchanged 24D request",exact=True)
    equal(inputs["root_seed_w"],np.asarray(pair["reset_seed"]["reset_root_pose_w"])[None],"authored seed")
    equal(inputs["common_lift_offset_m"],np.zeros(1),"zero external common lift",exact=True)
    for k in ("allowed","live"):equal(inputs[k],np.ones(1,bool),"original "+k,exact=True)
    action=torch.as_tensor(a["request"]["action24"],dtype=torch.float64)
    rates=dict(joint_rate_rad_s=ACTION_CONTRACT["joint_rate_rad_s"],wrist_rate_m_s=ACTION_CONTRACT["translation_rate_m_s"],rotation_rate_rad_s=ACTION_CONTRACT["rotation_rate_rad_s"])
    palms,angles=wrist_requested_targets(torch.as_tensor(inputs["palm_offsets_m"]),
        torch.as_tensor(inputs["angles_rad"]),action,rates,1/480)
    for k,v in (("requested_palm_offsets_m",palms),("requested_angles_rad",angles)):
        equal(inputs[k],v,"original action integration "+k)
        equal(a["request"][k],v,"same action readback "+k)
    for k in ("projected_root_poses","projected_angles"):
        equal(a["submission"][k],p[k],"actual SDK submitted "+k,exact=True)
    if a["submission"]["projection_source"]!=SOURCE:raise ValueError("original projection source")
    return dict(version=VERSION,ordinal=ordinal,safe_z_active=checked["safe_z_active"],
        original_projection_recomputed=True,fixed_unprojected_path_qualified=False,
        native_qualified=False,teacher_admission=False)

def validate_continuity(previous,current):
    p=current["wrist_projection"]["input_evidence"]["inputs"]
    pre=current["action_readback"]["pre"]["identity"][0]
    if previous is None:
        if pre["physics_step"]!=2:raise ValueError("fresh actual reset")
        equal(p["palm_offsets_m"],np.zeros((1,2,3)),"zero reset palm offset",exact=True)
        equal(p["angles_rad"],np.zeros((1,2,3)),"zero reset wrist angles",exact=True)
    else:
        old=previous["action_readback"]["post"]["identity"][0]
        for k in ("run_uuid","env_id","episode_uid","reset_id","rod_uid","physics_step","rod_tick","timestamp_s"):
            if old[k]!=pre[k]:raise ValueError("adjacent actual projection identities")
        equal(p["palm_offsets_m"],previous["wrist_projection"]["palm_offsets_m"],"applied offset continuity",exact=True)
        equal(p["angles_rad"],previous["wrist_projection"]["projected_angles"],"applied angle continuity",exact=True)
    return True

class _ProjectedGeometry(original.EffectiveRightGeometry):
    def __init__(self,project):
        with bindings((original,"plan",plan)):super().__init__(project)
        names=("src/go2_pvcnn/control/dh116/tabletop_projected_pinch_geometry.py",
            "src/go2_pvcnn/control/dh116/tabletop_wrist_projection_evidence.py")
        self.sources["owned_projection"]={n:hashlib.sha256((Path(project)/n).read_bytes()).hexdigest() for n in names}

    def _evaluate(self,raw,exchange):
        with bindings((original,"plan",plan),(original,"validate_projection",validate_projected_target)):
            return super()._evaluate(raw,exchange)

class ProjectedPinchGeometry(RetainedGeometryMixin,_ProjectedGeometry):
    pass
