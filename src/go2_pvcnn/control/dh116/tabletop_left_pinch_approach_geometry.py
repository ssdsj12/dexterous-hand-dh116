"""Both hands' exact targets; retain all original source/cooked physical guards."""
import hashlib
from pathlib import Path
import numpy as np
from scripts import prepare_dh116_left_pinch_approach as plan
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from . import tabletop_effective_right_pinch_geometry as original
from .tabletop_right_pinch_approach_geometry import _equal
from .tabletop_deformable_wrist_projection import SOURCE

VERSION="dh116-left-pinch-approach-target-contract-v1"

def validate_projection(pair,exchange,expected_actions=None):
    action=exchange["action_readback"];ids=action["post"]["identity"]
    if len(ids)!=1:raise ValueError("one actual bilateral identity")
    clock=ids[0];step=clock["physics_step"]
    if type(step) is not int:raise ValueError("integer physical step")
    ordinal=step-2;expected=plan.expected_boundary(pair,ordinal)
    if ordinal<1 or clock["policy_step"]!=(ordinal-1)//8 or clock["substep"]!=(ordinal-1)%8:
        raise ValueError("same policy/physical substep")
    actions=plan.action_schedule(pair).numpy() if expected_actions is None else expected_actions
    _equal(action["request"]["action24"],actions[(ordinal-1)//8][None],"24D bilateral action",exact=True)
    offsets=np.array([expected["left_offset_m"],expected["right_offset_m"]])[None]
    roots=expected["root_poses_w"][None];zeros=np.zeros((1,2,3))
    projection=exchange["wrist_projection"];submission=action["submission"]
    for value in (projection["palm_offsets_m"],action["request"]["requested_palm_offsets_m"]):
        _equal(value,offsets,"both frozen World offsets")
    for value in (projection["projected_angles"],action["request"]["requested_angles_rad"],submission["projected_angles"]):
        _equal(value,zeros,"no wrist rotation commands")
    for value in (projection["projected_root_poses"],submission["projected_root_poses"]):
        _equal(value,roots,"both SDK root targets")
    _equal(projection["safe_z_clipped"],np.zeros((1,2),bool),"safe-z flags",exact=True)
    _equal(projection["projection_reasons"],np.zeros((1,2),int),"projection reasons",exact=True)
    if projection["projection_source"]!=SOURCE or submission["projection_source"]!=SOURCE:
        raise ValueError("original per-hand safety source")
    return dict(version=VERSION,ordinal=ordinal,left_offset_m=offsets[0,0].tolist(),
        right_offset_m=offsets[0,1].tolist(),requested_projected_submitted_targets_agree=True,native_qualified=False)

class LeftPinchApproachGeometry(original.EffectiveRightGeometry):
    def __init__(self,project):
        with bindings((original,"plan",plan)):
            super().__init__(project)
        self.sources["left_pinch_approach"]={p:hashlib.sha256((Path(project)/p).read_bytes()).hexdigest()
            for p in (plan.CONFIG,"scripts/prepare_dh116_left_pinch_approach.py",
                "src/go2_pvcnn/control/dh116/tabletop_left_pinch_approach_geometry.py",
                "scripts/audit_dh116_left_pinch_tracking_path.py")}
        self.sources["left_pinch_approach"]["conditional_path_sha256"]=plan.PROOF_SHA
    def _evaluate(self,raw,exchange):
        with bindings((original,"plan",plan),(original,"validate_projection",validate_projection)):
            return super()._evaluate(raw,exchange)
