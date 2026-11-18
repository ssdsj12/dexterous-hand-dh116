"""Opt-in measured geometry scope for the exact right-owned approach.

The original zero-wrist scope is preserved. Only its zero-offset assertion is
superseded after exact schedule, physical identity, projection and submission
are checked; all collision, tracking, joint and depth guards remain.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
from scripts import prepare_dh116_right_pinch_owned_contact as plan
from .tabletop_segment_closure_geometry import SegmentClosureGeometry
from .tabletop_measured_closure_geometry import _bytes
from .tabletop_deformable_bimanual_step import _owned
from .tabletop_deformable_r0_replay import encode_owned,decode_owned
from .tabletop_deformable_wrist_projection import SOURCE

VERSION="dh116-right-owned-approach-geometry-v1"


def _equal(actual,expected,label,*,exact=False):
    a=_owned(actual);b=np.asarray(expected)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("finite same-shape "+label+" required")
    if not (np.array_equal(a,b) if exact else np.allclose(a,b,atol=1e-12,rtol=0)):
        raise ValueError("frozen "+label+" differs")


def validate_projection(pair,exchange):
    action=exchange["action_readback"];ids=action["post"]["identity"]
    if len(ids)!=1:raise ValueError("one right-approach physical identity required")
    clock=ids[0];step=clock["physics_step"]
    if type(step) is not int:raise ValueError("integer SDK step required")
    ordinal=step-2
    expected=plan.expected_boundary(pair,ordinal)
    if ordinal<1 or clock["policy_step"]!=(ordinal-1)//8 or clock["substep"]!=(ordinal-1)%8:
        raise ValueError("same frozen policy/physics substep required")
    _equal(action["request"]["action24"],plan.action_schedule(pair)[(ordinal-1)//8].numpy()[None],"24D action",exact=True)
    offsets=np.zeros((1,2,3));offsets[0,1,2]=-.0005*expected["fraction"]
    zeros=np.zeros((1,2,3));roots=expected["root_poses_w"][None]
    projection=exchange["wrist_projection"];submission=action["submission"]
    for value,label in ((projection["palm_offsets_m"],"projected offsets"),
                        (action["request"]["requested_palm_offsets_m"],"requested offsets")):
        _equal(value,offsets,label)
    for value,label in ((projection["projected_angles"],"projected angles"),
                        (action["request"]["requested_angles_rad"],"requested angles"),
                        (submission["projected_angles"],"submitted angles")):
        _equal(value,zeros,label)
    for value,label in ((projection["projected_root_poses"],"projected roots"),
                        (submission["projected_root_poses"],"submitted roots")):
        _equal(value,roots,label)
    _equal(projection["safe_z_clipped"],np.zeros((1,2),bool),"safe-z flag",exact=True)
    _equal(projection["projection_reasons"],np.zeros((1,2),int),"projection reasons",exact=True)
    if projection["projection_source"]!=SOURCE or submission["projection_source"]!=SOURCE:
        raise ValueError("original per-hand projection source required")
    return dict(ordinal=ordinal,version=VERSION,expected_right_z_offset_m=float(offsets[0,1,2]),
                requested_projected_submitted_targets_agree=True,native_qualified=False)


def retain_original_guards(packet):
    """Internal post-validation replacement of only the old zero-offset rule."""
    result=deepcopy(packet)
    result["failure_reasons"]=[r for r in result["failure_reasons"] if r!="unexpected_wrist_projection"]
    result["passed"]=not result["failure_reasons"]
    result["terminal_reason"]=None if result["passed"] else "closure_geometry_guard"
    return result


class RightApproachGeometry(SegmentClosureGeometry):
    def __init__(self,project):
        super().__init__(project)
        self._pair=plan.load_pair(project)
        paths=(plan.CONFIG,"scripts/prepare_dh116_right_pinch_owned_contact.py",
               "src/go2_pvcnn/control/dh116/tabletop_right_pinch_approach_geometry.py")
        self.sources["right_owned_approach"]={p:hashlib.sha256((Path(project)/p).read_bytes()).hexdigest() for p in paths}
        self.sources["right_owned_approach"]["static_proof_sha256"]=plan.PROOF_SHA

    def _evaluate(self,raw,exchange):
        checked=validate_projection(self._pair,exchange)
        original=super()._evaluate(raw,exchange)
        packet=retain_original_guards(original)
        packet.update(version=VERSION,original_zero_wrist_scope_replaced_only=True,
                      scoped_target_validation=checked)
        return packet

    def validate(self,packet,exchange):
        if packet.get("version")!=VERSION or packet.get("sources")!=self.sources:
            raise ValueError("right-owned geometry source/version differs")
        rebuilt=self._evaluate(decode_owned(encode_owned(packet["raw"])),exchange)
        if _bytes(rebuilt)!=_bytes(packet):raise ValueError("right-owned raw geometry recomputation differs")
        return rebuilt
