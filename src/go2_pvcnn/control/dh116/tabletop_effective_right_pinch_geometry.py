"""Measured right original-pad approach; original physical guards unchanged."""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
from scripts import prepare_dh116_effective_right_pinch as plan
from .tabletop_segment_closure_geometry import SegmentClosureGeometry
from .tabletop_right_pinch_approach_geometry import _equal,retain_original_guards
from .tabletop_bilateral_owned_geometry import separation_lower
from .tabletop_measured_closure_geometry import _bytes
from .tabletop_deformable_r0_replay import encode_owned,decode_owned
from .tabletop_deformable_wrist_projection import SOURCE
from .tabletop_deformable_pad_region import bind_geometry,classify_pair
from .tabletop_pair_substep_load_audit import audit_capture,bind_clock
from .tabletop_effective_pad import _patch
VERSION="dh116-effective-right-pinch-geometry-v1"

def validate_projection(pair,exchange,expected_actions=None):
    action=exchange["action_readback"];ids=action["post"]["identity"]
    if len(ids)!=1:raise ValueError("one exact right approach identity")
    clock=ids[0];step=clock["physics_step"]
    if type(step) is not int:raise ValueError("integer physical step")
    ordinal=step-2;expected=plan.expected_boundary(pair,ordinal)
    if ordinal<1 or clock["policy_step"]!=(ordinal-1)//8 or clock["substep"]!=(ordinal-1)%8:
        raise ValueError("same frozen policy/physics substep required")
    actions=plan.action_schedule(pair).numpy() if expected_actions is None else expected_actions
    _equal(action["request"]["action24"],actions[(ordinal-1)//8][None],"24D action",exact=True)
    offsets=np.zeros((1,2,3));offsets[0,1]=expected["right_offset_m"]
    zeros=np.zeros((1,2,3));roots=expected["root_poses_w"][None]
    projection=exchange["wrist_projection"];submission=action["submission"]
    for v,label in ((projection["palm_offsets_m"],"projected offsets"),(action["request"]["requested_palm_offsets_m"],"requested offsets")):
        _equal(v,offsets,label)
    for v in (projection["projected_angles"],action["request"]["requested_angles_rad"],submission["projected_angles"]):
        _equal(v,zeros,"no wrist rotation command")
    for v in (projection["projected_root_poses"],submission["projected_root_poses"]):_equal(v,roots,"root targets")
    _equal(projection["safe_z_clipped"],np.zeros((1,2),bool),"safe-z flag",exact=True)
    _equal(projection["projection_reasons"],np.zeros((1,2),int),"projection reasons",exact=True)
    if projection["projection_source"]!=SOURCE or submission["projection_source"]!=SOURCE:
        raise ValueError("original per-hand safety source required")
    return dict(ordinal=ordinal,version=VERSION,right_offset_m=offsets[0,1].tolist(),
        requested_projected_submitted_targets_agree=True,native_qualified=False)

class EffectiveRightGeometry(SegmentClosureGeometry):
    def __init__(self,project):
        super().__init__(project)
        self._pair=plan.load_pair(project)
        self._actions=plan.action_schedule(self._pair).numpy().copy();self._actions.flags.writeable=False
        self._atlas={s:{n:_patch(self._geometry[s]["source"][n],s,n) for n in self._grids[s]} for s in ("left","right")}
        paths=(plan.CONFIG,"scripts/prepare_dh116_effective_right_pinch.py",
            "src/go2_pvcnn/control/dh116/tabletop_effective_right_pinch_geometry.py",
            "src/go2_pvcnn/control/dh116/tabletop_deformable_pad_region.py",
            "src/go2_pvcnn/control/dh116/tabletop_pair_substep_load_audit.py")
        self.sources["effective_right"]={p:hashlib.sha256((Path(project)/p).read_bytes()).hexdigest() for p in paths}
        self.sources["effective_right"]["static_path_sha256"]=plan.PROOF_SHA

    def _evaluate(self,raw,exchange):
        checked=validate_projection(self._pair,exchange,self._actions)
        packet=retain_original_guards(super()._evaluate(raw,exchange))
        whole={}
        for side in ("left","right"):
            h=raw["hands"][side]
            poses={n:self._pose_matrix(np.r_[p[:3],p[6],p[3:6]]) for n,p in zip(h["body_names"],h["link_poses_xyzw"][0])}
            whole[side]=np.concatenate([self._transform(v,poses[n])
                for shapes in self._geometry[side].values() for n,v in shapes.items()])
        gap=separation_lower(whole["left"],whole["right"])
        if gap<0:
            packet["failure_reasons"].append("inter_hand_separation_not_proven")
            packet.update(passed=False,terminal_reason="closure_geometry_guard")
        rod=exchange["rod"];cap=rod["rod_mesh_contact_pairs"];clock=raw["identity"]
        bind_clock(cap,rod,clock,expected_physics_step=clock["physics_step"])
        audit_capture(cap,solver_substeps=12,complete_capture=True)
        regions=[];poses={}
        for pair in cap["pairs"]:
            name=pair["mesh_name"]
            if name=="table":continue
            side,pad=name.split(".")
            if name not in poses:poses[name]=bind_geometry(raw,clock,name,rod["mesh_bindings"][name])
            regions.append(dict(mesh_name=name,solver_substep=pair["substep"],node_index=pair["node_index"],
                normal_force_on_rod_n=pair["normal_force_on_rod_n"],
                region=classify_pair(pair,poses[name],self._atlas[side][pad])))
        packet.update(version=VERSION,original_zero_wrist_scope_replaced_only=True,
            scoped_target_validation=checked,inter_hand_source_cooked_separation_lower_m=gap,
            pad_region_diagnostic=dict(identity=clock,pairs=regions,
                accepted_by_original_region=sum(r["region"]["accepted_by_original_patch"] for r in regions),
                raw_hand_pairs=len(regions),positive_effective_contact_qualified=False,
                contact_depth_qualified=False,material_slip_qualified=False,teacher_admission=False))
        return packet

    def validate(self,packet,exchange):
        if packet.get("version")!=VERSION or packet.get("sources")!=self.sources:
            raise ValueError("exact effective-right geometry source/version")
        rebuilt=self._evaluate(decode_owned(encode_owned(packet["raw"])),exchange)
        if _bytes(rebuilt)!=_bytes(packet):raise ValueError("same-boundary raw geometry/pad-region recomputation differs")
        return rebuilt
