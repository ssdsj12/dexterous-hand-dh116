"""Opt-in owned evidence around the unchanged original wrist projection.

No dynamics, limits or original controller outputs are changed. These records
are diagnostic sources, not a native-success or Teacher admission certificate.
"""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from .tabletop_deformable_wrist_projection import project_policy_wrist_targets as original
from .tabletop_deformable_wrist_projection import SOURCE
from .tabletop_batch_geometry import collision_clearance
from .tabletop_deformable_bimanual_step import _owned
from .tabletop_deformable_r0_replay import encode_owned
from .tabletop_deformable_clearance_guard import GEOMETRY_SOURCE

VERSION="dh116-owned-wrist-projection-evidence-v1"
_GEOMETRY_FIELDS=("physics_step","rod_source_physics_step","geometry_phase","root_poses_w",
    "clearance_m","lowest_shape_index","source","projection_raw")
_INPUTS=("root_seed_w","measured_base_w","common_lift_offset_m","measured_clearance_m",
    "palm_offsets_m","requested_palm_offsets_m","angles_rad","requested_angles_rad","allowed","live")

def pins():
    root=Path(__file__).parent
    return {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
        (Path(__file__).name,"tabletop_deformable_wrist_projection.py","tabletop_command_kernels.py",
         "tabletop_deformable_live_geometry.py","tabletop_batch_geometry.py")}

def equal(a,b,label,*,exact=False):
    a,b=_owned(a),_owned(b)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("finite same-shape "+label)
    if not (np.array_equal(a,b) if exact else np.allclose(a,b,atol=1e-12,rtol=0)):
        raise ValueError(label+" differs")

def capture_geometry(reader,geometry,identity):
    """Read owned buffers at an already-owned SDK boundary; never advances it."""
    b=view=None
    try:
        clock=reader._clock()
        if (clock["physics_step"]!=identity["physics_step"]
            or any(identity.get(k)!=v for k,v in clock.items())):raise ValueError("same pre-action rod/SDK clock")
        raw=dict(identity=deepcopy(identity),link_poses_w=[],shapes=deepcopy(reader._shapes),
            body_names=[],base_indices=[],prim_paths=[],link_paths=[])
        for side,b in enumerate(reader._bindings):
            if reader._identity(b)!=reader._ids[side]:raise ValueError("actual geometry binding changed")
            view=b.hand.root_physx_view
            poses=_owned(view.get_link_transforms())[:,:,[0,1,2,6,3,4,5]].astype(float)
            raw["link_poses_w"].append(poses)
            raw["body_names"].append(list(b.hand.body_names))
            raw["base_indices"].append(b.mount.base_id)
            raw["prim_paths"].append(list(view.prim_paths));raw["link_paths"].append(deepcopy(view.link_paths))
        raw["link_poses_w"]=np.stack(raw["link_poses_w"],axis=1)
        if reader._clock()!=clock or any(reader._identity(b)!=reader._ids[i] for i,b in enumerate(reader._bindings)):
            raise ValueError("geometry changed while copying source")
        out={k:deepcopy(geometry[k]) for k in _GEOMETRY_FIELDS if k!="projection_raw"}
        out["projection_raw"]=raw
        validate_geometry(out,identity)
        return out
    finally:
        reader=geometry=b=view=None

def validate_geometry(g,c):
    raw=g["projection_raw"]
    if (raw["identity"]!=c or g["physics_step"]!=c["physics_step"]
        or g["rod_source_physics_step"]!=c["physics_step"] or g["geometry_phase"]!="synchronized"
        or g["source"]!=GEOMETRY_SOURCE or c["clock_valid"] is not True
        or type(c["physics_step"]) is not int or abs(c["timestamp_s"]-c["physics_step"]/480)>1e-9):
        raise ValueError("same complete pre-action geometry identity")
    poses=_owned(raw["link_poses_w"])
    if (poses.shape!=(1,2,23,7) or not np.isfinite(poses).all()
        or not np.allclose(np.linalg.norm(poses[...,3:],axis=-1),1.,atol=1e-5,rtol=0)):
        raise ValueError("both actual 23-link unit WXYZ buffers")
    shapes=raw["shapes"]
    if len(shapes)!=34:raise ValueError("34 original source-envelope shapes")
    for hi in range(2):
        names=raw["body_names"][hi];base=raw["base_indices"][hi]
        if len(names)!=23 or len(set(names))!=23 or names[base]!="base_link":
            raise ValueError("unique named base link")
        prim=raw["prim_paths"][hi];paths=raw["link_paths"][hi];side=("left","right")[hi]
        if (len(prim)!=1 or side+"Hand" not in prim[0].split("/") or len(paths)!=1
            or len(paths[0])!=23 or len(set(paths[0]))!=23
            or any(p.rsplit("/",1)[-1]!=n for p,n in zip(paths[0],names))
            or any(side+"Hand" not in p.split("/") for p in paths[0])):
            raise ValueError("actual own-side link paths")
        selected=[(bi,v) for h,bi,v in shapes if h==hi]
        if len(selected)!=17 or len({bi for bi,v in selected})!=17:
            raise ValueError("17 independently bound source shapes per side")
    for h,bi,v in shapes:
        a=np.asarray(v)
        if (type(h) is not int or type(bi) is not int or h not in (0,1) or not 0<=bi<23
            or a.ndim!=2 or a.shape[1]!=3 or len(a)<1 or not np.isfinite(a).all()):
            raise ValueError("finite local collision vertices")
    clear,lowest=collision_clearance(poses,shapes)
    equal(clear,g["clearance_m"],"raw source clearance")
    equal(lowest,g["lowest_shape_index"],"lowest source shape",exact=True)
    bases=np.stack([poses[:,hi,raw["base_indices"][hi]] for hi in range(2)],axis=1)
    equal(bases,g["root_poses_w"],"raw measured base")
    return clear,bases

def record_projection(geometry,identity,**kwargs):
    if set(kwargs)!=set(_INPUTS):raise ValueError("complete original projection arguments")
    # Compute the unmodified operation first. Inputs are copied, never patched.
    out=original(**kwargs)
    out["input_evidence"]=dict(version=VERSION,source_sha256=pins(),identity=deepcopy(identity),
        geometry={k:deepcopy(geometry[k]) for k in _GEOMETRY_FIELDS},
        inputs={k:_owned(v) for k,v in kwargs.items()},native_qualified=False,teacher_admission=False)
    validate_record(out,identity)
    return out

def validate_record(projection,identity):
    evidence=projection["input_evidence"]
    if evidence["version"]!=VERSION or evidence["source_sha256"]!=pins() or evidence["identity"]!=identity:
        raise ValueError("frozen source and pre-action identity")
    clear,bases=validate_geometry(evidence["geometry"],identity)
    values=evidence["inputs"]
    if set(values)!=set(_INPUTS):raise ValueError("complete projection inputs")
    equal(clear,values["measured_clearance_m"],"source measured clearance")
    equal(bases,values["measured_base_w"],"source measured root")
    args={k:torch.from_numpy(_owned(v)) for k,v in values.items()}
    # Original implementation validates shape, dtype, limits and unit quaternions.
    replay=original(**args)
    for k,v in replay.items():
        if isinstance(v,torch.Tensor):equal(v,projection[k],"replayed "+k,exact=True)
        elif projection[k]!=v:raise ValueError("replayed projection source differs")
    return dict(version=VERSION,identity=deepcopy(identity),
        independent_source_clearance_recomputed=True,
        safe_z_active=_owned(projection["safe_z_clipped"])[0].tolist(),
        source_clearance_m=clear[0].tolist(),native_qualified=False,teacher_admission=False)

def retain_failed_geometry(raw,exchange,evaluate):
    """After a successful actual read, a validation error must retain the raw copy."""
    try:return evaluate(raw,exchange)
    except (ValueError,KeyError) as error:
        snapshot=deepcopy(raw)
        return dict(version=VERSION,raw=snapshot,
            raw_sha256=hashlib.sha256(json.dumps(encode_owned(snapshot),sort_keys=True).encode()).hexdigest(),
            passed=False,terminal_reason="closure_geometry_source_invalid",
            failure_reasons=[type(error).__name__+": "+str(error)],
            native_qualification=False,teacher_admission=False)

class ProjectionEvidenceMixin:
    """Compose before the existing ClosureExchange; changes only owned evidence."""
    def _geometry(self,expected,*,previous_rod_clock=None):
        row=None
        try:
            row=super()._geometry(expected,previous_rod_clock=previous_rod_clock)
            if previous_rod_clock is None:
                self._owned_projection_geometry=capture_geometry(self._read_geometry,row,self._clock(self._evidence_substep)[0])
            return row
        finally:
            self=row=previous_rod_clock=None

    def _step(self,action,substep):
        from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
        from . import tabletop_deformable_physical_exchange as module
        self._evidence_substep=substep;self._owned_projection_geometry=None
        def recording(**kwargs):
            return record_projection(self._owned_projection_geometry,self._clock(substep)[0],**kwargs)
        try:
            if module.project_policy_wrist_targets is not original:raise ValueError("foreign projection binding")
            with bindings((module,"project_policy_wrist_targets",recording)):
                return super()._step(action,substep)
        finally:
            self._owned_projection_geometry=None
            self=action=recording=None

class RetainedGeometryMixin:
    """Keep complete owned raw geometry if the experiment target check fails."""
    def _evaluate(self,raw,exchange):
        result=retain_failed_geometry(raw,exchange,super()._evaluate)
        if result.get("version")==VERSION:
            result["sources"]=deepcopy(self.sources)
            result["evidence_source_sha256"]=pins()
        return result

    def validate(self,packet,exchange):
        if packet.get("version")!=VERSION:return super().validate(packet,exchange)
        if packet.get("sources")!=self.sources or packet.get("evidence_source_sha256")!=pins():
            raise ValueError("retained geometry source differs")
        from .tabletop_deformable_r0_replay import decode_owned
        rebuilt=self._evaluate(decode_owned(encode_owned(packet["raw"])),exchange)
        if encode_owned(rebuilt)!=encode_owned(packet):raise ValueError("retained failed geometry differs")
        return rebuilt
