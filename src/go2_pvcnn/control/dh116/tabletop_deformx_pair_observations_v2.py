"""Owned CPU validation of the separately built native v2 pair packet.

No backend import/read/step. A profile dictionary binds provenance but cannot
attest its own bytes or grant native contact/Reset/Teacher qualification.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import numpy as np

from . import tabletop_complete_pair_capture as complete
from .tabletop_deformable_r0_replay import _digest
from .tabletop_bimanual_table_contacts import _clock, IDENTITY
from .tabletop_deformable_solver_clock import solver_clock_matches

VERSION="dh116-native-pair-observations-consumer-v2"
BACKEND_PATH="/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/mesh-observations-build-20261114/build/python/ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so"
BACKEND_SHA="8817010163946945495c987923d11f3ade4fbd52482a83bbdd48f017d3149340"
SOURCE_TREE_SHA="34741b437191fd66bf151331f20476a86f8708414dcc2782a8c69dd15b0efb64"
PATCH_SHA="1918ebebfde3a00a9c0c3bf979313674d8015f9f5fa42220767b562d169bde41"
RAW_KEYS=complete.RAW_KEYS|frozenset((
    "schema","mesh_positions_world_m","mesh_quaternions_xyzw","surface_gap_m",
    "normal_force_n","sample_source","normal_force_semantics",
    "owner_time_s_before","owner_time_s_after","owner_time_source",
    "substep_timestamp_s","last_error"))
PROFILE_VERIFICATION="required_from_outer_source_basis_and_native_receipt"
EVIDENCE_KINDS=("synthetic_cpu","native_api_return_unqualified")


def _profile(profile):
    value=deepcopy(profile)
    expected=dict(backend_path=BACKEND_PATH,backend_sha256=BACKEND_SHA,
        source_tree_sha256=SOURCE_TREE_SHA,source_file_count=72,patch_delta_sha256=PATCH_SHA)
    if (not isinstance(value,dict) or any(value.get(k)!=v for k,v in expected.items())
            or type(value.get("source_file_count")) is not int
            or not isinstance(value.get("version"),str) or not value["version"]
            or not isinstance(value.get("sources"),dict)
            or value["sources"].get(BACKEND_PATH)!=BACKEND_SHA):
        raise ValueError("exact new backend profile identity required")
    for path,sha in value["sources"].items():
        if (not isinstance(path,str) or not Path(path).is_absolute()
                or not isinstance(sha,str) or len(sha)!=64
                or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError("profile source paths and SHA256 values required")
    _digest(value)  # No arbitrary/native objects can be retained.
    return value


def _sources():
    paths=(__file__,complete.__file__,complete.legacy.__file__)
    return {Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def _array(value,shape,label,dtype=np.float32):
    result=np.array(value,copy=True)
    if result.shape!=shape or result.dtype!=np.dtype(dtype) or not np.isfinite(result).all():
        raise ValueError("finite exact native dtype/shape required: "+label)
    return result


def _identity(value):
    if not isinstance(value,dict) or set(value)!=set(IDENTITY):
        raise ValueError("exact ten-field actual SDK/rod identity required")
    return _clock(lambda:value)


def parse_pair_observations(raw,*,dt_s,identity,source_instance_id,mesh_names,
                            source_profile,evidence_kind):
    """Parse one complete returned packet; never infer missing native fields."""
    profile=_profile(source_profile)
    if isinstance(mesh_names,(str,set,frozenset,dict)):
        raise ValueError("explicit ordered mesh names required")
    names=tuple(mesh_names)
    clock=_identity(identity)
    if (type(source_instance_id) is not int or source_instance_id<=0
            or evidence_kind not in EVIDENCE_KINDS):
        raise ValueError("explicit actual owner ID and evidence kind required")
    if not isinstance(raw,dict) or set(raw)!=RAW_KEYS:
        raise ValueError("exact complete native v2 raw packet required")
    raw=deepcopy(raw)
    if (raw["schema"]!="dh116.sdf_pair_samples.v2"
            or raw["sample_source"]!="cuda_surfaceReadout_same_end_substep_evaluation"
            or raw["normal_force_semantics"]!="unaveraged_penalty_normal_force_before_friction"
            or raw["owner_time_source"]!="native_host_owner_metadata_at_readback"
            or raw["substep_timestamp_s"] is not None
            or type(raw["last_error"]) is not int or raw["last_error"]!=0):
        raise ValueError("native sample source/error/timestamp contract differs")
    times=(raw["owner_time_s_before"],raw["owner_time_s_after"])
    if (any(type(t) not in (int,float) or not math.isfinite(t) or t<0 for t in times)
            or times[0]!=times[1] or times[0]!=clock["solver_time_s"]):
        raise ValueError("actual owner readback time differs from this frame")
    # The original complete projection retains all counter/index/normal/force/
    # aggregate/impulse gates, including exact ordered mesh names.
    legacy=complete.project({k:raw[k] for k in complete.RAW_KEYS},dt_s=dt_s,
        physics_step=clock["physics_step"],rod_tick=clock["rod_tick"],mesh_names=names)
    count=raw["attempted"];meshes=raw["mesh_count"]
    ids=_array(raw["indices"],(count,3),"indices",np.int32)
    vectors=_array(raw["vectors"],(count,12),"vectors")
    _array(raw["node_forces_on_rod_world_n"],
        (len(raw["node_forces_on_rod_world_n"]),3),"node forces")
    _array(raw["mesh_wrenches_on_mesh_world"],(meshes,6),"mesh wrenches")
    pos=_array(raw["mesh_positions_world_m"],(count,3),"sample mesh positions")
    quat=_array(raw["mesh_quaternions_xyzw"],(count,4),"sample XYZW quaternion")
    gap=_array(raw["surface_gap_m"],(count,),"native gap")
    native_n=_array(raw["normal_force_n"],(count,),"native penalty normal force")
    if (not np.allclose(np.linalg.norm(quat.astype(float),axis=1),1.,atol=2e-5,rtol=0)
            or np.any(gap>=0) or np.any(native_n<0)):
        raise ValueError("unit native pose, negative sampled gap and nonnegative native N required")
    # All nodes at the same mesh/substep used the same read-only P struct.
    poses={}
    for i,(sub,node,mesh) in enumerate(ids):
        key=(int(sub),int(mesh));value=(pos[i].tobytes(),quat[i].tobytes())
        if key in poses and poses[key]!=value:
            raise ValueError("same native mesh/substep has inconsistent sample pose")
        poses[key]=value
    # Use the existing aggregate arithmetic tolerance; this is not a new
    # physical clearance/load threshold. Contact point is rounded world float32.
    expected_torque=np.cross(vectors[:,:3].astype(float)-pos.astype(float),-vectors[:,6:9].astype(float))
    if not np.allclose(vectors[:,9:12],expected_torque,atol=3e-4,rtol=5e-5):
        raise ValueError("sample mesh origin does not reconcile with native torque")
    order=np.lexsort((ids[:,2],ids[:,1],ids[:,0]))
    observations=[]
    for pair,i in zip(legacy["pairs"],order):
        observations.append(dict(
            substep=pair["substep"],node_index=pair["node_index"],
            mesh_index=pair["mesh_index"],mesh_name=pair["mesh_name"],
            mesh_position_world_m=pos[i].tolist(),mesh_quaternion_xyzw=quat[i].tolist(),
            native_surface_gap_m=float(gap[i]),
            native_penalty_normal_force_n=float(native_n[i]),
            dot_total_force_normal_n=pair["normal_force_on_rod_n"],
            native_minus_projected_normal_force_n=float(native_n[i])-pair["normal_force_on_rod_n"],
            substep_timestamp_s=None,
            original_region_membership=None,effective_support=None))
    return dict(version=VERSION,evidence_kind=evidence_kind,identity=clock,
        source_instance_id=source_instance_id,dt_s=dt_s,mesh_names=list(names),
        source_profile=profile,source_profile_sha256=_digest(profile),
        source_profile_verification=PROFILE_VERIFICATION,producer_source_sha256=_sources(),
        raw=raw,raw_sha256=_digest(raw),legacy_capture=legacy,observations=observations,
        native_qualification=False,positive_contact_qualified=False,
        negative_contact_qualified=False,teacher_admission=False)


def validate_pair_observations(capture,*,source_profile):
    """Disk revalidation from retained raw values, not from a summary flag."""
    expected_profile=_profile(source_profile)
    if not isinstance(capture,dict) or capture.get("version")!=VERSION:
        raise ValueError("complete native pair observations capture required")
    if (capture.get("source_profile")!=expected_profile
            or capture.get("source_profile_sha256")!=_digest(expected_profile)):
        raise ValueError("expected outer source profile differs")
    raw=capture["raw"]
    if capture.get("raw_sha256")!=_digest(raw):
        raise ValueError("retained raw native packet changed")
    rebuilt=parse_pair_observations(raw,dt_s=capture["dt_s"],identity=capture["identity"],
        source_instance_id=capture["source_instance_id"],mesh_names=capture["mesh_names"],
        source_profile=expected_profile,evidence_kind=capture["evidence_kind"])
    if _digest(rebuilt)!=_digest(capture):
        raise ValueError("retained raw packet, projection or provenance differs")
    return dict(version=VERSION,complete=True,attempted=raw["attempted"],
        capacity=raw["capacity"],overflow=raw["overflow"],substeps=raw["substeps"],
        raw_sha256=capture["raw_sha256"],source_profile_sha256=capture["source_profile_sha256"],
        owner_time_s_before=raw["owner_time_s_before"],owner_time_s_after=raw["owner_time_s_after"],
        native_qualification=False,positive_contact_qualified=False,
        negative_contact_qualified=False,teacher_admission=False)


class PairObservationSequence:
    """Single reset episode packet consumer; only owned CPU metadata is retained."""
    def __init__(self,*,source_profile,reset_identity,source_instance_id,mesh_names,
                 expected_substeps,evidence_kind,dt_s=1/480):
        self.profile=_profile(source_profile)
        self.last=_identity(reset_identity)
        if (self.last["rod_tick"]!=0 or type(source_instance_id) is not int or source_instance_id<=0
                or type(expected_substeps) is not int or not 1<=expected_substeps<=64
                or evidence_kind not in EVIDENCE_KINDS
                or not math.isclose(dt_s,1/480,rel_tol=0,abs_tol=1e-12)):
            raise ValueError("reset identity, owner and original bounded substeps required")
        if isinstance(mesh_names,(str,set,frozenset,dict)):
            raise ValueError("explicit ordered mesh names required")
        self.owner=source_instance_id;self.names=tuple(mesh_names)
        self.substeps=expected_substeps;self.kind=evidence_kind;self.dt=dt_s
        self.failed=False

    def consume(self,raw,*,identity,source_instance_id):
        if self.failed:raise RuntimeError("failed pair sequence requires a new reset owner")
        try:
            clock=_identity(identity)
            if (source_instance_id!=self.owner
                    or any(clock[k]!=self.last[k] for k in ("run_uuid","env_id","episode_uid","reset_id","rod_uid"))
                    or clock["physics_step"]!=self.last["physics_step"]+1
                    or clock["rod_tick"]!=self.last["rod_tick"]+self.substeps
                    or raw.get("substeps")!=self.substeps
                    or not solver_clock_matches(self.last["solver_time_s"],clock["solver_time_s"],steps=1)):
                raise ValueError("same owner and exactly one new SDK/solver frame required")
            result=parse_pair_observations(raw,dt_s=self.dt,identity=clock,
                source_instance_id=source_instance_id,mesh_names=self.names,
                source_profile=self.profile,evidence_kind=self.kind)
            self.last=deepcopy(clock)
            return result
        except Exception:
            self.failed=True
            raise
