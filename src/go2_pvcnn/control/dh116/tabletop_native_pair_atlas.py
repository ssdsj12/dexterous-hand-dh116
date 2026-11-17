"""Offline original-pad classification at each validated native pair's actual P pose.

No endpoint interpolation, solver call, guard change, or contact qualification.
The loader is CPU-only/offline: its USD import must never be used before Kit in
a native worker. Source/profile dictionaries do not replace a native receipt.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from . import tabletop_deformx_pair_observations_v2 as pairs
from . import tabletop_deformable_pad_region as region
from . import tabletop_effective_pad as original_atlas
from . import tabletop_effective_pad_runtime as predicate
from .tabletop_dual_hand_rod_runtime import MESH_NAMES,HAND_NAMES
from .tabletop_deformable_r0_replay import _digest

VERSION="dh116-native-pair-original-atlas-v1"


def _source_hashes():
    return {Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in (__file__,pairs.__file__,region.__file__,original_atlas.__file__,predicate.__file__)}


class _OriginalAtlas:
    """Only owned original CAD regions and checked source identities survive loading."""
    def __init__(self,project):
        from scripts import audit_dh116_pinch_initial_pose as geometry
        records={};sources={};g=None
        try:
            for side in ("left","right"):
                g=geometry.load_geometry(Path(project),side=side)
                sources[side]=deepcopy(g["source_sha256"])
                for name in original_atlas.DISTAL_LINKS:
                    records[side+"."+name]=original_atlas._patch(
                        g["source_colliders"][name]["vertices"],side,name)
            self._regions=records
            self._source_basis=dict(
                atlas="dh116_effective_pad_v1",frame="PhysX_link_origin",unit="m",
                geometry=sources,geometry_loader_sha256=hashlib.sha256(
                    Path(geometry.__file__).read_bytes()).hexdigest(),
                implementation_sha256=_source_hashes(),
                geometry_source="original_locked_USD_convex_envelope_not_live_cooked",
                native_qualification=False)
            self._seal=_digest(dict(regions=self._regions,sources=self._source_basis))
        finally:g=None

    @property
    def regions(self):return deepcopy(self._regions)

    @property
    def source_basis(self):return deepcopy(self._source_basis)

    def _verify(self):
        if (set(self._regions)!=set(HAND_NAMES)
            or self._source_basis.get("implementation_sha256")!=_source_hashes()
            or self._seal!=_digest(dict(regions=self._regions,sources=self._source_basis))):
            raise ValueError("unchanged loaded original atlas and source identities required")


def load_original_atlas(project):
    """Load the existing locked source geometry; never create an SDK/solver owner."""
    return _OriginalAtlas(project)


def _gates():
    return dict(effective_support=None,fixed_material_slip_m=None,
        native_qualification=False,positive_contact_qualified=False,
        negative_contact_qualified=False,teacher_admission=False)


def classify_observations(capture,*,source_profile,atlas):
    """Revalidate a v2 capture, then classify all pairs using its same-event P."""
    if type(atlas) is not _OriginalAtlas:
        raise ValueError("original atlas loaded from its checked sources required")
    atlas._verify()
    checked=pairs.validate_pair_observations(capture,source_profile=source_profile)
    if tuple(capture["mesh_names"])!=MESH_NAMES:
        raise ValueError("original ordered eleven mesh slots required")
    old=capture["legacy_capture"]["pairs"];observations=capture["observations"]
    if len(old)!=len(observations):raise ValueError("all same-order pair observations required")
    rows=[];hand_count=accepted=0
    for pair,observation in zip(old,observations):
        keys=("substep","node_index","mesh_index","mesh_name")
        if any(pair[k]!=observation[k] for k in keys):
            raise ValueError("native pose and legacy pair identity differ")
        name=pair["mesh_name"]
        position=np.asarray(observation["mesh_position_world_m"],dtype=float)
        quat=np.asarray(observation["mesh_quaternion_xyzw"],dtype=float)
        # Parser has already checked finite native float32 values and unit XYZW.
        T=np.eye(4);T[:3,:3]=Rotation.from_quat(quat).as_matrix();T[:3,3]=position
        classified=None
        if name!="table":
            classified=region.classify_pair(pair,T,atlas._regions[name])
            hand_count+=1;accepted+=int(classified["accepted_by_original_patch"])
        rows.append(dict(identity=deepcopy(capture["identity"]),
            solver_substep=pair["substep"],node_index=pair["node_index"],
            mesh_index=pair["mesh_index"],mesh_name=name,
            source_instance_id=capture["source_instance_id"],
            actual_mesh_position_world_m=position.tolist(),
            actual_mesh_quaternion_xyzw=quat.tolist(),
            pose_source=capture["raw"]["sample_source"],
            contact_point_world_m=deepcopy(pair["contact_point_world_m"]),
            outward_normal_world=deepcopy(pair["outward_normal_world"]),
            dot_total_force_normal_n=pair["normal_force_on_rod_n"],
            native_penalty_normal_force_n=observation["native_penalty_normal_force_n"],
            native_surface_gap_m=observation["native_surface_gap_m"],
            impulse_on_rod_ns=deepcopy(pair["impulse_on_rod_ns"]),
            substep_timestamp_s=None,
            classification_status="original_atlas_checked" if classified is not None else "not_a_hand_pad",
            region=classified,**_gates()))
    return dict(version=VERSION,evidence_kind=capture["evidence_kind"],pairs=rows,
        identity=deepcopy(capture["identity"]),source_instance_id=capture["source_instance_id"],
        raw_sha256=capture["raw_sha256"],source_profile_sha256=capture["source_profile_sha256"],
        source_profile_verification=capture["source_profile_verification"],
        source_profile=deepcopy(capture["source_profile"]),
        pair_consumer_source_sha256=deepcopy(capture["producer_source_sha256"]),
        atlas_source_basis=atlas.source_basis,
        mesh_binding_verification="required_from_outer_original_eleven_mesh_owner",
        capture_counts={k:checked[k] for k in ("attempted","capacity","overflow","substeps")},
        owner_time_s_before=checked["owner_time_s_before"],owner_time_s_after=checked["owner_time_s_after"],
        owner_time_source=capture["raw"]["owner_time_source"],substep_timestamp_s=None,
        summary=dict(total_pairs=len(rows),hand_pairs=hand_count,table_pairs=len(rows)-hand_count,
            accepted_original_region_pairs=accepted,rejected_original_region_pairs=hand_count-accepted),
        gap_semantics="native phi minus rod radius; distinct from atlas supporting-plane distance",
        load_semantics="dot(total force,normal) drives original predicate; native N retained separately",
        no_native_calls=True,new_physics_steps=0,ppo_updates=0,**_gates())
