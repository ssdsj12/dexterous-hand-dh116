"""Synthetic native-shaped pairs and locked original USD atlas; no native execution."""
from copy import deepcopy
import importlib
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from go2_pvcnn.control.dh116 import tabletop_deformx_pair_observations_v2 as v2
from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as source
from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import MESH_NAMES
from go2_pvcnn.control.dh116.tabletop_deformable_pad_region import classify_pair
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest,encode_owned,decode_owned
import json

ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/"src/go2_pvcnn/control/dh116/tabletop_native_pair_atlas.py"
def api():
    assert MODULE.is_file(), "new actual-pose atlas classifier is missing"
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_native_pair_atlas")

@pytest.fixture(scope="module")
def atlas():
    return api().load_original_atlas(ROOT)

@pytest.fixture(scope="module")
def profile():
    return source.source_basis()

def capture(atlas,profile,specs=None):
    # Input tuple: mesh name, u offset, plane offset, normal sign, load.
    if specs is None:
        specs=[("left.finger14_link",0.,0.,1.,2.),
               ("right.finger23_link",0.,0.,1.,3.),("table",0.,0.,1.,4.)]
    ids=[];vectors=[];positions=[];quaternions=[]
    for j,(name,u,d,sign,load) in enumerate(specs):
        sub=j%2;node=j;mesh=MESH_NAMES.index(name)
        q=Rotation.from_rotvec([.2,-.3,.4+sub*.2]);R=q.as_matrix()
        P=np.array([.21+sub*.01,-.07,.3],np.float32)
        if name=="table":
            local=np.zeros(3);n=np.array([0.,0.,1.])
        else:
            a=atlas.regions[name];n=np.asarray(a["reference_outward_normal_local"])
            local=np.asarray(a["center_local_m"])+u*np.asarray(a["tangent_u"])+d*n
        point=(R@local+P).astype(np.float32);normal=(R@(sign*n)).astype(np.float32)
        force=(normal*load).astype(np.float32)
        torque=np.cross(point-P,-force).astype(np.float32)
        ids.append((sub,node,mesh));vectors.append(np.r_[point,normal,force,torque])
        positions.append(P);quaternions.append(q.as_quat())
    ids=np.asarray(ids,np.int32).reshape(-1,3);vectors=np.asarray(vectors,np.float32).reshape(-1,12)
    nodes=np.zeros((41,3),np.float32);meshes=np.zeros((11,6),np.float32)
    if len(ids):
        np.add.at(nodes,ids[:,1],vectors[:,6:9]/2)
        np.add.at(meshes,ids[:,2],np.c_[-vectors[:,6:9],vectors[:,9:12]]/2)
    order=np.arange(len(ids))[::-1] # Atomic output order is deliberately unsorted.
    raw=dict(schema="dh116.sdf_pair_samples.v2",substeps=2,mesh_count=11,capacity=902,
        attempted=len(ids),overflow=False,indices=ids[order],vectors=vectors[order],
        node_forces_on_rod_world_n=nodes,mesh_wrenches_on_mesh_world=meshes,
        mesh_positions_world_m=np.asarray(positions,np.float32).reshape(-1,3)[order],
        mesh_quaternions_xyzw=np.asarray(quaternions,np.float32).reshape(-1,4)[order],
        surface_gap_m=np.full(len(ids),-.004,np.float32),
        normal_force_n=np.arange(len(ids),dtype=np.float32)[order]+7,
        sample_source="cuda_surfaceReadout_same_end_substep_evaluation",
        normal_force_semantics="unaveraged_penalty_normal_force_before_friction",
        owner_time_s_before=1/480,owner_time_s_after=1/480,
        owner_time_source="native_host_owner_metadata_at_readback",substep_timestamp_s=None,last_error=0)
    identity=dict(run_uuid="cpu-only",env_id=0,episode_uid="episode",reset_id=1,rod_uid="rod",
        physics_step=3,rod_tick=2,timestamp_s=3/480,solver_time_s=1/480,clock_valid=True)
    return v2.parse_pair_observations(raw,dt_s=1/480,identity=identity,source_instance_id=123,
        mesh_names=MESH_NAMES,source_profile=profile,evidence_kind="synthetic_cpu")

def test_module_is_separate_and_opt_in():
    assert api().VERSION=="dh116-native-pair-original-atlas-v1"

def test_locked_atlas_has_ten_original_regions_and_sources(atlas):
    assert set(atlas.regions)==set(MESH_NAMES)-{"table"}
    assert set(atlas.source_basis["geometry"])=={"left","right"}
    for row in atlas.regions.values():
        assert row["half_widths_m"]==[.004,.0035]
        assert row["surface_tolerance_m"]==.001 and row["normal_cosine_min"]==.8
    assert atlas.source_basis["geometry"]["left"]["usd"]!=atlas.source_basis["geometry"]["right"]["usd"]

def test_actual_pair_pose_classifies_both_sides_and_preserves_sorted_identity(atlas,profile):
    c=capture(atlas,profile);out=api().classify_observations(c,source_profile=profile,atlas=atlas)
    assert len(out["pairs"])==3 and out["summary"]["hand_pairs"]==2
    for got,pair,obs in zip(out["pairs"],c["legacy_capture"]["pairs"],c["observations"]):
        assert got["identity"]==c["identity"]
        assert (got["solver_substep"],got["node_index"],got["mesh_index"],got["mesh_name"])==(
            pair["substep"],pair["node_index"],pair["mesh_index"],pair["mesh_name"])
        assert got["actual_mesh_position_world_m"]==obs["mesh_position_world_m"]
        assert got["actual_mesh_quaternion_xyzw"]==obs["mesh_quaternion_xyzw"]
        assert got["substep_timestamp_s"] is None
        assert got["impulse_on_rod_ns"]==pair["impulse_on_rod_ns"]
        if pair["mesh_name"]=="table":
            assert got["region"] is None and got["classification_status"]=="not_a_hand_pad"
        else:
            assert got["region"]["accepted_by_original_patch"]
            assert abs(got["region"]["plane_signed_distance_m"])<1e-7
            assert got["native_surface_gap_m"]<-.0039
            assert got["native_penalty_normal_force_n"]!=got["dot_total_force_normal_n"]
            T=np.eye(4);T[:3,:3]=Rotation.from_quat(obs["mesh_quaternion_xyzw"]).as_matrix()
            T[:3,3]=obs["mesh_position_world_m"]
            assert got["region"]==classify_pair(pair,T,atlas.regions[pair["mesh_name"]])
    assert out["owner_time_s_before"]==out["owner_time_s_after"]==1/480
    assert out["raw_sha256"]==c["raw_sha256"] and out["capture_counts"]["overflow"] is False
    assert out["summary"]["accepted_original_region_pairs"]==2
    for row in [out,*out["pairs"]]:
        assert row["effective_support"] is row["fixed_material_slip_m"] is None
        for key in ("native_qualification","positive_contact_qualified","negative_contact_qualified","teacher_admission"):
            assert row[key] is False

@pytest.mark.parametrize("u,d,sign,load,reason",[
    (.008,0.,1.,2.,"outside_inner_ellipse"),(0.,.002,1.,2.,"outside_source_surface"),
    (0.,0.,-1.,2.,"wrong_normal"),(0.,0.,1.,0.,"no_positive_compression"),
])
def test_original_rejection_conditions_stay_separate(atlas,profile,u,d,sign,load,reason):
    c=capture(atlas,profile,[("left.finger14_link",u,d,sign,load)])
    row=api().classify_observations(c,source_profile=profile,atlas=atlas)["pairs"][0]
    assert reason in row["region"]["overlap_rejections"]
    assert row["region"]["accepted_by_original_patch"] is False
    assert row["dot_total_force_normal_n"]==pytest.approx(load,abs=1e-6)

def test_endpoint_pose_cannot_replace_actual_sample_pose(atlas,profile):
    c=capture(atlas,profile,[("left.finger14_link",0.,0.,1.,2.)])
    got=api().classify_observations(c,source_profile=profile,atlas=atlas)["pairs"][0]
    assert got["region"]["accepted_by_original_patch"]
    assert not classify_pair(c["legacy_capture"]["pairs"][0],np.eye(4),
        atlas.regions["left.finger14_link"])["accepted_by_original_patch"]

@pytest.mark.parametrize("edit",[
    lambda c:c["observations"][0].pop("mesh_position_world_m"),
    lambda c:c["observations"][0].update(mesh_name="right.finger14_link"),
    lambda c:c["observations"].reverse(),
    lambda c:c["raw"].update(overflow=True),
    lambda c:c.pop("source_profile"),
    lambda c:c["source_profile"].update(backend_sha256="0"*64),
])
def test_unvalidated_or_misaligned_capture_is_rejected(atlas,profile,edit):
    c=capture(atlas,profile);edit(c)
    with pytest.raises((ValueError,KeyError)):
        api().classify_observations(c,source_profile=profile,atlas=atlas)

def test_missing_or_changed_atlas_source_rejected(atlas,profile):
    c=capture(atlas,profile)
    with pytest.raises(ValueError):api().classify_observations(c,source_profile=profile,atlas=None)
    changed=deepcopy(atlas);changed._regions["left.finger14_link"]["surface_tolerance_m"]=.002
    with pytest.raises(ValueError):api().classify_observations(c,source_profile=profile,atlas=changed)
    changed=deepcopy(atlas);changed._source_basis={}
    with pytest.raises(ValueError):api().classify_observations(c,source_profile=profile,atlas=changed)
    with pytest.raises(ValueError):api().classify_observations(c,source_profile={},atlas=atlas)

def test_empty_capture_does_not_become_negative_evidence(atlas,profile):
    out=api().classify_observations(capture(atlas,profile,[]),source_profile=profile,atlas=atlas)
    assert out["pairs"]==[] and out["summary"]["hand_pairs"]==0
    assert out["negative_contact_qualified"] is False

def test_only_owned_output_and_disk_roundtrip(atlas,profile):
    c=capture(atlas,profile);out=api().classify_observations(c,source_profile=profile,atlas=atlas)
    saved=_digest(out);c["raw"]["vectors"][:]=99
    copy=atlas.regions;copy.clear()
    assert _digest(out)==saved and len(atlas.regions)==10
    assert _digest(decode_owned(json.loads(json.dumps(encode_owned(out)))))==saved

def test_output_retains_full_profile_and_original_predicate_source(atlas,profile):
    out=api().classify_observations(capture(atlas,profile),source_profile=profile,atlas=atlas)
    assert out["source_profile"]==profile
    assert out["pair_consumer_source_sha256"]
    assert "tabletop_effective_pad_runtime.py" in atlas.source_basis["implementation_sha256"]

def test_overlap_reasons_are_preserved_without_new_contact_qualification(atlas,profile):
    c=capture(atlas,profile,[("left.finger14_link",.008,.002,-1.,2.)])
    row=api().classify_observations(c,source_profile=profile,atlas=atlas)["pairs"][0]
    assert set(row["region"]["overlap_rejections"])=={
        "outside_inner_ellipse","outside_source_surface","wrong_normal"}
    assert row["positive_contact_qualified"] is False
