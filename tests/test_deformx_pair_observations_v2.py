"""Synthetic CPU packets only; no imported backend, GPU, or native source qualification."""
from copy import deepcopy
import gc
import json
from pathlib import Path
import weakref
import numpy as np
import pytest
from go2_pvcnn.control.dh116 import tabletop_deformx_pair_observations_v2 as v2
from go2_pvcnn.control.dh116.tabletop_complete_pair_capture import RAW_KEYS,project
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest,encode_owned,decode_owned

def profile():
    return dict(version="synthetic-profile-for-cpu",
        backend_path=v2.BACKEND_PATH,backend_sha256=v2.BACKEND_SHA,
        source_tree_sha256=v2.SOURCE_TREE_SHA,source_file_count=72,
        patch_delta_sha256=v2.PATCH_SHA,sources={v2.BACKEND_PATH:v2.BACKEND_SHA},
        native_qualification=False)

def identity(step=3,tick=2,time=1/480):
    return dict(run_uuid="cpu-only",env_id=0,episode_uid="episode",reset_id=1,
        rod_uid="rod",physics_step=step,rod_tick=tick,timestamp_s=step/480,
        solver_time_s=time,clock_valid=True)

def raw():
    points=np.array([[.02,.01,.3],[0,0,.3]],np.float32)
    poses=np.array([[.01,.02,.3],[-.01,0,.3]],np.float32)
    normals=np.array([[0,0,1],[0,0,1]],np.float32)
    forces=np.array([[.2,0,4],[.1,0,2]],np.float32)
    torque=np.cross(points-poses,-forces)
    ids=np.array([[1,0,1],[0,1,0]],np.int32)
    nodes=np.zeros((2,3),np.float32);meshes=np.zeros((2,6),np.float32)
    np.add.at(nodes,ids[:,1],forces/2)
    np.add.at(meshes,ids[:,2],np.concatenate((-forces,torque),axis=1)/2)
    return dict(schema="dh116.sdf_pair_samples.v2",substeps=2,mesh_count=2,
        capacity=8,attempted=2,overflow=False,indices=ids,
        vectors=np.concatenate((points,normals,forces,torque),axis=1),
        mesh_positions_world_m=poses,
        mesh_quaternions_xyzw=np.array([[0,0,2**-.5,2**-.5],[0,0,0,1]],np.float32),
        surface_gap_m=np.array([-1e-4,-.002],np.float32),
        normal_force_n=np.array([4.000001,2.000001],np.float32),
        sample_source="cuda_surfaceReadout_same_end_substep_evaluation",
        normal_force_semantics="unaveraged_penalty_normal_force_before_friction",
        owner_time_s_before=1/480,owner_time_s_after=1/480,
        owner_time_source="native_host_owner_metadata_at_readback",
        substep_timestamp_s=None,last_error=0,node_forces_on_rod_world_n=nodes,
        mesh_wrenches_on_mesh_world=meshes)

def parse(packet=None,**kw):
    args=dict(dt_s=1/480,identity=identity(),source_instance_id=123,
        mesh_names=("left.finger14_link","table"),source_profile=profile(),
        evidence_kind="synthetic_cpu")
    args.update(kw)
    return v2.parse_pair_observations(raw() if packet is None else packet,**args)

def test_unsorted_native_rows_share_exact_order_with_all_new_fields():
    p=raw();r=parse(p)
    old=project({k:p[k] for k in RAW_KEYS},dt_s=1/480,physics_step=3,rod_tick=2,
        mesh_names=("left.finger14_link","table"))
    assert r["legacy_capture"]==old
    assert [x["mesh_name"] for x in r["observations"]]==["left.finger14_link","table"]
    for row,index in zip(r["observations"],[1,0]):
        assert row["mesh_position_world_m"]==p["mesh_positions_world_m"][index].tolist()
        assert row["mesh_quaternion_xyzw"]==p["mesh_quaternions_xyzw"][index].tolist()
        assert row["native_surface_gap_m"]==float(p["surface_gap_m"][index])
        assert row["native_penalty_normal_force_n"]==float(p["normal_force_n"][index])
    assert r["observations"][0]["dot_total_force_normal_n"]==2.
    assert r["observations"][0]["native_penalty_normal_force_n"]!=2.
    assert r["observations"][0]["native_surface_gap_m"]<-.001
    assert r["observations"][0]["original_region_membership"] is None
    assert r["observations"][0]["effective_support"] is None
    assert r["raw"]["capacity"]==8 and r["raw"]["attempted"]==2
    assert r["raw"]["substep_timestamp_s"] is None
    assert v2.validate_pair_observations(r,source_profile=profile())["complete"]
    for key in ("native_qualification","positive_contact_qualified",
                "negative_contact_qualified","teacher_admission"):
        assert r[key] is False

def test_owned_raw_and_cpu_json_round_trip_have_no_borrowed_views():
    p=raw();ref=weakref.ref(p["vectors"]);r=parse(p)
    expected=_digest(r)
    p["mesh_positions_world_m"][:]=99
    p["vectors"][:]=99
    assert _digest(r)==expected
    del p;gc.collect();assert ref() is None
    copy=decode_owned(json.loads(json.dumps(encode_owned(r))))
    assert v2.validate_pair_observations(copy,source_profile=profile())["complete"]

@pytest.mark.parametrize("edit",[
    lambda p:p.update(overflow=True),lambda p:p.update(attempted=9),
    lambda p:p.update(capacity=True),lambda p:p["indices"].__setitem__(0,p["indices"][1]),
    lambda p:p["indices"].__setitem__((0,0),2),
    lambda p:p["indices"].__setitem__((0,1),2),
    lambda p:p["indices"].__setitem__((0,2),2),
    lambda p:p["vectors"].__setitem__((0,5),.5),
    lambda p:p["vectors"].__setitem__((0,8),99),
    lambda p:p["node_forces_on_rod_world_n"].__setitem__((0,0),99),
    lambda p:p["mesh_wrenches_on_mesh_world"].__setitem__((0,0),99),
    lambda p:p["mesh_positions_world_m"].__setitem__((0,0),np.nan),
    lambda p:p.update(mesh_positions_world_m=np.zeros((1,3),np.float32)),
    lambda p:p.update(mesh_positions_world_m=np.zeros((2,3),np.float64)),
    lambda p:p["mesh_quaternions_xyzw"].__setitem__(0,[0,0,0,0]),
    lambda p:p["surface_gap_m"].__setitem__(0,0),
    lambda p:p["surface_gap_m"].__setitem__(0,np.inf),
    lambda p:p["normal_force_n"].__setitem__(0,-1),
    lambda p:p.update(last_error=None),lambda p:p.update(last_error=True),
    lambda p:p.update(last_error=1),lambda p:p.update(owner_time_s_after=2/480),
    lambda p:p.update(owner_time_s_before=0.,owner_time_s_after=0.),
    lambda p:p.update(substep_timestamp_s=1/5760),
    lambda p:p.update(sample_source="interpolated_endpoint"),
    lambda p:p.update(owner_time_source="nominal_step_div_480"),
    lambda p:p.pop("surface_gap_m"),
])
def test_incomplete_unknown_or_misaligned_native_packet_is_rejected(edit):
    p=raw();edit(p)
    with pytest.raises((ValueError,KeyError)):parse(p)

def test_actual_mesh_origin_reconciles_with_recorded_torque():
    p=raw();p["mesh_positions_world_m"][0,0]+=.1
    with pytest.raises(ValueError,match="torque"):parse(p)

@pytest.mark.parametrize("edit",[
    lambda r:r["observations"][0].update(native_surface_gap_m=-99),
    lambda r:r["observations"].reverse(),
    lambda r:r["legacy_capture"]["pairs"][0].update(normal_force_on_rod_n=99),
    lambda r:r["raw"]["normal_force_n"].__setitem__(0,99),
    lambda r:r.update(native_qualification=True),
    lambda r:r["identity"].update(physics_step=4),
    lambda r:r.update(raw_sha256="0"*64),
])
def test_disk_consumer_reprojects_raw_and_rejects_tampering(edit):
    r=parse();edit(r)
    with pytest.raises(ValueError):v2.validate_pair_observations(r,source_profile=profile())

def test_profile_is_bound_but_never_self_qualifies_the_source():
    p=profile();p["backend_sha256"]="0"*64
    with pytest.raises(ValueError,match="profile"):parse(source_profile=p)
    r=parse();p=profile();p["version"]="changed"
    with pytest.raises(ValueError,match="profile"):
        v2.validate_pair_observations(r,source_profile=p)
    assert parse()["source_profile_verification"]=="required_from_outer_source_basis_and_native_receipt"

def test_empty_complete_capture_does_not_grant_negative_contact():
    p=raw()
    p.update(attempted=0,indices=np.empty((0,3),np.int32),vectors=np.empty((0,12),np.float32),
        mesh_positions_world_m=np.empty((0,3),np.float32),
        mesh_quaternions_xyzw=np.empty((0,4),np.float32),
        surface_gap_m=np.empty((0,),np.float32),normal_force_n=np.empty((0,),np.float32),
        node_forces_on_rod_world_n=np.zeros((2,3),np.float32),
        mesh_wrenches_on_mesh_world=np.zeros((2,6),np.float32))
    r=parse(p)
    assert r["observations"]==[]
    assert v2.validate_pair_observations(r,source_profile=profile())["complete"]
    assert r["negative_contact_qualified"] is False

def sequence():
    return v2.PairObservationSequence(source_profile=profile(),reset_identity=identity(2,0,0),
        source_instance_id=123,mesh_names=("left.finger14_link","table"),
        expected_substeps=2,evidence_kind="synthetic_cpu")

def test_sequence_accepts_real_clock_progress_and_refuses_same_frame_replay():
    s=sequence();s.consume(raw(),identity=identity(),source_instance_id=123)
    p=raw();p.update(owner_time_s_before=2/480,owner_time_s_after=2/480)
    r=s.consume(p,identity=identity(4,4,2/480),source_instance_id=123)
    assert r["identity"]["physics_step"]==4
    with pytest.raises(ValueError):s.consume(p,identity=identity(4,4,2/480),source_instance_id=123)
    with pytest.raises(RuntimeError,match="failed"):s.consume(p,identity=identity(5,6,3/480),source_instance_id=123)

@pytest.mark.parametrize("change",["owner","sdk","tick","episode","clock","substeps"])
def test_wrong_owner_frame_or_sequence_fails_closed(change):
    s=sequence();p=raw();i=identity();owner=123
    if change=="owner":owner=124
    elif change=="sdk":i.update(physics_step=4,timestamp_s=4/480)
    elif change=="tick":i["rod_tick"]=4
    elif change=="episode":i["episode_uid"]="another"
    elif change=="clock":p.update(owner_time_s_before=2/480,owner_time_s_after=2/480)
    elif change=="substeps":p["substeps"]=3
    with pytest.raises(ValueError):s.consume(p,identity=i,source_instance_id=owner)
    with pytest.raises(RuntimeError,match="failed"):s.consume(raw(),identity=identity(),source_instance_id=123)

def test_real_built_v2_schema_is_the_contract_without_importing_extension():
    from scripts import prepare_dh116_deformx_mesh_observations as prep
    text=prep.planned_sources(Path(__file__).resolve().parents[1])["csrc/bindings.cpp"]
    method=text.split('.def("contact_pair_samples_v2",',1)[1].split('.def("mesh_state_snapshot",',1)[0]
    import re
    keys=set(re.findall(r'out\["([^"]+)"\]',method))
    assert keys==v2.RAW_KEYS

def test_generator_mesh_names_are_owned_once_for_projection_and_disk():
    r=parse(mesh_names=iter(("left.finger14_link","table")))
    assert r["mesh_names"]==["left.finger14_link","table"]
    assert v2.validate_pair_observations(r,source_profile=profile())["complete"]

def test_same_mesh_and_substep_cannot_claim_two_native_poses():
    p=raw()
    p["indices"]=np.array([[0,0,0],[0,1,0]],np.int32)
    p["mesh_wrenches_on_mesh_world"][:]=0
    np.add.at(p["mesh_wrenches_on_mesh_world"],p["indices"][:,2],
        np.concatenate((-p["vectors"][:,6:9],p["vectors"][:,9:12]),axis=1)/2)
    with pytest.raises(ValueError,match="same native mesh/substep"):parse(p)

@pytest.mark.parametrize("field",["run_uuid","rod_uid","reset_id","env_id"])
def test_sequence_preserves_every_episode_identity_field(field):
    s=sequence();i=identity()
    i[field]=(i[field]+1) if isinstance(i[field],int) else i[field]+"changed"
    with pytest.raises(ValueError):s.consume(raw(),identity=i,source_instance_id=123)

def test_profile_success_claim_cannot_qualify_synthetic_packet():
    p=profile();p.update(native_qualification=True,teacher_qualified=True)
    r=parse(source_profile=p)
    assert r["evidence_kind"]=="synthetic_cpu"
    assert r["native_qualification"] is False and r["teacher_admission"] is False

def test_surface_contact_lever_already_includes_radius_with_tangential_force():
    p=raw();radius=np.float32(.005)
    p["mesh_positions_world_m"][0]=[0,0,.3]
    point=p["mesh_positions_world_m"][0]-radius*p["vectors"][0,3:6]
    p["vectors"][0,:3]=point
    p["vectors"][0,6:9]=[1,0,4]
    p["vectors"][0,9:12]=np.cross(-radius*p["vectors"][0,3:6],-p["vectors"][0,6:9])
    p["node_forces_on_rod_world_n"][:]=0
    p["mesh_wrenches_on_mesh_world"][:]=0
    np.add.at(p["node_forces_on_rod_world_n"],p["indices"][:,1],p["vectors"][:,6:9]/2)
    np.add.at(p["mesh_wrenches_on_mesh_world"],p["indices"][:,2],
        np.concatenate((-p["vectors"][:,6:9],p["vectors"][:,9:12]),axis=1)/2)
    r=parse(p)
    assert r["legacy_capture"]["pairs"][1]["torque_on_mesh_about_origin_world_nm"][1]==pytest.approx(.005)
    from scripts import prepare_dh116_deformx_mesh_observations as prep
    source=prep.planned_sources(Path(__file__).resolve().parents[1])[
        "third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh"]
    assert "const vec3 r = (pos - P.pos) + dx - R * n;" in source
    assert "const vec3 tq = cross(r, -fm);" in source
    assert "tid % data->sdfEnvVerts, m, P.pos + r, n, fm, tq" in source

@pytest.mark.parametrize("names",["ab",{"left.finger14_link","table"},{"left.finger14_link":0,"table":1}])
def test_sequence_refuses_unordered_or_string_mesh_names(names):
    with pytest.raises(ValueError,match="ordered mesh"):
        v2.PairObservationSequence(source_profile=profile(),reset_identity=identity(2,0,0),
            source_instance_id=123,mesh_names=names,expected_substeps=2,evidence_kind="synthetic_cpu")
