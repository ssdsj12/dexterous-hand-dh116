"""CPU-only source/cooked coordinate diagnostics, not admission."""
import copy
import importlib
import hashlib
import json

import numpy as np
import pytest

NAME="scripts.audit_dh116_teacher_cooked_readback"


def row_fixture():
    from test_tripod_cooked_geometry import cube
    raw=cube()[0]
    eye=np.eye(4).tolist()
    row=dict(binding=dict(side="left",body_id=0,actual_body_path="/World/envs/env_0/leftHand/base_link",
                            source_manifest=dict(meters_per_unit=1.0)),
                source_geometry=dict(original_mesh_points=raw["vertices"]+raw["vertices"][::-1],
                    source_geometry_sha256="a"*64,live_mesh_geometry_sha256="a"*64,
                    original_mesh_to_body=copy.deepcopy(eye),live_mesh_to_body=copy.deepcopy(eye)),
                frames=dict(convention="column_vectors_world_from_body_SI_metres",
                    actual_body_link_pose_w=copy.deepcopy(eye),mesh_to_body=copy.deepcopy(eye),
                    usd_world_from_mesh_crosscheck_only=copy.deepcopy(eye)),
                cooked_hulls=[dict(raw=raw)])
    indices=raw["indices"];triangles=[]
    for polygon in raw["polygons"]:
        ids=indices[polygon["index_base"]:polygon["index_base"]+polygon["num_vertices"]]
        triangles.extend([[ids[0],ids[i],ids[i+1]] for i in range(1,len(ids)-1)])
    row["source_geometry"]["original_triangles"]=triangles
    row["binding"]["source_collision"]={"orientation":"rightHanded"}
    rehash_fixture(row)
    return row


def rehash_fixture(row):
    points=np.asarray(row["source_geometry"]["original_mesh_points"],dtype="<f8")
    indices=np.asarray(row["source_geometry"]["original_triangles"],dtype="<i8")
    digest=hashlib.sha256(b"DH116-triangle-surface-v1\0")
    digest.update(np.asarray(points.shape,dtype="<i8").tobytes())
    digest.update(points.tobytes());digest.update(indices.tobytes())
    digest.update(row["binding"]["source_collision"]["orientation"].encode("ascii"))
    sha=digest.hexdigest()
    row["source_geometry"]["source_geometry_sha256"]=sha
    row["source_geometry"]["live_mesh_geometry_sha256"]=sha
    row["binding"]["source_collision"]["geometry_sha256"]=sha


def audit(row,**kwargs):
    m=importlib.import_module(NAME)
    assert hasattr(m,"diagnose_cooked_representation"),"source/cooked coordinate auditor missing"
    return m.diagnose_cooked_representation(row,**kwargs)


def test_vertex_matching_handles_reordering_duplicates_and_unequal_counts():
    row=row_fixture();raw=row["cooked_hulls"][0]["raw"]
    # Source has repeated/reordered records; equality by array index is invalid.
    saved=copy.deepcopy(row)
    result=audit(row)
    assert result["available"] is True
    assert result["source_vertex_records"]==16 and result["source_unique_vertices"]==8
    assert result["hulls"][0]["cooked_vertices"]==8
    assert result["hulls"][0]["cooked_to_source_unique_vertex"]["max_m"]==0
    assert result["hulls"][0]["source_unique_to_cooked_vertex"]["max_m"]==0
    assert result["native_qualification"] is False and row==saved


def test_vertex_distance_does_not_claim_surface_error_or_geometry_corruption():
    row=row_fixture()
    row["source_geometry"]["original_mesh_points"].append([0,0,0])
    rehash_fixture(row)
    result=audit(row)
    assert result["available"] is True
    assert result["hulls"][0]["source_unique_to_cooked_vertex"]["max_m"]>0
    assert result["vertex_distances_are_surface_errors"] is False
    assert result["cooking_abi_alignment_proven"] is False
    assert result["live_shape_attachment_proven"] is False


def test_column_transform_roundtrip_and_world_origin_are_separate_from_local_matching():
    row=row_fixture();angle=.37;c,s=np.cos(angle),np.sin(angle)
    body=np.array([[c,-s,0,1000.],[s,c,0,-500.],[0,0,1,20.],[0,0,0,1]])
    mesh=np.eye(4);mesh[:3,3]=[.01,-.02,.03]
    row["frames"]["actual_body_link_pose_w"]=body.tolist()
    row["frames"]["mesh_to_body"]=mesh.tolist()
    row["frames"]["usd_world_from_mesh_crosscheck_only"]=(body@mesh).tolist()
    row["source_geometry"]["live_mesh_to_body"]=mesh.tolist()
    row["source_geometry"]["original_mesh_to_body"]=mesh.tolist()
    result=audit(row)
    assert result["available"] is True
    assert result["hulls"][0]["world_mesh_roundtrip_max_m"]<1e-10
    assert result["world_mesh_usd_crosscheck_max_abs_entry_delta"]==pytest.approx(0)
    assert result["hulls"][0]["cooked_to_source_unique_vertex"]["max_m"]==0


@pytest.mark.parametrize("bad",["transposed","nonrigid","wrong_convention","hash","units","nonfinite"])
def test_missing_or_wrong_coordinate_provenance_is_not_repaired_or_qualified(bad):
    row=row_fixture()
    if bad=="transposed":
        mat=np.eye(4);mat[0,3]=.2;row["frames"]["actual_body_link_pose_w"]=mat.T.tolist()
    elif bad=="nonrigid":row["frames"]["actual_body_link_pose_w"][0][0]=2.
    elif bad=="wrong_convention":row["frames"]["convention"]="row_vectors"
    elif bad=="hash":row["source_geometry"]["live_mesh_geometry_sha256"]="b"*64
    elif bad=="units":row["binding"]["source_manifest"]["meters_per_unit"]=.01
    else:row["source_geometry"]["original_mesh_points"][0][0]=float("inf")
    result=audit(row)
    assert result["available"] is False and result["error"]
    assert result["native_qualification"] is False


def test_loose_plane_counterfactual_never_changes_protected_validation():
    row=row_fixture();raw=row["cooked_hulls"][0]["raw"]
    raw["polygons"][0]["plane"][3]+=.00005
    saved=copy.deepcopy(row);result=audit(row)
    hull=result["hulls"][0]
    assert hull["existing_validator_closed_convex"] is False
    assert hull["plane_threshold_study"][0]["failing_faces"]==1
    suggested=next(x for x in hull["plane_threshold_study"] if x["name"]=="image_suggested_1e_4_m")
    assert suggested["failing_faces"]==0 and suggested["admission_applied"] is False
    assert result["validator_tolerance_changed"] is False and row==saved


@pytest.mark.parametrize("chunk",[0,False,65537])
def test_invalid_batch_budget_is_rejected(chunk):
    with pytest.raises(ValueError,match="chunk"):
        audit(row_fixture(),query_chunk_size=chunk)


def test_small_and_large_query_batches_return_same_numeric_diagnostics():
    row=row_fixture()
    a=audit(row,query_chunk_size=3);b=audit(row,query_chunk_size=65536)
    a.pop("query_chunk_size");b.pop("query_chunk_size")
    assert a==b
    assert json.loads(json.dumps(a,allow_nan=False))==a


def test_opt_in_cli_uses_actual_packet_clock_and_preserves_default_schema(tmp_path):
    row=row_fixture()
    source=tmp_path/"packet.json"
    clock=dict(sdk_step=2,sdk_time_s=2/480)
    packet=dict(capture=dict(complete=False,colliders=[row]),
                sdk_clock_before=clock,sdk_clock_after=copy.deepcopy(clock))
    source.write_text(json.dumps(packet))
    m=importlib.import_module(NAME)
    legacy=tmp_path/"legacy.json";new=tmp_path/"representation.json"
    assert m.main(["--capture",str(source),"--output",str(legacy)])==0
    assert m.main(["--capture",str(source),"--output",str(new),"--compare-source-vertices"])==0
    old=json.loads(legacy.read_text());current=json.loads(new.read_text())
    assert "representation_audit" not in old and old["schema"]=="dh116-offline-cooked-readback-audit-v1"
    assert current["representation_audit"]["clock_before"]==clock
    assert current["representation_audit"]["same_packet_sdk_boundary"] is True
    assert current["bodies"][0]["coordinate_diagnostics"]["available"] is True
    assert current["native_qualification"] is False
    with pytest.raises(FileExistsError):
        m.main(["--capture",str(source),"--output",str(new),"--compare-source-vertices"])


def test_mismatched_clock_cannot_report_same_capture_boundary():
    m=importlib.import_module(NAME)
    packet=dict(capture=dict(complete=False,colliders=[row_fixture()]),
                sdk_clock_before=dict(sdk_step=2,sdk_time_s=2/480),
                sdk_clock_after=dict(sdk_step=3,sdk_time_s=3/480))
    result=m.analyze_capture(packet,compare_source_vertices=True)
    assert result["representation_audit"]["same_packet_sdk_boundary"] is False
    assert result["native_qualification"] is False


def test_finite_input_matrices_with_overflowing_composition_fail_closed():
    row=row_fixture()
    row["frames"]["actual_body_link_pose_w"][0][3]=1e308
    for container,key in [(row["frames"],"mesh_to_body"),
            (row["source_geometry"],"live_mesh_to_body"),
            (row["source_geometry"],"original_mesh_to_body")]:
        container[key][0][3]=1e308
    # Old path returns available=True with NaN after combining finite matrices.
    with np.errstate(over="ignore",invalid="ignore"):
        result=audit(row)
    assert result["available"] is False and result["error"]
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize("bad",["point","triangle_bounds","triangle_dtype","orientation","binding_hash"])
def test_equal_saved_hash_strings_cannot_hide_modified_source_content(bad):
    row=row_fixture()
    if bad=="point":row["source_geometry"]["original_mesh_points"][0][0]+=.0001
    elif bad=="triangle_bounds":row["source_geometry"]["original_triangles"][0][0]=999
    elif bad=="triangle_dtype":row["source_geometry"]["original_triangles"][0][0]=0.5
    elif bad=="orientation":row["binding"]["source_collision"]["orientation"]="leftHanded"
    else:row["binding"]["source_collision"]["geometry_sha256"]="f"*64
    result=audit(row)
    assert result["available"] is False and result["error"]
    assert result["native_qualification"] is False
