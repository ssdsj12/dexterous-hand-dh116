"""CPU USD and API doubles; no PhysX/Isaac launch or native qualification."""
from copy import deepcopy
import importlib
from pathlib import Path


import numpy as np
import pytest

from scripts.audit_dh116_pinch_initial_pose import load_geometry
from test_tripod_cooked_geometry import cube

ROOT = Path(__file__).resolve().parents[1]


def api():
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_pinch_initial_geometry_v2")


@pytest.fixture(scope="module")
def geometry():
    return load_geometry(ROOT)


class Handle:
    def __init__(self,**kwargs):self.__dict__.update(kwargs)


class Cooking:
    def __init__(self):
        self.calls=[]
        self.owner_count=1
        self.mesh_count=1
        self.data=cube()[0]
    def get_nb_convex_mesh_data(self,path):
        self.calls.append(("count",path))
        return self.mesh_count if path.endswith("/mesh") else self.owner_count
    def get_convex_mesh_data(self,path,index):
        self.calls.append(("data",path,index))
        return self.data
    def request_convex_collision_representation(self,*args,**kwargs):
        raise AssertionError("read-only capture must never request cooking")


def setup(geometry):
    from pxr import Usd,UsdGeom
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    root="/World/DeformXPinch/leftHand"
    stage.DefinePrim(root).GetReferences().AddReference(geometry["usd"]["path"])
    names=list(reversed(geometry["source_colliders"]))
    poses=np.zeros((1,17,7));poses[:,:,2]=.25;poses[:,:,6]=1.
    view=Handle(prim_paths=[root],link_paths=[[root+"/"+n for n in names]],
        get_link_transforms=lambda:poses)
    hand=Handle(body_names=names,root_physx_view=view)
    clock=dict(sdk_step=2,sdk_time_s=2/480)
    cooking=Cooking()
    return stage,hand,poses,clock,cooking


def capture(geometry,fixture,**extra):
    stage,hand,poses,clock,cooking=fixture
    kwargs=dict(cooking_interface=cooking,read_clock=lambda:deepcopy(clock),
        table_center_m=[0,0,0],table_size_m=[2,2,.2],
        rod_nodes_w=[[-.1,0,.12],[.1,0,.12]],rod_radius_m=.005,
        minimum_table_gap_m=0.,minimum_rod_gap_m=0.)
    kwargs.update(extra)
    return api().capture_pinch_initial_geometry(stage,hand,geometry,**kwargs)


def test_all17_actual_poses_and5_locked_hulls_are_separate_from_stored_data(geometry):
    fixture=setup(geometry);stage,hand,poses,clock,cooking=fixture
    before=stage.GetRootLayer().ExportToString()
    result=capture(geometry,fixture)
    assert result["initial_geometry_passed"],result["reasons"]
    assert len(result["source_colliders"])==17 and len(result["locked_pads"])==5
    assert len(result["stored_cooking"])==17 and len(cooking.calls)==34
    assert result["stored_cooking_complete"]
    assert result["native_shape_attachment_proven"] is result["contact_free_proven"] is False
    assert result["native_qualification"] is result["teacher_admission"] is False
    assert stage.GetRootLayer().ExportToString()==before
    row=result["source_colliders"]["finger13_link"]
    assert row["body_index"]==hand.body_names.index("finger13_link")
    expected=geometry["source_colliders"]["finger13_link"]["vertices"]+[0,0,.25]
    np.testing.assert_allclose(row["world_vertices"],expected,atol=1e-10)
    assert row["table"]["top_clearance_m"]>.1
    assert all(r["coordinate_frame"]=="mesh_local_hypothesis_not_live_shape_attestation"
               for r in result["stored_cooking"].values())
    poses[:]=99
    assert result["actual_link_poses_xyzw"][0][0][2]==.25


def test_proximal_actual_table_overlap_blocks_source_gate(geometry):
    fixture=setup(geometry)
    fixture[2][0,fixture[1].body_names.index("finger13_link"),2]=0.
    result=capture(geometry,fixture)
    assert not result["initial_geometry_passed"]
    assert result["source_colliders"]["finger13_link"]["table"]["intersects"]


def test_locked_cooked_table_gate_cannot_be_replaced_by_source_gate(geometry,monkeypatch):
    # Counterfactual CPU hull verifies the independent gate; not real F4 evidence.
    changed=dict(geometry,cooked_pads=deepcopy(geometry["cooked_pads"]))
    changed["cooked_pads"]["finger14_link"]["vertices"][:,2]-=.25
    monkeypatch.setattr(api(),"load_geometry",lambda *a,**k:changed)
    fixture=setup(changed)
    result=capture(changed,fixture)
    assert result["source_geometry_passed"]
    assert not result["locked_pad_geometry_passed"]
    assert not result["initial_geometry_passed"]
    assert result["locked_pads"]["finger14_link"]["table"]["top_clearance_m"]<0


@pytest.mark.parametrize("fault",["zero","missing","bad_topology","throw"])
def test_stored_getter_failure_is_explicit_na_and_never_recooked(geometry,fault):
    fixture=setup(geometry);cooking=fixture[-1]
    if fault=="zero":cooking.owner_count=cooking.mesh_count=0
    elif fault=="missing":fixture=(*fixture[:-1],None)
    elif fault=="bad_topology":cooking.data["indices"][0]=99
    else:
        def fail(*args):raise RuntimeError("getter unavailable")
        cooking.get_nb_convex_mesh_data=fail
    result=capture(geometry,fixture)
    assert result["initial_geometry_passed"]
    assert not result["stored_cooking_complete"]
    assert all(not row["available"] and row["reasons"] for row in result["stored_cooking"].values())
    assert not result["native_shape_attachment_proven"]


def test_source_rod_gap_checks_between_nodes_not_only_endpoints(geometry):
    fixture=setup(geometry)
    name="base_link";index=fixture[1].body_names.index(name)
    v=geometry["source_colliders"][name]["vertices"]+[0,0,.25]
    center=v.mean(0)
    result=capture(geometry,fixture,rod_nodes_w=[center+[-1,0,0],center+[1,0,0]])
    assert not result["initial_geometry_passed"]
    assert result["source_colliders"][name]["rod"]["intersects_capsule"]


@pytest.mark.parametrize("fault",["clock","paths","source","offset","extra_collider"])
def test_changed_clock_or_source_identity_cannot_return_a_passing_gate(geometry,fault):
    fixture=setup(geometry);stage,hand,poses,clock,cooking=fixture
    extra={}
    if fault=="clock":
        def change():
            clock["sdk_step"]+=1;clock["sdk_time_s"]+=1/480
            return deepcopy(clock)
        extra["read_clock"]=change
    elif fault=="paths":
        hand.root_physx_view.link_paths[0][0]="/World/other/leftHand/"+hand.body_names[0]
    elif fault=="source":
        geometry=dict(geometry,usd=dict(geometry["usd"],sha256="0"*64))
    elif fault=="offset":
        from pxr import UsdGeom,Gf
        row=geometry["source_colliders"]["base_link"]["source"]
        root="/World/DeformXPinch/leftHand"
        source_root=row["body_path"].rsplit("/",1)[0]
        mesh=stage.GetPrimAtPath(root+row["collision_owner"][len(source_root):])
        instance=mesh
        while instance.IsInstanceProxy():instance=instance.GetParent()
        assert instance.IsInstance()
        path=mesh.GetPath()
        instance.SetInstanceable(False)
        stage.GetPrimAtPath(path).GetAttribute("xformOp:translate").Set(Gf.Vec3d(.1,0,0))
    else:
        from pxr import UsdGeom,UsdPhysics
        prim=UsdGeom.Cube.Define(stage,"/World/DeformXPinch/leftHand/extra").GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
    with pytest.raises((ValueError,RuntimeError)):
        capture(geometry,fixture,**extra)


def test_explicit_threshold_and_actual_rotation_are_consumed(geometry):
    fixture=setup(geometry)
    fixture[2][0,:,3:]=[0,0,np.sqrt(.5),np.sqrt(.5)]
    result=capture(geometry,fixture,minimum_table_gap_m=3.)
    assert not result["initial_geometry_passed"]
    row=result["source_colliders"]["finger13_link"]
    v=geometry["source_colliders"]["finger13_link"]["vertices"]
    np.testing.assert_allclose(row["world_vertices"],np.column_stack((-v[:,1],v[:,0],v[:,2]+.25)),atol=1e-10)


def test_exception_traceback_drops_stage_hand_view_and_callback_aliases(geometry):
    import gc,weakref
    fixture=setup(geometry);stage,hand,poses,clock,cooking=fixture
    view=hand.root_physx_view
    def fail():raise RuntimeError("native getter fault")
    view.get_link_transforms=fail
    class Clock:
        def __call__(self):return deepcopy(clock)
    reader=Clock()
    refs=[weakref.ref(x) for x in (stage,hand,view,cooking,reader)]
    caught=None
    try:
        api().capture_pinch_initial_geometry(stage,hand,geometry,
            cooking_interface=cooking,read_clock=reader,table_center_m=[0,0,0],
            table_size_m=[2,2,.2],rod_nodes_w=[[-.1,0,.12],[.1,0,.12]],rod_radius_m=.005,
            minimum_table_gap_m=0.,minimum_rod_gap_m=0.)
    except RuntimeError as error:
        caught=error
    assert caught is not None
    del fixture,stage,hand,view,cooking,reader
    gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.fixture(scope="module")
def captured_record(geometry):
    return capture(geometry,setup(geometry))


@pytest.mark.parametrize("damage",[None,"pose","path","source","vertices","gate","threshold","stored"])
def test_disk_validator_recomputes_owned_initial_geometry(geometry,captured_record,tmp_path,damage):
    import json
    record=deepcopy(captured_record)
    if damage=="pose":record["actual_link_poses_xyzw"][0][0][2]-=.001
    elif damage=="path":record["actual_link_paths"][0][0]="/World/wrong"
    elif damage=="source":record["source_sha256"]["usd"]="0"*64
    elif damage=="vertices":record["source_colliders"]["base_link"]["world_vertices"][0][2]+=1.
    elif damage=="gate":record["initial_geometry_passed"]=False
    elif damage=="threshold":record["thresholds"]["minimum_rod_gap_m"]=1.
    elif damage=="stored":record["stored_cooking"]["base_link"]["hulls"][0]["raw"]["vertices"][0][0]+=1.
    path=tmp_path/"initial-geometry.json"
    path.write_text(json.dumps(record,allow_nan=False))
    result=api().validate_pinch_initial_geometry(json.loads(path.read_text()),geometry,
        table_center_m=[0,0,0],table_size_m=[2,2,.2],
        rod_nodes_w=[[-.1,0,.12],[.1,0,.12]],rod_radius_m=.005,
        minimum_table_gap_m=0.,minimum_rod_gap_m=0.)
    assert result["valid"] is (damage is None),result["reasons"]
    assert result["native_qualification"] is False


def test_corrected_candidate_fk_cpu_surrogate_replays_without_native_claim(geometry,tmp_path):
    import json
    from scipy.spatial.transform import Rotation
    from scripts.audit_dh116_pinch_initial_pose import pose_matrix
    candidate=json.loads((ROOT/"configs/dh116_tasks/flexible_pinch_pose_candidate_v2.json").read_text())
    fixture=setup(geometry)
    delta=pose_matrix(candidate["base_pose_wxyz"])@np.linalg.inv(pose_matrix(geometry["pose"]))
    matrices={path.rsplit("/",1)[-1]:delta@matrix
              for path,matrix in geometry["fk"].poses(candidate["active_q_rad"]).items()}
    for i,name in enumerate(fixture[1].body_names):
        matrix=matrices[name]
        fixture[2][0,i,:3]=matrix[:3,3]
        fixture[2][0,i,3:]=Rotation.from_matrix(matrix[:3,:3]).as_quat()
    layout=geometry["reference"]["reference_layout"]
    kwargs=dict(table_center_m=layout["table_center"],table_size_m=layout["table_size"],
                rod_nodes_w=layout["wire_points"],rod_radius_m=geometry["old_candidate"]["rod"]["radius_m"],
                minimum_table_gap_m=0.,minimum_rod_gap_m=0.)
    record=capture(geometry,fixture,**kwargs)
    path=tmp_path/"candidate-fk-surrogate.json"
    path.write_text(json.dumps(record,allow_nan=False))
    replay=api().validate_pinch_initial_geometry(json.loads(path.read_text()),geometry,**kwargs)
    assert record["initial_geometry_passed"] and replay["valid"],replay["reasons"]
    assert replay["initial_geometry_passed"] and not record["native_qualification"]
    print(json.dumps(dict(scope="corrected_candidate_CPU_FK_surrogate_not_actual_SDK",
        source_min_table_m=min(r["table"]["top_clearance_m"] for r in record["source_colliders"].values()),
        locked_min_table_m=min(r["table"]["top_clearance_m"] for r in record["locked_pads"].values()),
        source_min_rod_m=min(r["rod"]["capsule_surface_gap_m"] for r in record["source_colliders"].values()),
        locked_min_rod_m=min(r["rod"]["capsule_surface_gap_m"] for r in record["locked_pads"].values()),
        stored_available=sum(r["available"] for r in record["stored_cooking"].values()),
        native_qualification=False)))
