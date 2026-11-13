"""Offline source/identity cell only; synthetic pair data never grant capability."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib
import json
import pytest

def api():
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_static_table_report_cell")

@pytest.fixture
def cell():
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import _load_offline_usd_schema
    _load_offline_usd_schema()
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.Xform.Define(stage,"/World/Table")
    table=UsdGeom.Cube.Define(stage,"/World/Table/geometry/mesh").GetPrim()
    UsdPhysics.CollisionAPI.Apply(table)
    body=UsdGeom.Sphere.Define(stage,"/World/Probe").GetPrim()
    UsdPhysics.CollisionAPI.Apply(body);UsdPhysics.RigidBodyAPI.Apply(body)
    PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr().Set(0.)
    kwargs=dict(table_root="/World/Table",moving_actor="/World/Probe",
                moving_collider="/World/Probe",stage_id=7)
    return stage,kwargs

def raw(reverse=False):
    header=dict(actor0="/World/Probe",collider0="/World/Probe",
        actor1="/World/Table/geometry/mesh",collider1="/World/Table/geometry/mesh",
        stage_id=7,contact_data_offset=0,num_contact_data=1,
        friction_anchors_offset=0,num_friction_anchors_data=1,type=0)
    if reverse:
        for field in ("actor","collider"):header[field+"0"],header[field+"1"]=header[field+"1"],header[field+"0"]
    return dict(headers=[header],contacts=[dict(position=[0.,0.,.3],impulse=[0.,0.,.1],
        normal=[0.,0.,1.],separation=-.0001)],
        friction_anchors=[dict(position=[0.,0.,.3],impulse=[.01,0.,0.])],friction_source_available=True)

@pytest.mark.parametrize("reverse",[False,True])
def test_exact_static_collider_both_orders_is_only_observation(cell,reverse):
    identity=api().source_identity(cell[0],**cell[1])
    result=api().classify_full_report(raw(reverse),identity)
    assert result["exact_table_contact_samples"]==1
    assert result["nonzero_contact_impulse_samples"]==1
    assert result["table_actor_identity_verified"] is None
    assert result["force_on_moving_body_world_n"] is None
    assert result["absence_verified"] is False and result["native_cell_qualified"] is False
    json.dumps(result,allow_nan=False)

@pytest.mark.parametrize("field,value",[("actor0","/World/Wrong"),("collider0","/World/Wrong"),
    ("collider1","/World/Table"),("collider1","/World/Table/other")])
def test_wrong_body_or_table_root_never_counts_as_positive(cell,field,value):
    identity=api().source_identity(cell[0],**cell[1]);report=raw()
    report["headers"][0][field]=value
    result=api().classify_full_report(report,identity)
    assert result["exact_table_contact_samples"]==0 and result["unmatched_headers"]==1

@pytest.mark.parametrize("fault",["stage","range","nan","normal","lost_data"])
def test_malformed_raw_buffers_fail_closed(cell,fault):
    identity=api().source_identity(cell[0],**cell[1]);report=raw()
    if fault=="stage":report["headers"][0]["stage_id"]=8
    elif fault=="range":report["headers"][0]["num_contact_data"]=2
    elif fault=="nan":report["contacts"][0]["position"][0]=float("nan")
    elif fault=="normal":report["contacts"][0]["normal"]=[0.,0.,0.]
    else:report["headers"][0]["type"]=1
    with pytest.raises(ValueError):api().classify_full_report(report,identity)

def test_empty_full_report_cannot_prove_absence(cell):
    identity=api().source_identity(cell[0],**cell[1])
    result=api().classify_full_report(dict(headers=[],contacts=[],friction_anchors=[],friction_source_available=True),identity)
    assert result["exact_table_contact_samples"]==0
    assert result["absence_verified"] is False and result["native_cell_qualified"] is False

@pytest.mark.parametrize("fault",["kinematic","missing_reporter","table_rigid","collision_disabled"])
def test_source_preflight_rejects_unsupported_cell(cell,fault):
    from pxr import UsdPhysics,PhysxSchema
    body=cell[0].GetPrimAtPath("/World/Probe")
    if fault=="kinematic":UsdPhysics.RigidBodyAPI(body).CreateKinematicEnabledAttr().Set(True)
    elif fault=="missing_reporter":body.RemoveAPI(PhysxSchema.PhysxContactReportAPI)
    elif fault=="table_rigid":UsdPhysics.RigidBodyAPI.Apply(cell[0].GetPrimAtPath("/World/Table"))
    else:UsdPhysics.CollisionAPI(body).CreateCollisionEnabledAttr().Set(False)
    with pytest.raises(ValueError):api().source_identity(cell[0],**cell[1])

def test_real_callback_copy_preserves_arrival_only_and_buffer_ownership(cell):
    from go2_pvcnn.control.dh116.tabletop_stereo_contact_calibration import ContactReportReceiptBuffer
    identity=api().source_identity(cell[0],**cell[1])
    report=raw()
    arrays=([SimpleNamespace(**r) for r in report["headers"]],
            [SimpleNamespace(**r) for r in report["contacts"]],
            [SimpleNamespace(**r) for r in report["friction_anchors"]])
    buffer=ContactReportReceiptBuffer(clock_reader=lambda:(3,3/480),path_decoder=str)
    buffer.begin_step(3);buffer.record("full",arrays);buffer.end_step()
    arrays[1][0].impulse[2]=900.
    receipt=buffer.drain()[0]
    assert receipt["raw_report"]["contacts"][0]["impulse"][2]==.1
    assert receipt["source_clock_association"]=="arrival_only_with_step_call_window_not_contact_timestamp"
    assert api().classify_full_report(receipt["raw_report"],identity)["exact_table_contact_samples"]==1

@pytest.mark.parametrize("side,pin",[("left","9ad911494f4d84296cb743b979d4294b1600491cafaf405daeaf2e55e10f2f85"),
    ("right","57fb5018546a15c1bc312663b73534fb8e192a47c5e5955910da174f00586b30")])
def test_real_thumb_archive_does_not_supply_static_table_evidence(cell,side,pin):
    path=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-contact-readback-20261112-WpVYXV")/side/"raw_contact_480hz.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pin
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==240
    counts=[]
    for row in rows:
        headers=row["full_report"]["headers"]
        if not headers:continue
        kwargs=dict(cell[1],stage_id=headers[0]["stage_id"])
        identity=api().source_identity(cell[0],**kwargs)
        counts.append(api().classify_full_report(row["full_report"],identity)["exact_table_contact_samples"])
    assert len(counts)>0 and sum(counts)==0
