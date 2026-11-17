"""CPU only: real source bytes, original reset, and explicit simulated getter arrays."""
from copy import deepcopy
import hashlib,importlib,json
from pathlib import Path
import numpy as np
import pytest
from test_mesh_reset_order import native_semantics,plan,next_step
ROOT=Path(__file__).resolve().parents[1]

def source():
    assert (ROOT/"src/go2_pvcnn/control/dh116/tabletop_mesh_observation_source.py").exists()
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_mesh_observation_source")

def readback():
    assert (ROOT/"src/go2_pvcnn/control/dh116/tabletop_mesh_reset_readback.py").exists()
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_mesh_reset_readback")

def test_actual_new_full_source_inventory_and_old_default_unchanged():
    m=source()
    from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as old
    prior=old.source_basis()
    b=m.source_basis()
    assert b["backend_sha256"]=="8817010163946945495c987923d11f3ade4fbd52482a83bbdd48f017d3149340"
    assert b["source_file_count"]==72 and len(b["changed_sources"])==5
    assert b["native_qualification"] is b["standalone_probe_qualified"] is False
    assert b["source_tree_sha256"]=="34741b437191fd66bf151331f20476a86f8708414dcc2782a8c69dd15b0efb64"
    assert old.source_basis()==prior

@pytest.mark.parametrize("which",["binary","unchanged_source","changed_source","inventory","patcher","patch","build_receipt","import_receipt"])
def test_actual_source_changes_fail_closed_without_mutation(monkeypatch,which):
    m=source()
    paths={"binary":m.BACKEND,"unchanged_source":m.BUILD/"source/csrc/batch.cu",
        "changed_source":m.BUILD/"source/csrc/sdf.cu","inventory":m.PREPARATION,
        "patcher":m.ROOT/m.PATCHER,"patch":m.PATCH,"build_receipt":m.BUILD_RECEIPT,
        "import_receipt":m.IMPORT_RECEIPT}
    original=Path.read_bytes
    monkeypatch.setattr(Path,"read_bytes",lambda p:original(p)+(b"changed" if p==paths[which] else b""))
    with pytest.raises(ValueError):m.source_basis()

def test_scope_live_and_postexit_share_source_and_restore_on_error():
    m=source()
    from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as p
    from go2_pvcnn.control.dh116 import tabletop_r0_health_separation as h
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as b
    from scripts import prepare_dh116_native_parameter_task as live
    from scripts import validate_dh116_native_parameter_task as audit
    before=(p.VERSION,p.BUILD,p.BACKEND,p.BACKEND_SHA,p.source_basis,h.BACKEND,h.BACKEND_SHA,b.BACKEND,b.BACKEND_SHA)
    with pytest.raises(RuntimeError,match="fixture"):
        with m.source_scope() as basis:
            assert live.reader is audit.reader is p
            assert p.VERSION==m.PARAMETER_VERSION
            assert p.source_basis()==basis
            assert p.BACKEND==h.BACKEND==b.BACKEND==m.BACKEND
            assert p.BACKEND_SHA==h.BACKEND_SHA==b.BACKEND_SHA==m.BACKEND_SHA
            with m.source_scope():assert p.source_basis()==basis
            raise RuntimeError("fixture")
    assert before==(p.VERSION,p.BUILD,p.BACKEND,p.BACKEND_SHA,p.source_basis,h.BACKEND,h.BACKEND_SHA,b.BACKEND,b.BACKEND_SHA)

def test_scope_rejects_foreign_binding_without_overwriting(monkeypatch):
    m=source()
    from go2_pvcnn.control.dh116 import tabletop_r0_health_separation as h
    monkeypatch.setattr(h,"BACKEND_SHA","foreign")
    with pytest.raises(ValueError,match="binding"):
        with m.source_scope():pass
    assert h.BACKEND_SHA=="foreign"

def setup_readback(monkeypatch):
    c=native_semantics(monkeypatch);p=plan(c,monkeypatch);m=source();r=readback()
    p.pop("sha256");p["source_basis"]=m.source_basis()
    from go2_pvcnn.control.dh116 import tabletop_mesh_reset_order as old
    p=old.seal_plan(p)
    monkeypatch.setattr(m,"verify_runtime",lambda rod:None)  # Explicit CPU backend double.
    def getter():
        c.events.append("mesh_state_snapshot")
        pos=np.array([[c.rope.current[i][0] for i in range(11)]],np.float32)
        q=np.array([[c.rope.current[i][1] for i in range(11)]],np.float32)
        return dict(schema="dh116.mesh_state_snapshot.v1",num_envs=1,mesh_count=11,
            current_positions_world_m=pos,current_quaternions_xyzw=q,
            target_positions_world_m=np.array([[c.rope.targets[i][0] for i in range(11)]],np.float32),
            target_quaternions_xyzw=np.array([[c.rope.targets[i][1] for i in range(11)]],np.float32),
            pending_teleport=np.array([[c.rope.tele[i] for i in range(11)]],np.uint8),
            solver_positions_world_m=pos.copy(),solver_quaternions_xyzw=q.copy(),
            device_state_source="cuda_device_copy",pending_set=c.rope.pending,moving=False,
            pair_capacity=c.rope.capacity,pair_ready=c.rope.pair_ready,
            flags_source="native_host_sdf_store",owner_time_s_before=c.rope.time,
            owner_time_s_after=c.rope.time,owner_time_source="native_host_owner_metadata_at_readback",last_error=0)
    c.rope.mesh_state_snapshot=getter
    return c,p,r

def execute(c,p,r,uid="one"):
    return r.MeasuredMeshReset(c.seed,episode_uid=uid,mesh_plan=p).run(
        c.sim,c.rod,c.bindings,owner_resets=c.owners)

def test_actual_reset_rebuild_plus_owned_mesh_getter_and_disk_revalidation(monkeypatch):
    c,p,r=setup_readback(monkeypatch);old=deepcopy(c.rope.current)
    result=execute(c,p,r);packet=result["mesh_state_readback"]
    check=r.validate_reset_readback(packet,reset_receipt=result,mesh_plan=p)
    assert check["eleven_reset_meshes_observed"] is True
    assert check["native_qualification"] is check["teacher_admission"] is False
    assert packet["before"]["clock_before"]==packet["before"]["clock_after"]
    assert packet["after"]["clock_before"]==packet["after"]["clock_after"]
    assert packet["after"]["snapshot"]["pending_set"] is False
    assert not np.array_equal(old[0][0],packet["after"]["snapshot"]["current_positions_world_m"][0,0])
    assert c.rope.steps==0 and c.sim.current_time_step_index==2
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
    disk=json.loads(json.dumps(encode_owned(result),allow_nan=False))
    assert r.validate_reset_readback(disk["mesh_state_readback"],reset_receipt=disk,mesh_plan=p)==check
    packet["after"]["snapshot"]["current_positions_world_m"][:]=99
    assert not np.any(c.rope.current[0][0]==99)

def test_reset_readback_also_handles_second_real_episode(monkeypatch):
    c,p,r=setup_readback(monkeypatch);execute(c,p,r);next_step(c);time=c.rope.time
    second=execute(c,p,r,"two")
    assert second["after_clock"]["reset_id"]==2
    assert c.rope.time==time and c.rope.steps==1 and c.rope.resets==2
    assert second["mesh_state_readback"]["before"]["snapshot"]["pair_ready"] is True
    assert second["mesh_state_readback"]["after"]["snapshot"]["pair_ready"] is False

@pytest.mark.parametrize("fault",["current","target","solver","quat","teleport","pending","moving","capacity","pair_ready","owner_time","last_error","clock_valid","sdk_time"])
def test_after_rebuild_getter_fault_poisoned_and_owned_failure_preserved(monkeypatch,fault):
    c,p,r=setup_readback(monkeypatch);get=c.rope.mesh_state_snapshot;captured=[]
    def bad():
        row=get()
        if c.rope.add_calls==11:
            if fault in ("current","target","solver"):row[fault+"_positions_world_m"][0,0,0]+=.001
            if fault=="quat":row["current_quaternions_xyzw"][0,0]=[0,0,1,0]
            if fault=="teleport":row["pending_teleport"][0,0]=1
            if fault=="pending":row["pending_set"]=True
            if fault=="moving":row["moving"]=True
            if fault=="capacity":row["pair_capacity"]=1
            if fault=="pair_ready":row["pair_ready"]=True
            if fault=="owner_time":row["owner_time_s_after"]+=.001
            if fault=="last_error":row["last_error"]=5
            if fault=="clock_valid":c.rod._needs_reset=True
            if fault=="sdk_time":c.sim.current_time+=.001
        return row
    c.rope.mesh_state_snapshot=bad
    transaction=r.MeasuredMeshReset(c.seed,episode_uid="bad",mesh_plan=p)
    with pytest.raises((ValueError,RuntimeError)):
        transaction.run(c.sim,c.rod,c.bindings,owner_resets=c.owners,record_readback=captured.append)
    assert transaction.status=="failed" and c.rod._needs_reset
    assert c.rope.steps==0 and captured[-1]["capture_error"]
    assert transaction.readback_receipt["complete"] is False
    with pytest.raises(RuntimeError):next_step(c)

@pytest.mark.parametrize("when",[0,11])
def test_getter_throw_preserves_failure_and_releases_native_aliases(monkeypatch,when):
    c,p,r=setup_readback(monkeypatch);get=c.rope.mesh_state_snapshot
    def bad():
        if c.rope.add_calls==when:raise RuntimeError("getter fixture")
        return get()
    c.rope.mesh_state_snapshot=bad
    t=r.MeasuredMeshReset(c.seed,episode_uid="failure",mesh_plan=p)
    with pytest.raises(ValueError,match="getter fixture") as caught:
        t.run(c.sim,c.rod,c.bindings,owner_resets=c.owners)
    assert t.readback_receipt["capture_error"] and t.status=="failed"
    tb=caught.value.__traceback__
    while tb:
        if tb.tb_frame.f_code.co_filename.endswith("tabletop_mesh_reset_readback.py"):
            for key in ("sim","rod","rope","bindings","owner_resets","record_readback"):
                if key in tb.tb_frame.f_locals:assert tb.tb_frame.f_locals[key] is None
        tb=tb.tb_next

def test_postexit_rejects_pose_clock_identity_or_source_tampering(monkeypatch):
    c,p,r=setup_readback(monkeypatch);result=execute(c,p,r)
    for fault in ("pose","clock","instance","backend","plan","missing"):
        bad=deepcopy(result["mesh_state_readback"])
        if fault=="pose":bad["after"]["snapshot"]["current_positions_world_m"][0,1,0]+=.01
        if fault=="clock":bad["after"]["clock_after"]["rod"]["reset_id"]+=1
        if fault=="instance":bad["after"]["source_instance_id"]+=1
        if fault=="backend":bad["sources"]["backend_sha256"]="old"
        if fault=="plan":bad["mesh_plan_sha256"]="bad"
        if fault=="missing":bad["after"]=None
        with pytest.raises((ValueError,KeyError,TypeError)):r.validate_reset_readback(bad,reset_receipt=result,mesh_plan=p)

def test_new_profile_exact_plan_still_revalidates_original_grids_and_table(monkeypatch):
    m=source();old=importlib.import_module("go2_pvcnn.control.dh116.tabletop_mesh_reset_order")
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import prepare_scene
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import load_candidate
    from types import SimpleNamespace
    prepared=prepare_scene(ROOT)
    with m.source_scope():
        sdf=old._cpu_sdf(m.BUILD/"build/python/ropesim/sdf.py")
        from go2_pvcnn.control.dh116.tabletop_native_parameter_endpoints import CONFIG
        rod=SimpleNamespace(bindings=deepcopy(prepared["pads"]),_base=SimpleNamespace(cfg=load_candidate(ROOT/CONFIG,ROOT)))
        monkeypatch.setattr(old,"verify_runtime_source",lambda rod:None)
        p=old.prepare_mesh_reset_plan(ROOT,prepared,rod,sdf_module=sdf)
        old.verify_plan_sources(p)
        assert p["source_basis"]["backend_sha256"]==m.BACKEND_SHA
        p.pop("sha256");p["entries"][0]["mu_s"]+=.1
        with pytest.raises(ValueError):old.verify_plan_sources(old.seal_plan(p))


@pytest.mark.parametrize("key",["sdk_physics_step","reset_id","clock_valid"])
def test_disk_clock_rejects_boolean_integer_or_untyped_valid(monkeypatch,key):
    c,p,r=setup_readback(monkeypatch);result=execute(c,p,r)
    bad=deepcopy(result["mesh_state_readback"])
    for edge in ("clock_before","clock_after"):
        clock=bad["after"][edge]
        if key=="sdk_physics_step":clock[key]=True
        elif key=="clock_valid":clock["rod"][key]=1
        else:clock["rod"][key]=True
    with pytest.raises(ValueError):r.validate_reset_readback(bad,reset_receipt=result,mesh_plan=p)


from test_flexible_parameter_readback import case

def test_existing_live_health_and_disk_validator_use_new_same_instance_profile(case):
    from test_r0_numerical_health_source import row_for
    from go2_pvcnn.control.dh116 import tabletop_health_getter_evidence as health
    from go2_pvcnn.control.dh116 import tabletop_r0_health_separation as reducer
    c=case;m=source()
    c.base.physics_step=3;c.base.rod_tick=12;c.rope.time=1/480
    c.sim.current_time_step_index=3;c.sim.current_time=3/480
    c.expected=c.runtime.clock();c.backend.__file__=str(m.BACKEND)
    with m.source_scope():
        row=row_for(c)
        got=health.capture_boundary(c.runtime,backend=c.backend,sim=c.sim,exchange=row)
        assert got["capture_error"] is None and c.events==[]
        assert got["solver_status"]["backend_sha256"]==m.BACKEND_SHA
        assert health.validate_capture(got,row)["last_error"]==0
        assert row["material_director"]["version"]==m.PARAMETER_VERSION
        observed=reducer.observe_health(row,solver_status=got["solver_status"])
        assert observed["numerical_health"]["value"] is True
        assert observed["motion"]["hold_motion"] is None
        assert observed["native_qualification"] is observed["teacher_admission"] is False
    with pytest.raises(ValueError):health.validate_capture(got,row)


def test_repeated_transaction_cannot_erase_success_receipt(monkeypatch):
    c,p,r=setup_readback(monkeypatch)
    transaction=r.MeasuredMeshReset(c.seed,episode_uid="once",mesh_plan=p)
    transaction.run(c.sim,c.rod,c.bindings,owner_resets=c.owners)
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
    before=_digest(transaction.readback_receipt)
    with pytest.raises(RuntimeError):transaction.run(c.sim,c.rod,c.bindings,owner_resets=c.owners)
    assert _digest(transaction.readback_receipt)==before
    assert transaction.status=="complete" and c.rod.clock()["clock_valid"]
