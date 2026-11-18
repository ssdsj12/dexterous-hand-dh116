"""CPU-only provenance and read-only replay contract; no Isaac imports."""
import json
import numpy as np
import pytest
from go2_pvcnn.control.dh116.tabletop_teacher_preview_recording import (
    NativePreviewRecorder, NativePreviewRecording, ReplayCursor,
)


def frame(step=0, **changes):
    data=dict(run_uid="run-A", episode_uid="episode-A", env_id=0, style="pinch",
              checkpoint_sha256="a"*64, sdk_step=2+step*8, sdk_time=(2+step*8)/480,
              policy_step=step, body_link_pose_w=np.ones((2,3,7),dtype=np.float32),
              rod_nodes=np.ones((5,3),dtype=np.float64), action=np.zeros(24,dtype=np.float32),
              net_lift=np.float64(step/100), wrench=None, support_ledger=None,
              validity={"wrench":False})
    data.update(changes)
    return data


def recording(style="pinch", count=3):
    r=NativePreviewRecorder(max_frames=4)
    for i in range(count): r.append(**frame(i,style=style))
    return r.finish()


def test_cpu_copy_is_detached_and_immutable():
    data=frame(wrench=np.ones((8,6),dtype=np.float64),support_ledger={"force":np.ones(3)})
    r=NativePreviewRecorder(max_frames=2)
    r.append(**data)
    for key in ("body_link_pose_w","rod_nodes","action","wrench"): data[key][...]=777
    data["support_ledger"]["force"][...]=777
    f=r.finish().frame(0)
    assert f["body_link_pose_w"][0,0,0] == 1
    assert f["action"][0] == 0
    assert f["wrench"][0,0] == 1
    assert f["support_ledger"]["force"][0] == 1
    with pytest.raises(ValueError): f["rod_nodes"][0,0]=0
    assert f["rod_nodes"].dtype == np.float64


@pytest.mark.parametrize("change", [dict(sdk_step=2),dict(sdk_time=0.),dict(policy_step=0),
    dict(policy_step=2),dict(episode_uid="other"),dict(run_uid="other"),dict(env_id=1),
    dict(style="power"),dict(checkpoint_sha256="b"*64)])
def test_mixed_identity_or_source_clock_rejected(change):
    r=NativePreviewRecorder(max_frames=3); r.append(**frame())
    with pytest.raises(ValueError): r.append(**frame(1,**change))
    assert r.frame_count == 1


def test_capacity_and_sealed_recording_reject_new_frames():
    r=NativePreviewRecorder(max_frames=1); r.append(**frame())
    with pytest.raises(OverflowError): r.append(**frame(1))
    assert r.finish().frame_count == 1
    with pytest.raises(RuntimeError): r.append(**frame(1))


def test_initial_frame_must_have_policy_step_zero():
    with pytest.raises(ValueError): NativePreviewRecorder(max_frames=4).append(**frame(1))


def test_missing_optional_measurements_remain_missing_without_fabricated_frames():
    rec=recording(count=2)
    assert rec.frame_count == 2
    assert rec.frame(0)["wrench"] is None
    assert rec.frame(0)["validity"]["wrench"] is False
    with pytest.raises(IndexError): rec.frame(2)
    with pytest.raises(IndexError): rec.frame(-1)
    r=NativePreviewRecorder(max_frames=2)
    with pytest.raises(ValueError): r.append(**frame(rod_nodes=None))
    assert r.frame_count == 0


@pytest.mark.parametrize("change",[dict(action=np.zeros(23)),dict(body_link_pose_w=np.zeros((2,3,6))),
    dict(rod_nodes=np.zeros((3,4))),dict(action=np.full(24,np.nan)),dict(validity={"wrench":True})])
def test_shapes_finiteness_and_false_validity_are_validated(change):
    with pytest.raises(ValueError): NativePreviewRecorder(max_frames=2).append(**frame(**change))


def test_npz_json_roundtrip_preserves_identity_dtype_and_sha(tmp_path):
    rec=recording(); rec.save(tmp_path/"trace")
    loaded=NativePreviewRecording.load(tmp_path/"trace")
    assert loaded.identity == rec.identity
    assert loaded.data_sha256 == rec.data_sha256
    assert loaded.frame_count == 3
    np.testing.assert_array_equal(loaded.body_link_pose_w,rec.body_link_pose_w)
    assert loaded.rod_nodes.dtype == np.float64
    metadata=json.loads((tmp_path/"trace.json").read_text())
    metadata["frame_count"]=999
    (tmp_path/"trace.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError): NativePreviewRecording.load(tmp_path/"trace")


def test_tampered_array_sha_rejected(tmp_path):
    rec=recording(); rec.save(tmp_path/"trace")
    with np.load(tmp_path/"trace.npz",allow_pickle=False) as f: arrays={k:f[k] for k in f.files}
    key=json.loads((tmp_path/"trace.json").read_text())["frames"][0]["arrays"]["action"]
    arrays[key][0]=2
    np.savez(tmp_path/"trace.npz",**arrays)
    with pytest.raises(ValueError,match="SHA"): NativePreviewRecording.load(tmp_path/"trace")


def test_replay_starts_at_zero_switches_style_and_never_calls_physics():
    # API takes recordings only: no environment/callback/physics object can be advanced.
    cursor=ReplayCursor({"pinch":recording(),"power":recording("power")})
    assert cursor.state == "waiting"
    assert cursor.current()["policy_step"] == 0
    cursor.play(); assert cursor.current()["policy_step"] == 0
    cursor.advance(); assert cursor.current()["policy_step"] == 1
    cursor.pause(); cursor.advance(); assert cursor.current()["policy_step"] == 1
    cursor.select_style("power")
    assert cursor.state == "paused" and cursor.current()["policy_step"] == 0
    cursor.start(); cursor.advance(); cursor.advance(); cursor.advance()
    assert cursor.current()["policy_step"] == 2 and cursor.state == "ended"
    cursor.play(); assert cursor.current()["policy_step"] == 0
    view=cursor.snapshot(frozen_sdk_step=987, frozen_sdk_time=2.5)
    assert view["source"] == "recorded_native_replay"
    assert view["source_sdk_step"] == 2 and view["frozen_sdk_step"] == 987
    assert view["counts_as_native_evaluation"] is False
    assert [e["event"] for e in cursor.events][-1] == "play"


def test_tensor_like_input_is_detached_and_copied_through_cpu():
    calls=[]
    class Tensor:
        def detach(self): calls.append("detach"); return self
        def cpu(self): calls.append("cpu"); return self
        def numpy(self): calls.append("numpy"); return raw
    raw=np.ones((2,3,7),dtype=np.float32)
    r=NativePreviewRecorder(max_frames=1)
    r.append(**frame(body_link_pose_w=Tensor()))
    raw[...] = 3
    assert calls == ["detach","cpu","numpy"]
    assert r.finish().body_link_pose_w[0,0,0,0] == 1


def test_optional_data_and_validity_roundtrip_without_filling_missing(tmp_path):
    r=NativePreviewRecorder(max_frames=2)
    r.append(**frame(wrench=np.ones((5,6),dtype=np.float32),
                     support_ledger={"finger0.force":np.arange(3,dtype=np.float64)},
                     validity={"wrench":True,"support_ledger":True,"finger1":False}))
    r.append(**frame(1))
    rec=r.finish(); rec.save(tmp_path/"optional.npz")
    loaded=NativePreviewRecording.load(tmp_path/"optional.json")
    assert loaded.data_sha256 == rec.data_sha256
    assert loaded.frame(0)["validity"]["finger1"] is False
    assert loaded.frame(0)["wrench"].dtype == np.float32
    np.testing.assert_array_equal(loaded.frame(0)["support_ledger"]["finger0.force"],np.arange(3))
    assert loaded.frame(1)["wrench"] is None
    assert loaded.frame(1)["support_ledger"] is None


def test_wait_resets_cursor_and_records_explicit_wait_event():
    cursor=ReplayCursor({"pinch":recording()})
    cursor.start(); cursor.advance(); cursor.wait()
    assert cursor.state == "waiting"
    assert cursor.current()["policy_step"] == 0
    cursor.advance()
    assert cursor.current()["policy_step"] == 0
    assert cursor.events[-1]["event"] == "wait"


def test_loader_rejects_object_pickle_without_deserializing(tmp_path):
    rec=recording(); rec.save(tmp_path/"objects")
    np.savez(tmp_path/"objects.npz",forbidden=np.array([{}],dtype=object))
    with pytest.raises(ValueError): NativePreviewRecording.load(tmp_path/"objects")


def test_replay_module_only_imports_numpy_and_standard_library():
    import ast
    import inspect
    from go2_pvcnn.control.dh116 import tabletop_teacher_preview_recording as module
    tree=ast.parse(inspect.getsource(module))
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imports.extend(a.name for a in node.names)
        elif isinstance(node,ast.ImportFrom): imports.append(node.module)
    assert set(imports) <= {"__future__","hashlib","json","pathlib","types","numpy"}


@pytest.mark.parametrize("validity", [[],[True],"wrench",{"wrench":1}])
def test_validity_must_be_named_boolean_mapping(validity):
    with pytest.raises(ValueError):
        NativePreviewRecorder(max_frames=2).append(**frame(validity=validity))


def test_frame_cannot_be_made_writable_again():
    a=recording().frame(0)["body_link_pose_w"]
    with pytest.raises(ValueError): a.setflags(write=True)


@pytest.mark.parametrize("corruption", ["shape","clock","validity"])
def test_loader_validates_structure_even_after_checksum_is_recomputed(tmp_path,corruption):
    from go2_pvcnn.control.dh116.tabletop_teacher_preview_recording import _sha
    rec=recording(); rec.save(tmp_path/"bad")
    meta=json.loads((tmp_path/"bad.json").read_text()); meta.pop("data_sha256")
    with np.load(tmp_path/"bad.npz",allow_pickle=False) as f: arrays={k:f[k] for k in f.files}
    if corruption == "shape":
        arrays[meta["frames"][0]["arrays"]["action"]]=np.zeros(23,dtype=np.float32)
    elif corruption == "clock":
        meta["frames"][1]["sdk_step"]=meta["frames"][0]["sdk_step"]
    else:
        meta["frames"][0]["validity"]["wrench"]=True
    meta["data_sha256"]=_sha(meta,arrays)
    np.savez(tmp_path/"bad.npz",**arrays)
    (tmp_path/"bad.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError): NativePreviewRecording.load(tmp_path/"bad")
