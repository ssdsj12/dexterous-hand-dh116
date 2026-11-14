"""Only the frozen real interval is admitted; raw pairs never imply support."""
import importlib,importlib.util
from copy import deepcopy
import pytest
import numpy as np

def api():
    name="scripts.audit_dh116_projected_pinch_contact"
    assert importlib.util.find_spec(name) is not None,"source-locked interval analysis missing"
    return importlib.import_module(name)

def rows():
    out=[]
    for tick in range(3,235):
        c=dict(run_uuid="cpu",env_id=0,episode_uid="cpu:1",reset_id=1,rod_uid="r",physics_step=tick,
            timestamp_s=tick/480,policy_step=(tick-3)//8)
        p=dict(pre_projection_palm_offsets_m=np.zeros((1,2,3)),palm_offsets_m=np.zeros((1,2,3)),
            safe_z_clipped=np.zeros((1,2),bool))
        out.append(dict(physics_step=tick,policy_step=c["policy_step"],guard_failure=None,
            action_readback=dict(post=dict(identity=[c])),wrist_projection=p))
    return out

def test_projection_summary_records_occurrence_identity():
    m=api();r=rows();r[173]["wrist_projection"]["palm_offsets_m"][0,0,2]=.000001
    r[173]["wrist_projection"]["safe_z_clipped"][0,0]=True
    s=m.projection_summary(r)
    assert s["clip_rows"]==1 and s["first_clip"]["identity"]["physics_step"]==176
    assert s["maximum_step_projection_mm"][0][2]==pytest.approx(.001)
    assert not s["teacher_admission"]

@pytest.mark.parametrize("bad",["gap","episode","time","terminal","nan","short"])
def test_foreign_or_incomplete_source_rejected(bad):
    r=rows()
    if bad=="gap":r[25]["physics_step"]+=1
    if bad=="episode":r[25]["action_readback"]["post"]["identity"][0]["episode_uid"]="different"
    if bad=="time":r[25]["action_readback"]["post"]["identity"][0]["timestamp_s"]+=1
    if bad=="terminal":r[25]["guard_failure"]="joint_limit"
    if bad=="nan":r[25]["wrist_projection"]["palm_offsets_m"][0,0,2]=np.nan
    if bad=="short":r.pop()
    with pytest.raises(ValueError):api().projection_summary(r)

@pytest.mark.parametrize("field,value",[("native_qualification",False),("native_execution_performed",False),
    ("outer_exit_code",1),("duration_seconds",181),("debugger_inferior_signal","SIGSEGV"),
    ("qualification_scope","foreign"),("code_config_sha256","0"*64)])
def test_receipt_cannot_promote_failed_foreign_or_unexecuted_result(tmp_path,field,value):
    import json,hashlib
    m=api()
    r=dict(qualification_scope=m.SCOPE,command_spec_sha256=m.RECIPE,code_config_sha256=m.CODE,
        native_qualification=True,native_execution_performed=True,outer_exit_code=0,
        debugger_inferior_exit_code=0,debugger_inferior_signal=None,duration_seconds=163.)
    r[field]=value;p=tmp_path/"receipt.json";p.write_text(json.dumps(r))
    with pytest.raises(ValueError):m.run(p,hashlib.sha256(p.read_bytes()).hexdigest())

def test_frozen_native_interval_is_replayed_without_new_physics():
    from pathlib import Path
    p=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f8-original-projected-pinch-native/d2b175b7654b4b30c67028a20a6b0e1a50eda98b3804414554c6d8c9fd3f7d58/attempt-zy2fao0h/receipt.json")
    r=api().run(p,"04ebfe2460b6366960d87cda6f36200f503072e4fdfa2abffd25bc8dfcf775c3")
    assert r["contact"]["raw_pairs"]["left.finger23_link"]==428
    assert r["contact"]["accepted_by_pad"].get("left.finger23_link",0)==0
    assert r["contact"]["accepted_by_pad"]["left.finger14_link"]==300
    assert r["original_region_windows"]["right"]["count"]==276
    assert r["original_region_windows"]["bilateral"]["count"]==0
    assert r["projection"]["clip_rows"]==36
    assert r["projection"]["first_clip"]["identity"]["physics_step"]==176
    assert r["source"]["outer_exit_code"]==0
    assert r["new_physics_steps"]==r["ppo_updates"]==0
    assert not r["native_success_qualified"] and not r["teacher_admission"]
