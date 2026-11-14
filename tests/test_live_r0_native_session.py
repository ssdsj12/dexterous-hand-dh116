"""Integration of live journal/initializer with the existing CPU SDK owner."""
import json
from pathlib import Path
import pytest
from test_dh116_dual_pinch_builder import prepared,geometry
from test_effective_right_pinch_owner import run_owner
from scripts import prepare_dh116_effective_right_native as previous
from scripts import prepare_dh116_live_r0_native as operation
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

def test_real_python_owner_initializes_before_first_action_then_settles(monkeypatch,tmp_path,prepared,geometry):
    monkeypatch.setattr(previous,"run_existing_app",operation.run_existing_app)
    out,_,_=run_owner(monkeypatch,tmp_path,prepared,geometry)
    init=decode_owned(json.loads((out/"live_r0_reset.json").read_text()))
    assert init["live_clock_before"]==init["live_clock_after"]
    assert init["live_clock_before"]["sdk_physics_step"]==2
    assert init["temporary_native_references_released"] is True
    assert init["initialization"]["initializer"]["reset_callbacks_executed"]==0
    rows=[decode_owned(json.loads(s)) for s in (out/"live_r0.jsonl").read_text().splitlines()]
    assert sum(r["operation"]=="physical" for r in rows)==232
    ends=[r["payload"] for r in rows if r["operation"]=="end"]
    assert len(ends)==29 and all(e["control_receipt"]["physical_samples"]==8 for e in ends)
    assert all(e["training_reward"] is None for e in ends)
    result=json.loads((out/"live_r0_summary.json").read_text())
    assert result["physical_rows"]==232 and result["control_windows"]==29
    assert result["new_physics_steps_by_observer"]==0 and result["teacher_admission"] is False
    assert result["status"]=="complete"
    from scripts.validate_dh116_live_r0_reset import audit
    checked=audit(out)
    assert checked["live_reset_initialization_verified"] and not checked["full_R0_ready"]
    # Audit must not accept a detached timestamp or stale native source digest.
    path=out/"live_r0.jsonl"; original=path.read_text()
    for key,value in (("identity",{}),("actual_event_sha256","0"*64)):
        damaged=[json.loads(line) for line in original.splitlines()]
        sample=next(row for row in damaged if row["operation"]=="physical")
        sample["payload"][key]=value
        path.write_text("\n".join(json.dumps(row) for row in damaged)+"\n")
        with pytest.raises(ValueError,match="physical identity"):audit(out)
    path.write_text(original)
    path.write_text(original.rstrip("\n"))
    with pytest.raises(ValueError,match="incomplete JSONL"):audit(out)
    path.write_text(original)

def test_native_refs_cleared_even_when_initializer_fails():
    session=object.__new__(operation.LiveSession)
    session.native_refs={"sim":object(),"rod":object()}
    session.reset_event=None
    with pytest.raises(ValueError):session.initialize(None,None,{},None)
    assert session.native_refs=={}
