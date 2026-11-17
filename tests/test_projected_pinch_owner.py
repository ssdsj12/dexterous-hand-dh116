"""Actual Python lifecycle with synthetic SDK fixtures, never native evidence."""
import json
import pytest
from test_dh116_dual_pinch_builder import prepared,geometry
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

@pytest.mark.parametrize("failure",[None,11])
def test_actual_owner_records_all_source_boundaries_and_terminal(monkeypatch,tmp_path,prepared,geometry,failure):
    import test_left_pinch_approach_owner as old
    from scripts import prepare_dh116_projected_pinch_native as op
    with bindings((old,"closure",op)):
        out,_,_=old.run_owner(monkeypatch,tmp_path,prepared,geometry,failure=failure)
    r=op.inspect_interval(out/"physical.jsonl",complete=failure is None)
    assert r["projection_source_rows"]==(232 if failure is None else 9)
    assert r["projection_continuity_verified"] is True
    assert json.loads((out/"app_close_returned.json").read_text())["app_close_returned"]
    if failure is None:
        assert json.loads((out/"projected_pinch_scope.json").read_bytes())==op.scope_marker()
    rows=[decode_owned(json.loads(l)["payload"])["exchange"] for l in (out/"physical.jsonl").read_text().splitlines()
          if json.loads(l).get("operation")=="physical"]
    assert rows[-1]["physics_step"]==(234 if failure is None else 11)
    assert all("input_evidence" in x["wrist_projection"] for x in rows)
