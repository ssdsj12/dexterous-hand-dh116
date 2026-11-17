"""Run the actual Python owner with synthetic SDK data, including early terminal."""
import json
from pathlib import Path
import pytest
from test_dh116_dual_pinch_builder import prepared,geometry
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116 import tabletop_deformable_physical_exchange as exchange

@pytest.mark.parametrize("fail",[False,True])
def test_original_owner_records_projection_inputs_without_changing_targets(monkeypatch,tmp_path,prepared,geometry,fail):
    from scripts import prepare_dh116_dual_pinch_director_interval as director
    from go2_pvcnn.control.dh116.tabletop_wrist_projection_evidence import ProjectionEvidenceMixin,validate_record
    from test_left_pinch_approach_owner import run_owner
    original=exchange.project_policy_wrist_targets
    class EvidenceExchange(ProjectionEvidenceMixin,director.DirectorExchange):pass
    with bindings((director,"DirectorExchange",EvidenceExchange)):
        out,_,_=run_owner(monkeypatch,tmp_path,prepared,geometry,failure=11 if fail else None)
    assert exchange.project_policy_wrist_targets is original
    rows=[decode_owned(json.loads(l)["payload"])["exchange"] for l in (out/"physical.jsonl").read_text().splitlines()
          if json.loads(l).get("operation")=="physical"]
    assert len(rows)==(9 if fail else 232)
    for row in rows:
        c=row["action_readback"]["pre"]["identity"][0]
        r=validate_record(row["wrist_projection"],c)
        assert r["identity"]["physics_step"]+1==row["physics_step"]
        assert r["independent_source_clearance_recomputed"]
        assert not r["native_qualified"]
    assert rows[-1]["guard_failure"]==("material_director_source_invalid" if fail else None)
    assert json.loads((out/"app_close_returned.json").read_text())["app_close_returned"]
