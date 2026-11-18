"""Fixed native episode replay; CPU diagnostics never confer grasp success."""
from pathlib import Path
import importlib
import pytest

def module():return importlib.import_module("scripts.audit_dh116_tracking_pinch_contact")

def test_missing_schema_rejected():
    m=module()
    with pytest.raises(ValueError):m.contiguous_windows([],0)
    with pytest.raises(ValueError):m.contiguous_windows([1,1],1/5760)

def test_measured_gaps_never_interpolated():
    m=module();r=m.contiguous_windows([1,2,5,6,7],.1)
    assert r["count"]==5 and r["longest_samples"]==3
    assert r["longest_duration_s"]==pytest.approx(.3)
    assert r["runs"]==[[1,2],[5,7]]
    assert m.contiguous_windows([],.1)["longest_samples"]==0

@pytest.fixture(scope="module")
def actual():return module().run()

def test_exact_source_clock_and_contact_regions(actual):
    assert actual["source"]["receipt_sha256"]==module().RECEIPT_SHA
    assert actual["source"]["outer_exit_code"]==0
    assert actual["source"]["duration_seconds"]<180
    assert actual["summary"]["raw_pairs"]=={"right.finger23_link":1116,"right.finger14_link":1100}
    assert actual["summary"]["accepted_by_pad"]=={"right.finger23_link":311,"right.finger14_link":276}
    assert actual["summary"]["accepted_by_original_region"]==587
    assert actual["summary"]["both_right_region_windows"]==276
    assert actual["windows"]["left"]["count"]==0
    assert actual["windows"]["right"]["count"]==276
    assert actual["windows"]["bilateral"]["count"]==0
    assert len(actual["measured_boundaries"])==232
    assert [r["physics_step"] for r in actual["measured_boundaries"]]==list(range(3,235))

def test_no_contact_success_or_teacher_promotion(actual):
    assert actual["teacher_admission"] is actual["positive_effective_contact_qualified"] is False
    assert actual["native_success_qualified"] is actual["fixed_material_slip_qualified"] is False
    assert actual["new_physics_steps"]==actual["ppo_updates"]==0
    assert actual["windows"]["right"]["longest_duration_s"]<.1
    assert actual["endpoint"]["new_physics_steps"]==0
    assert actual["endpoint"]["teacher_admission"] is False


def test_digest_mismatch_rejected(tmp_path):
    p=tmp_path/"raw.json";p.write_text("changed")
    with pytest.raises(ValueError):module().read_locked(p,"0"*64)

def test_actual_geometry_minima_remain_identified(actual):
    g=actual["geometry_minima_mm"]
    for side in ("left","right"):
        for key in ("minimum_table_clearance_m","minimum_non_target_gap_lower_m"):
            row=g[side][key]
            assert row["physics_step"] in range(3,235)
            assert row["episode_uid"]==actual["summary"]["identity"]["episode_uid"]
        assert g[side]["minimum_table_clearance_m"]["value_mm"]>=2.1
        assert g[side]["minimum_non_target_gap_lower_m"]["value_mm"]>=0
        assert all(row["value_mm"]>=2.1 for row in g[side]["cooked_table_by_pad"].values())
