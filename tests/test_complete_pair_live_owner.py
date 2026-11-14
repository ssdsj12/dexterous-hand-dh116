"""Existing full CPU owner, no native credit, complete atomic packet bindings."""
import json
from test_dh116_dual_pinch_builder import prepared,geometry
from test_live_r0_native_session import test_real_python_owner_initializes_before_first_action_then_settles as original_test
from scripts import prepare_dh116_live_r0_native as operation
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116 import tabletop_dual_hand_rod_runtime as runtime
from go2_pvcnn.control.dh116.tabletop_complete_pair_capture import read_complete_pair_capture
from go2_pvcnn.control.dh116.tabletop_live_effective_pad_source import CompletePairLiveR0Observer
from go2_pvcnn.control.dh116.tabletop_fast_owned_codec import codec_scope
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

def test_original_owner_complete_packet_contact_and_R0_share_clock(monkeypatch,tmp_path,prepared,geometry):
    with codec_scope(),bindings((runtime,"read_pair_capture",read_complete_pair_capture),
            (operation,"LiveR0Observer",CompletePairLiveR0Observer)):
        original_test(monkeypatch,tmp_path,prepared,geometry)
    paths=list(tmp_path.rglob("live_r0.jsonl"))
    assert len(paths)==1
    rows=[decode_owned(json.loads(s)) for s in paths[0].read_text().splitlines()]
    physical=[r["payload"] for r in rows if r["operation"]=="physical"]
    assert len(physical)==232
    for r in physical:
        # Raw and R0 evidence have the same true simulated clock even in doubles.
        evidence=r["r0_row"]["pair_diagnostics"]["complete_effective_pad_source"]
        assert evidence["identity"]==r["identity"]
        assert evidence["raw_capacity_overflow_retained"]
        assert evidence["capture_counts"]["overflow"] is False
        assert evidence["teacher_admission"] is False

    from scripts.validate_dh116_complete_pair_r0 import audit
    checked=audit(paths[0].parent)
    assert checked["physical_rows"]==232 and checked["atomic_aggregate_reconciled"]
    assert not checked["teacher_admission"] and checked["training_reward"] is None
