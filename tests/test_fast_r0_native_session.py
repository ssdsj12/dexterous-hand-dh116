"""Full original 234-step CPU owner under optimized encoding, no native credit."""
from test_dh116_dual_pinch_builder import prepared,geometry
from test_live_r0_native_session import test_real_python_owner_initializes_before_first_action_then_settles as original_test
from go2_pvcnn.control.dh116.tabletop_fast_owned_codec import codec_scope

def test_original_owner_reset_physics_actions_and_R0_audit_with_fast_codec(monkeypatch,tmp_path,prepared,geometry):
    with codec_scope():
        original_test(monkeypatch,tmp_path,prepared,geometry)
