import numpy as np
import pytest

from scripts import verify_dh116_902_flexible_reset as reset902


@pytest.mark.parametrize("grasp,pads", [("pinch", 2), ("tripod", 3)])
def test_reference_scene_uses_independent_seeds_with_original_fixed_stereo(grasp, pads):
    report, config, bindings = reset902.screen(grasp)
    assert report["geometry_passed"]
    assert not report["native_reset_verified"] and not report["grasp_qualified"]
    assert report["camera_sha256"] == reset902.sha(reset902.CAMERA)
    assert set(report["hands"]) == {"left", "right"}
    for hand in report["hands"].values():
        assert hand["count"] == 22
        assert len(hand["target_pad_rod_gaps_m"]) == pads
    assert config["hands"]["left"]["pose_wxyz"][1] > 0 > config["hands"]["right"]["pose_wxyz"][1]
    assert config["fixed_rod_nodes"] == []
    assert config["gravity_w_m_s2"] == [0., 0., -9.81]
    np.testing.assert_allclose(np.asarray(config["rod_nodes_w_m"])[:, 2], .365)
    assert reset902.CAMERA in bindings


def test_reference_cannot_switch_cameras_or_reuse_the_wrong_side_sources(monkeypatch, tmp_path):
    path = tmp_path / "camera.json"
    path.write_bytes(reset902.CAMERA.read_bytes() + b" ")
    monkeypatch.setattr(reset902, "CAMERA", path)
    with pytest.raises(ValueError, match="unchanged Student stereo"):
        reset902.screen("pinch")


def test_offline_initialization_cannot_claim_a_physical_grasp():
    report, config, _ = reset902.screen("pinch")
    assert config["scope"] == reset902.PINCH_SCOPE
    assert report["readback_source"] == "offline_source_and_locked_cooked_geometry"
    assert config["physics_qualified"] is False


def test_native_prescreen_uses_a_separate_cpu_process():
    import subprocess
    import sys
    from pathlib import Path

    snippet = '''
from scripts import verify_dh116_902_flexible_reset as reset902
from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary
assert reset902.run_screen("pinch")["geometry_passed"]
assert_clean_import_boundary()
'''
    subprocess.run([sys.executable, "-c", snippet], cwd=Path(reset902.ROOT), check=True)
