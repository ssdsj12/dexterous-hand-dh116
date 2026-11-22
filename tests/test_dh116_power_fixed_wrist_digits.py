from pathlib import Path
import json

import pytest

from scripts.audit_dh116_power_fixed_wrist_digits import audit_fixed_wrist


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/dc7-power-abduction-reset-probe.json")


def test_real_power_fixed_wrist_excludes_ring_and_little_through_continuous_joint_range():
    result = audit_fixed_wrist(json.loads(SOURCE.read_bytes()), project=ROOT)
    rows = result["joint_results"]
    assert len(rows) == 5 and all(len(d["samples"]) == 65 for d in rows.values())
    assert .0023 < rows["finger41"]["summary"]["continuous_minimum_lower_m"] < .0025
    assert .0101 < rows["finger51"]["summary"]["continuous_minimum_lower_m"] < .0103
    assert result["fixed_wrist_excluded_by"] == ["finger41", "finger51"]
    assert rows["finger31"]["tight_best_sample"]["lower_m"] > .00025
    assert result["native_contact_verified"] is False and result["teacher_admission"] is False


def test_claimed_open_seed_cannot_override_independent_physical_geometry():
    artifact = json.loads(SOURCE.read_bytes())
    row, = [row for row in artifact["left"] if row["abduction_fraction"] == .15 and row["roll"] == 1.45]
    row["root"][2] -= .04
    with pytest.raises(ValueError, match="palm witness"):
        audit_fixed_wrist(artifact, project=ROOT)


PALM_RECEIPT = SOURCE.parent / "ag2404-power-palm-witness-native/outer.json"


def test_measured_power_archive_keeps_clock_pose_and_wire_together():
    from scripts import audit_dh116_power_fixed_wrist_digits as audit

    row, scene, sources = audit.load_measured_palm_boundary(PALM_RECEIPT, project=ROOT)
    assert row["physics_step"] == row["sdk_hand_geometry"]["physics_step"] == 6504
    assert row["rod"]["node_positions"].shape == (1, 41, 3)
    assert scene["wire_radius_m"] == pytest.approx(.005)
    source = sources["motion_row_sha256"]
    assert source == "618f7f52161c7d72713608cfa1138f4600569da8c3cdb91722688441e5bde093"
    row["sdk_hand_geometry"]["physics_step"] -= 1
    with pytest.raises(ValueError, match="same physics step"):
        audit.audit_measured_wrist(row, scene, project=ROOT)


def test_measured_power_rejects_a_bilateral_wrap_if_one_distal_pad_cannot_reach(monkeypatch):
    from scripts import audit_dh116_power_fixed_wrist_digits as audit

    row, scene, _ = audit.load_measured_palm_boundary(PALM_RECEIPT, project=ROOT)
    measured = []

    def digit(geometry, wrist, q, index, pad, nodes, radius, supports, table, limits):
        measured.append((geometry["side"], pad))
        excluded = (geometry["side"], pad) == ("left", "finger53_link")
        gap = .003 if excluded else -.001
        return {"summary": {"fixed_wrist_reach_excluded": excluded, "continuous_minimum_lower_m": gap,
                            "best_sample_upper_m": gap}}

    monkeypatch.setattr(audit, "_digit_reach", digit)
    result = audit.audit_measured_wrist(row, scene, project=ROOT)
    assert len(measured) == 10 and len(set(measured)) == 10
    assert result["hands"]["left"]["excluded_pads"] == ["finger53_link"]
    assert not result["hands"]["right"]["excluded_pads"]
    assert result["fixed_wrist_excluded"]
    assert not result["simultaneous_closure_verified"]
