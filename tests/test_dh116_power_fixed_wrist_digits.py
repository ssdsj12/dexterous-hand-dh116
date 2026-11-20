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
