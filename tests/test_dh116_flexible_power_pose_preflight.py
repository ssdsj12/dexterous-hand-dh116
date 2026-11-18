import json

import pytest

from scripts.audit_dh116_flexible_power_pose import load_power_archive


@pytest.fixture
def archived(tmp_path):
    source = tmp_path / "power.json"
    row = {"start_index": 0, "qualified": False, "active_angles": [0.] * 6,
           "wrist_pose_w": [0., 0., .5, 1., 0., 0., 0.],
           "wrap": {"geometry_screen_passed": False, "native_contact_verified": False}}
    doc = {"status": "complete", "scope": "cpu_geometry_only", "source_unchanged": True,
           "geometry_candidates": 0, "native_contact_verified": False,
           "sides": [{"side": name, "rows": [row]} for name in ("left", "right")]}
    source.write_text(json.dumps(doc))
    return source, doc


def test_load_independent_negative_power_archive(archived):
    import hashlib

    source, _ = archived
    data = load_power_archive(source, hashlib.sha256(source.read_bytes()).hexdigest())
    assert sorted(data) == ["left", "right"]
    assert data["left"][0]["start_index"] == 0


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(scope="pinch"),
    lambda d: d.update(native_contact_verified=True),
    lambda d: d["sides"][1].update(side="left"),
    lambda d: d["sides"][0]["rows"][0].update(qualified=True),
    lambda d: d["sides"][0]["rows"].extend(d["sides"][0]["rows"]),
])
def test_does_not_treat_mislabeled_or_duplicate_archives_as_power(archived, mutate):
    import hashlib

    source, doc = archived
    mutate(doc)
    source.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        load_power_archive(source, hashlib.sha256(source.read_bytes()).hexdigest())


def test_reject_changed_archive_before_geometry(archived):
    source, _ = archived
    with pytest.raises(ValueError, match="SHA256"):
        load_power_archive(source, "0" * 64)
