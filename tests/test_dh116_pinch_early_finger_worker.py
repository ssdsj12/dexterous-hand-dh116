from scripts import run_dh116_pinch_early_finger as worker


def test_sdk34_quiet_prefix_passes_only_on_present_same_clock_source(monkeypatch):
    archived = {"example": "archived measurement"}
    loc = {"example": "current measurement"}
    same = True

    def read(row, *, expected_sdk):
        assert expected_sdk == 34
        return loc

    def compare(observed, expected):
        assert observed == loc and expected == archived
        return {"passed": same, "metrics": {"test_fixture": True}}

    monkeypatch.setattr(worker.late, "prefix_measurements", read)
    monkeypatch.setattr(worker.late, "compare_prefix", compare)
    row = {"physics_step": 34, "guard_failure": None}
    worker.guard_prefix(row, archived)
    assert row["early_finger_admission"]["passed"]
    assert row["guard_failure"] is None
    same = False
    worker.guard_prefix(row, archived)
    assert row["guard_failure"] == "early_left_finger_prefix_mismatch"


def test_preexisting_physics_guard_is_preserved():
    row = {"physics_step": 34, "guard_failure": "existing_native_guard"}
    worker.guard_prefix(row, {})
    assert row == {"physics_step": 34, "guard_failure": "existing_native_guard"}
