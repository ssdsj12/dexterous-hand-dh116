from scripts import run_dh116_pinch_wrist_relief as worker


def test_relief_guard_is_isolated_to_measured_branch_boundary(monkeypatch):
    def capture(row, *, expected_sdk):
        assert expected_sdk == 58
        return {"observation": "same tick"}

    monkeypatch.setattr(worker.early.late, "prefix_measurements", capture)
    monkeypatch.setattr(worker.early.late, "compare_prefix",
                        lambda actual, sealed: {"passed": actual == sealed})
    other = {"physics_step": 57, "guard_failure": None}
    worker.guard_relief(other, {})
    assert other == {"physics_step": 57, "guard_failure": None}
    at_branch = {"physics_step": 58, "guard_failure": None}
    worker.guard_relief(at_branch, {"observation": "same tick"})
    assert at_branch["guard_failure"] is None and at_branch["wrist_relief_admission"]["passed"]
    worker.guard_relief(at_branch, {"observation": "different tick"})
    assert at_branch["guard_failure"] == "wrist_relief_prefix_mismatch"
