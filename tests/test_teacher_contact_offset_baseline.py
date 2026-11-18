"""CPU fixtures for the exact-link PhysX offset diagnostic."""
from types import SimpleNamespace

import numpy as np
import pytest

from go2_pvcnn.control.dh116 import tabletop_contact_offset_baseline as probe
from scripts import diagnose_dh116_teacher_stat_cache_release as worker


CLOCK = dict(sdk_step=2, sdk_time_s=2/480)


def bindings():
    return [dict(actual_body_path=path) for path in probe.TARGET_LINKS]


class View:
    count = max_shapes = 1

    def __init__(self, path):
        self.prim_paths = [path]

    def get_contact_offsets(self):
        return np.array([[.0012]], dtype=float)

    def get_rest_offsets(self):
        return np.array([[-.0001]], dtype=float)


def read(create=View, *, clock=None, unit=1.0, rows=None):
    return probe.read_contact_offsets(
        SimpleNamespace(create_rigid_body_view=create), bindings() if rows is None else rows,
        meters_per_unit=unit, read_clock=(lambda: dict(CLOCK)) if clock is None else clock,
        expected_clock=CLOCK)


def test_single_native_shape_slot_is_read_without_guessing_face_identity():
    seen = []

    def make(path):
        seen.append(path)
        return View(path)

    report = read(make)
    assert seen == list(probe.TARGET_LINKS)
    assert report["complete"]
    assert report["sdk_clock_after"] == CLOCK
    for row in report["links"]:
        assert row["contact_offset_m"] == pytest.approx(.0012)
        assert row["rest_offset_m"] == pytest.approx(-.0001)
        assert row["native_shape_prim_path"] is None


@pytest.mark.parametrize("bad", ["ambiguous", "wrong_path", "getter", "nonfinite", "invalid_pair"])
def test_unresolved_or_invalid_readbacks_do_not_emit_measurements(bad):
    def make(path):
        v = View(path)
        if bad == "ambiguous":
            v.max_shapes = 2
        elif bad == "wrong_path":
            v.prim_paths = [path + "/wrong"]
        elif bad == "getter":
            def fail():
                raise RuntimeError("backend rejected")
            v.get_contact_offsets = fail
        elif bad == "nonfinite":
            v.get_contact_offsets = lambda: [[float("nan")]]
        else:
            v.get_rest_offsets = lambda: [[.002]]
        return v

    report = read(make)
    assert not report["complete"]
    assert all(row["contact_offset_m"] is None and row["rest_offset_m"] is None for row in report["links"])


def test_clock_change_discards_values_and_stops_further_reads():
    clocks = iter([CLOCK, dict(sdk_step=3, sdk_time_s=3/480)])
    seen = []

    def make(path):
        seen.append(path)
        return View(path)

    report = read(make, clock=lambda: next(clocks))
    assert len(seen) == 1
    assert not report["complete"]
    assert all(row["contact_offset_m"] is None for row in report["links"])


def test_wrong_unit_and_duplicate_bindings_decline_before_native_read():
    def make(path):
        pytest.fail("unexpected view creation")

    for args in [dict(unit=.01), dict(rows=bindings()+bindings()[:1])]:
        report = read(make, **args)
        assert not report["complete"]


def test_frozen_opt_in_is_bound_to_sdk_and_rejects_original_approval(tmp_path):
    root = worker.Path(worker.__file__).resolve().parents[1]
    output = tmp_path/"out"
    default = worker.freeze_plan(root, output, worker.legacy.BINDINGS)
    opted = worker.freeze_plan(root, output, worker.legacy.BINDINGS, contact_offset_baseline=True)
    assert "contact_offset_baseline" not in default
    assert opted["budget"] == default["budget"]
    contract = opted["contact_offset_baseline"]
    assert opted["frozen_files"][contract["api_path"]] == contract["api_sha256"]
    with pytest.raises(ValueError):
        worker.verify_plan(opted, worker.legacy.digest(default))
    worker.verify_plan(opted, worker.legacy.digest(opted))


def test_optional_readback_runs_after_raw_evidence_and_keeps_strict_rejection(tmp_path):
    import json

    packet = {"capture": {"complete": False, "reasons": ["STRICT_FAILURE"]}}
    seen = []

    def callback():
        assert json.loads((tmp_path/"stored_cooked_capture.json").read_text()) == packet
        seen.append("read")

    with pytest.raises(RuntimeError, match="STRICT_FAILURE"):
        worker.legacy.persist_capture(tmp_path, packet, after_write=callback)
    assert seen == ["read"]


def test_approved_opt_in_requires_same_cli_flag_before_native(tmp_path, monkeypatch):
    import json

    root = worker.Path(worker.__file__).resolve().parents[1]
    output, plan_path = tmp_path/"out", tmp_path/"plan.json"
    plan = worker.freeze_plan(root, output, worker.legacy.BINDINGS, contact_offset_baseline=True)
    plan_path.write_text(json.dumps(plan))
    arguments = ["--project", str(root), "--plan", str(plan_path), "--output", str(output),
                 "--execute-native", "--approved-frozen-sha256", worker.legacy.digest(plan)]
    calls = []
    monkeypatch.setattr(worker.legacy, "execute_native", lambda plan, **kw: calls.append(kw))
    with pytest.raises(ValueError, match="differs from approved plan"):
        worker.main(arguments)
    assert not calls
    assert worker.main(arguments + ["--contact-offset-baseline"]) == 0
    assert len(calls) == 1 and callable(calls[0]["after_cooked_capture"])
