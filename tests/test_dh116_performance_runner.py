import importlib.util
import json
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_rejects_nonpositive_repeats():
    with pytest.raises(SystemExit):
        load("run_performance").parse_args(
            ["--variant", "baseline", "--phase", "approach", "--repeat", "0"])


def test_report_uses_median_wall_time():
    report = load("collect_performance").compare_runs(
        [{"wall_s": 3.0}, {"wall_s": 2.0}, {"wall_s": 4.0}])
    assert report["median_wall_s"] == 3.0


def test_collector_marks_short_group_incomparable(tmp_path):
    for index, wall in enumerate((1., 2.)):
        run = tmp_path / f"run{index}"
        run.mkdir()
        (run / "comparison_manifest.json").write_text(json.dumps({
            "variant": "baseline", "phase": "approach", "repetition": index + 1}))
        (run / "perf_summary.json").write_text(json.dumps({
            "physics_dt_s": 1e-4,
            "stages": {"approach": {"steps": 10, "completed_steps": 10, "wall_s": wall}}}))
        (run / "run_manifest.json").write_text(json.dumps({
            "run_config": {"wire_py_dt": 1e-5, "contact_subsamples": 16}}))
    output = tmp_path / "report.json"
    report = load("collect_performance").collect(
        [tmp_path / "run0", tmp_path / "run1"], output)
    assert report["groups"][0]["comparable"] is False
