import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import probe_dh116_simulation_app_lifecycle as probe


def test_minimal_and_single_variable_launches():
    official = probe.launch_options("C0")
    no_stage = probe.launch_options("C1")
    stage = probe.launch_options("C2")
    no_camera = probe.launch_options("C3")
    assert official == ({"headless": True, "fast_shutdown": False}, {})
    assert stage[0] == dict(no_stage[0], create_new_stage=True)
    assert stage[1] == no_stage[1]
    assert no_stage[0]["create_new_stage"] is False
    assert no_camera[0] == no_stage[0]
    assert no_camera[1]["experience"] != no_stage[1]["experience"]


def test_runtime_map_only_reports_actual_mapped_paths():
    maps = [
        "0000-1000 r-xp 0 08:01 3 /env/kit/_appwindow.so",
        "1000-2000 r--p 0 08:01 3 /env/kit/_appwindow.so",
        "2000-3000 r-xp 0 08:01 4 /usr/lib/libcuda.so.1",
        "3000-4000 r-xp 0 08:01 0 [vdso]",
        "4000-5000 r-xp 0 08:01 5 /tmp/trace.log",
        "5000-6000 r-xp 0 08:01 6 /env/space path/libcudart.so.12 (deleted)",
    ]
    assert probe.mapped_runtime_libraries(maps) == [
        "/env/kit/_appwindow.so",
        "/env/space path/libcudart.so.12 (deleted)",
        "/usr/lib/libcuda.so.1",
    ]


def test_prelaunch_imports_exclude_sdk_project_and_torch():
    assert probe.preloaded_runtime_modules(["os", "omni.usd", "pxr", "carb", "torch.cuda",
                                            "go2_pvcnn.control"]) == [
        "carb", "go2_pvcnn.control", "omni.usd", "pxr", "torch.cuda",
    ]


def test_plan_does_not_import_or_launch_sdk(tmp_path):
    output = tmp_path / "preflight"
    result = subprocess.run([sys.executable, "-m", "scripts.probe_dh116_simulation_app_lifecycle",
                             "--case", "C0", "--output", str(output)],
                            check=True, capture_output=True, text=True)
    assert result.returncode == 0
    plan = json.loads((output / "plan.json").read_text())
    assert plan["native_execution_performed"] is False
    assert plan["preloaded_runtime_modules"] == []
    assert not (output / "app_close_returned.json").exists()


def test_preclose_evidence_survives_close_exception(tmp_path):
    class FakeApp:
        def update(self):
            pass

        def close(self):
            raise RuntimeError("close failed")

    events = []

    def snapshot(app, case, phase):
        events.append(phase)
        return {"phase": phase, "case": case}

    with pytest.raises(RuntimeError, match="close failed"):
        probe.run_lifecycle(FakeApp(), "C1", tmp_path, snapshot=snapshot)
    assert events == ["before_explicit_update", "before_app_close"]
    assert json.loads((tmp_path / "before_app_close.json").read_text())["case"] == "C1"
    assert not (tmp_path / "app_close_returned.json").exists()


def test_close_return_marker_never_asserts_process_exit(tmp_path):
    class FakeApp:
        updates = 0

        def update(self):
            self.updates += 1

        def close(self):
            assert self.updates == 1

    probe.run_lifecycle(FakeApp(), "C3", tmp_path, snapshot=lambda app, case, phase: {})
    marker = json.loads((tmp_path / "app_close_returned.json").read_text())
    assert marker == {"case": "C3", "app_close_returned": True, "process_exit_verified": False}


def test_display_rejected_before_kit_import(tmp_path):
    import os

    output = tmp_path / "bad-display"
    result = subprocess.run([sys.executable, "-m", "scripts.probe_dh116_simulation_app_lifecycle",
                             "--case", "C0", "--output", str(output), "--execute-native"],
                            env={**os.environ, "DISPLAY": ":7"}, capture_output=True, text=True)
    assert result.returncode == 1
    assert "require an empty display environment" in result.stderr
    assert not (output / "before_explicit_update.json").exists()


def test_kit_package_records_git_commit_without_loading_kit(tmp_path, monkeypatch):
    metadata = tmp_path / "PACKAGE-INFO.yaml"
    metadata.write_text("Package: OmniverseKit\nVersion: 106.5\nCommit: deadbeef\nTime: ignored\n")
    monkeypatch.setenv("CARB_APP_PATH", str(tmp_path))
    assert probe._kit_package() == {"path": str(metadata), "sha256": probe.sha256(metadata),
                                    "package": "OmniverseKit", "version": "106.5", "commit": "deadbeef"}



def test_all_camera_control_recipes_pin_same_debugger():
    configs = Path(probe.__file__).resolve().parents[1] / "configs" / "dh116_execution"
    recipes = {
        case: json.loads((configs / f"deformx_camera_{case.lower()}.json").read_text())
        for case in probe.CASES
    }
    gdb_sha = "a6222ff0f450ca040e3479c04fdb7d0f1e0cabb099688270e720b7115d933c3d"
    for recipe in recipes.values():
        assert recipe["debugger"] == "gdb_on_segv"
        assert recipe["inputs"]["/usr/bin/gdb"] == gdb_sha
    assert recipes["C1"]["inputs"] == recipes["C2"]["inputs"]
    assert recipes["C1"]["environment"] == recipes["C2"]["environment"]
