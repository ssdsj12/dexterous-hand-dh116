import importlib
import json


def module():
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_deformable_rod_bridge")


def test_groups_have_independent_minimal_dependencies():
    probe = importlib.import_module("scripts.verify_dh116_deformx_bridge_teardown")
    assert probe.group_features("A") == (False, False, False, False)
    assert probe.group_features("B") == (True, False, False, False)
    assert probe.group_features("C") == (True, True, False, False)
    assert probe.group_features("D") == (True, True, True, False)
    assert probe.group_features("E") == (True, True, True, True)


def test_ownership_trace_distinguishes_python_lifetime_and_sdk_release(tmp_path):
    class View:
        pass

    output = tmp_path / "ownership.jsonl"
    audit = module().NativeOwnershipTrace(output)
    value = View()
    audit.watch("view", value, owner="probe", release="invalidate", gpu_view=True)
    audit.checkpoint("before invalidation")
    audit.event("invalidate returned", resource="view", sdk_valid=False)
    del value
    audit.checkpoint("after dropping references")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["owner"] == "probe" and rows[0]["gpu_view"] is True
    assert rows[1]["python_liveness"] == {"view": True}
    assert rows[2]["sdk_valid"] is False
    assert rows[3]["python_liveness"] == {"view": False}


def test_legacy_rigid_entry_delegates_to_verified_full_bridge_lifecycle(monkeypatch):
    from argparse import Namespace
    from scripts import verify_dh116_deformx_bridge_teardown as teardown
    from scripts import verify_dh116_deformx_rigid_bridge as rigid

    args = Namespace(steps_per_phase=120)
    seen = []
    launcher = object()
    monkeypatch.setattr(teardown, "launch_one", lambda value, launch: seen.append(
        (value.group, value.steps_per_phase, launch)
    ))
    rigid.native_run(args, launcher)
    assert seen == [("E", 120, launcher)]
