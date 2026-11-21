"""Fresh17 scene assembly with original USD/SDFs and CPU SDK/solver doubles.

These exercise construction and provenance, never native contact qualification.
"""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_dh116_dual_pinch_builder import setup as original_setup, prepared, geometry, ROOT

MODULE = "go2_pvcnn.control.dh116.tabletop_fresh_supported_builder"
EVIDENCE = Path("/mnt/mydisk/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119")


def arrange(monkeypatch, tmp_path, prepared, geometry):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import NativeOwners
    from go2_pvcnn.control.dh116 import tabletop_mesh_reset_order as mesh
    profile = BackendProfile.load(ROOT)
    task = tmp_path/"tripod-task.json"
    frozen = EVIDENCE/"ref902/execution-closure-native-context-fixed/native/frozen_plan.json"
    task.write_text(json.dumps(json.loads(frozen.read_bytes())["reset_config"]))
    sim, trace, resets, sync, record, rows, events, refs, meta, drop = original_setup(
        monkeypatch, tmp_path, prepared, geometry, use_layout=True)
    package = sys.modules["ropesim"]
    package._ropesim.__file__ = profile.binary_path
    package._ropesim.RopeBatch = package.RopeBatch
    monkeypatch.setitem(sys.modules, "ropesim._ropesim", package._ropesim)
    sdf = mesh._cpu_sdf(Path(profile.binary_path).parent/"sdf.py")
    package.sdf = sdf
    monkeypatch.setitem(sys.modules, "ropesim.sdf", sdf)
    captures = []
    rope = package.RopeBatch
    capture = rope.enable_contact_pair_capture
    def enable(value, capacity):
        captures.append((len(value.slots), capacity))
        capture(value, capacity)
    monkeypatch.setattr(rope, "enable_contact_pair_capture", enable)
    extra = dict(rays=EVIDENCE/"ag2203-power-palm-rays-enabled-native/native/palm_rays.json",
        left_bake=EVIDENCE/"ag2184-left-power-sdf-native/bake",
        right_bake=EVIDENCE/"ag2200-right-power-sdf-native/bake",
        proximal_capture=EVIDENCE/"ag2312-power-proximal-tensor-ray-native",
        proximal_bake=EVIDENCE/"ag2314-power-proximal-sdf-half-mm/bake")
    return SimpleNamespace(sim=sim, trace=trace, resets=resets, rows=rows, events=events,
        record=record, meta=meta, drop=drop, owners=NativeOwners(), captures=captures,
        options=dict(project=ROOT, prepared=prepared, ownership=trace, record=record,
            backend_profile=profile, task_config=task,
            task_config_sha256=hashlib.sha256(task.read_bytes()).hexdigest(),
            extra_sources=extra, run_uuid="cpu-fresh17", episode_uid="cpu-fresh17:first",
            physics_budget_end=18))


def module():
    spec = importlib.util.find_spec(MODULE)
    assert spec is not None, "missing direct17 fresh supported scene builder"
    return importlib.import_module(MODULE)


def test_direct17_is_fresh_and_preserves_real_source_arrays_without_an_old_reset(monkeypatch, tmp_path, prepared, geometry):
    m = module()
    c = arrange(monkeypatch, tmp_path, prepared, geometry)
    from go2_pvcnn.control.dh116.tabletop_power_shared_rod import PROXIMAL_POWER_MESH_NAMES
    from go2_pvcnn.control.dh116 import tabletop_mesh_reset_order as mesh
    fresh = m.build_fresh_supported_owned(c.sim, c.owners, **c.options)
    episode = fresh.episode
    rod = episode._rod
    assert episode._state == "new" and episode._exchange is None
    assert tuple(rod.mesh_names) == PROXIMAL_POWER_MESH_NAMES and len(rod.bindings) == 14
    assert rod._base.physics_step is None and rod._base.reset_id == rod._base.rod_tick == 0
    assert rod.rope.steps == rod.rope.resets == 0 and rod._base.last is None
    assert c.captures == [(17, 41*17*12)]
    assert c.sim.current_time_step_index == 2 and c.sim.current_time == 2/480
    assert c.events.count("setup_reset") == 1
    assert not any(row.get("kind") in ("reset", "physical") for row in c.rows)
    assert [row["phase"] for row in fresh.setup_events] == ["before_reset", "after_reset"]
    assert c.owners.resources["runtime"] is rod and c.owners.resources["dual_episode"] is episode
    assert fresh.reset_seed["sha256"] == fresh.reset_seed_sha256
    np.testing.assert_array_equal(c.meta["constructor_nodes"][0],
        np.asarray(json.loads(c.options["task_config"].read_bytes())["rod_nodes_w_m"], np.float32))
    assert not c.meta["pinned"].any()
    # Constructor readback is independent of the copied/frozen plan.
    for name, (args, kw) in zip(rod.mesh_names, rod.rope.slots, strict=True):
        entry = next(row for row in fresh.mesh_plan["entries"] if row["name"] == name)
        np.testing.assert_array_equal(args[0], entry["values"])
        np.testing.assert_array_equal(args[1], entry["origin"])
        assert args[2] == entry["spacing"] and kw["mu_s"] == entry["mu_s"] and kw["mu_k"] == entry["mu_k"]
    assert fresh.mesh_plan["sha256"] == mesh._digest({k:v for k,v in fresh.mesh_plan.items() if k != "sha256"})
    for side in ("left", "right"):
        hand = c.owners.resources["scene"]["hands"][("left", "right").index(side)]
        for name, bound in rod.bindings.items():
            if bound["side"] != side:
                continue
            actual = hand.link_poses[0, bound["body_index"]].numpy()
            slot = rod.mesh_names.index(name)
            np.testing.assert_array_equal(np.concatenate((rod.rope.slots[slot][1]["pos"], rod.rope.slots[slot][1]["quat"])), actual)
    c.drop()


@pytest.mark.parametrize("damage", ["task_sha", "scope", "backend"])
def test_invalid_fresh_sources_fail_before_scene_or_sdk_mutation(monkeypatch, tmp_path, prepared, geometry, damage):
    m = module()
    c = arrange(monkeypatch, tmp_path, prepared, geometry)
    if damage == "task_sha":
        c.options["task_config"].write_bytes(c.options["task_config"].read_bytes()+b" ")
    elif damage == "scope":
        data = json.loads(c.options["task_config"].read_bytes())
        data["scope"] = "power_end_supported_contact_diagnostic"
        c.options["task_config"].write_text(json.dumps(data))
        c.options["task_config_sha256"] = hashlib.sha256(c.options["task_config"].read_bytes()).hexdigest()
    else:
        sys.modules["ropesim"]._ropesim.__file__ = "/wrong/backend.so"
    with pytest.raises(ValueError):
        m.build_fresh_supported_owned(c.sim, c.owners, **c.options)
    assert c.sim.current_time_step_index == 0 and not c.events
    assert not c.owners.resources and not c.captures
    c.drop()
