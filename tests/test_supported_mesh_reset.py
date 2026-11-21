"""Original clear/add semantics over17 supported slots; explicit CPU doubles."""
from copy import deepcopy
import importlib
import numpy as np
import pytest

from test_native_teacher_supported_reset import supported_reset_case
from test_mesh_reset_order import native_semantics
from go2_pvcnn.control.dh116 import tabletop_mesh_reset_order as original
from go2_pvcnn.control.dh116.tabletop_power_shared_rod import PROXIMAL_POWER_MESH_NAMES, END_SUPPORT_NAMES, HAND_NAMES, PALM_NAMES, PROXIMAL_NAMES, SIDES


def case(monkeypatch, *, nodes=None):
    c = native_semantics(monkeypatch, case=supported_reset_case(monkeypatch, nodes=nodes))
    m = importlib.import_module("go2_pvcnn.control.dh116.tabletop_supported_mesh_reset")
    entries = []
    for name, (args, kw) in zip(c.rod.mesh_names, c.slots, strict=True):
        entries.append(dict(name=name, values=np.asarray(args[0], np.float32),
            origin=np.asarray(args[1], np.float32), spacing=float(args[2]),
            mu_s=kw["mu_s"], mu_k=kw["mu_k"],
            static_pose=(np.asarray(kw["pos"], np.float32), np.asarray(kw.get("quat", [0.,0.,0.,1.]), np.float32))
                if name in ("table", *END_SUPPORT_NAMES) else None))
    plan = original.seal_plan(dict(version=m.PLAN_VERSION, entries=entries,
        bindings=deepcopy(c.rod.bindings), cfg=deepcopy(c.rod._base.cfg),
        source_basis=dict(source_tree_sha256="cpu_fixture"), source_scope="cpu_fixture"))
    monkeypatch.setattr(m, "verify_plan_sources", lambda p: None)
    monkeypatch.setattr(m, "source_basis", lambda: deepcopy(plan["source_basis"]))
    monkeypatch.setattr(m, "verify_runtime_source", lambda r: None)
    def snapshot():
        def array(source, index):
            return np.stack([source[i][index] for i in range(17)])[None].astype(np.float32)
        pos, quat = array(c.rope.current, 0), array(c.rope.current, 1)
        return dict(schema="dh116.mesh_state_snapshot.v1", num_envs=1, mesh_count=17,
            current_positions_world_m=pos, current_quaternions_xyzw=quat,
            target_positions_world_m=array(c.rope.targets, 0),
            target_quaternions_xyzw=array(c.rope.targets, 1),
            solver_positions_world_m=pos.copy(), solver_quaternions_xyzw=quat.copy(),
            pending_teleport=np.array([[c.rope.tele[i] for i in range(17)]], np.uint8),
            device_state_source="cuda_device_copy", flags_source="native_host_sdf_store",
            owner_time_source="native_host_owner_metadata_at_readback",
            owner_time_s_before=c.rope.time, owner_time_s_after=c.rope.time,
            pending_set=c.rope.pending, moving=False, pair_ready=c.rope.pair_ready,
            pair_capacity=c.rope.capacity, last_error=0)
    c.rope.mesh_state_snapshot = snapshot
    c.events.clear()
    return c, m, plan


def execute(c, m, plan, **kw):
    txn = m.MeasuredSupportedMeshReset(c.seed, episode_uid="cpu17:new", mesh_plan=plan)
    return txn, txn.run(c.sim, c.rod, c.bindings, owner_resets=c.owners, **kw)


def test_all17_are_committed_at_actual_reset_poses_before_the_first_moving_target(monkeypatch):
    c, m, plan = case(monkeypatch)
    before = (c.sim.current_time_step_index, c.sim.current_time, c.rope.time, c.rope.steps, id(c.rope))
    txn, receipt = execute(c, m, plan)
    assert before == (c.sim.current_time_step_index, c.sim.current_time, c.rope.time, c.rope.steps, id(c.rope))
    assert c.events.count("clear_meshes") == 1 and c.events.count("add_mesh") == 17
    assert c.events.index("clear_meshes") > c.events.index("kinematic_refresh")
    assert receipt["mesh_reset"]["mesh_names"] == list(PROXIMAL_POWER_MESH_NAMES)
    assert receipt["mesh_state_readback"]["complete"]
    assert receipt["mesh_state_readback"]["after"]["snapshot"]["pair_capacity"] == 41*17*12
    assert len(receipt["pad_poses_xyzw"]) == 14
    expected = {}
    for slot, entry in enumerate(plan["entries"]):
        pos, quat = entry["static_pose"] or receipt["pad_poses_xyzw"][entry["name"]]
        expected[slot] = np.asarray(pos, np.float32)
        np.testing.assert_array_equal(c.rope.current[slot][0], expected[slot])
        assert not c.rope.tele[slot]
        # First moving target must sweep from the new measured Reset, never
        # from an old pose or a made-up stationary first physical step.
        c.rope.set_mesh_poses(slot, expected[slot]+np.array([.001,0,0],np.float32), quat, teleport=False)
    c.rope.step_free(1/480,12)
    for slot in range(17):
        np.testing.assert_array_equal(c.rope.started_from[slot][0], expected[slot])
    assert not receipt["native_qualification"] and not receipt["teacher_admission"]
    assert not receipt["mesh_state_readback"]["teacher_admission"]
    # The old public validator remains an eleven-slot contract.
    from go2_pvcnn.control.dh116.tabletop_mesh_reset_readback import validate_capture
    with pytest.raises(ValueError, match="eleven"):
        validate_capture(receipt["mesh_state_readback"]["after"])


def test_seventeen_slot_readback_refuses_pose_capacity_owner_and_source_changes(monkeypatch):
    c, m, plan = case(monkeypatch)
    _, receipt = execute(c, m, plan)
    for damage in ("pose", "capacity", "owner", "slot_order", "pending", "rebuild_version", "step_counter"):
        packet = deepcopy(receipt["mesh_state_readback"])
        reset = deepcopy(receipt)
        if damage == "pose": packet["after"]["snapshot"]["current_positions_world_m"][0,16,0] += .01
        elif damage == "capacity": packet["after"]["snapshot"]["pair_capacity"] = 5412
        elif damage == "owner": packet["after"]["source_instance_id"] += 1
        elif damage == "slot_order": packet["mesh_names"][15:] = reversed(packet["mesh_names"][15:])
        elif damage == "pending": packet["after"]["snapshot"]["pending_set"] = True
        elif damage == "rebuild_version": reset["mesh_reset"]["version"] = "foreign_reset"
        else: reset["mesh_reset"]["new_solver_steps"] = 1
        with pytest.raises(ValueError):
            m.validate_supported_readback(packet, reset_receipt=reset, mesh_plan=plan)


def test_failed_rebuild_keeps_completed_reset_and_poisoned_clock_without_reusable_owner(monkeypatch):
    c, m, plan = case(monkeypatch)
    add = c.rope.add_mesh
    failure = RuntimeError("last proximal append failure")
    def bad(*args, **kw):
        slot = add(*args, **kw)
        if slot == 16: raise failure
        return slot
    c.rope.add_mesh = bad
    txn = m.MeasuredSupportedMeshReset(c.seed, episode_uid="cpu17:failed", mesh_plan=plan)
    rows = []
    with pytest.raises(RuntimeError) as caught:
        txn.run(c.sim,c.rod,c.bindings,owner_resets=c.owners,record_readback=rows.append)
    assert caught.value is failure and c.rod._needs_reset and txn.status == "failed"
    assert c.rope.steps == 0 and c.sim.current_time_step_index == 2
    assert rows[-1]["complete"] is False and rows[-1]["capture_error"]
    assert rows[-1]["completed_reset_receipt"]["status"] == "complete"
    with pytest.raises(RuntimeError):
        txn.run(c.sim,c.rod,c.bindings,owner_resets=c.owners)

def test_plan_reconstructs_real_locked_sources_and_refuses_resealed_constructor_changes(monkeypatch, tmp_path):
    """Real archived sources only; no native extension, Reset or snapshot claim."""
    import hashlib
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as profile
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import prepare_scene
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import load_candidate
    from scripts import verify_dh116_power_shared_rod as shared
    m = importlib.import_module("go2_pvcnn.control.dh116.tabletop_supported_mesh_reset")
    root = Path(__file__).resolve().parents[1]
    evidence = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119")
    native = evidence/"ref902/execution-closure-native-context-fixed/native"
    raw = (native/"tripod_control_764.json").read_bytes()
    outer = json.loads((native.parent/"outer.json").read_bytes())
    assert hashlib.sha256(raw).hexdigest() == outer["artifact_sha256"]["tripod_control_764.json"]
    bindings = deepcopy(decode_owned(json.loads(raw))["rod"]["mesh_bindings"])
    sources = dict(rays=Path("/mnt/mydisk/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/ag2203-power-palm-rays-enabled-native/native/palm_rays.json"),
        left_bake=Path("/mnt/mydisk/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/ag2184-left-power-sdf-native/bake"),
        right_bake=Path("/mnt/mydisk/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/ag2200-right-power-sdf-native/bake"),
        proximal_capture=Path("/mnt/mydisk/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/ag2312-power-proximal-tensor-ray-native"),
        proximal_bake=Path("/mnt/mydisk/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/ag2314-power-proximal-sdf-half-mm/bake"))
    # New constructor metadata is produced in this CPU fixture from authenticated
    # native source archives; the historical measurement record is not modified.
    for side in SIDES:
        archive = json.loads((sources["rays"].parent/(side+"_base_link.json")).read_bytes())
        key = side+".base_link"
        bindings[key].update(frame=archive["source"]["frame"],unit=archive["source"]["unit"],
                             sdf_path=str(sources[side+"_bake"]/(key+".npz")))
        archive = json.loads((sources["proximal_capture"]/"native"/(side+"_finger52_link.json")).read_bytes())
        key = side+".finger52_link"
        bindings[key].update(frame=archive["source"]["frame"],unit=archive["source"]["unit"],
                             sdf_path=str(sources["proximal_bake"]/(key+".npz")))
    prepared = prepare_scene(root)
    cfg = load_candidate(root/"configs/dh116_tasks/flexible_pinch_candidate_v1.json", root)
    rod = SimpleNamespace(bindings=bindings,_base=SimpleNamespace(cfg=cfg),
                          mesh_names=PROXIMAL_POWER_MESH_NAMES)
    task = tmp_path/"cpu-frozen-task.json"
    task.write_text(json.dumps(json.loads((native/"frozen_plan.json").read_bytes())["reset_config"]))
    monkeypatch.setattr(original,"verify_runtime_source",lambda rod: None)  # CPU constructor audit only.
    with profile.source_scope():
        sdf = original._cpu_sdf(profile.BUILD/"build/python/ropesim/sdf.py")
        plan = m.prepare_supported_mesh_reset_plan(
            root,prepared,rod,sdf_module=sdf,task_config=task,extra_sources=sources)
        m.verify_plan_sources(plan)
        assert [row["name"] for row in plan["entries"]] == list(PROXIMAL_POWER_MESH_NAMES)
        assert set(plan["original_plan"]["bindings"]) == set(HAND_NAMES)
        for key in PALM_NAMES+PROXIMAL_NAMES:
            entry = next(x for x in plan["entries"] if x["name"] == key)
            with np.load(bindings[key]["sdf_path"],allow_pickle=False) as grid:
                np.testing.assert_array_equal(entry["values"],grid["values"])
        # Independently compare support arrays with the actual retained helper.
        # Only the imported ropesim namespace is a CPU stand-in for its locked
        # NumPy sdf module; the production support generator itself is unchanged.
        import sys
        from go2_pvcnn.control.dh116.tabletop_deformx_end_supported_scene import make_support_sdf_grids
        support_scene = json.loads(task.read_bytes())["support_scene"]
        with monkeypatch.context() as patches:
            patches.setitem(sys.modules, "ropesim", SimpleNamespace(sdf=sdf))
            retained = make_support_sdf_grids(support_scene)
        for name, grid in zip(END_SUPPORT_NAMES, retained, strict=True):
            entry = next(x for x in plan["entries"] if x["name"] == name)
            for actual, expected in zip((entry["values"],entry["origin"],entry["spacing"]), grid.as_args(), strict=True):
                np.testing.assert_array_equal(actual, expected)
        # No CPU geometric change is allowed to masquerade as the same source
        # merely by recomputing an outer digest.
        for damage in ("extra_grid", "support_pose", "friction", "extra_binding", "implementation_source", "version"):
            bad = deepcopy(plan)
            bad.pop("sha256")
            if damage == "extra_grid": bad["entries"][16]["values"].flat[0] += .001
            elif damage == "support_pose": bad["entries"][13]["static_pose"][0][2] += .001
            elif damage == "friction": bad["entries"][15]["mu_s"] += .1
            elif damage == "extra_binding": bad["bindings"]["right.finger52_link"]["capture_sha256"] = "0"*64
            elif damage == "implementation_source": bad["implementation_sources"] = {}
            else: bad["version"] = "foreign_plan"
            with pytest.raises(ValueError):
                m.verify_plan_sources(original.seal_plan(bad))
