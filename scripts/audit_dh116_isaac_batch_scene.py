#!/usr/bin/env python3
"""Bounded headless Power scene/PhysX capacity probe, without a rod or PPO."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace
from weakref import ref

import numpy as np


def scene_lanes(count, table_size):
    size = np.asarray(table_size, dtype=float)
    if count not in (2, 32, 64) or size.shape != (3,) or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError("2, 32 or 64 lanes and a positive original table size required")
    columns = math.ceil(math.sqrt(count))
    spacing = float(max(size[:2]) + 1.0)
    return [(f"/World/DH116BatchAudit/lane_{i:03d}",
             (spacing * (i % columns), spacing * (i // columns), 0.0))
            for i in range(count)]


def verify_isolated_write(before, after, changed_lane):
    if before.shape != after.shape or not np.isfinite(after).all():
        raise ValueError("matching finite PhysX lane state required")
    if not np.array_equal(np.delete(before, changed_lane, axis=0),
                          np.delete(after, changed_lane, axis=0)):
        raise ValueError("joint write changed another lane")
    if np.array_equal(before[changed_lane], after[changed_lane]):
        raise ValueError("target lane did not read back its joint write")


def batch_snapshot(references):
    """Apply the existing lifetime observer in bounded chunks with no sampling."""
    from scripts.dh116_pinch_startup_v3 import snapshot_for_startup

    items = list(references.items())
    if len(items) > 8 * 64 + 1:
        raise ValueError("resource witnesses exceed the bounded 64-lane probe")
    summary = snapshot_for_startup({})
    for first in range(0, len(items), 32):
        snapshot = snapshot_for_startup(dict(items[first:first+32]))
        summary["resources"].update(snapshot["resources"])
    if items:
        summary.pop("observation", None)
    return summary


def verify_mesh_layout(style, names):
    from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import MESH_NAMES
    from go2_pvcnn.control.dh116.tabletop_power_shared_rod import PROXIMAL_POWER_MESH_NAMES

    expected = {"pinch": MESH_NAMES, "tripod": PROXIMAL_POWER_MESH_NAMES,
                "power": PROXIMAL_POWER_MESH_NAMES}
    if style not in expected or tuple(names) != expected[style]:
        raise ValueError("original grasp-specific mesh layout required")


def load_grasp_archive(style, receipt_path, digest):
    """Read the exact grasp's measured Reset inputs before loading Isaac."""
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    from go2_pvcnn.control.dh116.tabletop_dual_pinch_builder import load_bound_pair_proposal
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file

    outer = json.loads(checked_file(receipt_path, digest))
    root = receipt_path.parent / "native"
    inventory = outer["artifact_sha256"]

    def checked_json(name):
        return json.loads(checked_file(root/name, inventory[name]))

    if outer["exit_code"] != 0 or outer.get("forced_exit", False):
        raise ValueError("successful source-grasp Reset required")
    captured = checked_json("reset_capture.json")
    closed = checked_json("app_close_returned.json")
    if (captured["sdk_step"] != 2 or captured["actual_rod_tick"] != 0
            or not captured["clearance_passed"] or captured["ppo_updates"] != 0
            or not closed["app_close_returned"] or closed["close_mode"] != "normal"):
        raise ValueError("measured original two-step Reset and normal close required")
    mesh_plan = decode_owned(checked_json("mesh_reset_state.json"))["mesh_plan"]
    verify_mesh_layout(style, [row["name"] for row in mesh_plan["entries"]])
    if style == "pinch":
        if (outer.get("style") != style or outer.get("auto_kill") is not False
                or captured["pair_proposal"] != outer["pair_proposal"]):
            raise ValueError("archived Pinch pair ownership differs")
        packet = load_bound_pair_proposal(outer["pair_proposal"])
        frozen = json.loads(checked_file(root/"frozen_plan.json", outer["plan_sha256"]))
        if frozen["pair_proposal"] != outer["pair_proposal"]:
            raise ValueError("Pinch pair differs from the recorded launch")
        task = None
    else:
        if (captured["style"] != style or outer["debugger_inferior_exit_code"] != 0
                or outer["debugger_inferior_signal"] is not None
                or captured["mesh_plan_sha256"] != mesh_plan["sha256"]
                or captured["task_config"] != mesh_plan["task_config"]):
            raise ValueError("separate supported grasp Reset and source plan required")
        frozen = checked_json("frozen_plan.json")
        task_ref = captured["task_config"]
        task = json.loads(checked_file(Path(task_ref["path"]), task_ref["sha256"]))
        packet = None
    experience = frozen["experience"]
    checked_file(Path(experience), frozen["frozen_files"][experience])
    return SimpleNamespace(style=style, mesh_plan=mesh_plan, task=task, pair=packet,
                           experience=experience, reset_receipt_sha256=digest,
                           task_seed_sha256=captured.get("reset_seed_sha256"))


def _gpu_memory(gpu):
    return int(subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True).strip())


def _author(sim, owners, trace, project, prepared, archive, lanes):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as assets
    from go2_pvcnn.control.dh116.tabletop_deformx_end_supported_scene import _bind_support_material
    from go2_pvcnn.control.dh116.tabletop_dual_pinch_builder import _Composers
    from go2_pvcnn.control.dh116.tabletop_end_supported_scene import author_end_supports
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import (POWER_SCOPE, TRIPOD_SCOPE, prepare_initial_state)
    from go2_pvcnn.control.dh116.tabletop_vec_isaac import spawn_scene

    args, layout, sources = assets.spawn_parameters(project, prepared)
    if archive.style == "pinch":
        from go2_pvcnn.control.dh116.tabletop_dual_pinch_builder import _paired_sources
        _paired_sources(project, archive.pair, prepared)
        seed = archive.pair["reset_seed"]
        rod_nodes = np.asarray(archive.pair["rod_seed"]["proposed_nodes_m"], dtype=np.float32)
    else:
        scope = {"tripod": TRIPOD_SCOPE, "power": POWER_SCOPE}[archive.style]
        seed, active, rod_nodes, _ = prepare_initial_state(archive.task, prepared["table"], expected_scope=scope)
        if seed["sha256"] != archive.task_seed_sha256:
            raise ValueError("current supported reset seed differs from archived readback")
        for side_id, side in enumerate(assets.SIDES):
            layout["hand_roots"][side] = np.asarray(seed["authored_root_seed_w"][side_id]).copy()
            layout["hand_roots"][side][2] -= assets.BUILDER_OFFSET_M
            layout["active_joints"][side] = active[side_id].copy()

    scenes = owners.keep("scenes", [])
    for lane, (prefix, origin) in enumerate(lanes):
        scene = spawn_scene(sim, args, layout, sources, prefix, origin, external_wire=True)
        scenes.append(scene)
        if len(scene["hands"]) != 2 or scene["wire"] is not None or scene["sensors"]:
            raise ValueError("original hands and no duplicate wire required")
        scoped = dict(prepared, prefix=prefix, env_id=lane, table=dict(prepared["table"],
                      center_m=(np.asarray(prepared["table"]["center_m"]) + origin).tolist()))
        assets._bind_materials(sim.stage, scoped)
        report = assets.inspect_source_stage(sim.stage, scoped)
        if report["original_colliders"] != 34:
            raise ValueError("complete 34 original colliders per lane required")
        if archive.task is not None:
            support = deepcopy(archive.task["support_scene"])
            for box in support["supports"]:
                box["center_m"] = (np.asarray(box["center_m"]) + origin).tolist()
            paths = author_end_supports(sim.stage, prefix, support)
            material = prepared["candidate_physics"]["table"]
            _bind_support_material(sim.stage, prefix, paths, material["mu_s"], material["mu_k"])
        for side, hand in zip(assets.SIDES, scene["hands"], strict=True):
            trace.watch(f"{lane}_{side}_hand", hand, owner="batch_scene_audit",
                        release="owned scope then SDK STOP", callbacks=True)
    if sim.current_time_step_index != 0:
        raise ValueError("authoring must not advance physics")
    sim.reset()
    members = []
    for lane, scene in enumerate(scenes):
        for side, hand in zip(assets.SIDES, scene["hands"], strict=True):
            hand.update(1 / 480)
            for label, owned in (
                ("view", hand.root_physx_view),
                ("permanent", hand.permanent_wrench_composer),
                ("instantaneous", hand.instantaneous_wrench_composer),
            ):
                trace.watch(f"{lane}_{side}_{label}", owned, owner="batch_scene_audit",
                            release="owned scope then SDK STOP", callbacks=True)
                if label != "view":
                    members.append(owned)
    owners.keep("composer", _Composers(members))
    assets.validate_simulation_settings(sim, prepared)
    if sim.current_time_step_index != 2:
        raise ValueError("one global reset and two SDK initialization steps required")
    return scenes, seed, rod_nodes


def _exercise(sim, scenes, seed, prepared, output, gpu, write_new, *, rod_spec=None):
    import torch
    from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import _targets
    from go2_pvcnn.control.dh116.tabletop_palm_mount import JOINT_NAMES, MountView

    def sample():
        poses = []
        q = []
        for scene in scenes:
            local_poses = []
            local_q = []
            for side, hand in zip(("left", "right"), scene["hands"], strict=True):
                view = hand.root_physx_view
                root = f"{scene['prefix']}/{side}Hand"
                if list(view.prim_paths) not in ([root], [root + "/wrist_fixed"]):
                    raise ValueError(f"unexpected actual articulation identity for {root}")
                link = view.get_link_transforms().detach().cpu().numpy().copy()
                joints = view.get_dof_positions().detach().cpu().numpy().copy()
                if (link.shape != (1, 23, 7) or joints.shape != (1, 17)
                        or not np.isfinite(link).all() or not np.isfinite(joints).all()):
                    raise ValueError("finite original articulation tensors required on every lane")
                local_poses.append(link[0])
                local_q.append(joints[0])
            poses.append(local_poses)
            q.append(local_q)
        return np.asarray(q), np.asarray(poses)

    def write_seed(scene):
        for side_id, hand in enumerate(scene["hands"]):
            q, qd = _targets(seed, side_id, hand.joint_names)
            pose = np.asarray(seed["reset_root_pose_w"])[side_id:side_id + 1].copy()
            pose[:, :3] += scene["origin"]
            ids = torch.tensor([0], device=hand.device)
            def tensor(value):
                return torch.as_tensor(value, device=hand.device, dtype=torch.float32)
            hand.reset(ids)
            hand.write_root_pose_to_sim(tensor(pose), env_ids=ids)
            hand.write_joint_state_to_sim(tensor(q), tensor(qd), env_ids=ids)
            hand.set_joint_position_target(tensor(q), env_ids=ids)
            MountView(hand).reset(tensor(pose), ids)
            hand.write_data_to_sim()
    for scene in scenes:
        write_seed(scene)
    sim.physics_sim_view.update_articulations_kinematic()
    baseline_q, baseline_poses = sample()
    write_new(output / "initialized.json", dict(sdk_step=sim.current_time_step_index,
        environments=len(scenes), articulations=2*len(scenes), gpu_used_mib=_gpu_memory(gpu)))

    last = scenes[-1]["hands"][0]
    witness = baseline_q[-1, 0].copy()
    joint = last.joint_names.index(JOINT_NAMES[0])
    witness[joint] += 0.001
    def write_selected(value):
        pos = torch.as_tensor(value[None], device=last.device, dtype=torch.float32)
        last.write_joint_state_to_sim(pos, torch.zeros_like(pos))
        sim.physics_sim_view.update_articulations_kinematic()

    write_selected(witness)
    changed_q, changed_poses = sample()
    verify_isolated_write(baseline_q, changed_q, len(scenes)-1)
    verify_isolated_write(baseline_poses, changed_poses, len(scenes)-1)
    if not np.array_equal(changed_q[-1, 1], baseline_q[-1, 1]):
        raise ValueError("the other hand changed during the isolated write")
    if not np.isclose(changed_q[-1, 0, joint], witness[joint], atol=1e-6, rtol=0):
        raise ValueError("selected wrist did not read back the intended value")
    write_selected(baseline_q[-1, 0])
    restored_q, restored_poses = sample()
    if (not np.array_equal(restored_q, baseline_q)
            or not np.allclose(restored_poses, baseline_poses, atol=1e-6, rtol=0)
            or sim.current_time_step_index != 2):
        raise ValueError("no-step joint restore changed the reference state or clock")

    from go2_pvcnn.control.dh116.tabletop_batch_geometry import collision_clearance
    from go2_pvcnn.control.dh116.tabletop_deformable_dual_hand_episode import CLEARANCE_M
    shapes = [(i, scenes[0]["hands"][i].body_names.index(name), row["vertices"])
              for i, side in enumerate(("left", "right"))
              for name, row in prepared["hands"][side]["source_colliders"].items()]

    def clearance(poses):
        value = float(collision_clearance(poses[..., [0,1,2,6,3,4,5]], shapes)[0].min())
        if value < CLEARANCE_M:
            raise ValueError(f"original wrist clearance floor rejected {value} m")
        return value

    minimum = clearance(restored_poses)
    started = perf_counter()

    def step(expected):
        nonlocal minimum
        sim.step(render=False)
        if sim.current_time_step_index != expected:
            raise ValueError("unexpected shared 480 Hz SDK boundary")
        _, poses = sample()
        minimum = min(minimum, clearance(poses))

    def reset_hand_lane(scene):
        before, before_poses = sample()
        write_seed(scene)
        sim.physics_sim_view.update_articulations_kinematic()
        after, after_poses = sample()
        if not np.array_equal(before[:-1], after[:-1]) or not np.array_equal(before_poses[:-1], after_poses[:-1]):
            raise ValueError("zero-step restore changed another native hand lane")

    coupled = {"solver_steps": 0}
    if rod_spec is None:
        for expected in range(3, 11):
            step(expected)
    else:
        from scripts.audit_dh116_batch_rod_exchange import coupled_probe
        archive, nodes, owners = rod_spec
        coupled = coupled_probe(sim, scenes, archive, nodes, owners, reset_hand_lane, step, output)
    result = dict(physics_rows=8, environments=len(scenes), articulations=2*len(scenes),
        isolated_joint_restore=True, sdk_step=sim.current_time_step_index,
        min_source_clearance_m=minimum, step_and_readback_seconds=perf_counter()-started,
        gpu_used_mib=_gpu_memory(gpu), process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        teacher_queries=0, ppo_updates=0, **coupled)
    write_new(output / "scene_observations.json", result)
    return result


def _run(project, archive, gpu, num_envs, output, *, coupled, max_sdk_step=10, extra_exercise=None):
    from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary
    assert_clean_import_boundary()
    from isaaclab.app import AppLauncher
    argv = sys.argv
    try:
        sys.argv = argv[:1]
        app = AppLauncher(headless=True, device="cuda:0", enable_cameras=False,
                          fast_shutdown=False, experience=archive.experience).app
    finally:
        sys.argv = argv
    entered = False
    try:
        import carb.settings
        import isaaclab.sim as sim_utils
        from isaacsim.core.simulation_manager import SimulationManager
        import omni.timeline
        import torch
        from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_provider import _run_lifecycle_with_stat_cache
        from go2_pvcnn.control.dh116.tabletop_preview_launcher import commit_preview_stop
        from go2_pvcnn.control.dh116.tabletop_simulation_owner_cleanup import (
            clear_known_simulation_instance, simulation_source_contract)
        from go2_pvcnn.control.dh116.tabletop_stat_cache_release import stat_cache_source_contract
        from scripts import dh116_pinch_acquisition_v2 as lifecycle_bindings
        from scripts import probe_dh116_physx_lifetime as lifetime_trace
        from scripts.dh116_fault_safe_lifecycle_v3 import run_lifecycle
        from scripts.dh116_pinch_startup_v3 import StartupTrace
        from scripts.diagnose_dh116_teacher_initialization import clock, write_new
        from scripts.diagnose_dh116_teacher_stat_cache_release import make_before_collect
        from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

        settings = carb.settings.get_settings()
        clear_contract = simulation_source_contract()
        cache = make_before_collect(dict(output=str(output),
            stat_cache_release=dict(source_contract=stat_cache_source_contract(), max_reset_calls=1)))
        sim_reference = None
        completed_step = None

        def freeze():
            settings.set_bool("/app/player/playSimulations", False)
            return settings.get("/app/player/playSimulations")

        def make_sim():
            nonlocal sim_reference
            sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
                dt=1/480, device="cuda:0", render_interval=1, gravity=(0.,0.,-9.81),
                physx=sim_utils.PhysxCfg(solver_type=0)))
            sim_reference = ref(sim)
            return sim

        def acquire(sim, owners, trace, cleanup):
            nonlocal completed_step
            from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import prepare_scene
            prepared = prepare_scene(project)
            lanes = scene_lanes(num_envs, prepared["table"]["size_m"])
            scenes, seed, rod_nodes = _author(sim, owners, trace, project, prepared, archive, lanes)
            if extra_exercise is None:
                result = _exercise(sim, scenes, seed, prepared, output, gpu, write_new,
                    rod_spec=(archive, rod_nodes, owners) if coupled else None)
            else:
                result = extra_exercise(sim, scenes, seed, prepared, archive, rod_nodes, owners,
                                        output, gpu, max_sdk_step)
            result["style"] = archive.style
            completed_step = result["sdk_step"]
            return result

        def read_sim():
            return sim_reference() if sim_reference is not None else None

        def stop():
            def read_clock():
                simulation = read_sim()
                return clock(simulation) if simulation is not None else None
            result = commit_preview_stop(omni.timeline.get_timeline_interface(),
                                         freeze=freeze, read_clock=read_clock)
            write_new(output / "stop_commit.json", result)

        names = tuple(f"{i}_{side}_{suffix}" for i in range(num_envs)
                      for side in ("left", "right")
                      for suffix in ("hand", "view", "permanent", "instantaneous"))
        with bindings((lifecycle_bindings, "OWNED_NAMES", names),
                      (lifecycle_bindings, "LifetimeTrace", StartupTrace),
                      (lifecycle_bindings, "snapshot_weakrefs", batch_snapshot),
                      (lifetime_trace, "snapshot_weakrefs", batch_snapshot)):
            entered = True
            return _run_lifecycle_with_stat_cache(SimpleNamespace(update=app.update,
                close=lambda: app.close(wait_for_replicator=False)),
                lifecycle=run_lifecycle, cache_release=cache, read_sim=read_sim,
                read_completed_sdk_step=lambda: completed_step, max_sdk_step=max_sdk_step,
                make_sim=make_sim, acquire=acquire, synchronize=lambda: torch.cuda.synchronize(0),
                clear_sim=lambda sim: clear_known_simulation_instance(sim, contract=clear_contract),
                stop=stop, manager_is_none=lambda: SimulationManager.get_physics_sim_view() is None,
                freeze=freeze, output=output, cleanup="explicit")
    finally:
        if not entered:
            app.close(wait_for_replicator=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", choices=("pinch", "tripod", "power"), default="power")
    parser.add_argument("--coupled", action="store_true", help="add the original live rod and reaction exchange")
    parser.add_argument("--num-envs", type=int, choices=(32,64), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.getenv("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("explicit matching physical CUDA device required")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    from scripts.run_dh116_work_packages import NATIVE_LOCK, native_file_lock
    from scripts.diagnose_dh116_teacher_initialization import write_new

    project = Path(__file__).resolve().parents[1]
    archive = load_grasp_archive(args.style, args.receipt, args.receipt_sha256)
    args.output.mkdir(exist_ok=False)
    with native_file_lock(NATIVE_LOCK):
        if _gpu_memory(args.gpu) > 512:
            raise RuntimeError("GPU in use before batch scene launch")
        write_new(args.output/"probe_plan.json", dict(environments=args.num_envs, gpu=args.gpu,
            receipt_sha256=args.receipt_sha256, style=args.style, coupled=args.coupled,
            max_sdk_steps=10, ppo_updates=0))
        if args.coupled:
            from go2_pvcnn.control.dh116.tabletop_mesh_observation_source import BACKEND, BACKEND_SHA
            from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
            checked_file(BACKEND, BACKEND_SHA)
            sys.path.insert(0, str(BACKEND.parents[1]))
            from ropesim import _ropesim
            if Path(_ropesim.__file__).resolve() != BACKEND:
                raise ValueError("the locked DeformX binary must own the coupled probe")
        _run(project, archive, args.gpu, args.num_envs, args.output, coupled=args.coupled)


if __name__ == "__main__":
    main()
