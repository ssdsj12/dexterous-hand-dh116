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
    if count not in (32, 64) or size.shape != (3,) or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError("32 or 64 lanes and a positive original table size required")
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


def _gpu_memory(gpu):
    return int(subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True).strip())


def _author(sim, owners, trace, project, prepared, task, lanes):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as assets
    from go2_pvcnn.control.dh116.tabletop_deformx_end_supported_scene import _bind_support_material
    from go2_pvcnn.control.dh116.tabletop_dual_pinch_builder import _Composers
    from go2_pvcnn.control.dh116.tabletop_end_supported_scene import author_end_supports
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import prepare_initial_state
    from go2_pvcnn.control.dh116.tabletop_vec_isaac import spawn_scene

    seed, active, _, _ = prepare_initial_state(task, prepared["table"])
    args, layout, sources = assets.spawn_parameters(project, prepared)
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
        support = deepcopy(task["support_scene"])
        for box in support["supports"]:
            box["center_m"] = (np.asarray(box["center_m"]) + origin).tolist()
        assets._bind_materials(sim.stage, scoped)
        report = assets.inspect_source_stage(sim.stage, scoped)
        if report["original_colliders"] != 34:
            raise ValueError("complete 34 original colliders per lane required")
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
    return scenes, seed


def _exercise(sim, scenes, seed, prepared, output, gpu, write_new):
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

    for scene in scenes:
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
    for expected in range(3, 11):
        sim.step(render=False)
        if sim.current_time_step_index != expected:
            raise ValueError("unexpected shared 480 Hz SDK boundary")
        _, poses = sample()
        minimum = min(minimum, clearance(poses))
    result = dict(physics_rows=8, environments=len(scenes), articulations=2*len(scenes),
        isolated_joint_restore=True, sdk_step=sim.current_time_step_index,
        min_source_clearance_m=minimum, step_and_readback_seconds=perf_counter()-started,
        gpu_used_mib=_gpu_memory(gpu), process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        solver_steps=0, teacher_queries=0, ppo_updates=0)
    write_new(output / "scene_observations.json", result)
    return result


def _run(project, task, experience, gpu, num_envs, output):
    from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary
    assert_clean_import_boundary()
    from isaaclab.app import AppLauncher
    argv = sys.argv
    try:
        sys.argv = argv[:1]
        app = AppLauncher(headless=True, device="cuda:0", enable_cameras=False,
                          fast_shutdown=False, experience=experience).app
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
            scenes, seed = _author(sim, owners, trace, project, prepared, task, lanes)
            result = _exercise(sim, scenes, seed, prepared, output, gpu, write_new)
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
                read_completed_sdk_step=lambda: completed_step, max_sdk_step=10,
                make_sim=make_sim, acquire=acquire, synchronize=lambda: torch.cuda.synchronize(0),
                clear_sim=lambda sim: clear_known_simulation_instance(sim, contract=clear_contract),
                stop=stop, manager_is_none=lambda: SimulationManager.get_physics_sim_view() is None,
                freeze=freeze, output=output, cleanup="explicit")
    finally:
        if not entered:
            app.close(wait_for_replicator=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, choices=(32,64), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.getenv("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("explicit matching physical CUDA device required")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    from scripts.audit_dh116_power_solver_capacity import load_power_inputs
    from scripts.run_dh116_work_packages import NATIVE_LOCK, native_file_lock
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
    from scripts.diagnose_dh116_teacher_initialization import write_new

    project = Path(__file__).resolve().parents[1]
    plan, _ = load_power_inputs(args.receipt, args.receipt_sha256)
    task_ref = plan["task_config"]
    task = json.loads(checked_file(project/"configs/dh116_tasks"/Path(task_ref["path"]).name,
                                   task_ref["sha256"]))
    receipt = json.loads(checked_file(args.receipt, args.receipt_sha256))
    frozen = json.loads(checked_file(args.receipt.parent/"native/frozen_plan.json",
                                    receipt["artifact_sha256"]["frozen_plan.json"]))
    experience = frozen["experience"]
    checked_file(Path(experience), frozen["frozen_files"][experience])
    args.output.mkdir(exist_ok=False)
    with native_file_lock(NATIVE_LOCK):
        if _gpu_memory(args.gpu) > 512:
            raise RuntimeError("GPU in use before batch scene launch")
        write_new(args.output/"probe_plan.json", dict(environments=args.num_envs, gpu=args.gpu,
            receipt_sha256=args.receipt_sha256, task_config_sha256=task_ref["sha256"],
            max_sdk_steps=10, solver_steps=0, ppo_updates=0))
        _run(project, task, experience, args.gpu, args.num_envs, args.output)


if __name__ == "__main__":
    main()
