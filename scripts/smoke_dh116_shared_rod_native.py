#!/usr/bin/env python3
"""Bounded real Isaac/shared-DeformX reaction probe; no policy or PPO."""
import argparse
from copy import deepcopy
from functools import partial
import os
from pathlib import Path
import sys

import numpy as np

from scripts.audit_dh116_isaac_batch_scene import _run, _exercise, load_grasp_archive, _gpu_memory
from scripts.audit_dh116_deformx_shared_pairs import load_inputs, make_solver
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file


def coupled(sim, scenes, seed, prepared, archive, nodes, owners, output, gpu, end, *, reset, backend):
    import torch
    from scripts.audit_dh116_batch_rod_exchange import _live_contact_geometry
    from scripts.verify_dh116_deformx_thumb_contact import step_thumb_physics
    from scripts.diagnose_dh116_teacher_initialization import write_new
    from go2_pvcnn.control.dh116.tabletop_native_teacher_collector import NativeJournal
    from go2_pvcnn.control.dh116.tabletop_shared_rod_owner import SharedRodOwner
    from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import route_world_wrenches, write_composer_reactions
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, MIMIC, active_limits
    from go2_pvcnn.control.dh116.tabletop_task import joint_safety_failure
    from go2_pvcnn.control.dh116.tabletop_batch_geometry import collision_clearance
    from go2_pvcnn.control.dh116.tabletop_deformable_dual_hand_episode import CLEARANCE_M

    # Reuse measured original hand Reset, isolation and eight-step scene checks.
    _exercise(sim, scenes, seed, prepared, output, gpu, write_new)
    plan = archive.mesh_plan
    bindings = []
    for scene in scenes:
        names = deepcopy(plan["bindings"])
        for row in names.values():
            row["prim_path"] = f"{scene['prefix']}/{row['side']}Hand/{row['body_name']}"
        bindings.append(names)
    offsets = np.asarray([scene["origin"] for scene in scenes], dtype=np.float32)
    capacity = 41 * len(plan["entries"]) * 12
    solver = make_solver(backend, plan, reset, offsets, style=archive.style,
                         capacity=capacity, empty_lane=False)
    owners.keep("shared_rod_solver", solver)
    measured = [_live_contact_geometry(scene, names)
                for scene, names in zip(scenes, bindings, strict=True)]
    for slot, entry in enumerate(plan["entries"]):
        name = entry["name"]
        if name in bindings[0]:
            positions = np.asarray([row[0][name][0] for row in measured], dtype=np.float32)
            rotations = np.asarray([row[0][name][1] for row in measured], dtype=np.float32)
            solver.set_mesh_poses(slot, positions, rotations, teleport=True)
    runtime = SharedRodOwner(solver, run_uuid="native-shared-rod-probe",
        physics_step=sim.current_time_step_index,
        episodes=tuple(f"native:{i}:1" for i in range(len(scenes))),
        mesh_names=tuple(row["name"] for row in plan["entries"]),
        solver_substeps=12, device="cuda")
    shapes = [(side_id, scenes[0]["hands"][side_id].body_names.index(name), row["vertices"])
              for side_id, side in enumerate(("left", "right"))
              for name, row in prepared["hands"][side]["source_colliders"].items()]
    nonzero = [0] * len(scenes)
    reaction = None
    journal = NativeJournal(output / "shared_coupling.jsonl")
    try:
        for sdk in range(sim.current_time_step_index + 1, end + 1):
            for lane, (scene, names, (poses, centers)) in enumerate(zip(scenes, bindings, measured, strict=True)):
                w = (np.zeros((len(plan["entries"]), 6)) if reaction is None
                     else reaction["mesh_wrenches"][lane].detach().cpu().numpy())
                named = {entry["name"]: w[i] for i, entry in enumerate(plan["entries"])}
                routes, _ = route_world_wrenches(
                    {key: named[key] for key in (*names, "table")}, names, poses, centers)
                if any(np.linalg.norm(row["forces"], axis=-1).max() > 30 for row in routes.values()):
                    raise ValueError("original 30 N reaction guard failed")
                if any(np.any(row["forces"]) for row in routes.values()):
                    nonzero[lane] += 1
                composers = {side: hand.permanent_wrench_composer
                             for side, hand in zip(("left", "right"), scene["hands"], strict=True)}
                write_composer_reactions(composers, routes,
                    to_tensor=lambda value: torch.as_tensor(value, dtype=torch.float32, device=sim.device))
                for hand in scene["hands"]:
                    hand.write_data_to_sim()
            step_thumb_physics(sim, sdk - 1, budget_end=end)
            all_links = []
            for scene in scenes:
                links = []
                for hand in scene["hands"]:
                    hand.update(1 / 480)
                    q = hand.root_physx_view.get_dof_positions().detach().cpu().numpy()[0]
                    indices = [hand.joint_names.index(name) for name in ACTIVE]
                    mimic = [q[hand.joint_names.index(f)] - (ratio*q[hand.joint_names.index(m)]+offset)
                             for f, m, ratio, offset in MIMIC]
                    failure = joint_safety_failure(q[indices], np.stack(active_limits(), -1), mimic)
                    if failure:
                        raise ValueError("original joint guard: " + failure)
                    links.append(hand.root_physx_view.get_link_transforms().detach().cpu().numpy()[0])
                all_links.append(links)
            clearance = collision_clearance(np.asarray(all_links)[..., [0,1,2,6,3,4,5]], shapes)[0]
            if float(clearance.min()) < CLEARANCE_M:
                raise ValueError("original table clearance guard failed")
            measured = [_live_contact_geometry(scene, names)
                        for scene, names in zip(scenes, bindings, strict=True)]
            for slot, entry in enumerate(plan["entries"]):
                name = entry["name"]
                if name in bindings[0]:
                    solver.set_mesh_poses(slot,
                        np.asarray([row[0][name][0] for row in measured], dtype=np.float32),
                        np.asarray([row[0][name][1] for row in measured], dtype=np.float32), teleport=False)
            capture = runtime.advance(physics_step=sdk)
            reaction = runtime.consume(physics_step=sdk + 1)
            journal.emit("shared_step", dict(sdk_step=sdk, rod_ticks=runtime.ticks,
                counts=capture["counts"].cpu().numpy(), mesh_wrenches=capture["mesh_wrenches"].cpu().numpy(),
                source_solver_id=id(solver), min_table_clearance_m=float(clearance.min())))
        result = dict(sdk_step=end, environments=len(scenes), shared_solver=True,
            nonzero_reaction_steps=nonzero, ppo_updates=0, teacher_queries=0,
            physics_rows=end - 10, native_qualification=False, teacher_admission=False)
        write_new(output / "shared_result.json", result)
        return result
    finally:
        journal.close()


def main():
    from scripts.run_dh116_work_packages import NATIVE_LOCK, native_file_lock
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "smoke", "backend", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("receipt-sha256", "smoke-sha256", "smoke-ledger-sha256", "backend-sha256"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise ValueError("bounded probe uses physical GPU 3")
    checked_file(args.backend, args.backend_sha256)
    sys.path.insert(0, str(args.backend.parents[1]))
    from ropesim import _ropesim
    if Path(_ropesim.__file__).resolve() != args.backend.resolve():
        raise ValueError("explicit separate candidate backend required")
    archive = load_grasp_archive("pinch", args.receipt, args.receipt_sha256)
    _, reset = load_inputs("pinch", args.receipt, args.receipt_sha256,
        args.smoke, args.smoke_sha256, args.smoke_ledger_sha256)
    with native_file_lock(NATIVE_LOCK):
        if _gpu_memory(3) > 512:
            raise RuntimeError("GPU 3 is already occupied")
        args.output.mkdir(exist_ok=False)
        _run(Path(__file__).resolve().parents[1], archive, 3, 2, args.output,
             coupled=True, max_sdk_step=18, extra_exercise=partial(coupled, reset=reset, backend=_ropesim))


if __name__ == "__main__":
    main()
