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


def command_for_tick(commands, *, physics_step, start_step, env_id):
    """Keep the archived action unchanged for its original eight SDK ticks."""
    index = physics_step - start_step - 1
    if (type(physics_step) is not int or type(start_step) is not int
            or env_id not in (0, 1) or not 0 <= index < 8 * len(commands)):
        raise ValueError("bounded two-lane archived command interval required")
    action = np.asarray(commands[index // 8], dtype=np.float64)
    if action.shape != (24,) or not np.isfinite(action).all() or (np.abs(action) > 1).any():
        raise ValueError("archived finite normalized 24D command required")
    return action.copy() if env_id == 0 else np.zeros(24, dtype=np.float64)


def restore_hand_seed(sim, scenes, seed):
    """Restore the authenticated seed after the eight-step isolation test."""
    import torch
    from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import _targets
    from go2_pvcnn.control.dh116.tabletop_palm_mount import MountView

    before = sim.current_time_step_index
    for scene in scenes:
        for side, hand in enumerate(scene["hands"]):
            q, qd = _targets(seed, side, hand.joint_names)
            pose = np.asarray(seed["reset_root_pose_w"])[side:side+1].copy()
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
    if sim.current_time_step_index != before:
        raise ValueError("hand seed restore advanced SDK physics")


def reset_live_lane(sim, scenes, seed, archive, runtime, bindings, lane, *, start_step):
    """Reset one complete mechanical lane at a completed control boundary."""
    import torch
    from scripts.audit_dh116_batch_rod_exchange import _live_contact_geometry, translated_static_pose

    sdk = sim.current_time_step_index
    if (runtime.physics_step != sdk or (sdk-start_step) % 8
            or runtime._reaction is None or runtime._consumed):
        raise ValueError("completed shared window with pending reactions required")
    solver = runtime.solver
    before_time = float(solver.time)
    before_nodes = torch.as_tensor(solver.positions()).clone()
    before_velocity = torch.as_tensor(solver.velocities()).clone()
    before_mesh = solver.mesh_state_snapshot()
    before_reaction = runtime._reaction["mesh_wrenches"].clone()
    peers = [i for i in range(len(scenes)) if i != lane]

    def hand_state(scene):
        return [[value.detach().cpu().numpy().copy() for value in (
            hand.root_physx_view.get_dof_positions(),
            hand.root_physx_view.get_dof_velocities(),
            hand.root_physx_view.get_dof_position_targets(),
            hand.root_physx_view.get_link_transforms())] for hand in scene["hands"]]

    before_hands = {i: hand_state(scenes[i]) for i in peers}
    restore_hand_seed(sim, [scenes[lane]], seed)
    for hand in scenes[lane]["hands"]:
        hand.permanent_wrench_composer.reset()
        hand.instantaneous_wrench_composer.reset()
        hand.write_data_to_sim()
    measured, _ = _live_contact_geometry(scenes[lane], bindings[lane])
    positions, rotations = [], []
    for slot, entry in enumerate(archive.mesh_plan["entries"]):
        name = entry["name"]
        if name in measured:
            xyz, quat = measured[name]
        else:
            xyz, quat = translated_static_pose(archive.style, entry, np.zeros(3))
            xyz = np.asarray(xyz, dtype=np.float32) + np.asarray(scenes[lane]["origin"], dtype=np.float32)
            np.testing.assert_array_equal(before_mesh["current_positions_world_m"][lane, slot], xyz)
        positions.append(xyz)
        rotations.append(quat)
    positions = np.asarray(positions, dtype=np.float32)[None]
    rotations = np.asarray(rotations, dtype=np.float32)[None]
    runtime.reset_rod_lane(lane, episode_uid=f"native:{lane}:{runtime.resets[lane]+1}")
    solver.reset_env_mesh_poses(np.asarray([lane], dtype=np.int32), positions, rotations)
    after_mesh = solver.mesh_state_snapshot()
    for phase in ("current", "target", "solver"):
        np.testing.assert_array_equal(after_mesh[phase+"_positions_world_m"][lane], positions[0])
        np.testing.assert_allclose(after_mesh[phase+"_quaternions_xyzw"][lane], rotations[0], atol=2e-6, rtol=0)
    for key in ("solver_linear_velocity_world_m_s", "solver_angular_velocity_world_rad_s",
                "motion_delta_world_m", "motion_rotation_world_rad", "pending_teleport"):
        if np.any(after_mesh[key][lane]):
            raise ValueError("selected mesh Reset retained motion")
    for key, value in before_mesh.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(after_mesh[key][peers], value[peers])
    if [after_mesh["pair_counts"][i] for i in peers] != [before_mesh["pair_counts"][i] for i in peers]:
        raise ValueError("Reset changed another native pair count")
    for i in peers:
        for actual, expected in zip(hand_state(scenes[i]), before_hands[i], strict=True):
            for a, b in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(a, b)
    torch.testing.assert_close(torch.as_tensor(solver.positions())[peers], before_nodes[peers], rtol=0, atol=0)
    torch.testing.assert_close(torch.as_tensor(solver.velocities())[peers], before_velocity[peers], rtol=0, atol=0)
    torch.testing.assert_close(runtime._reaction["mesh_wrenches"][peers], before_reaction[peers], rtol=0, atol=0)
    if (runtime._reaction["mesh_wrenches"][lane].count_nonzero()
            or runtime.ticks[lane] != 0 or sim.current_time_step_index != sdk
            or float(solver.time) != before_time):
        raise ValueError("local Reset retained reaction or advanced a clock")
    return dict(sdk_step=sdk, env_id=lane, reset_id=runtime.resets[lane],
        episode_uid=runtime.episodes[lane], rod_ticks=list(runtime.ticks),
        current_target_solver_equal=True, mesh_motion_zero=True,
        neighbor_hand_and_rod_state_bitwise_preserved=True,
        neighbor_mesh_state_bitwise_preserved=True, neighbor_reaction_bitwise_preserved=True,
        solver_time_s=before_time, reset_sdk_steps=0, reset_solver_steps=0)


def coupled(sim, scenes, seed, prepared, archive, nodes, owners, output, gpu, end, *, reset, backend, commands=None, reset_last_window=False):
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
    if commands is not None:
        restore_hand_seed(sim, scenes, seed)
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
    from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import HandBinding, BimanualStepConnection
    from go2_pvcnn.control.dh116.tabletop_palm_mount import MountView
    from go2_pvcnn.control.dh116.tabletop_deformable_live_geometry import LiveGeometryReader
    from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import project_policy_wrist_targets

    start_step = sim.current_time_step_index
    control_index = substep_index = 0
    def lane_clock(lane):
        return dict(run_uuid=runtime.run_uuid, env_id=lane, rod_uid=f"rod:{lane}",
            episode_uid=runtime.episodes[lane], reset_id=runtime.resets[lane],
            physics_step=runtime.physics_step, rod_tick=runtime.ticks[lane],
            policy_step=control_index, substep=substep_index,
            timestamp_s=runtime.physics_step/480, solver_time_s=float(solver.time), clock_valid=True)

    hand_bindings, readers, roots = [], [], []
    for lane, (scene, names) in enumerate(zip(scenes, bindings, strict=True)):
        group = tuple(HandBinding(side, hand, {lane: hand.root_physx_view.prim_paths[0]}, MountView(hand))
            for side, hand in zip(("left", "right"), scene["hands"], strict=True))
        hand_bindings.append(group)
        readers.append(LiveGeometryReader(sim, group, dict(identity=[lane_clock(lane)],
            table_top_m=.30, pad_bindings=names, shapes=shapes), read_clock=lambda lane=lane: lane_clock(lane)))
        root = np.asarray(seed["reset_root_pose_w"]).copy()
        root[:, :3] += scene["origin"]
        roots.append(torch.as_tensor(root[None], dtype=torch.float64))
    palms = [torch.zeros((1, 2, 3), dtype=torch.float64) for _ in scenes]
    angles = [torch.zeros_like(value) for value in palms]
    nonzero = [0] * len(scenes)
    reaction = None
    reset_receipt = None
    journal = NativeJournal(output / "shared_coupling.jsonl")
    try:
        for sdk in range(start_step + 1, end + 1):
            control_index, substep_index = divmod(sdk - start_step - 1, 8)
            transactions = []
            for lane in range(len(scenes)):
                if commands is None:
                    transactions.append(None)
                    continue
                read_clock = lambda lane=lane: [lane_clock(lane)]
                geometry = readers[lane]()
                connection = BimanualStepConnection(env_ids=(lane,), solver_substeps=12)
                action = torch.as_tensor(command_for_tick(commands, physics_step=sdk,
                    start_step=start_step, env_id=lane))[None]
                request = connection.begin(hand_bindings[lane], action, read_clock=read_clock,
                    palms=palms[lane], angles=angles[lane])
                projection = project_policy_wrist_targets(root_seed_w=roots[lane],
                    measured_base_w=geometry["root_poses_w"], common_lift_offset_m=torch.zeros(1, dtype=torch.float64),
                    measured_clearance_m=geometry["clearance_m"],
                    palm_offsets_m=palms[lane], requested_palm_offsets_m=request["requested_palm_offsets_m"],
                    angles_rad=angles[lane], requested_angles_rad=request["requested_angles_rad"],
                    allowed=torch.ones(1, dtype=torch.bool), live=torch.ones(1, dtype=torch.bool))
                transactions.append((connection, projection, read_clock))
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
                transaction = transactions[lane]
                if transaction is not None:
                    connection, projection, read_clock = transaction
                    connection.submit(hand_bindings[lane], read_clock=read_clock,
                        projected_root_poses=projection["projected_root_poses"],
                        projected_angles=projection["projected_angles"],
                        projection_source=projection["projection_source"])
                else:
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
            action_readbacks = []
            for lane, transaction in enumerate(transactions):
                if transaction is not None:
                    connection, projection, read_clock = transaction
                    action_readbacks.append(connection.finish(hand_bindings[lane], read_clock=read_clock))
                    palms[lane], angles[lane] = projection["palm_offsets_m"], projection["projected_angles"]
            journal.emit("shared_step", dict(sdk_step=sdk, rod_ticks=list(runtime.ticks),
                action_readbacks=action_readbacks, counts=capture["counts"].cpu().numpy(), mesh_wrenches=capture["mesh_wrenches"].cpu().numpy(),
                source_solver_id=id(solver), min_table_clearance_m=float(clearance.min())))
            if reset_last_window and sdk == end - 8:
                reset_receipt = reset_live_lane(sim, scenes, seed, archive, runtime, bindings, 0, start_step=start_step)
                palms[0].zero_()
                angles[0].zero_()
                measured = [_live_contact_geometry(scene, names)
                            for scene, names in zip(scenes, bindings, strict=True)]
                journal.emit("local_reset", reset_receipt)
            reaction = runtime.consume(physics_step=sdk + 1)
        if reset_last_window and (reset_receipt is None or runtime.resets != [2, 1]
                                  or runtime.ticks != [96, (end-start_step)*12]):
            raise ValueError("complete local Reset and continued lane clocks required")
        result = dict(sdk_step=end, environments=len(scenes), shared_solver=True,
            nonzero_reaction_steps=nonzero, ppo_updates=0, teacher_queries=0,
            physics_rows=end - start_step, contact_replay=commands is not None,
            local_reset=reset_receipt, per_lane_resets=runtime.resets, per_lane_rod_ticks=runtime.ticks,
            native_qualification=False, teacher_admission=False)
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
    parser.add_argument("--contact-receipt", type=Path)
    parser.add_argument("--contact-receipt-sha256")
    parser.add_argument("--reset-last-window", action="store_true")
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
    if args.reset_last_window and (args.contact_receipt is None or not hasattr(_ropesim.RopeBatch, "reset_env_mesh_poses")):
        raise ValueError("complete local Reset needs the contact replay and selected-reset candidate backend")
    commands = None
    if args.contact_receipt is not None:
        if args.contact_receipt_sha256 is None:
            raise ValueError("original contact receipt SHA required")
        from scripts.smoke_dh116_teacher_batch_dynamic import archived_contact_prefix
        archive.pair, commands = archived_contact_prefix(args.contact_receipt, args.contact_receipt_sha256)
    with native_file_lock(NATIVE_LOCK):
        if _gpu_memory(3) > 512:
            raise RuntimeError("GPU 3 is already occupied")
        args.output.mkdir(exist_ok=False)
        _run(Path(__file__).resolve().parents[1], archive, 3, 2, args.output,
             coupled=True, max_sdk_step=18 if commands is None else 10 + 8 * len(commands),
             extra_exercise=partial(coupled, reset=reset, backend=_ropesim, commands=commands, reset_last_window=args.reset_last_window))


if __name__ == "__main__":
    main()
