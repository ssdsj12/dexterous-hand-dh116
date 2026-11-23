#!/usr/bin/env python3
"""Two-lane native Pinch smoke with a shared SDK clock and an isolated real Reset."""
import argparse
import json
from copy import deepcopy
import os
from functools import partial
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from scripts.audit_dh116_isaac_batch_scene import _gpu_memory, _run, load_grasp_archive


def lane_bindings(rod, scene):
    from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import HandBinding
    from go2_pvcnn.control.dh116.tabletop_palm_mount import MountView

    return tuple(HandBinding(side, hand, {rod.env_id: hand.root_physx_view.prim_paths[0]}, MountView(hand))
                     for side, hand in zip(("left", "right"), scene["hands"], strict=True))


def reset_lane_exchange(sim, rod, scene, seed, archive, prepared, *, end, owner_resets=None, bindings=None):
    import torch
    from go2_pvcnn.control.dh116.tabletop_deformable_clearance_guard import ClearanceGuardedExchange
    from go2_pvcnn.control.dh116.tabletop_deformable_live_geometry import LiveGeometryReader
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_reset as reset_owner
    from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
    from go2_pvcnn.control.dh116.tabletop_long_clock_v2 import _clock_long
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings as scoped_bindings
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_provider import _fresh_reset_owners
    from go2_pvcnn.control.dh116.tabletop_native_teacher_batch import translated_seed
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
    from scripts.audit_dh116_batch_rod_exchange import rebuild_reset_meshes, reset_rod_fingerprint

    if bindings is None:
        bindings = lane_bindings(rod, scene)
    if owner_resets is None:
        owner_resets = _fresh_reset_owners()
    lane_seed = translated_seed(seed, scene["origin"])
    episode_uid = f"{rod.run_uuid}:{rod.env_id}:{rod.clock()['reset_id'] + 1}"
    transaction = reset_owner.DualHandResetTransaction(lane_seed, episode_uid=episode_uid)
    with scoped_bindings((reset_owner, "_clock", partial(_clock_long, max_step=end))):
        receipt = transaction.run(sim, rod, bindings, owner_resets=owner_resets)
    poses = receipt["pad_poses_xyzw"]
    state = reset_rod_fingerprint(rod.rope, str(sim.device))
    rod._needs_reset = True
    snapshot = rebuild_reset_meshes(rod.rope, archive.style, archive.mesh_plan["entries"], poses,
                         scene["origin"], 41 * len(rod.mesh_names) * 12)
    if reset_rod_fingerprint(rod.rope, str(sim.device)) != state:
        raise ValueError("mesh clear/add changed actual same-lane rod state")
    if sim.current_time_step_index != receipt["after_clock"]["physics_step"]:
        raise ValueError("local Reset advanced the shared SDK")
    rod._needs_reset = False
    if rod.clock() != receipt["after_clock"]:
        raise ValueError("local Reset did not retain its committed rod identity")
    shapes = [(i, binding.hand.body_names.index(name), row["vertices"])
              for i, binding in enumerate(bindings)
              for name, row in prepared["hands"][binding.side]["source_colliders"].items()]
    if archive.style != "pinch":
        raise ValueError("the Pinch exchange requires its own eleven-slot Reset source")
    reader = LiveGeometryReader(sim, bindings, dict(identity=[rod.clock()], table_top_m=.30,
        pad_bindings=rod.bindings, shapes=shapes), read_clock=rod.clock)
    if not bool((reader()["clearance_m"] >= CLEARANCE_M).all()):
        raise ValueError("native Reset failed the original table clearance")
    receipt["batch_mesh_rebuild"] = dict(current_target_solver_readback=True, numeric_sha256=state,
        mesh_names=list(rod.mesh_names), source_instance_id=id(rod.rope), mesh_snapshot_sha256=_digest(snapshot))
    exchange = ClearanceGuardedExchange(sim, rod, bindings,
        root_seed_w=torch.as_tensor(lane_seed["reset_root_pose_w"], dtype=torch.float64)[None],
        read_geometry=reader, read_geometry_after_sdk=reader.read_after_sdk_step, physics_budget_end=end)
    return exchange, receipt


def _exercise(sim, scenes, seed, prepared, archive, nodes, owners, output, gpu, end, *, commands):
    import torch
    from go2_pvcnn.control.dh116.tabletop_native_teacher_batch import BatchBudget
    from go2_pvcnn.control.dh116.tabletop_teacher_physical_group import TeacherPhysicalGroup
    from go2_pvcnn.control.dh116.tabletop_native_teacher_collector import NativeJournal
    from scripts.audit_dh116_batch_rod_exchange import _build_rods, reset_rod_fingerprint
    from scripts.diagnose_dh116_teacher_initialization import write_new

    rods = _build_rods(scenes, archive, nodes, owners)
    budget = BatchBudget([0, 1], start_sdk_step=sim.current_time_step_index, max_sdk_step=end)
    last = {}
    hand_responses = {0: 0, 1: 0}
    with_journal = NativeJournal(output / "dynamic_physics.jsonl")
    durations = []
    peak_mib = _gpu_memory(gpu)
    try:
        def reset(env_id, old):
            other = rods[1 - env_id]
            other_before = (other.clock(), reset_rod_fingerprint(other.rope, str(sim.device)))
            other_joints = [hand.root_physx_view.get_dof_positions().detach().clone()
                            for hand in scenes[1 - env_id]["hands"]]
            exchange, receipt = reset_lane_exchange(sim, rods[env_id], scenes[env_id], seed, archive, prepared, end=end)
            after = (other.clock(), reset_rod_fingerprint(other.rope, str(sim.device)))
            if old is not None and (other_before != after or not all(
                    torch.equal(q, hand.root_physx_view.get_dof_positions())
                    for q, hand in zip(other_joints, scenes[1 - env_id]["hands"], strict=True))):
                raise ValueError("local Reset changed its neighboring native lane")
            budget.reset_lane(env_id, rods[env_id].clock())
            last.pop(env_id, None)
            with_journal.emit("lane_reset", dict(env_id=env_id, reset=receipt, budget=budget.snapshot()))
            return exchange

        group = TeacherPhysicalGroup(reset(i, None) for i in range(2))
        # The active lane replays an archived issued command. Its neighbor holds.
        actions = torch.zeros(2, 24, dtype=torch.float64)
        first_timeout_tick = end - 8

        def observe(row):
            with_journal.emit("physical", row)
            rod = row["rod"]
            env_id = rod["env_id"]
            actual = row["action_readback"]["request"]["action24"][0]
            if not torch.equal(actual, actions[env_id]):
                raise ValueError("measured lane command differs from its supplied 24D action")
            previous = last.get(env_id)
            for side, route in row["issued_composer_wrench"].items():
                if previous is None:
                    if row["reaction_source_physics_step"] is not None or np.any(route["forces"]) or np.any(route["torques"]):
                        raise ValueError("fresh Reset retained a stale reaction")
                elif row["reaction_source_physics_step"] != previous["physics_step"] or any(
                        not np.array_equal(route[k], previous["hand_wrenches_com_world"][side][k])
                        for k in ("forces", "torques", "body_ids")):
                    raise ValueError("composer reaction does not belong to the preceding same-lane solver")
            routes = rod["hand_wrenches_com_world"]
            pairs = rod["rod_mesh_contact_pairs"]["pairs"]
            if previous is not None and any(np.any(route["forces"]) for route in row["issued_composer_wrench"].values()):
                hand_responses[env_id] += 1
            if any(np.any(route["forces"]) for route in routes.values()) and not any(
                    pair["mesh_name"] in rod["mesh_bindings"] for pair in pairs):
                raise ValueError("nonzero hand reaction lacks a real hand pair")
            last[env_id] = deepcopy(rod)
            if env_id == 0 and row["physics_step"] == first_timeout_tick:
                if row["substep"] != 7:
                    raise ValueError("diagnostic timeout is outside the committed window boundary")
                return "time_limit"
            return None

        for command in commands:
            budget.require_window()
            actions[0] = torch.as_tensor(command, dtype=actions.dtype)
            started = perf_counter()
            report = group.control(actions, observe=observe, progress=budget,
                commit=lambda window: with_journal.emit("window_committed", dict(report=window, budget=budget.snapshot())))
            durations.append(perf_counter() - started)
            peak_mib = max(peak_mib, _gpu_memory(gpu))
            print("native window", budget.sdk_step, round(durations[-1], 3), "s", peak_mib, "MiB", flush=True)
            if report["aborted"]:
                raise ValueError("native smoke rejected original guard: " + repr(report["terminal_reasons"]))
            if report["terminal_reasons"]:
                group.reset_done(reset)
        if not sum(hand_responses.values()):
            raise ValueError("no nonzero native hand–rod reaction was actually routed")
        if [lane.resets for lane in budget.lanes.values()] != [2, 1]:
            raise ValueError("only the timed-out lane must have reset")
        result = dict(sdk_step=budget.sdk_step, budget=budget.snapshot(), environments=2,
                      window_seconds=durations, peak_observed_gpu_mib=peak_mib,
                      nonzero_routed_hand_steps=hand_responses, ppo_updates=0)
        write_new(output / "dynamic_result.json", result)
        return result
    except BaseException as exc:
        with_journal.emit("smoke_failure", dict(type=type(exc).__name__, error=str(exc),
            budget=budget.snapshot(), window_seconds=durations, nonzero_routed_hand_steps=hand_responses))
        raise
    finally:
        with_journal.close()



def archived_contact_prefix(path, expected_sha):
    """Use only original per-window issued vectors from a successful native trace."""
    from itertools import islice
    from go2_pvcnn.control.dh116.tabletop_dual_pinch_builder import load_bound_pair_proposal
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file

    status = json.loads(checked_file(path, expected_sha))
    if (status["exit_code"] != 0 or status["forced_exit"] is not False
            or status["debugger_inferior_exit_code"] != 0 or status["debugger_inferior_signal"] is not None):
        raise ValueError("original naturally exited native contact trace required")
    directory = path.parent / "native"
    plan = json.loads(checked_file(directory / "frozen_plan.json", status["artifact_sha256"]["frozen_plan.json"]))
    pair = load_bound_pair_proposal(plan["pair_proposal"])
    ledger = directory / "physical.jsonl"
    if ledger.is_symlink() or not ledger.is_file():
        raise ValueError("regular original native contact ledger required")
    with ledger.open("rb") as stream:
        import hashlib
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != status["artifact_sha256"]["physical.jsonl"]:
            raise ValueError("original contact ledger SHA differs")
    commands = []
    contact = False
    with ledger.open() as stream:
        physical = (event["payload"]["exchange"] for event in map(json.loads, stream)
                    if event.get("operation") == "physical")
        for index, row in enumerate(islice(physical, 26 * 8)):
            if row["policy_step"] != index // 8 or row["substep"] != index % 8:
                raise ValueError("archived action sequence is discontinuous")
            action = np.asarray(row["action_readback"]["request"]["action24"]["__dh116_array__"],
                                dtype=np.float64)
            if action.shape != (1, 24) or not np.isfinite(action).all() or (abs(action) > 1.).any():
                raise ValueError("original normalized finite 24D action required")
            if index % 8 == 0:
                commands.append(action[0].copy())
            elif not np.array_equal(commands[-1], action[0]):
                raise ValueError("shared PPO smoke requires a constant command in each window")
            rod = row["rod"]
            contact |= any(pair["mesh_name"] in rod["mesh_bindings"]
                           for pair in rod["rod_mesh_contact_pairs"]["pairs"])
    if len(commands) != 26 or not contact:
        raise ValueError("complete fixed prefix with measured hand contact required")
    # Window after local Reset verifies resumed dynamics and the neighbor's old clock.
    commands.append(np.zeros(24, dtype=np.float64))
    return pair, commands

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-receipt", type=Path, required=True)
    parser.add_argument("--contact-receipt-sha256", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("serial smoke requires the selected physical GPU as its only visible device")
    from go2_pvcnn.control.dh116.tabletop_mesh_observation_source import BACKEND, BACKEND_SHA
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
    from scripts.run_dh116_work_packages import NATIVE_LOCK, native_file_lock
    from scripts.diagnose_dh116_teacher_initialization import write_new

    archive = load_grasp_archive("pinch", args.receipt, args.receipt_sha256)
    checked_file(BACKEND, BACKEND_SHA)
    project = Path(__file__).resolve().parents[1]
    archive.pair, commands = archived_contact_prefix(args.contact_receipt, args.contact_receipt_sha256)
    max_sdk_steps = 2 + 8 * len(commands)
    with native_file_lock(NATIVE_LOCK):
        if _gpu_memory(args.gpu) > 512:
            raise RuntimeError("physical GPU is already occupied before serial smoke")
        args.output.mkdir(exist_ok=False)
        write_new(args.output / "probe_plan.json", dict(environments=2, style="pinch", gpu=args.gpu,
            source_reset_sha256=args.receipt_sha256, contact_replay_sha256=args.contact_receipt_sha256,
            max_sdk_steps=max_sdk_steps, ppo_updates=0))
        sys.path.insert(0, str(BACKEND.parents[1]))
        from ropesim import _ropesim
        if Path(_ropesim.__file__).resolve() != BACKEND:
            raise ValueError("native smoke loaded a different DeformX binary")
        _run(project, archive, args.gpu, 2, args.output, coupled=True, max_sdk_step=max_sdk_steps,
             extra_exercise=partial(_exercise, commands=commands))


if __name__ == "__main__":
    main()
