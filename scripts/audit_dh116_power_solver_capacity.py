#!/usr/bin/env python3
"""Measure independent and shared-SDF Power capacity on the locked DeformX build."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import material_mapping
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from go2_pvcnn.control.dh116.tabletop_mesh_observation_source import BACKEND, BACKEND_SHA
from go2_pvcnn.control.dh116.tabletop_power_shared_rod import PROXIMAL_POWER_MESH_NAMES


MIB = 1024 * 1024
GPU_RESERVE_MIB = 2048


def load_power_inputs(receipt, receipt_sha256):
    outer = json.loads(checked_file(receipt, receipt_sha256))
    if (outer["exit_code"] != 0 or outer["debugger_inferior_exit_code"] != 0
            or outer["debugger_inferior_signal"] is not None or outer["forced_exit"]):
        raise ValueError("natural native Power Reset receipt required")
    root = receipt.parent / "native"
    inventory = outer["artifact_sha256"]
    acquisition = json.loads(checked_file(root/"reset_capture.json", inventory["reset_capture.json"]))
    if acquisition["style"] != "power" or acquisition["sdk_step"] != 2:
        raise ValueError("archived same-clock Power Reset required")
    meshes = json.loads(checked_file(root/"mesh_reset_state.json", inventory["mesh_reset_state.json"]))
    mesh_plan = decode_owned(meshes["mesh_plan"])
    state = decode_owned(json.loads(checked_file(root/"episode_reset.json", inventory["episode_reset.json"])))["reset"]
    if (mesh_plan["sha256"] != acquisition["mesh_plan_sha256"]
            or tuple(row["name"] for row in mesh_plan["entries"]) != PROXIMAL_POWER_MESH_NAMES
            or not state["after_clock"]["clock_valid"] or state["after_clock"]["rod_tick"] != 0):
        raise ValueError("original 17-slot Power layout and fresh state required")
    return mesh_plan, state


def allocate_owners(requested, make_owner, free_mib):
    """Stop before creating another owner once the GPU reserve is reached."""
    owners = []
    available = free_mib()
    while len(owners) < requested and available > GPU_RESERVE_MIB:
        owners.append(make_owner(len(owners)))
        available = free_mib()
    return owners, available



def probe_shared_sdf(num_envs, nodes, make_rope, copy_nodes, capacity, mesh_count,
                     allocation_start, before_memory, gpu_memory, free_mib, torch):
    """Check the locked batch layout using native net node forces and mesh wrenches."""
    if free_mib() <= GPU_RESERVE_MIB:
        raise RuntimeError("insufficient GPU memory to start the shared-SDF probe")
    ids = np.arange(num_envs)
    offsets = np.zeros((num_envs, 3), dtype=np.float32)
    offsets[:, 0] = 4 * (ids % 32 - 16)
    offsets[:, 1] = 4 * (ids // 32 - 16)
    seed = nodes + offsets[:, None]
    seed[0, :, 0] += 2.
    rope = make_rope(seed, offsets)

    # Keep the rejected capability in the evidence. Raw pair support for this
    # layout would require its own validation before claiming equivalence.
    try:
        rope.enable_contact_pair_capture(capacity)
    except ValueError as error:
        if "pair capture requires one rope" not in str(error):
            raise
        raw_pair_rejection = str(error)
    else:
        raise RuntimeError("unexpected batch pair capture support; audit its schema")

    torch.cuda.synchronize()
    allocated_memory = gpu_memory()
    free_at_allocation = free_mib()
    if free_at_allocation <= GPU_RESERVE_MIB:
        raise RuntimeError("insufficient GPU memory to step the shared-SDF probe")
    allocation_seconds = time.perf_counter() - allocation_start
    original = copy_nodes(rope)
    start = time.perf_counter()
    ticks = []
    for _ in range(2):
        rope.step_free(1 / 480, 12)
        torch.cuda.synchronize()
        forces = rope.contact_forces()
        wrenches = rope.mesh_wrenches()
        if (rope.last_error or forces.shape != (num_envs, nodes.shape[-2], 3)
                or wrenches.shape != (num_envs, mesh_count, 6)
                or not np.isfinite(forces).all() or not np.isfinite(wrenches).all()):
            raise ValueError("finite per-environment force and wrench readback required")
        contact = np.any(forces != 0, axis=(1, 2))
        reaction = np.any(wrenches != 0, axis=(1, 2))
        if contact[0] or reaction[0] or not contact[1:].all() or not reaction[1:].all():
            raise ValueError("isolated no-contact lane or contact/reaction witnesses were lost")
        ticks.append(dict(lanes_with_contact=int(contact.sum()),
                          lanes_with_reaction=int(reaction.sum())))
    step_and_readback = time.perf_counter() - start

    before_reset = copy_nodes(rope)
    rope.reset_envs(np.array([0], dtype=np.int32))
    torch.cuda.synchronize()
    after_reset = copy_nodes(rope)
    if (np.array_equal(before_reset[0], original[0])
            or not np.array_equal(after_reset[0], original[0])
            or not np.array_equal(before_reset[1:], after_reset[1:])):
        raise ValueError("shared-SDF lane Reset must restore only the selected rod")

    print(json.dumps(dict(status="passed_aggregate_only", requested_solver_lanes=num_envs,
        num_solver_lanes=num_envs, num_solver_owners=1, physics_equivalent_steps=2,
        nodes_per_env=nodes.shape[-2], meshes_per_env=mesh_count, shared_sdf_meshes=mesh_count,
        backend_sha256=BACKEND_SHA, allocation_seconds=allocation_seconds,
        step_and_aggregate_readback_seconds=step_and_readback, tick_summary=ticks,
        gpu_usage_mib_before=before_memory, gpu_usage_mib_at_allocation=allocated_memory,
        gpu_usage_mib_end=gpu_memory(), gpu_free_mib_at_allocation=free_at_allocation,
        gpu_reserve_mib=GPU_RESERVE_MIB, isolated_reset_verified=True,
        raw_pair_capture=False, raw_pair_rejection=raw_pair_rejection,
        isaac_envs=0, ppo_updates=0, native_teacher_admission=False)))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--num-envs", type=int, metavar="2..1024", required=True)
    parser.add_argument("--shared-sdf", action="store_true",
                        help="measure aggregate forces; this layout has no raw pair support")
    args = parser.parse_args()

    if not 2 <= args.num_envs <= 1024:
        parser.error("--num-envs must be between 2 and 1024")

    if os.getenv("CUDA_VISIBLE_DEVICES") != str(args.gpu_index):
        raise ValueError("explicit matching CUDA_VISIBLE_DEVICES required")
    import torch
    from ropesim import RopeBatch, _ropesim

    if Path(_ropesim.__file__).resolve() != BACKEND.resolve():
        raise ValueError("actual locked Power backend required")
    checked_file(BACKEND, BACKEND_SHA)
    plan, reset = load_power_inputs(args.receipt, args.receipt_sha256)
    config = plan["cfg"]
    rod = config["rod"]
    nodes = reset["rod_reset"]["node_positions"]
    poses = reset["pad_poses_xyzw"]
    count, capacity = len(plan["entries"]), 41*17*12

    def gpu_memory():
        return int(subprocess.check_output(
            ["nvidia-smi", "-i", str(args.gpu_index), "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True).strip())

    def copy_nodes(solver):
        return torch.as_tensor(solver.positions()).detach().clone().cpu().numpy().copy()

    before_memory = gpu_memory()
    start = time.perf_counter()

    def make_rope(seed, offsets=None):
        rope = RopeBatch.from_positions(seed, **material_mapping(config)["constructor"],
            pinned=np.zeros(seed.shape[:-1], dtype=bool), fix_root_orientation=False,
            straight_rest=True, deterministic=True)
        rope.set_gravity(np.asarray(rod["gravity_world_m_s2"], dtype=np.float32))
        rope.set_contact(radius=rod["radius_m"], mu_s=rod["mu_s"], mu_k=rod["mu_k"],
                         v_slip=rod["v_slip_m_s"], kappa=rod["kappa"])
        rope.set_rope_contact(rod["self_collision"])
        for key in ("damping", "drag", "drag_quad", "drag_quad_t"):
            setattr(rope, key, rod[key])
        for slot, row in enumerate(plan["entries"]):
            xyz, quaternion = row["static_pose"] if row["static_pose"] is not None else poses[row["name"]]
            if offsets is not None:
                xyz = np.asarray(xyz, dtype=np.float32) + offsets
                quaternion = np.broadcast_to(quaternion, (len(offsets), 4)).astype(np.float32)
            actual = rope.add_mesh(row["values"], row["origin"], row["spacing"], pos=xyz,
                                   quat=quaternion, mu_s=row["mu_s"], mu_k=row["mu_k"])
            if actual != slot:
                raise ValueError("native mesh slot order changed")
        rope.enable_contact_forces(True)
        return rope

    def make_owner(env_id):
        seed = nodes.copy()
        if env_id == 0:
            # Move the isolated lane away from all of its meshes.
            seed[..., 0] += 2.
        rope = make_rope(seed)
        rope.enable_contact_pair_capture(capacity)
        return rope

    def free_mib():
        free_bytes, _ = torch.cuda.mem_get_info()
        return free_bytes // MIB

    if args.shared_sdf:
        return probe_shared_sdf(args.num_envs, nodes, make_rope, copy_nodes,
                                capacity, count, start, before_memory, gpu_memory, free_mib, torch)

    owners, free_at_allocation = allocate_owners(args.num_envs, make_owner, free_mib)
    torch.cuda.synchronize()
    allocated_memory = gpu_memory()
    allocations_s = time.perf_counter()-start

    if len(owners) < args.num_envs or free_at_allocation <= GPU_RESERVE_MIB:
        print(json.dumps(dict(status="gpu_memory_reserve_reached",
            requested_solver_owners=args.num_envs, num_solver_owners=len(owners),
            physics_equivalent_steps=0, backend_sha256=BACKEND_SHA,
            allocation_seconds=allocations_s, gpu_usage_mib_before=before_memory,
            gpu_usage_mib_at_allocation=allocated_memory, gpu_free_mib_at_allocation=free_at_allocation,
            gpu_reserve_mib=GPU_RESERVE_MIB, isolated_reset_verified=False,
            isaac_envs=0, ppo_updates=0, native_teacher_admission=False)), flush=True)
        return 2

    original = [copy_nodes(rope) for rope in owners]
    snapshots = []
    start = time.perf_counter()
    for _ in range(2):
        for rope in owners:
            rope.step_free(1/480, 12)
        torch.cuda.synchronize()
        rows = [rope.contact_pair_samples_v2() for rope in owners]
        if any(row["mesh_count"] != count or row["capacity"] != capacity or row["overflow"]
               or row["substeps"] != 12 for row in rows) or any(rope.last_error for rope in owners):
            raise ValueError("complete bounded pair rows required in every lane")
        snapshots.append([int(row["attempted"]) for row in rows])
    step_and_readback_s = time.perf_counter()-start
    before_reset = [copy_nodes(rope) for rope in owners]
    owners[0].reset_envs(np.array([0], dtype=np.int32))
    torch.cuda.synchronize()
    after_reset = [copy_nodes(rope) for rope in owners]
    if (not np.array_equal(after_reset[0], original[0])
            or any(not np.array_equal(before, after)
                   for before, after in zip(before_reset[1:], after_reset[1:], strict=True))
            or any(counts[0] != 0 or any(n <= 0 for n in counts[1:]) for counts in snapshots)):
        raise ValueError("distinct probe lane contact or reset isolation was lost")
    print(json.dumps(dict(status="passed", requested_solver_owners=args.num_envs,
        num_solver_owners=len(owners), physics_equivalent_steps=2,
        nodes_per_env=41, meshes_per_env=count, pairs_allocated_per_env=capacity,
        native_pair_counts_per_step=snapshots, backend_sha256=BACKEND_SHA,
        allocation_seconds=allocations_s, step_and_pair_readback_seconds=step_and_readback_s,
        gpu_usage_mib_before=before_memory, gpu_usage_mib_at_allocation=allocated_memory,
        gpu_usage_mib_end=gpu_memory(), gpu_free_mib_at_allocation=free_at_allocation,
        gpu_reserve_mib=GPU_RESERVE_MIB, isolated_reset_verified=True,
        isaac_envs=0, ppo_updates=0, native_teacher_admission=False)))


if __name__ == "__main__":
    raise SystemExit(main())
