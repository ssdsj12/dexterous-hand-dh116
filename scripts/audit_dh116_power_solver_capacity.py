#!/usr/bin/env python3
"""Size 32 independent DeformX owners with the archived real Power mesh slots."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import material_mapping
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from go2_pvcnn.control.dh116.tabletop_mesh_observation_source import BACKEND, BACKEND_SHA
from go2_pvcnn.control.dh116.tabletop_power_shared_rod import PROXIMAL_POWER_MESH_NAMES


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--num-envs", type=int, choices=(2, 32), required=True)
    args = parser.parse_args()

    import os
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
    owners = []
    for env_id in range(args.num_envs):
        seed = nodes.copy()
        if env_id == 0:
            # Keep the isolated lane outside the chamber. Raising it alone
            # can overlap the real palm geometry and legitimately create pairs.
            seed[..., 0] += 2.
        rope = RopeBatch.from_positions(seed, **material_mapping(config)["constructor"],
            pinned=np.zeros(41, dtype=bool), fix_root_orientation=False,
            straight_rest=True, deterministic=True)
        rope.set_gravity(np.asarray(rod["gravity_world_m_s2"], dtype=np.float32))
        rope.set_contact(radius=rod["radius_m"], mu_s=rod["mu_s"], mu_k=rod["mu_k"],
                         v_slip=rod["v_slip_m_s"], kappa=rod["kappa"])
        rope.set_rope_contact(rod["self_collision"])
        for key in ("damping", "drag", "drag_quad", "drag_quad_t"):
            setattr(rope, key, rod[key])
        for slot, row in enumerate(plan["entries"]):
            xyz, quaternion = row["static_pose"] if row["static_pose"] is not None else poses[row["name"]]
            actual = rope.add_mesh(row["values"], row["origin"], row["spacing"], pos=xyz,
                                   quat=quaternion, mu_s=row["mu_s"], mu_k=row["mu_k"])
            if actual != slot:
                raise ValueError("native mesh slot order changed")
        rope.enable_contact_forces(True)
        rope.enable_contact_pair_capture(capacity)
        owners.append(rope)
    torch.cuda.synchronize()
    allocated_memory = gpu_memory()
    allocations_s = time.perf_counter()-start

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
    print(json.dumps(dict(num_solver_owners=len(owners), physics_equivalent_steps=2,
        nodes_per_env=41, meshes_per_env=count, pairs_allocated_per_env=capacity,
        native_pair_counts_per_step=snapshots, backend_sha256=BACKEND_SHA,
        allocation_seconds=allocations_s, step_and_pair_readback_seconds=step_and_readback_s,
        gpu_usage_mib_before=before_memory, gpu_usage_mib_at_allocation=allocated_memory,
        gpu_usage_mib_end=gpu_memory(), isolated_reset_verified=True,
        isaac_envs=0, ppo_updates=0, native_teacher_admission=False)))


if __name__ == "__main__":
    main()
