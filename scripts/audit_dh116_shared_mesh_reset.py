#!/usr/bin/env python3
"""Audit selected zero-time CUDA collider Reset, without Isaac or PPO."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.audit_dh116_deformx_shared_pairs import DT, SUBSTEPS, load_inputs, make_solver
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from go2_pvcnn.control.dh116.tabletop_shared_rod_owner import SharedRodOwner


def audit(backend, plan, reset, count, *, style):
    if not hasattr(backend.RopeBatch, "reset_env_mesh_poses"):
        raise ValueError("selected zero-time mesh Reset API is missing")
    offsets = np.zeros((count, 3), dtype=np.float32)
    offsets[:, 0] = np.arange(count) * 3.
    solver = make_solver(backend, plan, reset, offsets, style=style,
        capacity=41*len(plan["entries"])*SUBSTEPS, empty_lane=False)
    names = tuple(row["name"] for row in plan["entries"])
    owner = SharedRodOwner(solver, run_uuid="mesh-reset-audit", physics_step=2,
        episodes=tuple(f"mesh-reset:{i}:1" for i in range(count)), mesh_names=names,
        solver_substeps=SUBSTEPS, device="cuda")
    captured = owner.advance(physics_step=3)
    before = solver.mesh_state_snapshot()
    positions = torch.as_tensor(solver.positions()).clone()
    velocities = torch.as_tensor(solver.velocities()).clone()
    forces = np.asarray(solver.contact_forces()).copy()
    wrenches = np.asarray(solver.mesh_wrenches()).copy()
    if not np.any(wrenches[0]) or not np.any(wrenches[1:]):
        raise ValueError("both selected and unselected real contact needed")
    target = before["current_positions_world_m"][:1].copy()
    rotation = before["current_quaternions_xyzw"][:1].copy()
    target[0, 0, 0] += .001
    invalid = [
        (np.array([0, 0], dtype=np.int32), np.repeat(target, 2, 0), np.repeat(rotation, 2, 0)),
        (np.array([count], dtype=np.int32), target, rotation),
        (np.array([0.], dtype=np.float64), target, rotation),
        (np.array([0], dtype=np.int32), target, rotation*2),
        (np.array([0], dtype=np.int32), target[:, :-1], rotation[:, :-1]),
    ]
    for ids, xyz, quat in invalid:
        try:
            solver.reset_env_mesh_poses(ids, xyz, quat)
        except (ValueError, TypeError):
            pass
        else:
            raise ValueError("invalid selected mesh Reset was accepted")
        observed = solver.mesh_state_snapshot()
        for key, value in before.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(observed[key], value)
        if float(solver.time) != owner._time:
            raise ValueError("invalid mesh Reset advanced time")
    owner.reset_rod_lane(0, episode_uid="mesh-reset:0:2")
    solver.reset_env_mesh_poses(np.array([0], dtype=np.int32), target, rotation)
    after = solver.mesh_state_snapshot()
    if after["schema"] != "dh116.mesh_state_snapshot.v2" or after["pair_ready"]:
        raise ValueError("explicit CUDA pose/velocity snapshot and invalidated pairs required")
    for phase in ("current", "target", "solver"):
        np.testing.assert_array_equal(after[phase+"_positions_world_m"][0], target[0])
        np.testing.assert_allclose(after[phase+"_quaternions_xyzw"][0], rotation[0], atol=2e-6, rtol=0)
    for key in ("solver_linear_velocity_world_m_s", "solver_angular_velocity_world_rad_s",
                "motion_delta_world_m", "motion_rotation_world_rad", "pending_teleport"):
        if np.any(after[key][0]):
            raise ValueError("selected mesh Reset retained motion")
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(after[key][1:], value[1:])
    if after["pair_counts"][0] != 0 or after["pair_counts"][1:] != before["pair_counts"][1:]:
        raise ValueError("selected Reset changed another pair bucket")
    torch.testing.assert_close(torch.as_tensor(solver.positions())[1:], positions[1:], rtol=0, atol=0)
    torch.testing.assert_close(torch.as_tensor(solver.velocities())[1:], velocities[1:], rtol=0, atol=0)
    if torch.as_tensor(solver.velocities())[0].count_nonzero():
        raise ValueError("selected rod Reset retained velocity")
    for actual, original in ((np.asarray(solver.contact_forces()), forces),
                             (np.asarray(solver.mesh_wrenches()), wrenches)):
        if np.any(actual[0]):
            raise ValueError("selected Reset retained force output")
        np.testing.assert_array_equal(actual[1:], original[1:])
    reaction = owner.consume(physics_step=4)
    if reaction["rod_ticks"] != (0, *((SUBSTEPS,)*(count-1))) or reaction["mesh_wrenches"][0].count_nonzero():
        raise ValueError("selected pending reaction/clock was retained")
    torch.testing.assert_close(reaction["mesh_wrenches"][1:], captured["mesh_wrenches"][1:], rtol=0, atol=0)
    if float(solver.time) != owner._time:
        raise ValueError("zero-time mesh Reset changed solver time")
    owner.advance(physics_step=4)
    owner.consume(physics_step=5)
    # A future mesh target is a separate state and must not be silently erased.
    snapshot = solver.mesh_state_snapshot()
    solver.set_mesh_poses(0, snapshot["current_positions_world_m"][:, 0],
                         snapshot["current_quaternions_xyzw"][:, 0], teleport=False)
    try:
        solver.reset_env_mesh_poses(np.array([0], dtype=np.int32), target, rotation)
    except (ValueError, RuntimeError):
        pass
    else:
        raise ValueError("pending mesh target was silently discarded")
    solver.step_free(DT, SUBSTEPS)
    if solver.last_error:
        raise ValueError("post-Reset continuation failed")
    return dict(grasp=style, environments=count, solver_calls=3, selected_reset_env=0,
        native_mesh_reset_calls=1, zero_time_reset=True, continued=True,
        neighbor_state_bitwise_preserved=True, neighbor_pending_reaction_bitwise_preserved=True,
        selected_force_cleared=True, invalid_requests_rejected=len(invalid),
        binary_snapshot_schema=after["schema"], isaac_envs=0, ppo_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp", choices=("pinch", "tripod", "power"), required=True)
    parser.add_argument("--num-envs", type=int, choices=(2, 128, 512), required=True)
    for name in ("receipt", "smoke", "backend"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("receipt-sha256", "smoke-sha256", "smoke-ledger-sha256", "backend-sha256"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    if not os.getenv("CUDA_VISIBLE_DEVICES", "").isdigit():
        raise ValueError("select exactly one physical GPU")
    checked_file(args.backend, args.backend_sha256)
    from ropesim import _ropesim
    if Path(_ropesim.__file__).resolve() != args.backend.resolve():
        raise ValueError("explicit candidate binary required")
    plan, reset = load_inputs(args.grasp, args.receipt, args.receipt_sha256,
        args.smoke, args.smoke_sha256, args.smoke_ledger_sha256)
    print(json.dumps(audit(_ropesim, plan, reset, args.num_envs, style=args.grasp)), flush=True)


if __name__ == "__main__":
    main()
