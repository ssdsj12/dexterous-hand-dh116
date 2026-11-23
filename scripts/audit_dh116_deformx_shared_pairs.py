#!/usr/bin/env python3
"""Compare isolated and shared DeformX contact buckets on a separate CUDA build."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import material_mapping
from go2_pvcnn.control.dh116.tabletop_deformx_pair_capture import validate_pair_capture
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from scripts.audit_dh116_isaac_batch_scene import load_grasp_archive
from scripts.audit_dh116_batch_rod_exchange import translated_static_pose


DT = 1 / 480
SUBSTEPS = 12
FIELDS = ("indices", "vectors", "mesh_positions_world_m", "mesh_quaternions_xyzw",
          "surface_gap_m", "normal_force_n")
FORCES = "node_forces_on_rod_world_n"
WRENCHES = "mesh_wrenches_on_mesh_world"


def load_inputs(style, receipt, receipt_sha, smoke, smoke_sha, ledger_sha):
    archive = load_grasp_archive(style, receipt, receipt_sha)
    if style == "pinch":
        from scripts.train_dh116_native_teacher_pinch_batch import _check_smoke

        if None in (smoke, smoke_sha, ledger_sha):
            raise ValueError("the accepted Pinch dynamic receipt supplies its measured lane-0 Reset")
        _check_smoke(smoke, smoke_sha)
        receipt_data = json.loads(checked_file(smoke, smoke_sha))
        plan = json.loads(checked_file(smoke.parent / "native" / "probe_plan.json",
                                       receipt_data["artifacts"]["probe_plan.json"]))
        if plan["source_reset_sha256"] != receipt_sha or plan["style"] != "pinch":
            raise ValueError("Pinch smoke and mesh archive provenance differ")
        raw = checked_file(smoke.parent / "native" / "dynamic_physics.jsonl", ledger_sha)
        first = json.loads(raw.split(b"\n", 1)[0])
        reset = decode_owned(first["payload"]["reset"])
        if (first["operation"] != "lane_reset" or first["payload"]["env_id"] != 0
                or reset["after_clock"]["physics_step"] != 2):
            raise ValueError("same measured original Pinch recipe and fresh Reset required")
    else:
        parent = json.loads(checked_file(receipt, receipt_sha))
        event = checked_file(receipt.parent / "native" / "episode_reset.json",
                             parent["artifact_sha256"]["episode_reset.json"])
        reset = decode_owned(json.loads(event))["reset"]
    if reset["status"] != "complete" or reset["after_clock"]["rod_tick"] != 0:
        raise ValueError("original fresh native Reset required")
    return archive.mesh_plan, reset


def make_solver(backend, plan, reset, offsets, *, style, capacity, empty_lane=True):
    nodes = np.asarray(reset["rod_reset"]["node_positions"], dtype=np.float32)
    initial = nodes + offsets[:, None, :]
    if empty_lane:
        initial[0, :, 0] += 2.
    cfg = plan["cfg"]
    rod = cfg["rod"]
    solver = backend.RopeBatch.from_positions(initial, **material_mapping(cfg)["constructor"],
        pinned=np.zeros(initial.shape[:-1], dtype=bool), fix_root_orientation=False,
        straight_rest=True, deterministic=True)
    solver.set_gravity(np.asarray(rod["gravity_world_m_s2"], dtype=np.float32))
    solver.set_contact(radius=rod["radius_m"], mu_s=rod["mu_s"], mu_k=rod["mu_k"],
                       v_slip=rod["v_slip_m_s"], kappa=rod["kappa"])
    solver.set_rope_contact(rod["self_collision"])
    for key in ("damping", "drag", "drag_quad", "drag_quad_t"):
        setattr(solver, key, rod[key])
    for slot, entry in enumerate(plan["entries"]):
        if entry["name"] in reset["pad_poses_xyzw"]:
            position, quaternion = reset["pad_poses_xyzw"][entry["name"]]
        else:
            position, quaternion = translated_static_pose(style, entry, np.zeros(3))
        xyz = np.asarray(position, dtype=np.float32) + offsets
        rotation = np.broadcast_to(quaternion, (len(offsets), 4)).astype(np.float32)
        actual = solver.add_mesh(entry["values"], entry["origin"], entry["spacing"], pos=xyz,
                                 quat=rotation, mu_s=entry["mu_s"], mu_k=entry["mu_k"])
        if actual != slot:
            raise ValueError("actual native mesh slot changed")
    solver.enable_contact_forces(True)
    if capacity is not None:
        if len(offsets) == 1:
            solver.enable_contact_pair_capture(capacity)
        else:
            solver.enable_batched_contact_pair_capture(capacity)
    return solver


def audit(backend, plan, reset, num_envs, *, style):
    lanes = np.arange(num_envs)
    from math import ceil, sqrt
    columns = ceil(sqrt(num_envs))
    table_size = plan.get("original_plan", plan)["prepared_scene"]["table"]["size_m"]
    spacing = max(table_size[:2]) + 1.
    offsets = np.stack((spacing * (lanes % columns), spacing * (lanes // columns),
                       np.zeros(num_envs)), axis=1).astype(np.float32)
    capacity = 41 * len(plan["entries"]) * SUBSTEPS
    batch = make_solver(backend, plan, reset, offsets, style=style, capacity=capacity)
    def position(solver):
        return torch.as_tensor(solver.positions()).detach().cpu().numpy().copy()

    aggregate_reference = make_solver(backend, plan, reset, offsets, style=style, capacity=None)
    reference_0 = make_solver(backend, plan, reset, offsets[:1], style=style, capacity=capacity)
    reference_1 = make_solver(backend, plan, reset, offsets[1:2], style=style, capacity=capacity, empty_lane=False)
    initial = position(batch)
    if not all(np.array_equal(initial[i], position(solver)[0])
               for i, solver in enumerate((reference_0, reference_1))):
        raise ValueError("identical independent constructor poses required")
    raw_times, counts = [], []
    errors = {}
    observer_error = 0.
    for _ in range(2):
        now = perf_counter()
        batch.step_free(DT, SUBSTEPS)
        raw = batch.contact_pair_samples_batched_v1()
        raw_times.append(perf_counter() - now)
        if (raw["schema"] != "dh116.sdf_pair_samples_batch.v1" or batch.last_error
                or len(raw["lanes"]) != num_envs
                or raw[FORCES].shape != (num_envs, 41, 3)
                or raw[WRENCHES].shape != (num_envs, len(plan["entries"]), 6)):
            raise ValueError("complete native per-environment contact buckets required")
        for env_id, lane in enumerate(raw["lanes"]):
            if (lane["env_id"] != env_id or lane["capacity"] != capacity
                    or lane["overflow"] or lane["last_error"] or lane["substeps"] != SUBSTEPS
                    or (lane["attempted"] == 0) != (env_id == 0)):
                raise ValueError("mixed or incomplete native pair bucket")
            validate_pair_capture(lane, dt_s=DT, node_forces=raw[FORCES][env_id],
                                  mesh_wrenches=raw[WRENCHES][env_id])
        aggregate_reference.step_free(DT, SUBSTEPS)
        for observed, reference in ((position(batch), position(aggregate_reference)),
                                     (raw[FORCES], aggregate_reference.contact_forces()),
                                     (raw[WRENCHES], aggregate_reference.mesh_wrenches())):
            observer_error = max(observer_error, float(np.abs(observed - reference).max(initial=0.)))
            np.testing.assert_allclose(observed, reference, atol=1e-5, rtol=0.)
        for env_id, solver in enumerate((reference_0, reference_1)):
            solver.step_free(DT, SUBSTEPS)
            packet = solver.contact_pair_samples_v2()
            candidate = raw["lanes"][env_id]
            a = np.lexsort(packet["indices"].T[::-1])
            b = np.lexsort(candidate["indices"].T[::-1])
            np.testing.assert_array_equal(packet["indices"][a], candidate["indices"][b])
            def compare(key, actual, expected):
                delta = float(np.abs(actual - expected).max(initial=0.))
                errors[key] = max(errors.get(key, 0.), delta)

            for key in FIELDS[1:]:
                compare(key, candidate[key][b], packet[key][a])
            for key in (FORCES, WRENCHES):
                compare(key, raw[key][env_id], packet[key])
        counts.append(int(sum(lane["attempted"] for lane in raw["lanes"])))

    before = position(batch)
    batch.reset_envs(np.array([0], dtype=np.int32))
    after = position(batch)
    if (np.array_equal(before[0], initial[0]) or not np.array_equal(after[0], initial[0])
            or not np.array_equal(after[1:], before[1:])):
        raise ValueError("shared solver Reset mutated the wrong world state")
    try:
        batch.contact_pair_samples_batched_v1()
    except ValueError as error:
        if "no completed pair capture" not in str(error):
            raise
    else:
        raise ValueError("pre-Reset contact evidence remained readable")
    batch.step_free(DT, SUBSTEPS)
    continued = batch.contact_pair_samples_batched_v1()
    if batch.last_error or continued["lanes"][0]["attempted"] or any(
            lane["overflow"] or not lane["attempted"] for lane in continued["lanes"][1:]):
        raise ValueError("native pair ownership after Reset differs")
    overflow = make_solver(backend, plan, reset, offsets[:2], style=style, capacity=1)
    overflow.step_free(DT, SUBSTEPS)
    witness = overflow.contact_pair_samples_batched_v1()["lanes"]
    if witness[0]["overflow"] or not witness[1]["overflow"] or witness[1]["capacity"] != 1:
        raise ValueError("overflow in a contacting lane must not affect its neighbor")
    max_error = max(errors.values())
    return dict(environments=num_envs, native_calls=3, isolated_overflow_probe_steps=1,
                per_field_independent_error=errors, reset_lane=0, continued_post_reset=True, counts=counts,
                step_and_pair_seconds=raw_times, maximum_shared_vs_independent_error=max_error,
                independent_parity_at_1e_5=max_error < 1e-5, maximum_capture_observer_error=observer_error,
                raw_pair_capacity_per_env=capacity, free_device_mib=torch.cuda.mem_get_info()[0] >> 20,
                isaac_envs=0, ppo_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp", required=True, choices=("pinch", "tripod", "power"))
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--smoke-sha256")
    parser.add_argument("--smoke-ledger-sha256")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--backend-sha256", required=True)
    parser.add_argument("--num-envs", type=int, choices=(2, 128, 512), required=True)
    args = parser.parse_args()
    if not os.getenv("CUDA_VISIBLE_DEVICES", "").isdigit():
        parser.error("select one physical GPU using CUDA_VISIBLE_DEVICES")
    from ropesim import _ropesim

    if Path(_ropesim.__file__).resolve() != args.backend.resolve():
        raise ValueError("separate compiled backend required")
    checked_file(args.backend, args.backend_sha256)
    plan, reset = load_inputs(args.grasp, args.receipt, args.receipt_sha256, args.smoke,
                              args.smoke_sha256, args.smoke_ledger_sha256)
    result = audit(_ropesim, plan, reset, args.num_envs, style=args.grasp)
    print(json.dumps(dict(grasp=args.grasp, backend_sha256=args.backend_sha256, **result)), flush=True)


if __name__ == "__main__":
    main()
