#!/usr/bin/env python3
"""Bounded N=2, two-regex Articulation API probe; no rod/reward/PPO parity.

Default CLI supervises one native child and admits its evidence only after a
natural zero exit and the existing lifecycle's verified normal App.close().
--preflight-only writes the exact plan without AppLauncher or GPU use.

Required audit._run hook (old defaults must remain unchanged):
  author_and_exercise=None, owned_names=None
The callback replaces BOTH old _author and _exercise, not merely _exercise.
Its signature matches author_and_exercise below. OWNED_NAMES replaces the old
per-environment ownership inventory. No old per-environment assets are made.
"""
from copy import deepcopy
import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

SIDES = ("left", "right")
OWNED_NAMES = tuple(f"regex_{side}_{kind}" for side in SIDES
                    for kind in ("hand", "view", "permanent", "instantaneous"))
MAX_SDK_STEP = 26
SCOPE = "regex_articulation_only"


def build_plan(*, style, receipt, receipt_sha256, gpu):
    """Pure CPU plan: fixed admission scope, two windows, no native attestation."""
    if style not in ("pinch", "tripod", "power"):
        raise ValueError("explicit Pinch/Tripod/Power style required")
    if (not isinstance(receipt_sha256, str) or len(receipt_sha256) != 64
            or any(c not in "0123456789abcdef" for c in receipt_sha256)
            or type(gpu) is not int or gpu < 0):
        raise ValueError("explicit receipt SHA256 and physical GPU identity required")
    # Small diagnostic normalized commands; original rate/lead/limit/clearance
    # kernels below determine physical targets. No learned policy is invented.
    command = np.linspace(.015, .04, 24, dtype=np.float64)
    first = np.stack((command, -command))
    return dict(version="dh116-regex-articulation-probe-v1", scope=SCOPE,
        style=style, receipt=str(receipt), receipt_sha256=receipt_sha256, gpu=gpu,
        environments=2, articulations=2, sdk_initialization_steps=2,
        setup_steps=8, measurement_steps=16, measurement_windows=2,
        measurement_start_sdk_step=10, max_sdk_step=MAX_SDK_STEP, reset_steps=0,
        physics_hz=480, control_hz=60, commands=[first.tolist(), (-first).tolist()],
        command_source="explicit bounded diagnostic 24D; original per-substep kernels",
        rod_steps=0, ppo_updates=0, teacher_queries=0, reward_evaluations=0,
        legacy_equivalence_proven=False, full_chain_qualified=False,
        native_execution_performed=False, grasp_qualified=False,
        planned_checks=["actual_two_regex_views", "native_target_readback",
            "nonzero_dynamic_response", "34_original_source_colliders_per_lane",
            "zero_step_selected_Reset_peer_state_and_pending_scratch_bitwise",
            "existing_lifecycle_normal_close_and_natural_child_exit"])


def action_for_tick(plan, sdk_step):
    index = sdk_step - plan["measurement_start_sdk_step"] - 1
    if type(sdk_step) is not int or not 0 <= index < 16:
        raise ValueError("exactly two eight-step measurement windows required")
    value = np.asarray(plan["commands"][index // 8], dtype=np.float64)
    if value.shape != (2, 24) or not np.isfinite(value).all() or (np.abs(value) > 1).any():
        raise ValueError("finite normalized distinct N2 24D measurement actions required")
    if np.array_equal(value[0], value[1]):
        raise ValueError("distinct per-lane 24D measurement commands required")
    return value.copy()


def verify_peer_preserved(before, after, *, selected_env_id):
    """Compare actual readback bytes; numerical equality does not prove bitwise."""
    if selected_env_id not in (0, 1) or set(before) != set(SIDES) or set(after) != set(SIDES):
        raise ValueError("complete bilateral N2 Reset snapshots required")
    peer = 1 - selected_env_id
    checked = []
    for side in SIDES:
        if not before[side] or set(before[side]) != set(after[side]):
            raise ValueError("matching nonempty peer field inventory required")
        for name, original in before[side].items():
            changed = np.asarray(after[side][name])
            original = np.asarray(original)
            if (original.shape != changed.shape or original.dtype != changed.dtype
                    or original.ndim < 1 or original.shape[0] != 2
                    or not np.isfinite(original).all() or not np.isfinite(changed).all()
                    or original[peer].tobytes() != changed[peer].tobytes()):
                raise ValueError(f"selected Reset changed peer {side}.{name}")
            checked.append(side + "." + name)
    return dict(selected_env_id=selected_env_id, peer_env_id=peer,
                peer_bitwise_preserved=True, checked_fields=checked, reset_sdk_steps=0)


def admit_natural_exit(returncode, close, lifecycle):
    """Admit only narrow hand-API evidence after the child actually exits."""
    actual = lifecycle.get("acquisition", {})
    expected = dict(scope=SCOPE, native_execution_performed=True,
        environments=2, articulations=2, sdk_step=MAX_SDK_STEP, physics_rows=16,
        setup_steps=8, measurement_windows=2, isolated_reset_peer_bitwise=True,
        distinct_24d_commands=True, dynamic_state_changed=True,
        all_source_clearance_passed=True, backend_targets_verified=True,
        legacy_equivalence_proven=False, full_chain_qualified=False, rod_steps=0, ppo_updates=0)
    if (returncode != 0 or close.get("app_close_returned") is not True
            or close.get("close_mode") != "normal" or lifecycle.get("close_mode") != "normal"
            or lifecycle.get("failure") is not None
            or lifecycle.get("lifetime_released_before_close") is not True
            or lifecycle.get("physics_steps") != MAX_SDK_STEP
            or any(actual.get(key) != value for key, value in expected.items())):
        raise ValueError("natural zero exit, complete bounded measurement and verified normal cleanup required")
    return dict(scope=SCOPE, native_articulation_api_passed=True,
        process_exit_verified=True, returncode=returncode, app_close_returned=True,
        close_mode="normal", lifetime_released_before_close=True,
        legacy_equivalence_proven=False, full_chain_qualified=False,
        rod_reward_exchange_qualified=False, grasp_qualified=False, teacher_admission=False,
        environments=2, articulations=2, sdk_step=MAX_SDK_STEP,
        setup_steps=8, measurement_steps=16, measurement_windows=2)


def _snapshot(batch):
    result = {}
    for side, state in batch.read().items():
        owner = batch.sides[side]
        values = dict(state)
        for kind in ("permanent", "instantaneous"):
            composer = getattr(owner.hand, kind + "_wrench_composer")
            for name in ("force", "torque"):
                values[kind + "_" + name] = getattr(composer, "composed_" + name + "_as_torch").index_select(0, owner.row_indices)
        result[side] = {key: value.detach().cpu().numpy().copy() for key, value in values.items()}
    return result


def _seed_packets(batch, seed, origins):
    import torch
    from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import _targets

    packets = {}
    for index, side in enumerate(SIDES):
        owner = batch.sides[side]
        q, qd = _targets(seed, index, owner.joint_names)
        pose = np.repeat(np.asarray(seed["reset_root_pose_w"])[index:index + 1], 2, axis=0)
        pose[:, :3] += origins
        packets[side] = {name: torch.as_tensor(value, device=owner.device, dtype=torch.float32)
            for name, value in dict(root_pose_wxyz=pose,
                joint_pos=np.repeat(q, 2, axis=0), joint_vel=np.repeat(qd, 2, axis=0)).items()}
    return packets


def _reset_hands(sim, batch, packets, effort_limit, env_ids):
    before = sim.current_time_step_index
    for side in SIDES:
        selected = {name: value[env_ids] for name, value in packets[side].items()}
        batch.sides[side].reset(env_ids=env_ids, active_effort_limit_nm=effort_limit, **selected)
    sim.physics_sim_view.update_articulations_kinematic()
    batch.update(1 / 480)
    if sim.current_time_step_index != before:
        raise ValueError("articulation Reset unexpectedly advanced physics")


def _selected_reset(sim, batch, packets, effort_limit):
    import torch

    # Nonzero scratch witnesses are NEVER submitted or stepped. This exercises
    # native composer preservation without adding a synthetic physical force.
    for side in SIDES:
        owner = batch.sides[side]
        force = torch.zeros((2, 23, 3), device=owner.device)
        torque = torch.zeros_like(force)
        force[:, 0, 0], torque[:, 0, 1] = .125, .0625
        for kind in ("permanent", "instantaneous"):
            owner.set_wrenches(forces=force, torques=torque, env_ids=[0, 1], composer=kind)
    before = _snapshot(batch)
    sdk_step = sim.current_time_step_index
    _reset_hands(sim, batch, packets, effort_limit, [1])
    after = _snapshot(batch)
    proof = verify_peer_preserved(before, after, selected_env_id=1)
    for side in SIDES:
        np.testing.assert_allclose(after[side]["q"][1], packets[side]["joint_pos"][1].cpu().numpy(), atol=1e-6, rtol=0)
        np.testing.assert_allclose(after[side]["qd"][1], packets[side]["joint_vel"][1].cpu().numpy(), atol=1e-6, rtol=0)
        np.testing.assert_allclose(after[side]["root_pose_wxyz"][1], packets[side]["root_pose_wxyz"][1].cpu().numpy(), atol=1e-6, rtol=0)
        if np.any(after[side]["root_velocity"][1]):
            raise ValueError("selected Reset retained root velocity")
        for kind in ("permanent", "instantaneous"):
            for field in ("force", "torque"):
                if np.any(after[side][kind + "_" + field][1]):
                    raise ValueError("selected Reset retained composer scratch")
    batch.reset_composers(env_ids=[0])  # Clear witness BEFORE any physical step.
    return dict(proof, sdk_step=sdk_step, scratch_sentinel_never_submitted=True)


def _measure(sim, batch, seed, prepared, origins, plan, output):
    import torch
    from go2_pvcnn.control.dh116.linear_coupling import MIMIC, active_limits
    from go2_pvcnn.control.dh116.tabletop_batch_geometry import collision_clearance
    from go2_pvcnn.control.dh116.tabletop_command_kernels import finger_pre_guard, surface_final_guard, wrist_requested_targets
    from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import project_policy_wrist_targets, CLEARANCE_M
    from go2_pvcnn.control.dh116.tabletop_palm_mount import JOINT_NAMES, ANGULAR_JOINT_NAMES, local_translation
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
    from scripts.diagnose_dh116_teacher_initialization import write_new

    device = batch.sides["left"].device
    tensor = lambda value: torch.as_tensor(value, device=device, dtype=torch.float64)
    packets = _seed_packets(batch, seed, origins)
    effort = float(prepared["candidate_physics"]["hand"]["active_effort_limit_nm"])
    shapes = [(index, batch.hands[index].body_names.index(name), row["vertices"])
        for index, side in enumerate(SIDES)
        for name, row in prepared["hands"][side]["source_colliders"].items()]
    lower, upper = (tensor(np.tile(value, 2)) for value in active_limits())
    limits = torch.stack((lower, upper), dim=-1)
    root_seed = tensor(np.repeat(np.asarray(seed["authored_root_seed_w"])[None], 2, axis=0))
    root_seed[:, :, :3] += tensor(origins)[:, None]
    offsets = tensor([seed["initial_offset_m"]] * 2)
    palms = torch.zeros((2, 2, 3), device=device, dtype=torch.float64)
    angles = torch.zeros_like(palms)
    rates = dict(joint_rate_rad_s=ACTION_CONTRACT["joint_rate_rad_s"],
        wrist_rate_m_s=ACTION_CONTRACT["translation_rate_m_s"], rotation_rate_rad_s=ACTION_CONTRACT["rotation_rate_rad_s"])
    minimum = float("inf")

    def capture():
        nonlocal minimum
        states = batch.read()
        poses = torch.stack([states[side]["link_pose_wxyz"] for side in SIDES], 1)
        clear, _ = collision_clearance(poses.detach().cpu().numpy(), shapes)
        minimum = min(minimum, float(clear.min()))
        if not np.isfinite(clear).all() or (clear < CLEARANCE_M).any():
            raise ValueError("original source clearance floor rejected native state")
        active_q, errors = [], []
        for side in SIDES:
            owner, state = batch.sides[side], states[side]
            if any(not bool(torch.isfinite(value).all()) for value in state.values()):
                raise ValueError("nonfinite actual Articulation readback")
            q = state["q"].double()
            active_q.append(q[:, owner.active_ids])
            errors.extend(q[:, owner.joint_names.index(f)] - ratio * q[:, owner.joint_names.index(m)] - offset
                          for f, m, ratio, offset in MIMIC)
        q = torch.cat(active_q, dim=1)
        errors = torch.stack(errors, dim=1)
        if bool(((q < limits[:, 0] - .02) | (q > limits[:, 1] + .02)).any()) or bool((errors.abs() > .02).any()):
            raise ValueError("original measured joint-limit/mimic safety guard rejected native state")
        return states, tensor(clear), q, errors

    def step(action, expected):
        nonlocal palms, angles
        states, clear, q, errors = capture()
        command = torch.cat([states[side]["backend_targets"][:, batch.sides[side].active_ids] for side in SIDES], 1).double()
        action = tensor(action)
        requested = finger_pre_guard(command, action, q, errors, lower, upper, 1 / 480, rates, policy_owned=True)
        _, requested = surface_final_guard(command, requested, lower, upper, 1 / 480)
        new_palms, new_angles = wrist_requested_targets(palms, angles, action, rates, 1 / 480)
        bases = torch.stack([states[side]["link_pose_wxyz"][:, batch.sides[side].body_names.index("base_link")] for side in SIDES], 1).double()
        projection = project_policy_wrist_targets(root_seed_w=root_seed, measured_base_w=bases,
            common_lift_offset_m=offsets, measured_clearance_m=clear,
            palm_offsets_m=palms, requested_palm_offsets_m=new_palms, angles_rad=angles,
            requested_angles_rad=new_angles, allowed=torch.ones(2, dtype=torch.bool, device=device),
            live=torch.ones(2, dtype=torch.bool, device=device))
        palms, angles = projection["palm_offsets_m"], projection["projected_angles"]
        targets = {}
        for index, side in enumerate(SIDES):
            owner, state = batch.sides[side], states[side]
            target = state["backend_targets"].clone()
            target[:, owner.active_ids] = requested[:, 6 * index:6 * index + 6].float()
            linear = [owner.joint_names.index(name) for name in JOINT_NAMES]
            angular = [owner.joint_names.index(name) for name in ANGULAR_JOINT_NAMES]
            target[:, linear] = local_translation(projection["projected_root_poses"][:, index], state["root_pose_wxyz"].double()).float()
            target[:, angular] = angles[:, index].float()
            owner.set_joint_position_target(target)
            targets[side] = target
        batch.write_data_to_sim()
        submitted = batch.read()
        for side in SIDES:
            if not torch.equal(submitted[side]["backend_targets"], targets[side]):
                raise ValueError("native position targets differ from actual submitted command")
        sim.step(render=False)
        if sim.current_time_step_index != expected:
            raise ValueError("unexpected shared 480Hz SDK boundary")
        batch.update(1 / 480)
        capture()

    _reset_hands(sim, batch, packets, effort, [0, 1])
    for expected in range(3, 11):
        step(np.zeros((2, 24)), expected)
    # Setup is excluded from measurement; restore the exact authenticated seed.
    _reset_hands(sim, batch, packets, effort, [0, 1])
    palms.zero_(); angles.zero_()
    start = _snapshot(batch)
    rows = []
    for expected in range(11, 27):
        action = action_for_tick(plan, expected)
        step(action, expected)
        state = _snapshot(batch)
        rows.append(dict(sdk_step=expected, window=(expected - 11) // 8,
            actions=action.tolist(), q={side: state[side]["q"].tolist() for side in SIDES},
            backend_targets={side: state[side]["backend_targets"].tolist() for side in SIDES}))
        if expected == 18:
            proof = _selected_reset(sim, batch, packets, effort)
            palms[1] = 0; angles[1] = 0
            write_new(output / "selected_reset.json", proof)
    finish = _snapshot(batch)
    if not all(any(not np.array_equal(start[side]["q"][lane], finish[side]["q"][lane])
                   for side in SIDES) for lane in (0, 1)):
        raise ValueError("distinct 24D native command lanes lacked measurable dynamic response")
    write_new(output / "measurement_rows.json", dict(rows=rows, windows=2, physics_steps=16))
    return dict(scope=SCOPE, native_execution_performed=True, environments=2,
        articulations=2, sdk_step=sim.current_time_step_index, physics_rows=16,
        setup_steps=8, measurement_windows=2, isolated_reset_peer_bitwise=True,
        distinct_24d_commands=True, dynamic_state_changed=True,
        all_source_clearance_passed=True, min_source_clearance_m=minimum,
        backend_targets_verified=True, legacy_equivalence_proven=False,
        full_chain_qualified=False, rod_steps=0, ppo_updates=0, reward_evaluations=0,
        native_nonzero_reaction_submitted=False, teacher_queries=0,
        physics_budget=dict(sdk_initialization_steps=2, setup_steps=8, measurement_steps=16,
            selected_reset_sdk_steps=0, rod_steps=0, max_sdk_step=MAX_SDK_STEP),
        row_env_ids={side: list(batch.sides[side].row_env_ids) for side in SIDES},
        actual_articulation_paths={side: list(batch.sides[side].hand.root_physx_view.prim_paths) for side in SIDES})


def author_and_exercise(sim, owners, trace, project, prepared, archive, lanes, output, gpu, max_sdk_step):
    """Single audit._run callback; never enters its old per-environment author."""
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as assets
    from go2_pvcnn.control.dh116.tabletop_dual_pinch_builder import _Composers
    from go2_pvcnn.control.dh116.tabletop_deformx_end_supported_scene import _bind_support_material
    from go2_pvcnn.control.dh116.tabletop_end_supported_scene import author_end_supports
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import POWER_SCOPE, TRIPOD_SCOPE, prepare_initial_state
    from go2_pvcnn.control.dh116.tabletop_regex_articulation import spawn_regex_articulations
    from go2_pvcnn.control.dh116.tabletop_vec_isaac import isolate_scenes
    from scripts.diagnose_dh116_teacher_initialization import write_new

    if len(lanes) != 2 or max_sdk_step != MAX_SDK_STEP or sim.current_time_step_index != 0:
        raise ValueError("explicit N2 startup and 2+8+16 SDK budget required")
    args, layout, sources = assets.spawn_parameters(project, prepared)
    if archive.style == "pinch":
        from go2_pvcnn.control.dh116.tabletop_dual_pinch_builder import _paired_sources
        _paired_sources(project, archive.pair, prepared)
        seed = archive.pair["reset_seed"]
    else:
        scope = {"tripod": TRIPOD_SCOPE, "power": POWER_SCOPE}[archive.style]
        seed, active, _, _ = prepare_initial_state(archive.task, prepared["table"], expected_scope=scope)
        if seed["sha256"] != archive.task_seed_sha256:
            raise ValueError("supported grasp seed differs from its separately authenticated receipt")
        for index, side in enumerate(SIDES):
            layout["hand_roots"][side] = np.asarray(seed["authored_root_seed_w"][index]).copy()
            layout["hand_roots"][side][2] -= assets.BUILDER_OFFSET_M
            layout["active_joints"][side] = active[index].copy()
    origins = np.asarray([origin for _, origin in lanes], dtype=float)
    batch = owners.keep("regex_batch", spawn_regex_articulations(sim, args, layout, sources, origins))
    for scene in batch.scenes:
        scoped = dict(prepared, prefix=scene["prefix"], table=dict(prepared["table"],
            center_m=(np.asarray(prepared["table"]["center_m"]) + scene["origin"]).tolist()))
        assets._bind_materials(sim.stage, scoped)
        if assets.inspect_source_stage(sim.stage, scoped)["original_colliders"] != 34:
            raise ValueError("all 34 original material-bound colliders per lane required")
        if archive.task is not None:
            support = deepcopy(archive.task["support_scene"])
            for box in support["supports"]:
                box["center_m"] = (np.asarray(box["center_m"]) + scene["origin"]).tolist()
            paths = author_end_supports(sim.stage, scene["prefix"], support)
            material = prepared["candidate_physics"]["table"]
            _bind_support_material(sim.stage, scene["prefix"], paths, material["mu_s"], material["mu_k"])
    isolate_scenes(sim.stage, batch.scenes)
    for side, hand in zip(SIDES, batch.hands, strict=True):
        trace.watch(f"regex_{side}_hand", hand, owner=SCOPE, release="owned scope then SDK STOP", callbacks=True)
    if sim.current_time_step_index != 0:
        raise ValueError("USD authoring advanced physics")
    sim.reset()
    batch.bind()
    members = []
    for side in SIDES:
        hand = batch.sides[side].hand
        for kind, value in (("view", hand.root_physx_view),
                ("permanent", hand.permanent_wrench_composer), ("instantaneous", hand.instantaneous_wrench_composer)):
            trace.watch(f"regex_{side}_{kind}", value, owner=SCOPE, release="owned scope then SDK STOP", callbacks=True)
            if kind != "view":
                members.append(value)
    owners.keep("composer", _Composers(members))
    assets.validate_simulation_settings(sim, prepared)
    if sim.current_time_step_index != 2:
        raise ValueError("one global reset with exactly two initialization steps required")
    plan = build_plan(style=archive.style, receipt="validated_by_parent",
        receipt_sha256=archive.reset_receipt_sha256, gpu=gpu)
    write_new(output / "initialized.json", dict(environments=2, articulations=2, sdk_step=2,
        real_root_suffixes=list(batch.root_suffixes), original_colliders_per_lane=34))
    result = _measure(sim, batch, seed, prepared, origins, plan, output)
    write_new(output / "regex_articulation_observations.json", result)
    return result


def _worker(args, project):
    from scripts.audit_dh116_isaac_batch_scene import _run, load_grasp_archive, _gpu_memory
    from scripts.run_dh116_work_packages import NATIVE_LOCK, native_file_lock

    if not {"author_and_exercise", "owned_names"}.issubset(inspect.signature(_run).parameters):
        raise RuntimeError("audit._run requires the explicit author_and_exercise/owned_names hook; refusing old per-environment author")
    if os.getenv("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("explicit matching physical CUDA device required")
    archive = load_grasp_archive(args.style, args.receipt, args.receipt_sha256)
    with native_file_lock(NATIVE_LOCK):
        if _gpu_memory(args.gpu) > 512:
            raise RuntimeError("GPU in use before regex articulation launch")
        _run(project, archive, args.gpu, 2, args.output, coupled=False,
             max_sdk_step=MAX_SDK_STEP, author_and_exercise=author_and_exercise, owned_names=OWNED_NAMES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", required=True, choices=("pinch", "tripod", "power"))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    plan = build_plan(style=args.style, receipt=args.receipt,
                      receipt_sha256=args.receipt_sha256, gpu=args.gpu)
    if args.worker:
        _worker(args, project)
        return
    from scripts.audit_dh116_isaac_batch_scene import load_grasp_archive
    from scripts.diagnose_dh116_teacher_initialization import write_new

    load_grasp_archive(args.style, args.receipt, args.receipt_sha256)
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / "preflight.json", plan)
    if args.preflight_only:
        return
    native = args.output / "native"
    native.mkdir()
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--style", args.style,
        "--receipt", str(args.receipt.resolve()), "--receipt-sha256", args.receipt_sha256,
        "--gpu", str(args.gpu), "--output", str(native.resolve())]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    with (args.output / "native.log").open("wb") as log:
        child = subprocess.run(command, cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    write_new(args.output / "process_exit.json", dict(returncode=child.returncode,
        process_exit_verified=True, exited_without_signal=child.returncode >= 0,
        signal=-child.returncode if child.returncode < 0 else None, supervisor_terminated_child=False))
    if child.returncode != 0:
        raise RuntimeError(f"native child exited with {child.returncode}; inspect native.log in the output directory")
    close = json.loads((native / "app_close_returned.json").read_text())
    lifecycle = json.loads((native / "result.json").read_text())
    proof = admit_natural_exit(child.returncode, close, lifecycle)
    write_new(args.output / "articulation_api_qualification.json", dict(proof,
        style=args.style, receipt_sha256=args.receipt_sha256,
        evidence=dict(lifecycle="native/result.json", close="native/app_close_returned.json",
            reset="native/selected_reset.json", rows="native/measurement_rows.json",
            process_exit="process_exit.json")))


if __name__ == "__main__":
    main()
