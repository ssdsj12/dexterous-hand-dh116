#!/usr/bin/env python3
"""Verify continuous coupling with two original palms and one free task rod."""
import argparse
from functools import partial
import json
from pathlib import Path
import traceback
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation

from scripts import record_dh116_teacher_numeric_reset as reset
from scripts import verify_dh116_power_palm_reactions as sources
from scripts.verify_dh116_deformx_thumb_contact import step_thumb_physics
from scripts.verify_dh116_power_live_palm_feedback import hand_safety

ROOT = Path(__file__).resolve().parents[1]
SDK_END = 10
DT = 1 / 480
SUBSTEPS = 12
SIDES = ("left", "right")
native = reset.native


class SharedPowerOwners:
    """Reserve all pair slots at the original builder's only allocation call."""

    def __init__(self, owners):
        self._owners = owners
        self.resources = owners.resources
        self.allocated_capacity = None

    def keep(self, name, value):
        value = self._owners.keep(name, value)
        if name == "runtime":
            value._base.rope = self._pair_allocation(value._base)
        return value

    def _pair_allocation(self, base):
        rope = base.rope
        node_substeps = len(base.nodes) * SUBSTEPS

        class CapacityReservation:
            def enable_contact_pair_capture(reservation, capacity):
                base.rope = rope
                if capacity != node_substeps * 11 or self.allocated_capacity is not None:
                    raise ValueError("exact original one-time pair-allocation call required")
                rope.enable_contact_pair_capture(node_substeps * 13)
                self.allocated_capacity = node_substeps * 13

        return CapacityReservation()


def measure_links(sim, bindings, mesh_bindings, clearance_shapes, *, sdk):
    from go2_pvcnn.control.dh116.tabletop_batch_geometry import collision_clearance
    from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import _owned
    from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M

    if native.clock(sim)["sdk_step"] != sdk:
        raise ValueError("live links must share the requested SDK boundary")
    poses, centers, frames = {}, {}, []
    for side in SIDES:
        hand = bindings[side].hand
        view = hand.root_physx_view
        native_poses, local_com = (_owned(fn()) for fn in (view.get_link_transforms, view.get_coms))
        if (native_poses.shape != (1, 23, 7) or local_com.shape != native_poses.shape
                or not all(np.isfinite(v).all() for v in (native_poses, local_com))
                or not np.allclose(np.linalg.norm(native_poses[..., 3:], axis=-1), 1, atol=1e-5, rtol=0)):
            raise ValueError("finite original DH116 pose and local COM buffers required")
        frames.append(native_poses[..., [0, 1, 2, 6, 3, 4, 5]])
        for name, record in mesh_bindings.items():
            if record["side"] != side:
                continue
            body = record["body_index"]
            if (hand.body_names[body] != record["body_name"]
                    or view.link_paths[0][body] != record["prim_path"]):
                raise ValueError("bound solver link differs from current PhysX instance")
            xyz, quat = native_poses[0, body, :3], native_poses[0, body, 3:]
            poses[name] = (xyz.copy(), quat.copy())
            centers[name] = xyz + Rotation.from_quat(quat).apply(local_com[0, body, :3])
    clearance, _ = collision_clearance(np.stack(frames, axis=1), clearance_shapes)
    if native.clock(sim)["sdk_step"] != sdk:
        raise ValueError("SDK moved during link and COM readback")
    if not np.isfinite(clearance).all() or np.any(clearance < CLEARANCE_M):
        raise ValueError("original full-hand table clearance protection failed")
    return poses, centers, clearance


def bind_palm_sources(sim, bindings, original, *, args):
    from ropesim import sdf
    from scripts.audit_dh116_power_palm_parity import read_palm_source

    rays = sources.checked_json(args.rays, sources.RAYS_SHA)["palms"]
    palm_bindings, palm_grids = {}, {}
    for side, bake in (("left", args.left_bake), ("right", args.right_bake)):
        grid_args, _, frozen, _ = sources.fixture(side, rays[side], bake)
        archive = sources.checked_json(args.rays.parent / f"{side}_base_link.json", frozen["gpu_row_sha256"])
        binding = bindings[side]
        body = binding.mount.base_id
        path = binding.hand.root_physx_view.link_paths[0][body]
        _, _, live = read_palm_source(sim.stage, path)
        parent = original.bindings[f"{side}.finger14_link"]
        if (binding.hand.body_names[body] != "base_link"
                or Path(path).parent != Path(parent["prim_path"]).parent
                or live["geometry_sha256"] != archive["source"]["geometry_sha256"]
                or not np.allclose(live["mesh_to_body"], archive["source"]["mesh_to_body"], atol=2e-6, rtol=0)):
            raise ValueError("Power palm source differs from the previously measured mounted link")
        key = f"{side}.base_link"
        palm_grids[key] = sdf.SdfGrid(*grid_args)
        palm_bindings[key] = dict(side=side, body_name="base_link", body_index=body, prim_path=path,
            source_usd_sha256=parent["source_usd_sha256"], source_geometry_sha256=live["geometry_sha256"],
            cooked_geometry_sha256=archive_geometry_digest(archive), capture_sha256=frozen["gpu_row_sha256"],
            sdf_sha256=frozen["sdf_file_sha256"])
    return palm_bindings, palm_grids


def archive_geometry_digest(archive):
    from scripts.audit_dh116_cooked_hull_device_parity import _pad_hull

    hull, _ = _pad_hull(archive)
    return native.digest(dict(vertices=np.asarray(hull["vertices"]).tolist(), faces=np.asarray(hull["faces"]).tolist()))


def acquire_original_power_reset(sim, owners, trace, output, *, profile):
    """Use the original measured Reset with the Power capture allocation reserved."""
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import profile_reader_scope
    from scripts import audit_dh116_pinch_initial_pose as geometry
    from scripts.audit_dh116_certified_segment_distance import segment_hull_distance
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    from scripts.verify_dh116_dual_pinch_interval import diagnostic_reset_owners

    run_uuid = "power-shared-rod-" + uuid4().hex
    index = 0

    def record(event):
        nonlocal index
        from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
        native.write_new(output/f"power_reset_source_{index}.json", encode_owned(event))
        index += 1

    try:
        with profile_reader_scope(profile), bindings(
                (builder, "BACKEND", Path(profile.binary_path)),
                (builder, "BACKEND_SHA", profile.binary_sha256),
                (geometry, "segment_hull_distance", segment_hull_distance)):
            return builder.build_reset_owned(sim, owners, project=ROOT, prepared=scene.prepare_scene(ROOT),
                ownership=trace, record=record, owner_resets=diagnostic_reset_owners(), run_uuid=run_uuid,
                episode_uid=run_uuid+":0", physics_budget_end=SDK_END)
    finally:
        sim = owners = trace = None


def run_shared(sim, owners, trace, output, *, profile, args, supported_acquire=None):
    import torch
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
    from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import measure_power_contact
    from go2_pvcnn.control.dh116.tabletop_power_shared_rod import (
        contact_links, PowerSharedRodRuntime, write_composer_reactions,
    )

    episode = original = power = bindings = composers = binding = scoped_owners = None
    setup = support_options = None
    try:
        if supported_acquire is None:
            scoped_owners = SharedPowerOwners(owners)
            setup = acquire_original_power_reset(sim, scoped_owners, trace, output, profile=profile)
        else:
            setup, support_options = supported_acquire(sim, owners, trace, output, profile=profile)
        episode = owners.resources["dual_episode"]
        original = owners.resources["runtime"]
        clearance_shapes = episode._exchange._read_geometry._shapes
        bindings = {binding.side:binding for binding in episode._bindings}
        if setup["clock"]["physics_step"] != 2 or native.clock(sim)["sdk_step"] != 2:
            raise ValueError("original same-clock task Reset and table clearance required")
        clock = original.clock()
        added, grids = bind_palm_sources(sim, bindings, original, args=args)
        meshes = {**original.bindings, **added}
        first_poses, _, _ = measure_links(sim, bindings, meshes, clearance_shapes, sdk=2)
        power = owners.keep("power_runtime", PowerSharedRodRuntime.from_original_reset(original,
            palm_bindings=added, palm_grids=grids, palm_poses={name:first_poses[name] for name in added},
            expected_clock=clock, **(support_options or {})))
        if power.rope is not original.rope or power.clock() != clock:
            raise ValueError("the shared rod or its reset boundary changed")
        nodes = len(power._base.nodes)
        capacity = nodes * len(power.mesh_names) * SUBSTEPS
        if scoped_owners is not None and scoped_owners.allocated_capacity != capacity:
            raise ValueError("the single original allocation must cover all Power mesh pairs")
        native.write_new(output/"power_binding.json", dict(reset_clock=clock, added=added, mesh_names=power.mesh_names,
                         rods_per_env=1, solver_steps_before_registration=0, solver_steps_after_registration=0,
                         pair_capacity=scoped_owners.allocated_capacity))
        composers = {side:bindings[side].hand.permanent_wrench_composer for side in SIDES}
        links = contact_links(power.bindings)
        to_tensor = partial(torch.as_tensor, dtype=torch.float32, device="cuda:0")
        route = {side:dict(body_ids=[meshes[f"{side}.{name}"]["body_index"] for name in links],
                          forces=np.zeros((1,len(links),3)), torques=np.zeros((1,len(links),3)), is_global=True)
                 for side in SIDES}
        loaded = {name:0 for name in power.mesh_names}
        max_residual = 0.
        contact_exchanges = 0
        opposing_exchanges = 0
        peak_support_loads = {name: 0. for name in ("table", "support_0", "support_1")}
        for before_sdk in range(2, SDK_END):
            if power.clock()["physics_step"] != before_sdk:
                raise ValueError("only the preceding same-rod reaction can be delivered")
            if any(np.linalg.norm(row["forces"], axis=-1).max() > 30. for row in route.values()):
                raise ValueError("unchanged excess force guard")
            write_composer_reactions(composers, route, to_tensor=to_tensor)
            for binding in bindings.values():
                binding.hand.write_data_to_sim()
            sdk = step_thumb_physics(sim, before_sdk, budget_end=SDK_END)
            for binding in bindings.values():
                binding.hand.update(DT)
                hand_safety(binding)
            poses, centers, clearance = measure_links(sim, bindings, meshes, clearance_shapes, sdk=sdk)
            row = power.step(poses, centers, physics_step=sdk)
            native_joint_state = None
            if supported_acquire is not None:
                from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import _owned
                from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import read_joint_target_evidence

                native_joint_state = {}
                for side, binding in bindings.items():
                    view = binding.hand.root_physx_view
                    target = read_joint_target_evidence(binding.hand)["backend"]
                    if not target["available"]:
                        raise ValueError("actual Power backend position targets are required")
                    native_joint_state[side] = dict(names=list(binding.hand.joint_names),
                        q_rad=_owned(view.get_dof_positions()), qd_rad_s=_owned(view.get_dof_velocities()),
                        target_rad=_owned(target["values"]), target_source=target["source"])
                for name, wrench in row["external_support_wrenches_world"].items():
                    peak_support_loads[name] = max(peak_support_loads[name], float(np.linalg.norm(wrench[:3])))
            native.write_new(output/f"power_exchange_{sdk}.json", encode_owned(dict(
                route=route, native_joint_state=native_joint_state, preceding_reaction_sdk_step=before_sdk if before_sdk>2 else None,
                clearance_m=clearance, rod=row, hand_link_poses=poses, hand_link_coms=centers)))
            if args.measure_contact_evidence:
                evidence = measure_power_contact(row)
                native.write_new(output/f"power_contact_{sdk}.json", evidence)
                contact_exchanges += 1
                opposing_exchanges += int(evidence["all_substeps_power_contacts"])
            route = row["hand_wrenches_com_world"]
            for pair in row["rod_mesh_contact_pairs"]["pairs"]:
                loaded[pair["mesh_name"]] += 1
            max_residual = max(max_residual, row["pair_force_residual_n"])
        result = dict(scope="original_dual_hand_shared_rod_no_action_power_layout",
            rods_per_env=1, sdk_steps=SDK_END, measured_exchanges=SDK_END-2,
            solver_substeps=power.clock()["rod_tick"], mesh_names=power.mesh_names,
            pair_counts=loaded, max_force_balance_residual_n=max_residual,
            native_qualification=False, grasp_qualified=False, teacher_admission=False, ppo_updates=0)
        if supported_acquire is not None:
            result.update(scope="measured_power_end_support_load_no_action", peak_support_force_n=peak_support_loads,
                          power_contact_qualified=False, supported_seed_clock=setup["clock"])
        if args.measure_contact_evidence:
            result.update(same_clock_contact_exchanges=contact_exchanges,
                          bilateral_opposed_contact_exchanges=opposing_exchanges)
        native.write_new(output/"power_shared_result.json", result)
        return result
    except Exception:
        sdk = native.clock(sim)["sdk_step"]
        if not 2 <= sdk <= SDK_END:
            raise
        failure = dict(failure=traceback.format_exc(), guarded_stop_sdk_step=sdk,
                       grasp_qualified=False, teacher_admission=False, ppo_updates=0)
        native.write_new(output/"power_failure.json", failure)
        return failure
    finally:
        sim = owners = trace = episode = original = power = bindings = composers = binding = scoped_owners = None
        view = setup = support_options = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("output", "plan", "rays", "left-bake", "right-bake"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--approved-frozen-sha256")
    parser.add_argument("--measure-contact-evidence", action="store_true")
    args = parser.parse_args(argv)
    native.validate_execution_context(ROOT)
    plan = reset.freeze_plan(args.output)
    plan.update(version="dh116-power-shared-task-rod-v1", preserved_scope="one original task rod, two hands, held joint targets")
    plan["budget"].update(mode="power_shared_task_rod", sdk_steps=SDK_END, solver_steps=(SDK_END-2)*SUBSTEPS,
                         new_power_mesh_registrations=2)
    rays = sources.checked_json(args.rays, sources.RAYS_SHA)["palms"]
    files = [args.rays, Path(__file__)]
    if args.measure_contact_evidence:
        plan["same_clock_power_contact_evidence"] = True
        files.append(ROOT / "src/go2_pvcnn/control/dh116/tabletop_power_contact_evidence.py")
    for side, bake in (("left", args.left_bake), ("right", args.right_bake)):
        _, _, frozen, _ = sources.fixture(side, rays[side], bake)
        archive = args.rays.parent / f"{side}_base_link.json"
        sources.checked_json(archive, frozen["gpu_row_sha256"])
        files.extend((archive, bake/"result.json", bake/f"{side}.base_link.npz"))
    plan["frozen_files"].update({str(path.resolve(strict=True)):native.sha(path) for path in files})
    digest = native.digest(plan)
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=digest, budget=plan["budget"])))
    else:
        if (args.approved_frozen_sha256 != digest or json.loads(args.plan.read_bytes()) != plan
                or args.output.exists()):
            raise ValueError("exact approved shared-rod recipe and fresh output required")
        from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
        reset.execute(plan, acquire_operation=partial(run_shared, profile=BackendProfile.load(ROOT), args=args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
