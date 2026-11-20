#!/usr/bin/env python3
"""Route original DeformX palm reactions to both mounted DH116 hands in Isaac."""
import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts import record_dh116_teacher_numeric_reset as reset
from scripts import verify_dh116_power_palm_reactions as solver
from scripts.verify_dh116_deformx_rigid_bridge import wrench_at_com
from scripts.verify_dh116_deformx_thumb_contact import step_thumb_physics

ROOT = Path(__file__).resolve().parents[1]
SIDES = ("left", "right")
END_SDK = 5
MAX_FORCE_N = 30.
MIN_RESPONSE_M_S = 2e-5
native = reset.native


def prepare_reaction(palm, solved, *, current_sdk_step):
    moment = solved["moment_reference_audit"]
    force = np.asarray(solved["force_on_mesh_n"], dtype=float)
    torque = np.asarray(solved["torque_about_mesh_origin_world_nm"], dtype=float)
    link, com = (np.asarray(palm[k], dtype=float) for k in ("xyz", "com"))
    if (palm["sdk_step"] != current_sdk_step or not palm["body_path"].endswith("/base_link")
            or type(palm["body_id"]) is not int or palm["body_id"] < 0
            or moment["frame"] != "world" or moment["about"] != "mesh_origin"
            or any(v.shape != (3,) or not np.isfinite(v).all() for v in (force, torque, link, com))
            or not np.array_equal(moment["origin_world_m"], link)
            or not 0. < np.linalg.norm(force) <= MAX_FORCE_N):
        raise ValueError("same-clock original palm mesh and bounded nonzero reaction required")
    wrench = wrench_at_com(np.r_[force, torque], link, com)
    return dict(body_ids=[palm["body_id"]], forces=wrench[None, None, :3].tolist(),
                torques=wrench[None, None, 3:].tolist(), is_global=True)


def audit_velocity_change(force, unloaded, loaded):
    force, unloaded, loaded = (np.asarray(v, dtype=float) for v in (force, unloaded, loaded))
    if (force.shape != (3,) or unloaded.shape != (2, 3) or loaded.shape != (3,)
            or not all(np.isfinite(v).all() for v in (force, unloaded, loaded))
            or np.linalg.norm(force) <= 0.):
        raise ValueError("invalid measured palm response")
    axis = force / np.linalg.norm(force)
    quiet = unloaded @ axis
    projected = float(loaded @ axis)
    threshold = max(4. * float(np.max(np.abs(quiet))), MIN_RESPONSE_M_S)
    if projected <= threshold:
        raise ValueError(f"palm response {projected:.8g} m/s does not exceed {threshold:.8g} m/s")
    return dict(response_detected=True, quiet_projected_m_s=quiet.tolist(),
                loaded_projected_m_s=projected, threshold_m_s=threshold)


def palm_snapshot(binding, sdk_step):
    from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import _owned

    hand, view = binding.hand, binding.hand.root_physx_view
    body_id = binding.mount.base_id
    prim = view.link_paths[0][body_id]
    if (len(view.prim_paths) != 1 or len(view.link_paths) != 1
            or hand.body_names[body_id] != "base_link" or not prim.endswith("/base_link")):
        raise ValueError("unique native mounted palm identity required")
    transforms, coms, velocities = (_owned(fn()) for fn in
        (view.get_link_transforms, view.get_coms, view.get_link_velocities))
    if (transforms.shape != (1, len(hand.body_names), 7) or coms.shape != transforms.shape
            or velocities.shape != (*transforms.shape[:-1], 6)
            or not all(np.isfinite(v).all() for v in (transforms, coms, velocities))):
        raise ValueError("complete original palm pose, COM and velocity required")
    xyz, quat = transforms[0, body_id, :3], transforms[0, body_id, 3:]
    if abs(np.linalg.norm(quat) - 1.) > 1e-5:
        raise ValueError("native unit palm rotation required")
    com = xyz + Rotation.from_quat(quat).apply(coms[0, body_id, :3])
    return dict(body_path=prim, body_id=body_id, sdk_step=sdk_step, xyz=xyz.tolist(), com=com.tolist(),
                quat_xyzw=quat.tolist(), velocity=velocities[0, body_id].tolist())


def hand_safety(binding):
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, MIMIC, active_limits
    from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import _owned
    from go2_pvcnn.control.dh116.tabletop_task import joint_safety_failure

    names = binding.hand.joint_names
    q = _owned(binding.hand.root_physx_view.get_dof_positions())[0]
    mimic = [q[names.index(f)] - (ratio*q[names.index(m)] + offset) for f, m, ratio, offset in MIMIC]
    failure = joint_safety_failure(q[[names.index(name) for name in ACTIVE]],
                                   np.stack(active_limits(), axis=-1), mimic)
    if failure:
        raise ValueError(f"{binding.side}: {failure}")


def run_feedback(sim, owners, trace, output, *, profile, args):
    import omni.usd
    from scripts import audit_dh116_power_palm_parity as palm_source
    import torch
    import ropesim
    from go2_pvcnn.control.dh116.tabletop_batch_geometry import collision_clearance
    from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import _owned
    from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M

    episode = bindings = hand = composer = None
    try:
        reset_outcome = reset._acquire(sim, owners, trace, output, profile)
        if reset_outcome["clearance_passed"] is not True:
            raise ValueError("original mounted hand reset clearance required")
        episode = owners.resources["dual_episode"]
        bindings = dict((item.side, item) for item in episode._bindings)
        if set(bindings) != set(SIDES) or bindings["left"].hand is bindings["right"].hand:
            raise ValueError("two distinct current mounted DH116 articulations required")
        shapes = episode._exchange._read_geometry._shapes
        stage = omni.usd.get_context().get_stage()
        rows = []
        destination = output / "palm_feedback"
        destination.mkdir()
        rays = solver.checked_json(args.rays, solver.RAYS_SHA)["palms"]
        cfg = solver.checked_json(ROOT/"configs/dh116_tasks/flexible_pinch_candidate_v1.json", solver.ROD_CONFIG_SHA)
        for sdk_step in range(2, END_SDK):
            before = {side:palm_snapshot(bindings[side], sdk_step) for side in SIDES}
            routes, reactions = {}, {}
            if sdk_step == END_SDK - 1:
                for side, bake in (("left", args.left_bake), ("right", args.right_bake)):
                    grids, rod_world, origin, witness = solver.fixture(side, rays[side], bake)
                    original_rotation = Rotation.from_quat(origin["quaternion_world_xyzw"])
                    local_nodes = original_rotation.inv().apply(rod_world[0] - origin["position_world_m"])
                    live = before[side]
                    _, _, source = palm_source.read_palm_source(stage, live["body_path"])
                    archived = solver.checked_json(args.rays.parent / f"{side}_base_link.json", origin["gpu_row_sha256"])
                    if (source["geometry_sha256"] != archived["source"]["geometry_sha256"]
                            or not np.allclose(source["mesh_to_body"], archived["source"]["mesh_to_body"],
                                               atol=2e-6, rtol=0)):
                        raise ValueError(f"{side}: current mounted palm does not match original SDF geometry")
                    rotation = Rotation.from_quat(live["quat_xyzw"])
                    world_nodes = (rotation.apply(local_nodes) + live["xyz"])[None].astype(np.float32)
                    live_pose = dict(position_world_m=live["xyz"], quaternion_world_xyzw=live["quat_xyzw"])
                    local_normal = original_rotation.inv().apply(witness["outward_world"])
                    witness = dict(witness, outward_world=rotation.apply(local_normal).tolist(),
                                   live_body=live["body_path"], live_geometry_sha256=source["geometry_sha256"],
                                   source_sdk_step=sdk_step)
                    reactions[side] = solver.run_fixture((grids, world_nodes, live_pose, witness), cfg, ropesim, destination)
                    routes[side] = prepare_reaction(live, reactions[side], current_sdk_step=sdk_step)
                native.write_new(destination / "queued.json", dict(clock=before, rod_reactions=reactions, routes=routes))
            issued_from_buffers = {}
            for side in SIDES:
                hand = bindings[side].hand
                composer = hand.permanent_wrench_composer
                composer.reset()
                if side in routes:
                    data = routes[side]
                    device = hand.data.joint_pos.device
                    composer.set_forces_and_torques(**dict(data,
                        forces=torch.as_tensor(data["forces"], device=device, dtype=torch.float32),
                        torques=torch.as_tensor(data["torques"], device=device, dtype=torch.float32)))
                    idx = before[side]["body_id"]
                    rotation = Rotation.from_quat(before[side]["quat_xyzw"])
                    applied = [
                        rotation.apply(t[0, idx].detach().cpu().numpy()).tolist()
                        for t in (composer.composed_force_as_torch, composer.composed_torque_as_torch)
                    ]
                    if not np.allclose(applied, [data[key][0][0] for key in ("forces", "torques")], atol=3e-5, rtol=2e-5):
                        raise ValueError("actual composer buffers differ from measured reaction")
                    issued_from_buffers[side] = applied
                hand.write_data_to_sim()
            after_step = step_thumb_physics(sim, sdk_step, budget_end=END_SDK)
            for binding in bindings.values():
                binding.hand.update(solver.DT)
                hand_safety(binding)
            poses = np.stack([_owned(bindings[side].hand.root_physx_view.get_link_transforms())
                              for side in SIDES], axis=1)[..., [0, 1, 2, 6, 3, 4, 5]]
            clearance, _ = collision_clearance(poses, shapes)
            if not np.isfinite(clearance).all() or np.any(clearance < CLEARANCE_M):
                raise ValueError("original full-hand clearance protection failed")
            after = {side:palm_snapshot(bindings[side], after_step) for side in SIDES}
            delta = {side: (np.asarray(after[side]["velocity"]) - before[side]["velocity"]).tolist() for side in SIDES}
            row = dict(before=before, after=after, delta_velocity=delta,
                       reaction_source_sdk_step=sdk_step if routes else None, native_issued_world=issued_from_buffers,
                       clearance_m=clearance.tolist())
            rows.append(row)
            native.write_new(destination / f"sdk-{after_step}.json", row)
        measured = {side:audit_velocity_change(reactions[side]["force_on_mesh_n"],
                       [row["delta_velocity"][side][:3] for row in rows[:2]], rows[-1]["delta_velocity"][side][:3])
                    for side in SIDES}
        result = dict(scope="isolated_mounted_articulation_palm_feedback", palms=measured, sdk_steps=END_SDK,
                      solver_substeps=24, control_steps=0, ppo_updates=0, native_qualification=False,
                      grasp_qualified=False, teacher_admission=False)
        native.write_new(destination / "feedback.json", result)
        return result
    finally:
        sim = owners = trace = episode = bindings = hand = composer = None
        binding = stage = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("output", "plan", "rays", "left-bake", "right-bake"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    native.validate_execution_context(ROOT)
    plan = reset.freeze_plan(args.output)
    plan.update(version="dh116-power-live-palm-feedback-v1",
                preserved_scope="isolated both original mounted palm reactions with no task action")
    plan["budget"].update(mode="isolated_live_palm_feedback", sdk_steps=END_SDK, solver_steps=24, new_power_mesh_registrations=2)
    for path in (args.rays, *(args.rays.parent / f"{side}_base_link.json" for side in SIDES), *(folder / "result.json" for folder in (args.left_bake, args.right_bake)),
                 args.left_bake/"left.base_link.npz", args.right_bake/"right.base_link.npz"):
        plan["frozen_files"][str(path.resolve(strict=True))] = native.sha(path)
    digest = native.digest(plan)
    if args.approved_frozen_sha256 is None:
        rays = solver.checked_json(args.rays, solver.RAYS_SHA)["palms"]
        for side, bake in (("left", args.left_bake), ("right", args.right_bake)):
            solver.fixture(side, rays[side], bake)
        native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=digest, budget=plan["budget"])))
    else:
        if (args.approved_frozen_sha256 != digest or json.loads(args.plan.read_bytes()) != plan
                or args.output.exists()):
            raise ValueError("exact approved palm feedback plan and fresh output required")
        from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
        reset.execute(plan, acquire_operation=partial(run_feedback, profile=BackendProfile.load(ROOT), args=args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
