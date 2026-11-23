"""Live hand/rod exchange for the bounded Isaac scene audit."""
from copy import deepcopy
import json
import os

import numpy as np


def _live_contact_geometry(scene, bindings):
    from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import _owned
    from go2_pvcnn.control.dh116.tabletop_deformable_live_geometry import rotation_wxyz

    poses, centers = {}, {}
    for side, hand in zip(("left", "right"), scene["hands"], strict=True):
        view = hand.root_physx_view
        xyzw = _owned(view.get_link_transforms())
        local = _owned(view.get_coms())
        if xyzw.shape != (1, 23, 7) or local.shape != xyzw.shape or not np.isfinite([xyzw, local]).all():
            raise ValueError("finite original hand link and COM views required")
        for name, row in bindings.items():
            if row["side"] == side:
                i = row["body_index"]
                if hand.body_names[i] != row["body_name"] or view.link_paths[0][i] != row["prim_path"]:
                    raise ValueError("live lane contact link identity differs")
                xyz, quat = xyzw[0, i, :3].copy(), xyzw[0, i, 3:].copy()
                poses[name] = (xyz, quat)
                centers[name] = xyz + rotation_wxyz(quat[[3, 0, 1, 2]]) @ local[0, i, :3]
    return poses, centers


def translated_static_pose(style, entry, origin):
    """Translate the measured static collider pose into its own Isaac lane."""
    pose_key = {"pinch": "table_pose", "tripod": "static_pose", "power": "static_pose"}.get(style)
    pose = entry.get(pose_key)
    if pose is None:
        raise ValueError("grasp-specific archived static pose required")
    position, quaternion = pose
    return np.asarray(position) + origin, np.asarray(quaternion).copy()


def _build_rods(scenes, archive, nodes, owners):
    from ropesim import sdf
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
    from go2_pvcnn.control.dh116.tabletop_native_teacher_lane_layout import verify_translated_bindings
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import FreeRodRuntime
    from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import DualHandRodRuntime, PADS
    from go2_pvcnn.control.dh116.tabletop_power_shared_rod import (
        END_SUPPORT_NAMES, PALM_NAMES, PROXIMAL_NAMES, PowerSharedRodRuntime)
    from uuid import uuid4

    plan = archive.mesh_plan
    entries = {row["name"]: row for row in plan["entries"]}
    grids = {name: sdf.SdfGrid(values=row["values"], origin=row["origin"], spacing=row["spacing"])
             for name, row in entries.items() if name != "table"}
    run = uuid4().hex
    rods = owners.keep("batch_rods", [])
    for env_id, scene in enumerate(scenes):
        bindings = deepcopy(plan["bindings"])
        for name, row in bindings.items():
            row["prim_path"] = f"{scene['prefix']}/{row['side']}Hand/{row['body_name']}"
        poses, _ = _live_contact_geometry(scene, bindings)
        center, rotation = translated_static_pose(archive.style, entries["table"], scene["origin"])
        if not np.array_equal(rotation, [0., 0., 0., 1.]):
            raise ValueError("original axis-aligned table pose required")
        base = FreeRodRuntime(deepcopy(plan["cfg"]), nodes + np.asarray(scene["origin"], dtype=np.float32),
            {n: grids["left."+n] for n in PADS}, {n: poses["left."+n] for n in PADS},
            center, scene["layout"]["table_size"],
            episode_uid=f"{run}:{env_id}:construction")
        args = dict(bindings=bindings, right_grids={n: grids["right."+n] for n in PADS},
                    right_poses={n: poses["right."+n] for n in PADS},
                    run_uuid=run, env_id=env_id, rod_uid=f"rod:{env_id}", require_pairs=True)
        if archive.style == "pinch":
            rod = DualHandRodRuntime(base, **args)
        else:
            support_poses = {n: translated_static_pose(archive.style, entries[n], scene["origin"])
                             for n in END_SUPPORT_NAMES}
            rod = PowerSharedRodRuntime(base, **args, support_poses=support_poses,
                support_grids={n: grids[n] for n in END_SUPPORT_NAMES},
                palm_grids={n: grids[n] for n in PALM_NAMES},
                palm_poses={n: poses[n] for n in PALM_NAMES},
                proximal_grids={n: grids[n] for n in PROXIMAL_NAMES},
                proximal_poses={n: poses[n] for n in PROXIMAL_NAMES})
        if tuple(entries) != tuple(rod.mesh_names):
            raise ValueError("live solver slots do not match this grasp's archive")
        verify_translated_bindings(plan["bindings"], rod.bindings, scene["prefix"], _digest(plan["bindings"]))
        rod.rope.enable_contact_pair_capture(len(nodes)*len(entries)*12)
        rods.append(rod)
    return rods


def rebuild_reset_meshes(rope, style, entries, poses, origin, capacity):
    """Use the pinned clear/add Reset entry points and verify actual CUDA readback."""
    targets = [poses[row["name"]] if row["name"] in poses else
               translated_static_pose(style, row, origin) for row in entries]
    rope.clear_meshes()
    for slot, (row, (position, orientation)) in enumerate(zip(entries, targets, strict=True)):
        actual = rope.add_mesh(row["values"], row["origin"], row["spacing"],
            pos=np.asarray(position, dtype=np.float32), quat=np.asarray(orientation, dtype=np.float32),
            mu_s=row["mu_s"], mu_k=row["mu_k"])
        if type(actual) is not int or actual != slot:
            raise ValueError("rebuilt slot differs from original source order")
    snapshot = rope.mesh_state_snapshot()
    positions = np.asarray([v[0] for v in targets], dtype=np.float32)[None]
    rotations = np.asarray([v[1] for v in targets], dtype=np.float32)[None]
    rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)
    if (snapshot["device_state_source"] != "cuda_device_copy" or snapshot["mesh_count"] != len(entries)
            or snapshot["pending_set"] or snapshot["moving"] or snapshot["pair_ready"]
            or np.any(snapshot["pending_teleport"]) or snapshot["pair_capacity"] != capacity):
        raise ValueError("rebuilt reset mesh state is stale or incomplete")
    for phase in ("current", "target", "solver"):
        if (not np.array_equal(snapshot[phase+"_positions_world_m"], positions)
                or not np.allclose(snapshot[phase+"_quaternions_xyzw"], rotations, atol=2e-6, rtol=0)):
            raise ValueError("rebuilt reset mesh state differs from original pose")
    return snapshot


def reset_rod_fingerprint(rope, device):
    from go2_pvcnn.control.dh116.tabletop_material_director_readback import _copy_endpoint
    from go2_pvcnn.control.dh116.tabletop_mesh_reset_numeric_state import _copy_velocity
    from go2_pvcnn.control.dh116.tabletop_mesh_reset_order import _digest

    parameters = rope.parameter_snapshot()
    if type(rope.last_error) is not int or rope.last_error != 0:
        raise ValueError("zero actual solver error required")
    positions, orientations = _copy_endpoint(rope, device)
    velocities = _copy_velocity(rope, device)
    for array, shape in ((positions, (1, 41, 3)), (orientations, (1, 41, 4)),
                         (velocities, (1, 41, 3))):
        if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
            raise ValueError("finite typed rod state required at the Reset boundary")
    return _digest(dict(positions=positions, orientations=orientations,
                        velocities=velocities, parameters=parameters))


def _reset_rod(scene, rod, sdk, archive, node_count):

    poses, _ = _live_contact_geometry(scene, rod.bindings)
    before = rod.clock()
    rod.reset(physics_step=sdk, episode_uid=f"{rod.run_uuid}:{rod.env_id}:{before['reset_id']+1}",
              reset_id=before["reset_id"]+1, pad_poses=poses)
    device = str(scene["hands"][0].device)
    state = reset_rod_fingerprint(rod.rope, device)
    rod._needs_reset = True
    rebuild_reset_meshes(rod.rope, archive.style, archive.mesh_plan["entries"], poses,
                         scene["origin"], node_count*len(rod.mesh_names)*12)
    if reset_rod_fingerprint(rod.rope, device) != state:
        raise ValueError("mesh reconstruction changed actual node or material state")
    after = rod.clock()
    if (after["physics_step"] != sdk or after["rod_tick"] != 0
            or after["solver_time_s"] != before["solver_time_s"]
            or after["reset_id"] != before["reset_id"]+1):
        raise ValueError("independent zero-step rod Reset not confirmed")
    rod._needs_reset = False
    return rod.clock()


def coupled_probe(sim, scenes, archive, nodes, owners, reset_hand_lane, hold_step, output):
    """Hold the actual reset pose, route raw reactions one tick later and reset one lane."""
    import torch
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
    from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import write_composer_reactions
    from go2_pvcnn.control.dh116.tabletop_power_shared_rod import write_composer_reactions as write_extended

    rods = _build_rods(scenes, archive, nodes, owners)
    writer = write_composer_reactions if archive.style == "pinch" else write_extended
    device = str(sim.device)
    last = [None] * len(rods)
    rows = pairs = hand_pairs = 0
    for scene, rod in zip(scenes, rods, strict=True):
        _reset_rod(scene, rod, sim.current_time_step_index, archive, len(nodes))
    with (output/"coupled.jsonl").open("x") as stream:
        for step in range(3, 11):
            before = sim.current_time_step_index
            for scene, rod, reaction in zip(scenes, rods, last, strict=True):
                hands = scene["hands"]
                composers = {side:hand.permanent_wrench_composer
                             for side,hand in zip(("left", "right"), hands, strict=True)}
                if reaction is None:
                    if rod.clock()["rod_tick"] != 0:
                        raise ValueError("reaction may be absent only after the current lane Reset")
                    for composer in composers.values():
                        composer.reset()
                else:
                    k, routes = reaction
                    if k != before:
                        raise ValueError("queued reaction does not belong to the prior SDK tick")
                    writer(composers, routes, to_tensor=lambda x: torch.as_tensor(
                           x, dtype=torch.float32, device=hands[0].device))
                for hand in hands:
                    hand.write_data_to_sim()
            hold_step(step)
            for env_id, (scene, rod) in enumerate(zip(scenes, rods, strict=True)):
                for hand in scene["hands"]:
                    hand.update(1/480)
                poses, com = _live_contact_geometry(scene, rod.bindings)
                row = rod.step(poses, com, physics_step=step)
                pair = row["rod_mesh_contact_pairs"]
                # Existing solver runtime validates atomically read forces and pair records.
                if (pair is None or row["pair_capture_status"] != "validated_solver_samples"
                        or row["env_id"] != env_id or row["physics_step"] != step):
                    raise ValueError("same-lane same-tick complete native contacts required")
                capture = rod.rope.contact_pair_samples_v2()
                if (capture["overflow"] or capture["substeps"] != 12
                        or capture["mesh_count"] != len(rod.mesh_names)):
                    raise ValueError("native pair capture is incomplete")
                routes = row["hand_wrenches_com_world"]
                if any(np.linalg.norm(v["forces"], axis=-1).max() > 30. for v in routes.values()):
                    raise ValueError("original hand reaction safety threshold rejected a lane")
                recipients = {s:f"{scene['prefix']}/{s}Hand" for s in routes}
                previous = last[env_id]
                stream.write(json.dumps(encode_owned(dict(
                    env_id=env_id, sdk_step=step, rod_clock=rod.clock(), raw_pairs=capture,
                    recipients=recipients, reaction_source_sdk_step=previous[0] if previous else None,
                    issued_reaction=previous[1] if previous else None, reaction=routes,
                    force_residual_n=row["pair_force_residual_n"])), allow_nan=False)+"\n")
                rows += 1
                pairs += capture["attempted"]
                moving = [i for i,n in enumerate(rod.mesh_names) if n in rod.bindings]
                hand_pairs += int(np.isin(capture["indices"][:, 2], moving).sum())
                last[env_id] = step, routes
            if step == 6:
                unaffected = [(reset_rod_fingerprint(rod.rope, device), rod.clock())
                              for rod in rods[:-1]]
                reset_hand_lane(scenes[-1])
                receipt = _reset_rod(scenes[-1], rods[-1], step, archive, len(nodes))
                last[-1] = None
                for rod, (state, clock) in zip(rods[:-1], unaffected, strict=True):
                    if reset_rod_fingerprint(rod.rope, device) != state or rod.clock() != clock:
                        raise ValueError("selected Reset changed another live rod")
                if sim.current_time_step_index != step:
                    raise ValueError("selected Reset advanced the global SDK clock")
                stream.write(json.dumps(dict(event="isolated_lane_reset", rod_clock=receipt))+"\n")
            stream.flush()
            os.fsync(stream.fileno())
    return dict(rod_owners=len(rods), solver_steps=8*len(rods), native_pair_rows=rows,
                native_contact_pairs=pairs, native_hand_contact_pairs=hand_pairs, isolated_rod_reset=True)
