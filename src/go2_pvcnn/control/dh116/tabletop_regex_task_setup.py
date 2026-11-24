"""Reusable original-task startup over two real regex Articulations.

Call prepare_regex_task only inside an externally owned live App/Sim scope.
Imports stay CPU-safe; native authoring is lazy. The function performs exactly
one SDK reset (two initialization ticks), then restores the explicit seed with
zero physics ticks. Rod construction, contact qualification, control, safety
projection and reward remain with the existing task owners.
"""
from copy import deepcopy
import importlib
import math

import numpy as np
import torch

from .linear_coupling import JOINTS

SIDES = ("left", "right")
DT = 1 / 480
OWNED_NAMES = tuple(f"regex_{side}_{kind}" for side in SIDES
                    for kind in ("hand", "view", "permanent", "instantaneous"))


def prepare_regex_task(sim, owners, trace, project, prepared, archive, lanes, seed_height=None):
    """Return batch, effective seed, original rod nodes, bindings and GPU state.

    bindings is ONE static mapping, keyed by the archived native mesh names.
    Each row retains its source/cooked/SDF identity and actual body_index;
    prim_path names env0 and prim_paths records all actual environment paths.
    There are no per-lane Articulation objects or wrappers.

    initial contains only same-device tensors in public environment order.
    Seed tensors are immutable-by-contract restore inputs, not measured state.
    COM velocities come from actual get_link_velocities (the installed SDK's
    body_com_vel_w source). World COM pose uses raw link quaternion and actual
    get_coms local transform. No root/COM/link frames are interchanged.

    seed_height is optional and explicit: delta_z_m/evidence_sha256 are passed
    to the original derive_height_seed. No grasp receives an automatic height
    change. Original physical authoring remains unchanged even for a candidate;
    only the later zero-step explicit seed restore uses the derived pose.
    """
    assets = importlib.import_module(__package__ + ".tabletop_dual_hand_scene_preparation")
    pinch = importlib.import_module(__package__ + ".tabletop_dual_pinch_builder")
    task = importlib.import_module(__package__ + ".tabletop_power_supported_task")
    supports = importlib.import_module(__package__ + ".tabletop_end_supported_scene")
    support_material = importlib.import_module(__package__ + ".tabletop_deformx_end_supported_scene")
    from .tabletop_dual_hand_reset import _seed
    from .tabletop_regex_articulation import spawn_regex_articulations
    from .tabletop_shared_training_height import derive_height_seed
    from .tabletop_vec_isaac import isolate_scenes

    lanes = tuple(lanes)
    origins = np.asarray([origin for _, origin in lanes], dtype=np.float64)
    if (sim.current_time_step_index != 0 or not 2 <= len(lanes) <= 512
            or origins.shape != (len(lanes), 3) or not np.isfinite(origins).all()
            or np.any(origins[:, 2] != 0) or archive.style not in ("pinch", "tripod", "power")):
        raise ValueError("fresh 480Hz Sim and explicit 2..512 horizontal original task lanes required")
    args, layout, sources = assets.spawn_parameters(project, prepared)
    if archive.style == "pinch":
        pinch._paired_sources(project, archive.pair, prepared)
        seed = _seed(archive.pair["reset_seed"])
        rod_nodes = np.array(archive.pair["rod_seed"]["proposed_nodes_m"], dtype=np.float64, copy=True)
    else:
        scope = {"tripod": task.TRIPOD_SCOPE, "power": task.POWER_SCOPE}[archive.style]
        seed, active, rod_nodes, _ = task.prepare_initial_state(archive.task, prepared["table"], expected_scope=scope)
        seed = _seed(seed)
        if seed["sha256"] != archive.task_seed_sha256:
            raise ValueError("supported grasp seed differs from its independently authenticated receipt")
        rod_nodes = np.array(rod_nodes, dtype=np.float64, copy=True)
        for index, side in enumerate(SIDES):
            layout["hand_roots"][side] = np.asarray(seed["authored_root_seed_w"][index]).copy()
            layout["hand_roots"][side][2] -= assets.BUILDER_OFFSET_M
            layout["active_joints"][side] = active[index].copy()
    if rod_nodes.shape != (41, 3) or not np.isfinite(rod_nodes).all():
        raise ValueError("all 41 original finite material rod nodes required")
    original_seed_sha = seed["sha256"]
    height_proof = None
    effective_seed = seed
    if seed_height is not None:
        if not isinstance(seed_height, dict) or set(seed_height) != {"delta_z_m", "evidence_sha256"}:
            raise ValueError("explicit height delta and evidence SHA required")
        effective_seed, height_proof = derive_height_seed(seed, grasp=archive.style, **seed_height)
    effort = float(prepared["candidate_physics"]["hand"]["active_effort_limit_nm"])
    if not math.isfinite(effort) or effort <= 0:
        raise ValueError("original explicit positive candidate effort limit required")

    batch = owners.keep("regex_batch", spawn_regex_articulations(sim, args, layout, sources, origins))
    for env_id, scene in enumerate(batch.scenes):
        scoped = dict(prepared, prefix=scene["prefix"], env_id=env_id, table=dict(prepared["table"],
            center_m=(np.asarray(prepared["table"]["center_m"]) + scene["origin"]).tolist()))
        assets._bind_materials(sim.stage, scoped)
        if assets.inspect_source_stage(sim.stage, scoped)["original_colliders"] != 34:
            raise ValueError("all 34 original material-bound colliders per lane required")
        if archive.task is not None:
            support = deepcopy(archive.task["support_scene"])
            for box in support["supports"]:
                box["center_m"] = (np.asarray(box["center_m"]) + scene["origin"]).tolist()
            paths = supports.author_end_supports(sim.stage, scene["prefix"], support)
            material = prepared["candidate_physics"]["table"]
            support_material._bind_support_material(sim.stage, scene["prefix"], paths,
                                                     material["mu_s"], material["mu_k"])
    isolate_scenes(sim.stage, batch.scenes)
    for side, hand in zip(SIDES, batch.hands, strict=True):
        trace.watch(f"regex_{side}_hand", hand, owner="regex_task_setup",
                    release="owned scope then SDK STOP", callbacks=True)
    if sim.current_time_step_index != 0:
        raise ValueError("original USD authoring advanced physics")
    sim.reset()
    batch.bind()
    members = []
    for side in SIDES:
        hand = batch.sides[side].hand
        for kind, value in (("view", hand.root_physx_view),
                ("permanent", hand.permanent_wrench_composer), ("instantaneous", hand.instantaneous_wrench_composer)):
            trace.watch(f"regex_{side}_{kind}", value, owner="regex_task_setup",
                        release="owned scope then SDK STOP", callbacks=True)
            if kind != "view":
                members.append(value)
    owners.keep("composer", pinch._Composers(members))
    assets.validate_simulation_settings(sim, prepared)
    if sim.current_time_step_index != 2:
        raise ValueError("one SDK reset with exactly two initialization ticks required")

    initial = _seed_tensors(batch, effective_seed, origins)
    initial.update(restore_regex_seed(sim, batch, initial, active_effort_limit_nm=effort))
    bindings = _actual_bindings(batch, archive.mesh_plan["bindings"])
    positions, rotations = _mesh_poses(archive, bindings, initial)
    initial["mesh_positions_world_m"], initial["mesh_quaternions_xyzw"] = positions, rotations
    initial["rod_positions_w"] = torch.as_tensor(rod_nodes, device=initial["q"].device,
        dtype=torch.float32)[None] + initial["env_origins"].float()[:, None]
    batch.task_setup_metadata = dict(style=archive.style, source_seed_sha256=original_seed_sha,
        effective_seed_sha256=effective_seed["sha256"], height_derivation=height_proof,
        active_effort_limit_nm=effort, original_colliders_per_lane=34,
        sdk_initialization_steps=2, setup_physics_steps=0,
        native_setup_requalified=False, teacher_admission=False)
    return batch, effective_seed, rod_nodes, bindings, initial


def restore_regex_seed(sim, batch, initial, *, env_ids=None, active_effort_limit_nm):
    """Zero-step original state restore; peers keep targets and both composers.

    initial is the prepare_regex_task tensor packet. Public IDs are its stable
    contiguous task row identities. This restores only hand mechanics: caller
    must reset rod, mesh poses, reaction ownership, controller and history in
    its same-boundary task transaction. No rollback is attempted on SDK faults.
    """
    n = len(batch.env_ids)
    if batch.env_ids != tuple(range(n)):
        raise ValueError("stable contiguous public task environment identities required")
    device = batch.sides["left"].device
    ids = initial["env_ids"] if env_ids is None else torch.as_tensor(env_ids, device=device)
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("integer selected environment identities required")
    ids = ids.to(dtype=torch.long)
    for side in SIDES:
        batch.sides[side].rows(ids)  # Reject invalid/duplicate IDs before mutation.
    before = sim.current_time_step_index
    for index, side in enumerate(SIDES):
        batch.sides[side].reset(env_ids=ids,
            root_pose_wxyz=initial["seed_root_pose_w"].index_select(0, ids)[:, index],
            joint_pos=initial["seed_q"].index_select(0, ids)[:, index],
            joint_vel=initial["seed_qd"].index_select(0, ids)[:, index],
            active_effort_limit_nm=active_effort_limit_nm)
    batch.invalidate_readback(reason="original zero-step selected Reset state writes")
    sim.physics_sim_view.update_articulations_kinematic()
    batch.update(DT)
    if sim.current_time_step_index != before:
        raise ValueError("original hand restore unexpectedly advanced SDK physics")
    return capture_regex_state(batch, body_com_pose_b_xyzw=initial.get("body_com_pose_b_xyzw"))


def capture_regex_state(batch, *, body_com_pose_b_xyzw=None):
    """Own current device state, reusing the immutable actual local COM cache.

    Omit the cache once at Startup to acquire get_coms. Pass that same verified
    cache for subsequent captures/Reset: the original task never changes COM.
    World COM always uses the freshly measured raw link quaternion and pose.
    """
    states = batch.read()
    result = {key: torch.stack([states[side][key] for side in SIDES], dim=1)
        for key in ("q", "qd", "backend_targets", "root_pose_wxyz", "root_pose_xyzw",
                    "root_velocity", "link_pose_wxyz", "link_pose_xyzw")}
    result["body_com_velocity_w"] = torch.stack([states[side]["link_velocity"] for side in SIDES], 1)
    if body_com_pose_b_xyzw is None:
        body_com_pose_b_xyzw = torch.stack([
            batch.sides[side].hand.root_physx_view.get_coms().to(
                device=batch.sides[side].device).index_select(0, batch.sides[side].row_indices)
            for side in SIDES], 1)
    if (not isinstance(body_com_pose_b_xyzw, torch.Tensor)
            or body_com_pose_b_xyzw.shape != result["link_pose_xyzw"].shape
            or body_com_pose_b_xyzw.device != result["q"].device
            or body_com_pose_b_xyzw.dtype != result["link_pose_xyzw"].dtype):
        raise ValueError("same-device actual original local COM pose cache required")
    result["body_com_pose_b_xyzw"] = body_com_pose_b_xyzw
    # Preserve LiveGeometryReader's original raw-quaternion FP64 COM arithmetic.
    # Casting only the finished world position loses micron-scale lever arms
    # for lane origins tens of meters from zero.
    link, local = result["link_pose_xyzw"].double(), result["body_com_pose_b_xyzw"].double()
    xyz, w = link[..., 3:6], link[..., 6:7]
    offset = local[..., :3]
    cross = 2 * torch.linalg.cross(xyz, offset)
    position = link[..., :3] + offset + w * cross + torch.linalg.cross(xyz, cross)
    local_xyz, local_w = local[..., 3:6], local[..., 6:7]
    quaternion_xyz = w * local_xyz + local_w * xyz + torch.linalg.cross(xyz, local_xyz)
    quaternion_w = w * local_w - (xyz * local_xyz).sum(dim=-1, keepdim=True)
    result["body_com_pose_w"] = torch.cat((position, quaternion_w, quaternion_xyz), dim=-1)
    result["base_link_pose_w"] = torch.stack([
        result["link_pose_wxyz"][:, index, batch.sides[side].body_names.index("base_link")]
        for index, side in enumerate(SIDES)], 1)
    result["base_link_com_velocity_w"] = torch.stack([
        result["body_com_velocity_w"][:, index, batch.sides[side].body_names.index("base_link")]
        for index, side in enumerate(SIDES)], 1)
    return result


def _seed_tensors(batch, seed, origins):
    device = batch.sides["left"].device
    n = len(batch.env_ids)
    roots = torch.tensor(seed["authored_root_seed_w"], device=device, dtype=torch.float64)[None].repeat(n, 1, 1)
    origin = torch.tensor(origins, device=device, dtype=torch.float64)
    roots[..., :3] += origin[:, None]
    reset_roots = torch.tensor(seed["reset_root_pose_w"], device=device, dtype=torch.float64)[None].repeat(n, 1, 1)
    reset_roots[..., :3] += origin[:, None]
    q, qd = [], []
    for index, side in enumerate(SIDES):
        names = batch.sides[side].joint_names
        values = torch.zeros((n, 17), device=device, dtype=torch.float32)
        velocity = torch.zeros_like(values)
        ids = [names.index(name) for name in JOINTS]
        values[:, ids] = torch.tensor(seed["joint_positions"][index], device=device, dtype=torch.float32)
        velocity[:, ids] = torch.tensor(seed["joint_velocities"][index], device=device, dtype=torch.float32)
        q.append(values); qd.append(velocity)
    return dict(env_ids=torch.arange(n, device=device), env_origins=origin,
        root_seed_w=roots, common_lift_offset_m=torch.full((n,), seed["initial_offset_m"], device=device, dtype=torch.float64),
        seed_root_pose_w=reset_roots.float(), seed_q=torch.stack(q, 1), seed_qd=torch.stack(qd, 1),
        palm_offsets_m=torch.zeros((n, 2, 3), device=device, dtype=torch.float64),
        angles_rad=torch.zeros((n, 2, 3), device=device, dtype=torch.float64))


def _actual_bindings(batch, archived):
    from .tabletop_power_shared_rod import contact_links
    contact_links(archived)  # Original bilateral pad/palm/proximal inventory gate.
    result = deepcopy(archived)
    for key, row in result.items():
        side, name = row["side"], row["body_name"]
        owner = batch.sides[side]
        if key != side + "." + name or owner.body_names.index(name) != row["body_index"]:
            raise ValueError("actual original contact-link body ordering differs from source archive")
        rows = owner.row_indices.detach().cpu().tolist()  # Startup-only static identity resolution.
        paths = tuple(owner.hand.root_physx_view.link_paths[index][row["body_index"]] for index in rows)
        expected = tuple(prefix + "/" + side + "Hand/" + name for prefix in batch.prefixes)
        if paths != expected:
            raise ValueError("actual original contact link paths differ from authored task lanes")
        row["prim_path"], row["prim_paths"] = paths[0], paths
    return result


def _mesh_poses(archive, bindings, initial):
    entries = archive.mesh_plan["entries"]
    names = tuple(entry["name"] for entry in entries)
    if len(set(names)) != len(names) or not set(bindings).issubset(names):
        raise ValueError("one original native mesh slot for each actual hand contact binding required")
    device, n = initial["q"].device, len(initial["q"])
    position = torch.empty((n, len(entries), 3), device=device, dtype=torch.float32)
    rotation = torch.empty((n, len(entries), 4), device=device, dtype=torch.float32)
    pose_key = "table_pose" if archive.style == "pinch" else "static_pose"
    for slot, entry in enumerate(entries):
        if entry["name"] in bindings:
            row = bindings[entry["name"]]
            side = SIDES.index(row["side"])
            pose = initial["link_pose_xyzw"][:, side, row["body_index"]]
            position[:, slot], rotation[:, slot] = pose[:, :3], pose[:, 3:]
        else:
            xyz, quat = entry[pose_key]
            position[:, slot] = torch.tensor(xyz, device=device, dtype=torch.float32) + initial["env_origins"].float()
            rotation[:, slot] = torch.tensor(quat, device=device, dtype=torch.float32)
    if (not bool(torch.isfinite(position).all()) or not bool(torch.isfinite(rotation).all())
            or not torch.allclose(rotation.norm(dim=-1), torch.ones_like(rotation[..., 0]), atol=2e-6, rtol=0)):
        raise ValueError("finite actual hand and unchanged static native mesh poses required")
    return position, rotation
