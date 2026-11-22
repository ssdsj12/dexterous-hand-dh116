"""One real resized supported-wire Reset inside an existing Isaac ownership scope."""
from functools import partial
from pathlib import Path
from uuid import uuid4

from .tabletop_supported_wire_specs import prepare_variant


def acquire_variant_reset(sim, owners, trace, output, *, project, profile, config, sdk_end,
                          variant_options, expected_seed_sha256, fixed_pregrasp=False, task="tripod"):
    """Bind a resized free rod to actual hands before the original shared registration."""
    from . import tabletop_dual_hand_scene_preparation as scene_api
    from . import tabletop_dual_pinch_builder as builder
    from .tabletop_deformable_bimanual_step import HandBinding
    from .tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode
    from .tabletop_deformable_grasp_runtime import FreeRodRuntime, load_candidate
    from .tabletop_deformable_r0_replay import encode_owned
    from .tabletop_deformx_end_supported_scene import make_support_sdf_grids
    from .tabletop_dual_hand_rod_runtime import DualHandRodRuntime, PADS
    from .tabletop_native_teacher_profile import profile_reader_scope
    from .tabletop_palm_mount import MountView
    from .tabletop_power_shared_rod import END_SUPPORT_NAMES
    from .tabletop_power_supported_task import assemble_supported_scene, guard_hand_supports
    from ropesim import sdf
    from scripts import audit_dh116_pinch_initial_pose as geometry
    from scripts.audit_dh116_certified_segment_distance import segment_hull_distance
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    from scripts.verify_dh116_dual_pinch_interval import diagnostic_reset_owners
    from scripts.verify_dh116_power_shared_rod import DT, SIDES, native

    scene = hands = hand = base = rod = mount_bindings = episode = value = None
    try:
        if native.clock(sim)["sdk_step"] != 0:
            raise ValueError("fresh App-owned variant scene required")
        prepared = scene_api.prepare_scene(project)
        cfg = load_candidate(project / "configs/dh116_tasks/flexible_pinch_candidate_v1.json", project)
        if fixed_pregrasp:
            from .tabletop_wire_holdout_resets import prepare_fixed_pregrasp_variant

            seed, active, nodes, support_poses, variant = prepare_fixed_pregrasp_variant(
                config, prepared["table"], cfg["rod"], **variant_options)
            if variant["grasp"] != task:
                raise ValueError("fixed pregrasp task differs from the native reset owner")
        else:
            if task != "tripod":
                raise ValueError("legacy resized reset requires Tripod")
            seed, active, nodes, support_poses, variant = prepare_variant(
                config, prepared["table"], cfg["rod"], **variant_options)
        if seed["sha256"] != expected_seed_sha256:
            raise ValueError("authored variant Reset seed differs from frozen CPU plan")
        cfg["rod"] = variant["rod"]
        support_scene = variant["scene"]["support_scene"]
        scene, paths, material = assemble_supported_scene(sim, project, prepared, seed, active, support_scene)
        owners.keep("scene", scene)
        hands = scene["hands"]
        if scene["wire"] is not None or scene["sensors"]:
            raise ValueError("one external rod and real static supports required")
        sim.reset()
        sdk = native.clock(sim)["sdk_step"]
        if sdk != 2:
            raise ValueError("variant scene must use exactly two SDK setup steps")
        settings = scene_api.validate_simulation_settings(sim, prepared)
        owners.keep("composer", builder._Composers(
            getattr(hand, attr) for hand in hands
            for attr in ("permanent_wrench_composer", "instantaneous_wrench_composer")))
        for side, hand in zip(SIDES, hands, strict=True):
            hand.update(DT)
            for suffix, value in (("hand", hand), ("view", hand.root_physx_view),
                                  ("asset_sim_view", hand._physics_sim_view),
                                  ("permanent_composer", hand.permanent_wrench_composer),
                                  ("instantaneous_composer", hand.instantaneous_wrench_composer)):
                trace.watch(side+"_"+suffix, value, owner=f"resized {task.title()} builder",
                            release="owned scope then SDK STOP", callbacks=True)
            value = None
        pad_bindings, pad_poses, view_paths = builder._initial_pads(prepared, hands)
        grids = {name: sdf.SdfGrid(**data) for name, data in scene_api.load_sdf_arrays(prepared).items()}
        run = f"{task}-wire-variant-"+uuid4().hex
        base = owners.keep("base_runtime", FreeRodRuntime(cfg, nodes,
            {name: grids["left."+name] for name in PADS}, {name: pad_poses["left."+name] for name in PADS},
            prepared["table"]["center_m"], prepared["table"]["size_m"], episode_uid=run+":construction"))
        rod = owners.keep("runtime", DualHandRodRuntime(base, bindings=pad_bindings,
            right_grids={name: grids["right."+name] for name in PADS},
            right_poses={name: pad_poses["right."+name] for name in PADS}, run_uuid=run,
            env_id=prepared["env_id"], require_pairs=True))
        trace.watch("runtime", rod, owner=f"resized {task.title()} builder", release="owned scope")
        trace.watch("rope", rod.rope, owner=f"resized {task.title()} runtime", release="owned scope")
        mount_bindings = owners.keep("hand_bindings", tuple(
            HandBinding(side, item, {prepared["env_id"]: view_paths[side]}, MountView(item))
            for side, item in zip(SIDES, hands, strict=True)))

        def record(row):
            record.count += 1
            native.write_new(output/f"{task}_reset_source_{record.count}.json", encode_owned(row))
        record.count = 0

        with profile_reader_scope(profile), bindings((geometry, "segment_hull_distance", segment_hull_distance)):
            episode = owners.keep("dual_episode", DualHandPhysicalEpisode(sim, rod, mount_bindings,
                backend_artifact=Path(profile.binary_path), backend_sha256=profile.binary_sha256,
                inspect_geometry=partial(builder._inspect_geometry, project, prepared, sim, hands, rod),
                physics_budget_end=sdk_end, record=record))
            trace.watch("episode", episode, owner=f"resized {task.title()} builder", release="owned scope")
            episode.reset(seed, episode_uid=run+":0", owner_resets=diagnostic_reset_owners())
            rows = list(builder._initial_geometry_rows(project, prepared, sim, hands, nodes, cfg["rod"]["radius_m"]))
        for row in rows:
            record(row)
        hulls = {f"{row['side']}.{group}.{name}": collider["world_vertices"]
                 for row in rows for group in ("source_colliders", "locked_pads")
                 for name, collider in row["evidence"][group].items()}
        if len(hulls) != 44:
            raise ValueError("34 source hulls and 10 cooked pads required")
        clearance = guard_hand_supports(hulls, support_scene)
        clock = rod.clock()
        if native.clock(sim)["sdk_step"] != sdk or clock["physics_step"] != sdk or clock["rod_tick"] != 0:
            raise ValueError("resized support data must belong to the native Reset")
        native.write_new(output/f"{task}_support_scene.json", dict(
            reset_clock=clock, support_paths=paths, supports=support_scene["supports"],
            support_material=material, settings=settings, world_hull_clearance=clearance,
            reset_seed_sha256=seed["sha256"], no_hidden_rod_pins=True, authored_rod_force=False,
            source_scope="variant_same_clock_geometry_diagnostic"))
        return dict(clock=clock, root_pose_wxyz=seed["reset_root_pose_w"]), dict(
            support_poses=support_poses,
            support_grids=dict(zip(END_SUPPORT_NAMES, make_support_sdf_grids(support_scene), strict=True)))
    finally:
        sim = owners = trace = scene = hands = hand = base = rod = mount_bindings = episode = value = None
