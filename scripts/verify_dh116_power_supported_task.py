#!/usr/bin/env python3
"""Measure an independent supported Power scene on the original two DH116 hands."""

import argparse
from functools import partial
import json
from pathlib import Path
from uuid import uuid4

from scripts import verify_dh116_power_shared_rod as shared

ROOT = shared.ROOT
CONFIG = ROOT / "configs/dh116_tasks/flexible_power_supported_contact_v1.json"
SETTLE = ROOT / "configs/dh116_tasks/flexible_power_supported_settle_v1.json"
native, reset, sources = shared.native, shared.reset, shared.sources


def acquire_supported_reset(sim, owners, trace, output, *, profile, config, sdk_end=shared.SDK_END):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene_api
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import HandBinding
    from go2_pvcnn.control.dh116.tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import FreeRodRuntime, load_candidate
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
    from go2_pvcnn.control.dh116.tabletop_deformx_end_supported_scene import make_support_sdf_grids
    from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import DualHandRodRuntime, PADS
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import profile_reader_scope
    from go2_pvcnn.control.dh116.tabletop_palm_mount import MountView
    from go2_pvcnn.control.dh116.tabletop_power_shared_rod import END_SUPPORT_NAMES
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import (
        assemble_supported_scene, guard_hand_supports, prepare_initial_state,
    )
    from ropesim import sdf
    from scripts import audit_dh116_pinch_initial_pose as geometry
    from scripts.audit_dh116_certified_segment_distance import segment_hull_distance
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    from scripts.verify_dh116_dual_pinch_interval import diagnostic_reset_owners

    hand = hands = scene = original = pad_bindings = mount_bindings = episode = value = None
    if native.clock(sim)["sdk_step"] != 0:
        raise ValueError("fresh App-owned Power scene required")
    try:
        prepared = scene_api.prepare_scene(ROOT)
        seed, active, nodes, support_poses = prepare_initial_state(config, prepared["table"])
        cfg = load_candidate(ROOT / "configs/dh116_tasks/flexible_pinch_candidate_v1.json", ROOT)
        support_scene = config["support_scene"]
        if (cfg["rod"]["gravity_world_m_s2"] != config["gravity_w_m_s2"]
                or cfg["rod"]["radius_m"] != support_scene["wire_radius_m"]):
            raise ValueError("retained rod geometry or gravity differs from the Power pilot")
        scene, paths, material = assemble_supported_scene(sim, ROOT, prepared, seed, active, support_scene)
        owners.keep("scene", scene)
        hands = scene["hands"]
        if scene["wire"] is not None or scene["sensors"]:
            raise ValueError("one external rod and no camera or rigid wire required")
        sim.reset()
        sdk = native.clock(sim)["sdk_step"]
        if sdk != 2:
            raise ValueError("native Power setup must use exactly 2 SDK steps")
        settings = scene_api.validate_simulation_settings(sim, prepared)
        owners.keep("composer", builder._Composers(
            getattr(item, attr) for item in hands
            for attr in ("permanent_wrench_composer", "instantaneous_wrench_composer")))
        for side, hand in zip(shared.SIDES, hands):
            hand.update(shared.DT)
            for suffix, value in (("hand", hand), ("view", hand.root_physx_view),
                                  ("asset_sim_view", hand._physics_sim_view),
                                  ("permanent_composer", hand.permanent_wrench_composer),
                                  ("instantaneous_composer", hand.instantaneous_wrench_composer)):
                trace.watch(side+"_"+suffix, value, owner="supported Power builder",
                            release="owned scope then SDK STOP", callbacks=True)
        pad_bindings, poses, view_paths = builder._initial_pads(prepared, hands)
        grids = {name: sdf.SdfGrid(**data) for name, data in scene_api.load_sdf_arrays(prepared).items()}
        run = "power-supported-" + uuid4().hex
        base = owners.keep("base_runtime", FreeRodRuntime(cfg, nodes,
            {name: grids["left."+name] for name in PADS}, {name: poses["left."+name] for name in PADS},
            prepared["table"]["center_m"], prepared["table"]["size_m"], episode_uid=run+":construction"))
        original = owners.keep("runtime", DualHandRodRuntime(base, bindings=pad_bindings,
            right_grids={name: grids["right."+name] for name in PADS},
            right_poses={name: poses["right."+name] for name in PADS}, run_uuid=run,
            env_id=prepared["env_id"], require_pairs=True))
        trace.watch("runtime", original, owner="supported Power builder", release="owned scope")
        trace.watch("rope", original.rope, owner="supported Power runtime", release="owned scope")
        mount_bindings = owners.keep("hand_bindings", tuple(
            HandBinding(side, item, {prepared["env_id"]: view_paths[side]}, MountView(item))
            for side, item in zip(shared.SIDES, hands)))

        def record(row):
            record.count += 1
            native.write_new(output/f"power_reset_source_{record.count}.json", encode_owned(row))
        record.count = 0

        with profile_reader_scope(profile), bindings((geometry, "segment_hull_distance", segment_hull_distance)):
            episode = owners.keep("dual_episode", DualHandPhysicalEpisode(sim, original, mount_bindings,
                backend_artifact=Path(profile.binary_path), backend_sha256=profile.binary_sha256,
                inspect_geometry=partial(builder._inspect_geometry, ROOT, prepared, sim, hands, original),
                physics_budget_end=sdk_end, record=record))
            trace.watch("episode", episode, owner="supported Power builder", release="owned scope")
            episode.reset(seed, episode_uid=run+":0", owner_resets=diagnostic_reset_owners())
            hulls = {}
            for row in builder._initial_geometry_rows(ROOT, prepared, sim, hands, nodes, cfg["rod"]["radius_m"]):
                record(row)
                for key in ("source_colliders", "locked_pads"):
                    for name, collider in row["evidence"][key].items():
                        hulls[f"{row['side']}.{key}.{name}"] = collider["world_vertices"]
        if len(hulls) != 44:
            raise ValueError("34 original source shapes and 10 locked cooked envelopes required")
        clearance = guard_hand_supports(hulls, support_scene)
        clock = original.clock()
        if native.clock(sim)["sdk_step"] != sdk or clock["physics_step"] != sdk or clock["rod_tick"] != 0:
            raise ValueError("support evidence must belong to the fresh same-clock Reset")
        trace.checkpoint("power_supported_reset_complete")
        native.write_new(output/"power_support_scene.json", dict(reset_clock=clock, support_paths=paths,
            supports=support_scene["supports"], support_material=material, settings=settings,
            world_hull_clearance=clearance, reset_seed_sha256=seed["sha256"],
            no_hidden_rod_pins=True, authored_rod_force=False, source_scope="same_clock_geometry_diagnostic"))
        return {"clock": clock}, dict(support_poses=support_poses,
            support_grids=dict(zip(END_SUPPORT_NAMES, make_support_sdf_grids(support_scene))))
    finally:
        sim = owners = trace = hands = hand = scene = original = episode = value = mount_bindings = None


def freeze(args):
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import prepare_initial_state

    config = json.loads(CONFIG.read_bytes())
    settle = args.settle_without_hand_contact
    sdk_end = shared.SDK_END
    if settle:
        experiment = json.loads(SETTLE.read_bytes())
        if (experiment["version"] != "dh116-power-support-settle-v1"
                or experiment["base_config_sha256"] != native.sha(CONFIG)
                or experiment["sdk_steps"] != 240
                or experiment["initial_hand_x_translation_m"] != .105):
            raise ValueError("independent bounded support settlement configuration required")
        sdk_end = experiment["sdk_steps"]
        for hand in config["hands"].values():
            hand["pose_wxyz"][0] += experiment["initial_hand_x_translation_m"]
    task = load_flexible_grasp_task(ROOT/"configs/dh116_tasks/flexible_dual_hand_grasp_v1.json", ROOT)
    table = dict(center_m=task["scene"]["table_center_m"], size_m=task["scene"]["table_size_m"])
    seed, _, _, _ = prepare_initial_state(config, table)
    plan = reset.freeze_plan(args.output)
    plan.update(version="dh116-power-supported-contact-pilot-v1",
                preserved_scope=config["scope"], power_seed_sha256=seed["sha256"])
    mode = "power_supported_settle_without_hand" if settle else "power_supported_contact_pilot"
    plan["budget"].update(mode=mode, sdk_steps=sdk_end,
                          solver_steps=(sdk_end-2)*shared.SUBSTEPS, new_power_mesh_registrations=4)
    src = ROOT / "src/go2_pvcnn/control/dh116"
    files = [CONFIG, Path(__file__), Path(shared.__file__), args.rays,
             src/"tabletop_power_supported_task.py", src/"tabletop_power_contact_evidence.py",
             src/"tabletop_power_shared_rod.py", src/"tabletop_end_supported_scene.py",
             src/"tabletop_deformx_end_supported_scene.py"]
    rays = sources.checked_json(args.rays, sources.RAYS_SHA)["palms"]
    for side, bake in (("left", args.left_bake), ("right", args.right_bake)):
        _, _, frozen, _ = sources.fixture(side, rays[side], bake)
        archive = args.rays.parent/f"{side}_base_link.json"
        sources.checked_json(archive, frozen["gpu_row_sha256"])
        files.extend((archive, bake/"result.json", bake/f"{side}.base_link.npz"))
    if settle:
        files.append(SETTLE)
    hashes = {str(path.resolve(strict=True)):native.sha(path) for path in files}
    for path, digest in hashes.items():
        if plan["frozen_files"].setdefault(path, digest) != digest:
            raise ValueError(f"conflicting frozen Power artifact: {path}")
    return plan, config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("output", "plan", "rays", "left-bake", "right-bake"):
        parser.add_argument("--"+flag, type=Path, required=True)
    parser.add_argument("--settle-without-hand-contact", dest="settle_without_hand_contact", action="store_true")
    parser.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    native.validate_execution_context(ROOT)
    plan, config = freeze(args)
    from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary

    assert_clean_import_boundary()
    digest = native.digest(plan)
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=digest, budget=plan["budget"])))
        return 0
    if (digest != args.approved_frozen_sha256 or json.loads(args.plan.read_bytes()) != plan
            or args.output.exists()):
        raise ValueError("exact frozen Power recipe and fresh isolated output required")
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile

    args.measure_contact_evidence = True
    end = plan["budget"]["sdk_steps"]
    reset.execute(plan, acquire_operation=partial(shared.run_shared, profile=BackendProfile.load(ROOT),
        args=args, sdk_end=end, require_no_hand_contact=args.settle_without_hand_contact,
        supported_acquire=partial(acquire_supported_reset, config=config, sdk_end=end)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
