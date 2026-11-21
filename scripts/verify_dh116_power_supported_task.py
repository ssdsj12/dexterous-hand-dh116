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
ACTION_PROBE = ROOT / "configs/dh116_tasks/flexible_power_supported_action_probe_v1.json"
NEAR_OPEN = ROOT / "configs/dh116_tasks/flexible_power_near_open_settle_v1.json"
PROXIMAL_ACTION = ROOT / "configs/dh116_tasks/flexible_power_proximal_action_v1.json"
native, reset, sources = shared.native, shared.reset, shared.sources


def acquire_supported_reset(sim, owners, trace, output, *, profile, config, sdk_end=shared.SDK_END, task="power"):
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
        POWER_SCOPE, TRIPOD_SCOPE, assemble_supported_scene, guard_hand_supports, prepare_initial_state,
    )
    from ropesim import sdf
    from scripts import audit_dh116_pinch_initial_pose as geometry
    from scripts.audit_dh116_certified_segment_distance import segment_hull_distance
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    from scripts.verify_dh116_dual_pinch_interval import diagnostic_reset_owners

    hand = hands = scene = original = pad_bindings = mount_bindings = episode = value = None
    scopes = {"power": POWER_SCOPE, "tripod": TRIPOD_SCOPE}
    if task not in scopes or native.clock(sim)["sdk_step"] != 0:
        raise ValueError("fresh App-owned supported task scene required")
    try:
        prepared = scene_api.prepare_scene(ROOT)
        seed, active, nodes, support_poses = prepare_initial_state(config, prepared["table"], expected_scope=scopes[task])
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
        run = f"{task}-supported-" + uuid4().hex
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
            native.write_new(output/f"{task}_reset_source_{record.count}.json", encode_owned(row))
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
        trace.checkpoint(f"{task}_supported_reset_complete")
        native.write_new(output/f"{task}_support_scene.json", dict(reset_clock=clock, support_paths=paths,
            supports=support_scene["supports"], support_material=material, settings=settings,
            world_hull_clearance=clearance, reset_seed_sha256=seed["sha256"],
            no_hidden_rod_pins=True, authored_rod_force=False, source_scope="same_clock_geometry_diagnostic"))
        return {"clock": clock, "root_pose_wxyz": seed["reset_root_pose_w"]}, dict(support_poses=support_poses,
            support_grids=dict(zip(END_SUPPORT_NAMES, make_support_sdf_grids(support_scene))))
    finally:
        sim = owners = trace = hands = hand = scene = original = episode = value = mount_bindings = None


def freeze(args):
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import prepare_initial_state
    from go2_pvcnn.control.dh116.tabletop_supported_rod_reference import load_parked_reference

    config = json.loads(CONFIG.read_bytes())
    probe = getattr(args,"verify_supported_action_handoff",False)
    near = getattr(args,"verify_near_open_settling",False)
    capture = getattr(args, "proximal_capture", None)
    bake = getattr(args, "proximal_bake", None)
    proximal_files = []
    action_reference = getattr(args, "proximal_action_reference", None)
    if capture is not None or bake is not None:
        if not near or capture is None or bake is None:
            raise ValueError("both proximal evidence inputs and the near-open diagnostic are required")
        _, proximal_files = shared.proximal_registration_inputs(capture, bake)
    settle = args.settle_without_hand_contact or probe or near
    sdk_end = shared.SDK_END
    if settle:
        experiment = json.loads(SETTLE.read_bytes())
        if (experiment["version"] != "dh116-power-support-settle-v1"
                or experiment["base_config_sha256"] != native.sha(CONFIG)
                or experiment["sdk_steps"] != 240
                or experiment["initial_hand_x_translation_m"] != .105):
            raise ValueError("independent bounded support settlement configuration required")
        sdk_end = experiment["sdk_steps"]
        if not near:
            for hand in config["hands"].values():
                hand["pose_wxyz"][0] += experiment["initial_hand_x_translation_m"]
    if near:
        candidate = json.loads(NEAR_OPEN.read_bytes())
        parent = getattr(args,"near_open_parent_receipt",None)
        screen = getattr(args,"near_open_screen",None)
        if parent is None or screen is None or native.sha(screen) != candidate["candidate_geometry_sha256"]:
            raise ValueError("exact native parent and near-open geometry evidence are required")
        load_parked_reference(parent, candidate["source_parked_outer_sha256"], 240)
        if (candidate["version"] != "dh116-power-near-open-settle-v1"
                or candidate["baseline_config_sha256"] != native.sha(CONFIG)
                or candidate["sdk_setup_and_measurement_steps"] != sdk_end
                or candidate["native_qualification"] is not False
                or set(candidate["candidate"]) != set(shared.SIDES)):
            raise ValueError("separate two-hand Power near-open Reset contract required")
        screen_rows = [json.loads(line) for line in screen.read_text().splitlines()]
        if sorted(row["side"] for row in screen_rows) != sorted(shared.SIDES):
            raise ValueError("exactly one certified geometry row is required for each hand")
        screened_by_side = {row["side"]: row["candidates"] for row in screen_rows}
        for side, specification in candidate["candidate"].items():
            if (specification not in screened_by_side[side] or specification["side"] != side
                    or specification["native_start_verified"] is not False
                    or specification["clearance_to_sampled_rod_tube_m"] <= .001
                    or specification["table_clearance_m"] < .0021
                    or specification["minimum_support_gap_m"] <= 0.):
                raise ValueError("original near-open geometry precheck is required for both hands")
            initial_xyz = [base+offset for base,offset in
                           zip(specification["target_pose_wxyz"][:3],specification["start_world_offset_m"],strict=True)]
            config["hands"][side] = dict(pose_wxyz=initial_xyz+specification["initial_quaternion_wxyz"],
                                           q_rad=specification["initial_q_rad"])
    action = None
    if action_reference is not None:
        if not proximal_files or sdk_end != 240:
            raise ValueError("proximal motion requires the settled near-open 17-slot scene")
        request = json.loads(PROXIMAL_ACTION.read_bytes())
        expected_action = [0.] * 24
        expected_action[12] = expected_action[15] = -.5
        if (request["version"] != "dh116-power-proximal-action-v1"
                or request["near_open_config_sha256"] != native.sha(NEAR_OPEN)
                or request["start_sdk_step"] != sdk_end or request["control_windows"] != 1
                or request["action24"] != expected_action):
            raise ValueError("exact original bounds and one independent 17-slot approach pulse required")
        action = request["action24"]
        row, reference_files = shared.proximal_action_boundary(action_reference, request["native_outer_sha256"])
        if row["rod"]["physics_step"] != sdk_end:
            raise ValueError("17-slot action reference has the wrong boundary")
        proximal_files.extend((PROXIMAL_ACTION, *reference_files))
        sdk_end += 8
    if probe:
        request = json.loads(ACTION_PROBE.read_bytes())
        action = request["action24"]
        expected_action = [0.]*24
        expected_action[12] = expected_action[15] = -.5
        if (request["version"] != "dh116-power-supported-action-probe-v1"
                or request["parked_config_sha256"] != native.sha(SETTLE)
                or request["control_windows"] != 1 or action != expected_action):
            raise ValueError("one independent 24D Power action probe required")
        sdk_end += 8
    task = load_flexible_grasp_task(ROOT/"configs/dh116_tasks/flexible_dual_hand_grasp_v1.json", ROOT)
    table = dict(center_m=task["scene"]["table_center_m"], size_m=task["scene"]["table_size_m"])
    seed, _, _, _ = prepare_initial_state(config, table)
    plan = reset.freeze_plan(args.output)
    plan.update(version="dh116-power-supported-contact-pilot-v1",
                preserved_scope=config["scope"], power_seed_sha256=seed["sha256"])
    mode = "power_supported_settle_without_hand" if settle else "power_supported_contact_pilot"
    if near:
        mode = ("power_supported_proximal_no_contact_settling" if proximal_files
                else "power_supported_near_open_no_contact_settling")
    if probe:
        mode = "power_supported_settle_and_action_probe"
    if action_reference is not None:
        mode = "power_supported_proximal_near_open_action_probe"
    if action is not None:
        plan["action24"] = action
    plan["budget"].update(mode=mode, sdk_steps=sdk_end, control_steps=int(action is not None),
                          solver_steps=(sdk_end-2)*shared.SUBSTEPS, new_power_mesh_registrations=6 if proximal_files else 4)
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
    if probe:
        files.append(ACTION_PROBE)
    if near:
        files.extend((NEAR_OPEN,parent,screen, *proximal_files))
    hashes = {str(path.resolve(strict=True)):native.sha(path) for path in files}
    for path, digest in hashes.items():
        if plan["frozen_files"].setdefault(path, digest) != digest:
            raise ValueError(f"conflicting frozen Power artifact: {path}")
    return plan, config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("output", "plan", "rays", "left-bake", "right-bake"):
        parser.add_argument("--"+flag, type=Path, required=True)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--settle-without-hand-contact", action="store_true")
    scope.add_argument("--verify-supported-action-handoff", action="store_true")
    scope.add_argument("--verify-near-open-settling", action="store_true")
    parser.add_argument("--near-open-parent-receipt",type=Path)
    parser.add_argument("--near-open-screen",type=Path)
    parser.add_argument("--proximal-capture", type=Path)
    parser.add_argument("--proximal-bake", type=Path)
    parser.add_argument("--proximal-action-reference", type=Path)
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
        args=args, sdk_end=end, require_no_hand_contact=(args.settle_without_hand_contact
            or args.verify_supported_action_handoff or args.verify_near_open_settling),
        action_after_settle=plan.get("action24"),
        supported_acquire=partial(acquire_supported_reset, config=config, sdk_end=end)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
