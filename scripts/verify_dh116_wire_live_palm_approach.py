"""One native pulse from actual palm-local wire geometry and measured contact."""
import argparse
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from go2_pvcnn.control.dh116.tabletop_wire_live_palm_approach import SIDES, choose_pulse, wire_tube
from scripts import verify_dh116_wire_variant_isaac as parked


ROOT, native, lifecycle = parked.ROOT, parked.native, parked.lifecycle
SDK_END = parked.SDK_END + 8
VERSION = "dh116-wire-live-palm-approach-v1"


def bind_distal_start(plan, run, receipt_sha256):
    """Match a completed native Reset that used the same resized wire and pose."""
    from scripts.prepare_dh116_tripod_contact_seed import read_receipt
    from go2_pvcnn.control.dh116.tabletop_wire_first_distal import (
        APPROACH_TRAVEL_M, APPROACH_WINDOWS, HOLD_WINDOWS, LEFT_TRIPOD_STATIONS,
    )

    receipt_file = run/"outer.json"
    receipt = read_receipt(run, receipt_sha256)
    plan_file, state_file = (run/"native"/name for name in ("frozen_plan.json", "result.json"))
    previous_plan = json.loads(checked_file(plan_file, receipt["artifact_sha256"][plan_file.name]))
    result = json.loads(checked_file(state_file, receipt["artifact_sha256"][state_file.name]))
    if (previous_plan["version"] != parked.VERSION
            or previous_plan["variant_options"]["stations"] != {"left": 35, "right": 3}
            or any(previous_plan[key] != plan[key] for key in (
                "reset_seed_sha256", "spec_sha256", "variant_options", "task_config"))
            or result["failure"] is not None or result["physics_steps"] != parked.SDK_END
            or result["acquisition"]["solver_substeps"] != (parked.SDK_END-2)*12):
        raise ValueError("successful native parked result for the same wire and Reset required")
    plan["first_distal_approach"] = dict(
        approach_windows=APPROACH_WINDOWS, hold_windows=HOLD_WINDOWS, travel_m=APPROACH_TRAVEL_M,
        stations={key: list(value) for key, value in LEFT_TRIPOD_STATIONS.items()},
        prior_receipt_sha256=receipt_sha256)
    count = APPROACH_WINDOWS + HOLD_WINDOWS
    sdk = parked.SDK_END + count*8
    plan["budget"].update(mode="measured_left_distal_approach", control_steps=count,
                          sdk_steps=sdk, solver_steps=(sdk-2)*12, soft_wall_seconds=300)
    plan["preserved_scope"] = "original Tripod left distal approach and measured reaction hold"
    for path in (receipt_file, plan_file, state_file,
                 ROOT/"src/go2_pvcnn/control/dh116/tabletop_wire_continuous_clearance.py",
                 ROOT/"src/go2_pvcnn/control/dh116/tabletop_wire_first_distal.py",
                 ROOT/"scripts/prepare_dh116_tripod_contact_seed.py"):
        path = path.resolve(strict=True)
        digest = native.sha(path)
        if plan["frozen_files"].setdefault(str(path), digest) != digest:
            raise ValueError("inconsistent distal approach source: "+str(path))


def bind_right_relief(plan, source, expected_sha):
    """Freeze a CPU-screened unilateral translation/rotation and its original Reset."""
    from go2_pvcnn.control.dh116.tabletop_wire_first_distal import right_relief_action

    proof = json.loads(checked_file(source, expected_sha))
    samples = proof["samples"]
    minima = {
        key: min(row[key] for row in samples) if samples else float("nan")
        for key in ("rod_gap_m", "support_gap_m", "table_gap_m")
    }
    if (len(samples) != 13
            or any(not np.isfinite(row[key]) for row in samples for key in minima)
            or any(proof["minimum_"+key] != value for key, value in minima.items())
            or proof["scope"] != "sampled_real_reset_right_relief"
            or proof["reset_seed_sha256"] != plan["reset_seed_sha256"]
            or proof["spec_sha256"] != plan["spec_sha256"]
            or proof["minimum_rod_gap_m"] <= .00025
            or proof["minimum_support_gap_m"] <= 0.
            or proof["minimum_table_gap_m"] < .0021):
        raise ValueError("matching original right relief path clearance evidence required")
    windows = 18
    action = right_relief_action(proof["direction_world"], travel_m=proof["travel_m"],
                                 rotation_rad=proof["right_intrinsic_z_rad"], windows=windows)
    start = plan.pop("first_distal_approach")
    plan["right_relief"] = dict(action24=action.tolist(), screen_sha256=expected_sha,
                                 prior_receipt_sha256=start["prior_receipt_sha256"])
    plan["frozen_files"][str(source.resolve(strict=True))] = expected_sha
    sdk = parked.SDK_END + windows*8
    plan["budget"].update(mode="measured_right_relief_approach", control_steps=windows,
                          sdk_steps=sdk, solver_steps=(sdk-2)*12, soft_wall_seconds=300)
    plan["preserved_scope"] = "right transverse approach with measured intrinsic Z relief"


def bind_settled_thumb(plan, run, receipt_sha, path, path_sha):
    """Freeze one still-supported reference and the selected left thumb path."""
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    from go2_pvcnn.control.dh116.tabletop_wire_first_distal import SETTLED_THUMB_STATIONS, settled_left_action
    from scripts.prepare_dh116_tripod_contact_seed import read_receipt

    proof = json.loads(checked_file(path, path_sha))
    receipt = read_receipt(run, receipt_sha)
    files = {str(path.resolve(strict=True)): path_sha, str((run/"outer.json").resolve(strict=True)): receipt_sha}

    def prior(name):
        actual = (run/"native"/name).resolve(strict=True)
        digest = receipt["artifact_sha256"][name]
        value = json.loads(checked_file(actual, digest))
        files[str(actual)] = digest
        return value

    archived = prior("frozen_plan.json")
    result = prior("result.json")
    quiet = prior("wire_variant_quiet_window.json")
    sdk = archived["quiet_settlement_sdk"]
    name = f"tripod_exchange_{sdk}.json"
    boundary = decode_owned(prior(name))
    path_ref = proof["rail"]
    waypoints = np.asarray(path_ref["relative_world_waypoints_m"], dtype=float)
    positions, rotations = boundary["hand_link_poses"]["left.base_link"]
    start = np.r_[positions, np.roll(rotations, 1)]
    if (archived["version"] != "dh116-wire-variant-isaac-quiet-v1"
            or any(archived[key] != plan[key] for key in (
                "reset_seed_sha256", "spec_sha256", "variant_options", "task_config"))
            or result["failure"] is not None or result["physics_steps"] != sdk
            or result["acquisition"]["quiet_settlement"] != quiet
            or quiet["qualified"] is not True or quiet["both_supports_only"] is not True
            or quiet["last_sdk"] != sdk or proof["source"] != f"same_clock_isaac_{sdk}"
            or proof["frozen_native_outer_sha256"] != receipt_sha
            or proof["native_terminal_archive_sha256"] != receipt["artifact_sha256"][name]
            or proof["right_hand_kept_at_original_park"] is not True
            or path_ref["ok"] is not True or path_ref["order"] != "linear"
            or path_ref["endpoint_node"] != 35 or not -.00015 <= path_ref["endpoint_sdf_gap_m"] < 0.
            or not path_ref["minimum_all_hull_gap_m"] >= -.00015
            or not path_ref["minimum_noncontact_hull_gap_m"] > .00025
            or waypoints.shape != (2, 3) or np.any(waypoints[0])
            or not np.array_equal(start, path_ref["measured_root_start_wxyz"])
            or not np.allclose(start[:3]+waypoints[1], path_ref["goal_world_xyz_m"], rtol=0, atol=1e-12)):
        raise ValueError("screened thumb path must match the same successful quiet wire and hand boundary")
    parked.variant_settlement_budget(plan, sdk)
    drive, hold = 96, 2
    end = sdk + 8*(drive+hold)
    plan.update(version="dh116-wire-settled-left-thumb-approach-v1",
                preserved_scope="original compliant supported rod, one left thumb reaction and measured transfer")
    plan["budget"].update(mode="quiet_wire_then_left_thumb_transfer", sdk_steps=end,
                          solver_steps=(end-2)*12, control_steps=drive+hold)
    plan["first_distal_approach"] = dict(
        approach_windows=drive, action24=settled_left_action(waypoints[1], windows=drive).tolist(),
        stations={key: list(value) for key, value in SETTLED_THUMB_STATIONS.items()},
        quiet_archive=str((run/"native"/name).resolve(strict=True)),
        archive_sha256=receipt["artifact_sha256"][name], path_sha256=path_sha)
    for name in ("scripts/prepare_dh116_tripod_owned_approach.py", "scripts/prepare_dh116_tripod_contact_seed.py",
                 "src/go2_pvcnn/control/dh116/tabletop_wire_continuous_clearance.py",
                 "src/go2_pvcnn/control/dh116/tabletop_wire_first_distal.py"):
        actual = (ROOT/name).resolve(strict=True)
        files[str(actual)] = native.sha(actual)
    for key, value in files.items():
        if plan["frozen_files"].setdefault(key, value) != value:
            raise ValueError("inconsistent settled thumb source: " + key)


def bind_left_opposition(plan, proof_path, proof_sha):
    """Bind a proved static candidate to a separate bounded live recipe."""
    from go2_pvcnn.control.dh116.tabletop_wire_live_opposition import HOLD_WINDOWS, MODE, VERSION, index_closure

    proof = json.loads(checked_file(proof_path, proof_sha))
    path = np.asarray(proof["canonical_proposed_pivot_actions"], dtype=float)
    windows = len(path)
    if (plan["version"] != "dh116-wire-settled-left-thumb-approach-v1"
            or proof["version"] != "dh116-tripod-static-opposition-route-v1"
            or not proof["proved_static_hull_path"] or proof["native_measurement"] is not False
            or proof["grasp_qualified"] is not False or proof["issued_action_count"] != 0
            or proof["basis"]["wire_spec_sha256"] != plan["spec_sha256"]
            or proof["basis"]["station"] != plan["variant_options"]["stations"]["left"]
            or len(proof["pivot"]) != windows or not all(row["proved"] for row in proof["pivot"])
            or not proof["close"]["proved"] or proof["right_fixed_minimum_margin_m"] <= 0.
            or not 1 <= windows <= 256 or path.shape != (windows, 24)
            or not np.isfinite(path).all() or np.any(path[:, np.r_[:12, 15:18, 21:24]])):
        raise ValueError("a separate proved left opposition candidate on this settled wire is required")
    rotvec = np.asarray(proof["world_rotvec_rad"], dtype=float)
    if rotvec.shape != (3,) or not np.isfinite(rotvec).all() or not 0. < np.linalg.norm(rotvec) <= .16:
        raise ValueError("the approved candidate must use a bounded world pivot")
    increment = proof["index_increment_rad"]
    _, close = index_closure(increment)
    plan["left_opposition"] = dict(pivot_windows=windows, close_windows=close, hold_windows=HOLD_WINDOWS,
        approach_windows=plan["budget"]["control_steps"], world_rotvec_rad=rotvec.tolist(),
        index_increment_rad=increment, route_proof_sha256=proof_sha, basis=proof["basis"])
    extra = windows + close + HOLD_WINDOWS
    end = plan["budget"]["sdk_steps"] + 8*extra
    plan.update(version=VERSION, preserved_scope="same live supported rod and left two-pad opposition diagnostic")
    plan["budget"].update(mode=MODE, sdk_steps=end, solver_steps=(end-2)*12, soft_wall_seconds=900)
    plan["budget"]["control_steps"] += extra
    sources = {str(proof_path.resolve(strict=True)): proof_sha}
    for name in ("scripts/certify_dh116_tripod_opposition.py",
                 "src/go2_pvcnn/control/dh116/tabletop_wire_live_opposition.py"):
        file = (ROOT/name).resolve(strict=True)
        sources[str(file)] = native.sha(file)
    for file, digest in sources.items():
        if plan["frozen_files"].setdefault(file, digest) != digest:
            raise ValueError("inconsistent opposition source: " + file)


def validate_settled_handoff(reference, live, sdk):
    """Compare the same measured pose, joints and centerline used by the frozen path."""
    from scipy.spatial.transform import Rotation

    rods = reference["rod"], live["rod"]
    if any(row["physics_step"] != sdk or row["rod_tick"] != (sdk-2)*12 for row in rods):
        raise ValueError("quiet handoff requires the same solver and SDK boundary")
    nodes = [np.asarray(row["node_positions"], dtype=float) for row in rods]
    if any(p.shape != (1, 41, 3) or not np.isfinite(p).all() for p in nodes):
        raise ValueError("quiet handoff requires both intact 41-node centerlines")
    centerline = float(np.linalg.norm(nodes[0]-nodes[1], axis=-1).max())
    if centerline > 1e-6:
        raise ValueError("quiet centerline differs from the screened boundary")
    old = reference["hand_link_poses"]
    current = live["hand_link_poses"]
    if set(current) != set(old) or not set(old) >= {"left.base_link", "right.base_link"}:
        raise ValueError("quiet handoff original measured bodies required")
    position, angle, joint = 0., 0., 0.
    for link in old:
        p0, q0 = map(np.asarray, old[link])
        p1, q1 = map(np.asarray, current[link])
        if (p0.shape != (3,) or p1.shape != (3,) or q0.shape != (4,) or q1.shape != (4,)
                or not all(np.isfinite(value).all() for value in (p0, p1, q0, q1))):
            raise ValueError("finite quiet handoff body pose required")
        position = max(position, float(np.linalg.norm(p1-p0)))
        angle = max(angle, (Rotation.from_quat(q0).inv()*Rotation.from_quat(q1)).magnitude())
    if position > 1e-6 or angle > 1e-5:
        raise ValueError("quiet measured link poses differ from screened geometry")
    for side in SIDES:
        h0 = reference["native_joint_state"][side]
        h1 = live["native_joint_state"][side]
        if h0["names"] != h1["names"]:
            raise ValueError("quiet native joint identities differ")
        for key in ("position", "target"):
            x, y = np.asarray(h0[key]), np.asarray(h1[key])
            if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("finite native joint readback required")
            joint = max(joint, float(np.abs(x-y).max()))
    if joint > 1e-5:
        raise ValueError("quiet joint pose or backend target differs from screened geometry")
    return dict(node_max_error_m=centerline, link_max_error_m=position,
                link_max_error_rad=float(angle), joint_max_native_unit_error=joint)


def freeze(args):
    plan = parked.freeze(args)
    plan.update(version=VERSION, preserved_scope="one live transverse pre-contact wrist pulse")
    plan["budget"].update(mode="measured_wire_micro_approach", sdk_steps=SDK_END,
                          solver_steps=(SDK_END-2)*12, control_steps=1,
                          soft_wall_seconds=180)
    for path in (Path(__file__),
                 ROOT/"src/go2_pvcnn/control/dh116/tabletop_wire_live_palm_approach.py"):
        key, digest = str(path.resolve(strict=True)), native.sha(path)
        if plan["frozen_files"].setdefault(key, digest) != digest:
            raise ValueError("conflicting approach source: " + key)
    distal_parked = getattr(args, "distal_parked_run", None)
    distal_sha = getattr(args, "distal_outer_sha256", None)
    if (distal_parked is None) != (distal_sha is None):
        raise ValueError("distal approach needs both the prior parked run and its outer receipt SHA")
    if distal_parked is not None:
        bind_distal_start(plan, distal_parked, distal_sha)
    relief = getattr(args, "right_relief_screen", None)
    relief_sha = getattr(args, "right_relief_sha256", None)
    if (relief is None) != (relief_sha is None) or (relief is not None and distal_parked is None):
        raise ValueError("right relief requires a full clearance screen and matching parked receipt")
    if relief is not None:
        bind_right_relief(plan, relief, relief_sha)
    settled = tuple(getattr(args, name, None) for name in (
        "settled_run", "settled_outer_sha256", "settled_path", "settled_path_sha256"))
    if any(item is not None for item in settled):
        if any(item is None for item in settled) or distal_parked is not None:
            raise ValueError("settled thumb requires its own complete receipt and path evidence")
        bind_settled_thumb(plan, *settled)
    opposition = getattr(args, "opposition_proof", None)
    proof_sha = getattr(args, "opposition_proof_sha256", None)
    if (opposition is None) != (proof_sha is None):
        raise ValueError("opposition requires its source proof and SHA")
    if opposition is not None:
        bind_left_opposition(plan, opposition, proof_sha)
    if getattr(args, "compose_pivot_preview", False):
        if opposition is None:
            raise ValueError("composed pivot preview requires a frozen opposition plan")
        plan["left_opposition"]["composed_pivot_preview"] = True
    if getattr(args, "thumb_reseat", False):
        if not getattr(args, "compose_pivot_preview", False):
            raise ValueError("thumb re-seat requires the guarded composed opposition plan")
        plan["left_opposition"]["thumb_reseat"] = True
    return plan


def pack_geometry(sdk, poses, centers, clearance):
    """Convert measured PhysX XYZW to the existing float64 wrist projection input."""
    import torch

    roots = []
    for side in SIDES:
        position, xyzw = poses[f"{side}.base_link"]
        roots.append(np.r_[position, xyzw[[3, 0, 1, 2]]])
    return dict(physics_step=sdk, root_poses_w=torch.as_tensor(np.stack(roots), dtype=torch.float64)[None],
                clearance_m=torch.as_tensor(clearance, dtype=torch.float64).reshape(1, 2),
                pad_poses=poses, centers_of_mass=centers)


def acquire(sim, owners, trace, output, *, plan):
    import torch
    from go2_pvcnn.control.dh116.tabletop_batch_geometry import rotated_points
    from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import _owned
    from go2_pvcnn.control.dh116.tabletop_deformable_physical_exchange import FlexiblePhysicalExchange
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import prepare_scene
    from go2_pvcnn.control.dh116.tabletop_material_director_readback import _copy_endpoint
    from go2_pvcnn.control.dh116.tabletop_supported_wire_specs import observe_wire_pose_and_reaction, prepare_variant
    from go2_pvcnn.control.dh116.tabletop_power_shared_rod import END_SUPPORT_NAMES
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
    from scripts import verify_dh116_power_shared_rod as shared
    from scripts.screen_dh116_tripod_supported_reset import separation

    runtime = control = episode = None
    bindings = None
    distal = plan.get("first_distal_approach")
    right_relief = plan.get("right_relief")
    original_pulse = distal is None and right_relief is None
    end_sdk = plan["budget"]["sdk_steps"]
    start_sdk = plan.get("quiet_settlement_sdk", parked.SDK_END)
    settled_thumb = "quiet_settlement_sdk" in plan
    opposition = plan.get("left_opposition")
    phase = "approach"
    index_reaction = None
    motion_stop = None
    frames_sdk = prior_frames = live_frames = prior_nodes = live_nodes = None
    recent_load = None
    opposition_rows = 0
    max_opposed_substeps = 0
    loaded_opposed_ticks = 0
    index_peak_n = 0.
    thumb_peak_n = 0.
    try:
        settled = parked.acquire(sim, owners, trace, output, plan=plan)
        if settled.get("failure"):
            return settled
        clock = native.clock(sim)["sdk_step"]
        if clock != start_sdk:
            raise ValueError("approach requires the completed native parked exchange")
        if settled_thumb:
            from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

            if settled["quiet_settlement"]["qualified"] is not True:
                raise ValueError("left thumb requires the same live quiet window")
            saved = decode_owned(json.loads(checked_file(distal["quiet_archive"], distal["archive_sha256"])))
            live = decode_owned(json.loads((output/f"tripod_exchange_{start_sdk}.json").read_bytes()))
            verified = validate_settled_handoff(saved, live, start_sdk)
            native.write_new(output/"settled_thumb_handoff.json", verified)
        episode = owners.resources["dual_episode"]
        runtime = owners.resources["power_runtime"]
        bindings = {binding.side: binding for binding in episode._bindings}
        shapes = episode._exchange._read_geometry._shapes
        scene = json.loads(checked_file(plan["task_config"]["path"], plan["task_config"]["sha256"]))
        rod_path = ROOT/parked.BASE_ROD
        rod = json.loads(checked_file(rod_path, plan["frozen_files"][str(rod_path)]))["rod"]
        seed, _, _, _, variant = prepare_variant(scene, prepare_scene(ROOT)["table"], rod, **plan["variant_options"])
        if variant["spec_sha256"] != plan["spec_sha256"] or seed["sha256"] != plan["reset_seed_sha256"]:
            raise ValueError("approach cannot change the wire or hand Reset")
        spec = dict(rod=variant["rod"], length_m=variant["length_m"])
        initial = json.loads((output/"wire_variant_terminal_feedback.json").read_bytes())
        radius = initial["readback"]["measured_radius_m"]
        supports = variant["scene"]["support_scene"]
        tube_clock, tube, envelope = None, None, None
        geometry_row = None
        minimum_gap = np.inf
        stations_by_body = {}
        if not original_pulse:
            from scipy.spatial.transform import Rotation
            from scripts.audit_dh116_pinch_initial_pose import load_geometry
            from scripts.verify_dh116_supported_distal_transfer import target_shape_keys
            from scripts.prepare_dh116_pinch_closure_margin import sample_contact
            from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import load_sdf_arrays
            from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import SUBSTEPS, measure_supported_distal_contact
            from go2_pvcnn.control.dh116.tabletop_wire_continuous_clearance import WireClearance
            from go2_pvcnn.control.dh116.tabletop_wire_first_distal import FirstDistalContact, HeldPalmTarget

            stations = {} if distal is None else distal["stations"]
            if opposition:
                from go2_pvcnn.control.dh116.tabletop_wire_live_opposition import (
                    INDEX, THUMB, IndexReaction, choose_thumb_reseat, close_action_joint_names, compose_projected_pivot,
                    guard_prediction, index_closure,
                    left_loads, ready_after_thumb,
                )
                from scripts.audit_dh116_pinch_initial_pose import path_displacement_bound
                from scripts.certify_dh116_tripod_opposition import pivot_commands

                stations = {**stations, INDEX: (34, 35, 36)}
            sources = target_shape_keys(runtime.bindings, shapes, stations)
            stations_by_body = {(0, runtime.bindings[link]["body_index"]):link for link in stations}
            if set(stations_by_body) != sources or len(shapes) != 34:
                raise ValueError("complete original source hulls and explicit distal stations required")
            cooked = []
            for side_index, side in enumerate(SIDES):
                pads = load_geometry(ROOT, side=side)["cooked_pads"]
                if len(pads) != 5:
                    raise ValueError("each hand must supply five archived cooked pad hulls")
                for name, mesh in pads.items():
                    binding = runtime.bindings[side+"."+name]
                    cooked.append((side_index, binding["body_index"], mesh["vertices"]))
            shapes = tuple(shapes) + tuple(cooked)
            grids = load_sdf_arrays(prepare_scene(ROOT))


        def geometry():
            nonlocal tube_clock, tube, envelope, geometry_row, minimum_gap
            nonlocal frames_sdk, prior_frames, live_frames
            sdk = native.clock(sim)["sdk_step"]
            rod_row = runtime._base.last
            before = runtime.clock()
            if before["physics_step"] not in (sdk-1, sdk) or rod_row["physics_step"] != before["physics_step"]:
                raise ValueError("rod geometry must use the present or immediately preceding SDK tick")
            if before != tube_clock:
                if original_pulse:
                    tube = wire_tube(rod_row["node_positions"], radius_m=radius)
                else:
                    envelope = WireClearance(rod_row["node_positions"], radius_m=radius)
                tube_clock = before
            poses, centers, clearance = shared.measure_links(sim, bindings, runtime.bindings, shapes,
                                                             sdk=sdk, support_scene=supports)
            frames = []
            for side in SIDES:
                transform = _owned(bindings[side].hand.root_physx_view.get_link_transforms())
                if transform.shape != (1, 23, 7):
                    raise ValueError("complete DH116 pose readback required")
                frames.append(transform[0][:, [0, 1, 2, 6, 3, 4, 5]])
            if opposition and frames_sdk != sdk:
                prior_frames, live_frames = live_frames, np.stack(frames)
                frames_sdk = sdk
            pad_sdf = {}
            if original_pulse:
                gap = min(separation(rotated_points(frames[hand][body], vertices), *tube)
                          for hand, body, vertices in shapes)
                if not np.isfinite(gap) or gap <= .00025:
                    native.write_new(output/f"wire_clearance_stop_{sdk}.json", dict(
                        physics_step=sdk, rod_source_physics_step=before["physics_step"],
                        minimum_rod_clearance_m=gap))
                    raise ValueError("original non-contact rod clearance failed")
            else:
                gap = np.inf
                selected_bounds = {}
                for i, (hand, body, vertices) in enumerate(shapes):
                    link = stations_by_body.get((hand, body))
                    enabled = link is not None and (not opposition or link != INDEX or phase in ("close", "hold"))
                    selected = enabled and sdk > start_sdk
                    guard = envelope.certify(rotated_points(frames[hand][body], vertices), selected=selected)
                    if not guard["passed"]:
                        native.write_new(output/f"wire_clearance_stop_{sdk}.json", dict(
                            physics_step=sdk, rod_source_physics_step=before["physics_step"],
                            hand=SIDES[hand], hull="source" if i < 34 else "cooked",
                            link=bindings[SIDES[hand]].hand.body_names[body], **guard))
                        raise ValueError("original radius-aware continuous hull clearance failed")
                    gap = min(gap, guard["lower_m"])
                    if enabled:
                        selected_bounds[link] = min(selected_bounds.get(link, np.inf), guard["lower_m"])
                for link, bound in selected_bounds.items():
                    xyz, quat = poses[link]
                    transform = np.eye(4)
                    transform[:3,:3] = Rotation.from_quat(quat).as_matrix()
                    transform[:3,3] = xyz
                    local = sample_contact(grids[link], transform, rod_row["node_positions"][0],
                                           radius=radius, dtype=np.float64)
                    pad_sdf[link] = local
                    observed = local["minimum_gap_m"]
                    floor = .00025 if sdk == start_sdk else -.00015
                    if (observed is None and bound <= .00025) or (observed is not None and observed < floor):
                        native.write_new(output/f"wire_pad_sdf_stop_{sdk}.json", dict(
                            physics_step=sdk, rod_source_physics_step=before["physics_step"],
                            selected_link=link, minimum_gap_m=observed, clearance_floor_m=floor))
                        raise ValueError("original selected distal SDF bound failed")
            if native.clock(sim)["sdk_step"] != sdk or runtime.clock() != before:
                raise ValueError("clearance readback should not advance either clock")
            minimum_gap = min(minimum_gap, gap)
            geometry_row = dict(physics_step=sdk, poses=poses, rod_source_physics_step=before["physics_step"],
                                minimum_rod_clearance_m=gap, pad_sdf=pad_sdf, hull_count=len(shapes))
            return pack_geometry(sdk, poses, centers, clearance)

        def after_sdk(previous):
            row = geometry()
            if row["physics_step"] != previous["physics_step"] + 1:
                raise ValueError("expected completed SDK before matching solver step")
            return dict(row, geometry_phase="sdk_complete_rod_pending",
                        rod_source_physics_step=previous["physics_step"])

        def feedback(geom):
            before = runtime.clock()
            live_nodes, _ = _copy_endpoint(runtime.rope, "cuda:0")
            raw = runtime.rope.parameter_snapshot()
            if (before != runtime.clock() or before["physics_step"] != geom["physics_step"]
                    or before["physics_step"] != geom["rod_source_physics_step"]):
                raise ValueError("palm, reaction and parameter snapshots require the same physics tick")
            exchange = dict(rod=runtime._base.last, sdk_hand_geometry=geom,
                            material_director=dict(identity=before, node_positions_m=live_nodes,
                                                   raw_native_parameters=raw))
            result = observe_wire_pose_and_reaction(
                exchange, spec, grasp="tripod", grasp_material_ids=plan["variant_options"]["stations"])
            if result["spec_sha256"] != plan["spec_sha256"]:
                raise ValueError("approach readback changed the frozen wire specification")
            return result

        root = torch.as_tensor(seed["reset_root_pose_w"], dtype=torch.float64)[None]
        geometry()
        start_geometry = geometry_row
        measured = feedback(start_geometry)
        contact_owner = None
        if original_pulse:
            action = choose_pulse(measured, start_geometry)
            first, last = ACTION_CONTRACT["slices"]["world_translation_velocity"]
            if not np.linalg.norm(action[first:last].reshape(2, 3), axis=1).all():
                raise ValueError("both unloaded hands require a measured approach direction")
        elif right_relief is not None:
            action = np.asarray(right_relief["action24"], dtype=float)
        else:
            if not measured["reaction_valid"] or any(
                    pad["mean_normal_n"] > 0. for side in measured["measured_hands"].values()
                    for pad in side["reaction"].values()):
                raise ValueError("left distal approach requires zero initial hand contact")
            previous = dict(physics_step=start_sdk, rod=runtime._base.last)
            if settled_thumb:
                action = np.asarray(distal["action24"], dtype=float)
                contact_owner = FirstDistalContact(previous, approach=action)
            else:
                contact_owner = FirstDistalContact(previous)
                action = np.zeros(ACTION_CONTRACT["size"])
                action[14] = distal["travel_m"]*60/(distal["approach_windows"]*ACTION_CONTRACT["translation_rate_m_s"])
        control = owners.keep("power_control_exchange", FlexiblePhysicalExchange(
            sim, runtime, [bindings[side] for side in SIDES], root_seed_w=root,
            read_geometry=geometry, read_geometry_after_sdk=after_sdk,
            physics_budget_end=end_sdk, continue_supported_rod=True))
        trace.watch("power_control_exchange", control, owner="wire pose and reaction pulse", release="owned scope")
        native.write_new(output/"wire_pulse_request.json", encode_owned(
            dict(geometry=start_geometry, reaction=measured, action24=action)))
        held_palm = None
        if not original_pulse:
            held_palm = HeldPalmTarget("left" if right_relief is not None else "right",
                                       root[0, 0 if right_relief is not None else 1])

        transfer_complete = "settled_thumb_reaction_delivered"

        def observe(row):
            nonlocal prior_nodes, live_nodes, recent_load
            nonlocal opposition_rows, max_opposed_substeps, loaded_opposed_ticks, index_peak_n, thumb_peak_n
            sdk = row["physics_step"]
            if geometry_row["physics_step"] != sdk or row["reaction_source_physics_step"] != sdk-1:
                raise ValueError("geometry and reaction transport clocks do not match")
            native.write_new(output/f"wire_pulse_step_{sdk}.json", encode_owned(
                dict(row, sdk_hand_geometry=geometry_row)))
            if held_palm is not None and not held_palm.observe(row["action_readback"]):
                return f"{held_palm.side}_target_changed_during_unilateral_approach"
            if settled_thumb:
                from go2_pvcnn.control.dh116.tabletop_palm_mount import ANGULAR_JOINT_NAMES, JOINT_NAMES

                pre, post = (row["action_readback"][key]["hands"]["left"] for key in ("pre", "post"))
                if phase == "pivot":
                    moving = JOINT_NAMES + ANGULAR_JOINT_NAMES
                elif phase == "close":
                    issued = row["action_readback"]["request"]["action24"][0]
                    moving = close_action_joint_names(issued, allow_thumb_reseat=opposition.get("thumb_reseat", False))
                else:
                    moving = JOINT_NAMES if phase == "approach" else ()
                held = [i for i, name in enumerate(pre["joint_names"]) if name not in moving]
                if (not np.array_equal(pre["backend_targets"][:, held], post["backend_targets"][:, held])
                        or int(row["wrist_projection"]["projection_reasons"][0, 0])):
                    return "settled_left_finger_or_wrist_target_changed"
            if contact_owner is not None:
                evidence = measure_supported_distal_contact(row["rod"], grasp="tripod")
                native.write_new(output/f"wire_distal_evidence_{sdk}.json", evidence)
                if evidence["wrong_side_contacts"] or evidence["unexpected_hand_contacts"]:
                    return "unexpected_distal_station"
                stop = contact_owner.observe(row)
                if opposition and stop is None:
                    prior_nodes, live_nodes = live_nodes, row["rod"]["node_positions"].copy()
                    recent_load = left_loads(evidence["per_hand"]["left"]["per_substep"])
                    if phase != "approach":
                        index_reaction.observe(sdk, recent_load, row["issued_composer_wrench"]["left"])
                        opposition_rows += 1
                        max_opposed_substeps = max(max_opposed_substeps, recent_load["opposed_substeps"])
                        loaded_opposed_ticks += int(recent_load["loaded_opposed_substeps"] == 12)
                        index_peak_n = max(index_peak_n, recent_load["index_peak_n"])
                        thumb_peak_n = max(thumb_peak_n, recent_load["thumb_peak_n"])
                        native.write_new(output/f"wire_opposition_step_{sdk}.json",
                            dict(sdk=sdk, phase=phase, loads=recent_load,
                                 index_contact_sdk=index_reaction.first_contact_sdk,
                                 index_transfer_sdk=index_reaction.first_transfer_sdk))
                if (settled_thumb and stop is None and row["guard_failure"] is None
                        and contact_owner.first_transfer_sdk == sdk):
                    native.write_new(output/"settled_thumb_transfer.json", encode_owned(dict(
                        first_contact_sdk=contact_owner.first_contact_sdk, first_transfer_sdk=sdk,
                        first_pairs=contact_owner.first_pairs, issued_wrench=row["issued_composer_wrench"],
                        teacher_admission=False, grasp_qualified=False)))
                    if not opposition:
                        return transfer_complete
                return stop
            rod_row = row["rod"]
            pairs = rod_row["rod_mesh_contact_pairs"]["pairs"]
            if any(pair["mesh_name"] in runtime.bindings for pair in pairs):
                return "first_hand_reaction"
            if right_relief is not None and any(
                    np.any(hand[key]) for hand in rod_row["hand_wrenches_com_world"].values()
                    for key in ("forces", "torques")):
                return "unexpected_hand_wrench"
            wrenches = rod_row["external_support_wrenches_world"]
            if (np.linalg.norm(wrenches["table"][:3]) > 1e-6
                    or any(wrenches[name][2] >= 0. for name in END_SUPPORT_NAMES)):
                return "unexpected_support_transition"
            return None

        reserved_controls = opposition["approach_windows"] if opposition else plan["budget"]["control_steps"]
        n_controls = 0
        for i in range(reserved_controls):
            command = action if distal is None or i < distal["approach_windows"] else np.zeros_like(action)
            result = control.control(torch.as_tensor(command)[None], observe=observe,
                select_substep_action=None if contact_owner is None else contact_owner.select)
            n_controls += 1
            if settled_thumb and result["termination_reason"] == transfer_complete:
                break
            if result["termination_reason"] or result["physical_samples"] != 8:
                raise ValueError(f"approach stopped: {result['termination_reason']}")
            if opposition and ready_after_thumb(contact_owner.first_transfer_sdk, contact_owner.previous):
                break

        probe_result = None
        if opposition:
            previous = contact_owner.previous
            if not ready_after_thumb(contact_owner.first_transfer_sdk, previous):
                raise ValueError("opposition needs live thumb transfer and a whole control boundary")
            opposition_start = previous["physics_step"]
            geometry()
            hand = previous["action_readback"]["post"]["hands"]["left"]
            active = hand["q"][0, hand["active_indices"]]
            left = load_geometry(ROOT, side="left")
            close_action, close_windows = index_closure(opposition["index_increment_rad"])
            index_delta = close_action[:6] * ACTION_CONTRACT["joint_rate_rad_s"] / 480
            travel_bounds = [path_displacement_bound(left, active, index_delta, cooked=cooked)
                             for cooked in (False, True)]
            finger_travel = {i: travel_bounds[int(i >= 34)][bindings["left"].hand.body_names[body]]
                             for i, (side, body, _) in enumerate(shapes) if side == 0}
            table = prepare_scene(ROOT)["table"]
            table_top = table["center_m"][2] + table["size_m"][2]/2
            boxes = [(np.asarray(box["center_m"])-np.asarray(box["size_m"])/2,
                      np.asarray(box["center_m"])+np.asarray(box["size_m"])/2)
                     for box in supports["supports"]]
            idx = runtime.bindings[INDEX]["body_index"]
            index_reaction = IndexReaction(body_id=idx)
            close_start = None
            position, rotation = geometry_row["poses"]["left.base_link"]
            projection = previous["action_readback"]["submission"]
            _, _, pivot_actions = pivot_commands(np.r_[position, np.roll(rotation, 1)],
                projection["projected_root_poses"][0, 0, 3:], projection["projected_angles"][0, 0],
                live_nodes[0, plan["variant_options"]["stations"]["left"]],
                np.asarray(opposition["world_rotvec_rad"]), windows=opposition["pivot_windows"])

            def preview_opposition(candidate, *, reseat_geometry=None):
                selected = {key for key, link in stations_by_body.items() if link in contact_owner.stations}

                def projected_pivot():
                    live = geometry() if reseat_geometry is None else reseat_geometry
                    targets = contact_owner.previous["wrist_projection"]
                    return compose_projected_pivot(candidate, root_seed_w=root,
                        measured_base_w=live["root_poses_w"], clearance_m=live["clearance_m"],
                        palm_offsets_m=targets["palm_offsets_m"], angles_rad=targets["projected_angles"],
                        allow_translation_only=reseat_geometry is not None)

                if reseat_geometry is not None and projected_pivot() is None:
                    return dict(passed=False, reason="thumb_wrist_projection")
                return guard_prediction(action=candidate, frames=live_frames, before_frames=prior_frames,
                    nodes=live_nodes, before_nodes=prior_nodes, shapes=shapes, selected=selected,
                    left_palm=geometry_row["poses"]["left.base_link"][0],
                    finger_travel=finger_travel if phase == "close" and reseat_geometry is None else {},
                    boxes=boxes, table_top=table_top, radius=radius,
                    composed_pivot=projected_pivot if phase == "pivot"
                    and opposition.get("composed_pivot_preview") else None)

            def translated_thumb_gap(translation):
                xyz, quat = geometry_row["poses"][THUMB]
                transform = np.eye(4)
                transform[:3, :3] = Rotation.from_quat(quat).as_matrix()
                transform[:3, 3] = xyz + translation
                sample = sample_contact(grids[THUMB], transform, live_nodes[0], radius=radius, dtype=np.float64)
                if sample["minimum_node_id"] in contact_owner.stations[THUMB]:
                    return sample["minimum_gap_m"]
                return None

            def full_thumb_load():
                return recent_load["thumb_loaded_substeps"] == SUBSTEPS

            def select_opposition(proposed, clock):
                nonlocal motion_stop
                sdk = clock["physics_step"]
                if sdk != frames_sdk or sdk != contact_owner.previous["physics_step"]:
                    raise ValueError("opposition prediction must use the live pre-command SDK and solver endpoint")
                hold = phase == "hold" or motion_stop is not None or index_reaction.first_contact_sdk is not None
                if hold:
                    return torch.zeros_like(proposed), dict(paused=True, reason="retain_opposition_targets")
                candidate = proposed[0].numpy()
                if phase == "close" and opposition.get("thumb_reseat"):
                    if np.any(candidate[12:15]):
                        if full_thumb_load():
                            return torch.zeros_like(proposed), dict(paused=True, reason="thumb_load_restored")
                        if np.any(candidate[np.r_[:12, 15:24]]):
                            raise ValueError("thumb re-seat only permits left wrist translation")
                        live = geometry()
                        gap = translated_thumb_gap(np.zeros(3))
                        shift = candidate[12:15] * ACTION_CONTRACT["translation_rate_m_s"] / 480
                        next_gap = translated_thumb_gap(shift)
                        advances = (gap is not None and next_gap is not None
                                    and gap > -2e-6 and -.00015 <= next_gap < gap)
                        if not advances:
                            return torch.zeros_like(proposed), dict(paused=True, reason="thumb_sdf_no_improvement")
                        preview = preview_opposition(candidate, reseat_geometry=live)
                        return (proposed if preview["passed"] else torch.zeros_like(proposed),
                                dict(paused=not preview["passed"], reason="thumb_wrist_reseat",
                                     gap_m=gap, next_gap_m=next_gap, preview=preview))
                    if not full_thumb_load() or not np.any(candidate):
                        return torch.zeros_like(proposed), dict(paused=True, reason="await_full_thumb_load")
                preview = preview_opposition(candidate)
                if not preview["passed"]:
                    motion_stop = dict(sdk=sdk, phase=phase, preview=preview)
                return (torch.zeros_like(proposed) if motion_stop else proposed,
                        dict(paused=motion_stop is not None, reason="predicted_geometry_guard" if motion_stop
                            else "measured_one_step_opposition_preview", preview=preview))

            if recent_load["thumb_loaded_substeps"] == 0:
                motion_stop = dict(sdk=opposition_start, phase=phase, reason="no_thumb_load_at_boundary")
            batches = (("pivot", pivot_actions), ("close", (close_action,)*close_windows),
                       ("hold", (np.zeros(ACTION_CONTRACT["size"]),)*opposition["hold_windows"]))
            reseat_windows = 0
            for stage, batch in batches:
                if motion_stop and not index_reaction.deliver_after_stop(stage):
                    break
                phase = stage
                if stage == "close":
                    close_start = native.clock(sim)["sdk_step"]
                    contact_owner.stations = stations
                for additional in batch:
                    if stage == "close" and opposition.get("thumb_reseat") and not full_thumb_load():
                        live = geometry()
                        if geometry_row["rod_source_physics_step"] != contact_owner.previous["physics_step"]:
                            raise ValueError("thumb re-seat proposal requires the completed rod boundary")
                        reseat = choose_thumb_reseat(translated_thumb_gap,
                            lambda action: preview_opposition(action, reseat_geometry=live))
                        additional = reseat["action"]
                        reseat_windows += 1
                    result = control.control(torch.as_tensor(additional)[None], observe=observe,
                                             select_substep_action=select_opposition)
                    n_controls += 1
                    if result["termination_reason"] or result["physical_samples"] != 8:
                        raise ValueError(f"opposition stopped: {result['termination_reason']}")
                    if motion_stop or stage == "close" and index_reaction.first_contact_sdk is not None:
                        break
            probe_result = dict(start_sdk=opposition_start, close_start_sdk=close_start,
                last_sdk=native.clock(sim)["sdk_step"], physical_samples=opposition_rows, motion_stop=motion_stop,
                first_thumb_contact_sdk=contact_owner.first_contact_sdk,
                first_thumb_transfer_sdk=contact_owner.first_transfer_sdk,
                first_index_contact_sdk=index_reaction.first_contact_sdk,
                first_index_transfer_sdk=index_reaction.first_transfer_sdk, index_transfer_force_n=index_reaction.transfer_force_n,
                max_opposed_substeps=max_opposed_substeps, full_threshold_sdk_count=loaded_opposed_ticks,
                peak_normal_n=dict(thumb=thumb_peak_n,index=index_peak_n),
                teacher_admission=False, grasp_qualified=False, lift_promoted=False)
            if opposition.get("thumb_reseat"):
                probe_result["close_windows_reserved_for_reseat"] = reseat_windows
            native.write_new(output/"settled_left_opposition.json", probe_result)

        last_tick = native.clock(sim)["sdk_step"]
        if (not start_sdk < last_tick <= end_sdk
                or not opposition and last_tick != end_sdk and result["termination_reason"] != transfer_complete):
            raise ValueError("approach ended at the wrong SDK boundary")
        geometry()
        end = feedback(geometry_row)
        start, _ = ACTION_CONTRACT["slices"]["world_translation_velocity"]
        travel = {}
        for index, side in enumerate(SIDES):
            key = f"{side}.base_link"
            delta = np.asarray(geometry_row["poses"][key][0]) - start_geometry["poses"][key][0]
            direction = action[start+3*index:start+3*index+3]
            magnitude = np.linalg.norm(direction)
            signed = float(delta @ direction / magnitude) if magnitude > 0. else None
            travel[side] = dict(world_delta_m=delta.tolist(), signed_travel_m=signed)
        report = dict(start_sdk=start_sdk, end_sdk=last_tick, measured_radius_m=radius,
                      minimum_clearance_m=minimum_gap, signed_palms=travel, readback=end,
                      terminal_rod_clearance_physics_step=geometry_row["rod_source_physics_step"],
                      control_steps=n_controls, optimizer_updates=0, teacher_admission=False,
                      grasp_qualified=False)
        scope = "measured_wire_precontact_palm_feedback"
        if contact_owner is not None:
            scope = "measured_settled_wire_left_thumb" if settled_thumb else "measured_wire_left_distal_approach"
            report.update(first_contact_sdk=contact_owner.first_contact_sdk,
                first_transfer_sdk=contact_owner.first_transfer_sdk,
                contact_reached=bool(contact_owner.first_contact_sdk and contact_owner.first_transfer_sdk),
                first_pairs=contact_owner.first_pairs, contact_samples=contact_owner.contact_samples,
                terminal_violations=contact_owner.violations, lift_promoted=False)
        if right_relief is not None:
            scope = "measured_wire_right_relief_approach"
            initial_rotation = Rotation.from_quat(start_geometry["poses"]["right.base_link"][1])
            endpoint = Rotation.from_quat(geometry_row["poses"]["right.base_link"][1])
            relief_rad = float((initial_rotation.inv()*endpoint).as_rotvec()[2])
            report.update(right_intrinsic_z_rad=relief_rad, left_world_target_unchanged=True,
                          first_contact_sdk=None, lift_promoted=False)
        if probe_result is not None:
            report["left_opposition"] = probe_result
            scope = "measured_settled_wire_left_opposition_probe"
        native.write_new(output/"wire_pulse_result.json", report)
        if right_relief is not None and not (travel["right"]["signed_travel_m"] > 0. and relief_rad > 0.):
            raise ValueError("actual right translation and intrinsic-Z rotation must move as requested")
        if original_pulse and any(record["signed_travel_m"] <= 0. for record in travel.values()):
            raise ValueError("positive actual motion along each measured direction required")
        if settled_thumb and contact_owner.first_transfer_sdk is None:
            raise ValueError("settled left thumb did not deliver a measured reaction within the bounded path")
        if contact_owner is not None and contact_owner.first_contact_sdk is not None and contact_owner.first_transfer_sdk is None:
            raise ValueError("actual first contact must have its next completed reaction transfer")
        return dict(settled, scope=scope, sdk_steps=last_tick, solver_substeps=end["clock"]["rod_tick"],
                    control_steps=n_controls, measured_exchanges=settled["measured_exchanges"]+last_tick-start_sdk,
                    approach_feedback=report)
    except Exception:
        sdk = native.clock(sim)["sdk_step"]
        failure = dict(failure=traceback.format_exc(), guarded_stop_sdk_step=sdk,
                       teacher_admission=False, grasp_qualified=False, ppo_updates=0)
        native.write_new(output/"wire_pulse_failure.json", failure)
        if not 2 <= sdk <= end_sdk:
            raise
        return failure
    finally:
        sim = owners = trace = runtime = control = episode = bindings = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in parked.PATH_ARGS:
        parser.add_argument("--"+name, type=Path, required=True)
    for name in parked.VALUE_ARGS:
        kind = int if name.endswith("material-id") else str
        parser.add_argument("--"+name, type=kind, required=True)
    parser.add_argument("--distal-parked-run", type=Path)
    parser.add_argument("--distal-outer-sha256")
    parser.add_argument("--right-relief-screen", type=Path)
    parser.add_argument("--right-relief-sha256")
    parser.add_argument("--settled-run", type=Path)
    parser.add_argument("--settled-outer-sha256")
    parser.add_argument("--settled-path", type=Path)
    parser.add_argument("--settled-path-sha256")
    parser.add_argument("--opposition-proof", type=Path)
    parser.add_argument("--opposition-proof-sha256")
    parser.add_argument("--compose-pivot-preview", action="store_true")
    parser.add_argument("--thumb-reseat", action="store_true")
    parser.add_argument("--approved-frozen-sha256")
    parser.add_argument("--verify-existing-plan", action="store_true")
    args = parser.parse_args()
    native.validate_execution_context(ROOT)
    if args.verify_existing_plan or args.approved_frozen_sha256 is None:
        frozen = freeze(args)
        if args.verify_existing_plan:
            if args.approved_frozen_sha256 is not None or json.loads(args.plan.read_bytes()) != frozen:
                raise ValueError("independent native approach plan differs")
        else:
            native.write_new(args.plan, frozen)
        print(json.dumps(dict(frozen_sha256=native.digest(frozen), budget=frozen["budget"])))
        return 0
    frozen = json.loads(args.plan.read_bytes())
    if native.digest(frozen) != args.approved_frozen_sha256 or args.output.exists():
        raise ValueError("exact approved native approach recipe and fresh output required")
    child_args = [part for name in (*parked.PATH_ARGS, *parked.VALUE_ARGS)
                  for part in ("--"+name, str(getattr(args, name.replace("-", "_"))))]
    if args.distal_parked_run is not None:
        child_args.extend(("--distal-parked-run", str(args.distal_parked_run),
                           "--distal-outer-sha256", args.distal_outer_sha256))
    if args.right_relief_screen is not None:
        child_args.extend(("--right-relief-screen", str(args.right_relief_screen),
                           "--right-relief-sha256", args.right_relief_sha256))
    if args.settled_run is not None:
        child_args.extend(("--settled-run", str(args.settled_run), "--settled-outer-sha256", args.settled_outer_sha256,
                           "--settled-path", str(args.settled_path), "--settled-path-sha256", args.settled_path_sha256))
    if args.opposition_proof is not None:
        child_args.extend(("--opposition-proof", str(args.opposition_proof),
                           "--opposition-proof-sha256", args.opposition_proof_sha256))
    if args.compose_pivot_preview:
        child_args.append("--compose-pivot-preview")
    if args.thumb_reseat:
        child_args.append("--thumb-reseat")
    subprocess.run([sys.executable, "-B", "-m", "scripts.verify_dh116_wire_live_palm_approach",
                    *child_args, "--verify-existing-plan"], check=True)
    for path, digest in frozen["frozen_files"].items():
        checked_file(path, digest)
    if args.output.exists() or json.loads(args.plan.read_bytes()) != frozen:
        raise ValueError("approach inputs changed after source verification")
    lifecycle.execute(frozen, acquire_operation=partial(acquire, plan=frozen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
