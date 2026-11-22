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
    try:
        settled = parked.acquire(sim, owners, trace, output, plan=plan)
        if settled.get("failure"):
            return settled
        clock = native.clock(sim)["sdk_step"]
        if clock != parked.SDK_END:
            raise ValueError("approach requires the completed native parked exchange")
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
        tube_clock, tube = None, None
        geometry_row = None

        def geometry():
            nonlocal tube_clock, tube, geometry_row
            sdk = native.clock(sim)["sdk_step"]
            rod_row = runtime._base.last
            before = runtime.clock()
            if before["physics_step"] not in (sdk-1, sdk) or rod_row["physics_step"] != before["physics_step"]:
                raise ValueError("rod geometry must use the present or immediately preceding SDK tick")
            if before != tube_clock:
                tube = wire_tube(rod_row["node_positions"], radius_m=radius)
                tube_clock = before
            poses, centers, clearance = shared.measure_links(sim, bindings, runtime.bindings, shapes,
                                                             sdk=sdk, support_scene=supports)
            frames = []
            for side in SIDES:
                transform = _owned(bindings[side].hand.root_physx_view.get_link_transforms())
                if transform.shape != (1, 23, 7):
                    raise ValueError("complete DH116 pose readback required")
                frames.append(transform[0][:, [0, 1, 2, 6, 3, 4, 5]])
            gap = min(separation(rotated_points(frames[hand][body], vertices), *tube)
                      for hand, body, vertices in shapes)
            if not np.isfinite(gap) or gap <= .00025:
                native.write_new(output/f"wire_clearance_stop_{sdk}.json", dict(
                    physics_step=sdk, rod_source_physics_step=before["physics_step"],
                    minimum_rod_clearance_m=gap))
                raise ValueError("original non-contact rod clearance failed")
            if native.clock(sim)["sdk_step"] != sdk or runtime.clock() != before:
                raise ValueError("clearance readback should not advance either clock")
            geometry_row = dict(physics_step=sdk, poses=poses, rod_source_physics_step=before["physics_step"],
                                minimum_rod_clearance_m=gap)
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
        action = choose_pulse(measured, start_geometry)
        first, last = ACTION_CONTRACT["slices"]["world_translation_velocity"]
        if not np.linalg.norm(action[first:last].reshape(2, 3), axis=1).all():
            raise ValueError("both unloaded hands require a measured approach direction")
        control = owners.keep("power_control_exchange", FlexiblePhysicalExchange(
            sim, runtime, [bindings[side] for side in SIDES], root_seed_w=root,
            read_geometry=geometry, read_geometry_after_sdk=after_sdk,
            physics_budget_end=SDK_END, continue_supported_rod=True))
        trace.watch("power_control_exchange", control, owner="wire pose and reaction pulse", release="owned scope")
        native.write_new(output/"wire_pulse_request.json", encode_owned(
            dict(geometry=start_geometry, reaction=measured, action24=action)))
        minimum_gap = start_geometry["minimum_rod_clearance_m"]

        def observe(row):
            nonlocal minimum_gap
            sdk = row["physics_step"]
            if geometry_row["physics_step"] != sdk or row["reaction_source_physics_step"] != sdk-1:
                raise ValueError("geometry and reaction transport clocks do not match")
            minimum_gap = min(minimum_gap, geometry_row["minimum_rod_clearance_m"])
            native.write_new(output/f"wire_pulse_step_{sdk}.json", encode_owned(
                dict(row, sdk_hand_geometry=geometry_row)))
            rod_row = row["rod"]
            pairs = rod_row["rod_mesh_contact_pairs"]["pairs"]
            if any(pair["mesh_name"] in runtime.bindings for pair in pairs):
                return "first_hand_reaction"
            wrenches = rod_row["external_support_wrenches_world"]
            if (np.linalg.norm(wrenches["table"][:3]) > 1e-6
                    or any(wrenches[name][2] >= 0. for name in END_SUPPORT_NAMES)):
                return "unexpected_support_transition"
            return None

        result = control.control(torch.as_tensor(action)[None], observe=observe)
        last_tick = native.clock(sim)["sdk_step"]
        if result["termination_reason"] or last_tick != SDK_END or result["physical_samples"] != 8:
            raise ValueError(f"approach stopped at SDK {last_tick}: {result['termination_reason']}")
        geometry()
        minimum_gap = min(minimum_gap, geometry_row["minimum_rod_clearance_m"])
        end = feedback(geometry_row)
        start, _ = ACTION_CONTRACT["slices"]["world_translation_velocity"]
        travel = {}
        for index, side in enumerate(SIDES):
            key = f"{side}.base_link"
            delta = np.asarray(geometry_row["poses"][key][0]) - start_geometry["poses"][key][0]
            direction = action[start+3*index:start+3*index+3]
            signed = float(delta @ direction / np.linalg.norm(direction))
            travel[side] = dict(world_delta_m=delta.tolist(), signed_travel_m=signed)
        report = dict(start_sdk=parked.SDK_END, end_sdk=last_tick, measured_radius_m=radius,
                      minimum_clearance_m=minimum_gap, signed_palms=travel, readback=end,
                      terminal_rod_clearance_physics_step=geometry_row["rod_source_physics_step"],
                      control_steps=1, optimizer_updates=0, teacher_admission=False,
                      grasp_qualified=False)
        native.write_new(output/"wire_pulse_result.json", report)
        if any(record["signed_travel_m"] <= 0. for record in travel.values()):
            raise ValueError("positive actual motion along each measured direction required")
        return dict(settled, scope="measured_wire_precontact_palm_feedback", sdk_steps=last_tick, solver_substeps=end["clock"]["rod_tick"],
                    control_steps=1, measured_exchanges=settled["measured_exchanges"]+8,
                    approach_feedback=report)
    except Exception:
        sdk = native.clock(sim)["sdk_step"]
        failure = dict(failure=traceback.format_exc(), guarded_stop_sdk_step=sdk,
                       teacher_admission=False, grasp_qualified=False, ppo_updates=0)
        native.write_new(output/"wire_pulse_failure.json", failure)
        if not 2 <= sdk <= SDK_END:
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
