"""Measure a resized supported Tripod wire in one existing Isaac lifecycle."""
import argparse
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import traceback

import numpy as np

from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile, checked_file
from go2_pvcnn.control.dh116.tabletop_supported_wire_specs import catalog_wire_specs, prepare_variant
from scripts import record_dh116_tripod_supported_reset as original


ROOT, native, lifecycle = original.ROOT, original.native, original.lifecycle
VERSION = "dh116-wire-variant-isaac-settle-v1"
SDK_END = 10
BASE_ROD = "configs/dh116_tasks/flexible_pinch_candidate_v1.json"
SOURCE_FILES = (
    BASE_ROD,
    "scripts/verify_dh116_power_shared_rod.py",
    "scripts/verify_dh116_power_supported_task.py",
    "src/go2_pvcnn/control/dh116/tabletop_supported_wire_specs.py",
    "src/go2_pvcnn/control/dh116/tabletop_supported_variant_builder.py",
    "src/go2_pvcnn/control/dh116/tabletop_material_director_readback.py",
)
PATH_ARGS = ("output", "plan", "task-config", "catalog", "rays", "left-bake", "right-bake",
             "proximal-capture", "proximal-bake")
VALUE_ARGS = ("task-sha256", "catalog-sha256", "variant", "left-material-id", "right-material-id")


def validate_settlement(result, rod_row):
    """Verify measured loads on both supports and no hand pairs in the parked interval."""
    peak = result["peak_support_force_n"]
    wrenches = rod_row["external_support_wrenches_world"]
    support_names = ("support_0", "support_1")
    if set(peak) != set(wrenches) or set(peak) != {"table", *support_names}:
        raise ValueError("exact table and both support sources required")
    forces = {}
    for name, wrench in wrenches.items():
        wrench = np.asarray(wrench)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise ValueError("finite world wrench on each registered mesh required")
        forces[name] = float(np.linalg.norm(wrench[:3]))
    if (not np.isfinite(tuple(peak.values())).all() or peak["table"] > 1e-6
            or forces["table"] > 1e-6
            or any(peak[name] <= 0. or wrenches[name][2] >= 0. for name in support_names)):
        raise ValueError("rod must finish settled on both supports without touching the table")
    if any(result["pair_counts"][name] for name in rod_row["mesh_bindings"]):
        raise ValueError("unexpected hand pair during support settlement")
    return dict(peak_support_force_n=peak, terminal_support_force_n=forces,
                terminal_support_wrench_on_mesh_w=wrenches)



def _choice(catalog, candidate_id):
    catalog_ref = catalog["reference_rod"]
    if catalog_ref["path"] != BASE_ROD:
        raise ValueError("original wire family catalog required")
    rod = json.loads(checked_file(ROOT / BASE_ROD, catalog_ref["sha256"]))["rod"]
    train = catalog_wire_specs(catalog, rod)["train"]
    selected = [case for case in train if case["id"] == candidate_id]
    if len(selected) != 1:
        raise ValueError("explicit training-only wire specimen required for the native pilot")
    case = selected[0]
    return rod, case


def freeze(args):
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import prepare_scene

    extra = {name: getattr(args, name) for name in
             ("rays", "left_bake", "right_bake", "proximal_capture", "proximal_bake")}
    plan = original.freeze_plan(args.output, task_config=args.task_config,
                                task_config_sha256=args.task_sha256, extra_sources=extra)
    catalog = json.loads(checked_file(args.catalog, args.catalog_sha256))
    scene = json.loads(checked_file(args.task_config, args.task_sha256))
    rod, case = _choice(catalog, args.variant)
    options = dict(radius_m=case["rod"]["radius_m"], length_m=case["length_m"],
                   stations=dict(left=args.left_material_id, right=args.right_material_id))
    if "elastic_modulus_scale" in case:
        options["elastic_modulus_scale"] = case["elastic_modulus_scale"]
    seed, _, _, _, variant = prepare_variant(scene, prepare_scene(ROOT)["table"], rod, **options)
    if variant["spec_sha256"] != case["spec_sha256"]:
        raise ValueError("scene and catalog wire specification differ")
    plan.update(version=VERSION, variant_id=args.variant, spec_sha256=case["spec_sha256"],
                reset_seed_sha256=seed["sha256"], variant_options=options,
                preserved_scope="resized supported Tripod Reset2 and eight stationary physical exchanges")
    plan["budget"].update(mode="resized_tripod_supported_settle", sdk_steps=SDK_END,
                          solver_steps=(SDK_END-2)*12, soft_wall_seconds=180)
    for path in (args.catalog, Path(__file__), *(ROOT / name for name in SOURCE_FILES)):
        if path.is_symlink() or not path.is_file():
            raise ValueError("regular immutable source required: " + str(path))
        key, digest = str(path.resolve(strict=True)), native.sha(path)
        if plan["frozen_files"].setdefault(key, digest) != digest:
            raise ValueError("conflicting variant source: " + key)
    return plan


def acquire(sim, owners, trace, output, *, plan):
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import prepare_scene
    from go2_pvcnn.control.dh116.tabletop_material_director_readback import _copy_endpoint
    from go2_pvcnn.control.dh116.tabletop_supported_wire_specs import observe_wire_pose_and_reaction, verify_wire_getter
    from go2_pvcnn.control.dh116.tabletop_supported_variant_builder import acquire_variant_reset
    from scripts import verify_dh116_power_shared_rod as shared

    runtime = None
    try:
        profile = BackendProfile.load()
        scene = json.loads(checked_file(plan["task_config"]["path"], plan["task_config"]["sha256"]))
        cfg_path = ROOT / BASE_ROD
        rod = json.loads(checked_file(cfg_path, plan["frozen_files"][str(cfg_path)]))["rod"]
        _, _, _, _, variant = prepare_variant(scene, prepare_scene(ROOT)["table"], rod, **plan["variant_options"])
        spec = dict(rod=variant["rod"], length_m=variant["length_m"])
        if variant["spec_sha256"] != plan["spec_sha256"]:
            raise ValueError("native selected wire specification differs")
        args = SimpleNamespace(**{name: Path(path) for name, path in plan["extra_sources"].items()},
                               verify_near_open_settling=True, measure_contact_evidence=False)

        def acquire_reset(sim, owners, trace, output, *, profile):
            try:
                setup, supports = acquire_variant_reset(
                    sim, owners, trace, output, project=ROOT, profile=profile, config=scene,
                    sdk_end=SDK_END, variant_options=plan["variant_options"],
                    expected_seed_sha256=plan["reset_seed_sha256"])
                original_runtime = owners.resources["runtime"]
                raw = original_runtime.rope.parameter_snapshot()
                getter = verify_wire_getter(spec, raw, solver_time_s=0.)
                if (setup["clock"] != original_runtime.clock() or setup["clock"]["physics_step"] != 2
                        or setup["clock"]["rod_tick"] != 0):
                    raise ValueError("resized native Reset must be at same SDK2 and rod tick zero")
                native.write_new(output/"wire_variant_reset_getters.json",
                                 dict(**getter, spec_sha256=plan["spec_sha256"], clock=setup["clock"]))
                return setup, supports
            finally:
                sim = owners = trace = original_runtime = None

        result = shared.run_shared(sim, owners, trace, output, profile=profile, args=args,
                                   supported_acquire=acquire_reset, sdk_end=SDK_END,
                                   require_no_hand_contact=True, task="tripod",
                                   live_support_scene=variant["scene"]["support_scene"])
        if result.get("failure"):
            return result
        runtime = owners.resources["power_runtime"]
        before = runtime.clock()
        terminal = decode_owned(json.loads((output/f"tripod_exchange_{SDK_END}.json").read_bytes()))
        raw = runtime.rope.parameter_snapshot()
        live_nodes, _ = _copy_endpoint(runtime.rope, "cuda:0")
        if before != runtime.clock() or before["physics_step"] != SDK_END:
            raise ValueError("read-only terminal variant parameters advanced the clock")
        rod_row = terminal["rod"]
        exchange = dict(rod=rod_row, sdk_hand_geometry=dict(physics_step=SDK_END, poses=terminal["hand_link_poses"]),
                        material_director=dict(identity=before, node_positions_m=live_nodes,
                                               raw_native_parameters=raw))
        observation = observe_wire_pose_and_reaction(
            exchange, spec, grasp="tripod", grasp_material_ids=plan["variant_options"]["stations"])
        if observation["spec_sha256"] != plan["spec_sha256"]:
            raise ValueError("terminal palm and reaction readback differs from declared variant")
        settlement = validate_settlement(result, rod_row)
        native.write_new(output/"wire_variant_terminal_feedback.json", observation)
        native.write_new(output/"wire_variant_settlement.json", {
            key: value if key != "terminal_support_wrench_on_mesh_w"
            else {name: wrench.tolist() for name, wrench in value.items()}
            for key, value in settlement.items()
        })
        return dict(result, spec_sha256=plan["spec_sha256"], variant_id=plan["variant_id"],
                    teacher_admission=False, grasp_qualified=False)
    except Exception:
        sdk = native.clock(sim)["sdk_step"]
        failure = dict(failure=traceback.format_exc(), guarded_stop_sdk_step=sdk,
                       teacher_admission=False, grasp_qualified=False, ppo_updates=0)
        native.write_new(output/"wire_variant_failure.json", failure)
        if not 2 <= sdk <= SDK_END:
            raise
        return failure
    finally:
        sim = owners = trace = runtime = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PATH_ARGS:
        parser.add_argument("--"+name, required=True, type=Path)
    for name in ("task-sha256", "catalog-sha256", "variant"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--left-material-id", type=int, required=True)
    parser.add_argument("--right-material-id", type=int, required=True)
    parser.add_argument("--approved-frozen-sha256")
    parser.add_argument("--verify-existing-plan", action="store_true")
    args = parser.parse_args()
    native.validate_execution_context(ROOT)
    if args.verify_existing_plan or args.approved_frozen_sha256 is None:
        expected = freeze(args)
        if args.verify_existing_plan:
            if args.approved_frozen_sha256 is not None or json.loads(args.plan.read_bytes()) != expected:
                raise ValueError("independent variant source freeze differs")
        else:
            native.write_new(args.plan, expected)
        print(json.dumps(dict(frozen_sha256=native.digest(expected), budget=expected["budget"])))
        return 0
    expected = json.loads(args.plan.read_bytes())
    if native.digest(expected) != args.approved_frozen_sha256 or args.output.exists():
        raise ValueError("exact approved variant recipe and fresh output required")
    child_args = [part for name in (*PATH_ARGS, *VALUE_ARGS)
                  for part in ("--" + name, str(getattr(args, name.replace("-", "_"))))]
    subprocess.run([sys.executable, "-B", "-m", "scripts.verify_dh116_wire_variant_isaac",
                    *child_args, "--verify-existing-plan"], check=True)
    for path, sha256 in expected["frozen_files"].items():
        checked_file(path, sha256)
    if args.output.exists() or json.loads(args.plan.read_bytes()) != expected:
        raise ValueError("variant execution inputs changed after source verification")
    lifecycle.execute(expected, acquire_operation=partial(acquire, plan=expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
