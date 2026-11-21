#!/usr/bin/env python3
"""Screen and verify independent 902-inspired Pinch and Tripod DeformX Reset poses."""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "configs/dh116_tasks/flexible_902_distal_reset_v1.json"
BASE = ROOT / "configs/dh116_tasks/flexible_power_supported_contact_v1.json"
CAMERA = ROOT / "configs/dh116_training/stereo_camera_front_640x480_v3.json"
HAND_ORDER_CAMERA = ROOT / "configs/dh116_training/stereo_camera_hand_order_640x480_v1.json"
PINCH_SCOPE = "pinch_end_supported_contact_diagnostic"
TRIPOD_SCOPE = "tripod_end_supported_contact_diagnostic"
PADS = ("finger14_link", "finger23_link", "finger33_link")
SDK_END = 240


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidate(grasp):
    from scripts.screen_dh116_tripod_supported_reset import REQUEST, ROUTES, scene_request

    scope = {"pinch": PINCH_SCOPE, "tripod": TRIPOD_SCOPE}[grasp]
    reference = json.loads(REFERENCE.read_bytes())
    if (sha(CAMERA) != reference["student_stereo_file_sha256"]
            or sha(HAND_ORDER_CAMERA) != reference["hand_order_stereo_file_sha256"]):
        raise ValueError("the unchanged Student stereo calibration file is required")
    if sha(BASE) != reference["supported_rod_scene_sha256"]:
        raise ValueError("the original supported free-rod scene is required")
    config = json.loads(BASE.read_bytes())
    config.update(scope=scope, offline_candidate_sha256=sha(REFERENCE))
    dependencies = [Path(__file__), REFERENCE, BASE, CAMERA, HAND_ORDER_CAMERA]
    if grasp == "pinch":
        config["hands"] = reference["pinch"]["hands"]
    else:
        config["hands"] = scene_request()[1]["hands"]
        dependencies.extend((REQUEST, *ROUTES.values()))
    return reference, config, tuple(dependencies)


def screen(grasp):
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import guard_hand_supports, prepare_initial_state
    from scripts import audit_dh116_pinch_initial_pose as geometry
    from scripts.audit_dh116_certified_segment_distance import segment_hull_distance

    reference, config, files = load_candidate(grasp)
    table = reference["table"]
    seed, active, nodes, _ = prepare_initial_state(config, table, expected_scope=config["scope"])
    support = config["support_scene"]
    table_top = table["center_m"][2] + table["size_m"][2] / 2
    reports = {}
    for index, side in enumerate(("left", "right")):
        source = geometry.load_geometry(ROOT, side=side)
        if source["usd"]["sha256"] != reference["hand_usd_sha256"][side]:
            raise ValueError("the original side-specific hand USD is required")
        pose = seed["reset_root_pose_w"][index]
        if (1 if side == "left" else -1) * pose[1] <= 0:
            raise ValueError("left and right Reset wrists must be on their own wire halves")
        hulls = {}
        for cooked, group in ((False, "source"), (True, "locked_cooked")):
            hulls.update({f"{group}:{link}": verts for link, verts in
                          geometry.world_vertices(source, pose, active[index], cooked=cooked).items()})
        if len(hulls) != 22:
            raise ValueError("17 original envelopes and 5 locked Cooked pads are required per hand")
        gaps = {name: segment_hull_distance(verts, nodes[0], nodes[-1]) - support["wire_radius_m"]
                for name, verts in hulls.items()}
        rod_gap = min(gaps.values())
        table_gap = min(float(verts[:, 2].min() - table_top) for verts in hulls.values())
        support_gap = guard_hand_supports(hulls, support)["minimum_gap_m"]
        pad_count = 2 if grasp == "pinch" else 3
        reports[side] = dict(count=len(hulls), minimum_rod_gap_m=rod_gap, minimum_table_gap_m=table_gap,
                             minimum_support_gap_m=support_gap,
                             target_pad_rod_gaps_m={link: gaps[f"locked_cooked:{link}"]
                                                    for link in PADS[:pad_count]},
                             passed=rod_gap > .001 and table_gap >= .0021 and support_gap > .001)
    report = dict(grasp=grasp, readback_source="offline_source_and_locked_cooked_geometry", hands=reports,
                  geometry_passed=all(hand["passed"] for hand in reports.values()),
                  supported_scene_sha256=sha(BASE), camera_sha256=sha(CAMERA), reset_seed_sha256=seed["sha256"],
                  native_reset_verified=False, grasp_qualified=False, teacher_admission=False)
    return report, config, (*files, Path(geometry.__file__), Path(__file__).with_name(
        "audit_dh116_certified_segment_distance.py"))


def run_screen(grasp):
    # Standalone USD work must finish before Kit chooses the process's USD bindings.
    with TemporaryDirectory(prefix="dh116-902-screen-") as folder:
        output = Path(folder) / "screen.json"
        subprocess.run([sys.executable, "-B", "-m", "scripts.verify_dh116_902_flexible_reset",
                        "--grasp", grasp, "--output", str(output)], cwd=ROOT, check=True)
        return json.loads(output.read_bytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp", choices=("pinch", "tripod"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--rays", type=Path)
    parser.add_argument("--left-bake", type=Path)
    parser.add_argument("--right-bake", type=Path)
    parser.add_argument("--proximal-capture", type=Path)
    parser.add_argument("--proximal-bake", type=Path)
    parser.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    if args.plan is None:
        report, _, _ = screen(args.grasp)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2)
        return int(not report["geometry_passed"])
    report = run_screen(args.grasp)
    _, config, files = load_candidate(args.grasp)

    from scripts import verify_dh116_power_supported_task as supported
    from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary

    shared = supported.shared
    native, sources, reset = shared.native, shared.sources, shared.reset
    native.validate_execution_context(ROOT)
    if not all((args.rays, args.left_bake, args.right_bake, args.proximal_capture, args.proximal_bake)):
        parser.error("native verification requires the previously measured palm and proximal SDF evidence")
    plan = reset.freeze_plan(args.output)
    plan.update(version="dh116-902-flexible-supported-reset-v1", screen=report)
    plan["budget"].update(mode="902_" + args.grasp + "_supported_settle", sdk_steps=SDK_END, control_steps=0,
                          solver_steps=(SDK_END - 2) * shared.SUBSTEPS, new_power_mesh_registrations=6)
    _, proximal_inputs = shared.proximal_registration_inputs(args.proximal_capture, args.proximal_bake)
    scripts = ROOT / "scripts"
    files = [*files, Path(supported.__file__), Path(shared.__file__), args.rays, *proximal_inputs,
             scripts / "audit_dh116_pinch_initial_pose.py", scripts / "audit_dh116_certified_segment_distance.py"]
    palms = sources.checked_json(args.rays, sources.RAYS_SHA)["palms"]
    for side, bake in (("left", args.left_bake), ("right", args.right_bake)):
        _, _, record, _ = sources.fixture(side, palms[side], bake)
        raw = args.rays.parent / f"{side}_base_link.json"
        sources.checked_json(raw, record["gpu_row_sha256"])
        files.extend((raw, bake / "result.json", bake / f"{side}.base_link.npz"))
    plan["frozen_files"].update({str(path.resolve(strict=True)): native.sha(path) for path in files})
    assert_clean_import_boundary()
    digest = native.digest(plan)
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=digest, budget=plan["budget"])))
        return 0
    if args.approved_frozen_sha256 != digest or json.loads(args.plan.read_bytes()) != plan or args.output.exists():
        raise ValueError("exact approved 902 Reset recipe and fresh output required")

    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile

    args.verify_near_open_settling = True
    args.measure_contact_evidence = False
    acquire = partial(supported.acquire_supported_reset, config=config, sdk_end=SDK_END, task=args.grasp)
    reset.execute(plan, acquire_operation=partial(shared.run_shared, profile=BackendProfile.load(ROOT), args=args,
        supported_acquire=acquire, sdk_end=SDK_END, require_no_hand_contact=True, task=args.grasp,
        live_support_scene=config["support_scene"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
