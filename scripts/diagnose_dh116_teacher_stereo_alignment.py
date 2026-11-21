#!/usr/bin/env python3
"""Frozen N=1 Teacher initialization and same-boundary Student RGB registration."""
import argparse
import json
from pathlib import Path

from scripts import diagnose_dh116_teacher_stat_cache_release as cleanup
from go2_pvcnn.control.dh116 import tabletop_teacher_stereo_alignment as alignment
from go2_pvcnn.control.dh116.tabletop_tiled_stereo_resources import verify_tiled_sources


ROOT = Path(__file__).resolve().parents[1]
legacy = cleanup.legacy
VERSION = "dh116-teacher-student-camera-native-v1"


def freeze_plan(root, output, bindings, *, camera_profile="front"):
    plan = cleanup.freeze_plan(root, output, bindings)
    rendering = legacy.EXPERIENCE.with_name("isaaclab.python.headless.rendering.kit")
    camera = alignment.student_camera_path(root, camera_profile)
    calibration = alignment.student_camera_calibration(root, camera_profile)
    plan["frozen_files"][str(camera.resolve())] = legacy.sha(camera)
    plan["version"] = VERSION
    plan["experience"] = str(rendering)
    plan["frozen_files"][str(rendering)] = legacy.sha(rendering)
    for row in verify_tiled_sources(root).values():
        plan["frozen_files"][row["path"]] = row["sha256"]
    plan["teacher_stereo"] = dict(camera_config=str(camera.relative_to(root)), camera_profile=camera_profile,
                                  calibration_sha256=calibration.sha256,
                                  resolution_wh=[calibration.width, calibration.height],
                                  capture_pairs=1, max_render_attempts=4)
    return plan


def verify_plan(plan, approved):
    if plan.get("version") != VERSION or legacy.digest(plan) != approved:
        raise ValueError("exact independently approved Teacher camera fingerprint required")
    profile = plan["teacher_stereo"].get("camera_profile", "front")
    if plan != freeze_plan(plan["project"], plan["output"], plan["binding_reports"], camera_profile=profile):
        raise ValueError("frozen Student camera or Teacher source changed")
    if Path(plan["output"]).exists():
        raise ValueError("fresh exclusive output directory required")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binding-reports", type=Path, default=legacy.BINDINGS)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--student-camera-profile", choices=alignment.STUDENT_CAMERA_PROFILES, default="front")
    parser.add_argument("--execute-native", action="store_true")
    parser.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    if args.project.resolve() != ROOT:
        raise ValueError("Teacher stereo worker and worktree identity differ")
    cleanup.validate_execution_context(ROOT)
    if not args.execute_native:
        plan = freeze_plan(args.project, args.output, args.binding_reports,
                           camera_profile=args.student_camera_profile)
        legacy.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=legacy.digest(plan), native_started=False)))
        return 0

    plan = json.loads(args.plan.read_text())
    verify_plan(plan, args.approved_frozen_sha256)
    if (str(args.output.resolve()) != plan["output"]
            or str(args.binding_reports.resolve()) != plan["binding_reports"]
            or args.student_camera_profile != plan["teacher_stereo"]["camera_profile"]):
        raise ValueError("CLI identity differs from the frozen Teacher camera plan")

    def capture(runtime):
        alignment.capture_initial_pair(runtime, plan, read_clock=legacy.clock, write_receipt=legacy.write_new)

    legacy.execute_native(plan, before_owner_collect=cleanup.make_before_collect(plan),
                          teacher_stereo_capture=capture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
