#!/usr/bin/env python3
"""Execute the verified Power retreat and opening on the original shared rod."""

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np

from scripts import screen_dh116_power_retreat_then_open as screen
from scripts import verify_dh116_power_supported_task as supported

shared = supported.shared
native, reset, sources = shared.native, shared.reset, shared.sources
PROOF_SHA = "455d727a032f52bd40c91fe380e30d01550e53745aa722dee664548e971fae40"


def compile_schedule(row, proof):
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, expand
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT

    retreat, opened = np.zeros(24), np.zeros(24)
    first_windows, second_windows = screen.WINDOWS
    seconds = 8 * shared.DT
    for i, side in enumerate(shared.SIDES):
        record = proof["hands"][side]
        hand = row["native_joint_state"][side]
        ids = [hand["names"].index(name) for name in ACTIVE]
        goal = np.asarray(record["opened_active_rad"])
        expand(goal)
        current = np.asarray(hand["target"])[0, ids]
        delta = goal - current
        if abs(delta[0]) > 2e-6 or (delta[1:] > 2e-6).any() or np.max(np.abs(delta)) > .20001:
            raise ValueError("Power opening must preserve its thumb axis and original bounded extension")
        opened[6*i:6*i+6] = delta / (second_windows * seconds * ACTION_CONTRACT["joint_rate_rad_s"])
        pose = row["hand_link_poses"][f"{side}.base_link"][0]
        delta_x = record["retreat_pose_wxyz"][0] - pose[0]
        if not np.isclose(delta_x, .004, atol=1e-9, rtol=0):
            raise ValueError("exact screened 4 mm retreat required")
        retreat[12+3*i] = delta_x / (first_windows * seconds * ACTION_CONTRACT["translation_rate_m_s"])
    commands = np.vstack((np.repeat(retreat[None], first_windows, axis=0),
                          np.repeat(opened[None], second_windows, axis=0)))
    return shared.supported_action_schedule(commands, len(commands) * 8)


def freeze(args):
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

    args.verify_near_open_settling = True
    args.settle_without_hand_contact = False
    plan, config = supported.freeze(args)
    refs, row, files = screen.load_archive(args.retreat_reference)
    proof = sources.checked_json(args.retreat_screen, PROOF_SHA)
    if (plan["budget"]["sdk_steps"] != screen.SDK or plan["budget"]["new_power_mesh_registrations"] != 6
            or proof["all_passed"] is not True or proof["before_sdk"] != screen.SDK
            or proof["outer_sha256"] != native.sha(args.retreat_reference)
            or proof["request_sha256"] != native.sha(screen.REQUEST)
            or proof["last_row_sha256"] != native.sha(files[-1])
            or proof["number_of_path_samples"] != screen.SAMPLES
            or set(proof["hands"]) != set(shared.SIDES)
            or list(refs[-1].support_boxes) != config["support_scene"]["supports"]):
        raise ValueError("immutable Power settled source and bilateral swept certificates required")
    schedule = compile_schedule(decode_owned(row), proof)
    sdk_end = screen.SDK + 8 * len(schedule)
    plan["power_retreat_then_open"] = schedule.tolist()
    plan["budget"].update(mode="power_retreat_then_open", sdk_steps=sdk_end, control_steps=len(schedule),
                          solver_steps=(sdk_end - 2)*shared.SUBSTEPS, soft_wall_seconds=420)
    for path in (Path(__file__), Path(screen.__file__), screen.REQUEST, args.retreat_screen, *files):
        plan["frozen_files"][str(path.resolve(strict=True))] = native.sha(path)
    return plan, config, row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "plan", "rays", "left-bake", "right-bake", "near-open-parent-receipt",
                 "near-open-screen", "proximal-capture", "proximal-bake", "retreat-reference", "retreat-screen"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    native.validate_execution_context(screen.measured.ROOT)
    plan, config, row = freeze(args)
    from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary

    assert_clean_import_boundary()
    digest = native.digest(plan)
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=digest, budget=plan["budget"])))
        return 0
    if (digest != args.approved_frozen_sha256 or json.loads(args.plan.read_bytes()) != plan or args.output.exists()):
        raise ValueError("approved Power opening proof with fresh native output required")
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile

    args.measure_contact_evidence = False
    end = plan["budget"]["sdk_steps"]
    reset.execute(plan, acquire_operation=partial(shared.run_shared, profile=BackendProfile.load(screen.measured.ROOT),
        args=args, sdk_end=end, require_no_hand_contact=True, live_support_scene=config["support_scene"],
        power_open_archive=row, power_open_boundary_sdk=screen.SDK+8*screen.WINDOWS[0],
        action_after_settle=plan["power_retreat_then_open"],
        supported_acquire=partial(supported.acquire_supported_reset, config=config, sdk_end=end)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
