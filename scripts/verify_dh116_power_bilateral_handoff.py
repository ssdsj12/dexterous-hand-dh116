"""Replay the Power prefix and diagnose partial dual-hand contact on a live rod."""
import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from scripts import verify_dh116_power_retreat_then_open as prefix
from scripts.audit_dh116_power_fixed_wrist_digits import load_measured_palm_boundary, measured_palm_boundary
from scripts.verify_dh116_power_partial_stage import _selected_links

shared, native = prefix.shared, prefix.native
READINESS = "docs/contracts/dh116_teacher_power_bilateral_detour_v1.json"
ANCHOR_SDK = 6504
WINDOWS = (284, 202, 88, 82, 307)
STAGES = ("wrist_reseat", "thumb_first", "two_pad_closure", "right_approach", "left_approach")
HOLD_WINDOWS = 1
_PUBLICATION_RECORDS = (
    "docs/contracts/dh116_release_manifest.json",
    "docs/contracts/dh116_work_package_registry.json",
    "docs/contracts/dh116_teacher_power_bilateral_handoff_v1.json",
)


def execution_file_inventory(files, *, project):
    """Pin runtime inputs independently of records that report this plan's digest."""
    publication = {str((project / item).resolve()) for item in _PUBLICATION_RECORDS}
    return {path: digest for path, digest in files.items() if path not in publication}


def compile_commands(blocks):
    """Check the 60 Hz signed 24D rate contract against every archived waypoint."""
    if not blocks:
        raise ValueError("nonempty Power target route required")
    previous = None
    result = []
    rates = np.repeat([ACTION_CONTRACT["joint_rate_rad_s"], ACTION_CONTRACT["translation_rate_m_s"],
                       ACTION_CONTRACT["rotation_rate_rad_s"]], [12, 6, 6])
    for points, actions in blocks:
        points, actions = np.asarray(points, dtype=float), np.asarray(actions, dtype=float)
        windows = len(actions)
        if (windows == 0 or points.shape != (windows+1, 2, 12) or actions.shape != (windows, 24)
                or not np.isfinite(points).all() or not np.isfinite(actions).all()):
            raise ValueError("finite paired hand targets and canonical action rows required")
        if previous is not None and not np.array_equal(points[0], previous):
            raise ValueError("Power target route must be exactly continuous at each handoff")
        if (np.abs(actions[:, :12]).max() > 1.
                or np.linalg.norm(actions[:, 12:].reshape(windows, 4, 3), axis=-1).max() > 1.):
            raise ValueError("Power commands exceed original joint or wrist rate limits")
        delta = np.diff(points, axis=0) * 60.
        expected = np.concatenate((delta[:, :, 6:].reshape(windows, 12),
                                   delta[:, :, :3].reshape(windows, 6),
                                   delta[:, :, 3:6].reshape(windows, 6)), axis=1) / rates
        if not np.allclose(actions, expected, atol=1e-12, rtol=0.):
            raise ValueError("Power action does not match the archived target increments")
        result.append(actions)
        previous = points[-1]
    return np.concatenate(result)


def phase_stations(anchor, windows, palms, stations):
    """Keep each stage's link and material ownership explicit in the plan."""
    _selected_links(stations, 41)
    palm_names = {"left.base_link", "right.base_link"}
    if set(palms) != palm_names or any(len(nodes) != 1 for nodes in palms.values()):
        raise ValueError("one original palm witness station per side required")
    if len(windows) != len(STAGES) or any(type(w) is not int or w <= 0 for w in windows):
        raise ValueError("five explicit Power control stages required")
    settled_palms = {name: stations[name] for name in sorted(palm_names)}
    if any(type(palms[name][0]) is not int or len(nodes) != 1 or abs(palms[name][0]-nodes[0]) > 2
           for name, nodes in settled_palms.items()):
        raise ValueError("only the short original palm station transition is allowed")
    moving_palms = {name: list(range(min(palms[name][0], nodes[0]), max(palms[name][0], nodes[0])+1))
                    for name, nodes in settled_palms.items()}
    thumbs = {name: nodes for name, nodes in stations.items() if name.endswith(".finger14_link")}
    contacts = [moving_palms, {**settled_palms, **thumbs}, stations, stations, stations, stations]
    return [dict(stage=stage, end_sdk=anchor+8*int(end), stations=allowed)
            for stage,end,allowed in zip((*STAGES,"fixed_target_observation"),
                np.cumsum([*windows,HOLD_WINDOWS]), contacts, strict=True)]


def verify_same_rod(row, live):
    """Require the geometry recheck to read the completed rod of this exact exchange."""
    rod = row["rod"]
    nodes = np.asarray(rod["node_positions"])
    if (rod["physics_step"] != row["physics_step"] or live["physics_step"] != row["physics_step"]
            or nodes.shape != (1, 41, 3) or not np.isfinite(nodes).all()
            or not np.array_equal(nodes, live["node_positions"])):
        raise ValueError("geometry must bind the completed live rod at this SDK tick")


def load_handoff(args, *, project):
    ready = json.loads((project / READINESS).read_bytes())["replay"]
    staged, _, source = load_measured_palm_boundary(args.staged_receipt, project=project)
    clock, _, _ = measured_palm_boundary(staged)
    refs = (("base_route_sha256", args.continuous_audit),
            ("right_route_sha256", args.right_route_audit),
            ("artifact_sha256", args.bilateral_detour_audit))
    base, right, left = (prefix.sources.checked_json(path, ready[key]) for key, path in refs)
    if (clock["physics_step"] != ANCHOR_SDK or clock != ready["clock"] or source != ready["archive"]
            or any(packet["clock"] != clock or packet["archive"] != source or packet["control_hz"] != 60
                   or packet["fixed_rod_only"] is not True or packet["continuous_target_geometry_proved"] is not True
                   or packet["native_execution_ready"] is not False for packet in (base, right, left))
            or left["base_route_sha256"] != ready["base_route_sha256"]
            or right["previous_route_sha256"] != ready["base_route_sha256"]
            or left["right_route_sha256"] != ready["right_route_sha256"]
            or right["source_stations"] != ready["source_stations"] or left["source_stations"] != ready["source_stations"]
            or right["right_selected_sdf_samples"]["passed"] is not True
            or left["left_target_sdf_samples"]["passed"] is not True
            or tuple(stage["stage"] for stage in base["stages"]) != STAGES[:3]):
        raise ValueError("one exact archived live boundary and the verified fixed-rod Power route are required")
    blocks = [(s["target_waypoints"], s["planned_24d_commands"]) for s in base["stages"]]
    blocks += [(right["target_waypoints"], right["planned_24d_commands"]),
               (left["left_target_waypoints"], left["left_planned_24d_commands"])]
    if tuple(len(commands) for _, commands in blocks) != WINDOWS:
        raise ValueError("the proven stage window boundaries differ")
    directory = args.staged_receipt.parent / "native"
    files = (project/READINESS, args.staged_receipt, *(file for _,file in refs),
             directory/f"power_control_{ANCHOR_SDK}.json", directory/"power_support_scene.json",
             directory/"app_close_returned.json")
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned

    return compile_commands(blocks), encode_owned(staged), ready["source_stations"], files


def freeze(args):
    plan, config, parked = prefix.freeze(args)
    plan["frozen_files"] = execution_file_inventory(plan["frozen_files"], project=shared.ROOT)
    commands, archive, stations, files = load_handoff(args, project=shared.ROOT)
    route = plan["power_high_route"]
    if plan["budget"]["sdk_steps"] != ANCHOR_SDK:
        raise ValueError("the initial live replay must end at the sealed palm anchor")
    phases = phase_stations(ANCHOR_SDK, WINDOWS, route["palm_witness"]["stations"], stations)
    schedule = np.vstack((plan["power_retreat_then_open"], commands, np.zeros((HOLD_WINDOWS, 24))))
    plan["power_retreat_then_open"] = schedule.tolist()
    route["bilateral_partial"] = dict(anchor_sdk=ANCHOR_SDK, archive=archive, phases=phases,
                                      final_stations=stations)
    end = phases[-1]["end_sdk"]
    plan["budget"].update(mode="power_bilateral_partial_contact", sdk_steps=end, control_steps=len(schedule),
                          solver_steps=(end-2)*shared.SUBSTEPS, soft_wall_seconds=args.soft_wall_seconds)
    dependencies = (
        Path(__file__), shared.ROOT/"scripts/verify_dh116_power_bilateral_handoff_probe.py",
        shared.ROOT/"scripts/verify_dh116_power_partial_stage.py",
        shared.ROOT/"scripts/verify_dh116_supported_distal_transfer.py",
        shared.ROOT/"scripts/audit_dh116_certified_segment_distance.py",
    )
    for path in (*dependencies, *files):
        key = str(path.resolve(strict=True))
        digest = native.sha(path)
        if plan["frozen_files"].setdefault(key, digest) != digest:
            raise ValueError(f"conflicting handoff source: {key}")
    return plan, config, parked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "plan", "rays", "left-bake", "right-bake", "near-open-parent-receipt",
                 "near-open-screen", "proximal-capture", "proximal-bake", "retreat-reference", "retreat-screen",
                 "high-bypass-reference", "palm-witness-reference", "staged-receipt",
                 "continuous-audit", "right-route-audit", "bilateral-detour-audit"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--soft-wall-seconds", type=int, required=True)
    parser.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    if args.soft_wall_seconds <= 0:
        parser.error("positive soft wall budget required")
    native.validate_execution_context(shared.ROOT)
    plan, config, parked = freeze(args)
    from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary

    assert_clean_import_boundary()
    digest = native.digest(plan)
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=digest, budget=plan["budget"])))
        return 0
    if (digest != args.approved_frozen_sha256 or json.loads(args.plan.read_bytes()) != plan or args.output.exists()):
        raise ValueError("approved separate Power partial contact plan and fresh output required")
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile

    end = plan["budget"]["sdk_steps"]
    args.measure_contact_evidence = False
    prefix.reset.execute(plan, acquire_operation=partial(shared.run_shared, profile=BackendProfile.load(shared.ROOT),
        args=args, sdk_end=end, require_no_hand_contact=True, live_support_scene=config["support_scene"],
        power_open_archive=parked, power_open_boundary_sdk=prefix.screen.SDK+8*prefix.screen.WINDOWS[0],
        action_after_settle=plan["power_retreat_then_open"], power_route=plan["power_high_route"],
        supported_acquire=partial(prefix.supported.acquire_supported_reset, config=config, sdk_end=end)))
    summary = args.output/"power_palm_probe_summary.json"
    if not summary.is_file() or (args.output/"power_failure.json").exists():
        return 1
    return int(json.loads(summary.read_bytes())["local_opposition_with_reaction"] is not True)


if __name__ == "__main__":
    raise SystemExit(main())
