"""Select one reachable fixed starting wrist pair per grasp using training wire only."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.linear_coupling import active_limits, expand
from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import prepare_scene
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import CONTRACT as WRIST
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from go2_pvcnn.control.dh116.tabletop_power_supported_task import POWER_SCOPE, prepare_initial_state
from go2_pvcnn.control.dh116.tabletop_supported_wire_specs import build_variant, catalog_wire_specs, prepare_variant
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from go2_pvcnn.control.dh116.tabletop_wire_holdout_resets import place_fixed_wrist_variant
from scripts.diagnose_dh116_teacher_initialization import digest, sha, write_new
from scripts.audit_dh116_pinch_initial_pose import load_geometry, world_vertices
from scripts.prepare_dh116_power_palm_seed import thumb_clearance
from scripts.prepare_dh116_wire_holdout_resets import CLEARANCE_M, check_static_clearance
from scripts.prepare_dh116_tripod_contact_seed import read_receipt
from scripts.verify_dh116_wire_variant_isaac import BASE_ROD, QUIET_SAMPLES, validate_quiet_settlement

ROOT = Path(__file__).resolve().parents[1]
TRAINING_CATALOG = ROOT / "configs/dh116_training/supported_wire_generalization_v2.json"
SCREEN_CONFIG = ROOT / "configs/dh116_training/supported_wire_common_pregrasp_v1.json"


def open_flexion(q, fraction, *, thumb_scale=1.):
    """Open the five flexion masters with an optional Power thumb offset."""
    lower, _ = active_limits()
    opened = np.asarray(q, dtype=float).copy()
    opened[0] *= thumb_scale
    opened[1:] = (1. - fraction) * opened[1:] + fraction * lower[1:]
    expand(opened)
    return opened


def authored_side_hulls(side, hand, geometry):
    hulls = {(side, link, cooked): vertices
             for cooked in (False, True)
             for link, vertices in world_vertices(geometry, hand["pose_wxyz"], hand["q_rad"], cooked=cooked).items()}
    if len(hulls) != 22:
        raise ValueError("all 17 source envelopes and five locked pads required per hand")
    return hulls


def select_side(side, hand, geometry, table, layouts, target_offsets, candidates, fractions, *, thumb_scale=1.):
    """Select the least opening fraction and shortest shift that pass the entire training cohort."""
    wrist_lower, wrist_upper = np.asarray(WRIST["palm_lower_m"]), np.asarray(WRIST["palm_upper_m"])
    trial_rows = []
    for fraction in fractions:
        q = open_flexion(hand["q_rad"], fraction, thumb_scale=thumb_scale)
        hulls = authored_side_hulls(side, dict(hand, q_rad=q.tolist()), geometry)
        thumb_gap = thumb_clearance({f"{cooked}:{link}": vertices
                                     for (_, link, cooked), vertices in hulls.items()})
        if thumb_gap <= 0.:
            trial_rows.append(dict(open_fraction=float(fraction), thumb_to_other_digits_gap_m=thumb_gap))
            continue
        for offset in candidates:
            return_offsets = target_offsets - offset
            margin = float(np.minimum(return_offsets - wrist_lower, wrist_upper - return_offsets).min())
            row = dict(offset_m=offset.tolist(), open_fraction=float(fraction), wrist_component_margin_m=margin)
            if margin < 0.:
                trial_rows.append(dict(row, rejected="material_station_return_outside_original_wrist_workspace"))
                continue
            moved = {key: vertices + offset for key, vertices in hulls.items()}
            results = []
            for case_id, placement, variant in layouts:
                clearance = check_static_clearance(variant, table, moved)
                results.append(clearance)
                if not clearance["passed"]:
                    row["first_failed_layout"] = dict(variant_id=case_id, placement=placement, **clearance)
                    break
            trial_rows.append(dict(row, checked_layouts=len(results)))
            if "first_failed_layout" in row:
                continue
            return (
                dict(pose_wxyz=np.r_[np.asarray(hand["pose_wxyz"][:3]) + offset, hand["pose_wxyz"][3:]].tolist(),
                     q_rad=q.tolist()),
                moved,
                dict(status="cpu_reset_candidate", selected=row, thumb_to_other_digits_gap_m=thumb_gap,
                     nominal_return_distance_m=float(np.linalg.norm(offset)),
                     nominal_minimum_wrist_duration_s=float(np.linalg.norm(offset) / ACTION_CONTRACT["translation_rate_m_s"]),
                     nominal_minimum_finger_duration_s=float(
                         np.max(np.abs(np.asarray(hand["q_rad"]) - q)) / ACTION_CONTRACT["joint_rate_rad_s"]),
                     minimum_training_clearance_m={key: min(item[key] for item in results)
                                                   for key in ("table_margin_m", "support_gap_m", "wire_gap_lower_m")},
                     trials=trial_rows),
            )
    return None, {}, dict(status="no_common_training_reset", trials=trial_rows)


def load_settled_training_layout(run, receipt_sha256, scene, scene_sha256, table, baseline, stations):
    """Use a sealed quiet Tripod training curve as an additional CPU layout."""
    receipt = read_receipt(run, receipt_sha256)

    def read(name):
        return json.loads(checked_file(run / "native" / name, receipt["artifact_sha256"][name]))

    plan = read("frozen_plan.json")
    result = read("result.json")
    quiet = read("wire_variant_quiet_window.json")
    end = plan["quiet_settlement_sdk"]
    if (plan["version"] != "dh116-wire-variant-isaac-quiet-v1" or digest(plan) != receipt["frozen_sha256"]
            or plan["task_config"]["sha256"] != scene_sha256 or plan["variant_options"]["stations"] != stations
            or result["failure"] is not None or result["physics_steps"] != end or result["ppo_updates"] != 0
            or result["acquisition"]["quiet_settlement"] != quiet or not quiet["qualified"]):
        raise ValueError("source must be the sealed supported quiet run for this training scene")
    seed, _, _, _, variant = prepare_variant(scene, table, baseline, **plan["variant_options"])
    if variant["spec_sha256"] != plan["spec_sha256"] or seed["sha256"] != plan["reset_seed_sha256"]:
        raise ValueError("quiet run did not use the expected training rod and reset")
    rows = [decode_owned(read(f"tripod_exchange_{tick}.json"))["rod"]
            for tick in range(end - QUIET_SAMPLES + 1, end + 1)]
    reset_nodes = variant["scene"]["rod_nodes_w_m"]
    if validate_quiet_settlement(rows, reset_nodes, weight_n=quiet["measured_weight_n"]) != quiet:
        raise ValueError("quiet-window readback differs from the sealed settlement check")
    variant["scene"]["rod_nodes_w_m"] = rows[-1]["node_positions"][0].tolist()
    return dict(variant, variant_id=plan["variant_id"], outer_receipt_sha256=receipt_sha256,
                terminal_archive_sha256=receipt["artifact_sha256"][f"tripod_exchange_{end}.json"],
                terminal_sdk=end)


def select_common_pregrasp(scene, table, baseline, *, catalog, stations, placements,
                           wrist_candidates_m, opening_fractions, return_reserve_m, geometries, settled_layout=None,
                           power_left_thumb_scale=1.):
    """Choose one pose per hand from training and retain every rejection."""
    if not 0. <= power_left_thumb_scale <= 1. or (power_left_thumb_scale != 1. and scene["scope"] != POWER_SCOPE):
        raise ValueError("left thumb opposition adjustment requires a Power scene and scale in [0, 1]")
    train = catalog_wire_specs(catalog, baseline)["train"]
    if settled_layout is not None and not any(
            case["spec_sha256"] == settled_layout["spec_sha256"] for case in train):
        raise ValueError("settled layout is not a training specimen")
    shifts = np.asarray(placements, dtype=float)
    offsets = np.asarray(wrist_candidates_m, dtype=float)
    fractions = np.asarray(opening_fractions, dtype=float)
    if (not train or set(geometries) != {"left", "right"} or fractions.ndim != 1 or not fractions.size
            or not np.isfinite(fractions).all() or fractions[0] != 0. or fractions[-1] > 1.
            or np.any(np.diff(fractions) <= 0.)):
        raise ValueError("training shapes and increasing opening fractions from zero to at most one required")
    if (shifts.ndim != 2 or shifts.shape[1:] != (3,) or not len(shifts) or not np.isfinite(shifts).all()
            or len(np.unique(shifts, axis=0)) != len(shifts)):
        raise ValueError("distinct finite training XY and yaw positions required")
    if (offsets.ndim != 2 or offsets.shape[1:] != (3,) or not len(offsets)
            or not np.isfinite(offsets).all() or len(np.unique(offsets, axis=0)) != len(offsets)
            or not np.any(np.all(offsets == 0., axis=1))):
        raise ValueError("distinct finite wrist offsets including the unchanged start required")
    reserve = np.array([return_reserve_m, return_reserve_m, 0.])
    if (not math.isfinite(return_reserve_m) or return_reserve_m <= 0.
            or np.max(np.abs(shifts[:, :2])) > return_reserve_m
            or np.any(-offsets - reserve < np.asarray(WRIST["palm_lower_m"]))
            or np.any(-offsets + reserve > np.asarray(WRIST["palm_upper_m"]))):
        raise ValueError("retreat and reserve must fit the original wrist return travel")
    offsets = sorted(offsets, key=lambda offset: (np.linalg.norm(offset), *offset))

    prepare_initial_state(scene, table, expected_scope=scene["scope"])
    rod = np.asarray(scene["rod_nodes_w_m"])
    layouts, station_offsets = [], {side: [] for side in scene["hands"]}
    for case in train:
        for placement in shifts:
            variant = build_variant(scene, baseline, radius_m=case["rod"]["radius_m"], length_m=case["length_m"],
                                    elastic_modulus_scale=case.get("elastic_modulus_scale", 1.),
                                    grasp_material_ids=stations)
            if variant["spec_sha256"] != case["spec_sha256"]:
                raise ValueError("training variant differs from its solver specification")
            nodes = place_fixed_wrist_variant(variant, scene, table, placement)
            layouts.append((case["id"], placement.tolist(), variant))
            for side, station in stations.items():
                station_offsets[side].append(nodes[station] - rod[station])

    if settled_layout is not None:
        settled_nodes = np.asarray(settled_layout["scene"]["rod_nodes_w_m"], dtype=float)
        layouts.append((settled_layout["variant_id"], "measured_settled", settled_layout))
        for side, station in stations.items():
            station_offsets[side].append(settled_nodes[station] - rod[station])

    result = dict(version="dh116-wire-common-pregrasp-v1", scope=scene["scope"],
                  training_spec_sha256={case["id"]: case["spec_sha256"] for case in train},
                  training_layout_count=len(layouts), return_reserve_m=return_reserve_m, hands={},
                  native_reset_verified=False, approach_verified=False, grasp_qualified=False)
    if settled_layout is not None:
        result["settled_training"] = {key: settled_layout[key] for key in (
            "outer_receipt_sha256", "terminal_archive_sha256", "terminal_sdk", "spec_sha256", "variant_id")}
    if power_left_thumb_scale != 1.:
        result["power_left_thumb_scale"] = power_left_thumb_scale
    reset_hands, reset_hulls = {}, {}
    for side, hand in scene["hands"].items():
        reset_hands[side], reset_hulls[side], result["hands"][side] = select_side(
            side, hand, geometries[side], table, layouts, np.asarray(station_offsets[side]), offsets, fractions,
            thumb_scale=power_left_thumb_scale if side == "left" else 1.)
    if any(hand is None for hand in reset_hands.values()):
        return dict(result, status="no_common_training_reset")
    between_hands = (min(float(hull[:, 1].min()) for hull in reset_hulls["left"].values())
                     - max(float(hull[:, 1].max()) for hull in reset_hulls["right"].values()))
    result["between_hands_world_y_gap_m"] = between_hands
    if between_hands <= CLEARANCE_M:
        return dict(result, status="no_certified_hand_pair")

    candidate = deepcopy(scene)
    candidate["hands"] = reset_hands
    source = dict(original_scene_sha256=digest(scene), training_spec_sha256=result["training_spec_sha256"],
                  training_placements=shifts.tolist(), hands=reset_hands)
    if settled_layout is not None:
        source["settled_training"] = result["settled_training"]
    candidate["offline_candidate_sha256"] = digest(source)
    seed, _, _, _ = prepare_initial_state(candidate, table, expected_scope=scene["scope"])
    return dict(result, status="cpu_reset_candidate", config=candidate, reset_seed_sha256=seed["sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--left-material-id", required=True, type=int)
    parser.add_argument("--right-material-id", required=True, type=int)
    parser.add_argument("--power-left-thumb-scale", type=float, default=1.)
    parser.add_argument("--settled-run", type=Path)
    parser.add_argument("--settled-outer-sha256")
    args = parser.parse_args()
    if (args.settled_run is None) != (args.settled_outer_sha256 is None):
        parser.error("both settled-run and settled-outer-sha256 are required together")

    scene_file = json.loads(args.scene.read_bytes())
    scene = scene_file.get("config", scene_file)
    catalog = json.loads(TRAINING_CATALOG.read_bytes())
    screen = json.loads(SCREEN_CONFIG.read_bytes())
    baseline_path = ROOT / BASE_ROD
    if (screen["version"] != "dh116-wire-common-pregrasp-screen-v1"
            or catalog["reference_rod"] != dict(path=BASE_ROD, sha256=sha(baseline_path))):
        raise ValueError("known training screen and source-bound baseline solver required")
    baseline = json.loads(baseline_path.read_bytes())["rod"]
    table = prepare_scene(ROOT)["table"]
    geometries = {side: load_geometry(ROOT, side=side) for side in scene["hands"]}
    hull_sources = {side: geometry["source_sha256"] for side, geometry in geometries.items()}
    stations = dict(left=args.left_material_id, right=args.right_material_id)
    settled_layout = None
    offsets = screen["wrist_candidates_m"]
    if args.settled_run is not None:
        settled_layout = load_settled_training_layout(
            args.settled_run, args.settled_outer_sha256, scene, sha(args.scene), table, baseline, stations)
        offsets = sorted({tuple(offset) for offset in offsets} | {
            (x, y, settle_z) for x, y, _ in offsets for settle_z in screen["settled_reset_z_m"]})
    result = select_common_pregrasp(
        scene, table, baseline, catalog=catalog, stations=stations, placements=screen["training_xy_m_yaw_rad"],
        wrist_candidates_m=offsets, opening_fractions=screen["opening_fractions"],
        return_reserve_m=screen["return_reserve_m"], geometries=geometries, settled_layout=settled_layout,
        power_left_thumb_scale=args.power_left_thumb_scale)
    result["source_sha256"] = dict(scene=sha(args.scene), catalog=sha(TRAINING_CATALOG), screen=sha(SCREEN_CONFIG),
                                   baseline=sha(baseline_path), hulls=hull_sources)
    write_new(args.output, result)
    if "config" not in result:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
