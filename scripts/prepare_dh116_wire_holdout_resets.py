"""Freeze a CPU reset matrix for unseen wire sizes, stiffness and placement."""
import argparse
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import prepare_scene
from go2_pvcnn.control.dh116.tabletop_supported_wire_specs import catalog_wire_specs
from go2_pvcnn.control.dh116.tabletop_wire_holdout_resets import prepare_holdout_variant
from go2_pvcnn.control.dh116.tabletop_wire_continuous_clearance import WireClearance
from go2_pvcnn.control.dh116.tabletop_wire_live_opposition import CLEARANCE_M
from scripts.certify_dh116_tripod_distal_route import support_clearance_lower
from scripts.diagnose_dh116_teacher_initialization import sha, write_new
from scripts.verify_dh116_wire_variant_isaac import BASE_ROD

ROOT = Path(__file__).resolve().parents[1]


def check_static_clearance(variant, table, hulls):
    """Measure the untouched reset against the original noncontact limits."""
    table_top = table["center_m"][2] + table["size_m"][2] / 2
    supports = []
    for support in variant["scene"]["support_scene"]["supports"]:
        center = np.asarray(support["center_m"])
        half_size = np.asarray(support["size_m"]) / 2
        supports.append((center - half_size, center + half_size))
    nodes = np.asarray(variant["scene"]["rod_nodes_w_m"])
    envelope = WireClearance(nodes[None], radius_m=variant["rod"]["radius_m"])
    failures = []
    minimum_table = minimum_support = minimum_wire = float("inf")
    unresolved_wire = False
    for (side, link, cooked), vertices in hulls.items():
        table_margin = float(vertices[:, 2].min() - table_top - CLEARANCE_M)
        support_gap = min(support_clearance_lower(vertices, lo, hi) for lo, hi in supports)
        try:
            wire = envelope.certify(vertices, selected=False)
        except ValueError as error:
            if str(error) != "continuous patch/convex-hull distance did not converge":
                raise
            wire = dict(passed=False, lower_m=None, unresolved=str(error))
        minimum_table = min(minimum_table, table_margin)
        minimum_support = min(minimum_support, support_gap)
        if wire["lower_m"] is None:
            unresolved_wire = True
        else:
            minimum_wire = min(minimum_wire, wire["lower_m"])
        if table_margin <= 0. or support_gap <= 0. or not wire["passed"]:
            failures.append(dict(side=side, link=link, cooked=cooked, table_margin_m=table_margin,
                                 support_gap_m=float(support_gap), wire=wire))
    return dict(passed=not failures, table_margin_m=minimum_table, support_gap_m=float(minimum_support),
                wire_gap_lower_m=None if unresolved_wire else minimum_wire, failures=failures)


def prepare_matrix(scene, table, baseline, *, catalog, stations, placements, hulls=None):
    """Reuse each placement across the held-out specimens and reject identical physical resets."""
    cases = catalog_wire_specs(catalog, baseline)["test"]
    if not placements or not cases or (hulls is not None and not hulls):
        raise ValueError("nonempty held-out specimens and matched placements required")
    seen = set()
    rows = []
    for case in cases:
        for placement in placements:
            variant = prepare_holdout_variant(
                scene, table, baseline, catalog=catalog, variant_id=case["id"],
                stations=stations, placement=placement)
            identity = variant["physical_initial_condition_sha256"]
            if identity in seen:
                raise ValueError("duplicate physical initial condition in held-out matrix")
            seen.add(identity)
            row = dict(variant_id=case["id"], placement=list(placement),
                       spec_sha256=variant["spec_sha256"], reset_seed_sha256=variant["reset_seed"]["sha256"],
                       physical_initial_condition_sha256=identity, native_reset_verified=False)
            if hulls is not None:
                row["cpu_clearance"] = check_static_clearance(variant, table, hulls)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--placements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-hulls", action="store_true")
    parser.add_argument("--left-material-id", type=int, required=True)
    parser.add_argument("--right-material-id", type=int, required=True)
    args = parser.parse_args()

    scene_source = json.loads(args.scene.read_bytes())
    scene = scene_source.get("config", scene_source)
    catalog = json.loads(args.catalog.read_bytes())
    baseline_path = ROOT / catalog["reference_rod"]["path"]
    if (baseline_path != ROOT / BASE_ROD
            or sha(baseline_path) != catalog["reference_rod"]["sha256"]):
        raise ValueError("held-out catalog must bind the original solver candidate")
    baseline = json.loads(baseline_path.read_bytes())["rod"]
    placement_spec = json.loads(args.placements.read_bytes())
    if placement_spec["version"] != "dh116-wire-holdout-placements-v1":
        raise ValueError("known wire holdout placement schema required")
    stations = dict(left=args.left_material_id, right=args.right_material_id)
    table = prepare_scene(ROOT)["table"]
    hulls = hull_sources = None
    if args.audit_hulls:
        from scripts.audit_dh116_pinch_initial_pose import load_geometry, world_vertices

        hulls, hull_sources = {}, {}
        for side, hand in scene["hands"].items():
            info = load_geometry(ROOT, side=side)
            hull_sources[side] = info["source_sha256"]
            for cooked in (False, True):
                for link, vertices in world_vertices(info, hand["pose_wxyz"], hand["q_rad"], cooked=cooked).items():
                    hulls[side, link, cooked] = vertices
        if len(hulls) != 44:
            raise ValueError("all 34 source envelopes and 10 locked cooked pads required")
    rows = prepare_matrix(scene, table, baseline, catalog=catalog, stations=stations,
                          placements=placement_spec["world_xy_m_yaw_rad"], hulls=hulls)
    sources = dict(scene=sha(args.scene), catalog=sha(args.catalog),
                   baseline=sha(baseline_path), placements=sha(args.placements))
    write_new(args.output, dict(version="dh116-wire-holdout-reset-matrix-v1", scope=scene["scope"],
                               source_sha256=sources, hull_source_sha256=hull_sources, stations=stations,
                               trial_count=len(rows), trials=rows,
                               native_reset_verified=False, generalization_qualified=False))


if __name__ == "__main__":
    main()
