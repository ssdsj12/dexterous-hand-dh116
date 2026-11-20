#!/usr/bin/env python3
"""Audit whether all five Power digits can reach from the selected left wrist."""
import argparse
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from scripts.audit_dh116_pinch_initial_pose import DISTAL_PAD_LINKS, checked_json
from scripts.audit_dh116_power_material_reach import (
    ACTIVE, _pad_gap_interval, active_limits, endpoint_gap, load_geometry,
    path_displacement_bound, seated_geometry, shapes_world, summarize_middle_reach,
    support_clearance, supported_seed_plan,
)


PROBE_SHA = "bed01826298bfbb88ce573ee324a593341275e77523ff77a75a868feeb085c17"
SAMPLES = 65


def _digit_reach(geometry, wrist, open_q, index, pad, nodes, radius, supports, table, limits):
    lower, upper = limits
    q = open_q.copy()
    q[index] = lower[index]
    delta = np.zeros_like(q)
    delta[index] = upper[index] - lower[index]
    reserve = path_displacement_bound(geometry, q, delta, cooked=True)[pad] / (2 * (SAMPLES - 1))
    pad_key = "cooked:" + pad
    prefix = f"finger{index}"
    samples = []
    for angle in np.linspace(lower[index], upper[index], SAMPLES):
        q[index] = angle
        digit = {key: vertices for key, vertices in shapes_world(geometry, wrist, q).items()
                 if key.split(":")[1].startswith(prefix)}
        gap = _pad_gap_interval(digit[pad_key], nodes, radius)
        table_gap = min(float(vertices[:, 2].min() - table) for vertices in digit.values())
        hits, support_gap = support_clearance(digit, supports)
        rod_clear = True
        for vertices in digit.values():
            check = _pad_gap_interval(vertices, nodes, radius)
            if check["lower_m"] < 0 <= check["upper_m"]:
                check = endpoint_gap(vertices, nodes, radius)
            if check["lower_m"] < 0:
                rod_clear = False
                break
        samples.append(dict(angle_rad=float(angle), gap_lower_m=gap["lower_m"], gap_upper_m=gap["upper_m"],
                            clear=bool(rod_clear and table_gap >= CLEARANCE_M
                                       and not hits and support_gap >= CLEARANCE_M)))
    best = min(samples, key=lambda row: row["gap_upper_m"])
    q[index] = best["angle_rad"]
    tight = endpoint_gap(shapes_world(geometry, wrist, q)[pad_key], nodes, radius)
    return dict(link=pad, summary=summarize_middle_reach(samples, point_travel_m=reserve),
                travel_reserve_m=reserve, tight_best_sample=tight, samples=samples)


def audit_fixed_wrist(probe, *, project):
    candidate, = [item for item in probe["left"] if item["abduction_fraction"] == .15 and item["roll"] == 1.45]
    wrist = np.asarray(candidate["root"], dtype=float)
    open_q = np.asarray(candidate["open_q"], dtype=float)
    lower, upper = active_limits()
    expected_q = lower.copy()
    expected_q[0] += .15 * (upper[0] - lower[0])
    if (wrist.shape != (7,) or not np.isfinite(wrist).all()
            or not np.isclose(np.linalg.norm(wrist[3:]), 1., atol=1e-10, rtol=0.)
            or not np.array_equal(open_q, expected_q)):
        raise ValueError("original fixed left wrist and 15% thumb abduction required")
    plan, _ = supported_seed_plan(project)
    nodes = np.asarray(plan["seeds_by_style"]["power"]["nodes"], dtype=float)
    supports = plan["scene"]["supports"]
    geometry = load_geometry(project, side="left")
    layout = geometry["reference"]["reference_layout"]
    table = layout["table_center"][2] + layout["table_size"][2] / 2
    radius = geometry["old_candidate"]["rod"]["radius_m"]
    hulls = shapes_world(geometry, wrist, open_q)
    palm = endpoint_gap(hulls["source:base_link"], nodes, radius)
    if not 0 <= palm["lower_m"] <= palm["upper_m"] <= .0005:
        raise ValueError("fixed wrist must retain the original palm witness")
    if not seated_geometry(hulls, nodes, supports, table, radius)["eligible"]:
        raise ValueError("fixed wrist must pass the original open geometry guards")
    digits = {
        ACTIVE[index]: _digit_reach(geometry, wrist, open_q, index, pad, nodes, radius, supports,
                                   table, (lower, upper))
        for index, pad in enumerate(DISTAL_PAD_LINKS, 1)
    }
    excluded = [key for key, row in digits.items() if row["summary"]["fixed_wrist_reach_excluded"]]
    return dict(joint_results=digits, fixed_wrist_excluded_by=excluded, palm_upper_m=palm["upper_m"],
                independent_digit_screen=True, side="left", native_contact_verified=False,
                teacher_admission=False, actual_isaac_steps=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    probe = checked_json(args.probe, PROBE_SHA)
    result = audit_fixed_wrist(probe, project=Path(__file__).resolve().parents[1])
    result["probe_sha256"] = PROBE_SHA
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
