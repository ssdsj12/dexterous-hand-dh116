"""Read-only flexible-scene geometry screen for the independent Power archive."""
import argparse
import hashlib
import json
from pathlib import Path


def load_power_archive(path, expected_sha):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Power archive SHA256 differs")
    report = json.loads(raw)
    if (report.get("status") != "complete" or report.get("scope") != "cpu_geometry_only"
            or report.get("source_unchanged") is not True
            or report.get("native_contact_verified") is not False
            or type(report.get("geometry_candidates")) is not int
            or report["geometry_candidates"] != 0):
        raise ValueError("expected a completed negative Power geometry archive")
    sides = {}
    for entry in report["sides"]:
        side = entry["side"]
        if side not in ("left", "right") or side in sides or not entry["rows"]:
            raise ValueError("expected distinct left and right Power candidates")
        seen = set()
        for row in entry["rows"]:
            index = row["start_index"]
            if (type(index) is not int or index < 0 or index in seen
                    or row["qualified"] is not False
                    or row["wrap"]["native_contact_verified"] is not False):
                raise ValueError("ambiguous Power candidate identity or qualification")
            seen.add(index)
        sides[side] = entry["rows"]
    if set(sides) != {"left", "right"}:
        raise ValueError("both Power sides required")
    return sides


def audit_power_archive(project, archive, expected_sha):
    from scripts.audit_dh116_pinch_initial_pose import load_geometry, shape_metrics, world_vertices

    results = {}
    for side, rows in load_power_archive(archive, expected_sha).items():
        geometry = load_geometry(project, side=side)
        layout = geometry["reference"]["reference_layout"]
        radius = geometry["old_candidate"]["rod"]["radius_m"]
        results[side] = []
        for row in rows:
            args = geometry, row["wrist_pose_w"], row["active_angles"]
            source = shape_metrics(world_vertices(*args), layout, radius)
            cooked = shape_metrics(world_vertices(*args, cooked=True), layout, radius)
            results[side].append({
                "start_index": row["start_index"],
                "rigid_archive_wrap_screen_passed": row["wrap"]["geometry_screen_passed"],
                "source_table_overlaps": [k for k, v in source.items() if v["table"]["intersects"]],
                "cooked_pad_table_overlaps": [k for k, v in cooked.items() if v["table"]["intersects"]],
                "source_rod_overlaps": [k for k, v in source.items() if v["rod"]["intersects_capsule"]],
                "cooked_pad_rod_overlaps": [k for k, v in cooked.items() if v["rod"]["intersects_capsule"]],
                "min_source_table_clearance_m": min(v["table"]["top_clearance_m"] for v in source.values()),
                "cooked_pad_rod_gap_m": {k: v["rod"]["capsule_surface_gap_m"] for k, v in cooked.items()},
            })
    return {"scope": "archived_power_initial_pose_screen_only", "archive_sha256": expected_sha,
            "candidates_by_side": results, "native_ready": False, "native_execution_performed": False,
            "blockers": ["missing_flexible_power_approach_and_bilateral_reach_evidence",
                         "nonpad_live_cooked_hulls_unavailable",
                         "native_reset_and_contact_unqualified"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_power_archive(Path(__file__).resolve().parents[1], args.archive, args.archive_sha256)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
