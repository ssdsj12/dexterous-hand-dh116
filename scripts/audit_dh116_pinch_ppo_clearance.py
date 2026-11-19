#!/usr/bin/env python3
"""Read the existing Pinch Teacher journal through the first table-clearance failure."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M


def audit_clearance(rows):
    owner = previous = None
    measured = 0
    first_failure = None
    for row in rows:
        if row["operation"] != "physical":
            continue
        exchange = row["payload"]["exchange"]
        after = exchange["post_step_clearance"]
        identity = after["identity"]
        step = exchange["physics_step"]
        identity_keys = ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid")
        current_owner = tuple(identity[key] for key in identity_keys)
        if (identity.get("clock_valid") is not True or type(step) is not int
                or step != identity["physics_step"] or (owner is not None and
                     (current_owner != owner or step != previous["step"] + 1))):
            raise ValueError("same-episode consecutive physical clearance rows required")
        owner = current_owner
        measured += 1
        floor = after["floor_m"]
        clearances = after["clearance_m"][0]
        if (floor != CLEARANCE_M or len(clearances) != 2
                or not all(math.isfinite(value) for value in clearances)):
            raise ValueError("the original two-hand measured clearance contract required")
        guard = exchange.get("original_guard_failure")
        if first_failure is None and isinstance(guard, str) and guard.startswith("post_step_source_clearance:"):
            for index, side in enumerate(("left", "right")):
                clearance = clearances[index]
                if clearance >= floor:
                    continue
                geometry = exchange["closure_geometry"]
                if geometry.get("source_complete") is not True:
                    raise ValueError("complete same-clock link geometry required")
                links = geometry["sides"][side]["table_clearance_m"]
                measured_link = min(links, key=links.get)
                source, link = measured_link.split(":", 1)
                if not math.isfinite(links[measured_link]) or links[measured_link] >= floor:
                    raise ValueError("raw geometry must corroborate the failed side")
                first_failure = dict(identity={key: identity[key] for key in (*identity_keys, "physics_step")},
                    policy_step=exchange["policy_step"], substep=exchange["substep"], side=side,
                    collider_source=source, link=link, guard=guard, minimum_clearance_m=clearance,
                    measured_link_clearance_m=links[measured_link], floor_m=floor,
                    previous_clearance_m=None if previous is None else previous["clearances"][index],
                    observed_floor_deficit_m=floor-clearance,
                    wrist_projection_clipped=exchange["wrist_projection"]["safe_z_clipped"]["__dh116_array__"][0][index],
                    hand_table_contacts=after["hand_table_contacts"])
                break
            if first_failure is None:
                raise ValueError("clearance guard has no corroborating measured side")
        previous = dict(step=step, clearances=clearances)
    if not measured:
        raise ValueError("recorded physical samples required")
    return dict(physical_steps_scanned=measured, first_clearance_failure=first_failure,
                native_qualification=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--journal-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    raw = args.journal.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != args.journal_sha256 or not raw.endswith(b"\n"):
        raise ValueError("complete source journal and matching SHA256 required")
    rows = [json.loads(line) for line in raw.splitlines()]
    if any(type(row.get("sequence")) is not int or row["sequence"] != index for index, row in enumerate(rows)):
        raise ValueError("complete original journal sequence required")
    result = dict(audit_clearance(rows), journal_sha256=sha)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
