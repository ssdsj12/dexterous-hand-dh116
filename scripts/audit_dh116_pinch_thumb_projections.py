#!/usr/bin/env python3
"""Attribute measured left Pinch opposition losses to normal and tangential loads."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_deformable_pair_source_basis import compare_contacts
from scripts.bake_dh116_deformx_pad_sdfs import write_json


FIRST_SDK, LAST_SDK, SOLVER_SUBSTEPS = 299, 362, 12
INDEX, THUMB = "left.finger23_link", "left.finger14_link"
LEDGER_SHA256 = {
    "fixed": "7bde45de8bb66ddfc5a5d01c98c299be110f57afbbfbbdf2e77e255cb96811e1",
    "longitudinal": "31f4a762204b11fb5a8de14da53530b047d739062f907810ca3d59722c09ae04",
}


def measure_step(exchange):
    tick = exchange["physics_step"]
    contacts = {}
    for sample in exchange["rod"]["rod_mesh_contact_pairs"]["pairs"]:
        if sample["mesh_name"] not in (INDEX, THUMB):
            continue
        key = (sample["mesh_name"], sample["substep"])
        if key in contacts:
            raise ValueError("multiple pad contacts in one substep require separate attribution")
        contacts[key] = sample
    expected = list(range(SOLVER_SUBSTEPS))
    if [x["substep"] for x in exchange["stage_reaction"]["per_substep"]] != expected:
        raise ValueError("original staged per-substep record is missing or out of order")
    rows = []
    for substep in expected:
        index, thumb = contacts.get((INDEX, substep)), contacts.get((THUMB, substep))
        if index is None or thumb is None:
            raise ValueError("paired left contact evidence missing")
        record = compare_contacts(index, thumb)
        total = np.asarray(record["inward_force_projections_n"], dtype=float)
        normal = np.asarray(record["inward_normal_cosines"], dtype=float) * record["normal_loads_n"]
        if total.shape != (2,) or not np.isfinite(total).all() or not np.isfinite(normal).all():
            raise ValueError("inward projection was not observed")
        rows.append(dict(sdk=tick, substep=substep, original_signed=record["geometric_opposition_sign_candidate"],
            normal_n=normal.tolist(), tangential_n=(total-normal).tolist(), inward_n=total.tolist(),
            material_nodes=[index["node_index"], thumb["node_index"]]))
    return rows


def summarize(rows):
    if not rows:
        raise ValueError("at least one verified pressure sample required")
    thumb_inward = np.asarray([row["inward_n"][1] for row in rows])
    losses = [row for row in rows if row["original_signed"] is not True]
    nodes = Counter(tuple(row["material_nodes"]) for row in rows)
    return dict(substeps=len(rows), thumb_inward_p05_n=float(np.percentile(thumb_inward, 5)),
                thumb_inward_min_n=float(thumb_inward.min()), sign_losses=len(losses),
                negative_thumb_inward_count=int((thumb_inward <= 0).sum()),
                nodes_hist=[[*ids, count] for ids, count in sorted(nodes.items())], loss_samples=losses)


def read_settled_samples(path, expected_sha):
    rows, ticks = [], []
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            packet = json.loads(line)
            if packet.get("operation") != "physical":
                continue
            row = packet["payload"]["exchange"]
            tick = row["physics_step"]
            if FIRST_SDK <= tick <= LAST_SDK:
                rows.extend(measure_step(row))
                ticks.append(tick)
    if digest.hexdigest() != expected_sha:
        raise ValueError("exact preserved original Pinch ledger required")
    if ticks != list(range(FIRST_SDK, LAST_SDK + 1)):
        raise ValueError("preserved settled SDK window must be consecutive")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*LEDGER_SHA256, "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = {name:dict(ledger_sha256=digest, **summarize(read_settled_samples(getattr(args, name), digest)))
              for name, digest in LEDGER_SHA256.items()}
    write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
