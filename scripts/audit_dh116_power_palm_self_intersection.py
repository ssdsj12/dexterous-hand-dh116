#!/usr/bin/env python3
"""Audit intersections in the original cooked Power palm triangle surfaces."""
import argparse
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from scripts import audit_dh116_power_palm_surface as surface
from scripts.audit_dh116_cooked_hull_device_parity import _pad_hull


def reduce_equations(rows, unknowns):
    rows = [list(row) for row in rows]
    rank = 0
    for col in range(unknowns):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        divisor = rows[rank][col]
        rows[rank] = [x / divisor for x in rows[rank]]
        for i, row in enumerate(rows):
            if i != rank and row[col]:
                rows[i] = [a - row[col] * b for a, b in zip(row, rows[rank])]
        rank += 1
    if any(row[-1] for row in rows[rank:]):
        return None
    return rows[:rank]


def crosses_shared_boundary(points, left, right):
    # A feasible vertex of the barycentric polytope exposes any overlap outside shared edges or vertices.
    shared = set(left) & set(right)
    if len(shared) == 3:
        return True
    indices = (*left, *right)
    zero, one = Fraction(0), Fraction(1)
    matrix = [[one] * 3 + [zero] * 3 + [one], [zero] * 3 + [one] * 3 + [one]]
    matrix.extend([[points[i][axis] for i in left] + [-points[i][axis] for i in right] + [zero]
                   for axis in range(3)])
    equations = reduce_equations(matrix, 6)
    if equations is None:
        return False
    rank = len(equations)
    for columns in combinations(range(6), rank):
        system = reduce_equations([[row[i] for i in columns] + [row[-1]] for row in equations], rank)
        if system is None or len(system) != rank:
            continue
        weights = [row[-1] for row in system]
        if min(weights) >= 0 and any(w > 0 and indices[i] not in shared for i, w in zip(columns, weights)):
            return True
    return False


def audit_intersections(vertices, faces):
    points = tuple(tuple(Fraction(float(a)) for a in v) for v in vertices)
    tris = vertices[faces]
    low, high = tris.min(1), tris.max(1)
    tested = 0
    crossing = []
    for i, j in combinations(range(len(faces)), 2):
        if np.any(low[i] > high[j]) or np.any(low[j] > high[i]):
            continue
        tested += 1
        if crosses_shared_boundary(points, faces[i], faces[j]):
            crossing.append((i, j))
    return dict(face_pairs=len(faces)*(len(faces)-1)//2, exact_pairs=tested, crossings=crossing)


def evaluate(capture):
    original = surface.evaluate(capture)
    palms = {}
    for side in ("left", "right"):
        path = capture / "gpu" / "native" / f"{side}_base_link.json"
        hull, _ = _pad_hull(json.loads(path.read_bytes()))
        palms[side] = dict(
            source_row_sha256=original["palms"][side]["gpu_row_sha256"],
            **audit_intersections(np.asarray(hull["vertices"]), np.asarray(hull["faces"])),
        )
    return dict(schema="dh116-power-palm-triangle-intersections-v1", palms=palms,
                parity_sha256=original["parity_sha256"], no_self_intersections=all(
                    not palm["crossings"] for palm in palms.values()),
                physx_shape_equivalence_verified=False, right_sdf_baked=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(args.capture.resolve(strict=True))
    surface.parity.write_json(args.output, result)
    return 0 if result["no_self_intersections"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
