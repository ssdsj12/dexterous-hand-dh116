"""Measure fixed node/mesh force continuity in a sealed supported-wire archive."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import (
    DT, MIN_LINK_LOAD_N, NODE_COUNT, SUBSTEPS, _power_pairs,
)


def contact_runs(loaded):
    runs = []
    for (link, node), ticks in sorted(loaded.items()):
        start = previous = ticks[0]
        for tick in ticks[1:]:
            if tick <= previous:
                raise ValueError("loaded pair ticks must be unique and strictly increasing")
            if tick != previous + 1:
                runs.append(dict(link=link, node_index=node, first_rod_tick=start,
                                 last_rod_tick=previous, solver_steps=previous - start + 1))
                start = tick
            previous = tick
        runs.append(dict(link=link, node_index=node, first_rod_tick=start,
                         last_rod_tick=previous, solver_steps=previous - start + 1))
    return runs


def _read_json(path, expected_sha):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"archive SHA differs for {path.name}")
    return json.loads(raw)


def audit(receipt_path, receipt_sha, *, grasp, first, last):
    if grasp not in ("pinch", "tripod", "power") or not 3 <= first <= last:
        raise ValueError("supported grasp and nonempty post-Reset SDK interval required")
    receipt = _read_json(receipt_path, receipt_sha)
    if (any(type(receipt[k]) is not int or receipt[k] != 0
            for k in ("exit_code", "debugger_inferior_exit_code"))
            or receipt["forced_exit"] is not False or receipt["exceeded_budget"] is not False):
        raise ValueError("independent native exit 0 within budget required")
    archive = receipt_path.parent / "native"
    inventory = receipt["artifact_sha256"]
    geometry_name = f"{grasp}_reset_source_2.json"
    geometry = decode_owned(_read_json(archive / geometry_name, inventory[geometry_name]))
    radius_m = geometry["evidence"]["rod_radius_m"]
    if geometry["kind"] != "initial_geometry" or not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("measured Reset rod radius required")

    loaded, sample_counts = defaultdict(list), Counter()
    current_epoch = None
    current_binding_sha = None
    current_rest_lengths = None
    previous_rod_tick = first_rod_tick = None
    max_center_residual_m = None
    for sdk in range(first, last + 1):
        name = f"{grasp}_control_{sdk}.json"
        row = decode_owned(_read_json(archive / name, inventory[name]))
        rod = row["rod"]
        pairs = _power_pairs(rod)
        epoch = tuple(rod[key] for key in ("run_uuid", "episode_uid", "env_id", "reset_id", "rod_uid"))
        nodes = np.asarray(rod["node_positions"], dtype=float)
        rest_lengths = np.asarray(rod["rest_lengths_m"], dtype=float)
        binding = {"ordered_mesh_names": rod["mesh_names"], "bindings": rod["mesh_bindings"]}
        binding_sha = hashlib.sha256(json.dumps(binding, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if rest_lengths.shape != (NODE_COUNT - 1,) or not np.all(np.isfinite(rest_lengths) & (rest_lengths > 0)):
            raise ValueError("finite original material rest segments required")
        if (row["physics_step"] != sdk or rod["physics_step"] != sdk
                or nodes.shape != (1, NODE_COUNT, 3) or not np.isfinite(nodes).all()
                or current_epoch is not None and (epoch != current_epoch
                    or rod["rod_tick"] != previous_rod_tick + SUBSTEPS)):
            raise ValueError("continuous same-owner SDK and rod clock required")
        if current_epoch is None:
            current_epoch = epoch
            current_binding_sha = binding_sha
            current_rest_lengths = rest_lengths
            first_rod_tick = rod["rod_tick"] - SUBSTEPS + 1
        elif binding_sha != current_binding_sha or not np.array_equal(rest_lengths, current_rest_lengths):
            raise ValueError("mesh registry or material rest segments changed within one capture")
        previous_rod_tick = rod["rod_tick"]

        for pair in pairs:
            link = rod["mesh_names"][pair["mesh_index"]]
            if not link.startswith(("left.", "right.")):
                continue
            node = pair["node_index"]
            normal = np.asarray(pair["outward_normal_world"])
            sample_counts["raw_hand_pairs"] += 1
            if pair["substep"] == SUBSTEPS - 1:
                sphere_center = np.asarray(pair["contact_point_world_m"]) + radius_m * normal
                residual = float(np.linalg.norm(sphere_center - nodes[0, node]))
                max_center_residual_m = max(max_center_residual_m or 0., residual)
                sample_counts["endpoint_center_checks"] += 1
            load = float(normal @ np.asarray(pair["force_on_rod_world_n"]))
            if load > MIN_LINK_LOAD_N:
                step = rod["rod_tick"] - SUBSTEPS + 1 + pair["substep"]
                loaded[link, node].append(step)
                sample_counts["loaded_individual_pairs"] += 1

    longest_by_link = {}
    for run in contact_runs(loaded):
        link = run["link"]
        if link not in longest_by_link or run["solver_steps"] > longest_by_link[link]["solver_steps"]:
            longest_by_link[link] = {**run, "duration_s": run["solver_steps"] * DT / SUBSTEPS}
    return {
        "scope": "historical_native_fixed_node_mesh_pair_continuity",
        "grasp": grasp,
        "receipt_sha256": receipt_sha,
        "first_sdk_step": first,
        "last_sdk_step": last,
        "first_rod_tick": first_rod_tick,
        "last_rod_tick": previous_rod_tick,
        "mesh_binding_sha256": current_binding_sha,
        "sample_counts": {name: sample_counts[name] for name in (
            "raw_hand_pairs", "endpoint_center_checks", "loaded_individual_pairs")},
        "individual_pair_normal_threshold_n": MIN_LINK_LOAD_N,
        "longest_loaded_run_by_link": longest_by_link,
        "maximum_endpoint_sphere_center_residual_m": max_center_residual_m,
        "native_material_slip_m": None,
        "native_material_slip_qualified": False,
        "new_native_steps": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--grasp", required=True, choices=("pinch", "tripod", "power"))
    parser.add_argument("--first", required=True, type=int)
    parser.add_argument("--last", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.receipt, args.receipt_sha256, grasp=args.grasp, first=args.first, last=args.last)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
