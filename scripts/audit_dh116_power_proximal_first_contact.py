#!/usr/bin/env python3
"""Classify the predicted initial little-finger approach direction using sealed SDFs."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_supported_rod_reference import load_parked_reference
from scripts.audit_dh116_pinch_initial_pose import load_geometry
from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_grid
from scripts.verify_dh116_power_shared_rod import native, proximal_registration_inputs

ROOT = Path(__file__).resolve().parents[1]
ROD_RECEIPT_SHA = "c1683e8b44aa5e123aeee49a3adfd070fdf738ea20858ff0106a5756abda0aea"
ROUTE_SWEEP_SHA = "400b15c016b04db34eb94493f5550d4c0e91b0067d388051863e0e8c4101c073"
MODES = ("translate_alone","undo_rotation_first","simultaneous_wrap","translate_rotate_then_close")


def transverse_alignment(normal, rod_direction, to_palm):
    normal, rod_direction, to_palm = (np.asarray(vec, dtype=float)
                                      for vec in (normal, rod_direction, to_palm))
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (normal, rod_direction, to_palm)):
        raise ValueError("finite world 3-vectors required")
    n, rod = np.linalg.norm(normal), np.linalg.norm(rod_direction)
    if min(n, rod) <= 1e-12:
        raise ValueError("nondegenerate original normal and rod direction required")
    axis = rod_direction/rod
    transverse = to_palm - (to_palm@axis)*axis
    size = np.linalg.norm(transverse)
    return None if size <= 1e-12 else float(np.clip((normal/n)@(transverse/size), -1., 1.))


def first_overlap(grid, link, palm, cloud, nodes, *, subdivisions):
    local = (cloud-link[:3,3])@link[:3,:3]
    phi, grad, valid = sample_grid(grid, local, dtype=np.float64)
    overlap = np.flatnonzero(valid & (phi < .005))
    if not len(overlap):
        return None
    index = int(overlap[np.argmin(phi[overlap])])
    station = index/subdivisions
    segment = min(int(station), len(nodes)-2)
    normal = link[:3,:3]@grad[index]
    alignment = transverse_alignment(normal,nodes[segment+1]-nodes[segment],palm[:3,3]-cloud[index])
    normal /= np.linalg.norm(normal)
    return dict(station=station, grid_gap_m=float(phi[index]-.005), toward_palm_cosine=alignment,
                outward_normal_world=normal.tolist(), point_world_m=cloud[index].tolist())


def audit(routes, reference, packets):
    modes = defaultdict(list)
    for line in routes.read_text().splitlines():
        row = json.loads(line)
        modes[row["side"],row["mode"]].append(row)
    nodes = reference.node_positions
    f = np.arange(5)/5
    dense = ((1-f[None,:,None])*nodes[:-1,None]+f[None,:,None]*nodes[1:,None]).reshape(-1,3)
    clouds = (("solver_node",nodes,1),("subdivided_polyline",np.vstack((dense,nodes[-1])),5))
    result = []
    if set(modes) != {(side,mode) for side in ("left","right") for mode in MODES}:
        raise ValueError("complete matched bilateral route grid required")
    for side in ("left","right"):
        geometry = load_geometry(ROOT, side=side)
        _, grid, binding = packets[side+".finger52_link"]
        for mode in MODES:
            first, source = {}, None
            rows = modes[side,mode]
            if not np.array_equal([row["t"] for row in rows],np.linspace(0,1,21)):
                raise ValueError("exact archived route sampling order required")
            for row in rows:
                if source is None and row["min_rod_gap_m"] < 0.:
                    shape = min(row["rod_gap_m"], key=row["rod_gap_m"].get)
                    source = dict(t=row["t"], first_unproved_shape=shape, gap_lower_m=row["min_rod_gap_m"])
                fk = fk_poses(geometry,row["pose"],row["active"])
                for kind, cloud, n in clouds:
                    if kind in first:
                        continue
                    sample = first_overlap(grid,fk["finger52_link"],fk["base_link"],cloud,nodes,subdivisions=n)
                    if sample is not None:
                        first[kind] = dict(t=row["t"], **sample)
            result.append(dict(side=side, mode=mode, source_envelope=source,
                               predicted_overlap=first, grid_sha256=binding["grid_sha256"]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("route-sweep","parked-receipt","capture","bake","output"):
        parser.add_argument("--"+option, type=Path, required=True)
    args = parser.parse_args(argv)
    if native.sha(args.route_sweep) != ROUTE_SWEEP_SHA:
        raise ValueError("sealed full Power route sweep required")
    reference = load_parked_reference(args.parked_receipt,ROD_RECEIPT_SHA,240)
    packets,_ = proximal_registration_inputs(args.capture,args.bake)
    result = dict(scope="power_proximal_sampled_overlap_direction",
        routes=audit(args.route_sweep,reference,packets), route_sweep_sha256=ROUTE_SWEEP_SHA,
        native_reference_sample_sha256=reference.sample_sha256,
        physical_contact_verified=False, grasp_verified=False, teacher_admission=False)
    native.write_new(args.output,result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
