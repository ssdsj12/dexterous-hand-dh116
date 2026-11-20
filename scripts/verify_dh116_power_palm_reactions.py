#!/usr/bin/env python3
"""Measure reconstructed left and right Power palm reactions in separate CUDA fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import DT, material_mapping
from go2_pvcnn.control.dh116.tabletop_deformable_solver_clock import solver_clock_matches
from go2_pvcnn.control.dh116.tabletop_deformx_pair_capture import validate_pair_capture
from go2_pvcnn.control.dh116.tabletop_material import hand_material_contract
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
from scripts.bake_dh116_deformx_pad_sdfs import checked_json, write_json


ROOT = Path(__file__).resolve().parents[1]
RAYS_SHA = "6e5a7a500a89f3081b660cfcea836fe426dea51989238d1f6b3631c2d8872659"
ROD_CONFIG_SHA = "9f751680de805c5075282c16bc5c5e5e30b8b7c261a3dd36f3b7b1cbab8f35c8"
GRID_RIGHT_SHA = "e5153e58c8b17b30d6ab2e2535a8c5d1bcf7c1d6f2c581fa6b00f0fa5fe53d7c"
GRID_LEFT_SHA = "824df146c5bd50c3e878a577e4f91ebce66b7b76fc2873c54aacef11c6a9e961"
BAKE = {"left": ("8316ca79897e63f12fb5c7908544d549312a93c9b3cb66beec64b78fd6bb9cdf", GRID_LEFT_SHA),
        "right": ("57727e345cb03586daeccd17a4e9589b68191740c2260595cea990e011e42884", GRID_RIGHT_SHA)}
OVERLAP_M = .00035


def _unit(vector):
    value = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(value)
    if value.shape != (3,) or not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("nondegenerate finite 3D direction required")
    return value / norm


def make_tangent_nodes(hit, normal, tangent_hint):
    outward = _unit(normal)
    hint = np.asarray(tangent_hint)
    tangent = _unit(hint - (hint @ outward) * outward)
    center = np.asarray(hit) + (.005 - OVERLAP_M) * outward
    return (center + np.linspace(-.25, .25, 41)[:, None] * tangent).astype(np.float32)[None]


def reaction_sign(force_on_palm, outward):
    signed = float(-force_on_palm @ _unit(outward))
    if not np.isfinite(signed) or signed <= 0.:
        raise ValueError("measured reaction must point inward along the palm normal")
    return signed



def audit_moment_origin(pairs, origin):
    """Independently verify the contact-point moment about the named palm mesh."""
    origin = np.asarray(origin, dtype=float)
    if not pairs or origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("contact pairs and a finite mesh origin are required")
    position, force, torque = (
        np.asarray([pair[key] for pair in pairs], dtype=float)
        for key in ("contact_point_world_m", "force_on_rod_world_n", "torque_on_mesh_about_origin_world_nm")
    )
    if any(x.shape != (len(pairs), 3) or not np.isfinite(x).all() for x in (position, force, torque)):
        raise ValueError("finite world position, force, and torque required")
    local_expected = np.cross(position - origin, -force)
    world_expected = np.cross(position, -force)
    mesh_residual = float(np.max(np.linalg.norm(torque - local_expected, axis=1)))
    world_residual = float(np.max(np.linalg.norm(torque - world_expected, axis=1)))
    if mesh_residual > 5e-5:
        raise ValueError("pair torques do not reconcile about the mesh origin")
    return dict(frame="world", about="mesh_origin", origin_world_m=origin.tolist(),
                mesh_origin_residual_nm=mesh_residual, world_origin_residual_nm=world_residual)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(side, rays, folder):
    bake_sha, grid_sha = BAKE[side]
    name = side + ".base_link"
    record = checked_json(folder/"result.json", bake_sha)["palms"][name]
    path = folder/(name+".npz")
    if sha(path) != grid_sha or record["sdf_file_sha256"] != grid_sha:
        raise ValueError("sealed original power palm grid required")
    with np.load(path, allow_pickle=False) as data:
        values, origin, spacing = data["values"], data["origin"], float(data["spacing"])
    axis = tuple(float(origin[k]) + spacing * np.arange(n) for k, n in enumerate(values.shape))
    sample = RegularGridInterpolator(axis, values, bounds_error=False, fill_value=np.inf)
    choices = [x for x in rays["rays"] if x["native_hit_distance_m"] is not None]
    valid = []
    for hit in choices:
        n = _unit(hit["native_normal_link"])
        for idx, hint in enumerate(np.eye(3)):
            if abs(n @ hint) > .85:
                continue
            points = make_tangent_nodes(hit["native_hit_link_m"], n, hint)
            phi = sample(points[0])
            if float(phi.min()) <= .001 or not np.any(phi < .00480):
                continue
            valid.append((int(np.count_nonzero(phi < .005)), -abs(float(phi[20])-.00465),
                          hit["triangle"], idx, hit, points, phi))
    if not valid:
        raise ValueError("no safe tangent rod fixture on the verified palm surface")
    count, _, index, tangent_axis, hit, local, phi = max(valid, key=lambda x:x[:4])
    rotation = Rotation.from_quat(record["quaternion_world_xyzw"])
    offset = np.asarray(record["position_world_m"])
    world = (rotation.apply(local[0]) + offset).astype(np.float32)[None]
    return (values, origin, spacing), world, record, dict(mesh=name, triangle=index, tangent_axis=tangent_axis,
        initial_contacts_predicted=count, minimum_initial_grid_m=float(min(phi)),
        outward_world=rotation.apply(hit["native_normal_link"]).tolist(),
        source_body=rays["source_body"], source_colliders=rays["source_colliders"], sdf_sha256=grid_sha)


def run_fixture(packet, cfg, ropesim, output):
    grids, world, pose, witness = packet
    r = cfg["rod"]
    hand = hand_material_contract(cfg["hand"]["material_profile"])["hand_contact_material"]
    rod = ropesim.RopeBatch.from_positions(world, **material_mapping(cfg)["constructor"],
        pinned=np.zeros(world.shape[1], dtype=bool), fix_root_orientation=False, straight_rest=True,
        deterministic=True)
    rod.set_gravity(np.asarray(r["gravity_world_m_s2"], dtype=np.float32))
    rod.set_contact(radius=r["radius_m"], mu_s=r["mu_s"], mu_k=r["mu_k"], v_slip=r["v_slip_m_s"], kappa=r["kappa"])
    rod.set_rope_contact(r["self_collision"])
    for key in ("damping", "drag", "drag_quad", "drag_quad_t"):
        setattr(rod, key, r[key])
    index = rod.add_mesh(*grids, pos=np.asarray(pose["position_world_m"], dtype=np.float32),
        quat=np.asarray(pose["quaternion_world_xyzw"], dtype=np.float32),
        mu_s=max(r["mu_s"], hand["static_friction"]), mu_k=max(r["mu_k"], hand["dynamic_friction"]))
    if index != 0 or rod.fixed_root_orientation or rod.num_envs != 1:
        raise ValueError("expected only the isolated original palm and one free rod")
    rod.enable_contact_forces(True)
    rod.enable_contact_pair_capture(41 * 12)
    start = float(rod.time)
    rod.step_free(DT, 12)
    raw = rod.contact_pair_samples()
    write_json(output / (witness["mesh"] + "-raw.json"),
               {name:value.tolist() if isinstance(value, np.ndarray) else value for name, value in raw.items()})
    if rod.last_error or not solver_clock_matches(start, float(rod.time), steps=1):
        raise ValueError("isolated palm solver step did not advance cleanly")
    node_forces = rod.contact_forces()[0]
    wrench = rod.mesh_wrenches()[0]
    measured = validate_pair_capture(raw, dt_s=DT, node_forces=node_forces, mesh_wrenches=wrench)
    residual = float(np.linalg.norm(node_forces.sum(axis=0) + wrench[0, :3]))
    if not measured["pairs"] or residual > 1e-3:
        raise ValueError("missing real palm pair data or failed aggregate reaction balance")
    moment = audit_moment_origin(measured["pairs"], pose["position_world_m"])
    reaction = reaction_sign(wrench[0, :3], witness["outward_world"])
    return dict(**witness, mesh_index=index, solver_steps=12, contact_pairs=len(measured["pairs"]),
                loaded_solver_substeps=len({x["substep"] for x in measured["pairs"]}),
                force_on_mesh_n=wrench[0, :3].tolist(), torque_about_mesh_origin_world_nm=wrench[0, 3:].tolist(), moment_reference_audit=moment,
                positive_inward_reaction_n=reaction, force_residual_n=residual, raw_validated=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rays", "left-bake", "right-bake"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rays = checked_json(args.rays, RAYS_SHA)["palms"]
    cfg = checked_json(ROOT/"configs/dh116_tasks/flexible_pinch_candidate_v1.json", ROD_CONFIG_SHA)
    packets = {side:fixture(side, rays[side], path) for side, path in (("left", args.left_bake), ("right", args.right_bake))}
    if args.output is None:
        print(json.dumps({side:item[-1] for side,item in packets.items()}))
        return 0
    profile = BackendProfile.load(ROOT)
    binary = Path(profile.binary_path).resolve(strict=True)
    sys.path.insert(0, str(binary.parents[1]))
    import ropesim
    from ropesim import _ropesim
    if Path(_ropesim.__file__).resolve() != binary or sha(binary) != profile.binary_sha256:
        raise ValueError("solver must match the current native DH116 profile")
    args.output.mkdir()
    result = {side:run_fixture(packet, cfg, ropesim, args.output) for side,packet in packets.items()}
    write_json(args.output/"result.json", dict(scope="independent_original_palm_solver_reaction_fixtures",
        configured_binary_sha256=profile.binary_sha256, ray_source_sha256=RAYS_SHA, palms=result,
        isaac_instances=0, ppo_updates=0, live_articulation_feedback=False, grasp_verified=False,
        teacher_admission=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
