"""Zero-step CUDA verification of alternate wire construction, without Isaac."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import material_mapping
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile, checked_file, digest
from go2_pvcnn.control.dh116.tabletop_supported_wire_specs import MATERIAL_NODES, verify_wire_getter


def probe(variants, selected):
    if (variants["version"] != "dh116-wire-variant-scene-candidates-v1"
            or variants["runtime_qualified"] is not False or variants["teacher_admission"] is not False
            or not 1 <= len(selected) <= 3 or len(set(selected)) != len(selected)):
        raise ValueError("one to three explicit, unqualified wire variant constructions required")
    specs, identities = {}, set()
    for group in variants["groups"].values():
        for name, case in group.items():
            identity = digest(dict(rod=case["rod"], length_m=case["length_m"]))
            if name in specs or identity in identities or identity != case["spec_sha256"]:
                raise ValueError("duplicate or inconsistent physical specification in variant manifest")
            specs[name] = case
            identities.add(identity)
    if not set(selected) <= set(specs):
        raise ValueError("requested wire specifications absent from the frozen catalog")
    profile = BackendProfile.load()
    binary = Path(profile.binary_path)
    sys.path.insert(0, str(binary.parent.parent))
    import ropesim
    import torch

    loaded = Path(ropesim._ropesim.__file__).resolve(strict=True)
    if loaded != binary or hashlib.sha256(loaded.read_bytes()).hexdigest() != profile.binary_sha256:
        raise ValueError("loaded CUDA solver differs from the original verified backend")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("one specifically selected CUDA device required")
    results = {}
    for name in selected:
        variant = specs[name]
        spec = dict(rod=variant["rod"], length_m=variant["length_m"])
        if digest(spec) != variant["spec_sha256"]:
            raise ValueError("variant content differs from its physical specification")
        nodes = np.asarray(variant["scene"]["rod_nodes_w_m"], dtype=np.float32)
        if nodes.shape != (MATERIAL_NODES, 3) or not np.isfinite(nodes).all():
            raise ValueError("actual 41-node constructor positions required")
        rod = spec["rod"]
        rope = ropesim.RopeBatch.from_positions(
            nodes[None], **material_mapping(dict(rod=rod, material_status="simulation_candidate"))["constructor"],
            pinned=np.zeros(MATERIAL_NODES, dtype=bool), fix_root_orientation=False,
            straight_rest=True, deterministic=True)
        rope.set_gravity(np.asarray(rod["gravity_world_m_s2"], dtype=np.float32))
        rope.set_contact(radius=rod["radius_m"], mu_s=rod["mu_s"], mu_k=rod["mu_k"],
                         v_slip=rod["v_slip_m_s"], kappa=rod["kappa"])
        raw = rope.parameter_snapshot()
        actual = verify_wire_getter(spec, raw, solver_time_s=0.)
        if rope.time != 0. or rope.last_error or rope.fixed_root_orientation:
            raise RuntimeError("zero-step free-root solver verification failed")
        torch.cuda.synchronize()
        results[name] = dict(**actual, spec_sha256=variant["spec_sha256"],
                             axial_stiffness_n=raw["stretch_ea_n"][0, 0].item(),
                             bend_stiffness_n_m2=raw["bend_ei_n_m2"][0, 0].item())
        del rope
    return dict(scope="standalone_cuda_constructor_and_parameter_getters", device=torch.cuda.get_device_name(0),
                backend_sha256=profile.binary_sha256, variants=results, isaac_steps=0,
                solver_steps=0, ppo_updates=0, grasp_qualified=False, teacher_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--variants-sha256", required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("independent output required")
    variants = json.loads(checked_file(args.variants, args.variants_sha256))
    with open("/tmp/dh116-isaac-single-instance.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = probe(variants, args.variant)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
