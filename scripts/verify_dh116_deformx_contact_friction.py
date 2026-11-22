"""Bounded standalone source probe; no Isaac, hands, Teacher or PPO.

Each of three isolated plane-mesh cases uses exactly four 480Hz rod advances
of twelve substeps. This is observation/physics parity evidence, not slip
uncertainty calibration or actual DH116 grasp qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

DT = 1/480
SUBSTEPS = 12
CASES = ("static", "translation", "rotation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("fresh exclusive probe output required")
    sys.path.insert(0, str(args.package_root.resolve(strict=True)))
    import ropesim
    import torch
    binary = Path(ropesim._ropesim.__file__).resolve()
    actual_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    if (binary.parent != (args.package_root/"ropesim").resolve()
            or actual_sha != args.binary_sha256):
        raise ValueError("exact independent loaded backend required")
    result = dict(binary_sha256=actual_sha, cases={}, new_solver_advances=0,
                  solver_substeps=0, isaac_steps=0, ppo_updates=0, slip_qualified=False)
    def owned(value):
        return torch.as_tensor(value).clone().cpu().numpy().copy()
    def packet(value):
        return {key: (array.tolist() if isinstance(array, np.ndarray) else array)
                for key, array in value.items()}
    for mode in CASES:
        nodes = np.zeros((1, 21, 3), np.float32)
        nodes[0, :, 0] = np.linspace(.02, .12, 21)
        nodes[0, :, 2] = .005
        rope = ropesim.RopeBatch.from_positions(nodes, .065, 800., .018, 1/5760, 8,
                    pinned=np.zeros(21, bool), deterministic=True)
        rope.set_gravity(np.array([0., 0., -9.81], np.float32))
        rope.set_contact(.005)
        grid = ropesim.sdf.from_function(lambda p: p[..., 2],
                            [-.1, -.1, -.03], [.3, .1, .03], .005)
        slot = rope.add_mesh(*grid.as_args())
        if slot != 0:
            raise ValueError("one controlled mesh required")
        rope.enable_contact_forces(True)
        rope.enable_contact_pair_capture(21*12)
        before = None
        target_pos = np.zeros(3, np.float32)
        target_quat = np.array([0., 0, 0, 1.], np.float32)
        for step in range(4):
            if step == 3:
                before = owned(rope.positions())
                if mode == "translation":
                    target_pos[0] = .01*DT
                elif mode == "rotation":
                    target_quat = Rotation.from_rotvec([0, 0, .1*DT]).as_quat().astype(np.float32)
            rope.set_mesh_poses(slot, target_pos, target_quat, teleport=False)
            rope.step_free(DT, SUBSTEPS)
            result["new_solver_advances"] += 1
            result["solver_substeps"] += SUBSTEPS
            if rope.last_error:
                raise RuntimeError("actual solver health failure")
        old = rope.contact_pair_samples()
        observed = rope.contact_pair_samples_v2()
        if (old["overflow"] or observed["overflow"] or old["substeps"] != 12
                or old["attempted"] != observed["attempted"]):
            raise ValueError("complete original pair capture required")
        record = dict(positions=owned(rope.positions()).tolist(),
            velocities=owned(rope.velocities()).tolist(), before_positions=before.tolist(),
            contact_forces=np.asarray(rope.contact_forces()).tolist(),
            mesh_wrenches=np.asarray(rope.mesh_wrenches()).tolist(),
            old_pair=packet(old), pose_pair=packet(observed),
            target_pos=target_pos.tolist(), target_quat=target_quat.tolist(),
            solver_time_s=float(rope.time))
        if args.extended:
            extra = rope.contact_friction_samples_v1()
            ids = np.asarray(extra["indices"])
            d = np.asarray(extra["solver_friction_displacement_world_m"])
            if (extra["schema"] != "dh116.sdf_pair_friction_samples.v1"
                    or extra["slip_qualified"] is not False
                    or extra["slip_uncertainty_m"] is not None
                    or d.shape != (extra["attempted"], 3) or not np.isfinite(d).all()
                    or extra["owner_time_s_before"] != extra["owner_time_s_after"]
                    or not np.array_equal(ids, old["indices"])
                    or not np.array_equal(extra["vectors"], old["vectors"])):
                raise ValueError("new observation is not the same complete owned native capture")
            record["friction_pair"] = packet(extra)
        result["cases"][mode] = record
        torch.cuda.synchronize()
        del rope
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(binary_sha256=actual_sha, cases=list(result["cases"]),
        new_solver_advances=result["new_solver_advances"], solver_substeps=result["solver_substeps"],
        isaac_steps=0, ppo_updates=0, slip_qualified=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
