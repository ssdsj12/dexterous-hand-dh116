#!/usr/bin/env python3
"""Compare batch tensor arithmetic with the archived native Pinch first window."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from go2_pvcnn.control.dh116.linear_coupling import JOINTS
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_privileged_teacher import _features
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import (
    CLEARANCE_M, project_policy_wrist_targets,
)
from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import prepare_scene
from go2_pvcnn.control.dh116.tabletop_teacher_tensor_math import (
    TensorSourceClearance, joint_force_guard, reward_window,
)


SIDES = ("left", "right")
GUARD_CODES = (None, "left:nonfinite_joint_state", "left:joint_limit", "left:mimic_error",
               "right:nonfinite_joint_state", "right:joint_limit", "right:mimic_error", "excess_force")


def _matrix(hands, field):
    return np.asarray([hands[side][field][0][..., [0, 1, 2, 6, 3, 4, 5]]
                       for side in SIDES], dtype=np.float64)


def _read_lanes(archive):
    paths = sorted(archive.glob("lane_*.jsonl"))
    if len(paths) != 64:
        raise ValueError("the original 64 independent lane journals are required")
    rows, windows, sources, body_name_sets = [], [], [], []
    for lane, path in enumerate(paths):
        seed = None
        sdk = []
        for line in path.open():
            event = json.loads(line)
            operation = event["operation"]
            if operation not in ("owner_event", "physical", "teacher_input_source", "batch_settlement"):
                continue
            payload = decode_owned(event["payload"])
            if operation == "owner_event" and payload["kind"] == "reset":
                hands = payload["reset"]["hands"]
                seed = np.asarray(payload["reset"]["seed"]["reset_root_pose_w"])
                previous_pose = _matrix(hands, "body_poses_xyzw")
                previous_clearance = payload["clearance_m"][0]
                previous_palms = previous_angles = np.zeros((2, 3))
                names = [hands[side]["body_names"] for side in SIDES]
                body_name_sets.append(names)
            elif operation == "physical":
                raw = payload["exchange"]
                rod = raw["rod"]
                if seed is None or rod["env_id"] != lane or raw["substep"] != len(sdk):
                    raise ValueError("same-lane consecutive archived sources required")
                sdk.append(rod["physics_step"])
                base = np.stack([previous_pose[i, names[i].index("base_link")] for i in range(2)])
                post = raw["action_readback"]["post"]["hands"]
                request = raw["action_readback"]["request"]
                projection = raw["wrist_projection"]
                new_poses = _matrix(raw["closure_geometry"]["raw"]["hands"], "link_poses_xyzw")
                new_clearance = np.asarray(raw["post_step_clearance"]["clearance_m"])[0]
                rows.append(dict(seed=seed, base=base, clearance=previous_clearance,
                    palms=previous_palms, angles=previous_angles, requested_palms=request["requested_palm_offsets_m"][0],
                    requested_angles=request["requested_angles_rad"][0],
                    projected_root=projection["projected_root_poses"][0], reasons=projection["projection_reasons"][0],
                    pose=new_poses, measured_clearance=new_clearance,
                    lowest=raw["post_step_clearance"]["lowest_shape_index"][0],
                    guard=GUARD_CODES.index(raw["original_guard_failure"]),
                    q=np.stack([post[s]["q"][0][[post[s]["joint_names"].index(n) for n in JOINTS]] for s in SIDES]),
                    forces=np.stack([rod["hand_wrenches_com_world"][s]["forces"][0] for s in SIDES])))
                previous_pose, previous_clearance = new_poses, new_clearance
                previous_palms, previous_angles = projection["palm_offsets_m"][0], projection["projected_angles"][0]
            elif operation == "teacher_input_source":
                packet = payload["buffers"]
                if (payload["purpose"] != "bootstrap" or payload["clock"]["env_id"] != lane
                        or payload["clock"]["physics_step"] != 10 or
                        packet["pre_action"]["common"]["identity"]["physics_step"] != 10):
                    raise ValueError("actual first-window bootstrap observation source required")
                sources.append(packet)
            else:
                if sdk != list(range(3, 11)) or payload["reward"]["endpoint"]["stamp"]["env_id"] != lane:
                    raise ValueError("the complete, owned first eight-tick window is required")
                windows.append(payload["reward"])
                break
    if (len(rows) != 512 or len(windows) != 64 or len(sources) != 64
            or any(names != body_name_sets[0] for names in body_name_sets)):
        raise ValueError("one complete step set and original body order per lane required")
    return rows, windows, sources, body_name_sets[0]


def audit(archive, device):
    rows, windows, sources, names = _read_lanes(archive)
    project = Path(__file__).resolve().parents[1]
    prepared = prepare_scene(project)
    shapes = [(i, names[i].index(name), collider["vertices"])
              for i, side in enumerate(SIDES)
              for name, collider in prepared["hands"][side]["source_colliders"].items()]

    def measure(where):
        def tensor(name):
            return torch.as_tensor(np.stack([row[name] for row in rows]), device=where)
        def reward_field(name):
            return torch.as_tensor(np.asarray([row[name] for row in windows]), device=where)

        floor, lowest = TensorSourceClearance(shapes, device=where)(tensor("pose"))
        status = joint_force_guard(tensor("q"), tensor("forces"), JOINTS)
        projection = project_policy_wrist_targets(
            root_seed_w=tensor("seed"), measured_base_w=tensor("base"),
            common_lift_offset_m=torch.zeros(len(rows), dtype=torch.float64, device=where),
            measured_clearance_m=tensor("clearance"), palm_offsets_m=tensor("palms"),
            requested_palm_offsets_m=tensor("requested_palms"), angles_rad=tensor("angles"),
            requested_angles_rad=tensor("requested_angles"),
            allowed=torch.ones(len(rows), dtype=torch.bool, device=where),
            live=torch.ones(len(rows), dtype=torch.bool, device=where))
        gamma = windows[0]["gamma"]
        if any(window["gamma"] != gamma for window in windows):
            raise ValueError("one frozen reward coefficient required")
        reward = reward_window(reward_field("phi_pre"), reward_field("phi_post_raw"),
                               reward_field("cost_per_second"), gamma, reward_field("event"),
                               reward_field("terminated"))
        fields = ("node_positions", "velocities", "tangent", "curvature", "rest_lengths_m", "deformation")
        snapshots = [packet["pre_action"]["rod_boundary"]["rod_snapshot"] for packet in sources]
        snapshot = {key: torch.as_tensor(np.concatenate([row[key] for row in snapshots]),
                                         dtype=torch.float32, device=where) for key in fields}
        common = torch.as_tensor(np.concatenate([p["observation"]["common371"] for p in sources]), device=where)
        mask = torch.as_tensor(np.concatenate([p["observation"]["rod_mask"] for p in sources]), device=where)
        features = _features(snapshot, common, mask)
        return {key: value.cpu().numpy() for key, value in dict(
            clearance=floor, lowest=lowest, guard=status, targets=projection["projected_root_poses"],
            projection_reasons=projection["projection_reasons"], reward=reward, rod_features=features).items()}

    cpu, cuda = measure("cpu"), measure(device)
    archived = dict(clearance=np.stack([row["measured_clearance"] for row in rows]),
                    lowest=np.asarray([row["lowest"] for row in rows]),
                    guard=np.asarray([row["guard"] for row in rows]),
                    targets=np.stack([row["projected_root"] for row in rows]),
                    projection_reasons=np.stack([row["reasons"] for row in rows]),
                    reward=np.asarray([window["reward"] for window in windows]),
                    rod_features=np.concatenate([packet["observation"]["rod_nodes"] for packet in sources]))
    numeric = ("clearance", "targets", "reward", "rod_features")
    error = {f"{label}_vs_archive_{key}": float(np.max(np.abs(candidate[key] - archived[key])))
             for label, candidate in (("cpu", cpu), ("cuda", cuda)) for key in numeric}
    error.update({f"cpu_vs_cuda_{key}": float(np.max(np.abs(cpu[key] - cuda[key]))) for key in numeric})
    flags = {f"{label}_{key}": bool(np.array_equal(candidate[key], archived[key]))
             for label, candidate in (("cpu", cpu), ("cuda", cuda))
             for key in ("lowest", "guard", "projection_reasons")}
    flags["cuda_clearance_guard"] = bool(np.array_equal(cuda["clearance"] < CLEARANCE_M,
                                                       archived["clearance"] < CLEARANCE_M))
    return dict(rows=len(rows), windows=len(windows), max_abs_error=error, flags_exact=flags,
                subset_passed=all(v < 1e-5 for v in error.values()) and all(flags.values()),
                complete_native_gpu_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("a real CUDA device is required for parity")
    result = audit(args.archive, "cuda:0")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["subset_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
