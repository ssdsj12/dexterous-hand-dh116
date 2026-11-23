"""Compare two measured Tripod resets and measure archived supported-wire drift."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_fast_owned_codec import decode_owned, encode_owned

NODE_COUNT = 41
WINDOW_STEPS = 8
SOLVER_SUBSTEPS = 12


def array_agreement(old, new):
    a, b = np.asarray(old), np.asarray(new)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("finite, nonempty arrays with matching shape required")
    error = float(np.max(np.abs(a.astype(float) - b.astype(float))))
    return {"byte_equal": a.dtype == b.dtype and a.tobytes() == b.tobytes(),
            "max_abs_difference": error}


def window_displacements(positions):
    """Observed maximum displacement from each archived command boundary."""
    points = np.asarray(positions, dtype=float)
    if (points.ndim != 3 or points.shape[1:] != (NODE_COUNT, 3)
            or len(points) <= WINDOW_STEPS or (len(points) - 1) % WINDOW_STEPS
            or not np.isfinite(points).all()):
        raise ValueError("finite rod positions for complete eight-step windows required")
    return np.array([
        np.linalg.norm(points[i + 1:i + 1 + WINDOW_STEPS] - points[i], axis=-1).max(axis=0)
        for i in range(0, len(points) - 1, WINDOW_STEPS)
    ])


def _read(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"SHA differs for {path.name}")
    return decode_owned(json.loads(raw))


def _exited(receipt_path, digest):
    receipt = _read(receipt_path, digest)
    if (any(type(receipt[k]) is not int or receipt[k] != 0
            for k in ("exit_code", "debugger_inferior_exit_code"))
            or receipt["forced_exit"] is not False or receipt["exceeded_budget"] is not False
            or receipt["debugger_status_verified"] is not True
            or receipt["debugger_inferior_signal"] is not None):
        raise ValueError("independent natural Isaac exit within budget required")
    return receipt_path.parent / "native", receipt["artifact_sha256"]


def _reset_arrays(directory, inventory):
    capture = {
        key: _read(directory / name, inventory[name]) for key, name in (
            ("numeric", "mesh_reset_numeric_state.json"), ("geometry", "reset_geometry.json"),
            ("episode", "episode_reset.json"), ("common", "pre_action_common.json"),
        )
    }
    numeric = capture["numeric"]
    if (numeric["operation_completed"] is not True or len(numeric["samples"]) != 2
            or not numeric["byte_equal"]):
        raise ValueError("complete unchanged numeric Reset readback required")
    state = numeric["samples"][-1]["state"]
    episode = capture["episode"]["reset"]
    common = capture["common"]
    valid = np.asarray(common["valid"])
    if (episode["status"] != "complete" or state["last_error"] != 0
            or valid.shape != (371,) or valid.dtype.kind != "b" or int(valid.sum()) != 66):
        raise ValueError("completed Reset and exact 66-field validity required")
    arrays = {f"rod.{k}": state[k] for k in ("positions", "velocities", "quaternions")}
    arrays.update(common_valid=valid, common_valid_values=np.asarray(common["values"])[valid])
    shapes = 0
    if [row["side"] for row in capture["geometry"]] != ["left", "right"]:
        raise ValueError("independent left and right reset geometry required")
    for side, hand in episode["hands"].items():
        if side not in ("left", "right"):
            raise ValueError("unexpected hand identity in Reset")
        for field in ("q", "qd", "backend_targets", "root_pose_xyzw", "body_poses_xyzw"):
            arrays[f"{side}.{field}"] = hand[field]
    for row in capture["geometry"]:
        for group in ("source_colliders", "locked_pads"):
            for link, shape in row["evidence"][group].items():
                arrays[f"{row['side']}.{group}.{link}"] = shape["world_vertices"]
                shapes += 1
    if shapes != 44 or set(episode["hands"]) != {"left", "right"}:
        raise ValueError("34 source and 10 cooked shapes and both hands required")
    task = _read(directory / "frozen_plan.json", inventory["frozen_plan.json"])["task_config"]
    return arrays, dict(seed_sha256=episode["seed"]["sha256"], task_sha256=task["sha256"],
                         common_contract_sha256=common["contract_sha256"],
                         solver_parameters=encode_owned(state["parameters"]))


def _historical_positions(directory, inventory):
    capture = _read(directory / "settling_capture.json", inventory["settling_capture.json"])
    windows = capture["diagnostic_control_windows"]
    expected_rows = windows * WINDOW_STEPS + 1
    if (windows != 30 or capture["failure"] is not None or capture["on_policy"] is not False
            or capture["ppo_updates"] != 0 or capture["persisted_physics_rows"] != expected_rows
            or capture["source_valid_rows"] != expected_rows):
        raise ValueError("completed historical zero-action settling required")
    digest = hashlib.sha256()
    positions, speeds = [], []
    episode = None
    with (directory / "physics.jsonl").open("rb") as stream:
        for index, line in enumerate(stream):
            digest.update(line)
            if not line.endswith(b"\n"):
                raise ValueError("incomplete archived physics row")
            entry = decode_owned(json.loads(line))
            row = entry["payload"]["exchange"]
            rod = row["rod"]
            clock = tuple(rod[k] for k in ("run_uuid", "episode_uid", "env_id", "reset_id", "rod_uid"))
            if episode is None:
                episode = clock
            window, substep = divmod(max(0, index - 1), WINDOW_STEPS)
            command = np.asarray(row["action_readback"]["request"]["action24"])
            if (entry["raw_sequence"] != index or entry["payload"]["on_policy"] is not False
                    or clock != episode or row["physics_step"] != index + 3
                    or rod["physics_step"] != index + 3 or rod["rod_tick"] != SOLVER_SUBSTEPS * (index + 1)
                    or (row["policy_step"], row["substep"]) != (window, substep)
                    or row["guard_failure"] is not None or command.shape != (1, 24)
                    or not np.isfinite(command).all() or np.any(command)):
                raise ValueError("consecutive same-episode zero-action evidence required")
            points = np.asarray(rod["node_positions"], dtype=float)
            velocity = np.asarray(rod["velocities"], dtype=float)
            if (points.shape != (1, NODE_COUNT, 3) or not np.isfinite(points).all()
                    or velocity.shape != points.shape or not np.isfinite(velocity).all()):
                raise ValueError("complete finite historical material state required")
            positions.append(points[0])
            speeds.append(float(np.linalg.norm(velocity[0], axis=-1).max()))
    if digest.hexdigest() != inventory["physics.jsonl"] or len(positions) != expected_rows:
        raise ValueError("complete SHA-bound historical physics journal required")
    return positions, speeds


def audit(historical_receipt, historical_sha, current_receipt, current_sha):
    old, old_files = _exited(historical_receipt, historical_sha)
    new, new_files = _exited(current_receipt, current_sha)
    old_arrays, old_ids = _reset_arrays(old, old_files)
    new_arrays, new_ids = _reset_arrays(new, new_files)
    if set(old_arrays) != set(new_arrays):
        raise ValueError("Reset fields or shape identities differ")
    agreement = {k: array_agreement(old_arrays[k], new_arrays[k]) for k in old_arrays}
    positions, speeds = _historical_positions(old, old_files)
    displacements = window_displacements(positions)
    return {
        "historical_receipt_sha256": historical_sha, "current_receipt_sha256": current_sha,
        "measured_reset_matches": old_ids == new_ids and all(v["byte_equal"] for v in agreement.values()),
        "reset_fields": agreement, "reset_seed_sha256": new_ids["seed_sha256"],
        "historical_physics_sha256": old_files["physics.jsonl"], "historical_rows": len(positions),
        "historical_window_peak_displacement_m": displacements.tolist(),
        "historical_peak_node_speed_m_s": max(speeds),
        "historical_last_window_peak_node_speed_m_s": max(speeds[-WINDOW_STEPS:]),
        "scope": "measured_reset_correspondence_and_historical_zero_action_only",
        "current_settling_execution_verified": False,
        "teacher_admission": False, "grasp_qualified": False, "new_native_steps": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("historical", "current"):
        parser.add_argument(f"--{label}-receipt", type=Path, required=True)
        parser.add_argument(f"--{label}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.historical_receipt, args.historical_sha256, args.current_receipt, args.current_sha256)
    with args.output.open("x") as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
