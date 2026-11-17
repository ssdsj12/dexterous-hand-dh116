"""CPU preflight or one gated CUDA-only contact instrumentation calibration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from go2_pvcnn.control.dh116.tabletop_deformx_pair_capture import validate_pair_capture
from go2_pvcnn.control.dh116.tabletop_stereo_native import native_instance_lock

WORKER_ENV = "DH116_DEFORMX_PAIR_WORKER_V1"
SCOPE = "one_free_rod_one_table_sdf_pair_capture_calibration"
DT = 1 / 480


def verify_worker(output, extension, expected_sha):
    path = Path(os.environ.get(WORKER_ENV, ""))
    if (not path.is_absolute() or path.name != "intent.json"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "3"):
        raise ValueError("calibration requires the registered GPU 3 scheduler worker")
    from scripts import run_dh116_work_packages as scheduler

    intent = json.loads(path.read_text(encoding="utf-8"))
    repo = scheduler.Repository(Path(__file__).resolve().parents[1], evidence_dir=path.parents[3])
    package = repo.registry()["packages"][intent["package"]]
    recipe = repo.native_recipe(package)
    identity = {key: recipe[key] for key in ("command_spec_sha256", "code_config_sha256")}
    approval = scheduler.native_approval(intent["approval"], repo)
    expected_argv = ["--output", "{output}", "--extension", str(extension),
                     "--extension-sha256", expected_sha, "--execute-native"]
    if (path.resolve() != repo.private / "native-attempts" / intent["package"]
            / identity["code_config_sha256"] / path.parent.name / "intent.json"
            or repo.head() != intent["head"]
            or recipe["spec"]["argv"] != expected_argv
            or recipe["spec"]["entrypoint"] != "scripts/verify_dh116_deformx_pair_capture.py"
            or identity != approval["packages"].get(intent["package"])
            or any(intent[key] != value for key, value in identity.items())
            or output.resolve() != Path(intent["output"])):
        raise ValueError("worker output or current recipe does not match the registered approval")


def validate_artifacts(output):
    raw = json.loads((output / "raw_pairs.json").read_text(encoding="utf-8"))
    report = json.loads((output / "pairs.json").read_text(encoding="utf-8"))
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    if (raw["mesh_count"] != 1 or raw["substeps"] != 12 or raw["capacity"] != 36
            or len(raw["node_forces_on_rod_world_n"]) != 3):
        raise ValueError("packet differs from the one-mesh, twelve-substep budget")
    pairs = validate_pair_capture(raw, dt_s=DT,
        node_forces=raw["node_forces_on_rod_world_n"],
        mesh_wrenches=raw["mesh_wrenches_on_mesh_world"])
    if ({key: report[key] for key in pairs} != pairs or result["pair_count"] != len(pairs["pairs"])
            or not any(row["normal_force_on_rod_n"] > 0 for row in pairs["pairs"])
            or not np.isclose(report["solver_clock_after_s"] - report["solver_clock_before_s"],
                              DT, atol=1e-7, rtol=0)
            or report["stale_reset_rejected"] is not True):
        raise ValueError("stored calibration evidence does not reconcile")


def run_calibration(extension, expected_sha):
    sys.path.insert(0, str(extension.parent.parent))
    from ropesim import RopeBatch, _ropesim, sdf

    loaded = Path(_ropesim.__file__).resolve(strict=True)
    if loaded != extension or hashlib.sha256(loaded.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("loaded solver binary differs from the frozen candidate")
    nodes = np.array([[[0., -.01, .004], [0., 0., .004], [0., .01, .004]]], dtype=np.float32)
    rod = RopeBatch.from_positions(nodes, density=.065, k_stretch=800., k_bend=.018,
        max_h=DT / 12, num_itr=8, fix_root_orientation=False,
        pinned=np.zeros(3, dtype=bool), straight_rest=True, deterministic=True)
    rod.set_gravity(np.array([0., 0., -9.81], dtype=np.float32))
    rod.set_contact(radius=.005, mu_s=.8, mu_k=.6, v_slip=.001, kappa=1.5)
    grid = sdf.from_function(lambda p: p[..., 2], [-.05, -.05, -.02], [.05, .05, .04], .005)
    slot = rod.add_mesh(*grid.as_args(), pos=np.zeros(3, dtype=np.float32),
                        quat=np.array([0, 0, 0, 1], dtype=np.float32), mu_s=.8, mu_k=.6)
    if slot != 0:
        raise ValueError("expected one identified calibration mesh")
    rod.enable_contact_forces(True)
    rod.enable_contact_pair_capture(3 * 12)
    clock_before = float(rod.time)
    rod.step_free(DT, 12)
    packet = rod.contact_pair_samples()
    result = validate_pair_capture(packet, dt_s=DT,
        node_forces=packet["node_forces_on_rod_world_n"],
        mesh_wrenches=packet["mesh_wrenches_on_mesh_world"])
    raw = {key: value.tolist() if isinstance(value, np.ndarray) else value
           for key, value in packet.items()}
    clock_after = float(rod.time)
    if not np.isclose(clock_after - clock_before, DT, atol=1e-7, rtol=0):
        raise ValueError("solver clock differs from the one-step budget")
    if not any(row["normal_force_on_rod_n"] > 0 for row in result["pairs"]):
        raise ValueError("native calibration did not produce measured positive contact")
    rod.reset_envs([0])
    try:
        rod.contact_pair_samples()
    except ValueError:
        stale_reset_rejected = True
    else:
        raise ValueError("reset exposed stale contact pairs")
    del rod
    return {**result, "solver_clock_before_s": clock_before, "solver_clock_after_s": clock_after,
            "stale_reset_rejected": stale_reset_rejected}, raw


def write_new(path, record):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--execute-native", action="store_true")
    args = parser.parse_args()
    ext = args.extension.resolve(strict=True)
    if hashlib.sha256(ext.read_bytes()).hexdigest() != args.extension_sha256:
        parser.error("frozen extension SHA256 differs")
    if ext.parent.name != "ropesim" or not ext.name.startswith("_ropesim.cpython-310-"):
        parser.error("expected the isolated cp310 ropesim extension")
    plan = {"scope": SCOPE, "extension_sha256": args.extension_sha256, "cuda_solver_steps": 1,
            "rod_substeps": 12, "isaac_physics_steps": 0, "ppo_updates": 0}
    if not args.execute_native:
        print(json.dumps({**plan, "native_execution_performed": False}))
        return
    verify_worker(args.output, ext, args.extension_sha256)
    with native_instance_lock():
        args.output.mkdir(parents=True, exist_ok=False)
        write_new(args.output / "plan.json", plan)
        result, raw = run_calibration(ext, args.extension_sha256)
        write_new(args.output / "raw_pairs.json", raw)
        write_new(args.output / "pairs.json", result)
        write_new(args.output / "result.json", {**plan, "native_execution_performed": True,
            "reconciled_pairs": True, "pair_count": len(result["pairs"]),
            "stale_reset_rejected": result["stale_reset_rejected"]})


if __name__ == "__main__":
    main()
