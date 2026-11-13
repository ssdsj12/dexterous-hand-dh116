"""Versioned registered-worker correction; unchanged one-step CUDA calibration."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from go2_pvcnn.control.dh116.tabletop_deformx_pair_capture import validate_pair_capture
from scripts import verify_dh116_deformx_pair_capture as original

WORKER_ENV = "DH116_DEFORMX_PAIR_WORKER_V2"
ENTRY = "scripts/verify_dh116_deformx_pair_capture_v2.py"
DT = original.DT


def verify_worker(output, extension, expected_sha):
    path = Path(os.environ.get(WORKER_ENV, ""))
    if (not path.is_absolute() or path.name != "intent.json"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "3"):
        raise ValueError("calibration requires the registered GPU 3 scheduler worker")
    # evidence/native-attempts/package/fingerprint/attempt/intent.json
    attempt = path.parent
    fingerprint = attempt.parent
    package_dir = fingerprint.parent
    native_attempts = package_dir.parent
    evidence = native_attempts.parent
    if native_attempts.name != "native-attempts":
        raise ValueError("worker intent is outside the registered attempt hierarchy")
    from scripts import run_dh116_work_packages as scheduler

    intent = json.loads(path.read_text(encoding="utf-8"))
    repo = scheduler.Repository(Path(__file__).resolve().parents[1], evidence_dir=evidence)
    package = repo.registry()["packages"][intent["package"]]
    recipe = repo.native_recipe(package)
    identity = {key: recipe[key] for key in ("command_spec_sha256", "code_config_sha256")}
    approval = scheduler.native_approval(intent["approval"], repo)
    expected_argv = ["--output", "{output}", "--extension", str(extension),
                     "--extension-sha256", expected_sha, "--execute-native"]
    expected_output = Path(approval["output_root"]) / intent["package"] / attempt.name / "native"
    if (path.resolve() != repo.private / "native-attempts" / intent["package"]
            / identity["code_config_sha256"] / attempt.name / "intent.json"
            or repo.head() != intent["head"]
            or recipe["spec"]["argv"] != expected_argv
            or recipe["spec"]["entrypoint"] != ENTRY
            or identity != approval["packages"].get(intent["package"])
            or any(intent[key] != value for key, value in identity.items())
            or output.resolve() != Path(intent["output"])
            or output.resolve() != expected_output):
        raise ValueError("worker output or current recipe does not match the registered approval")


# Copied from frozen v1; only the post-step error check and its scalar record differ.
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
    solver_last_error = int(rod.last_error)
    if solver_last_error != 0:
        raise ValueError(f"solver last_error after calibration step: {solver_last_error}")
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
            "stale_reset_rejected": stale_reset_rejected, "solver_last_error": solver_last_error}, raw


def validate_artifacts(output):
    original.validate_artifacts(output)
    report = json.loads((output / "pairs.json").read_text(encoding="utf-8"))
    error = report.get("solver_last_error")
    if type(error) is not int or error != 0:
        raise ValueError("calibration did not record a zero solver last_error")


def main():
    bindings = {
        "WORKER_ENV": WORKER_ENV,
        "verify_worker": verify_worker,
        "run_calibration": run_calibration,
    }
    saved = {name: getattr(original, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(original, name, value)
        return original.main()
    finally:
        for name, value in saved.items():
            setattr(original, name, value)


if __name__ == "__main__":
    main()
