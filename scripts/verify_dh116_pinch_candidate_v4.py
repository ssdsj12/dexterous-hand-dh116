#!/usr/bin/env python3
"""Bounded fixed-base left Pinch with measured-node closure and explicit cleanup."""

from copy import deepcopy
from functools import partial
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from go2_pvcnn.control.dh116 import tabletop_deformable_grasp_runtime as runtime
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, active_limits
from go2_pvcnn.control.dh116.tabletop_deformable_pinch_force_precision import validate_ledger
from go2_pvcnn.control.dh116.tabletop_simulation_owner_cleanup import (
    clear_known_simulation_instance, probe_simulation_owner_edges, simulation_source_contract,
)
from scripts import dh116_pinch_acquisition_v2 as native
from scripts import verify_dh116_pinch_candidate_v3 as prior
from scripts.dh116_pinch_startup_v3 import StartupTrace, snapshot_for_startup
from scripts.probe_dh116_physx_lifetime import write_json
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings


ENTRY = "scripts/verify_dh116_pinch_candidate_v4.py"
WRAPPER = "scripts/run_dh116_pinch_work_packages_v4.py"
VERSION = "dh116-single-left-pinch-v4"
WORKER_ENV = "DH116_PINCH_WORKER_V4"
POSE = "configs/dh116_tasks/flexible_pinch_pose_candidate_v4.json"
SCOPE = "single_left_fixed_base_free_rod_measured_node_contact_v4"
TABLE_FLOOR_M = .0019
PROOF = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/"
             "pinch-closure-margin-20261113-_dmnkcda/export/candidate.json")
PROOF_SHA = "3d04ccca63263b4480e1831693b8d7c70f1336170f2006b5c6a3a1f414553fe0"
OFFLINE_PROOF_SHA = "448630ca4f54c3b4b35dbeb4246201445aff1fc55753c7631412062b80389e1b"
V3_RECEIPT = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/"
    "flex.f4-pinch-left-v3-native/113882bbab640846e6b3d3d5042bdfa6487ddbffbd497869d71acd8e11eb8362/"
    "attempt-dsfwx5kr/receipt.json")
NEW_SOURCES = (ENTRY, WRAPPER, POSE, "scripts/validate_dh116_pinch_v4.py",
    "scripts/prepare_dh116_pinch_closure_margin.py", "tests/test_dh116_pinch_v4_worker.py",
    "tests/test_dh116_pinch_v4_audit.py",
    "src/go2_pvcnn/control/dh116/tabletop_simulation_owner_cleanup.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_pinch_force_precision.py")


def read(path):
    return json.loads(Path(path).read_bytes())


def validate_candidate(cfg, original, proof, contract):
    """Allow only the identified pose and two closure rates in the original scope."""
    expected = deepcopy(original)
    expected["scope"] = SCOPE
    pose = expected["initial_pose"]
    action = proof["action"]
    seed = proof["candidate_initial_q_rad"]
    endpoint = proof["target_end_q_rad"]
    original_pose = original["initial_pose"]
    lower, upper = active_limits()
    gates = proof["gates"]
    quiet = proof["quiet_counterfactual"]
    if (proof["scope"] != "single_left_fixed_base_offline_contact_discovery_only"
            or proof["side"] != "left" or proof["base_fixed"] is not True
            or proof["base_pose_world_wxyz"] != original_pose["base_pose_wxyz"]
            or proof["original_initial_q_rad"] != original_pose["active_q_rad"]
            or proof["original_geometry_source_sha256"] != original_pose["source_sha256"]
            or proof["active_joint_names"] != list(ACTIVE)
            or proof["sdf_bindings"] != contract["pad_bindings"]
            or any(proof[field] is not False for field in ("material_configuration_changed",
                "rod_seed_changed", "native_ready", "contact_qualified", "teacher_admitted"))):
        raise ValueError("offline evidence does not bind the original fixed-base scope")
    if (len(seed) != 6 or len(endpoint) != 6 or seed[3:] != original_pose["active_q_rad"][3:]
            or any(not lo <= value <= hi for values in (seed, endpoint)
                   for value, lo, hi in zip(values, lower, upper))
            or (action["physics_hz"], action["quiet_steps"], action["move_steps"],
                action["hold_steps"], action["active_velocity_limit_rad_s"]) != (480,48,160,30,.3)
            or gates["single_hand_static_table_minimum_m"] != TABLE_FLOOR_M
            or not TABLE_FLOOR_M <= gates["actual_table_lower_m"] < .0021
            or gates["policy_projection_qualified"] is not False
            or not gates["non_target_geometry_lower_m"] > 0
            or not gates["initial_all_shape_rod_lower_m"] > 0
            or quiet["frames"] != 48 or quiet["predicted_contact_predicates"] != 0
            or not quiet["minimum_in_lattice_gap_m"] > 0):
        raise ValueError("measured-node closure geometry or original bounds differ")
    action24 = action["canonical_action24"]
    velocities = action["thumb_index_velocity_rad_s"]
    if (len(action24) != 24 or len(velocities) != 2
            or any(value != 0 for i, value in enumerate(action24) if i not in (1,2))
            or any(not 0 < value <= 1 for value in action24[1:3])
            or not all(math.isclose(a*.3, v, rel_tol=0, abs_tol=1e-12)
                       and math.isclose(seed[i]+v/3, endpoint[i], rel_tol=0, abs_tol=1e-12)
                       for i, a, v in zip((1,2), action24[1:3], velocities))
            or any(seed[i] != endpoint[i] for i in (0,3,4,5))):
        raise ValueError("two bounded independent digit rates required")
    pose["active_q_rad"] = seed
    pose["approach"] = dict(move_steps=160, physics_hz=480, velocity_rad_s=velocities,
        active_delta_rad=[end-start for start, end in zip(seed, endpoint)],
        original_active_velocity_limit_rad_s=.3, palm_motion="fixed_base", is_commanded_FK_only=True)
    expected["action"].update(thumb_velocity_normalized=action24[1],
                              index_velocity_normalized=action24[2])
    if cfg != expected:
        raise ValueError("candidate differs from original physics or frozen offline pose/action proof")
    return cfg


def preflight(project):
    """Read source and evidence bytes only, leaving USD and Kit unloaded."""
    contract, original = prior.preflight(project)
    if prior.sha(PROOF) != PROOF_SHA:
        raise ValueError("offline candidate evidence bytes changed")
    offline = PROOF.parent.parent / "offline-proof.json"
    if prior.sha(offline) != OFFLINE_PROOF_SHA:
        raise ValueError("independent offline proof bytes changed")
    proof = read(PROOF)
    cfg = validate_candidate(read(project / POSE), original, proof, contract)
    sources = deepcopy(contract["sources"])
    for name in NEW_SOURCES:
        sources["local_sources"][name] = prior.sha(project / name)
    external = sources["external_sources"]
    for name, expected in ((V3_RECEIPT, proof["original_native_receipt_sha256"]),
            (proof["original_path_result_path"], proof["original_path_result_sha256"]),
            (proof["candidate_result_path"], proof["candidate_result_sha256"]),
            (PROOF, PROOF_SHA), (offline, OFFLINE_PROOF_SHA)):
        if prior.sha(name) != expected:
            raise ValueError("closure provenance bytes differ")
        external[str(name)] = expected
    v3_output = Path(read(V3_RECEIPT)["output"])
    for name, expected in proof["original_native_ledger_sha256"].items():
        path = v3_output / name
        if prior.sha(path) != expected:
            raise ValueError("original native ledger changed")
        external[str(path)] = expected
    release = simulation_source_contract()
    for row in release["sources"].values():
        external[row["path"]] = row["sha256"]
    fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()
    return dict(contract, version=VERSION, scope=SCOPE, code_config_sha256=fingerprint,
        sources=sources, pose_candidate_sha256=prior.sha(project / POSE),
        cpu_closure_proof_sha256=PROOF_SHA, simulation_clear_source_contract=release,
        policy_wrist_clearance_qualified=False, full_state371_bound=False), cfg


def require_table_floor(packet):
    rows = [*packet["source_colliders"].values(), *packet["locked_pads"].values()]
    if not rows or packet["initial_geometry_passed"] is not True or any(
            row["table"]["intersects"] is not False
            or not math.isfinite(row["table"]["top_clearance_m"])
            or row["table"]["top_clearance_m"] < TABLE_FLOOR_M for row in rows):
        raise ValueError("actual initial source/locked geometry fails fixed-base 1.9 mm floor")
    return min(row["table"]["top_clearance_m"] for row in rows)


class ReleaseTrace(StartupTrace):
    def checkpoint(self, phase):
        super().checkpoint(phase)
        name = phase.replace(" ", "_").replace(",", "")
        reference = self.references.get("simulation")
        record = (probe_simulation_owner_edges(reference) if reference is not None
                  else dict(alive=None, reason="no_simulation_reference_registered"))
        write_json(self.path.parent / (name + ".simulation-owners.json"), record)


def _clear_sim(sim, *, output, contract):
    try:
        receipt = clear_known_simulation_instance(sim, contract=contract)
        write_json(output / "simulation_clear_receipt.json", receipt)
    finally:
        sim = None


def execute(args, contract, cfg, project):
    boundary = prior.assert_clean_import_boundary()
    write_json(args.output / "before_app_launcher.json", boundary)
    old_builder, old_lifecycle = native.build_native, native.run_lifecycle

    def build_checked(*args, **kwargs):
        bundle = None
        try:
            bundle = old_builder(*args, **kwargs)
            require_table_floor(read(output / "initial_geometry.json"))
            return bundle
        finally:
            args = kwargs = bundle = None

    def lifecycle(*args, **kwargs):
        try:
            kwargs["clear_sim"] = partial(_clear_sim, output=output,
                contract=contract["simulation_clear_source_contract"])
            return old_lifecycle(*args, **kwargs)
        finally:
            args = kwargs = None

    output = args.output
    with bindings((native, "VERSION", VERSION), (native, "LifetimeTrace", ReleaseTrace),
                  (native, "snapshot_weakrefs", snapshot_for_startup),
                  (native, "run_lifecycle", lifecycle), (native, "build_native", build_checked),
                  (runtime, "validate_pinch_ledger", validate_ledger)):
        return native.execute(args, contract, cfg, project)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup", choices=["explicit"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--execute-native", action="store_true")
    parser.add_argument("--approved-code-config-sha256")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.steps != 240:
        parser.error("exactly 240 total SDK steps required")
    if args.execute_native:
        if args.output is None or not args.headless or args.device != "cuda:0":
            parser.error("fresh output/headless/cuda:0 required")
        with bindings((prior, "ENTRY", ENTRY), (prior, "WRAPPER", WRAPPER),
                      (prior, "VERSION", VERSION), (prior, "WORKER_ENV", WORKER_ENV)):
            prior.require_worker(args, project)
        prior.assert_clean_import_boundary()
    contract, cfg = preflight(project)
    if not args.execute_native:
        print(json.dumps(contract, indent=2))
        return
    if args.approved_code_config_sha256 != contract["code_config_sha256"]:
        parser.error("current candidate code/config approval differs")
    args.output.mkdir(parents=True, exist_ok=False)
    args.ropesim_python = prior.PAIR_ROOT
    sys.path.insert(0, str(args.ropesim_python))
    execute(args, contract, cfg, project)


if __name__ == "__main__":
    main()
