#!/usr/bin/env python3
"""CPU-only same-episode comparison: raw pad counts versus measured negatives."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from scripts import audit_dh116_effective_right_tracking as pinned
from scripts import replay_dh116_native_parameter_r0 as old
from scripts.replay_dh116_segment_contact_r0 import stream_events, prefix_scope
from scripts.replay_dh116_bilateral_contact_r0 import summary_gate
from go2_pvcnn.control.dh116.tabletop_r0_effective_pad_source import (
    FrozenEffectivePadSource, EffectivePadR0Diagnostic,
)
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import (
    PhysicalR0Diagnostic, encode_owned, decode_owned, _digest,
)
from go2_pvcnn.control.dh116.tabletop_r0_measured_approach import MeasuredPinchApproach
from go2_pvcnn.control.dh116.tabletop_effective_right_pinch_geometry import EffectiveRightGeometry, VERSION as GEOMETRY
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task, R0_TERMS

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dh116-effective-pad-negative-r0-replay-v1"
CONFIG = "configs/dh116_training/effective_pad_negative_r0_replay_v1.json"

class EffectiveRightMeasuredApproach(MeasuredPinchApproach):
    def __init__(self, project):
        self.project = Path(project).resolve()
        self.geometry = EffectiveRightGeometry(self.project)
        self.source_sha256 = pinned.digest(__file__)

    def _metric(self, packet, radius):
        result = super()._metric(packet, radius)
        result["version"] = VERSION
        result["source"].update(version=VERSION, geometry_contract_version=GEOMETRY,
            inherited_metric_source_sha256=pinned.digest(
                ROOT / "src/go2_pvcnn/control/dh116/tabletop_r0_measured_approach.py"))
        result["source_sha256"] = _digest(result["source"])
        sample = result["auxiliary"]["samples"]["approach_distance_m"]
        sample["source"] = None if sample["value"] is None else (
            "actual_SDK_solver_nodes_effective_right_SDF_discrete_approach:" + result["source_sha256"])
        return result

def configuration():
    cfg = json.loads((ROOT / CONFIG).read_bytes())
    if cfg != dict(version=VERSION, enabled_by_default=False,
        source_receipt_sha256=pinned.RECEIPT_SHA, expected_measured_steps=232,
        new_physics_steps=0, ppo_updates=0, training_reward_enabled=False,
        coefficient_source="existing_CPU_fixture_ordinal_weights_not_training_coefficients",
        teacher_admission=False):
        raise ValueError("immutable offline-only replay configuration required")
    return cfg

def run(result, emit):
    cfg = configuration()
    source = FrozenEffectivePadSource(ROOT, pinned.RECEIPT,
        expected_receipt_sha256=pinned.RECEIPT_SHA, expected_scope=pinned.SCOPE)
    receipt = source.receipt
    native = Path(receipt["output"])
    result.update(source_receipt_sha256=pinned.RECEIPT_SHA,
        source_ledger_sha256=receipt["artifact_sha256"]["physical.jsonl"],
        source_scope=pinned.SCOPE, config_sha256=pinned.digest(ROOT / CONFIG),
        coefficient_source=cfg["coefficient_source"])
    reader = EffectiveRightMeasuredApproach(ROOT)
    p0 = decode_owned(json.loads((native / "director_reset.json").read_bytes())["payload"])
    task = load_flexible_grasp_task(ROOT / "configs/dh116_tasks/flexible_dual_hand_grasp_v1.json", ROOT)
    budget = json.loads((ROOT / "configs/dh116_training/deformable_task_budget_examples_v2.json").read_bytes())["plans"]["frozen_eval"]
    budget["setup_physics_steps_per_env"] = 2
    budget["physics"]["solver_substeps_per_physics_step"] = 12
    weights = {name: float(i + 1) for i, name in enumerate(R0_TERMS)}
    observers, controls = {}, {"legacy": [], "negative_source": []}
    blocked = {mode: Counter() for mode in controls}
    unknown = Counter()
    raw_pairs = Counter()
    counts = Counter()
    fields = None
    for entry in stream_events(native / "physical.jsonl", result["source_ledger_sha256"]):
        op = entry.get("operation")
        counts[str(op)] += 1
        if op in (None, "setup", "initial_geometry"): continue
        p = entry["payload"]
        if op == "reset":
            if observers: raise ValueError("exactly one native reset")
            radius = old.check_radius_packet(p0, p["reset"]["rod_reset"])
            reset_approach = reader.capture_reset(p["reset"], p0)
            result["reset_approach"] = reset_approach
            for mode, cls in (("legacy", PhysicalR0Diagnostic), ("negative_source", EffectivePadR0Diagnostic)):
                extra = dict(effective_pad_source=source) if mode == "negative_source" else {}
                obs = cls(task, "pinch", env_id=0, budget_config=budget,
                    diagnostic_coefficients=weights, backend_artifact=old.BACKEND,
                    backend_sha256=old.BACKEND_SHA,
                    radius_sample=dict(value=radius["value"], source=
                        "native_host_metadata:RopeBatch.parameter_snapshot.contact_radius_m:" + old.BACKEND_SHA),
                    **extra)
                adapted = deepcopy(task)
                adapted["backend"]["extension_sha256"] = old.BACKEND_SHA
                obs._reducer = old._ResetBoundApplicableR0(adapted, "pinch", env_id=0,
                    budget_config=budget, diagnostic_coefficients=weights,
                    reset_approach=reset_approach["reset_approach"])
                obs.record(p)
                observers[mode] = obs
        elif not observers: raise ValueError("reset required before data")
        elif op == "begin":
            current = [i for i, valid in enumerate(p["pre_action_state"]["common"]["valid"]) if valid]
            if fields is not None and fields != current: raise ValueError("common validity changed")
            fields = current
            for obs in observers.values(): obs.begin(**p)
        elif op == "physical":
            measured = reader.capture(p["exchange"])
            comparison = {}
            for mode, obs in observers.items():
                obs.record(p, auxiliary=measured["auxiliary"])
                row = obs._rows[-1]
                blocked[mode].update(row["r0"]["blocked_terms"])
                unknown[mode] += any(s in row["unavailable_sources"] for s in ("left", "right"))
                comparison[mode] = dict(terms=row["r0"]["terms"],
                    unavailable_sources=row["unavailable_sources"],
                    blocked_terms=row["r0"]["blocked_terms"],
                    contact_measurements={s: row["measurements"][s] for s in ("left", "right")},
                    training_reward=row["r0"]["training_reward"])
            evidence = observers["negative_source"]._rows[-1]["pair_diagnostics"]["effective_pad_negative_source"]
            for side, h in evidence["hands"].items(): raw_pairs[side] += h["raw_pair_count"]
            emit(dict(operation="same_step_comparison", identity=evidence["identity"],
                negative_proof=evidence, comparison=comparison))
        elif op == "end":
            for mode, obs in observers.items():
                control = obs.end(p)["control_receipt"]
                if control is None or control["physical_samples"] != 8:
                    raise ValueError("complete actual eight-step interval required")
                controls[mode].append(control)
                emit(dict(operation="control_receipt", mode=mode, payload=control))
        else: raise ValueError("unsupported recorded operation")
    if counts != Counter({"None":1, "setup":2, "initial_geometry":2, "reset":1,
                          "begin":29, "physical":232, "end":29}):
        raise ValueError("complete original native event stream required")
    result.update(prefix_scope(29, 232, 2), status="complete",
        controls=controls, comparison={mode: summary_gate(c) for mode, c in controls.items()},
        blocked_physical_term_counts={m: dict(c) for m, c in blocked.items()},
        unknown_contact_rows=dict(unknown), raw_hand_pairs=dict(raw_pairs),
        common_valid_indices=fields, common_valid_count=len(fields), common_total=371,
        common_missing_policy="NaN_with_false_validity",
        source_operation_counts=dict(counts), original_R0_reset_native_order_qualified=False,
        positive_effective_contact_qualified=False, contact_depth_qualified=False,
        material_anchor_slip_qualified=False, full_reward_ready=False,
        original_reward_coefficients_unchanged=True,
        missing_sources=["qualified_stability_predicate", "native_R0_reset_owner_order",
            "training_reward_coefficients", "positive_contact_and_slip_source_when_applicable",
            "hand_table_contact_source"])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = dict(version=VERSION, status="partial", new_physics_steps=0,
        ppo_updates=0, teacher_admission=False, native_qualification=False, training_reward=None)
    previous = signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError("600s CPU budget")))
    signal.setitimer(signal.ITIMER_REAL, 600)
    try:
        with (args.output / "progress.jsonl").open("x") as f:
            def emit(row):
                f.write(json.dumps(encode_owned(row), allow_nan=False) + "\n")
                f.flush()
            try: run(result, emit)
            finally: f.flush(); os.fsync(f.fileno())
    except Exception as e:
        result.update(status="blocked", reason=type(e).__name__ + ": " + str(e))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    result.update(elapsed_s=time.monotonic()-start, implementation_sha256=pinned.digest(__file__),
                  output_sha256=pinned.digest(args.output / "progress.jsonl"))
    with (args.output / "result.json").open("x") as f:
        json.dump(encode_owned(result), f, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    print(json.dumps({k: result.get(k) for k in ("status", "reason", "elapsed_s",
        "unknown_contact_rows", "raw_hand_pairs", "blocked_physical_term_counts", "teacher_admission")}))
    return int(result["status"] != "complete")
if __name__ == "__main__": raise SystemExit(main())
