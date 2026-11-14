"""Read-only negative effective-contact evidence from frozen native pairs.

Necessary-condition rejection can prove absence, never positive grasp. The
legacy serializer omits raw capacity/overflow counters: this reader therefore
requires an immutable completed native receipt, not arbitrary pair arrays.
No live sensor, R0 training qualification, or reward coefficient is added.
"""
from copy import deepcopy
import hashlib
import json
import subprocess
from pathlib import Path
from types import MappingProxyType

from .tabletop_deformable_r0_replay import (
    PhysicalR0Diagnostic, UNKNOWN_SLIP, _digest, _measurement, decode_owned,
)
from .tabletop_deformable_contact_diagnostics import summarize_contact_pairs
from .tabletop_deformable_pad_region import bind_geometry, classify_pair
from .tabletop_pair_substep_load_audit import bind_clock, audit_capture, SIDES, PADS
from .tabletop_effective_pad import _patch

VERSION = "dh116-frozen-effective-pad-negative-source-v1"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def negative_side_result(classifications, *, complete, rod_uid):
    if complete is not True:
        raise ValueError("incomplete capture cannot establish absence")
    if any(type(c) is not bool for c in classifications):
        raise ValueError("explicit original-patch classifications required")
    if not isinstance(rod_uid, str) or not rod_uid:
        raise ValueError("explicit rod identity required")
    known = not any(classifications)
    return dict(
        value=(dict(rod_uid=rod_uid, pads=[], opposed=False,
                    **dict.fromkeys(UNKNOWN_SLIP)) if known else None),
        known_no_effective_contact=known, raw_pair_count=len(classifications),
        region_candidate_count=sum(classifications),
        reason=None if known else "region_candidate_requires_depth_and_effective_contact_source",
        positive_effective_contact_qualified=False,
    )

class FrozenEffectivePadSource:
    """An explicit offline source; complete captured rows cannot be substituted."""
    def __init__(self, project, receipt_path, *, expected_receipt_sha256, expected_scope):
        from scripts.audit_dh116_pinch_initial_pose import load_geometry
        from scripts.audit_dh116_bilateral_effective_pad_regions import validate_binding
        from scripts.replay_dh116_segment_contact_r0 import stream_events

        self.project = Path(project).resolve()
        path = Path(receipt_path)
        if path.is_symlink() or sha(path) != expected_receipt_sha256:
            raise ValueError("immutable native receipt required")
        r = json.loads(path.read_bytes())
        if (r.get("native_execution_performed") is not True
            or r.get("native_qualification") is not True
            or r.get("qualification_scope") != expected_scope
            or r.get("outer_exit_code") != 0
            or r.get("debugger_inferior_signal") is not None
            or not 0 < r.get("duration_seconds", float("inf")) <= 180):
            raise ValueError("successful bounded execution receipt with exact scope required")
        paths = ("src/go2_pvcnn/control/dh116/tabletop_dual_hand_rod_runtime.py",
                 "src/go2_pvcnn/control/dh116/tabletop_deformx_pair_capture.py")
        self.capture_source_sha256 = {}
        for name in paths:
            historical = subprocess.run(
                ["git", "-c", "safe.directory=" + str(self.project), "-C", str(self.project),
                 "show", r["tested_head"] + ":" + name],
                check=True, capture_output=True).stdout
            expected = hashlib.sha256(historical).hexdigest()
            if sha(self.project / name) != expected:
                raise ValueError("audited native capture implementation changed")
            self.capture_source_sha256[name] = expected
        self.receipt = r
        self.receipt_sha256 = expected_receipt_sha256
        native = Path(r["output"])
        # Validate the exact complete native artifact set, including reset.
        for name, expected in r["artifact_sha256"].items():
            p = native / name
            if p.is_symlink() or not p.is_relative_to(native) or sha(p) != expected:
                raise ValueError("frozen native artifact changed: " + name)
        self.exchanges = []
        for event in stream_events(native / "physical.jsonl", r["artifact_sha256"]["physical.jsonl"]):
            if event.get("operation") == "physical":
                self.exchanges.append(event["payload"]["exchange"])
        if len(self.exchanges) != 232:
            raise ValueError("complete original 232 measured steps required")
        hashes = {}
        episode = None
        for step, x in enumerate(self.exchanges, 3):
            clock = x["action_readback"]["post"]["identity"][0]
            key = tuple(clock[k] for k in ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid"))
            if (episode is not None and key != episode or clock["physics_step"] != step):
                raise ValueError("complete same-episode source required")
            episode = key
            hashes[step] = _digest(x)
        self._hashes = MappingProxyType(hashes)
        self._atlas = {}
        self._bindings = {}
        self._geometry = {}
        for side in SIDES:
            g = load_geometry(self.project, side=side)
            self._geometry[side] = g
            for pad in PADS:
                key = side + "." + pad
                self._atlas[key] = _patch(g["source_colliders"][pad]["vertices"], side, pad)
                binding = self.exchanges[0]["rod"]["mesh_bindings"][key]
                validate_binding(binding, g, pad)
                self._bindings[key] = _digest(binding)
        self._validate_binding = validate_binding

    def capture(self, exchange):
        clock = exchange["action_readback"]["post"]["identity"][0]
        step = clock["physics_step"]
        if self._hashes.get(step) != _digest(exchange):
            raise ValueError("exact frozen exchange required; no partial or changed evidence")
        rod = exchange["rod"]
        if (rod.get("contact_valid") is not True
            or rod.get("pair_capture_status") != "validated_solver_samples"):
            raise ValueError("producer-validated complete native capture required")
        capture = rod["rod_mesh_contact_pairs"]
        bind_clock(capture, rod, clock, expected_physics_step=step)
        # Reconcile original retained aggregates as well as every individual pair.
        summarize_contact_pairs(rod, grasp="pinch")
        audit_capture(capture, solver_substeps=12, complete_capture=True)
        raw = exchange["closure_geometry"]["raw"]
        transforms = {}
        for side in SIDES:
            for pad in PADS:
                key = side + "." + pad
                binding = rod["mesh_bindings"][key]
                if self._bindings[key] != _digest(binding):
                    raise ValueError("same immutable source/cooked/SDF binding required")
                self._validate_binding(binding, self._geometry[side], pad)
                transforms[key] = bind_geometry(raw, clock, key, binding)
        classified = {s: [] for s in SIDES}
        records = []
        for pair in capture["pairs"]:
            key = pair["mesh_name"]
            if key == "table":
                continue
            result = classify_pair(pair, transforms[key], self._atlas[key])
            side = key.split(".")[0]
            classified[side].append(result["accepted_by_original_patch"])
            records.append(dict(mesh_name=key, solver_substep=pair["substep"],
                                node_index=pair["node_index"], region=result))
        return dict(
            version=VERSION, identity=deepcopy(clock),
            hands={s: negative_side_result(classified[s], complete=True, rod_uid=clock["rod_uid"])
                   for s in SIDES},
            pairs=records, source_receipt_sha256=self.receipt_sha256,
            source_ledger_sha256=self.receipt["artifact_sha256"]["physical.jsonl"],
            source_exchange_sha256=self._hashes[step],
            completeness_source="frozen_native_producer_validated_capture_and_reconciled_aggregates",
            raw_capacity_overflow_retained=False,
            capture_producer_sha256=dict(self.capture_source_sha256),
            producer_tested_head=self.receipt["tested_head"],
            native_pair_capture_status=rod["pair_capture_status"],
            timing_scope="post_SDK_link_pose_held_during_12_solver_substeps",
            contact_depth_qualified=False, material_slip_qualified=False,
            new_physics_steps=0, ppo_updates=0, teacher_admission=False,
        )

class EffectivePadR0Diagnostic(PhysicalR0Diagnostic):
    def __init__(self, *args, effective_pad_source, **kwargs):
        super().__init__(*args, **kwargs)
        self.effective_pad_source = effective_pad_source

    def _measurements(self, event, clock, auxiliary):
        measurements, diagnostics, unavailable = super()._measurements(event, clock, auxiliary)
        evidence = self.effective_pad_source.capture(event["exchange"])
        if evidence["identity"] != clock:
            raise ValueError("same R0 physical endpoint required")
        for side in SIDES:
            hand = evidence["hands"][side]
            source = None if hand["value"] is None else (
                VERSION + ":" + evidence["source_exchange_sha256"] + ":" + side)
            measurements[side] = _measurement(hand["value"], source)
            unavailable.pop(side, None)
            if hand["reason"]:
                unavailable[side] = hand["reason"]
        diagnostics["effective_pad_negative_source"] = evidence
        return measurements, diagnostics, unavailable
