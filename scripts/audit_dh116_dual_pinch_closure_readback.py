#!/usr/bin/env python3
"""Summarize the archived dual Pinch closure readbacks; never step or qualify."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
from go2_pvcnn.control.dh116.tabletop_deformable_dual_evidence import inspect_evidence, _Sequence
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned, encode_owned
from scripts.prepare_dh116_dual_pinch_closure_interval import sdk_clock_sample

VERSION = "dh116-dual-pinch-closure-readback-cpu-v1"
EVIDENCE = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence")
RECEIPT = EVIDENCE / ("native-attempts/flex.f7-dual-pinch-guarded-closure-native/"
    "74f8378ee4648f30c8ecb2ba8f0aa5f2caf9ef396a8e49b9c532cbf44ff7fad6/"
    "attempt-1ouifpxg/receipt.json")
RECEIPT_SHA = "dd64cce38312cb3fe044e210c8560f44fae33b218e5b8e2ba97435a3969c9bc0"
PROFILE = EVIDENCE / "dual-pinch-readback-profile-20261113-dxcb62it/run02/profile.json"
PROFILE_SHA = "67fe4ac5a231355aac627d2ce4e7bd8186b138273d80df3c30a5c5406c4e0498"
AUDIT_NAME = "dual-pinch-closure-post-exit-audit.json"
SIDES = ("left", "right")
REQUIRED = ("finger14_link", "finger23_link")
OTHER_PADS = ("finger33_link", "finger43_link", "finger53_link")
SOURCE = ("base_link",) + tuple("finger%d%d_link" % (i, j)
    for i in range(1, 6) for j in range(1, 5 if i == 1 else 4))
TABLE_KEYS = {"source:" + n for n in SOURCE} | {"cooked:" + n for n in REQUIRED + OTHER_PADS}
SOURCE_FILES = (
    "scripts/audit_dh116_dual_pinch_closure_readback.py",
    "scripts/prepare_dh116_dual_pinch_closure_interval.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_dual_evidence.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_r0_replay.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_bimanual_step.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_solver_clock.py",
    "src/go2_pvcnn/control/dh116/tabletop_teacher_mounted_wrist_source.py",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(encode_owned(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_pinned(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, str(path) + " SHA differs")
    return raw


def strict_json(raw):
    def invalid(value):
        raise ValueError("nonstandard JSON " + value)
    return json.loads(raw, parse_constant=invalid)


def load_archive(*, receipt=RECEIPT, receipt_sha=RECEIPT_SHA):
    """Bind existing receipts and bytes, not rerun the native acceptance validator."""
    receipt = Path(receipt)
    source = strict_json(read_pinned(receipt, receipt_sha))
    require(source["native_execution_performed"] is True and source["native_qualification"] is True
            and type(source["exit_code"]) is int and source["exit_code"] == 0
            and type(source["outer_exit_code"]) is int and source["outer_exit_code"] == 0,
            "successful archived native execution receipt required")
    native = Path(source["output"])
    files = {}
    for name in ("physical.jsonl", "result.json", "sdk_clock.jsonl"):
        files[name] = read_pinned(native / name, source["artifact_sha256"][name])
    audit_sha = source["lifetime_post_exit_audit_sha256"]
    audit = strict_json(read_pinned(receipt.parent / AUDIT_NAME, audit_sha))["evidence"]
    result = strict_json(files["result.json"])
    acquisition = result["acquisition"]
    profile = strict_json(read_pinned(PROFILE, PROFILE_SHA))
    require(audit["lifecycle_qualified"] is True
            and audit["qualification_scope"] == "exact_2_setup_plus_16_measured_guarded_closure_execution_only",
            "archived closure post-exit scope differs")
    require(result["version"] == "dh116-dual-pinch-guarded-closure-v1"
            and result["physics_steps"] == 18 and result["measurement_steps"] == 16
            and result["ppo_updates"] == 0 and result["failure"] is None,
            "exact archived 2+8+8 result required")
    require(audit["code_config_sha256"] == result["code_config_sha256"] == acquisition["code_config_sha256"],
            "operation source identity differs")
    require(audit["profile_sha256"] == acquisition["profile_sha256"] == PROFILE_SHA
            and source["dependency_receipt_sha256"] == profile["native_parent_receipt_sha256"]
            and acquisition["setup"]["paired_evidence_sha256"] == profile["paired_reset_sha256"]
            and acquisition["setup"]["reset_seed_sha256"] == profile["reset_seed_sha256"],
            "profile parent/reset binding differs")
    for name in files:
        require(audit["available_artifact_sha256"][name] == source["artifact_sha256"][name],
                "post-exit artifact identity differs")
    inspection = inspect_evidence(native / "physical.jsonl")
    require(inspection["record_sha256"] == source["artifact_sha256"]["physical.jsonl"],
            "ledger changed during inspection")
    require(all(acquisition["journal"].get(k) == v for k, v in inspection.items())
            and acquisition["journal"] == audit["journal"], "saved journal summary differs")
    rows = [decode_owned(strict_json(line)) for line in files["physical.jsonl"].splitlines()]
    context = dict(receipt_sha256=receipt_sha, receipt_path=str(receipt), native_output=str(native),
        artifact_sha256={n: source["artifact_sha256"][n] for n in files},
        post_exit_audit_sha256=audit_sha, profile_sha256=PROFILE_SHA,
        parent_receipt_sha256=source["dependency_receipt_sha256"],
        registry_recipe_sha256=source["code_config_sha256"],
        executed_operation_sha256=result["code_config_sha256"], tested_head=source["tested_head"],
        source_native_qualification_scope=source["qualification_scope"],
        profile=profile, journal=inspection, setup=acquisition["setup"],
        schedule=acquisition["action_schedule"],
        sdk_clock=[strict_json(line) for line in files["sdk_clock.jsonl"].splitlines()])
    return rows, context


def vector(hand, name):
    a = np.asarray(hand[name])
    require(a.shape == (1, 17) and a.dtype.kind == "f" and np.isfinite(a).all(),
            "finite measured mounted17 " + name + " required")
    return a[0].astype(float)


def validate_records(rows, context):
    operations = ["setup", "setup", "reset", "initial_geometry", "initial_geometry"]
    operations += (["begin"] + ["physical"] * 8 + ["end"]) * 2
    require(len(rows) == 26 and [e.get("operation") for e in rows[1:]] == operations,
            "exact two complete 8-row windows required")
    require(rows[0]["physics_budget_end"] == 18 and rows[0]["solver_substeps"] == 12,
            "exact archived physics budget required")
    sequence = _Sequence(18, 12)
    for entry in rows[1:]:
        sequence.accept(entry["operation"], entry["payload"])
    require(all(context["journal"].get(k) == v for k, v in sequence.summary().items()),
            "recorded window summary differs")
    expected = [p["action24"] for p in context["profile"]["phases"]]
    require(context["schedule"] == expected, "profile action schedule differs")
    begins = [e["payload"] for e in rows[1:] if e["operation"] == "begin"]
    require(all(np.array_equal(b["action"], [a]) for b, a in zip(begins, expected)),
            "zero then one fixed action required")
    physical = [e["payload"] for e in rows[1:] if e["operation"] == "physical"]
    clock = [sdk_clock_sample(row, i) for i, row in enumerate(physical, 1)]
    require(clock == context["sdk_clock"], "SDK durable clock differs")
    reset = rows[3]["payload"]["reset"]["after_clock"]
    require(reset == context["setup"]["clock"], "actual reset/episode identity differs")
    sources = physical[0]["exchange"]["closure_geometry"]["sources"]
    previous = None
    for row in physical:
        ex = row["exchange"]; trace = ex["action_readback"]; geometry = ex["closure_geometry"]
        require(ex["guard_failure"] is None and ex["original_guard_failure"] is None
                and ex["geometry_capture_error"] is None and ex["table_capture_error"] is None,
                "complete nonterminal recorded observations required")
        require(geometry["passed"] is True and geometry["terminal_reason"] is None
                and geometry["failure_reasons"] == [] and geometry["table_floor_m"] == .0021,
                "recorded original geometry guard differs")
        raw = geometry["raw"]
        require(geometry["sources"] == sources
                and hashlib.sha256(canonical(raw)).hexdigest() == geometry["raw_sha256"]
                and raw["identity"] == trace["post"]["identity"][0]
                and canonical(raw["joint_readback"]) == canonical(trace["post"])
                and np.array_equal(raw["rod_nodes_m"], ex["rod"]["node_positions"][0]),
                "same-clock owned geometry/source identity differs")
        for side in SIDES:
            for phase in ("pre", "post"):
                hand = trace[phase]["hands"][side]
                names = hand["joint_names"]
                require(len(names) == len(set(names)) == 17
                        and hand["active_indices"] == [names.index(n) for n in ACTIVE]
                        and hand["mounted"] is True, "actual named active joints differ")
                for field in ("q", "qd", "backend_targets"):
                    vector(hand, field)
                if phase == "pre" and previous is not None:
                    require(canonical(hand) == canonical(previous[side]),
                            "post-to-next-pre actual joint state differs")
            post = trace["post"]["hands"][side]
            submitted = trace["submission"]["expected_backend_targets"][side]
            require(np.array_equal(post["backend_targets"][:, submitted["joint_indices"]], submitted["values"]),
                    "submitted targets differ from actual backend target readback")
            geo = geometry["sides"][side]
            table = geo["table_clearance_m"]
            require(set(table) == TABLE_KEYS and all(np.isfinite(v) for v in table.values())
                    and min(table.values()) == geo["minimum_table_clearance_m"]
                    and min(table.values()) >= .0021, "complete source17/locked5 table record differs")
            non_target = geo["non_target_surface_gap_lower_m"]
            require(min(non_target.values()) == geo["minimum_non_target_gap_lower_m"]
                    and all(np.isfinite(v) and v >= 0 for v in non_target.values()),
                    "recorded non-target bound differs")
            require(set(geo["required_pad_sdf"]) == set(REQUIRED), "required pad SDF record differs")
            for pad in REQUIRED:
                sdf = geo["required_pad_sdf"][pad]; gaps = sdf["node_gaps_m"]
                valid = [i for i, v in enumerate(gaps) if v is not None]
                require(len(gaps) == 41 and valid == sdf["in_lattice_nodes"] and valid
                        and all(np.isfinite(gaps[i]) for i in valid)
                        and min(gaps[i] for i in valid) == sdf["minimum_gap_m"]
                        and gaps[sdf["minimum_node_id"]] == sdf["minimum_gap_m"]
                        and np.isfinite(sdf["gradient_norm_at_minimum"]) and sdf["gradient_norm_at_minimum"] > 0,
                        "recorded SDF minimum/coverage differs")
        table = ex["hand_table_contacts"]
        require(canonical(table) == canonical(row["hand_table_contacts"])
                and table["available"] is False and table["hands"] is None
                and table["total_contacts"] is None and table["source"]["table_is_static"] is True,
                "static GPU hand/table contact must remain unavailable")
        previous = trace["post"]["hands"]
    return physical


def joint_window(rows, side):
    traces = [r["exchange"]["action_readback"] for r in rows]
    first = traces[0]["pre"]["hands"][side]
    last = traces[-1]["post"]["hands"][side]
    result = {}
    for name in ACTIVE:
        index = first["joint_names"].index(name)
        q = [vector(t["post"]["hands"][side], "q")[index] for t in traces]
        qd = [vector(t["post"]["hands"][side], "qd")[index] for t in traces]
        targets = [vector(t["post"]["hands"][side], "backend_targets")[index] for t in traces]
        q0 = vector(first, "q")[index]; t0 = vector(first, "backend_targets")[index]
        requested_index = SIDES.index(side) * 6 + ACTIVE.index(name)
        requested = [float(t["request"]["requested_finger_targets_rad"][0, requested_index]) for t in traces]
        error = np.asarray(targets) - q
        result[name] = dict(q_before_rad=q0, q_end_rad=q[-1], actual_q_delta_rad=q[-1]-q0,
            backend_target_before_rad=t0, backend_target_end_rad=targets[-1],
            backend_target_delta_rad=targets[-1]-t0,
            qd_before_rad_s=vector(first, "qd")[index], qd_end_rad_s=qd[-1],
            max_abs_qd_rad_s=max(abs(v) for v in qd),
            target_error_before_rad=t0-q0, target_error_end_rad=float(error[-1]),
            max_abs_target_error_rad=float(abs(error).max()), requested_targets_rad=requested,
            actual_q_rad=q, actual_qd_rad_s=qd, backend_targets_rad=targets)
    return dict(active_joint_order=list(ACTIVE), active_joints=result)


def series_summary(values):
    require(len(values) == 16, "exact recorded 16-sample series required")
    return dict(first_post_sdk3=values[0], zero_end_sdk10=values[7],
        action_first_post_sdk11=values[8], action_end_sdk18=values[-1],
        action_interval_change=values[-1]-values[7], minimum=min(values),
        minimum_sdk_step=3+values.index(min(values)), maximum=max(values), by_sdk_step=values)


def summarize(rows, context):
    """Aggregate saved measured responses. No FK, SDF sampling, SDK or R0 call."""
    physical = validate_records(rows, context)
    windows = []
    for i in range(2):
        block = physical[i*8:(i+1)*8]
        windows.append(dict(policy_step=i, sdk_steps=[r["exchange"]["physics_step"] for r in block],
            action24=context["schedule"][i],
            hands={side: joint_window(block, side) for side in SIDES}))
    geometry = {}
    for side in SIDES:
        geos = [r["exchange"]["closure_geometry"]["sides"][side] for r in physical]
        table = {key: series_summary([g["table_clearance_m"][key] for g in geos])
                 for key in sorted(TABLE_KEYS)}
        pads = {}
        for pad in REQUIRED:
            saved = [g["required_pad_sdf"][pad] for g in geos]
            pads[pad] = dict(gap_m=series_summary([s["minimum_gap_m"] for s in saved]),
                minimum_node_ids=[s["minimum_node_id"] for s in saved],
                in_lattice_nodes=[s["in_lattice_nodes"] for s in saved],
                gradient_norm_at_minimum=[s["gradient_norm_at_minimum"] for s in saved],
                contact_predicate_nodes=[s["contact_predicate_nodes"] for s in saved],
                source="recorded_geometry_from_measured_pose", native_contact_force=None)
        geometry[side] = dict(table_clearance_m=table, required_pad_sdf=pads,
            other_pad_sdf={p: None for p in OTHER_PADS},
            other_pad_sdf_reason="not recorded; no new SDF sampling in this readback",
            minimum_table_clearance_m=min(t["minimum"] for t in table.values()),
            minimum_table_margin_above_original_floor_m=min(t["minimum"] for t in table.values())-.0021,
            minimum_non_target_gap_lower_m=min(g["minimum_non_target_gap_lower_m"] for g in geos),
            maximum_wrist_tracking_error_m=max(g["wrist_tracking_error_m"] for g in geos),
            maximum_wrist_world_com_linear_speed_m_s=max(g["wrist_world_com_linear_speed_m_s"] for g in geos))
    clocks = [r["exchange"]["action_readback"]["post"]["identity"][0] for r in physical]
    pairs = Counter(p["mesh_name"] for r in physical for p in r["exchange"]["rod"]["rod_mesh_contact_pairs"]["pairs"])
    lineage = {k: v for k, v in context.items() if k not in ("profile", "journal", "setup", "schedule", "sdk_clock")}
    return dict(version=VERSION, source=lineage, journal=context["journal"],
        source_native_execution_qualified=True, qualification_scope="archived_response_summary_only",
        command_semantics="backend targets are SDK commands; measured q/qd are distinct actual responses",
        geometry_semantics="recorded_geometry_from_measured_pose; SDF is not contact force",
        geometry_recomputed=False, sdf_recomputed=False, physical_samples=16,
        clock=dict(identity={k: clocks[0][k] for k in ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid")},
            reset_sdk_step=2, reset_rod_tick=0, sdk_steps=[c["physics_step"] for c in clocks],
            rod_ticks=[c["rod_tick"] for c in clocks], timestamps_s=[c["timestamp_s"] for c in clocks],
            solver_times_s=[c["solver_time_s"] for c in clocks],
            sdk_increment=1, rod_increment=12, complete_windows=2),
        windows=windows, geometry=geometry, original_table_floor_m=.0021,
        geometry_sources=physical[0]["exchange"]["closure_geometry"]["sources"],
        raw_pairs_by_mesh=dict(pairs), raw_hand_pair_samples=sum(v for k, v in pairs.items() if k != "table"),
        hand_table_contacts=None, hand_table_source_available=False, no_hand_table_contact_proven=False,
        hand_table_unavailable_reasons=sorted({r["hand_table_contacts"]["reason"] for r in physical}),
        new_physics_steps=0, optimizer_updates=0, new_geometry_search_states=0,
        native_qualification=False, grasp_qualified=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument("--receipt-sha256", default=RECEIPT_SHA)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; preserve original evidence")
    rows, context = load_archive(receipt=args.receipt, receipt_sha=args.receipt_sha256)
    summary = summarize(rows, context)
    native = sorted(n for n in sys.modules if n.split(".")[0] in ("omni", "isaaclab", "isaacsim", "ropesim", "warp", "pxr"))
    require(not native, "native import crossed the read-only CPU boundary")
    result = dict(summary=summary, native_imports=native,
        implementation_sha256={name: digest(ROOT/name) for name in SOURCE_FILES})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), physical_samples=16, new_physics_steps=0,
                          optimizer_updates=0, native_imports=native)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
