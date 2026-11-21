#!/usr/bin/env python3
"""One bounded observed Pinch quiet-and-close diagnostic on the sealed staged Reset."""
import argparse
import functools
import json
import math
from pathlib import Path

import numpy as np

from scripts import record_dh116_teacher_numeric_reset as lifecycle
from scripts.audit_dh116_pinch_staged_closure import compile_close
from scripts.audit_dh116_rod_quiet_window import extract, summarize_window
from scripts.prepare_dh116_tripod_owned_approach import QUIET_LIMITS
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import DT
from go2_pvcnn.control.dh116.tabletop_deformable_clearance_guard import (
    ClearanceGuardedExchange, frozen_clearance_sources, validate_post_step_clearance,
)
from go2_pvcnn.control.dh116.tabletop_deformable_dual_evidence import DualHandEvidenceJournal, inspect_evidence
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_pair_source_basis import compare_contacts
from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import MESH_NAMES, SIDES
from go2_pvcnn.control.dh116.tabletop_deformx_pair_capture import validate_pair_capture
from go2_pvcnn.control.dh116.tabletop_teacher_force_closure import _opposition, _pad_summary

ROOT = lifecycle.ROOT
VERSION = "dh116-staged-pinch-contact-diagnostic-v1"
SUBSTEPS = 12
QUIET_STEPS = 8
CLOSE_STEPS = 64
MEASURED_STEPS = QUIET_STEPS + CLOSE_STEPS
SDK_STEPS = 2 + MEASURED_STEPS
CANDIDATE = ROOT / "configs/dh116_execution/pinch-staged-open-and-close-archived-v1.json"
CANDIDATE_SHA = "afc7e79c75a5b0436d4b9224c3b9d556cdcb4b9a933cbd1dea615153f4fd186e"
SEALED_SHA = "2f8ec96ba01922b81502b4575a7b9657bbae0bc8c928c14698b3811936e2e9a8"


def schedule(plan):
    if any(plan[side]["steps"] != CLOSE_STEPS for side in SIDES):
        raise ValueError("exact 64 SDK steps from the staged candidate required")
    return np.vstack((np.zeros((1, 24)), np.tile(compile_close(plan), (CLOSE_STEPS // 8, 1))))


def same_substep_reaction(rod):
    packet = rod["rod_mesh_contact_pairs"]
    if (rod["clock_valid"] is not True or rod["pair_capture_status"] != "validated_solver_samples"
            or tuple(rod["mesh_names"]) != MESH_NAMES or not isinstance(packet, dict)
            or packet["scope"] != "rod_node_to_sdf_mesh_substep_force"
            or (packet["physics_step"], packet["rod_tick"]) != (rod["physics_step"], rod["rod_tick"])
            or not math.isclose(packet["substep_dt_s"], DT / SUBSTEPS, abs_tol=1e-12, rel_tol=0.)):
        raise ValueError("actual same-boundary 11-slot solver pair capture required")
    ids, vectors = [], []
    for pair in packet["pairs"]:
        mesh = pair["mesh_index"]
        if (type(mesh) is not int or not 0 <= mesh < len(MESH_NAMES) or pair["mesh_name"] != MESH_NAMES[mesh]
                or type(pair["substep"]) is not int or type(pair["node_index"]) is not int):
            raise ValueError("verified original mesh/solver substep/node identity required")
        ids.append((pair["substep"], pair["node_index"], mesh))
        vectors.append([*pair["contact_point_world_m"], *pair["outward_normal_world"],
                        *pair["force_on_rod_world_n"], *pair["torque_on_mesh_about_origin_world_nm"]])
    positions = np.asarray(rod["distributed_contact"])
    if positions.shape != (1, 41, 3):
        raise ValueError("original one-rod 41-node force array required")
    clean = validate_pair_capture(dict(substeps=SUBSTEPS, mesh_count=len(MESH_NAMES), capacity=max(1, len(ids)),
        attempted=len(ids), overflow=False, indices=np.asarray(ids, dtype=np.int64).reshape(-1, 3),
        vectors=np.asarray(vectors, dtype=float).reshape(-1, 12)),
        dt_s=DT, node_forces=positions[0], mesh_wrenches=[rod["named_wrenches"][name] for name in MESH_NAMES])
    by_group = {(step, mesh): [] for step in range(SUBSTEPS) for mesh in range(len(MESH_NAMES))}
    for pair in clean["pairs"]:
        by_group[pair["substep"], pair["mesh_index"]].append(pair)
    rows, common = [], []
    for step in range(SUBSTEPS):
        sides = {}
        for side, mesh in zip(SIDES, (0, 6)):
            thumb, index = by_group[step, mesh], by_group[step, mesh+1]
            pads = dict(thumb=_pad_summary(thumb, 1, mesh), index=_pad_summary(index, 1, mesh+1))
            loads = [p["normal_load_n"] for p in pads.values()]
            oppose = _opposition(pads, "Pinch")
            signed = any(compare_contacts(a, b)["geometric_opposition_sign_candidate"] is True
                         for a in thumb for b in index)
            loaded = min(loads) > .05 and sum(loads) > .3 and oppose["effective"] and signed
            sides[side] = dict(thumb_load_n=loads[0], index_load_n=loads[1],
                               opposition_cos=oppose["opposition_cos"], signed_opposition=signed,
                               loaded_opposed=bool(loaded))
        if all(sides[side]["loaded_opposed"] for side in SIDES):
            common.append(step)
        rows.append(dict(substep=step, **sides))
    return dict(physics_step=rod["physics_step"], rod_tick=rod["rod_tick"],
                pairs=len(clean["pairs"]), per_substep=rows,
                bilateral_loaded_opposed_substeps=common,
                all_12_loaded_opposed=len(common) == SUBSTEPS,
                evidence_scope="measured_simulated_reaction", grasp_qualified=False)


def quiet_gate(rows, reset_nodes):
    metrics = summarize_window(rows, QUIET_STEPS)
    metrics["source_scope"] = "same_instance_Pinch_quiet_prefix"
    drift = max(float(np.linalg.norm(row["positions"] - reset_nodes, axis=-1).max()) for row in rows)
    bad = []
    if metrics["max_node_speed_m_s"] > QUIET_LIMITS["max_node_speed_m_s"]:
        bad.append("rod_speed")
    if drift > QUIET_LIMITS["max_node_drift_m"]:
        bad.append("rod_drift")
    if metrics["table_loaded_solver_substeps"] != SUBSTEPS * QUIET_STEPS:
        bad.append("table_support")
    if metrics["hand_loaded_solver_substeps"]:
        bad.append("hand_contact")
    for side, joints in metrics["hand_joint_speed"].items():
        for name, value in joints.items():
            limit = QUIET_LIMITS["max_joint_angular_speed_rad_s"] if value["unit"] == "rad/s" else QUIET_LIMITS["max_mount_linear_speed_m_s"]
            if value["max_abs"] > limit:
                bad.append(side + ":" + name + ":speed")
    return dict(passed=not bad, reasons=bad, drift_m=drift, **metrics)


class ObservedClose(ClearanceGuardedExchange):
    def _step(self, action, substep):
        from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import profile_reader_scope
        row = None
        try:
            row = super()._step(action, substep)
            row["original_guard_failure"] = row["guard_failure"]
            try:
                with profile_reader_scope(self.profile):
                    row["material_director"] = self.parameter_reader.capture(
                        row["rod"], rope=self._rod.rope, backend=self.backend, sim=self._sim, read_clock=self._rod.clock)
                    row["closure_geometry"] = self.pinch_geometry.capture(
                        row, sim=self._sim, bindings=self._bindings, read_clock=self._rod.clock)
                info = row["closure_geometry"]
                row["stage_reaction"] = same_substep_reaction(row["rod"])
                if row["guard_failure"] is None and not info["passed"]:
                    row["guard_failure"] = info["terminal_reason"] or "staged_geometry_source_invalid"
                if row["policy_step"] == 0 and row["guard_failure"] is None:
                    self.quiet_rows.append(extract(dict(exchange=row)))
                    if substep == QUIET_STEPS - 1:
                        row["quiet_admission"] = quiet_gate(self.quiet_rows, self.reset_nodes)
                        if not row["quiet_admission"]["passed"]:
                            row["guard_failure"] = "staged_initial_quiet_guard"
            except Exception as error:
                row["stage_capture_error"] = type(error).__name__ + ": " + str(error)
                row["guard_failure"] = row["guard_failure"] or "staged_required_source_invalid"
            return row
        finally:
            self = action = row = None


def inspect_interval(path, actions, *, sdk_steps=SDK_STEPS, feedback_factory=None, sealed_archive=None):
    archived_sources = frozen_clearance_sources(sealed_archive["frozen_files"]) if sealed_archive is not None else None
    audit = inspect_evidence(path)
    if audit["physics_budget_end"] != sdk_steps or audit["solver_substeps"] != SUBSTEPS:
        raise ValueError("ledger differs from frozen close budget")
    quiet, begins, rows, expected_actions, decisions = [], [], [], [], []
    feedback = feedback_factory() if feedback_factory is not None else None
    reset_nodes = None
    with Path(path).open() as f:
        next(f)
        for line in f:
            entry = json.loads(line)
            data = decode_owned(entry["payload"])
            if entry["operation"] == "reset":
                reset_nodes = np.asarray(data["reset"]["rod_reset"]["node_positions"], dtype=float)[0]
            if entry["operation"] == "begin":
                index = len(begins)
                if index >= len(actions):
                    raise ValueError("more control requests than the frozen budget")
                expected = actions[index]
                if feedback is not None:
                    expected, selection = feedback(index, expected, rows[-8:])
                    if selection is not None:
                        decisions.append(selection)
                expected_actions.append(expected)
                begins.append(data["action"])
            if entry["operation"] == "physical":
                row = data["exchange"]
                validate_post_step_clearance(dict(row, guard_failure=row["original_guard_failure"]),
                                             sealed_archive_sources=archived_sources)
                if "stage_reaction" in row and row["stage_reaction"] != same_substep_reaction(row["rod"]):
                    raise ValueError("durable reaction differs from independently reread solver pairs")
                if "quiet_admission" in row:
                    quiet.append(row["quiet_admission"])
                rows.append(row)
    if quiet:
        recalculated = quiet_gate([extract(dict(exchange=dict(row, guard_failure=row["original_guard_failure"])))
                                   for row in rows[:QUIET_STEPS]], reset_nodes)
        if quiet != [recalculated]:
            raise ValueError("quiet receipt differs from current-instance raw rows")
    if (len(begins) > len(actions)
            or any(not np.array_equal(observed, expected[None,:]) for observed, expected in zip(begins, expected_actions))
            or (len(begins) > 1 and (len(quiet) != 1 or not quiet[0]["passed"]))):
        raise ValueError("close was not gated by its own quiet window and frozen actions")
    reactions = [(x["physics_step"], x["stage_reaction"]) for x in rows if "stage_reaction" in x]
    common = [step for step, data in reactions if data["bilateral_loaded_opposed_substeps"]]
    full = [step for step, data in reactions if data["all_12_loaded_opposed"]]
    if feedback is not None:
        audit["fixed_wrist_force_decisions"] = decisions
    return dict(audit, quiet_admission=quiet[0] if quiet else None,
                rows_with_simultaneous_loaded_opposition=common, rows_all_12_opposed=full,
                final_8_sdk_all_substeps_opposed=(len(rows) == sdk_steps - 2
                    and full[-8:] == list(range(sdk_steps-7, sdk_steps+1))),
                teacher_admission=False, grasp_qualified=False)


def acquire(sim, owners, trace, output, *, plan, sdk_steps=SDK_STEPS, feedback_factory=None):
    import sys
    from uuid import uuid4
    import torch
    from scripts.verify_dh116_dual_pinch_interval import diagnostic_reset_owners
    from go2_pvcnn.control.dh116 import tabletop_deformable_dual_hand_episode as physical
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116.tabletop_native_parameter_endpoints import NativeParameterEndpoints
    from go2_pvcnn.control.dh116.tabletop_native_teacher_policy_pinch_distance import PolicyOwnedPinchGeometry
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile, profile_reader_scope

    fresh = episode = exchange = journal = result = None
    feedback = feedback_factory() if feedback_factory is not None else None
    last_rows, live_decisions = [], []

    def record(row):
        journal.record(row)
        if row["kind"] == "physical":
            last_rows.append(row["exchange"])
            del last_rows[:-8]

    try:
        profile = BackendProfile.load(ROOT)
        source = plan["pair_proposal"]
        seed = builder.load_bound_pair_proposal(source)["reset_seed"]
        prepared = scene.prepare_scene(ROOT)
        run = "staged-pinch-" + uuid4().hex
        journal_path = Path(output) / "physical.jsonl"
        journal = DualHandEvidenceJournal(journal_path, physics_budget_end=sdk_steps, solver_substeps=SUBSTEPS)
        with profile_reader_scope(profile), bindings(
                (builder, "BACKEND", Path(profile.binary_path)), (builder, "BACKEND_SHA", profile.binary_sha256),
                (physical, "FlexiblePhysicalExchange", ObservedClose)):
            fresh = builder.build_fresh_owned(sim, owners, project=ROOT, prepared=prepared,
                ownership=trace, record=journal.record if feedback is None else record,
                run_uuid=run, episode_uid=run+":0",
                physics_budget_end=sdk_steps, expected_reset_seed=seed, pair_proposal=source)
            episode = fresh.episode
            reset = episode.reset(seed, episode_uid=run+":0", owner_resets=diagnostic_reset_owners())
            for item in fresh.initial_geometry():
                journal.record(item)
            exchange = episode._exchange
            exchange.profile = profile
            exchange.backend = sys.modules["ropesim._ropesim"]
            exchange.pinch_geometry = PolicyOwnedPinchGeometry(ROOT, scope="owner_source_unqualified")
            exchange.quiet_rows = []
            rod_reset = reset["reset"]["rod_reset"]
            exchange.reset_nodes = np.asarray(rod_reset["node_positions"], dtype=float)[0].copy()
            exchange.parameter_reader = NativeParameterEndpoints(exchange.reset_nodes,
                device=str(exchange._bindings[0].hand.data.joint_pos.device))
            exchange.parameter_reader.capture_reset(rod_reset, material_reference_uid=run+":fixed_material41",
                rope=exchange._rod.rope, backend=exchange.backend, sim=sim, read_clock=exchange._rod.clock)
            for index, action in enumerate(plan["action_schedule"]):
                if feedback is not None:
                    action, selection = feedback(index, action, last_rows)
                    if selection is not None:
                        lifecycle.native.write_new(Path(output)/f"fixed_wrist_force_{index}.json", selection)
                        live_decisions.append(selection)
                command = torch.tensor(action, dtype=torch.float64)[None,:]
                journal.begin(episode.pre_action_state(), command)
                result = episode.control(command)
                journal.end(result)
                if result["termination_reason"]:
                    break
        journal.close()
        audit = inspect_interval(journal_path, np.asarray(plan["action_schedule"]), sdk_steps=sdk_steps,
                                 feedback_factory=feedback_factory)
        if feedback is not None and audit["fixed_wrist_force_decisions"] != live_decisions:
            raise ValueError("independent force decisions differ from the physical control ledger")
        report = dict(version=VERSION, physics_rows=audit["measured_physics_rows"],
                      close_evidence=audit, ppo_updates=0, grasp_qualified=False, teacher_admission=False)
        if audit["terminal_reason"]:
            report.update(failure=audit["terminal_reason"], guarded_stop_sdk_step=audit["last_sdk_step"])
        elif audit["last_sdk_step"] != sdk_steps or audit["settled_control_windows"] != len(plan["action_schedule"]):
            raise ValueError("incomplete staged close without measured terminal")
        lifecycle.native.write_new(Path(output)/"staged_close_audit.json", report)
        return report
    finally:
        if journal is not None:
            journal.close()
        sim = owners = trace = fresh = episode = exchange = journal = result = None


def _checked(path, expected):
    if path.resolve(strict=True) != path or lifecycle.native.sha(path) != expected:
        raise ValueError("canonical frozen source bytes differ")
    return json.loads(path.read_bytes())


def freeze(output, sealed):
    manifest = _checked(sealed, SEALED_SHA)
    parent = manifest["source_refs"]["original_receipt"]
    parent_path = Path(parent["path"])
    received = _checked(parent_path, parent["sha256"])
    lifecycle._verify_outer(Path(received["output"]), parent_path, parent["sha256"])
    if received["pair_proposal"] != manifest["pair_proposal"] or received.get("forced_exit") is not False:
        raise ValueError("sealed same-candidate natural Reset required")
    from go2_pvcnn.control.dh116.tabletop_dual_pinch_builder import load_bound_pair_proposal
    bound = load_bound_pair_proposal(manifest["pair_proposal"])
    candidate = _checked(CANDIDATE, CANDIDATE_SHA)
    if bound["reset_seed"] != manifest["reset_seed"] or bound["staged_closure_source"]["config_sha256"] != CANDIDATE_SHA:
        raise ValueError("close and measured open Reset sources differ")
    plan = lifecycle.freeze_plan(output, pair_proposal=manifest["pair_proposal"])
    plan["frozen_files"].update({str(sealed): SEALED_SHA, str(CANDIDATE): CANDIDATE_SHA,
                                  str(parent_path): parent["sha256"]})
    plan.update(staged_version=VERSION, action_schedule=schedule(candidate["hands"]).tolist(),
                parent_reset_sha256=SEALED_SHA, preserved_scope="Pinch quiet plus fixed close, no lift or PPO")
    plan["budget"].update(control_steps=(QUIET_STEPS+CLOSE_STEPS)//8, sdk_steps=SDK_STEPS,
                          solver_steps=MEASURED_STEPS*SUBSTEPS)
    return plan


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("output", "plan", "sealed-reset"):
        p.add_argument("--"+n, type=Path, required=True)
    p.add_argument("--approved-frozen-sha256")
    args = p.parse_args(argv)
    lifecycle.native.validate_execution_context(ROOT)
    plan = freeze(args.output, args.sealed_reset)
    digest = lifecycle.native.digest(plan)
    if args.approved_frozen_sha256 is None:
        lifecycle.native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=digest, budget=plan["budget"])))
    else:
        if args.approved_frozen_sha256 != digest or json.loads(args.plan.read_text()) != plan or args.output.exists():
            raise ValueError("exact staged diagnostic recipe and fresh output required")
        lifecycle.execute(plan, acquire_operation=functools.partial(acquire, plan=plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
