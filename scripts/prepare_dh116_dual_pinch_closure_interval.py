"""Two fixed no-lift intervals on the original paired Pinch physical owner."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path

from scripts import prepare_dh116_dual_pinch_interval as original
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from scripts.run_dh116_work_packages import native_physics_budget
from go2_pvcnn.control.dh116.tabletop_deformable_clearance_guard import ClearanceGuardedExchange

VERSION = "dh116-dual-pinch-guarded-closure-v1"
SETUP_STEPS = 2
MEASURED_STEPS = 16
TOTAL_STEPS = SETUP_STEPS + MEASURED_STEPS
PROFILE_SHA = None
run_original_existing_app = original.run_existing_app


def action_schedule():
    import torch

    actions = torch.zeros((2, 24), dtype=torch.float64)
    actions[1, [1, 2, 7, 8]] = torch.tensor([.1, .1, .05, .1], dtype=actions.dtype)
    return actions


def make_geometry(project):
    # USD/source loading is deliberately deferred until after App creation.
    from go2_pvcnn.control.dh116.tabletop_measured_closure_geometry import MeasuredClosureGeometry
    return MeasuredClosureGeometry(project)


def observation_failure(row):
    """Validate packet/error consistency and retain the original guard first."""
    for name, field in (("geometry", "closure_geometry"), ("table", "hand_table_contacts")):
        error, packet = row[name + "_capture_error"], row[field]
        if error is None:
            if not isinstance(packet, dict):
                raise ValueError(name + " successful capture requires an owned packet")
        elif not isinstance(error, str) or not error.strip() or packet is not None:
            raise ValueError(name + " failed capture requires error text and an unknown packet")
    reason = row["original_guard_failure"]
    if reason is None:
        if row["geometry_capture_error"] is not None:
            reason = "closure_geometry_source_invalid"
        elif row["closure_geometry"]["terminal_reason"] is not None:
            reason = row["closure_geometry"]["terminal_reason"]
        elif row["table_capture_error"] is not None:
            reason = "closure_table_source_invalid"
    return reason


def complete_observation(row, *, geometry, table):
    """A failed auxiliary read must not discard the already completed SDK row."""
    read = None
    try:
        row["original_guard_failure"] = row["guard_failure"]
        for name, read, field in (("geometry", geometry, "closure_geometry"),
                                  ("table", table, "hand_table_contacts")):
            try:
                packet = read()
                if not isinstance(packet, dict):
                    raise ValueError("owned observation packet required")
                if name == "geometry" and (packet.get("passed") not in (True, False)
                        or "terminal_reason" not in packet):
                    raise ValueError("geometry result and terminal reason required")
                row[field] = packet
                row[name + "_capture_error"] = None
            except Exception as exc:
                row[field] = None
                row[name + "_capture_error"] = type(exc).__name__ + ": " + str(exc)
        row["guard_failure"] = observation_failure(row)
        return row
    finally:
        row = geometry = table = read = None


class ClosureExchange(ClearanceGuardedExchange):
    """Complete original physics first, then attach same-boundary observations."""

    def _step(self, action, substep):
        row = None
        try:
            from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts import read_bimanual_table_contacts
            row = super()._step(action, substep)
            return complete_observation(row,
                geometry=lambda: self._closure_geometry.capture(row, sim=self._sim,
                    bindings=self._bindings, read_clock=self._rod.clock),
                table=lambda: read_bimanual_table_contacts(self._closure_contacts, read_clock=self._rod.clock))
        finally:
            self = action = row = None


def sdk_clock_sample(row, ordinal):
    exchange = row["exchange"]
    post = exchange["action_readback"]["post"]["identity"]
    rod = exchange["rod"]
    step = exchange["physics_step"]
    sdk_time = exchange["post_step_clearance"]["sdk_time_s"]
    if (type(ordinal) is not int or ordinal < 1 or type(step) is not int
            or step != ordinal + SETUP_STEPS or len(post) != 1
            or rod["physics_step"] != step or post[0]["physics_step"] != step
            or type(sdk_time) not in (int, float) or not math.isfinite(sdk_time)
            or not math.isclose(sdk_time, rod["timestamp_s"], rel_tol=0, abs_tol=2e-7)
            or not math.isclose(rod["timestamp_s"], step / 480, rel_tol=0, abs_tol=2e-7)
            or not math.isclose(post[0]["timestamp_s"], rod["timestamp_s"], rel_tol=0, abs_tol=2e-7)):
        raise ValueError("actual SDK, rod and post-action clock differ")
    return dict(phase_step=ordinal, sdk_step=step, policy_step=exchange["policy_step"],
                substep=exchange["substep"], timestamp_s=rod["timestamp_s"], sdk_time_s=sdk_time)


def inspect_interval(path, *, complete=True):
    import numpy as np
    from go2_pvcnn.control.dh116.tabletop_deformable_clearance_guard import validate_post_step_clearance
    from go2_pvcnn.control.dh116.tabletop_deformable_dual_evidence import inspect_evidence
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned, encode_owned
    from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts import IDENTITY

    audit = inspect_evidence(path)
    if audit["physics_budget_end"] != TOTAL_STEPS or audit["solver_substeps"] != 12:
        raise ValueError("exact frozen closure physics budget required")
    if complete and (audit["last_sdk_step"] != TOTAL_STEPS or audit["measured_physics_rows"] != MEASURED_STEPS
            or audit["complete_control_windows"] != 2 or audit["settled_control_windows"] != 2
            or audit["unsettled_control_window"] or audit["terminal_reason"] is not None):
        raise ValueError("two complete safe windows required for closure execution qualification")
    geometry = make_geometry(Path(__file__).resolve().parents[1])
    begins = []
    source_errors = {"geometry": 0, "table": 0}
    table_na_rows = 0
    min_by_window = [None, None]
    contacts = {}
    table_path = "/World/DeformXDualHand/Table"
    with Path(path).open() as stream:
        next(stream)
        for line in stream:
            entry = json.loads(line)
            row = decode_owned(entry["payload"])
            if entry["operation"] == "begin":
                begins.append(row["action"])
                if int(np.asarray(row["pre_action_state"]["common"]["valid"]).sum()) != 66:
                    raise ValueError("original 66 valid common fields required")
            if entry["operation"] != "physical":
                continue
            exchange = row["exchange"]
            original_exchange = dict(exchange, guard_failure=exchange["original_guard_failure"])
            actual = validate_post_step_clearance(original_exchange)
            reason = observation_failure(exchange)
            if reason != exchange["guard_failure"]:
                raise ValueError("closure terminal differs from original guard and observation results")
            for source in source_errors:
                source_errors[source] += int(exchange[source + "_capture_error"] is not None)
            if exchange["geometry_capture_error"] is None:
                geometry.validate(exchange["closure_geometry"], original_exchange)
            if encode_owned(row["hand_table_contacts"]) != encode_owned(exchange["hand_table_contacts"]):
                raise ValueError("durable table packet differs from completed exchange")
            index = exchange["policy_step"]
            gap = actual["minimum_clearance_m"]
            min_by_window[index] = gap if min_by_window[index] is None else min(gap, min_by_window[index])
            table = row["hand_table_contacts"]
            rod = exchange["rod"]
            if table is not None:
                table_na_rows += 1
            if table is not None and (table["available"] is not False or table["hands"] is not None
                    or table["total_contacts"] is not None or table["native_qualified"] is not False
                    or table["source"]["table_is_static"] is not True
                    or table["source"]["table_path"] != table_path
                    or table["source"]["table_colliders"]["collider_paths"] != [table_path + "/geometry/mesh"]
                    or table["identity"] != {key: rod[key] for key in IDENTITY}):
                raise ValueError("same-clock actual static table with unknown contact required")
            pairs = rod["rod_mesh_contact_pairs"]
            for pair in pairs["pairs"]:
                mesh = pair["mesh_name"]
                contacts[mesh] = contacts.get(mesh, 0) + 1
    schedule = action_schedule().numpy()
    if (len(begins) > len(schedule) or any(not np.array_equal(actual, expected[None])
            for actual, expected in zip(begins, schedule)) or (complete and len(begins) != len(schedule))):
        raise ValueError("frozen zero then finger-only action sequence differs")
    return dict(audit, static_table_contact_na_rows=table_na_rows, observation_source_error_rows=source_errors,
                minimum_source_clearance_m_by_window=min_by_window, rod_contact_pairs_by_mesh=contacts,
                hand_table_source_available=False, no_hand_table_contact_proven=False)


def _owned_interval(sim, owners, trace, *, output, project, prepared, owner_resets,
                    run_uuid, episode_uid, code_config_sha256):
    episode = rod = contacts = journal = scene = clock_stream = geometry = exchange = None
    measured_rows = 0
    try:
        from go2_pvcnn.control.dh116 import tabletop_deformable_dual_hand_episode as physical
        from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
        from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts_v2 import open_bimanual_table_contacts
        from go2_pvcnn.control.dh116.tabletop_deformable_dual_evidence import DualHandEvidenceJournal

        journal_path = Path(output) / "physical.jsonl"
        journal = DualHandEvidenceJournal(journal_path, physics_budget_end=TOTAL_STEPS, solver_substeps=12)
        clock_stream = (Path(output) / "sdk_clock.jsonl").open("x")

        def record(row):
            nonlocal measured_rows
            if row["kind"] == "physical":
                row = dict(row, hand_table_contacts=deepcopy(row["exchange"]["hand_table_contacts"]))
            journal.record(row)
            if row["kind"] == "physical":
                measured_rows += 1
                clock_stream.write(json.dumps(sdk_clock_sample(row, measured_rows)) + "\n")
                clock_stream.flush()
                os.fsync(clock_stream.fileno())

        with bindings((physical, "FlexiblePhysicalExchange", ClosureExchange)):
            setup = builder.build_reset_owned(
                sim, owners, project=project, prepared=prepared, ownership=trace, record=record,
                owner_resets=owner_resets, run_uuid=run_uuid, episode_uid=episode_uid, physics_budget_end=TOTAL_STEPS,
            )
        episode = owners.resources["dual_episode"]
        rod = owners.resources["runtime"]
        scene = owners.resources["scene"]
        contacts = owners.keep("table_contact_source", open_bimanual_table_contacts(
            sim.stage, scene["hands"], prepared, read_clock=rod.clock,
        ))
        trace.watch("table_contact_source", contacts, owner="dual interval", release="owned scope after sync")
        geometry = make_geometry(project)
        exchange = episode._exchange
        exchange._closure_geometry = geometry
        exchange._closure_contacts = contacts
        for action in action_schedule():
            action = action.unsqueeze(0)
            journal.begin(episode.pre_action_state(), action)
            result = episode.control(action)
            journal.end(result)
            if result["termination_reason"] is not None:
                raise RuntimeError(result["termination_reason"])
        journal.close()
        audit = inspect_interval(journal_path)
        clock = dict(sdk_initial_step=0, sdk_after_setup_step=2, sdk_first_measurement_step=3,
            sdk_last_measurement_step=TOTAL_STEPS, sdk_final_step=rod.clock()["physics_step"],
            setup_steps=SETUP_STEPS, measured_steps=MEASURED_STEPS, total_steps=TOTAL_STEPS)
        budget = native_physics_budget(initialization_physics_steps=SETUP_STEPS, warmup_physics_steps=0,
            measurement_physics_steps=MEASURED_STEPS, other_physics_advancing_steps=0)
        return dict(version=VERSION, setup=setup, physics_rows=MEASURED_STEPS, physics_budget=budget,
            physics_clock=clock, journal=audit, hand_table_source_available=False,
            no_hand_table_contact_proven=False, native_qualification=False, teacher_admission=False,
            grasp_qualified=False, ppo_updates=0, code_config_sha256=code_config_sha256,
            action_schedule=action_schedule().tolist(), profile_sha256=PROFILE_SHA)
    finally:
        if journal is not None:
            journal.close()
        if clock_stream is not None:
            clock_stream.close()
        sim = owners = trace = prepared = owner_resets = episode = rod = contacts = journal = scene = clock_stream = geometry = exchange = None


def run_existing_app(*args, **kwargs):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation_v2 import external_wire_layout

    try:
        if kwargs["prepared"] is None:
            kwargs["prepared"] = scene.prepare_scene(kwargs["project"])
        with external_wire_layout(), bindings(
                (original, "VERSION", VERSION), (original, "_owned_interval", _owned_interval)):
            return run_original_existing_app(*args, **kwargs)
    finally:
        args = kwargs = None
