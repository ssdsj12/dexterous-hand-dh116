#!/usr/bin/env python3
"""CPU-prepared dual Pinch operation; no native launcher, recipe or approval."""
import argparse
import json
from pathlib import Path
import sys

VERSION="dh116-dual-pinch-one-interval-preparation-v1"
BUDGET=10
OWNED_NAMES=tuple(side+"_"+suffix for side in ("left","right") for suffix in
    ("hand","view","asset_sim_view","permanent_composer","instantaneous_composer"))+(
    "runtime","rope","episode","table_contact_source","simulation")


def assert_clean_import_boundary():
    loaded=sorted(n for n in sys.modules if n.split(".")[0] in ("pxr","PhysxSchema"))
    if loaded:raise RuntimeError("USD imported before AppLauncher: "+", ".join(loaded))
    return dict(loaded_usd_modules=[])


def _owned_interval(sim,owners,trace,*,output,project,prepared,owner_resets,run_uuid,episode_uid,code_config_sha256):
    episode=rod=contacts=journal=record=scene=None
    try:
        import torch
        from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
        from go2_pvcnn.control.dh116.tabletop_deformable_dual_evidence import DualHandEvidenceJournal
        from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts import (
            open_bimanual_table_contacts,read_bimanual_table_contacts)
        journal=DualHandEvidenceJournal(Path(output)/"physical.jsonl",
            physics_budget_end=BUDGET,solver_substeps=12)
        def record(row):
            if row["kind"]=="physical":
                row=dict(row,hand_table_contacts=read_bimanual_table_contacts(contacts,read_clock=rod.clock))
            journal.record(row)
        setup=builder.build_reset_owned(sim,owners,project=project,prepared=prepared,
            ownership=trace,record=record,owner_resets=owner_resets,run_uuid=run_uuid,
            episode_uid=episode_uid,physics_budget_end=BUDGET)
        episode=owners.resources["dual_episode"];rod=owners.resources["runtime"]
        scene=owners.resources["scene"]
        contacts=owners.keep("table_contact_source",open_bimanual_table_contacts(
            sim.stage,scene["hands"],prepared,read_clock=rod.clock))
        trace.watch("table_contact_source",contacts,owner="dual interval",release="owned scope after sync")
        if contacts.metadata["table_is_static"] is not True:
            raise ValueError("unchanged original static table required")
        pre=episode.pre_action_state()
        action=torch.zeros((1,24),dtype=torch.float64)
        journal.begin(pre,action)
        result=episode.control(action)
        journal.end(result)
        journal.close()
        audit=inspect_interval(Path(output)/"physical.jsonl")
        from scripts.run_dh116_work_packages import native_physics_budget
        if code_config_sha256 is not None:
            sdk_rows = write_sdk_clock(Path(output)/"physical.jsonl", Path(output)/"sdk_clock.jsonl")
            clock = dict(sdk_initial_step=0, sdk_after_setup_step=2,
                sdk_first_measurement_step=sdk_rows[0]["sdk_step"],
                sdk_last_measurement_step=sdk_rows[-1]["sdk_step"], sdk_final_step=rod.clock()["physics_step"],
                setup_steps=2, measured_steps=len(sdk_rows), total_steps=10)
            budget = native_physics_budget(initialization_physics_steps=2, warmup_physics_steps=0,
                measurement_physics_steps=len(sdk_rows), other_physics_advancing_steps=0)
        else:
            clock = rod.clock()
            budget = dict(total_sdk_steps=10,setup_steps=2,measured_steps=8,solver_substeps=12)
        return dict(version=VERSION,setup=setup,physics_rows=audit["measured_physics_rows"],
            physics_budget=budget,physics_clock=clock,journal=audit,hand_table_source_available=False,
            no_hand_table_contact_proven=False,native_qualification=False,teacher_admission=False,
            grasp_qualified=False,ppo_updates=0,code_config_sha256=code_config_sha256)
    finally:
        if journal is not None:journal.close()
        # NativeOwners retains the contact views through composer reset and
        # synchronization. Do not release them early from this inner scope.
        sim=owners=trace=prepared=owner_resets=episode=rod=contacts=journal=record=scene=None


def inspect_interval(path):
    """Independently replay the existing journal and this operation's NA scope."""
    import numpy as np
    from go2_pvcnn.control.dh116.tabletop_deformable_dual_evidence import inspect_evidence
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    audit=inspect_evidence(path)
    if (audit["physics_budget_end"]!=10 or audit["solver_substeps"]!=12
            or audit["last_sdk_step"]!=10 or audit["measured_physics_rows"]!=8
            or audit["complete_control_windows"]!=1 or audit["settled_control_windows"]!=1
            or audit["unsettled_control_window"] or audit["terminal_reason"] is not None):
        raise ValueError("one complete zero-action 2+8 diagnostic interval required")
    count=0;begin=0
    with Path(path).open() as stream:
        next(stream)
        for line in stream:
            entry=json.loads(line);row=decode_owned(entry["payload"])
            if entry["operation"]=="begin":
                begin+=1
                if np.any(row["action"]):raise ValueError("exact zero action required; no lift")
                if int(np.asarray(row["pre_action_state"]["common"]["valid"]).sum())!=66:
                    raise ValueError("original partial 66/371 state required")
            if entry["operation"]=="physical":
                packet=row["hand_table_contacts"];clock=row["exchange"]["rod"]
                if (packet["available"] is not False or packet["total_contacts"] is not None
                        or packet["hands"] is not None
                        or packet["source"]["table_is_static"] is not True
                        or "static GPU" not in packet["reason"]
                        or packet["native_qualified"] is not False
                        or any(value!=clock[key] for key,value in packet["identity"].items())):
                    raise ValueError("static hand-table source must remain same-boundary NA")
                count+=1
    if count!=8 or begin!=1:raise ValueError("exact one decision/eight contact NA rows required")
    return dict(audit,static_table_contact_na_rows=count,hand_table_source_available=False,
                no_hand_table_contact_proven=False)


def write_sdk_clock(journal_path, path):
    """Copy scheduler clock rows from the independently inspected physical journal."""
    import os

    rows = []
    with Path(journal_path).open() as stream:
        next(stream)
        for line in stream:
            entry = json.loads(line)
            if entry["operation"] == "physical":
                exchange = entry["payload"]["exchange"]
                rows.append(dict(phase_step=len(rows)+1,sdk_step=exchange["physics_step"]))
    if [row["sdk_step"] for row in rows] != list(range(3, 11)):
        raise ValueError("expected exact measured SDK clock in durable physical journal")
    with Path(path).open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row)+"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return rows


def run_existing_app(app,*,make_sim,synchronize,clear_sim,stop,manager_is_none,freeze,
                     output,project,prepared,owner_resets,run_uuid,episode_uid,code_config_sha256=None):
    """Compose existing lifecycle; caller owns a future separately gated App.

    clear_sim must use the source-verified simulation-owner helper in a future
    native package. CPU tests inject SDK-shaped lifecycle doubles. This function
    imports no AppLauncher and acquires no lock/worker. The existing lifecycle
    invokes the caller-provided make_sim; native adapters require a future gate.
    """
    acquire=None
    try:
        from scripts import dh116_pinch_acquisition_v2 as lifecycle
        from scripts.dh116_pinch_startup_v3 import StartupTrace,snapshot_for_startup
        from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
        def acquire(sim,owners,trace,cleanup):
            try:
                return _owned_interval(sim,owners,trace,output=output,project=project,prepared=prepared,
                    owner_resets=owner_resets,run_uuid=run_uuid,episode_uid=episode_uid,
                    code_config_sha256=code_config_sha256)
            finally:sim=owners=trace=None
        with bindings((lifecycle,"VERSION",VERSION),(lifecycle,"OWNED_NAMES",OWNED_NAMES),
                      (lifecycle,"LifetimeTrace",StartupTrace),
                      (lifecycle,"snapshot_weakrefs",snapshot_for_startup)):
            return lifecycle.run_lifecycle(app,make_sim=make_sim,acquire=acquire,
                synchronize=synchronize,clear_sim=clear_sim,stop=stop,manager_is_none=manager_is_none,
                freeze=freeze,output=output,cleanup="explicit")
    finally:
        app=make_sim=synchronize=clear_sim=stop=manager_is_none=freeze=prepared=owner_resets=acquire=None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-native",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.execute_native:
        parser.error("native execution unavailable: no dedicated frozen worker, recipe or approval")
    assert_clean_import_boundary()
    print(json.dumps(dict(version=VERSION,native_execution_available=False,
        intended_sdk_steps=10,setup_steps=2,measured_steps=8,action="exact_zero_24D",
        physics_steps_executed=0,optimizer_updates=0,hand_table_source_available=False,
        no_hand_table_contact_proven=False,teacher_admission=False)))

if __name__=="__main__":main()
