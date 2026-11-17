"""Frozen dual-hand lifetime checks with an explicit final step count.

Copied from validate_dh116_dual_pinch_interval._lifetime without changing the
original source. Only the fixed ten-step comparison is parameterized. The
test compares the original AST body after that exact substitution.
"""
import math
import re
from scripts.validate_dh116_dual_pinch_interval import (
    BOUNDARY_VERSION,PHASES,OWNED_NAMES,DT,require,exact,read,rows,sha,native_artifact)


def lifetime_evidence(output,result,*,expected_steps):
    require(type(expected_steps) is int and expected_steps>0,"positive expected SDK step required")
    clocks=result["clocks"]
    require([r.get("phase") for r in clocks]==list(PHASES[2:]),"lifecycle clock phases differ")
    require(clocks[0].get("clock_available") is True,"pre-STOP actual clock missing")
    for row in clocks:
        require(type(row.get("clock_available")) is bool,"unknown lifecycle clock availability")
        if row["clock_available"]:
            require(type(row.get("sdk_step")) is int and row["sdk_step"]==expected_steps
                and math.isclose(row["sdk_time_s"],expected_steps*DT,rel_tol=0,abs_tol=2e-7),
                "clock advanced/reset after measured interval")
        else:require("sdk_step" not in row and "sdk_time_s" not in row,"fabricated unavailable clock")
    stop=[r for r in rows(native_artifact(output,"ownership.jsonl")) if r.get("phase")=="SDK STOP"]
    require(len(stop)==1 and stop[0].get("manager_view_released") is True,"manager root view survived STOP")
    liveness={};simulation={}
    for phase in PHASES:
        state=read(native_artifact(output,phase+".refs.json"))["resources"]
        expected=set(OWNED_NAMES)-({"table_contact_source"} if phase==PHASES[0] else set())
        require(expected<=set(state),"actual owned resource liveness missing")
        require(re.search(r"(?m)^[0-9a-f]+-[0-9a-f]+\s",native_artifact(output,phase+".maps").read_text()),
                "phase process maps missing")
        liveness[phase]={name:item.get("alive") for name,item in state.items()}
        simulation[phase]=state["simulation"].get("alive")
        require(simulation[phase] is None or type(simulation[phase]) is bool,"invalid simulation liveness")
        if phase==PHASES[0]:
            require(all(state[n].get("alive") is True for n in expected),"initial actual owners not observed alive")
        if phase in PHASES[3:]:
            require(all(v is False for n,v in liveness[phase].items() if n!="simulation"),
                    "non-Sim owner live or unknown after STOP/close")
    full_released=all(v is False for v in liveness["before_app_close"].values())
    exact(result,dict(lifetime_released_before_close=full_released),"full liveness result contradicts observation")
    before=read(native_artifact(output,"before_app_close.json"))
    exact(before,dict(case="explicit",play_simulations=False,lifetime_released=full_released,
                      process_exit_verified=False),"pre-close frozen/liveness evidence differs")
    require(before["clocks"]==clocks[:3],"pre-close clock evidence differs")
    closed=read(native_artifact(output,"app_close_returned.json"))
    require(closed==dict(case="explicit",app_close_returned=True,process_exit_verified=False),"actual App close return missing")
    from go2_pvcnn.control.dh116.tabletop_simulation_owner_cleanup import VERSION as CLEAR_VERSION
    receipt=read(native_artifact(output,"simulation_clear_receipt.json"))
    expected=dict(version=CLEAR_VERSION,subscription_status="explicitly_unsubscribed",
        unsubscribe_returned=True,subscription_field_cleared=True,original_clear_instance_returned=True,
        extra_gc_calls=0,native_release_verified=False,simulation_liveness_after_return=None,
        constructor_frame_owner_resolved=False,physics_steps_executed=0)
    require(set(receipt)==set(expected),"PHYSICS_READY clear receipt fields differ")
    exact(receipt,expected,"PHYSICS_READY unsubscribe/original clear sequence incomplete")
    boundary=read(native_artifact(output,"before_app_launcher.json"))
    require(boundary==dict(version=BOUNDARY_VERSION,phase="before_isaaclab_app_import",
        loaded_usd_modules=[],native_execution_performed=False),"clean import boundary differs")
    return dict(liveness=liveness,simulation_liveness=simulation,owned_non_sim_released=True,
                simulation_clear_receipt_sha256=sha(output/"simulation_clear_receipt.json"),
                simulation_release_required_for_execution_qualification=False)
