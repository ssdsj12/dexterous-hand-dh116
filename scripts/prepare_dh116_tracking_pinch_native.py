"""Tracking-aware right candidate; reuse complete original physics/lifetime owner."""
from contextlib import contextmanager
from pathlib import Path
from scripts import prepare_dh116_tracking_pinch as plan
from scripts import prepare_dh116_effective_right_native as previous
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

VERSION="dh116-tracking-aware-pinch-contact-v1"
SCOPE="tracking_aware_pinch_2_setup_232_original_guard_contact_diagnostic_only"
ROOT=Path(__file__).resolve().parents[1]
CONFIG=plan.CONFIG
SETUP_STEPS,MEASURED_STEPS,TOTAL_STEPS=2,232,234
PROFILE_SHA=None
load_pair=plan.load_pair
action_schedule=plan.action_schedule
read_locked=previous.read_locked
sdk_clock_sample=previous.sdk_clock_sample

def make_geometry(project):
    from go2_pvcnn.control.dh116.tabletop_tracking_pinch_geometry import TrackingPinchGeometry
    return TrackingPinchGeometry(project)

@contextmanager
def scenario_scope(project=ROOT):
    from scripts import validate_dh116_effective_right_contact as validator
    from scripts import dh116_pinch_margin_contact_audit as audit
    import sys
    # Bind the owner only. Historical plan modules remain unchanged, including
    # their expected-boundary functions used by correlated tracking proofs.
    with bindings((previous,"VERSION",VERSION),(previous,"SCOPE",SCOPE),
        (previous,"CONFIG",CONFIG),(previous,"PROFILE_SHA",PROFILE_SHA),
        (previous,"load_pair",load_pair),(previous,"action_schedule",action_schedule),
        (previous,"make_geometry",make_geometry),(validator,"SCOPE",SCOPE)):
        with previous.scenario_scope(project):
            with bindings((audit,"operation",sys.modules[__name__])):yield

def inspect_interval(path,*,complete=True):
    with scenario_scope():return previous.inspect_interval_bound(path,complete=complete)

def scope_marker():
    from scripts.dh116_deferred_physical_audit import VERSION as seal
    from go2_pvcnn.control.dh116.tabletop_owned_clock_projection import VERSION as clock
    return dict(version=VERSION,seal_version=seal,clock_projection_version=clock,
        bindings_restored=True,post_exit_complete_audit_required=True,
        native_qualification=False,teacher_admission=False)

def run_existing_app(*args,**kwargs):
    from scripts.dh116_deferred_physical_audit import seal_pending
    from go2_pvcnn.control.dh116.tabletop_owned_clock_projection import clock_projection_scope
    import json,os
    parameter=previous.previous
    old=parameter.inspect_interval
    def producer_audit(path,*,complete=True):
        return seal_pending(path) if complete else old(path,complete=False)
    output=Path(kwargs["output"])
    try:
        with scenario_scope(kwargs.get("project",ROOT)),clock_projection_scope(),bindings(
                (parameter,"inspect_interval",producer_audit)):
            result=parameter.run_existing_app(*args,**kwargs)
        if parameter.inspect_interval is not old:raise RuntimeError("producer audit binding not restored")
        with (output/"tracking_pinch_scope.json").open("x") as file:
            json.dump(scope_marker(),file,allow_nan=False);file.write("\n");file.flush();os.fsync(file.fileno())
        return result
    finally:args=kwargs=None
