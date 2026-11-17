"""Independent right-owned approach on original bounded parameter/lifetime owner."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
from scripts import prepare_dh116_bilateral_owned_pinch as plan
from scripts import prepare_dh116_pinch_margin_contact as margin
from scripts import prepare_dh116_native_parameter_task as previous
from scripts import prepare_dh116_dual_pinch_closure_interval as closure
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

VERSION="dh116-bilateral-owned-pinch-v1"
SCOPE="bilateral_owned_pinch_2_setup_232_guarded_contact_diagnostic_only"
PROFILE_SHA=None
ROOT=Path(__file__).resolve().parents[1]
CONFIG=plan.CONFIG
SETUP_STEPS,MEASURED_STEPS,TOTAL_STEPS=2,232,234
load_pair=plan.load_pair
action_schedule=plan.action_schedule
read_locked=margin.read_locked
sdk_clock_sample=margin.sdk_clock_sample

def make_geometry(project):
    from go2_pvcnn.control.dh116.tabletop_bilateral_owned_geometry import BilateralOwnedGeometry
    return BilateralOwnedGeometry(project)

@contextmanager
def scenario_scope(project=ROOT):
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116 import tabletop_measured_closure_geometry as geometry
    from scripts import dh116_pinch_margin_contact_audit as audit
    from scripts import validate_dh116_dual_pinch_director_interval as director_audit
    from scripts import validate_dh116_dual_pinch_closure_interval as closure_audit
    from scripts import validate_dh116_bilateral_owned_contact as own_audit
    packet=load_pair(project);path=Path(project)/CONFIG
    # Keep margin.load_pair untouched: plan.derive_pair reads the ORIGINAL left
    # candidate. A separate validator binds the new operation without mutating
    # the version used by the old candidate derivation.
    with bindings((builder,"PAIR",path),(builder,"PAIR_SHA",hashlib.sha256(path.read_bytes()).hexdigest()),
        (builder,"SEED_SHA",packet["reset_seed"]["sha256"]),(builder,"load_pair_proposal",lambda:load_pair(project)),
        (closure,"MEASURED_STEPS",MEASURED_STEPS),(closure,"TOTAL_STEPS",TOTAL_STEPS),
        (closure,"action_schedule",action_schedule),(closure,"make_geometry",make_geometry),
        (previous.director,"inspect_closure",audit.inspect_closure),
        (director_audit,"inspect_directors",audit.inspect_directors),
        (closure_audit,"validate_artifacts",own_audit.validate_artifacts),
        (geometry,"required_sdf_valid",margin.required_sdf_valid),
        (previous,"VERSION",VERSION),(previous,"PROFILE_SHA",PROFILE_SHA),
        (audit,"operation",sys.modules[__name__]),(audit,"SCOPE",SCOPE)):
        yield

def inspect_interval_bound(path,*,complete=True):
    return previous.inspect_interval(path,complete=complete)

def inspect_interval(path,*,complete=True):
    with scenario_scope():
        return inspect_interval_bound(path,complete=complete)

def run_existing_app(*args,**kwargs):
    try:
        with scenario_scope(kwargs.get("project",ROOT)):
            return previous.run_existing_app(*args,**kwargs)
    finally:
        args=kwargs=None
