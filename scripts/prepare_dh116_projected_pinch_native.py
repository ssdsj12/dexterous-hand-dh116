"""Independent diagnostic accepting only the recomputed ORIGINAL safety projection."""
from contextlib import contextmanager
from pathlib import Path
import sys,json
from scripts import prepare_dh116_left_pinch_approach_native as previous
from scripts import prepare_dh116_left_pinch_approach as plan
from scripts import dh116_pinch_margin_contact_audit as audit
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
VERSION="dh116-original-projected-pinch-contact-v1"
SCOPE="pinch_original_safety_projection_2_setup_232_diagnostic_only"
ROOT=Path(__file__).resolve().parents[1]
CONFIG=plan.CONFIG
SETUP_STEPS,MEASURED_STEPS,TOTAL_STEPS=2,232,234
PROFILE_SHA=None
load_pair=plan.load_pair
action_schedule=plan.action_schedule
read_locked=previous.read_locked
sdk_clock_sample=previous.sdk_clock_sample
_BASE_INSPECT=audit.inspect_closure
def make_geometry(project):
    from go2_pvcnn.control.dh116.tabletop_projected_pinch_geometry import ProjectedPinchGeometry
    return ProjectedPinchGeometry(project)

def inspect_projected_closure(path,*,complete=True):
    from go2_pvcnn.control.dh116.tabletop_projected_pinch_geometry import validate_projected_target,validate_continuity
    result=_BASE_INSPECT(path,complete=complete)
    old=None;count=active=0;pair=load_pair()
    with Path(path).open() as stream:
        for line in stream:
            entry=json.loads(line)
            if entry.get("operation")!="physical":continue
            current=decode_owned(entry["payload"])["exchange"]
            checked=validate_projected_target(pair,current)
            validate_continuity(old,current)
            active+=int(any(checked["safe_z_active"]));count+=1;old=current
    if count!=result["measured_physics_rows"]:raise ValueError("all measured projections required")
    return dict(result,projection_source_rows=count,projection_continuity_verified=True,
        safe_z_active_rows=active,fixed_unprojected_path_qualified=False)

@contextmanager
def scenario_scope(project=ROOT):
    from scripts import prepare_dh116_dual_pinch_director_interval as director
    from go2_pvcnn.control.dh116.tabletop_wrist_projection_evidence import ProjectionEvidenceMixin
    # Nesting execute/run/post-exit scopes must not wrap a reader twice.
    original=director.DirectorExchange
    if issubclass(original,ProjectionEvidenceMixin):owned=original
    else:
        class OwnedProjectionExchange(ProjectionEvidenceMixin,original):pass
        owned=OwnedProjectionExchange
    with bindings((previous,"VERSION",VERSION),(previous,"SCOPE",SCOPE),
        (previous,"PROFILE_SHA",PROFILE_SHA),(previous,"make_geometry",make_geometry),
        (audit,"inspect_closure",inspect_projected_closure),(director,"DirectorExchange",owned)):
        with previous.scenario_scope(project):
            with bindings((audit,"operation",sys.modules[__name__])):yield

def inspect_interval(path,*,complete=True):
    with scenario_scope():return previous.previous.inspect_interval_bound(path,complete=complete)

def scope_marker():
    from go2_pvcnn.control.dh116.tabletop_wrist_projection_evidence import VERSION as evidence
    from go2_pvcnn.control.dh116.tabletop_projected_pinch_geometry import VERSION as target
    return dict(previous.scope_marker(),version=VERSION,projection_input_evidence=evidence,
        target_contract=target,original_safety_projection_unchanged=True,
        fixed_unprojected_path_qualified=False)

def run_existing_app(*args,**kwargs):
    import os
    marker=scope_marker()
    output=Path(kwargs["output"])
    try:
        with scenario_scope(kwargs.get("project",ROOT)),bindings((previous,"scope_marker",lambda:marker)):
            result=previous.run_existing_app(*args,**kwargs)
        with (output/"projected_pinch_scope.json").open("x") as f:
            json.dump(marker,f,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
        return result
    finally:args=kwargs=None
