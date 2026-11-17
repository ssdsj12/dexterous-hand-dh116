"""Independent same-clock host-getter evidence on the unchanged projected owner."""
from contextlib import contextmanager
from pathlib import Path
import json,os
from scripts import prepare_dh116_projected_pinch_native as previous
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

VERSION="dh116-projected-pinch-health-getter-v1"
SCOPE="projected_pinch_2_setup_232_same_clock_health_getter_only"
ROOT=Path(__file__).resolve().parents[1]
CONFIG=previous.CONFIG
SETUP_STEPS,MEASURED_STEPS,TOTAL_STEPS=2,232,234
PROFILE_SHA=None
load_pair=previous.load_pair
action_schedule=previous.action_schedule
sdk_clock_sample=previous.sdk_clock_sample
read_locked=previous.read_locked

@contextmanager
def scenario_scope(project=ROOT):
    from scripts import prepare_dh116_dual_pinch_director_interval as director
    from go2_pvcnn.control.dh116.tabletop_health_getter_evidence import HealthEvidenceMixin
    with bindings((previous,"VERSION",VERSION),(previous,"SCOPE",SCOPE),(previous,"PROFILE_SHA",PROFILE_SHA)):
        with previous.scenario_scope(project):
            original=director.DirectorExchange
            if issubclass(original,HealthEvidenceMixin):owned=original
            else:
                class OwnedHealthExchange(HealthEvidenceMixin,original):pass
                owned=OwnedHealthExchange
            with bindings((director,"DirectorExchange",owned)):yield

def inspect_interval(path,*,complete=True):
    from scripts.validate_dh116_health_projected_pinch import inspect_health
    with scenario_scope():
        health=inspect_health(path,complete=complete)
        if complete:return dict(previous.inspect_interval(path,complete=True),health_getter_audit=health)
        # The real journal's terminal is retained. New health failures cannot be
        # rewritten into a passing old partial contract.
        return dict(health["journal"],health_getter_audit=health,full_parent_audit_performed=False)

def scope_marker():
    from go2_pvcnn.control.dh116.tabletop_health_getter_evidence import VERSION as evidence
    return dict(previous.scope_marker(),version=VERSION,health_evidence_version=evidence,
        live_r0_computed=False,health_native_qualified=False,
        mesh_reset_qualified=False,substep_contact_frame_qualified=False)

def run_existing_app(*args,**kwargs):
    output=Path(kwargs["output"]);marker=scope_marker()
    try:
        with scenario_scope(kwargs.get("project",ROOT)),bindings((previous,"scope_marker",lambda:marker)):
            result=previous.run_existing_app(*args,**kwargs)
        with (output/"health_getter_scope.json").open("x") as stream:
            json.dump(marker,stream,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
        return result
    finally:args=kwargs=None
