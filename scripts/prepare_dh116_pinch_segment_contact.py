"""Same 234-step Pinch reset/actions; independently versioned 3D geometry proof."""
from contextlib import contextmanager
from pathlib import Path
from scripts import prepare_dh116_pinch_margin_contact as margin
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

VERSION="dh116-pinch-segment-contact-v1"
SCOPE="left_pinch_segment_2_setup_232_guarded_contact_diagnostic_only"
PROFILE_SHA=None
ROOT=Path(__file__).resolve().parents[1]
CONFIG=margin.CONFIG
SETUP_STEPS,MEASURED_STEPS,TOTAL_STEPS=2,232,234
load_pair=margin.load_pair
action_schedule=margin.action_schedule
read_locked=margin.read_locked

def make_geometry(project):
    from go2_pvcnn.control.dh116.tabletop_segment_closure_geometry import SegmentClosureGeometry
    return SegmentClosureGeometry(project)

@contextmanager
def scenario_scope(project=ROOT):
    with margin.scenario_scope(project),bindings(
            (margin,"make_geometry",make_geometry),(margin.closure,"make_geometry",make_geometry),
            (margin.previous,"VERSION",VERSION),(margin.previous,"PROFILE_SHA",PROFILE_SHA)):
        yield

def inspect_interval(path,*,complete=True):
    with scenario_scope():
        return margin.previous.inspect_interval(path,complete=complete)

def run_existing_app(*args,**kwargs):
    try:
        with scenario_scope(kwargs.get("project",ROOT)):
            return margin.run_previous(*args,**kwargs)
    finally:
        args=kwargs=None
