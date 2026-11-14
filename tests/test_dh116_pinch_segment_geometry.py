import hashlib
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]

def test_scope_installs_new_factory_and_restores_on_exception():
    from scripts import prepare_dh116_pinch_segment_contact as op
    from scripts import prepare_dh116_pinch_margin_contact as old
    from scripts import prepare_dh116_dual_pinch_closure_interval as owner
    before=(old.make_geometry,owner.make_geometry,old.previous.VERSION)
    with pytest.raises(RuntimeError):
        with op.scenario_scope():
            assert old.make_geometry is owner.make_geometry is op.make_geometry
            assert old.previous.VERSION==op.VERSION
            raise RuntimeError("restore")
    assert (old.make_geometry,owner.make_geometry,old.previous.VERSION)==before

def test_scenario_has_identical_reset_action_and_original_guards():
    from scripts import prepare_dh116_pinch_segment_contact as op
    from scripts import prepare_dh116_pinch_margin_contact as old
    assert op.load_pair()==old.load_pair()
    assert np.array_equal(op.action_schedule().numpy(),old.action_schedule().numpy())
    assert op.CONFIG==old.CONFIG
    for n,h in {
        "src/go2_pvcnn/control/dh116/tabletop_measured_closure_geometry.py":"3e90dca3dfc2448ad2a1e82f545c29e7229fbda7510b90fba45bf88a5a7132de",
        "scripts/prepare_dh116_pinch_margin_contact.py":"09a369027259e3aa7218a662ea72214ae2203344065221512f066ecccb066b9f"
    }.items():
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h

def test_legacy_packet_cannot_be_validated_as_new_contract():
    from go2_pvcnn.control.dh116.tabletop_segment_closure_geometry import SegmentClosureGeometry
    g=object.__new__(SegmentClosureGeometry);g.sources={}
    with pytest.raises(ValueError):g.validate({"version":"dh116-measured-closure-geometry-v1","sources":{}},{})
