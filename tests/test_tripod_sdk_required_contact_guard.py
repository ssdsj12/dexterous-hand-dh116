"""Required-pad clearance must use the locked cooked collision hull."""
import importlib
import pytest

def test_required_cooked_hull_and_nonrequired_sources_use_original_floors():
    m=importlib.import_module("scripts.audit_dh116_tripod_sdk_path")
    bounds,floors=m.path_constraints({"source:finger14_link":.1,
                                      "source:finger13_link":.2,
                                      "cooked:finger14_link":.1,
                                      "cooked:finger33_link":.3})
    assert "rod:source:finger14_link" not in floors, "CAD source hull is not the locked cooked pad"
    assert floors["rod:cooked:finger14_link"]==pytest.approx(-.00005)
    assert floors["rod:cooked:finger33_link"]==pytest.approx(-.00005)
    assert floors["rod:source:finger13_link"]==0
    assert floors["table:source:finger14_link"]==pytest.approx(.0021)
    assert len(bounds)==len(floors)==7
