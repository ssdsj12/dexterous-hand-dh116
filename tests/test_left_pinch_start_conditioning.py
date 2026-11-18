import importlib
import numpy as np
import pytest

@pytest.fixture(scope="module")
def m():return importlib.import_module("scripts.solve_dh116_left_pinch_start_conditioned")

def test_endpoint_only_candidate_is_rejected_at_actual_start(m):
    r,g=m.load_inputs()
    row=g.evaluate(np.asarray(r["search"]["best"]["x"]),[r["source"]["scenarios"][-1]])
    assert row["endpoint"]["q_rad"]==r["all_observed_validation"]["endpoint"]["q_rad"]
    assert min(v for k,v in row["constraints"].items() if k.startswith("start:table:"))<0
    assert row["start_state"]["source_left_was_held"] is True
    assert row["native_ready"] is False
    assert "start:initial_rod:cooked:finger14_link" in row["constraints"]

def test_start_joint_delta_stays_original(m):
    r,g=m.load_inputs();x=np.asarray(r["search"]["best"]["x"]);row=g.evaluate(x,[])
    np.testing.assert_allclose(np.array(row["endpoint"]["q_rad"])-row["start_state"]["commanded"]["q_rad"],
        r["source"]["original_joint_delta_rad"],atol=1e-14)
    assert m.CONFIG["maximum_evaluations"]==600
    assert m.CONFIG["native_execution"] is False
