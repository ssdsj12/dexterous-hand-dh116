"""Actual immutable CPU replay evidence; never substitute for native runs."""
import hashlib
import json
from pathlib import Path
import pytest

PATH=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/tripod-vertical-sdf-20261114-run01/result.json")
SHA="d0040fc6f282687b16bbc1336f7cbfdabc54bac52fc6b696facdf720e6ec4ba1"

@pytest.fixture
def result():
    if not PATH.exists():pytest.skip("private actual evidence is not installed on this host")
    assert hashlib.sha256(PATH.read_bytes()).hexdigest()==SHA
    return json.loads(PATH.read_text())

def test_complete_205_node_records_and_48_piecewise_polynomials(result):
    r=result["continuous_reference"]
    assert result["status"]=="complete"
    assert len(r["nodes"])==205
    assert len({(x["pad"],x["node_id"]) for x in r["nodes"]})==205
    assert sum(len(x["segments"]) for x in r["nodes"])==48
    assert r["counts"]=={"outside_sampler_lattice":194,"reference_clear_with_guard":11}

def test_three_actual_pad_minima_and_old_geometric_margins(result):
    r=result["continuous_reference"]["pads"]
    for key,value in {"finger14_link":.0017706380354958716,
                      "finger23_link":.0017478742061911154,
                      "finger33_link":.0026155552494421617}.items():
        assert r[key]["minimum_reference_gap_m"]==pytest.approx(value,abs=1e-13)
    g=result["geometry_recheck"]
    assert min(g["table_gap_by_shape_m"].values())==pytest.approx(.00240361967582986)
    assert min(g["all_shape_rod_envelope_lower_m"].values())==pytest.approx(.00047128339696340195)

def test_reference_clear_does_not_grant_physics_reset_or_teacher(result):
    assert result["continuous_reference"]["reference_no_contact"] is True
    assert result["continuous_reference"]["free_rod_trajectory_verified"] is False
    assert result["continuous_reference"]["gpu_bitwise_verified"] is False
    assert result["native_ready"] is False
    assert result["teacher_admitted"] is False
    assert result["reset_verified"] is False
    assert result["physics_steps"]==result["ppo_updates"]==0
