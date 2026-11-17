"""Same-clock geometry attribution is not a native policy counterfactual."""
import importlib
from copy import deepcopy
import numpy as np
import pytest

def api():
    return importlib.import_module("scripts.audit_dh116_projected_pinch_tracking")

def test_sdk_targets_follow_actual_dof_names():
    m=api();names=["extra",*reversed(m.ACTIVE)]
    row={"joint_names":names}
    target={"joint_indices":list(range(1,7)),"values":np.arange(6,dtype=float)[None]}
    np.testing.assert_array_equal(m.named_sdk_active(row,target),np.arange(6)[::-1])
    bad=deepcopy(target);bad["joint_indices"][0]=bad["joint_indices"][1]
    with pytest.raises(ValueError):m.named_sdk_active(row,bad)
    bad=deepcopy(target);bad["values"][0,0]=np.nan
    with pytest.raises(ValueError):m.named_sdk_active(row,bad)

def test_decomposition_is_ordered_not_causal():
    m=api()
    v={k:{"ellipse_value":n} for n,k in enumerate(m.STAGES)}
    r=m.ellipse_differences(v)
    assert r["total"]==4.
    assert sum(r["ordered_deltas"].values())==4.
    assert r["causal_effect_established"] is False
    v[m.STAGES[2]]["ellipse_value"]=float("nan")
    with pytest.raises(ValueError):m.ellipse_differences(v)

@pytest.fixture(scope="module")
def packet():
    return api().load_source()

def test_source_has_real_full_same_episode_and_source_sha(packet):
    assert len(packet["rows"])==232
    assert packet["receipt"]["native_qualification"] is True
    assert packet["source"]["physical_sha256"]=="9cec6ed43c06f23cf23b327a22009c7f1b3a98574de27883ba89c60b0bf255d4"

def test_complete_native_endpoint_separates_sdk_tracking_and_rod(packet):
    m=api();g=m.Geometry()
    r=g.compare(packet["rows"][-1],packet["plan"])
    assert r["identity"]["physics_step"]==234
    p=r["pads"]["left.finger23_link"]
    assert p["stages"]["planned_nominal"]["ellipse_value"]<1
    assert p["stages"]["measured_links_actual_rod"]["ellipse_value"]>1.12
    assert p["actual_region_pairs"]>0 and p["actual_original_region_accepted"]==0
    assert abs(p["ordered_geometry_difference"]["total"]-sum(
        p["ordered_geometry_difference"]["ordered_deltas"].values()))<1e-12
    assert abs(r["hands"]["left"]["root_actual_minus_sdk_mm"][2])<.01
    assert max(abs(v) for v in r["hands"]["left"]["active_tracking_error_rad"])>.001
    assert not r["teacher_admission"] and not r["native_counterfactual"]

@pytest.mark.parametrize("bad",["joint_clock","pose_shape","nan_rod","terminal"])
def test_uncorrelated_measurements_rejected(packet,bad):
    m=api();x=deepcopy(packet["rows"][-1])
    if bad=="joint_clock":x["closure_geometry"]["raw"]["joint_readback"]["identity"][0]["physics_step"]-=1
    if bad=="pose_shape":x["closure_geometry"]["raw"]["hands"]["left"]["link_poses_xyzw"]=np.zeros((1,1,7))
    if bad=="nan_rod":x["closure_geometry"]["raw"]["rod_nodes_m"][0,0]=np.nan
    if bad=="terminal":x["guard_failure"]="joint_limit"
    with pytest.raises(ValueError):m.validate_boundary(x)

def test_receipt_wrong_hash_not_repaired():
    with pytest.raises(ValueError):api().load_source(expected_sha="0"*64)
