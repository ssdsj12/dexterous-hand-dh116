import json
import numpy as np
import pytest
from scripts import audit_dh116_tripod_two_phase_sdf as m


def parent():
    return dict(side="left",independent_style="tripod",independent_seed_sha256=m.TRIPOD_SHA,
        base_pose_wxyz=[.5,.2,.4,1.,0.,0.,0.],active_q_rad=[.3,.5,.7,.8,.7,.95])


def test_two_phase_preserves_independent_root_and_nonclosing_joints():
    p=parent();r=m.command(p,[.01,.02,.03])
    assert r["base_pose_wxyz"]==p["base_pose_wxyz"]
    np.testing.assert_allclose(r["initial_q_rad"],[.3,.47,.67,.77,.7,.95])
    np.testing.assert_allclose(r["endpoint_q_rad"],[.3,.51,.72,.83,.7,.95])
    assert max(r["active_velocity_rad_s"])<=.3
    assert r["quiet_steps"]==48 and r["closure_steps"]==160
    assert p==parent() and not r["reset_verified"]


@pytest.mark.parametrize("field,value",[("independent_style","pinch"),("independent_seed_sha256","0"*64),("side","right")])
def test_wrong_source_or_side_rejected(field,value):
    p=parent();p[field]=value
    with pytest.raises(ValueError):m.command(p,[.01]*3)


@pytest.mark.parametrize("delta",[[.071,0,0],[-.001,0,0],[np.nan,0,0],[0,0]])
def test_nonlegal_step_rejected(delta):
    with pytest.raises(ValueError):m.command(parent(),delta)


def test_two_scale_prediction_is_one_bounded_direction_only():
    calls=[]
    def query(d):
        calls.append(d.copy());return np.array([.001,.002,.003])-np.array([.05,.1,.15])*d
    r=m.direction(query)
    assert len(calls)==13
    np.testing.assert_allclose(r["proposed_extra_rad"],[.018,.019,.01933333333333333])
    assert r["jacobian_relative_difference"]<1e-10


def test_unstable_derivative_refused():
    def query(d):return .001+1e6*d**3
    with pytest.raises(ValueError,match="unstable"):m.direction(query)


def test_flat_or_wrong_direction_refused():
    with pytest.raises(ValueError,match="decreasing"):m.direction(lambda d:np.ones(3)*.001)


def test_path_lower_bound_uses_half_interval_cover_and_original_floor():
    r=m.path_summary({"source:a":[.0022]*65},{"source:a":[.0002]*65},{"source:a":.0128})
    assert r["minimum_table_lower_m"]==pytest.approx(.0021)
    assert r["minimum_nonrequired_lower_m"]==pytest.approx(.0001)
    assert m.RUNTIME_FLOOR==.0021


def test_missing_nonuniform_path_rejected():
    with pytest.raises(ValueError):m.path_summary({"x":[1.]*64},{"x":[1.]*65},{"x":0.})


def test_output_exclusive(tmp_path):
    with pytest.raises(FileExistsError):m.main(["--output",str(tmp_path)])


def test_partial_timeout_saves_no_qualification(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"run",lambda *a:(_ for _ in ()).throw(m.BudgetExpired()))
    out=tmp_path/"partial";assert m.main(["--output",str(out)])==0
    r=json.loads((out/"result.json").read_text())
    assert r["status"]=="partial_timeout" and not r["native_ready"]
    assert r["physics_steps"]==r["training_updates"]==0


def test_positive_table_does_not_hide_proximal_collision():
    r=m.path_summary({"a":[.003]*65},{"a":[-.0001]*65},{"a":0.})
    assert r["table_2p1mm_passed"] and not r["nonrequired_settled_axis_clear"]


def test_coupled_limit_violation_refused():
    p=parent();p["active_q_rad"][1]=.01
    with pytest.raises(ValueError):m.command(p,[0,0,0])


def test_nonfinite_sdf_refused_before_prediction():
    with pytest.raises(ValueError):m.direction(lambda d:[None,.001,.001])


def test_parent_result_identity_is_mandatory(tmp_path):
    p=tmp_path/"tampered.json";p.write_text("{}")
    with pytest.raises(ValueError):m.load_parent(p)
