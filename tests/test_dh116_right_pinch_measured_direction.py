import numpy as np
import pytest

def module():
    from scripts import audit_dh116_right_pinch_measured_direction as m
    return m

def test_actual_right_source_keeps_sdk234_identity_and_zeros_wrist():
    m=module();a=m.load_anchor()
    assert a["side"]=="right" and a["identity"]["physics_step"]==234
    assert a["q"].shape==(6,) and len(a["actual"])==23
    assert np.array_equal(a["issued_action24"][6:12],np.zeros(6))
    assert a["native_steps"]==0

@pytest.mark.parametrize("side",["left","both","unknown"])
def test_independent_right_source_required(side):
    with pytest.raises(ValueError):module().load_anchor(side=side)

def test_finite_difference_solves_original_bounded_two_finger_direction():
    m=module();calls=[]
    def measured(x):
        calls.append(x.copy());return np.array([.002,.003])+np.diag([-.04,-.05])@x
    r=m.direction(measured,np.array([.1,.1]))
    np.testing.assert_allclose(r["delta_rad"],[.05375,.063],atol=1e-12)
    assert len(calls)==9 and r["jacobian_relative_error"]<1e-10

@pytest.mark.parametrize("bad",[np.array([1.,.1]),np.array([-.1,.1]),np.array([np.nan,.1])])
def test_direction_rejects_budget_expansion(bad):
    with pytest.raises(ValueError):module().direction(lambda x:np.array([.001,.001])-x,bad)

def test_excess_motion_refused_instead_of_clipping():
    with pytest.raises(ValueError,match="direction"):
        module().direction(lambda x:np.array([1.,1.])-x,np.array([.1,.1]))

def test_missing_node_support_cannot_be_replaced_by_zero():
    with pytest.raises(ValueError):module().direction(lambda x:np.array([np.nan,.002]),np.array([.1,.1]))

def test_path_gate_requires_all_physical_geometry_conditions():
    m=module()
    assert m.gates(.0022,.0001,-.0002,-.0002,-.8)["static_path_passed"]
    for args in [(0.0020,.0001,-.0002,-.0002,-.8),(.0022,-.00001,-.0002,-.0002,-.8),(.0022,.0001,-.0004,-.0002,-.8),(.0022,.0001,-.0002,.001,-.8),(.0022,.0001,-.0002,-.0002,.8)]:
        r=m.gates(*args)
        assert not r["static_path_passed"] and not r["native_ready"]


def test_source_fk_only_controls_existing_collision_links_not_mount_scaffolding():
    m=module();old={"finger14_link":np.eye(4)};new={"finger14_link":np.eye(4)}
    actual={"finger14_link":np.eye(4),"wrist_anchor":np.eye(4)}
    out=m.predicted_poses(actual,old,new,["finger14_link"])
    assert set(out)=={"finger14_link"}
    with pytest.raises(ValueError):m.predicted_poses(actual,old,new,["finger23_link"])

def test_refinement_is_bounded_by_original_total_motion_not_new_budget():
    m=module();start=np.array([.03,.02])
    f=lambda x:np.array([.002,.003])+np.diag([-.04,-.05])@x
    r=m.refine_direction(f,start,np.array([.1,.1]))
    np.testing.assert_allclose(r["delta_rad"],[.05375,.063])
    assert r["parent_delta_rad"]==[.03,.02]
    with pytest.raises(ValueError):m.refine_direction(f,np.array([.1,.02]),np.array([.1,.1]))

def test_diagnostic_path_choice_does_not_turn_missing_contact_into_success():
    rows=[dict(delta_rad=[x,x],pads={n:dict(minimum_gap_m=v) for n in ("finger14_link","finger23_link")})
          for x,v in ((.02,.002),(.03,.001),(.04,.0004))]
    m=module();choice=m.failed_path_choice(rows)
    assert choice["delta_rad"]==[.04,.04]
    assert choice["contact_margin_passed"] is False
