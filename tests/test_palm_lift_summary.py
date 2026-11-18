import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_palm_lift_summary import summarize_palm_lift


def test_left_and_right_maxima_can_belong_to_different_environments():
    peak=torch.tensor([[-.1,.02],[.07,.01],[.03,.04],[.02,.09]],dtype=torch.float64)
    last=torch.tensor([[.01,-.01],[.02,.02],[.03,.03],[.04,.04]],dtype=torch.float64)
    valid=torch.tensor([[True,True],[True,False],[False,True],[True,True]])
    result=summarize_palm_lift(peak,last,valid)
    assert result==pytest.approx(dict(left_palm_lift_max_m=.07,right_palm_lift_max_m=.09,
        left_palm_lift_last_mean_m=.07/3,right_palm_lift_last_mean_m=.02,
        left_palm_lift_max_env_id=1,right_palm_lift_max_env_id=3,
        left_palm_lift_valid_envs=3,right_palm_lift_valid_envs=3))
    assert type(result["left_palm_lift_max_env_id"]) is int
    assert type(result["right_palm_lift_valid_envs"]) is int


def test_negative_only_lift_is_not_clamped_to_zero():
    peak=torch.tensor([[-.2,-.1],[-.3,-.4],[-.1,-.2]])
    last=peak-.05
    result=summarize_palm_lift(peak,last,torch.ones(3,2,dtype=torch.bool))
    assert result["left_palm_lift_max_m"]==pytest.approx(-.1)
    assert result["right_palm_lift_max_m"]==pytest.approx(-.1)
    assert result["left_palm_lift_last_mean_m"]==pytest.approx(-.25)
    assert result["right_palm_lift_last_mean_m"]==pytest.approx(-.85/3)
    assert result["left_palm_lift_max_env_id"]==2
    assert result["right_palm_lift_max_env_id"]==0


def test_all_invalid_is_unknown_and_invalid_nonfinite_values_are_allowed():
    peak=torch.tensor([[float("nan"),float("-inf")],[float("inf"),float("nan")]])
    result=summarize_palm_lift(peak,peak.clone(),torch.zeros(2,2,dtype=torch.bool))
    for hand in ("left","right"):
        assert result[f"{hand}_palm_lift_max_m"] is None
        assert result[f"{hand}_palm_lift_last_mean_m"] is None
        assert result[f"{hand}_palm_lift_max_env_id"] is None
        assert result[f"{hand}_palm_lift_valid_envs"]==0


def test_group_range_uses_global_environment_ids_and_only_selected_rows():
    peak=torch.tensor([[10.,10.],[9.,9.],[.01,.02],[.08,.03],[.04,.09],[8.,8.]])
    last=torch.tensor([[10.,10.],[9.,9.],[.01,.02],[.02,.03],[.03,.04],[8.,8.]])
    valid=torch.ones(6,2,dtype=torch.bool)
    # A nonfinite measurement outside the requested group does not affect it.
    peak[0,0]=float("nan")
    result=summarize_palm_lift(peak,last,valid,start=2,stop=5)
    assert result["left_palm_lift_max_env_id"]==3
    assert result["right_palm_lift_max_env_id"]==4
    assert result["left_palm_lift_max_m"]==pytest.approx(.08)
    assert result["right_palm_lift_max_m"]==pytest.approx(.09)
    assert result["left_palm_lift_last_mean_m"]==pytest.approx(.02)
    assert result["right_palm_lift_last_mean_m"]==pytest.approx(.03)
    assert result["left_palm_lift_valid_envs"]==3


def test_each_hand_validity_is_independent_and_padding_does_not_contribute():
    peak=torch.tensor([[float("nan"),.02],[float("-inf"),.04]])
    last=torch.tensor([[float("nan"),-.01],[float("inf"),.03]])
    valid=torch.tensor([[False,True],[False,True]])
    result=summarize_palm_lift(peak,last,valid)
    assert result["left_palm_lift_max_m"] is None
    assert result["left_palm_lift_last_mean_m"] is None
    assert result["left_palm_lift_max_env_id"] is None
    assert result["left_palm_lift_valid_envs"]==0
    assert result["right_palm_lift_max_m"]==pytest.approx(.04)
    assert result["right_palm_lift_last_mean_m"]==pytest.approx(.01)
    assert result["right_palm_lift_max_env_id"]==1
    assert result["right_palm_lift_valid_envs"]==2


@pytest.mark.parametrize("name",["peak","last"])
@pytest.mark.parametrize("bad",[float("nan"),float("inf"),float("-inf")])
def test_nonfinite_valid_measurement_is_rejected(name,bad):
    inputs={"peak":torch.zeros(2,2),"last":torch.zeros(2,2),"valid":torch.ones(2,2,dtype=torch.bool)}
    inputs[name][1,0]=bad
    with pytest.raises(ValueError,match="finite"):
        summarize_palm_lift(**inputs)


@pytest.mark.parametrize("name,shape",[("peak",(4,)),("peak",(2,3)),
    ("last",(3,2)),("valid",(2,1)),("valid",(2,2,1))])
def test_incompatible_shapes_are_rejected(name,shape):
    inputs={"peak":torch.zeros(2,2),"last":torch.zeros(2,2),"valid":torch.ones(2,2,dtype=torch.bool)}
    inputs[name]=torch.zeros(shape,dtype=torch.bool if name=="valid" else torch.float32)
    with pytest.raises(ValueError,match="shape"):
        summarize_palm_lift(**inputs)


@pytest.mark.parametrize("start,stop",[(-1,None),(0,3),(2,1),(0,-1),(0.0,None),(False,None),(0,True)])
def test_invalid_ranges_are_rejected(start,stop):
    with pytest.raises(ValueError,match="range"):
        summarize_palm_lift(torch.zeros(2,2),torch.zeros(2,2),
            torch.ones(2,2,dtype=torch.bool),start=start,stop=stop)


def test_empty_range_reports_unknown():
    result=summarize_palm_lift(torch.zeros(2,2),torch.zeros(2,2),
        torch.ones(2,2,dtype=torch.bool),start=2)
    assert result["left_palm_lift_max_m"] is None
    assert result["right_palm_lift_valid_envs"]==0


def test_inputs_and_rng_are_unchanged_and_equal_maxima_choose_first_global_id():
    peak=torch.tensor([[.01,.04],[.03,.04],[.03,.02]])
    last=peak/2;valid=torch.ones(3,2,dtype=torch.bool)
    before=[t.clone() for t in (peak,last,valid)]
    rng=torch.get_rng_state().clone()
    result=summarize_palm_lift(peak,last,valid,start=1)
    for actual,expected in zip((peak,last,valid),before): assert torch.equal(actual,expected)
    assert torch.equal(torch.get_rng_state(),rng)
    assert result["left_palm_lift_max_env_id"]==1
    assert result["right_palm_lift_max_env_id"]==1


def test_measurements_must_be_floating_tensors_and_validity_must_be_boolean():
    peak=torch.zeros(2,2);valid=torch.ones(2,2,dtype=torch.bool)
    for args in ((peak.long(),peak,valid),(peak,peak,valid.float()),
            (peak.tolist(),peak,valid)):
        with pytest.raises(ValueError):summarize_palm_lift(*args)
