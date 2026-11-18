"""CPU geometry-reference tests; simulated pressure is not a force predictor."""
import importlib

import pytest
import torch


def pressure(**kwargs):
    module = importlib.import_module("go2_pvcnn.control.dh116.tabletop_teacher_pressure_reference")
    return module.pressure_reference_targets(**kwargs)


def inputs(n=2):
    points = torch.full((n, 2, 5, 3), .2, dtype=torch.float64)
    vectors = torch.zeros_like(points)
    vectors[..., 0] = 2.
    return dict(points=points, vectors_on_hand=vectors,
        loads=torch.ones(n, 2, 5, dtype=torch.float64),
        valid=torch.ones(n, 2, 5, dtype=torch.bool),
        target_loads=torch.full((n, 2, 5), 1.5, dtype=torch.float64), horizon=4)


def test_measured_force_on_hand_sets_inward_sign_and_horizon_units():
    data = inputs()
    result = pressure(**data)
    assert result["target_points"].shape == (2, 4, 2, 5, 3)
    expected = data["points"][:, None].repeat(1, 4, 1, 1, 1)
    expected[..., 0] -= torch.arange(1, 5, dtype=torch.float64)[None, :, None, None] * (.0004 * .5 / 60)
    torch.testing.assert_close(result["target_points"], expected, rtol=0, atol=1e-16)
    assert result["weights"].shape == (2, 4, 2, 5)
    assert bool((result["weights"] == 1).all())
    assert result["eligible"].all() and result["validity"].all()
    assert result["mode"] == "geometry_velocity_from_simulated_force"
    assert result["force_prediction"] is False


def test_overload_retracts_and_each_hand_environment_remains_independent():
    data = inputs()
    data["loads"][0, 0, 0] = 2.5
    data["vectors_on_hand"][1, 1, 0, 0] = -2.
    result = pressure(**data)
    delta = result["target_points"][:, 0] - data["points"]
    assert delta[0, 0, 0, 0] > 0  # overload follows outward force-on-hand
    assert delta[0, 1, 0, 0] < 0
    assert delta[1, 0, 0, 0] < 0
    assert delta[1, 1, 0, 0] > 0  # opposite hand has opposite measured normal
    assert bool((delta[..., 1:] == 0).all())


def test_speed_is_capped_in_metres_per_second_before_horizon_integration():
    data = inputs()
    data["target_loads"][0, 0, 0] = 1000.
    data["loads"][0, 0, 1] = 1000.
    result = pressure(**data, dt=.02)
    assert result["speed_m_s"][0, 0, 0] == .001
    assert result["speed_m_s"][0, 0, 1] == -.001
    displacement = (result["target_points"] - data["points"][:, None]).norm(dim=-1)
    assert float(displacement.max()) <= 4 * .02 * .001 + 1e-16


def test_missing_zero_and_disabled_pads_never_create_contact_acquisition_targets():
    data = inputs()
    data["loads"][0, 0, :2] = torch.tensor([0., .05], dtype=data["loads"].dtype)
    data["valid"][0, 0, 2] = False
    data["vectors_on_hand"][0, 0, 3] = 0.
    data["target_loads"][0, 0, 4] = 0.
    result = pressure(**data)
    assert not result["eligible"][0, 0].any()
    assert not result["weights"][0, :, 0].any()
    assert not result["speed_m_s"][0, 0].any()
    torch.testing.assert_close(result["target_points"][0, :, 0],
        data["points"][0, 0].expand(4, -1, -1), rtol=0, atol=0)
    assert result["eligible"][0, 1].all() and result["eligible"][1].all()
    assert bool((result["reason_code"][0, 0] != result["reason_codes"]["eligible"]).all())


@pytest.mark.parametrize("field", ["points", "vectors_on_hand", "loads", "target_loads", "effective_depth_m"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_data_only_masks_the_affected_pad(field, bad):
    data = inputs()
    data["effective_depth_m"] = torch.zeros_like(data["loads"])
    data[field][0, 0, 0] = bad
    result = pressure(**data)
    assert not result["validity"][0, 0, 0]
    assert not result["eligible"][0, 0, 0]
    assert not result["weights"][0, :, 0, 0].any()
    assert result["eligible"].sum() == 19
    assert torch.isfinite(result["target_points"]).all()
    assert torch.isfinite(result["speed_m_s"]).all()
    assert result["reason_code"][0, 0, 0] == result["reason_codes"]["nonfinite_measurement"]


def test_actual_depth_override_retreats_only_above_threshold_and_only_on_bearing_pads():
    data = inputs()
    depth = torch.zeros_like(data["loads"])
    depth[0, 0, :3] = torch.tensor([.0006, .000601, .001], dtype=depth.dtype)
    data["loads"][0, 0, 2] = 0.
    result = pressure(**data, effective_depth_m=depth)
    assert result["speed_m_s"][0, 0, 0] > 0
    assert result["speed_m_s"][0, 0, 1] == -.001
    assert result["speed_m_s"][0, 0, 2] == 0
    assert result["target_points"][0, 0, 0, 1, 0] > data["points"][0, 0, 1, 0]


def test_input_tensors_and_output_storage_are_independent():
    data = inputs()
    data["effective_depth_m"] = torch.zeros_like(data["loads"])
    before = {k: v.clone() for k, v in data.items() if isinstance(v, torch.Tensor)}
    result = pressure(**data)
    for key, original in before.items():
        torch.testing.assert_close(data[key], original, rtol=0, atol=0)
    result["target_points"].zero_()
    result["eligible"].zero_()
    for key, original in before.items():
        torch.testing.assert_close(data[key], original, rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [dict(horizon=0), dict(horizon=True), dict(horizon=1.2),
    dict(dt=0.), dict(dt=float("nan")), dict(dt=True), dict(dt=1e308), dict(gain_m_per_n_s=-1.),
    dict(max_speed_m_s=0.), dict(retreat_depth_m=-.001)])
def test_invalid_configuration_is_rejected(kwargs):
    data = inputs()
    data.update(kwargs)
    with pytest.raises(ValueError):
        pressure(**data)


def test_shape_dtype_and_boolean_validity_contracts_are_explicit():
    for field, replacement in [("points", torch.zeros(2, 2, 4, 3)),
            ("loads", torch.zeros(2, 2, 4)), ("valid", torch.ones(2, 2, 5)),
            ("points", torch.zeros(2, 2, 5, 3, dtype=torch.int64))]:
        data = inputs()
        data[field] = replacement
        with pytest.raises(ValueError):
            pressure(**data)


def test_negative_measurements_are_masked_without_affecting_other_pads():
    data = inputs()
    data["loads"][0, 0, 0] = -1.
    data["target_loads"][0, 0, 1] = -1.
    data["effective_depth_m"] = torch.zeros_like(data["loads"])
    data["effective_depth_m"][0, 0, 2] = -.001
    result = pressure(**data)
    assert not result["validity"][0, 0, :3].any()
    assert result["eligible"].sum() == 17


def test_float32_horizon_overflow_is_a_configuration_error():
    data = inputs()
    data = {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
        for k, v in data.items()}
    with pytest.raises(ValueError):
        pressure(**data, dt=1e39)
