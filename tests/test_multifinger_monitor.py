"""Pure Torch contracts for multifinger support and hand-relative slip (no Isaac)."""

import importlib
import itertools
import math

import pytest
import torch


def api():
    return importlib.import_module(
        "go2_pvcnn.control.dh116.tabletop_multifinger_monitor"
    )


def poses(n=1, dtype=torch.float64, device="cpu"):
    obj = torch.zeros((n, 7), dtype=dtype, device=device)
    obj[:, 3] = 1
    return obj, obj[:, None].expand(-1, 2, -1).clone()


def samples(n=1, fingers=((0, 1), (0, 1)), dtype=torch.float64, device="cpu"):
    obj, hands = poses(n, dtype, device)
    points = api().rod_points_in_hands(obj, hands)
    loads = torch.zeros((n, 2, 5), dtype=dtype, device=device)
    normals = torch.zeros((n, 2, 5, 3), dtype=dtype, device=device)
    for hand, digits in enumerate(fingers):
        loads[:, hand, list(digits)] = 1
        normals[:, hand, list(digits), 0] = 1
        normals[:, hand, digits[-1], 0] = -1
    return loads, points, normals


def step(monitor, data, dt=0.1, valid=None, reset=None):
    loads, points, normals = data
    n, device = loads.shape[0], loads.device
    if valid is None:
        valid = torch.ones(n, dtype=torch.bool, device=device)
    if reset is None:
        reset = torch.zeros(n, dtype=torch.bool, device=device)
    return monitor.update(loads, points, normals, dt, valid, reset)


def fill(monitor, data, steps=4, **kwargs):
    for _ in range(steps):
        result = step(monitor, data, **kwargs)
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rod_three_points_wxyz_and_independent_hand_translation(dtype):
    obj, hands = poses(2, dtype)
    hands[:, 1, 0] = 0.03
    actual = api().rod_points_in_hands(obj, hands)
    local = torch.tensor([[0, -0.2, 0], [0, 0.2, 0], [0.008, 0, 0]], dtype=dtype)
    assert actual.shape == (2, 2, 3, 3)
    assert actual.dtype == dtype and actual.device == obj.device
    torch.testing.assert_close(actual[:, 0], local.expand(2, -1, -1))
    torch.testing.assert_close(actual[:, 1], (local - local.new_tensor([0.03, 0, 0])).expand(2, -1, -1))


def test_common_rigid_rotation_and_translation_do_not_slip():
    obj, hands = poses()
    initial = api().rod_points_in_hands(obj, hands)
    obj[:, :3] = torch.tensor([0.7, -0.3, 1.2])
    obj[:, 3:] = torch.tensor([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    hands[:] = obj[:, None]
    moved = api().rod_points_in_hands(obj, hands)
    torch.testing.assert_close(moved, initial)
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    data = samples()
    fill(monitor, data)
    data[1].copy_(moved)
    result = step(monitor, data)
    assert result["ready"].item()
    assert result["relative_drift_m"].item() < 1e-12


def test_each_hand_inverse_rotation_and_nonunit_quaternion():
    obj, hands = poses()
    hands[:, 1, 3:] = torch.tensor([2 * math.sqrt(0.5), 0, 0, 2 * math.sqrt(0.5)])
    points = api().rod_points_in_hands(obj, hands)
    torch.testing.assert_close(points[0, 1], points.new_tensor([[-0.2, 0, 0], [0.2, 0, 0], [0, -0.008, 0]]), atol=1e-12, rtol=1e-6)
    hands[:, :, 3:] *= -1
    torch.testing.assert_close(api().rod_points_in_hands(obj, hands), points)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_relative_object_rotation_detects_tilt_and_axial_roll(axis):
    obj, hands = poses()
    initial = api().rod_points_in_hands(obj, hands)
    obj[:, 3] = math.cos(0.1)
    obj[:, 4 + axis] = math.sin(0.1)
    moved = api().rod_points_in_hands(obj, hands)
    assert torch.linalg.vector_norm(moved - initial, dim=-1).amax() > 0.0005


def test_default_window_is_144_complete_samples_and_outputs_have_batch_shapes():
    monitor = api().GraspStabilityMonitor(2, "cpu")
    data = samples(2)
    result = fill(monitor, data, steps=143)
    assert not result["ready"].any()
    result = step(monitor, data)
    assert result["ready"].tolist() == [True, True]
    assert result["loaded"].shape == result["force_cv"].shape == (2, 2, 5)
    assert result["loaded"].dtype == torch.bool
    for key in ("relative_drift_m", "slip_rate_m_s"):
        assert result[key].shape == (2,)
        assert result[key].dtype == data[0].dtype
    torch.testing.assert_close(result["force_cv"], torch.zeros_like(data[0]))
    assert torch.equal(monitor.ready, result["ready"])
    assert torch.equal(monitor.loaded, result["loaded"])


PAIRS = list(itertools.combinations(range(5), 2))


@pytest.mark.parametrize("left,right", list(zip(PAIRS, reversed(PAIRS))))
def test_any_opposing_finger_pair_can_support_each_hand(left, right):
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    data = samples(fingers=(left, right))
    result = fill(monitor, data)
    assert result["ready"].item()
    assert torch.equal(result["loaded"], data[0] > 0.05)


@pytest.mark.parametrize("digits", [(0, 2, 4), (1, 2, 3, 4), (0, 1, 2, 3, 4)])
def test_three_to_five_fingers_can_share_load(digits):
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    assert fill(monitor, samples(fingers=(digits, digits)))["ready"].item()


@pytest.mark.parametrize("force", [0.0, 0.05])
def test_zero_force_or_contact_threshold_is_never_ready(force):
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    data = samples()
    data[0].fill_(force)
    result = fill(monitor, data)
    assert not result["ready"].item()
    assert not result["loaded"].any()


def test_one_finger_or_one_hand_cannot_qualify():
    for digits in (((2,), (3, 4)), ((1, 2), ())):
        monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
        data = samples() if not digits[1] else samples(fingers=digits)
        if not digits[1]:
            data[0][:, 1] = 0
        assert not fill(monitor, data)["ready"].item()


def test_contact_must_be_continuously_loaded_throughout_window():
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    data = samples()
    fill(monitor, data)
    data[0][0, 0, 1] = 0
    assert not step(monitor, data)["ready"].item()
    data[0][0, 0, 1] = 1
    assert not fill(monitor, data, steps=3)["ready"].item()
    assert step(monitor, data)["ready"].item()


@pytest.mark.parametrize("forces", [[0.6, 1.4, 0.6, 1.4], [1, 1, 1, 5]])
def test_raw_force_cv_rejects_oscillations_and_impact_average(forces):
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    data = samples()
    for force in forces:
        data[0][data[0] > 0] = force
        result = step(monitor, data)
    expected = torch.tensor(forces, dtype=torch.float64)
    cv = expected.std(correction=0) / expected.mean()
    assert result["loaded"].sum() == 4
    torch.testing.assert_close(result["force_cv"][result["loaded"]], cv.expand(4))
    assert cv > 0.25 and not result["ready"].item()


@pytest.mark.parametrize("pulses", [[0, 6, 0, 6], [0, 0, 0, 6], [0, 0.12, 0, 0.12]])
def test_intermittent_extra_load_cannot_be_hidden_by_two_stable_fingers(pulses):
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    data = samples()  # Both hands retain two constant 1 N fingers throughout.
    for force in pulses:
        data[0][:, :, 2] = force
        data[2][:, :, 2, 0] = 1
        torch.testing.assert_close(data[0][:, 0].sum(-1), data[0][:, 1].sum(-1))
        result = step(monitor, data)
    assert result["loaded"][0, :, :2].all()
    assert not result["loaded"][0, :, 2].any()
    assert result["force_cv"][0, :, 2].gt(0.25).all()
    assert not result["ready"].item()
    data[0][:, :, 2] = 0
    assert fill(monitor, data)["ready"].item()


@pytest.mark.parametrize("amplitude,ready", [(0.25, True), (0.26, False)])
def test_population_cv_threshold(amplitude, ready):
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    data = samples()
    for force in (1 - amplitude, 1 + amplitude) * 2:
        data[0][data[0] > 0] = force
        result = step(monitor, data)
    assert result["ready"].item() == ready


@pytest.mark.parametrize("forces,ready", [([1, 4], False), ([0.5, 1, 1.5], True), ([0.49, 1, 1.51], False)])
def test_participating_fingers_must_share_their_mean(forces, ready):
    digits = tuple(range(len(forces)))
    data = samples(fingers=(digits, digits))
    data[0][:, :, list(digits)] = torch.tensor(forces, dtype=data[0].dtype)
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    assert fill(monitor, data)["ready"].item() == ready


@pytest.mark.parametrize("ratio,ready", [(1.5, True), (1.51, False), (2, False)])
def test_total_hand_load_ratio(ratio, ready):
    data = samples()
    data[0][:, 1] *= ratio
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    assert fill(monitor, data)["ready"].item() == ready


@pytest.mark.parametrize("dot,ready", [(-1, True), (-0.25, True), (-0.24, False), (1, False)])
def test_normals_must_oppose_with_dot_at_most_minus_point_25(dot, ready):
    data = samples()
    data[2][:, :, 0] *= 3
    data[2][:, :, 1] = data[2].new_tensor([dot, math.sqrt(1 - dot * dot), 0]) * 7
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    assert fill(monitor, data)["ready"].item() == ready


def test_opposition_is_required_on_both_hands_at_every_sample_and_only_loaded_digits():
    data = samples()
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    fill(monitor, data)
    data[2][0, 1, 1] = data[2].new_tensor([1, 0, 0])
    data[2][0, 1, 4] = data[2].new_tensor([-1, 0, 0])  # Unloaded cannot oppose.
    assert not step(monitor, data)["ready"].item()
    data[2][0, 1, 1, 0] = -1
    assert not fill(monitor, data, steps=3)["ready"].item()
    assert step(monitor, data)["ready"].item()


def test_zero_loaded_normal_cannot_supply_opposition():
    data = samples()
    data[2][0, 0, 1] = 0
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    assert not fill(monitor, data)["ready"].item()


@pytest.mark.parametrize("hand", [0, 1])
def test_single_hand_relative_displacement_prevents_ready(hand):
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    data = samples()
    fill(monitor, data)
    data[1][:, hand, :, 0] += 0.0006
    result = step(monitor, data, dt=1)
    assert result["relative_drift_m"].item() == pytest.approx(0.0006)
    assert not result["ready"].item()


def test_window_captures_excursion_and_fast_motion_that_returns_to_start():
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    data = samples()
    origin = data[1].clone()
    for shift in (0, 0.0004, -0.0004, 0):
        data[1].copy_(origin)
        data[1][:, 1, :, 0] += shift
        result = step(monitor, data, dt=1)
    assert result["relative_drift_m"].item() == pytest.approx(0.0008)
    assert not result["ready"].item()


def test_peak_slip_rate_uses_each_real_timestep_not_window_average():
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    data = samples()
    step(monitor, data, dt=0.5)
    data[1][:, 0, :, 0] += 0.0002
    step(monitor, data, dt=0.01)
    data[1][:, 0, :, 0] -= 0.0002
    step(monitor, data, dt=0.5)
    result = step(monitor, data, dt=0.5)
    assert result["relative_drift_m"].item() < 0.0005
    assert result["slip_rate_m_s"].item() == pytest.approx(0.02)
    assert not result["ready"].item()


def test_slow_small_motion_can_be_ready_and_ring_expires_old_bad_samples():
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    data = samples()
    for _ in range(4):
        data[1][:, 0, :, 0] += 0.0001
        result = step(monitor, data, dt=0.2)
    assert result["ready"].item()
    assert result["relative_drift_m"].item() == pytest.approx(0.0003)
    assert result["slip_rate_m_s"].item() == pytest.approx(0.0005)
    data[1][:, 0, :, 0] += 0.01
    assert not step(monitor, data)["ready"].item()
    assert fill(monitor, data)["ready"].item()


@pytest.mark.parametrize("fault", ["valid", "reset", "loads_nan", "points_inf", "normals_nan", "negative_load"])
def test_bad_rows_and_partial_reset_clear_only_the_affected_history_and_anchor(fault):
    monitor = api().GraspStabilityMonitor(3, "cpu", window_steps=4)
    data = samples(3)
    assert fill(monitor, data)["ready"].all()
    monitor.latch(torch.arange(3), data[1])
    valid = torch.ones(3, dtype=torch.bool)
    reset = torch.zeros(3, dtype=torch.bool)
    if fault == "valid":
        valid[1] = False
    elif fault == "reset":
        reset[1] = True
    elif fault == "loads_nan":
        data[0][1, 0, 0] = float("nan")
    elif fault == "points_inf":
        data[1][1, 0, 0, 0] = float("inf")
    elif fault == "normals_nan":
        data[2][1, 0, 0, 0] = float("nan")
    else:
        data[0][1, 0, 0] = -0.1
    result = step(monitor, data, valid=valid, reset=reset)
    assert result["ready"].tolist() == [True, False, True]
    assert not result["loaded"][1].any()
    good_data = samples(3)
    error = monitor.anchor_error(good_data[1])
    assert error[[0, 2]].eq(0).all() and torch.isinf(error[1])
    assert not fill(monitor, good_data, steps=3)["ready"][1]
    assert step(monitor, good_data)["ready"].all()


def test_latched_anchor_is_copied_and_does_not_follow_rolling_window():
    monitor = api().GraspStabilityMonitor(2, "cpu", window_steps=4)
    data = samples(2)
    fill(monitor, data)
    monitor.latch(torch.tensor([1]), data[1])
    assert torch.isinf(monitor.anchor_error(data[1])[0])
    for _ in range(12):
        data[1][1, 1, :, 0] += 0.0001
        step(monitor, data, dt=1)
    assert monitor.anchor_error(data[1])[1].item() == pytest.approx(0.0012)
    assert monitor.relative_drift_m[1] < 0.0005
    monitor.latch(torch.tensor([1]), data[1])
    assert monitor.anchor_error(data[1])[1] == 0


@pytest.mark.parametrize("field", ["loads_shape", "points_shape", "normals_shape", "valid_shape", "reset_dtype", "loads_dtype", "mixed_dtype"])
def test_structural_validation_raises_without_mutating_good_history(field):
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    data = samples()
    fill(monitor, data)
    loads, points, normals = data
    valid, reset = torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
    if field == "loads_shape": loads = loads[:, :, :4]
    elif field == "points_shape": points = points[:, :, :2]
    elif field == "normals_shape": normals = normals[..., :2]
    elif field == "valid_shape": valid = valid[:, None]
    elif field == "reset_dtype": reset = reset.float()
    elif field == "loads_dtype": loads = loads.long()
    else: points = points.float()
    with pytest.raises((ValueError, TypeError)):
        monitor.update(loads, points, normals, 0.1, valid, reset)
    assert step(monitor, data)["ready"].all()


@pytest.mark.parametrize("dt", [0, -0.1, float("nan"), float("inf")])
def test_bad_timestep_is_rejected(dt):
    monitor = api().GraspStabilityMonitor(1, "cpu", window_steps=4)
    with pytest.raises((ValueError, TypeError)):
        step(monitor, samples(), dt=dt)


@pytest.mark.parametrize("n,window", [(0, 4), (-1, 4), (1, 1), (1, 0), (1, 2.5)])
def test_bad_monitor_dimensions_are_rejected(n, window):
    with pytest.raises((ValueError, TypeError)):
        api().GraspStabilityMonitor(n, "cpu", window_steps=window)


@pytest.mark.parametrize("fault", ["object_shape", "hand_shape", "integer", "nonfinite", "zero_quaternion"])
def test_pose_validation(fault):
    obj, hands = poses()
    if fault == "object_shape": obj = obj[:, :6]
    elif fault == "hand_shape": hands = hands[:, :1]
    elif fault == "integer": obj = obj.long()
    elif fault == "nonfinite": obj[0, 0] = float("nan")
    else: hands[0, 1, 3:] = 0
    with pytest.raises((ValueError, TypeError)):
        api().rod_points_in_hands(obj, hands)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_monitor_preserves_device_dtype_and_does_not_retain_autograd_graph(device):
    monitor = api().GraspStabilityMonitor(2, device, window_steps=3)
    data = samples(2, dtype=torch.float32, device=device)
    data[0].requires_grad_()
    data[1].requires_grad_()
    result = fill(monitor, data, steps=3)
    assert result["ready"].all()
    monitor.latch(torch.tensor([0, 1], device=device), data[1])
    for value in (*result.values(), monitor.anchor_error(data[1])):
        assert value.device == data[0].device and not value.requires_grad


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("as_mask", [False, True])
def test_explicit_physical_reset_clears_selected_history_anchor_and_public_metrics(device, as_mask):
    monitor = api().GraspStabilityMonitor(3, device, window_steps=4)
    data = samples(3, device=device)
    for k in range(5):  # Reset when the shared ring cursor is not at zero.
        data[0][data[0] > 0] = 1 + 0.1 * (-1) ** k
        data[1][..., 0] += 0.00005
        result = step(monitor, data, dt=0.2)
    assert result["ready"].all()
    for key in ("force_cv", "relative_drift_m", "slip_rate_m_s"):
        assert result[key].abs().sum() > 0
    monitor.latch(torch.arange(3, device=device), data[1])
    data[1][..., 0] += 0.0001
    before_error = monitor.anchor_error(data[1]).clone()
    before = {name: value.clone() for name, value in result.items()}
    ids = torch.tensor([False, True, False] if as_mask else [1], device=device)
    monitor.reset(ids)
    for name, old in before.items():
        now = getattr(monitor, name)
        torch.testing.assert_close(now[[0, 2]], old[[0, 2]])
        assert not now[1].any()
    error = monitor.anchor_error(data[1])
    torch.testing.assert_close(error[[0, 2]], before_error[[0, 2]])
    assert torch.isinf(error[1])
    for _ in range(3):
        result = step(monitor, data, dt=0.2)
        assert result["ready"].tolist() == [True, False, True]
    assert step(monitor, data, dt=0.2)["ready"].all()
    assert torch.isinf(monitor.anchor_error(data[1])[1])  # Refill cannot relatch.


def test_explicit_reset_is_safe_before_first_update_and_for_empty_selection():
    monitor = api().GraspStabilityMonitor(2, "cpu", window_steps=4)
    monitor.reset(torch.tensor([0]))
    monitor.reset([])
    data = samples(2)
    assert torch.isinf(monitor.anchor_error(data[1])).all()
    assert fill(monitor, data)["ready"].all()
    monitor.latch(torch.arange(2), data[1])
    monitor.reset(torch.empty(0, dtype=torch.long))
    assert monitor.ready.all()
    assert monitor.anchor_error(data[1]).eq(0).all()
    monitor.reset(torch.ones(2, dtype=torch.bool))
    assert not monitor.ready.any()
    assert torch.isinf(monitor.anchor_error(data[1])).all()


@pytest.mark.parametrize("ids", [torch.tensor([-1]), torch.tensor([3]), torch.tensor([0.5]), torch.tensor([[1]]), torch.tensor([True])])
def test_explicit_reset_rejects_invalid_selection_without_mutation(ids):
    monitor = api().GraspStabilityMonitor(3, "cpu", window_steps=4)
    data = samples(3)
    fill(monitor, data)
    monitor.latch(torch.arange(3), data[1])
    with pytest.raises((ValueError, TypeError)):
        monitor.reset(ids)
    assert monitor.ready.all()
    assert monitor.anchor_error(data[1]).eq(0).all()
    assert step(monitor, data)["ready"].all()


def test_anchor_error_reduces_euclidean_deviation_over_three_points_and_both_hands():
    monitor = api().GraspStabilityMonitor(2, "cpu", window_steps=4)
    data = samples(2)
    monitor.latch(torch.arange(2), data[1])
    moved = data[1].clone()
    moved[0, 1, 2] += moved.new_tensor([0.0003, 0.0004, 0])
    moved[0, 0, 0, 0] += 0.0002
    moved[1, 0, 1, 2] += 0.0007
    torch.testing.assert_close(monitor.anchor_error(moved), moved.new_tensor([0.0005, 0.0007]))


def test_world_to_hand_uses_supplied_palm_link_origin_and_orientation():
    obj, palm_link_pose_w = poses()
    # Deliberately choose a nonzero body origin and rotation. No COM transform
    # or fixed palm offset is part of this function's geometry contract.
    obj[:, :3] = obj.new_tensor([0.7, -0.2, 0.9])
    palm_link_pose_w[:, :, :3] = obj.new_tensor([0.4, -0.1, 0.5])
    palm_link_pose_w[:, :, 3:] = obj.new_tensor([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    points = api().rod_points_in_hands(obj, palm_link_pose_w)
    expected = obj.new_tensor([[-0.3, -0.3, 0.4], [0.1, -0.3, 0.4], [-0.1, -0.308, 0.4]])
    torch.testing.assert_close(points, expected[None, None].expand(1, 2, 3, 3))
