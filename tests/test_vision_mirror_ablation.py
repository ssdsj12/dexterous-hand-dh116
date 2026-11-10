import torch
import pytest

from go2_pvcnn.control.dh116.tabletop_vision_mirror_ablation import (
    MIRROR_MODES,
    evaluate_mirror_outputs,
    m2_action_to_physics,
    m2_observation_for_actor,
    mirror_action24,
    mirror_observation,
    mirror_observation_roundtrip_metrics,
    prepare_mirror_pair,
)
from go2_pvcnn.control.dh116.tabletop_vision_student import POLICY_DIM


def _observations(n=2):
    generator = torch.Generator().manual_seed(203)
    value = torch.randn(n, POLICY_DIM, generator=generator)
    frames = value.reshape(n, 4, 961)
    frames[..., 768:896] = 1.
    for start in (923, 930):
        q = frames[..., start:start + 4]
        q /= q.norm(dim=-1, keepdim=True)
    return value


def test_mirror_action24_is_an_involution_with_explicit_sign_contract():
    action = torch.arange(48, dtype=torch.float32).reshape(2, 24) / 10.
    mirrored = mirror_action24(action)
    assert torch.equal(mirror_action24(mirrored), action)
    assert torch.equal(mirrored[:, 0:6], action[:, 6:12])
    assert torch.equal(mirrored[:, 6:12], action[:, 0:6])
    assert torch.equal(mirrored[:, 12], -action[:, 15])
    assert torch.equal(mirrored[:, 15], -action[:, 12])
    assert torch.equal(mirrored[:, 18], action[:, 21])
    assert torch.equal(mirrored[:, 19], -action[:, 22])
    assert torch.equal(mirrored[:, 20], -action[:, 23])


def test_mirror_observation_is_an_involution_and_preserves_contract():
    observation = _observations()
    mirrored = mirror_observation(observation)
    assert mirrored.shape == observation.shape == (2, POLICY_DIM)
    assert torch.isfinite(mirrored).all()
    torch.testing.assert_close(mirror_observation(mirrored), observation, atol=2e-6, rtol=0.)
    metrics = mirror_observation_roundtrip_metrics(observation)
    assert all(value["max_abs"] <= 2e-6 for value in metrics.values())


def test_prepare_pair_has_explicit_m0_m1_m2_modes():
    observation = _observations()
    for mode in MIRROR_MODES:
        original, paired, metadata = prepare_mirror_pair(observation, mode)
        assert original.shape == paired.shape == observation.shape
        assert metadata["mode"] == mode
        assert metadata["offline_only"] is (mode == "M2_full_mirror")


def test_m2_native_wrapper_has_explicit_observation_and_action_inverse():
    observation = _observations()
    actor_input = m2_observation_for_actor(observation)
    assert actor_input.shape == observation.shape
    torch.testing.assert_close(
        actor_input,
        prepare_mirror_pair(observation, "M1_quaternion")[1],
        atol=2e-6, rtol=0.,
    )
    action = torch.arange(48, dtype=torch.float32).reshape(2, 24) / 100.
    torch.testing.assert_close(m2_action_to_physics(m2_action_to_physics(action)), action)


def test_mirror_output_metrics_report_l2_direction_and_per_dimension_error():
    expected = torch.zeros(1, 24)
    actual = torch.zeros(1, 24)
    expected[0, :4] = torch.tensor([1., -1., 0., 0.])
    actual[0, :4] = torch.tensor([1., 1., 0.5, 0.])
    metrics = evaluate_mirror_outputs(actual, expected)
    assert metrics["sample_count"] == 1
    assert metrics["action_dim"] == 24
    assert metrics["l2_mean"] == torch.linalg.vector_norm(actual - expected, dim=-1).item()
    assert metrics["per_dim_abs_mean"][:4] == [0., 2., .5, 0.]
    assert metrics["direction_consistency_rate"] == pytest.approx(22. / 24.)
