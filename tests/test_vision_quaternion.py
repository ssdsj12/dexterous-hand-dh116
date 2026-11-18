"""Opt-in quaternion representation invariance without changing the sensor contract."""
import importlib

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_vision_student import VisionStudentActor, FRAME_DIM, POLICY_DIM
from go2_pvcnn.control.dh116.tabletop_vision_squashed_policy import UnboundedVisionStudentActor


MODE = 'wrist_quaternion_hemisphere_v1'


def helper():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_quaternion')


def observations(dtype=torch.float32):
    generator = torch.Generator().manual_seed(71)
    value = torch.randn(3, 3844, generator=generator, dtype=dtype)
    frames = value.reshape(3, 4, 961)
    frames[..., 768:896] = 1.
    for lo in (923, 930):
        q = frames[..., lo:lo+4]
        q /= q.norm(dim=-1, keepdim=True)
    return value


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_canonicalization_is_invariant_to_independent_quaternion_signs(dtype):
    original = observations(dtype)
    changed = original.clone().reshape(3, 4, 961)
    changed[0, 0, 923:927] *= -1
    changed[1, 2, 930:934] *= -1
    changed[2, :, 923:927] *= -1
    a = helper().canonicalize_wrist_quaternion_history(original)
    b = helper().canonicalize_wrist_quaternion_history(changed.flatten(1))
    assert torch.equal(a, b)
    assert a.dtype == dtype and a.device == original.device


def test_only_quaternion_signs_change_and_input_is_owned_without_normalization():
    original = observations(torch.float64)
    original.reshape(3, 4, 961)[..., 923:927] *= 3.
    before = original.clone()
    canonical = helper().canonicalize_wrist_quaternion_history(original)
    mask = torch.ones(4, 961, dtype=torch.bool)
    mask[:, 923:927] = False; mask[:, 930:934] = False
    assert original.numpy().tobytes() == before.numpy().tobytes()
    assert canonical[:, mask.flatten()].numpy().tobytes() == original[:, mask.flatten()].numpy().tobytes()
    for lo in (923, 930):
        old = original.reshape(3, 4, 961)[..., lo:lo+4]
        new = canonical.reshape(3, 4, 961)[..., lo:lo+4]
        assert torch.equal(old.abs(), new.abs())
        dominant = new.gather(-1, new.abs().argmax(-1, keepdim=True))
        assert (dominant >= 0).all()
    canonical.zero_()
    assert torch.equal(original, before)


def test_zero_history_and_zero_scalar_component_have_deterministic_signs():
    value = torch.zeros(1, 3844)
    frame = value.reshape(1, 4, 961)
    frame[0, -1, 923:927] = torch.tensor([0., -1., 0., 0.])
    frame[0, -1, 930:934] = torch.tensor([0., -.5, .5, 0.])
    actual = helper().canonicalize_wrist_quaternion_history(value).reshape(1, 4, 961)
    assert not actual[:, :3].any()
    assert torch.equal(actual[0, -1, 923:927], torch.tensor([0., 1., 0., 0.]))
    assert torch.equal(actual[0, -1, 930:934], torch.tensor([0., .5, -.5, 0.]))


@pytest.mark.parametrize('actor_class', [VisionStudentActor, UnboundedVisionStudentActor])
def test_opt_in_actor_has_identical_output_for_antipodal_quaternions(actor_class):
    actor = actor_class(wrist_quaternion_preprocessing=MODE).eval()
    value = observations(); flipped = value.clone().reshape(3, 4, 961)
    flipped[..., 923:927] *= -1; flipped[..., 930:934] *= -1
    with torch.inference_mode():
        assert torch.equal(actor(value), actor(flipped.flatten(1)))


@pytest.mark.parametrize('actor_class', [VisionStudentActor, UnboundedVisionStudentActor])
def test_opt_in_actor_supports_finite_training_gradients_without_mutating_input(actor_class):
    actor = actor_class(wrist_quaternion_preprocessing=MODE)
    value = observations().requires_grad_()
    before = value.detach().clone()
    actor(value).square().mean().backward()
    assert torch.equal(value.detach(), before)
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())


@pytest.mark.parametrize('actor_class', [VisionStudentActor, UnboundedVisionStudentActor])
def test_default_and_explicit_raw_preserve_legacy_output_and_parameter_schema(actor_class):
    original = actor_class().eval()
    raw = actor_class(wrist_quaternion_preprocessing='raw').eval()
    canonical = actor_class(wrist_quaternion_preprocessing=MODE).eval()
    raw.load_state_dict(original.state_dict(), strict=True)
    canonical.load_state_dict(original.state_dict(), strict=True)
    assert list(raw.state_dict()) == list(original.state_dict()) == list(canonical.state_dict())
    assert len(original.state_dict()) == 12
    assert original.wrist_quaternion_preprocessing == 'raw'
    value = observations()
    with torch.inference_mode():
        assert torch.equal(original(value), raw(value))
        expected = raw(helper().canonicalize_wrist_quaternion_history(value))
        assert torch.equal(canonical(value), expected)
    assert POLICY_DIM == 3844 and FRAME_DIM == 961


@pytest.mark.parametrize('bad', [None, True, 'hemisphere', '', 1])
def test_unknown_preprocessing_mode_rejected(bad):
    with pytest.raises(ValueError, match='quaternion'):
        VisionStudentActor(wrist_quaternion_preprocessing=bad)


@pytest.mark.parametrize('bad', ['shape', 'integer', 'nan_quaternion'])
def test_pure_helper_rejects_invalid_quaternion_input(bad):
    value = observations()
    if bad == 'shape': value = value[:, :960]
    elif bad == 'integer': value = value.long()
    else: value.reshape(3, 4, 961)[0, 3, 923] = float('nan')
    with pytest.raises(ValueError, match='quaternion'):
        helper().canonicalize_wrist_quaternion_history(value)
