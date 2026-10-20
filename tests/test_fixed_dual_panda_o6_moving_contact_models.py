from pathlib import Path

import numpy as np
import pytest


EXTERNAL = Path(__file__).resolve().parents[1]/"vendor/DeformX"
if not EXTERNAL.is_dir():
    pytest.skip("PyElastica-Mesh checkout is absent", allow_module_level=True)
pytest.MonkeyPatch().syspath_prepend(str(EXTERNAL))

from co_sim.moving_contact_models import (  # noqa: E402
    ContactImpulse,
    MeshMotion,
    predict_motion,
    validate_motion,
)


def _motion(**changes):
    values = dict(
        position_w=np.zeros(3),
        rotation_wb=np.eye(3),
        velocity_w=np.zeros(3),
        omega_w=np.zeros(3),
        acceleration_w=np.zeros(3),
        alpha_w=np.zeros(3),
    )
    values.update(changes)
    return MeshMotion(**values)


def test_trusted_prediction_reuses_boundary_validation_without_changing_result(monkeypatch):
    from co_sim import moving_contact_models as module
    current=validate_motion(_motion(velocity_w=np.array([.01,-.02,.03]),
        omega_w=np.array([.1,.2,-.3]),acceleration_w=np.array([.4,-.2,.1])))
    expected=predict_motion(current,.0001)
    def duplicate_validation(value):
        pytest.fail('trusted internal motion was redundantly revalidated')
    monkeypatch.setattr(module,'validate_motion',duplicate_validation)
    actual=module._predict_validated_motion(current,.0001)
    for field in ('position_w','rotation_wb','velocity_w','omega_w','acceleration_w','alpha_w'):
        np.testing.assert_array_equal(getattr(actual,field),getattr(expected,field))
        assert not getattr(actual,field).flags.writeable


def test_trusted_prediction_still_rejects_nonfinite_result():
    from co_sim import moving_contact_models as module
    current=validate_motion(_motion(velocity_w=np.full(3,1e308)))
    with pytest.raises(ValueError,match='finite'):
        module._predict_validated_motion(current,10.)


def test_predict_motion_uses_updated_linear_velocity_for_position():
    predicted = predict_motion(_motion(acceleration_w=np.array([2.0, 0.0, 0.0])), 0.1)

    np.testing.assert_allclose(predicted.velocity_w, [0.2, 0.0, 0.0])
    np.testing.assert_allclose(predicted.position_w, [0.02, 0.0, 0.0])


def test_predict_motion_preserves_constant_velocity_translation():
    predicted = predict_motion(
        _motion(position_w=np.array([1.0, 2.0, 3.0]), velocity_w=np.array([0.5, -1.0, 2.0])),
        0.2,
    )

    np.testing.assert_allclose(predicted.position_w, [1.1, 1.8, 3.4])
    np.testing.assert_array_equal(predicted.velocity_w, [0.5, -1.0, 2.0])


def test_predict_motion_left_multiplies_world_angular_increment():
    initial_rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    predicted = predict_motion(
        _motion(rotation_wb=initial_rotation, alpha_w=np.array([0.0, 0.0, np.pi / 2.0])),
        1.0,
    )
    world_z_quarter_turn = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    np.testing.assert_allclose(
        predicted.rotation_wb, world_z_quarter_turn @ initial_rotation, atol=1.0e-15
    )
    np.testing.assert_allclose(predicted.omega_w, [0.0, 0.0, np.pi / 2.0])


def test_zero_rotation_branch_is_exact_and_finite():
    initial = _motion()
    predicted = predict_motion(initial, 0.1)

    np.testing.assert_array_equal(predicted.rotation_wb, np.eye(3))
    assert np.isfinite(predicted.rotation_wb).all()


@pytest.mark.parametrize("dt", [True, False, 0, -0.1, np.nan, np.inf, "0.1"])
def test_predict_motion_rejects_invalid_dt(dt):
    with pytest.raises((TypeError, ValueError)):
        predict_motion(_motion(), dt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("position_w", np.zeros(2)),
        ("velocity_w", np.array([0.0, np.nan, 0.0])),
        ("omega_w", np.array([0.0, np.inf, 0.0])),
        ("acceleration_w", [0.0, True, 0.0]),
        ("acceleration_w", np.array([0.0, True, 0.0], dtype=object)),
        ("alpha_w", np.array([0.0, 0.0, 1.0j])),
        ("rotation_wb", np.eye(2)),
        ("rotation_wb", np.diag([1.0, 1.0, -1.0])),
        ("rotation_wb", np.array([[1.0, 1.0e-5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
    ],
)
def test_mesh_motion_rejects_invalid_arrays(field, value):
    with pytest.raises((TypeError, ValueError)):
        _motion(**{field: value})


def test_motion_inputs_and_results_are_defensive_read_only_copies():
    position = np.array([1.0, 2.0, 3.0])
    rotation = np.eye(3)
    motion = _motion(position_w=position, rotation_wb=rotation)
    validated = validate_motion(motion)
    predicted = predict_motion(motion, 0.1)

    position[:] = 99.0
    rotation[:] = 88.0
    np.testing.assert_array_equal(motion.position_w, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(motion.rotation_wb, np.eye(3))
    assert validated is not motion
    for value in (
        motion.position_w,
        motion.rotation_wb,
        validated.position_w,
        predicted.position_w,
    ):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flat[0] = 7.0
        with pytest.raises(ValueError):
            value.setflags(write=True)


def test_failed_prediction_does_not_change_an_earlier_result():
    earlier = predict_motion(_motion(velocity_w=np.array([1.0, 0.0, 0.0])), 0.25)
    earlier_position = earlier.position_w.copy()

    with pytest.raises(ValueError):
        predict_motion(_motion(), np.nan)

    np.testing.assert_array_equal(earlier.position_w, earlier_position)


def test_contact_impulse_is_validated_defensive_and_read_only():
    linear = np.array([1.0, 2.0, 3.0])
    angular = np.array([4.0, 5.0, 6.0])
    impulse = ContactImpulse(linear, angular, peak_force_n=7.0, contact_duration_s=0.1)
    linear[:] = 99.0
    angular[:] = 88.0

    np.testing.assert_array_equal(impulse.linear_w, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(impulse.angular_com_w, [4.0, 5.0, 6.0])
    assert not impulse.linear_w.flags.writeable
    assert not impulse.angular_com_w.flags.writeable


@pytest.mark.parametrize(
    "kwargs",
    [
        {"linear_w": np.zeros(2)},
        {"linear_w": np.array([0.0, np.nan, 0.0])},
        {"angular_com_w": [0.0, True, 0.0]},
        {"angular_com_w": np.array([0.0, True, 0.0], dtype=object)},
        {"peak_force_n": True},
        {"peak_force_n": -1.0},
        {"peak_force_n": np.inf},
        {"contact_duration_s": False},
        {"contact_duration_s": -0.1},
        {"contact_duration_s": np.nan},
    ],
)
def test_contact_impulse_rejects_invalid_values(kwargs):
    values = dict(
        linear_w=np.zeros(3),
        angular_com_w=np.zeros(3),
        peak_force_n=0.0,
        contact_duration_s=0.0,
    )
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        ContactImpulse(**values)
