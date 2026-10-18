from pathlib import Path

import numpy as np
import pytest


EXTERNAL = Path(__file__).resolve().parents[1]/"vendor/DeformX"
if not EXTERNAL.is_dir():
    pytest.skip("PyElastica-Mesh checkout is absent", allow_module_level=True)
pytest.MonkeyPatch().syspath_prepend(str(EXTERNAL))

import elastica  # noqa: F401 -- import failure is a real test failure, not a skip
from co_sim.free_rod import FreeRodEngine
from co_sim.models import CoSimConfig, RodInitialState


def _config(**changes):
    values = dict(
        py_dt=1.0e-4,
        isaac_dt=1.0e-3,
        n_elem=4,
        base_length=0.5,
        base_radius=0.008,
        density=1000.0,
        youngs_modulus=1.0e4,
        damping_constant=0.0,
        rod_start=np.array([0.0, 0.0, 0.2]),
        rod_direction=np.array([1.0, 0.0, 0.0]),
        rod_normal=np.array([0.0, 0.0, 1.0]),
    )
    values.update(changes)
    return CoSimConfig(**values)


def _assert_snapshot_equal(left, right):
    assert left["mode"] == right["mode"]
    assert left["active_constraints"] == right["active_constraints"]
    assert left["sim_time"] == right["sim_time"]
    for name in ("rod_position", "rod_velocity", "rod_director", "rod_omega"):
        np.testing.assert_array_equal(left[name], right[name])


def test_free_rod_gravity_moves_every_node_without_rotating_or_pinning():
    engine = FreeRodEngine(_config(isaac_dt=0.01))
    initial = engine.snapshot()
    result = engine.advance()

    np.testing.assert_allclose(
        result["rod_position"][2] - initial["rod_position"][2],
        -0.5 * 9.81 * 0.01**2,
        atol=2e-8,
    )
    np.testing.assert_allclose(result["rod_velocity"][2], -9.81 * 0.01, atol=2e-8)
    np.testing.assert_allclose(result["rod_velocity"][:2], 0.0, atol=2e-10)
    np.testing.assert_allclose(result["rod_director"], initial["rod_director"], atol=2e-10)
    assert result["mode"] == "FREE"
    assert result["active_constraints"] == ()


def test_remainder_and_sequential_advances_preserve_time_and_rod_identity():
    engine = FreeRodEngine(_config(py_dt=3.0e-4))
    rod_id = id(engine.rod)
    assert engine.advance(7.0e-4)["sim_time"] == pytest.approx(7.0e-4)
    assert engine.advance(2.0e-4)["sim_time"] == pytest.approx(9.0e-4)
    assert id(engine.rod) == rod_id


def test_damped_final_partial_step_matches_explicit_real_step_subdivision():
    config = _config(py_dt=3.0e-4, damping_constant=20.0)
    actual = FreeRodEngine(config)
    reference = FreeRodEngine(config)

    actual_result = actual.advance(7.0e-4)
    reference.advance(3.0e-4)
    reference.advance(3.0e-4)
    reference_result = reference.advance(1.0e-4)

    np.testing.assert_allclose(
        actual_result["rod_velocity"], reference_result["rod_velocity"], atol=1e-15
    )
    np.testing.assert_allclose(
        actual_result["rod_position"], reference_result["rod_position"], atol=1e-15
    )


def test_snapshot_and_constructor_inputs_are_isolated_copies():
    config = _config()
    state = RodInitialState(
        start=np.array([0.0, 0.0, 0.2]),
        direction=np.array([1.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
    )
    engine = FreeRodEngine(config, state)
    expected = engine.snapshot()
    state.start[:] = 99.0
    config.rod_start[:] = 88.0
    returned = engine.snapshot()
    returned["rod_position"][:] = 77.0
    returned["rod_velocity"][:] = 77.0
    _assert_snapshot_equal(expected, engine.snapshot())


@pytest.mark.parametrize("duration", [True, False, "0.1", 0.0, -1.0, np.nan, np.inf])
def test_invalid_advance_is_rejected_without_state_change(duration):
    engine = FreeRodEngine(_config())
    before = engine.snapshot()
    with pytest.raises((TypeError, ValueError)):
        engine.advance(duration)
    _assert_snapshot_equal(before, engine.snapshot())


@pytest.mark.parametrize(
    "changes",
    [
        {"n_elem": 1}, {"n_elem": True}, {"py_dt": 0.0}, {"py_dt": "0.001"},
        {"isaac_dt": np.nan},
        {"base_length": -1.0}, {"base_radius": np.inf}, {"density": 0.0},
        {"youngs_modulus": 0.0}, {"shear_modulus_ratio": 0.0},
        {"axial_stretch_stiffening": 0.0}, {"damping_constant": -1.0},
    ],
)
def test_invalid_consumed_config_is_rejected(changes):
    with pytest.raises((TypeError, ValueError)):
        FreeRodEngine(_config(**changes))


@pytest.mark.parametrize(
    "state",
    [
        RodInitialState(np.zeros(2), np.array([1., 0., 0.]), np.array([0., 0., 1.])),
        RodInitialState(np.array([np.nan, 0., 0.]), np.array([1., 0., 0.]), np.array([0., 0., 1.])),
        RodInitialState(np.zeros(3), np.array([2., 0., 0.]), np.array([0., 0., 1.])),
        RodInitialState(np.zeros(3), np.array([1., 0., 0.]), np.array([1., 0., 0.])),
    ],
)
def test_invalid_initial_state_is_rejected(state):
    with pytest.raises(ValueError):
        FreeRodEngine(_config(), state)


@pytest.mark.parametrize(
    "changes",
    [{"use_ground_contact": True}, {"settle_time": 0.1}, {"initial_wire_theta": 0.0}],
)
def test_bound_only_capabilities_are_explicitly_rejected(changes):
    with pytest.raises((TypeError, ValueError), match="unsupported"):
        FreeRodEngine(_config(**changes))


def test_unrepresentable_time_increment_is_rejected_before_state_change():
    engine = FreeRodEngine(_config())
    engine.time = np.float64(1.0e20)
    before = engine.snapshot()
    with pytest.raises(ValueError, match="represent"):
        engine.advance(1.0)
    _assert_snapshot_equal(before, engine.snapshot())


def test_nonfinite_step_result_marks_engine_unusable():
    engine = FreeRodEngine(_config())

    class BadStepper:
        def step(self, sim, time, dt):
            engine.rod.position_collection[0, 0] = np.nan
            return time + dt

    engine.stepper = BadStepper()
    with pytest.raises(RuntimeError, match="non-finite"):
        engine.advance()
    with pytest.raises(RuntimeError, match="unusable"):
        engine.advance()
    with pytest.raises(RuntimeError, match="non-finite"):
        engine.snapshot()


def test_nonprogressing_stepper_fails_without_hanging_and_marks_unusable():
    engine = FreeRodEngine(_config())

    class StuckStepper:
        def step(self, sim, time, dt):
            return time

    engine.stepper = StuckStepper()
    with pytest.raises(RuntimeError, match="progress"):
        engine.advance()
    with pytest.raises(RuntimeError, match="unusable"):
        engine.advance()


def test_nonfinite_stepper_time_marks_engine_unusable():
    engine = FreeRodEngine(_config())

    class NonfiniteTimeStepper:
        def step(self, sim, time, dt):
            return np.nan

    engine.stepper = NonfiniteTimeStepper()
    with pytest.raises(RuntimeError, match="finite forward progress"):
        engine.advance()
    with pytest.raises(RuntimeError, match="unusable"):
        engine.advance()
