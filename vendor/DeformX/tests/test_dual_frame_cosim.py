import numpy as np

from co_sim.engine import CoSimEngine
from co_sim.models import CoSimConfig, FrameState


def _frame(position):
    return FrameState(
        position=np.asarray(position, dtype=float),
        director=np.eye(3),
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        omega=np.zeros(3),
        alpha=np.zeros(3),
    )


def test_dual_frame_engine_advances_one_rod_and_reports_both_impulses():
    config = CoSimConfig(
        py_dt=1.0e-5,
        isaac_dt=1.0e-5,
        n_elem=4,
        base_length=0.2,
        base_radius=0.002,
        density=1000.0,
        youngs_modulus=1.0e4,
        damping_constant=0.01,
        joint_k=100.0,
        joint_nu=2.0,
        settle_time=0.0,
    )
    left = _frame([0.0, 0.0, 0.0])
    right = _frame([0.2, 0.0, 0.0])
    engine = CoSimEngine(
        config,
        frame_initial_state=left,
        second_frame_initial_state=right,
    )

    result = engine.update_dual_frame_state(left, right)
    snapshot = engine.snapshot()

    assert snapshot.rod_position.shape == (3, 5)
    assert result.left_linear_impulse.shape == (3,)
    assert result.right_linear_impulse.shape == (3,)
    assert np.isfinite(snapshot.rod_position).all()
    assert result.elapsed_time == config.isaac_dt


def test_dual_frame_engine_reacts_to_each_endpoint_motion():
    config = CoSimConfig(
        py_dt=1.0e-5,
        isaac_dt=1.0e-4,
        n_elem=4,
        base_length=0.2,
        base_radius=0.002,
        density=1000.0,
        youngs_modulus=1.0e4,
        damping_constant=0.01,
        joint_k=1000.0,
        joint_nu=2.0,
        settle_time=0.0,
    )
    left = _frame([0.0, 0.0, 0.0])
    right = _frame([0.2, 0.0, 0.0])
    engine = CoSimEngine(
        config,
        frame_initial_state=left,
        second_frame_initial_state=right,
    )

    moved_left = _frame([0.0, 0.001, 0.0])
    left_result = engine.update_dual_frame_state(moved_left, right)
    moved_right = _frame([0.2, -0.001, 0.0])
    right_result = engine.update_dual_frame_state(moved_left, moved_right)

    assert np.linalg.norm(left_result.left_linear_impulse) > 0.0
    assert np.linalg.norm(right_result.right_linear_impulse) > 0.0
    assert np.isfinite(engine.snapshot().rod_position).all()
