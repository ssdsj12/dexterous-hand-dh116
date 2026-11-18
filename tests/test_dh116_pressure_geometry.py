import numpy as np
import pytest

from go2_pvcnn.control.dh116 import tabletop_pressure_geometry as geometry


def test_pressure_direction_closes_without_first_order_height_or_midpoint_drift():
    # Two pad Jacobians with shared upward drift and asymmetric thumb motion.
    jac = np.zeros((2, 3, 6))
    jac[0, :, :3] = [[-.03, -.04, 0], [.01, .02, 0], [.02, .05, 0]]
    jac[1, :, :3] = [[0, 0, .06], [0, 0, -.02], [0, 0, .04]]
    direction, compensation = geometry.pressure_direction(jac, [1., 0., 0.])
    velocity = jac @ direction + compensation
    assert velocity[0, 0] < 0 < velocity[1, 0]
    np.testing.assert_allclose(velocity[:, 1:], 0., atol=1e-12)
    np.testing.assert_allclose(velocity.mean(axis=0), 0., atol=1e-12)
    assert np.linalg.norm(direction) == pytest.approx(1.)
    np.testing.assert_array_equal(direction[3:], 0.)


def test_uncontrollable_relative_geometry_is_rejected():
    with pytest.raises(ValueError, match='rank'):
        geometry.pressure_direction(np.zeros((2, 3, 6)), [1., 0., 0.])
    with pytest.raises(ValueError, match='finite'):
        geometry.pressure_direction(np.full((2, 3, 6), np.nan), [1., 0., 0.])


def test_compensation_shares_existing_palm_envelope_and_reports_clipping():
    palm = np.zeros((2, 3)); palm[:, 2] = -1.
    compensation = np.array([[0, .01, -.04], [0, -.01, -.03]])
    action, clipped = geometry.compensated_palm_action(palm, [.08, .08], compensation)
    np.testing.assert_array_equal(action[:, 2], [-1., -1.])
    assert np.all(clipped[:, 2])
    assert not np.any(clipped[:, :2])
    np.testing.assert_allclose(action[:, 1], [.0008/.003, -.0008/.003])
    batch, batch_clipped = geometry.compensated_palm_action(
        np.stack([palm, palm]), np.full((2, 2), .08), compensation)
    np.testing.assert_allclose(batch, np.stack([action, action]))
    np.testing.assert_array_equal(batch_clipped, np.stack([clipped, clipped]))


def test_invalid_compensation_cannot_reach_physics():
    with pytest.raises(ValueError):
        geometry.compensated_palm_action(np.zeros((2, 3)), [.08, .08], np.full((2, 3), np.inf))


def test_environment_compensates_pressure_in_backend_request_and_exposes_error():
    from test_dh116_batched_task import make_env
    compensation = np.array([[0., 0., -.04], [0., 0., -.03]])
    env = make_env(0., compensation)
    env.force_controller.offset[:] = .02
    env.step(np.zeros(20))
    assert np.all(env.backend.adaptive_palm_offset[:, 2] < 0)
    diagnostics = env._info()['pressure_geometry']
    assert np.all(np.asarray(diagnostics['requested_compensation_m'])[:, 2] < 0)
    assert np.max(abs(np.asarray(diagnostics['palm_target_error_m']))) > 0


def test_offline_usd_fk_reproduces_independently_recorded_contact_pose():
    from pathlib import Path
    from go2_pvcnn.control.dh116.hand_point_kinematics import HandPointFK
    root = Path(__file__).resolve().parents[1]
    source = root/'assets/fixed_dual_panda_dh116_v1/dependencies/d134680ab619_DH116_left.usd'
    pose = [.5020698308944702,.18861708045005798,.42923831939697266,
            -.42810311913490295,-.34228965640068054,.7910710573196411,.2716103792190552]
    q = np.array([.3203346132196089,.4102028952660908,.6967254947035454,.2,.2,.2])
    fk = HandPointFK(source, 'left', pose)
    np.testing.assert_allclose(fk.points(q), [[.438,.2,.308],[.422,.2,.308]], atol=1e-7, rtol=0)
    direction, compensation = geometry.pressure_direction(fk.jacobian(q), [1.,0.,0.])
    displaced = fk.points(q+.08*direction)+.08*compensation
    assert abs(displaced[:,2]-.308).max() < .0004
