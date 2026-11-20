import subprocess
import sys
import numpy as np
import pytest

from scripts import verify_dh116_power_live_palm_feedback as feedback


def test_prepare_reaction_preserves_force_and_moves_mesh_moment_to_com():
    palm = dict(body_path="/World/LeftHand/base_link", body_id=8, sdk_step=7,
                xyz=[1., 2., 3.], com=[1.2, 2., 3.], quat_xyzw=[0., 0., 0., 1.])
    solved = dict(force_on_mesh_n=[0., 0., 5.], torque_about_mesh_origin_world_nm=[0., .4, 0.],
                  moment_reference_audit=dict(frame="world", about="mesh_origin", origin_world_m=palm["xyz"]))
    route = feedback.prepare_reaction(palm, solved, current_sdk_step=7)
    assert route["body_ids"] == [8] and route["is_global"] is True
    np.testing.assert_allclose(route["forces"], [[[0., 0., 5.]]])
    np.testing.assert_allclose(route["torques"], [[[0., 1.4, 0.]]])


@pytest.mark.parametrize("reason", ["stale", "wrong-origin", "excess-load"])
def test_prepare_reaction_rejects_unowned_or_unsafe_packet(reason):
    palm = dict(body_path="/World/LeftHand/base_link", body_id=8, sdk_step=7,
                xyz=[1., 2., 3.], com=[1.2, 2., 3.], quat_xyzw=[0., 0., 0., 1.])
    solved = dict(force_on_mesh_n=[0., 0., 5.], torque_about_mesh_origin_world_nm=[0., .4, 0.],
                  moment_reference_audit=dict(frame="world", about="mesh_origin", origin_world_m=palm["xyz"]))
    sdk = 7
    if reason == "stale":
        sdk += 1
    if reason == "wrong-origin":
        solved["moment_reference_audit"]["origin_world_m"] = [0., 0., 0.]
    if reason == "excess-load":
        solved["force_on_mesh_n"] = [0., 0., 31.]
    with pytest.raises(ValueError):
        feedback.prepare_reaction(palm, solved, current_sdk_step=sdk)


def test_feedback_requires_the_measured_loaded_step_after_unloaded_controls():
    force = np.array([0., 0., 4.])
    baseline = [[0., 0., .00002], [0., 0., -.00002]]
    report = feedback.audit_velocity_change(force, baseline, [0., 0., .001])
    assert report["response_detected"]
    with pytest.raises(ValueError, match="response"):
        feedback.audit_velocity_change(force, baseline, [0., 0., -.001])


def test_feedback_worker_leaves_usd_for_app_launcher():
    worker = subprocess.run([sys.executable, "-c",
        "import scripts.verify_dh116_power_live_palm_feedback; "
        "from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary; "
        "assert_clean_import_boundary()"], capture_output=True, text=True)
    assert worker.returncode == 0, worker.stderr
