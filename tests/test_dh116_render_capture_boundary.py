import numpy as np
import pytest


from go2_pvcnn.control.dh116 import tabletop_stereo_capture_boundary as module

def test_one_render_fans_out_and_copies_both_camera_results():
    state = {"physics_tick": 8, "rod_step": 13, "state_sha256": "a" * 64}
    buffers = {"left": np.array([1, 2]), "right": np.array([3, 4])}
    renders = []

    def clock(side):
        return (10 + len(renders), 8, 8)

    result = module.capture_at_boundary(lambda: state, lambda: renders.append(1), clock, {
        side: (lambda side=side: {"rgb": buffers[side]}) for side in buffers
    })
    buffers["left"][:] = -1
    assert renders == [1]
    assert result["state"] == state
    assert result["render_clock"] == (11, 8, 8)
    np.testing.assert_array_equal(result["captures"]["left"]["rgb"], [1, 2])


@pytest.mark.parametrize("failure", ["stale", "cross_camera", "read_clock", "physics", "rod"])
def test_render_admission_rejects_stale_frames_and_advancing_state(failure):
    state = {"physics_tick": 8, "rod_step": 13, "state_sha256": "a" * 64}
    epoch = [10]

    def render():
        if failure != "stale":
            epoch[0] += 1
        if failure == "physics":
            state["physics_tick"] += 1
        if failure == "rod":
            state["rod_step"] += 1

    def clock(side):
        mismatch = failure == "cross_camera" and side == "right" and epoch[0] > 10
        return (epoch[0] + mismatch, 8, 8)

    def read_left():
        if failure == "read_clock":
            epoch[0] += 1
        return {"rgb": np.array([1])}

    with pytest.raises(ValueError, match="capture boundary"):
        module.capture_at_boundary(lambda: state, render, clock, {
            "left": read_left, "right": lambda: {"rgb": np.array([2])},
        })


def test_radial_and_optical_z_differ_off_axis_and_preserve_invalid_values():
    K = np.array([[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]])
    radial = np.full((3, 3), 2.0)
    radial[0, 0] = np.inf
    z, valid = module.pinhole_radial_to_z(radial, K)
    assert not valid[0, 0] and np.isnan(z[0, 0])
    assert z[1, 1] == pytest.approx(2.0)
    assert z[2, 2] == pytest.approx(2.0 / np.sqrt(1.5))
    metrics = module.audit_depth_pair(radial, z, K)
    assert metrics["p95_abs_error_m"] == 0.0
    assert metrics["center_pixels"] > 0 and metrics["edge_pixels"] > 0


def test_depth_audit_flags_radial_relabelled_as_optical_z():
    K = np.array([[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]])
    radial = np.full((3, 3), 2.0)
    metrics = module.audit_depth_pair(radial, radial.copy(), K)
    assert metrics["p95_abs_error_m"] > 0.2


def test_verified_artifacts_and_app_close_cannot_qualify_exit139():
    status = module.lifecycle_status(artifact_integrity=True, synchronized=True, optical_depth=True,
                                     app_close_returned=True, outer_exit_code=139)
    assert status == {
        "artifact_integrity_verified": True,
        "capture_synchronization_verified": True,
        "depth_semantics_verified": True,
        "app_close_returned": True,
        "process_exit_verified": False,
        "restart_verified": False,
        "outer_exit_code": 139,
    }
    assert module.lifecycle_status(
        artifact_integrity=False, synchronized=False, optical_depth=False,
        app_close_returned=True, outer_exit_code=0,
    )["process_exit_verified"]
