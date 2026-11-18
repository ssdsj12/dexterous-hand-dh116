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
    assert result["physical_state_trace"] == {phase: state for phase in ("before_render", "before_read", "after_read")}
    assert result["render_clock"] == (11, 8, 8)
    assert result["camera_render_clocks"] == {
        side: {"before_render": (10, 8, 8), "before_read": (11, 8, 8), "after_read": (11, 8, 8)}
        for side in buffers
    }
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


def test_sdk_default_stage_option_only_changes_one_launch_config_key(tmp_path, monkeypatch):
    import hashlib

    from scripts import probe_dh116_render_lifecycle as probe
    from scripts import run_dh116_deformx_render_smoke as smoke

    kit = tmp_path / "isaaclab.python.headless.rendering.kit"
    kit.write_bytes(b"fixed experience fixture")
    monkeypatch.setattr(smoke, "HEADLESS_RENDERING_EXPERIENCE_SHA256",
                        hashlib.sha256(kit.read_bytes()).hexdigest())
    baseline, options = probe.standalone_launch_options(kit, False)
    changed, changed_options = probe.standalone_launch_options(kit, True)
    assert changed == {**baseline, "create_new_stage": False}
    assert options == changed_options
    assert "create_new_stage" not in baseline



class _Readback:
    def __init__(self, value):
        self.value = value

    def get_data(self, **kwargs):
        return self.value


def _camera_readers(width, height, *, radial, optical, ids):
    return {
        "rgb": _Readback(np.zeros((height, width, 4), dtype=np.uint8)),
        "radial_distance_m": _Readback(radial),
        "optical_z_m": _Readback(optical),
        "native_instance_id": _Readback({"data": ids, "info": {"idToLabels": {"0": "BACKGROUND"}}}),
    }


@pytest.mark.parametrize("wrong_field", ["radial_distance_m", "optical_z_m", "native_instance_id"])
def test_native_camera_rejects_transposed_scalar_image(wrong_field):
    from scripts import probe_dh116_render_lifecycle as probe

    depth = np.ones((3, 5), dtype=np.float32)
    ids = np.zeros((3, 5), dtype=np.uint32)
    readers = _camera_readers(5, 3, radial=depth, optical=depth, ids=ids)
    if wrong_field == "native_instance_id":
        readers[wrong_field].value["data"] = ids.T
    else:
        readers[wrong_field].value = depth.T
    with pytest.raises(ValueError, match=wrong_field):
        probe._read_products(readers, 5, 3)


def test_native_camera_retains_independent_depth_validity():
    from scripts import probe_dh116_render_lifecycle as probe

    radial = np.ones((3, 5, 1), dtype=np.float32)
    optical = np.ones((3, 5), dtype=np.float32)
    radial[0, 0, 0] = np.inf
    optical[0, 1] = np.nan
    readers = _camera_readers(5, 3, radial=radial, optical=optical,
                              ids=np.zeros((3, 5), dtype=np.int32))
    products = probe._read_products(readers, 5, 3)
    assert products["radial_valid"][0, :2].tolist() == [False, True]
    assert products["depth_valid"][0, :2].tolist() == [True, False]
    assert np.isinf(products["radial_distance_m"][0, 0])
    assert products["native_instance_id"].dtype == np.uint32


@pytest.mark.parametrize("failure", [None, "relabeled_depth", "late_read", "changed_state"])
def test_v3_artifact_audit_recomputes_depth_and_clock(tmp_path, failure):
    import json
    from scripts import probe_dh116_render_lifecycle as probe

    width, height = 96, 64
    k = np.array([[80., 0., width / 2], [0., 80., height / 2], [0., 0., 1.]])
    rows, cols = np.indices((height, width))
    optical = np.full((height, width), 1.98, dtype=np.float32)
    radial = (optical * np.sqrt(1 + ((cols + .5 - width / 2) / 80) ** 2
                                + ((rows + .5 - height / 2) / 80) ** 2)).astype(np.float32)
    image = tmp_path / "plane-single.npz"
    if failure == "relabeled_depth":
        optical = radial.copy()
    np.savez(image, rgb=np.zeros((height, width, 4), dtype=np.uint8),
             radial_distance_m=radial, optical_z_m=optical,
             radial_valid=np.isfinite(radial), depth_valid=np.isfinite(optical),
             native_instance_id=np.zeros((height, width), dtype=np.uint32))
    state = {"physics_tick": None, "rod_step": None, "state_sha256": "a" * 64, "simulation_time": 0.}
    trace = {phase: state.copy() for phase in ("before_render", "before_read", "after_read")}
    if failure == "changed_state":
        trace["after_read"]["state_sha256"] = "b" * 64
    (tmp_path / "plan.json").write_text(json.dumps({"case": "V3"}))
    (tmp_path / "plane-boundary.json").write_text(json.dumps({
        "state": state, "physical_state_trace": trace,
        "render_clock": [11, 0, 1, 4, 0.],
        "camera_render_clocks": {"single": {
            "before_render": [10, 0, 1, 3, 0.], "before_read": [11, 0, 1, 4, 0.],
            "after_read": [12 if failure == "late_read" else 11, 0, 1, 4, 0.],
        }},
        "cameras": {"single": {
            "capture_sha256": probe._sha(image.read_bytes()),
            "camera_intrinsics": k.tolist(),
            "camera_intrinsics_sha256": probe._state_sha([k]),
        }},
        "capture_synchronization_verified": True, "depth_semantics_verified": True,
    }))
    evidence = probe.audit_record(tmp_path, "V3", 255)
    assert evidence["artifact_integrity_verified"] is (failure is None)
    assert evidence["depth_semantics_verified"] is (failure is None)
    assert evidence["process_exit_verified"] is False


def test_ready_task_stereo_boundary_renders_once_and_archives_both_views(tmp_path, monkeypatch):
    import json
    from scripts import probe_dh116_render_lifecycle as probe

    width, height = 5, 3
    k = np.array([[4., 0., width / 2], [0., 4., height / 2], [0., 0., 1.]])
    rows, cols = np.indices((height, width))
    optical = np.ones((height, width), dtype=np.float32)
    radial = np.sqrt(1 + ((cols + .5 - width / 2) / 4) ** 2
                     + ((rows + .5 - height / 2) / 4) ** 2).astype(np.float32)
    cameras = {
        side: _camera_readers(width, height, radial=radial, optical=optical,
                              ids=np.zeros((height, width), dtype=np.uint32))
        for side in ("left", "right")
    }
    state = {"physics_tick": 8, "rod_step": None, "state_sha256": "a" * 64, "simulation_time": .5}
    calls = []
    monkeypatch.setattr(probe, "_clock", lambda reader: (10 + len(calls), 15, 30, 15, .5))
    result = probe._capture_pair(tmp_path, "post_zero_action", lambda: state,
                                 lambda: calls.append("render"), cameras,
                                 {side: k for side in cameras}, width, height, warmup=False)
    assert calls == ["render"]
    assert result["render_clock"] == (11, 15, 30, 15, .5)
    record = json.loads((tmp_path / "post_zero_action-boundary.json").read_text())
    probe._audit_render_clock(record, set(cameras))
    for side in cameras:
        probe._audit_saved_products(tmp_path / f"post_zero_action-{side}.npz",
                                    record["cameras"][side], width, height, None)
