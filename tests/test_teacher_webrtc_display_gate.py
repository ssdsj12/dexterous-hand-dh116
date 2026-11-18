import json
from types import SimpleNamespace

import pytest

from scripts import view_dh116_teacher_grasps_webrtc as viewer


def audit(position=0., rotation=0.):
    return dict(physics_clock=(10, 10/480), updated_links=102,
                max_position_error_m=position, max_rotation_error_rad=rotation)


@pytest.mark.parametrize("position, rotation", [
    (2e-4, 0.), (0., 2e-3), (float("nan"), 0.), (0., float("inf")),
])
def test_unverified_frame_is_logged_and_hidden_without_interrupting_controls(tmp_path, position, rotation):
    viewport = SimpleNamespace(visible=True)
    assert viewer._publish_audited_view(audit(position, rotation), viewport, tmp_path, 1) is False
    assert not viewport.visible
    rows = (tmp_path/"display_rejections.jsonl").read_text().splitlines()
    assert len(rows) == 1
    record = json.loads(rows[0])
    assert (record["control_step"], record["sdk_step"], record["updated_links"]) == (1, 10, 102)
    assert record["display_qualified"] is False
    assert viewer._publish_audited_view(audit(1e-4, 1e-3), viewport, tmp_path, 2) is True
    assert viewport.visible
    assert (tmp_path/"display_rejections.jsonl").read_text().splitlines() == rows


def test_headless_viewer_still_records_pose_rejections(tmp_path):
    assert viewer._publish_audited_view(audit(0., .01), None, tmp_path, 1) is False
    assert (tmp_path/"display_rejections.jsonl").is_file()


def test_checkpoint_root_keeps_the_same_sha_when_sharing_models(tmp_path):
    checkpoint = tmp_path/"best.pt"
    checkpoint.write_bytes(b"original bytes")
    from hashlib import sha256

    config = dict(checkpoints={"pinch": "best.pt"}, checkpoint_sha256={"pinch": sha256(b"original bytes").hexdigest()})
    assert viewer._read_frozen_checkpoints(tmp_path, config) == {"pinch": b"original bytes"}
    checkpoint.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="checkpoint changed"):
        viewer._read_frozen_checkpoints(tmp_path, config)



@pytest.mark.parametrize("sync_fails", [False, True])
def test_shutdown_records_phases_and_keeps_errors_visible(tmp_path, sync_fails):
    calls = []

    def callback(name):
        def run():
            calls.append(name)
            if name == "cuda_synchronize" and sync_fails:
                raise RuntimeError("CUDA error")
            return {"done": True} if name == "stop" else None
        return run

    steps = ("freeze", "cuda_synchronize", "release", "stop", "app_close")
    def close():
        viewer._shutdown_preview(tmp_path, [(name, callback(name)) for name in steps[:-1]],
                                 callback(steps[-1]))

    if sync_fails:
        with pytest.raises(RuntimeError, match="CUDA error"):
            close()
        assert calls == ["freeze", "cuda_synchronize", "app_close"]
    else:
        close()
        assert calls == list(steps)
    rows = [json.loads(line) for line in (tmp_path/"shutdown_phases.jsonl").read_text().splitlines()]
    assert [row["phase"] for row in rows[::2]] == calls
    assert rows[-1]["event"] == "returned"
    assert rows[-1]["phase"] == "app_close"
    assert rows[-1]["process_exit_verified"] is False
    if sync_fails:
        assert rows[3]["event"] == "error"
    else:
        assert rows[-3]["result"] == {"done": True}
