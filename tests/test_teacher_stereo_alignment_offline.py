"""Offline acceptance checks for frozen Teacher/Student alignment receipts."""
import json

import numpy as np
import pytest
from PIL import Image

from go2_pvcnn.control.dh116 import tabletop_teacher_stereo_alignment as alignment
from scripts import audit_dh116_teacher_stereo_alignment as audit
from test_teacher_stereo_alignment import scene


def write_frame(directory, tick=0, epoch=80):
    directory.mkdir()
    cal, state, pair, shift = scene()
    sdk_step = 2 + 8 * tick
    identity = pair["identity"]
    identity.update(run_uuid="run", env_id=0, episode_uid="episode", reset_id=0,
                    policy_tick=tick, physics_tick_end=sdk_step, sdk_step=sdk_step,
                    sdk_timestamp_s=sdk_step / 480, simulation_timestamp=sdk_step / 480,
                    rod_uid="rod", material_reference_uid="material")
    clock = dict(frame_id=epoch, render_frame_number=epoch, simulation_time=0.0)
    pair["render_timing"].update(capture_id=tick + 1, physical_step=sdk_step,
                                  physical_timestamp_s=sdk_step / 480,
                                  raw_left_clock=clock, raw_right_clock=dict(clock))
    result = alignment.aligned_payload(state, identity, pair, cal, shift[None])
    for side in ("left", "right"):
        path = directory / f"{side}.png"
        Image.new("RGB", (cal.width, cal.height)).save(path)
        result["cameras"][side].update(rgb_file=path.name, rgb_sha256=audit.file_sha(path))
    path = directory / "teacher_stereo_alignment.json"
    path.write_text(json.dumps(result))
    return cal, path, result


def test_recomputes_geometry_and_checks_independent_pixel_error(tmp_path):
    cal, path, receipt = write_frame(tmp_path / "f0")
    report = audit.audit_frame(path, cal)
    assert report["reprojection"] is None
    assert report["privileged_state_digest_verified"] is False
    labels = {"receipt_sha256": audit.file_sha(path), "source": "manual_rgb_annotation",
              "observations": []}
    for side, err in (("left", 1.0), ("right", 2.0)):
        camera = receipt["cameras"][side]
        labels["observations"].append(
            dict(side=side, node_index=0, rgb_sha256=camera["rgb_sha256"],
                 pixel_uv=(np.asarray(camera["rod_pixels_uv"][0]) + [err, 0]).tolist()))
    result = audit.audit_frame(path, cal, labels)["reprojection"]
    assert result["left"]["max_px"] == pytest.approx(1)
    assert result["left"]["all_strictly_under_2px"]
    assert not result["right"]["all_strictly_under_2px"]
    assert result["right"]["samples"] == 1


@pytest.mark.parametrize("fault", ["pixels", "palms", "lens", "clock", "rgb", "labels"])
def test_cannot_use_tampered_geometry_clock_images_or_bound_labels(tmp_path, fault):
    cal, path, data = write_frame(tmp_path / "f0")
    labels = None
    if fault == "pixels":
        data["cameras"]["left"]["rod_pixels_uv"][0][0] += 1
    elif fault == "palms":
        data["rod_nodes_palm_m"]["left"][0][0] += .01
    elif fault == "lens":
        data["cameras"]["left"]["K"][0][2] += 1
    elif fault == "clock":
        data["render_timing"]["raw_right_clock"]["render_frame_number"] += 1
    elif fault == "rgb":
        (path.parent / "right.png").write_bytes(b"changed")
    else:
        labels = dict(receipt_sha256="a"*64, source="manual_rgb_annotation", observations=[])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        audit.audit_frame(path, cal, labels)


def test_audits_consecutive_physics_boundaries_and_keeps_clocks_distinct(tmp_path):
    cal, first, _ = write_frame(tmp_path / "f0")
    _, second, _ = write_frame(tmp_path / "f1", tick=1, epoch=84)
    a, b = [audit.audit_frame(p, cal) for p in (first, second)]
    result = audit.audit_sequence([a, b])
    assert result["intervals"] == 1
    assert result["physical_delta_ms"]["max"] == pytest.approx(1000/60)
    assert result["wall_capture_latency_ms"] is None
    assert result["complete_episodes_verified"] == 0
    assert result["actions_verified"] == 0

    _, stale, _ = write_frame(tmp_path / "f2", tick=1, epoch=80)
    with pytest.raises(ValueError, match="renderer epoch"):
        audit.audit_sequence([a, audit.audit_frame(stale, cal)])
    _, skipped, _ = write_frame(tmp_path / "f3", tick=2, epoch=88)
    with pytest.raises(ValueError, match="policy"):
        audit.audit_sequence([a, audit.audit_frame(skipped, cal)])
