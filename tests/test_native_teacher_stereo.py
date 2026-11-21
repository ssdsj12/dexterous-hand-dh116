"""CPU evidence fixtures for the native Teacher to Student camera join."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from test_teacher_stereo_alignment import scene
from go2_pvcnn.control.dh116.tabletop_native_teacher_stereo import (
    capture_identity, save_aligned_rgb_pair)


def fixture():
    cal, state, packet, _ = scene()
    clock = dict(run_uuid="teacher", env_id=0, episode_uid="episode", reset_id=2,
                 rod_uid="rod", physics_step=2, rod_tick=0, policy_step=0, timestamp_s=2/480,
                 phase="pre_action_before_target_write", clock_valid=True, solver_time_s=0.0)
    values = np.full(371, np.nan)
    valid = np.zeros(371, dtype=bool)
    values[0], valid[0] = .2, True
    pre = dict(common=dict(identity=clock, values=values, valid=valid,
                           source_packets=[dict(side=side, identity=clock, body_link_pose_w=pose)
                               for side, pose in zip(("left", "right"), state["palm_poses_world_xyz_wxyz"])]),
              rod_boundary=dict(run_uuid="teacher", rod_uid="rod", rod_tick=0,
                  rod_snapshot=dict(env_ids=[0], episode_uids=["episode"], reset_ids=[2], rod_uids=["rod"],
                                    rod_tick=0, physics_step=2, timestamp_s=2/480,
                                    node_positions=torch.tensor(state["rod_nodes_world_m"])[None])))
    clock = dict(frame_id=7, render_frame_number=7, simulation_time=2/480)
    sdk_stamp = 0.004166666883975267
    packet["render_timing"].update(capture_id=1, raw_left_clock=clock, raw_right_clock=dict(clock),
                                  physical_timestamp_s=sdk_stamp)
    packet.update(identity=capture_identity(pre, cal.sha256, sdk_timestamp_s=sdk_stamp), source_scope="cpu_fixture",
                  left_rgb=torch.zeros(1, 3, cal.height, cal.width),
                  right_rgb=torch.ones(1, 3, cal.height, cal.width))
    return cal, pre, packet


def test_saves_same_tick_stereo_geometry_and_preserves_unknown_privileged_fields(tmp_path):
    import json

    cal, pre, packet = fixture()
    label = save_aligned_rgb_pair(pre, packet, cal, tmp_path)
    saved = json.loads((tmp_path / label["receipt_path"]).read_text())
    assert saved["identity"]["episode_uid"] == "episode"
    assert saved["privileged_values"][:2] == [.2, None]
    assert saved["privileged_valid"][:2] == [True, False]
    assert saved["cameras"]["left"]["rod_pixels_uv"][0] == pytest.approx([320, 240], abs=1e-4)
    assert saved["rod_nodes_palm_m"]["left"][0] == pytest.approx([0, 0, 0], abs=1e-6)
    assert saved["identity"]["sdk_timestamp_s"] != saved["identity"]["simulation_timestamp"]
    assert saved["native_qualification"] is False
    assert label["state_sha256"] == packet["identity"]["state_sha256"]
    assert len(list((tmp_path / "frame-000001").glob("*.png"))) == 2


@pytest.mark.parametrize("fault", ["clock", "rod", "frame", "source"])
def test_mismatched_source_never_produces_an_aligned_label(tmp_path, fault):
    cal, pre, packet = fixture()
    pre = deepcopy(pre)
    if fault == "clock":
        packet["render_timing"]["physical_step"] += 1
    elif fault == "rod":
        pre["rod_boundary"]["rod_snapshot"]["reset_ids"] = [1]
    elif fault == "frame":
        packet["render_timing"]["raw_right_clock"]["render_frame_number"] += 1
    else:
        pre["common"]["source_packets"][1]["identity"]["episode_uid"] = "other"
    with pytest.raises(ValueError):
        save_aligned_rgb_pair(pre, packet, cal, tmp_path)
    assert not list(tmp_path.iterdir())

@pytest.mark.parametrize("delta_s,usable", [(0., True), (.00199, True), (.00201, False), (-.003, False)])
def test_frame_timing_flags_are_saved_with_raw_evidence(tmp_path, delta_s, usable):
    import json

    cal, pre, packet = fixture()
    timing = packet["render_timing"]
    for side in ("raw_left_clock", "raw_right_clock"):
        timing[side]["simulation_time"] = timing["physical_timestamp_s"] + delta_s
    result = save_aligned_rgb_pair(pre, packet, cal, tmp_path)
    receipt = json.loads((tmp_path/result["receipt_path"]).read_text())
    assert receipt["student_sync_usable"] is result["student_sync_usable"] is usable
    assert receipt["rgb_sdk_delta_ms"] == pytest.approx(delta_s*1000)
    assert receipt["student_sync_rejection"] == (
        None if usable else "rgb_sdk_skew_at_or_above_2ms")
    assert (tmp_path/result["receipt_path"]).with_name("right.png").is_file()


@pytest.mark.parametrize("swap,verified", [(False, True), (True, False)])
def test_hand_order_rig_binds_actual_palms_and_retains_rejected_evidence(tmp_path, swap, verified):
    import json
    from pathlib import Path
    from go2_pvcnn.control.dh116.tabletop_teacher_stereo_alignment import student_camera_calibration

    _, pre, packet = fixture()
    cal = student_camera_calibration(Path(__file__).resolve().parents[1], "hand_order")
    palms = [[.53, .13, .43, 1., 0., 0., 0.], [.47, -.13, .45, 1., 0., 0., 0.]]
    if swap:
        palms.reverse()
    for source, pose in zip(pre["common"]["source_packets"], palms, strict=True):
        source["body_link_pose_w"] = pose
    for side in ("left", "right"):
        packet[f"K_{side}"] = getattr(cal, f"K_{side}").numpy()[None]
        packet[f"T_world_{side}"] = getattr(cal, f"T_world_{side}").numpy()[None]
    packet["calibration_sha256"] = cal.sha256
    packet["identity"] = capture_identity(
        pre, cal.sha256, sdk_timestamp_s=packet["render_timing"]["physical_timestamp_s"])
    label = save_aligned_rgb_pair(pre, packet, cal, tmp_path, camera_profile="hand_order")
    receipt = json.loads((tmp_path / label["receipt_path"]).read_text())
    assert receipt["student_sync_usable"] is True
    assert label["student_hand_order_usable"] is verified
    assert receipt["student_hand_order_usable"] is verified
    assert (tmp_path / label["receipt_path"]).with_name("left.png").is_file()
    for side in ("left", "right"):
        palms_uv = receipt["cameras"][side]["physical_palm_pixels_uv"]
        assert ((palms_uv["left"][0] < palms_uv["right"][0]) is verified)
