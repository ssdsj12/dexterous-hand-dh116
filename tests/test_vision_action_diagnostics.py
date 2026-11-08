import json
import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_vision_action_diagnostics import (
    PhysicsActionDiagnosticWriter, VisionActionDiagnosticRecorder,
    relative_palm_pose_m, reset_relative_world_base, split_action24,
    summarize_action24)


def test_reset_relative_world_base_adds_task_offset_only_to_z():
    root_seed = np.array([[[0.11, 0.22, 0.33], [0.44, 0.55, 0.66]]])
    offset = np.array([0.004])
    result = reset_relative_world_base(root_seed, offset)
    expected = root_seed.copy()
    expected[:, :, 2] += offset[:, None]
    np.testing.assert_allclose(result, expected)


def test_relative_palm_pose_uses_fixed_episode_reset_baseline():
    reset_pose = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
    actual_pose = reset_pose.copy()
    actual_pose[:, :, 2] += 0.02
    np.testing.assert_allclose(relative_palm_pose_m(actual_pose, reset_pose),
                               np.array([[[0., 0., .02], [0., 0., .02]]]))


def test_split_action24_preserves_left_right_contract():
    action = np.arange(48, dtype=np.float64).reshape(2, 24) / 48.
    parts = split_action24(action)
    assert np.array_equal(parts["left_finger6"], action[:, :6])
    assert np.array_equal(parts["right_finger6"], action[:, 6:12])
    assert np.array_equal(parts["left_palm_translation3"], action[:, 12:15])
    assert np.array_equal(parts["right_palm_translation3"], action[:, 15:18])
    assert np.array_equal(parts["left_rotation3"], action[:, 18:21])
    assert np.array_equal(parts["right_rotation3"], action[:, 21:24])


def test_summary_reports_per_hand_saturation():
    action = np.zeros((2, 24), dtype=np.float64)
    action[0, 12:15] = 1.
    action[1, 15:18] = -1.
    result = summarize_action24(action)
    assert result["left"]["palm_translation3"]["saturation_rate"] == [0.5, 0.5, 0.5]
    assert result["right"]["palm_translation3"]["saturation_rate"] == [0.5, 0.5, 0.5]


def test_recorder_keeps_env_episode_timestamp_and_projection():
    n = 2
    recorder = VisionActionDiagnosticRecorder()
    action = np.zeros((n, 24), dtype=np.float64)
    zeros = np.zeros((n, 2, 3), dtype=np.float64)
    projected = zeros.copy(); projected[1, 1, 2] = .004
    out = recorder.record(
        env_id=[3, 9], episode_id=[11, 12], physics_step=[100, 101], timestamp_s=[.2, .202],
        raw_action24=action, requested_palm_target_m=zeros,
        pre_projection_palm_target_m=zeros, projected_palm_target_m=projected,
        final_sdk_palm_target_m=projected, actual_palm_pose_m=zeros,
        actual_palm_velocity_m_s=zeros, table_clearance_m=np.ones((n, 2)),
        safe_z_m=np.zeros((n, 2)), projection_reasons=[None, "table_clearance"])
    assert len(recorder.rows) == 2
    assert out["rows"][1]["env_id"] == 9
    assert out["rows"][1]["episode_id"] == 12
    assert out["rows"][1]["projection_reasons"] == "table_clearance"
    assert out["rows"][0]["palm_displacement_m"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert out["projection"]["projection_rate"][1][2] == .5
    assert recorder.summary()["by_hand"]["right"]["sdk_delta_abs_mean"][2] == 0.


def test_physics_writer_keeps_terminal_sample_authoritative(tmp_path):
    path = tmp_path / "physics.jsonl"
    n = 1
    zeros23 = np.zeros((n, 2, 3), dtype=np.float64)
    action = np.zeros((n, 24), dtype=np.float64)
    audit = dict(
        raw_action24=action,
        requested_palm_target_m=zeros23,
        pre_projection_palm_target_m=zeros23,
        projected_palm_target_m=zeros23,
        final_sdk_palm_target_m=zeros23,
        actual_palm_pose_m=zeros23,
        palm_displacement_m=zeros23,
        actual_palm_velocity_m_s=zeros23,
        table_clearance_m=np.zeros((n, 2)),
        safe_z_m=np.zeros((n, 2)),
        contact_loads_n=np.zeros((n, 2)),
        contact_valid=np.zeros((n, 2), dtype=bool),
        projection_reasons=[0],
    )
    with path.open("w", encoding="utf-8") as stream:
        writer = PhysicsActionDiagnosticWriter(stream)
        writer.record_batch([0], [4], 7, [10], [0.1], [True], [False], audit)
        writer.record_batch([0], [4], 7, [11], [0.2], [True], [True], audit)
        writer.record_batch([0], [4], 7, [12], [0.3], [False], [False], audit)
        assert writer.terminal_rows == 1
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[1]["terminal_sample"] is True
    assert rows[2]["post_terminal_noop"] is True
    assert rows[2]["terminal_sample"] is False


def test_physics_writer_episode_uid_changes_after_repeated_reset(tmp_path):
    path = tmp_path / "physics.jsonl"
    zeros23 = np.zeros((1, 2, 3), dtype=np.float64)
    audit = dict(
        raw_action24=np.zeros((1, 24)), requested_palm_target_m=zeros23,
        pre_projection_palm_target_m=zeros23, projected_palm_target_m=zeros23,
        final_sdk_palm_target_m=zeros23, actual_palm_pose_m=zeros23,
        palm_displacement_m=zeros23, actual_palm_velocity_m_s=zeros23,
        table_clearance_m=np.zeros((1, 2)), safe_z_m=np.zeros((1, 2)),
        contact_loads_n=np.zeros((1, 2)), contact_valid=np.zeros((1, 2), dtype=bool),
        projection_reasons=[0])
    with path.open("w", encoding="utf-8") as stream:
        writer = PhysicsActionDiagnosticWriter(stream)
        writer.record_batch([0], [0], 0, [1], [0.1], [True], [False], audit)
        writer.begin_reset()
        writer.record_batch([0], [0], 0, [2], [0.2], [True], [False], audit)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["episode_uid"] != rows[1]["episode_uid"]


def test_physics_writer_rejects_nonmonotonic_physical_timestamps(tmp_path):
    path = tmp_path / "physics.jsonl"
    zeros23 = np.zeros((1, 2, 3), dtype=np.float64)
    audit = dict(
        raw_action24=np.zeros((1, 24)), requested_palm_target_m=zeros23,
        pre_projection_palm_target_m=zeros23, projected_palm_target_m=zeros23,
        final_sdk_palm_target_m=zeros23, actual_palm_pose_m=zeros23,
        palm_displacement_m=zeros23, actual_palm_velocity_m_s=zeros23,
        table_clearance_m=np.zeros((1, 2)), safe_z_m=np.zeros((1, 2)),
        contact_loads_n=np.zeros((1, 2)), contact_valid=np.zeros((1, 2), dtype=bool),
        projection_reasons=[0])
    with path.open("w", encoding="utf-8") as stream:
        writer = PhysicsActionDiagnosticWriter(stream)
        writer.record_batch([0], [0], 0, [2], [0.2], [True], [False], audit)
        with pytest.raises(ValueError, match="monotonic"):
            writer.record_batch([0], [0], 0, [1], [0.1], [True], [False], audit)


@pytest.mark.parametrize("bad", [np.zeros((2, 23)), np.full((2, 24), np.nan)])
def test_bad_action_rejected(bad):
    with pytest.raises(ValueError):
        split_action24(bad)
