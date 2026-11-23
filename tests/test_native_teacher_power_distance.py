import numpy as np
import pytest


def test_power_pose_follows_the_same_material_station_in_world_space():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_power_distance import power_pose_error

    initial = np.column_stack((np.zeros(41), np.arange(41) * .0125, np.ones(41)))
    targets = np.array([[.01, .1, .99], [.02, .3875, 1.02]])
    positions = targets.copy()
    positions[0, 2] += .003
    shift = np.array([.2, -.3, .1])
    distances, stations = power_pose_error(initial, targets, initial, positions)
    translated, shifted_stations = power_pose_error(initial, targets, initial + shift, positions + shift)
    assert stations == shifted_stations == [8, 31]
    assert distances == pytest.approx([.003, 0.])
    assert translated == pytest.approx(distances)


def test_power_pose_error_rejects_incomplete_measured_rod():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_power_distance import power_pose_error

    initial = np.zeros((41, 3))
    wrist_targets = np.zeros((2, 3))
    with pytest.raises(ValueError, match="41"):
        power_pose_error(initial, wrist_targets, initial[:40], wrist_targets)



def test_power_source_reads_validity_from_the_rod_record():
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import IDENTITY, encode_owned
    from go2_pvcnn.control.dh116.tabletop_native_teacher_power_distance import NativePowerPoseApproach
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import digest
    from go2_pvcnn.control.dh116.tabletop_power_shared_rod import PROXIMAL_POWER_MESH_NAMES
    from go2_pvcnn.control.dh116.tabletop_training_reward import SourceFailure

    metric = object.__new__(NativePowerPoseApproach)
    metric.scope = "cpu_fixture"
    metric.reference_nodes = np.zeros((41, 3))
    metric.targets = np.zeros((2, 3))
    metric.goal_sha = metric.task_sha = "fixture"
    bindings = {"left.base_link": {"prim_path": "/leftHand/base_link"}}
    metric.mesh_sha = digest(encode_owned(bindings))
    rod = dict.fromkeys(IDENTITY, 0)
    rod.update(clock_valid=True, node_positions=np.zeros((1, 41, 3)),
               mesh_names=PROXIMAL_POWER_MESH_NAMES, mesh_bindings=bindings)
    packet = metric._metric(rod, metric.targets, binding="fixture")
    assert packet["auxiliary"]["samples"]["approach_distance_m"]["value"] == [0., 0.]
    assert packet["contact_observed"] is None
    rod["clock_valid"] = False
    with pytest.raises(SourceFailure, match="same-clock"):
        metric._metric(rod, metric.targets, binding="fixture")



def test_power_reward_accepts_native_float32_clock_but_rejects_a_skipped_sdk_step():
    from go2_pvcnn.control.dh116.tabletop_training_reward import SourceStamp, _solver_interval_matches

    def stamp(tick, seconds):
        return SourceStamp("power-episode", tick, seconds, "unqualified_source",
                           "policy_owned_power_endpoint_v1", env_id=0, reset_uid="1", rod_uid="rod0")

    previous = stamp(17, 0.03125)
    current = stamp(18, 0.03333333507180214)
    skipped = stamp(19, 0.03541666641831398)
    assert _solver_interval_matches(previous, current, native_smoke=True)
    assert not _solver_interval_matches(previous, skipped, native_smoke=True)
