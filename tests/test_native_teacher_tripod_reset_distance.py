"""Saved SDK2 pose buffers plus explicit CPU parameter fixtures, not a native rerun."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from test_native_teacher_context import backend_profile
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import profile_reader_scope

ROOT = Path(__file__).resolve().parents[1]
RESET = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/ref902/execution-closure-native-context-fixed/native/tripod_reset_source_1.json")
RESET_SHA = "1afdf3954fd98871b9735f4168d412e69de0b56cb770417ad17b1882c4825da4"


@pytest.fixture
def reset_case(backend_profile):
    from test_native_teacher_model import raw_cpu_packets
    raw = RESET.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RESET_SHA
    reset = decode_owned(json.loads(raw))["reset"]
    _, _, parameters, _ = raw_cpu_packets(backend_profile)
    parameters = deepcopy(parameters)
    # Parameter data is explicitly CPU-scoped, never a retagged native getter.
    assert parameters["source_scope"] == "cpu_fixture"
    clock = deepcopy(reset["after_clock"])
    parameters["identity"] = clock
    from go2_pvcnn.control.dh116.tabletop_deformx_parameter_snapshot import validate_snapshot
    parameters["raw_native_parameters"]["time_s"] = clock["solver_time_s"]
    parameters["native_parameters"] = validate_snapshot(
        parameters["raw_native_parameters"], expected_num_envs=1, expected_nodes=41,
        expected_time_s=clock["solver_time_s"])
    parameters["node_positions_m"] = reset["rod_reset"]["node_positions"].copy()
    parameters["reward_radius_sample"]["source"]["identity"] = deepcopy(clock)
    return reset, parameters


def test_tripod_reset_has_six_real_pad_queries_at_sdk2_without_promotion(reset_case, backend_profile):
    module = importlib.import_module("go2_pvcnn.control.dh116.tabletop_tripod_reset_distance")
    reader = module.MeasuredTripodResetApproach(ROOT, scope="cpu_fixture")
    reset, parameters = reset_case
    with profile_reader_scope(backend_profile):
        result = reader.capture_reset(reset, parameters)
    assert result["version"] == "dh116-measured-tripod-reset-distance-v1"
    assert result["source_scope"] == "cpu_fixture" and result["reset_baseline_available"]
    expected = {
        "left": [.002395946066826582, .001586157362908125, .0018587964586913586],
        "right": [.00667161587625742, .009878989309072495, .012432756833732128]}
    for side in ("left", "right"):
        row = result["per_hand"][side]
        assert row["required_pads"] == ["finger14_link", "finger23_link", "finger33_link"]
        assert [row["pad_samples"][name]["minimum_gap_m"] for name in row["required_pads"]] == pytest.approx(expected[side], abs=1e-12)
        assert row["positive_gap_mean_m"] == pytest.approx(sum(expected[side])/3, abs=1e-12)
    identity = result["reset_approach"]["sample_identity"]
    assert identity["physics_step"] == 2 and identity["rod_tick"] == 0
    assert result["new_physics_steps"] == result["ppo_updates"] == 0
    assert result["teacher_admission"] is result["r0_native_reset_complete"] is False
    assert result["coupling_mesh_reset_qualified"] is False
    assert result["observed_mesh_count"] == 11  # Historical Reset distance, not a new17-slot Reset claim.
    assert result["contact_observed"] is None and not result["contact_inferred_from_geometry"]


def test_tripod_reset_refuses_wrong_pose_copy_time_or_radius_identity(reset_case, backend_profile):
    module = importlib.import_module("go2_pvcnn.control.dh116.tabletop_tripod_reset_distance")
    reader = module.MeasuredTripodResetApproach(ROOT, scope="cpu_fixture")
    reset, parameters = reset_case
    # One immutable physical fixture covers distinct source boundaries; no rebuild per fault.
    for damage, message in (("pose_copy", "pose"), ("clock", "clock"), ("radius_identity", "identit"),
                            ("parameter_source", "source")):
        row, packet = deepcopy(reset), deepcopy(parameters)
        if damage == "pose_copy":
            row["pad_poses_xyzw"]["right.finger33_link"][0][0] += .001
        elif damage == "clock":
            row["after_actual_clocks"]["sdk_physics_step"] += 1
        elif damage == "radius_identity":
            packet["reward_radius_sample"]["source"]["identity"]["episode_uid"] = "foreign"
        else:
            packet["sources"]["backend_sha256"] = "0"*64
        with profile_reader_scope(backend_profile), pytest.raises(ValueError, match=message):
            reader.capture_reset(row, packet)


def test_tripod_reset_missing_middle_is_unknown_and_other_grasp_is_refused(reset_case, backend_profile):
    module = importlib.import_module("go2_pvcnn.control.dh116.tabletop_tripod_reset_distance")
    reader = module.MeasuredTripodResetApproach(ROOT, scope="cpu_fixture")
    reset, parameters = reset_case
    row = deepcopy(reset)
    hand = row["hands"]["left"]
    i = hand["body_names"].index("finger33_link")
    hand["body_poses_xyzw"][0, i, 0] += .2
    row["pad_poses_xyzw"]["left.finger33_link"] = (
        hand["body_poses_xyzw"][0, i, :3].copy(), hand["body_poses_xyzw"][0, i, 3:].copy())
    with profile_reader_scope(backend_profile):
        result = reader.capture_reset(row, parameters)
    assert result["auxiliary"]["samples"]["approach_distance_m"]["value"] is None
    assert result["reset_baseline_available"] is False
    assert result["per_hand"]["left"]["pad_samples"]["finger33_link"]["minimum_gap_m"] is None
    with pytest.raises(ValueError, match="Tripod"):
        reader.capture_reset(reset, parameters, grasp="pinch")


def test_unknown_parameter_scope_cannot_be_treated_as_a_native_getter(reset_case, backend_profile):
    module = importlib.import_module("go2_pvcnn.control.dh116.tabletop_tripod_reset_distance")
    reader = module.MeasuredTripodResetApproach(ROOT, scope="owner_source_unqualified")
    reset, parameters = reset_case
    parameters["source_scope"] = "unrecognized_not_native"
    with profile_reader_scope(backend_profile), pytest.raises(ValueError, match="scope"):
        reader.capture_reset(reset, parameters)
