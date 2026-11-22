"""Regression coverage for the causal, specimen-independent grasp feedback."""
from copy import deepcopy

import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_wire_grasp_feedback import pack_wire_grasp_feedback


def sample():
    clock = dict(run_uuid="run", env_id=0, episode_uid="episode", reset_id=1,
                 rod_uid="rod", physics_step=10, rod_tick=96, timestamp_s=10/480,
                 solver_time_s=8/480, clock_valid=True)
    hand = dict(xyz_in_palm_m=[0., .012, -.016], tangent_in_palm=[0., 1., 0.],
                reaction={"finger14_link": dict(minimum_normal_n=0., mean_normal_n=.2,
                                                mean_force_on_hand_palm_n=[0., 0., -.1]),
                          "finger23_link": dict(minimum_normal_n=0., mean_normal_n=0.,
                                                mean_force_on_hand_palm_n=[0., 0., 0.])})
    return dict(clock=clock, measured_hands={side:deepcopy(hand) for side in ("left", "right")},
                reaction_valid=True, readback={"weight_n": .4}, spec_sha256="old")


def test_grasp_feedback_uses_palm_geometry_and_measured_force_without_spec_identity():
    sample_a = sample()
    a = pack_wire_grasp_feedback(sample_a, sample_a["clock"])
    sample_b = deepcopy(sample_a)
    sample_b["spec_sha256"] = "held-out"
    sample_b["readback"].update(axial_stiffness_n=960., material="unseen", weight_n=.5)
    assert a == pack_wire_grasp_feedback(sample_b, sample_b["clock"])
    assert a["position_in_palm_m"][0] == [0., .012, -.016]
    assert a["reaction_valid"][0][:3] == [True, True, False]
    np.testing.assert_allclose(a["reaction_force_n"][0][0], [0., .2, 0., 0., -.1])
    assert a["reaction_force_n"][0][1] == [0.]*5
    sample_b["measured_hands"]["left"]["xyz_in_palm_m"][1] += .002
    sample_b["measured_hands"]["right"]["reaction"]["finger14_link"]["mean_normal_n"] *= 2
    b = pack_wire_grasp_feedback(sample_b, sample_b["clock"])
    assert b["position_in_palm_m"][0][1] == pytest.approx(.014)
    assert b["reaction_force_n"][1][0][1] == pytest.approx(.4)


def test_grasp_feedback_rejects_stale_or_cross_reset_sources_and_masks_invalid_reaction():
    data = sample()
    for key in ("physics_step", "rod_tick", "reset_id"):
        clock = dict(data["clock"], **{key: data["clock"][key] + 1})
        if key == "physics_step":
            clock["timestamp_s"] = clock["physics_step"]/480
        with pytest.raises(ValueError, match="same completed"):
            pack_wire_grasp_feedback(data, clock)
    data["reaction_valid"] = False
    a = pack_wire_grasp_feedback(data, data["clock"])
    assert not np.any(a["reaction_valid"])
    assert not np.any(a["reaction_force_n"])
    assert a["position_in_palm_m"][0] == [0., .012, -.016]


def test_grasp_feedback_rejects_nonfinite_trusted_reaction():
    data = sample()
    data["measured_hands"]["left"]["reaction"]["finger14_link"]["mean_normal_n"] = float("nan")
    with pytest.raises(ValueError, match="finite measured"):
        pack_wire_grasp_feedback(data, data["clock"])

def test_distal_and_power_sources_use_same_canonical_row():
    data = sample()
    distal = deepcopy(data)
    for hand in distal["measured_hands"].values():
        hand["reaction"] = {name: pad for name, pad in zip(
            ("thumb", "index"), hand["reaction"].values(), strict=True)}
    expected = pack_wire_grasp_feedback(data, data["clock"])
    assert pack_wire_grasp_feedback(distal, distal["clock"]) == expected
    distal["measured_hands"]["left"]["reaction"]["finger14_link"] = (
        data["measured_hands"]["left"]["reaction"]["finger14_link"])
    with pytest.raises(ValueError, match="unique canonical"):
        pack_wire_grasp_feedback(distal, distal["clock"])

def test_grasp_feedback_rejects_empty_valid_hand_source():
    data = sample()
    data["measured_hands"]["left"]["reaction"] = {}
    with pytest.raises(ValueError, match="missing measured reaction"):
        pack_wire_grasp_feedback(data, data["clock"])
