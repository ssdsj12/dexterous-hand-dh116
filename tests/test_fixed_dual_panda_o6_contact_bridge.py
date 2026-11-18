from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


EXTERNAL = Path(__file__).resolve().parents[1]/"vendor/DeformX"
pytest.MonkeyPatch().syspath_prepend(str(EXTERNAL))

from co_sim.moving_contact_models import ContactImpulse
from go2_pvcnn.control.fixed_dual_panda_o6.contact_bridge import (
    CHANNEL_BODY_NAMES,
    BodyState,
    ContactReactionBridge,
    compose_articulation_arrays,
    mesh_motions_from_bodies,
    normalized_wxyz_to_rotation,
    reaction_wrenches,
    static_offset_torque,
    validate_pad_manifest,
)


CHANNELS = tuple(CHANNEL_BODY_NAMES)


def state(channel, *, token="sample-1", time=1.0, link=(1, 2, 3), com=(1, 2, 2.9),
          velocity=(0.1, 0.2, 0.3), omega=(0, 0, 2), rotation=None):
    return BodyState(
        body_name=CHANNEL_BODY_NAMES[channel], link_position_w=link,
        rotation_wb=np.eye(3) if rotation is None else rotation,
        physical_com_position_w=com, physical_com_velocity_w=velocity,
        omega_w=omega, sample_time_s=time, sample_token=token,
    )


def states(**changes):
    result = {channel: state(channel) for channel in CHANNELS}
    result.update(changes)
    return result


def impulses(duration=.005):
    return {channel: ContactImpulse(np.array([1., 2., 3.]), np.array([.2, .3, .4]), 4., duration)
            for channel in CHANNELS}


def diagnostics(origin=(.5, .6, .7)):
    return {"integrated_duration_s": .01, "table_label": "table", "evaluation_count": 16, "channels": {channel: {
        "mesh_angular_origin_w": np.asarray(origin),
        "rod_angular_origin_w": -np.asarray(origin),
        "mesh_linear_w": np.array([1., 2., 3.]),
        "rod_linear_w": np.array([-1., -2., -3.]),
        "linear_balance_w": np.zeros(3),
        "angular_origin_balance_w": np.zeros(3),
        "evaluations": 4,
    } for channel in CHANNELS}}


def test_body_state_copies_inputs_and_rejects_invalid_values():
    link = np.array([1., 2., 3.])
    value = state("left_thumb", link=link)
    link[:] = 99
    np.testing.assert_array_equal(value.link_position_w, [1, 2, 3])
    assert not value.link_position_w.flags.writeable
    with pytest.raises(ValueError, match="rotation"):
        state("left_thumb", rotation=np.ones((3, 3)))
    with pytest.raises(TypeError, match="bool"):
        state("left_thumb", time=True)
    with pytest.raises(ValueError, match="finite"):
        state("left_thumb", velocity=(0, np.nan, 0))


def test_measured_float32_quaternion_is_normalized_in_float64_before_rotation():
    q = np.array([.9238795, 0., 0., .3826834], dtype=np.float32) * np.float32(1.000001)
    rotation = normalized_wxyz_to_rotation(q)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
    assert np.linalg.det(rotation) == pytest.approx(1., abs=1e-14)
    with pytest.raises(ValueError, match="norm"):
        normalized_wxyz_to_rotation([0, 0, 0, 0])


def test_mesh_motion_uses_link_pose_but_physical_com_twist_origin():
    offsets = {channel: np.array([.2, 0., 0.]) for channel in CHANNELS}
    motions = mesh_motions_from_bodies(states(), offsets, None, .005)
    motion = motions["left_thumb"]
    np.testing.assert_allclose(motion.position_w, [1.2, 2, 3])
    np.testing.assert_allclose(motion.velocity_w, [.1, .6, .3])
    np.testing.assert_array_equal(motion.acceleration_w, np.zeros(3))
    np.testing.assert_array_equal(motion.alpha_w, np.zeros(3))


def test_mesh_motion_finite_differences_actual_world_twists():
    offsets = {channel: np.zeros(3) for channel in CHANNELS}
    previous = {channel: state(channel, token="old", time=.995,
                               velocity=(0, 0, 0), omega=(0, 0, 1)) for channel in CHANNELS}
    current = {channel: state(channel, token="new", time=1.,
                              velocity=(.01, 0, 0), omega=(0, 0, 2)) for channel in CHANNELS}
    motion = mesh_motions_from_bodies(current, offsets, previous, .005)["left_thumb"]
    np.testing.assert_allclose(motion.acceleration_w, [2, 0, 0])
    np.testing.assert_allclose(motion.alpha_w, [0, 0, 200])


@pytest.mark.parametrize("bad", [True, 0., -1., np.nan, np.inf])
def test_mesh_motion_rejects_invalid_or_stale_dt_without_mutation(bad):
    offsets = {channel: np.zeros(3) for channel in CHANNELS}
    with pytest.raises((TypeError, ValueError)):
        mesh_motions_from_bodies(states(), offsets, None, bad)
    old = {channel: state(channel, token="old", time=.99) for channel in CHANNELS}
    with pytest.raises(ValueError, match="time|dt"):
        mesh_motions_from_bodies(states(), offsets, old, .005)


def test_mesh_motion_rejects_missing_extra_and_swapped_identity():
    offsets = {channel: np.zeros(3) for channel in CHANNELS}
    missing = states(); missing.pop("right_index")
    with pytest.raises(ValueError, match="channels"):
        mesh_motions_from_bodies(missing, offsets, None, .005)
    swapped = states(); swapped["left_thumb"] = state("right_thumb")
    with pytest.raises(ValueError, match="identity"):
        mesh_motions_from_bodies(swapped, offsets, None, .005)


def test_reaction_uses_integrated_world_origin_impulse_at_explicit_com():
    body_positions = {channel: np.array([.1, -.2, .3]) for channel in CHANNELS}
    result = reaction_wrenches(impulses(), diagnostics(), body_positions, .01)
    expected_force = np.array([100., 200., 300.])
    np.testing.assert_allclose(result["left_thumb"].force_w, expected_force)
    np.testing.assert_allclose(
        result["left_thumb"].torque_com_w,
        (np.array([.5, .6, .7]) - np.cross(body_positions["left_thumb"], [1, 2, 3])) / .01,
    )
    np.testing.assert_array_equal(result["left_thumb"].position_w, [.1, -.2, .3])


def test_reaction_rejects_bad_diagnostics_duration_or_mapping():
    positions = {channel: np.zeros(3) for channel in CHANNELS}
    bad = diagnostics(); bad["channels"]["left_thumb"]["linear_balance_w"] = np.ones(3)
    with pytest.raises(ValueError, match="linear balance"):
        reaction_wrenches(impulses(), bad, positions, .01)
    with pytest.raises(ValueError, match="duration"):
        reaction_wrenches(impulses(.02), diagnostics(), positions, .01)
    incomplete = impulses(); incomplete.pop("left_thumb")
    with pytest.raises(ValueError, match="channels"):
        reaction_wrenches(incomplete, diagnostics(), positions, .01)
    bad = diagnostics(); bad["channels"]["left_thumb"]["mesh_linear_w"] = np.zeros(3)
    with pytest.raises(ValueError, match="impulse"):
        reaction_wrenches(impulses(), bad, positions, .01)
    bad = diagnostics(); bad["integrated_duration_s"] = .009
    with pytest.raises(ValueError, match="integrated duration"):
        reaction_wrenches(impulses(), bad, positions, .01)
    bad = diagnostics(); bad["channels"]["left_thumb"]["evaluations"] = 0
    with pytest.raises(ValueError, match="evaluation"):
        reaction_wrenches(impulses(), bad, positions, .01)
    bad = diagnostics(); bad["evaluation_count"] = 15
    with pytest.raises(ValueError, match="evaluation"):
        reaction_wrenches(impulses(), bad, positions, .01)


def test_static_offset_helper_and_rotating_force_shortcut_counterexample():
    np.testing.assert_allclose(static_offset_torque([0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 2, 0]),
                               [0, 0, 3])
    # Two equal impulses at different mesh poses cannot be shifted with the final pose.
    exact_origin = np.cross([1, 0, 0], [0, 1, 0]) + np.cross([0, 1, 0], [1, 0, 0])
    shortcut = np.cross([0, 1, 0], np.array([0, 1, 0]) + np.array([1, 0, 0]))
    assert not np.allclose(exact_origin, shortcut)


class Sink:
    def __init__(self, fail_on=None): self.calls = []; self.fail_on = fail_on
    def submit(self, side, force, torque, position):
        self.calls.append((side, force.copy(), torque.copy(), position.copy()))
        if side == self.fail_on: raise RuntimeError("injected submit")


def test_bridge_consumes_once_after_full_validation_and_submits_each_robot_once():
    sink = Sink(); bridge = ContactReactionBridge(sink)
    result = bridge.publish("batch-1", 1., impulses(), diagnostics(),
                            {channel: np.zeros(3) for channel in CHANNELS}, .01)
    assert result.token == "batch-1" and [call[0] for call in sink.calls] == ["left", "right"]
    assert sink.calls[0][1].shape == (2, 3)
    with pytest.raises(ValueError, match="duplicate|stale"):
        bridge.publish("batch-1", 1., impulses(), diagnostics(),
                       {channel: np.zeros(3) for channel in CHANNELS}, .01)
    assert len(sink.calls) == 2
    bridge.publish("batch-2", 2., impulses(), diagnostics(),
                   {channel: np.zeros(3) for channel in CHANNELS}, .01)
    with pytest.raises(ValueError, match="duplicate"):
        bridge.publish("batch-1", 3., impulses(), diagnostics(),
                       {channel: np.zeros(3) for channel in CHANNELS}, .01)


def test_invalid_publish_has_no_side_effect_and_submit_failure_latches():
    sink = Sink(); bridge = ContactReactionBridge(sink)
    with pytest.raises(ValueError):
        bridge.publish("batch-1", 1., impulses(.02), diagnostics(),
                       {channel: np.zeros(3) for channel in CHANNELS}, .01)
    assert sink.calls == []
    sink.fail_on = "right"
    with pytest.raises(RuntimeError, match="injected"):
        bridge.publish("batch-1", 1., impulses(), diagnostics(),
                       {channel: np.zeros(3) for channel in CHANNELS}, .01)
    with pytest.raises(RuntimeError, match="unusable"):
        bridge.publish("batch-2", 2., impulses(), diagnostics(),
                       {channel: np.zeros(3) for channel in CHANNELS}, .01)


def test_force_gate_is_checked_before_any_sink_side_effect():
    sink = Sink(); bridge = ContactReactionBridge(sink, max_force_n=300.)
    diag = diagnostics(); diag["integrated_duration_s"] = .005
    with pytest.raises(ValueError, match="force gate"):
        bridge.publish("batch-1", 1., impulses(), diag,
                       {channel: np.zeros(3) for channel in CHANNELS}, .005)
    assert sink.calls == []


def test_full_articulation_arrays_are_zero_except_unique_distal_bodies():
    values = reaction_wrenches(impulses(), diagnostics(),
                               {channel: np.zeros(3) for channel in CHANNELS}, .01)
    arrays = compose_articulation_arrays(values, {
        "left": {"left_thumb": 2, "left_index": 4},
        "right": {"right_thumb": 1, "right_index": 5},
    }, {"left": 7, "right": 6})
    assert arrays["left"].force_w.shape == (1, 7, 3)
    assert arrays["right"].position_w.shape == (1, 6, 3)
    np.testing.assert_array_equal(arrays["left"].force_w[0, [0, 1, 3, 5, 6]], 0.)
    np.testing.assert_array_equal(arrays["left"].force_w[0, [2, 4]], [[100, 200, 300]] * 2)
    with pytest.raises(ValueError, match="unique"):
        compose_articulation_arrays(values, {
            "left": {"left_thumb": 2, "left_index": 2},
            "right": {"right_thumb": 1, "right_index": 5},
        }, {"left": 7, "right": 6})


# The original O6 pad-asset integration test is intentionally excluded from
# this DH116-only project. DH116 manifest identity is covered separately in
# test_dh116_contact_runtime.py and test_dh116_contact_meshes.py.
