"""Guarded closed-loop wrist re-seat with no changes to the index action."""

import numpy as np

from go2_pvcnn.control.dh116.tabletop_wire_live_opposition import (
    ACTION_CONTRACT, DT, choose_thumb_reseat,
)


def test_thumb_reseat_uses_current_clearance_and_defers_index_closure():
    calls = []

    def current_gap(delta):
        return 3e-6 - delta[0]

    def preview(action):
        calls.append(action.copy())
        return {"passed": np.linalg.norm(action) < .2}

    result = choose_thumb_reseat(current_gap, preview)
    assert result["reason"] == "thumb_wrist_reseat"
    # The full 5 micrometres is rejected. The smaller candidate improves gap.
    np.testing.assert_allclose(result["action"][12:15] *
        ACTION_CONTRACT["translation_rate_m_s"] * DT, [2.5e-6, 0., 0.])
    assert len(calls) == 2
    assert not np.any(result["action"][np.r_[:12, 15:24]])
    assert result["predicted_gap_m"] < result["gap_m"]


def test_thumb_reseat_holds_for_invalid_or_non_improving_geometry_and_projection():
    for current_gap, preview in (
        (lambda delta: None, lambda action: {"passed": True}),
        (lambda delta: -3e-6, lambda action: {"passed": True}),
        (lambda delta: 3e-6 - delta[0], lambda action: {"passed": False}),
    ):
        result = choose_thumb_reseat(current_gap, preview)
        assert result["reason"] == "thumb_reseat_hold"
        assert not np.any(result["action"])

def test_translation_only_is_an_explicit_unclamped_opposition_candidate():
    from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import CONTRACT
    from go2_pvcnn.control.dh116.tabletop_wire_live_opposition import compose_projected_pivot

    seed = np.array([[[0., 0., .5, 1., 0., 0., 0.], [.2, 0., .5, 1., 0., 0., 0.]]])
    command = np.zeros(24)
    command[12] = .3
    offsets = np.zeros((1, 2, 3))
    kwargs = dict(root_seed_w=seed, measured_base_w=seed.copy(), clearance_m=np.full((1, 2), .2),
                  palm_offsets_m=offsets, angles_rad=np.zeros((1, 2, 3)))
    assert compose_projected_pivot(command, **kwargs) is None
    _, translations = compose_projected_pivot(command, **kwargs, allow_translation_only=True)
    np.testing.assert_allclose(translations[-1], [command[12] * ACTION_CONTRACT["translation_rate_m_s"] * DT, 0., 0.])
    offsets[0, 0, 0] = CONTRACT["palm_upper_m"][0]
    kwargs["measured_base_w"][:, :, :3] += offsets
    assert compose_projected_pivot(command, **kwargs, allow_translation_only=True) is None
