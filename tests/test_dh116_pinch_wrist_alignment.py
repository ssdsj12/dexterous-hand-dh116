import numpy as np
import pytest

from scripts import run_dh116_pinch_wrist_alignment as alignment


def test_alignment_uses_wrist_rate_balls_and_keeps_finger_targets():
    old = np.zeros((13, 24))
    state = {s: (np.zeros(12), np.zeros(12)) for s in ("left", "right")}
    offsets = {"left": [0.00034, 0., -.00012, 0., -.002, 0.],
               "right": [0., -.0004, -.00012, 0., 0., 0.]}
    actions, targets = alignment.compile_alignment(old, state, offsets)
    np.testing.assert_array_equal(actions[:13], old)
    np.testing.assert_array_equal(actions[13:, :12], 0.)
    for sl in alignment.prelift.VECTORS:
        assert np.max(np.linalg.norm(actions[13:, sl], axis=-1)) <= 1. + 1e-12
    target = alignment.prelift.global_channel_order(targets["left"], targets["right"])
    np.testing.assert_allclose(actions[13:].sum(0) * alignment.prelift.WINDOW_SCALE, target, atol=1e-14)
    assert not np.any(actions[-1])


def test_alignment_rejects_arm_path_beyond_frozen_budget():
    state = {s: (np.zeros(12), np.zeros(12)) for s in ("left", "right")}
    bad = {"left": [.01, 0., 0., 0., 0., 0.], "right": np.zeros(6)}
    with pytest.raises(ValueError, match="frozen alignment budget"):
        alignment.compile_alignment(np.zeros((13, 24)), state, bad)

def test_index_pacing_live_gate_roundtrips_with_a_signed_suffix(monkeypatch):
    pacing = alignment.pacing
    low, high = pacing.prelift.active_limits()
    q = np.r_[(low + high) / 2, np.zeros(6)]
    synthetic = {side: (q, q.copy()) for side in ("left", "right")}
    recipe, _ = pacing.compile_index_pacing(np.zeros((7, 24)), synthetic)
    extended = np.vstack((recipe, np.zeros((alignment.ALIGN_WINDOWS, 24))))
    plan = dict(action_schedule=extended.tolist(), index_pacing=dict(prefix={}))
    monkeypatch.setattr(pacing.late, "prefix_measurements", lambda *args, **kw: {})
    monkeypatch.setattr(pacing.late, "compare_prefix", lambda *args: {"passed": True})
    monkeypatch.setattr(pacing.prelift, "joint_and_wrist", lambda row, side: synthetic[side])
    monkeypatch.setattr(pacing, "certify_pacing",
                        lambda *values: {s: {"proved": True} for s in synthetic})
    def owned(*values, **kwargs):
        row = {"physics_step": pacing.pair.LAST_SDK, "guard_failure": None}
        kwargs["observe"](row)
        return row
    monkeypatch.setattr(pacing.pair, "acquire", owned)
    row = pacing.acquire(None, None, None, None, plan=plan, sdk_steps=alignment.LAST_SDK)
    assert row["guard_failure"] is None
    assert row["index_pacing_gate"]["passed"]
