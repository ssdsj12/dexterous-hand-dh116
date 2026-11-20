import numpy as np

from scripts import run_dh116_pinch_index_pacing as pacing


def test_tracking_domain_contains_opposite_axis_commands_and_previous_catchup():
    actual = np.zeros(12)
    prior = actual.copy()
    prior[[1, 2, 6, 8, 10]] = [.002, -.003, .001, -.0002, .009]
    target = prior.copy()
    target[[1, 2, 6, 8, 10]] = [-.004, 0., -.004, -.0003, .001]
    reference, domain = pacing.tracking_domain(actual, prior, target)
    low, high = domain * (reference - actual)[pacing.TRACKED]
    lo, hi = np.minimum(low, high), np.maximum(low, high)
    for observed in (actual, prior, target):
        assert np.all(observed[pacing.TRACKED] <= hi)
        assert np.all(observed[pacing.TRACKED] >= lo)
    assert lo[0] < 0 < hi[0]
    assert hi[1] == 0 and lo[1] == -.003
    assert hi[2] > 0 and lo[2] < 0
    assert hi[-2] == .009


def test_schedule_rolls_back_right_fingers_once_and_leads_left_index_only():
    previous = np.zeros((7, 24))
    low, high = pacing.prelift.active_limits()
    actual = np.r_[(low + high) / 2, np.zeros(6)]
    source = {
        "left": (actual, actual.copy()),
        "right": (actual, actual + np.r_[0., .001, .002, np.zeros(9)]),
    }
    schedule, targets = pacing.compile_index_pacing(previous, source)
    assert schedule.shape == (13, 24)
    assert targets["left"][2] == actual[2] + pacing.INDEX_LEAD_RAD
    np.testing.assert_array_equal(schedule[:7], previous)
    np.testing.assert_allclose(schedule[7, 7:9], [-.2, -.4])
    np.testing.assert_array_equal(schedule[8:, 7:9], 0.)
    assert np.all(schedule[7:11, 2] == 1.)
    assert not np.any(schedule[12])


def test_geometry_rejects_missing_axis_or_lagged_right_thumb(monkeypatch):
    current = np.zeros(12)
    prior = current.copy()
    prior[1] = .003
    goals = {"left": current, "right": current}
    source = {"left": (current, current), "right": (current, prior)}
    calls = []

    def verify(row, actual, issued, target, envelope, **kwargs):
        calls.append((kwargs["side"], kwargs["independent_wrist"], kwargs["require_pad_clearance"],
                      target[1], np.array(envelope["index_pacing"]).shape))
        return {"index_pacing": {"proved": True}}, None

    proofs = pacing.certify_pacing({"rod": {"node_positions": np.zeros((1, 41, 3))}}, source, goals, prove=verify)
    assert all(p["proved"] for p in proofs.values())
    assert calls == [("left", True, True, 0., (2, 8)), ("right", True, False, .003, (2, 8))]
