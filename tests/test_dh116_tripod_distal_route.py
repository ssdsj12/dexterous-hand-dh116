import numpy as np
import pytest

from scripts.certify_dh116_tripod_distal_route import prove_segment


def test_rejects_an_unsafe_midpoint_between_clear_endpoints():
    def margins(t):
        return {"rod": abs(t - .5) - .05, "support": .1}
    proof = prove_segment(margins, {"rod": 1., "support": 0.}, max_samples=64)
    assert proof["proved"] is False and proof["violations"][0]["margins"]["rod"] < 0.


def test_covers_whole_interval_with_positive_displacement_reserve():
    def margins(t):
        return {"rod": .025 + .1 * t, "table": .020}
    proof = prove_segment(margins, {"rod": .10, "table": .02}, max_samples=64)
    assert proof["proved"] is True
    assert min(box["minimum_margin_m"] for box in proof["certified_boxes"]) > 0.
    assert sum(np.asarray(box["upper"])[0] - np.asarray(box["lower"])[0]
               for box in proof["certified_boxes"]) == pytest.approx(1.)
