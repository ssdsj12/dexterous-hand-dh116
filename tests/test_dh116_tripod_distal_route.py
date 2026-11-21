import numpy as np
import pytest

from scripts.certify_dh116_tripod_distal_route import prove_segment, support_clearance_lower


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


def test_uses_convex_separation_when_world_aabbs_overlap():
    from scipy.spatial.transform import Rotation

    local = np.array([[x,y,z] for x in (-1.,1.) for y in (-.1,.1) for z in (-.1,.1)])
    vertices=Rotation.from_euler("z",45,degrees=True).apply(local)
    low=np.array([.5,-.6,-.05])
    high=np.array([.6,-.5,.05])
    assert np.all(vertices.max(0)>low) and np.all(vertices.min(0)<high)
    assert support_clearance_lower(vertices,low,high) > .5
    assert support_clearance_lower(vertices,-.02*np.ones(3),.02*np.ones(3)) < 0
