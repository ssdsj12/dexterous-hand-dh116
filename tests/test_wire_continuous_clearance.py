"""Check the radius-aware fast/certified bounds for actual material segments."""
import importlib
from itertools import product

import numpy as np
import pytest


def module():
    name = "go2_pvcnn.control.dh116.tabletop_wire_continuous_clearance"
    assert importlib.util.find_spec(name), "continuous wire clearance is missing"
    return importlib.import_module(name)


def wire():
    return np.c_[np.zeros(41), np.linspace(-.25, .25, 41), np.zeros(41)][None]


def hull(gap_from_axis, *, azimuth=0.):
    # True closest hull plane is gap_from_axis from the wire axis.
    x, z = np.array([np.cos(azimuth), np.sin(azimuth)]).reshape(2)
    along_y = np.array([0., .001, 0.])
    radial = np.array([x, 0., z])
    tangential = np.array([-z, 0., x])*.0001
    return np.stack([((gap_from_axis+rr*.0001)*radial + yy*along_y + zz*tangential)
                     for rr, yy, zz in product((0, 1), (-1, 1), (-1, 1))])


def test_original_contact_and_clearance_thresholds_apply_at_the_measured_radius():
    a = module().WireClearance(wire(), radius_m=.0045)
    b = module().WireClearance(wire(), radius_m=.005)
    point = hull(.00446)
    assert a.certify(point, selected=True)["passed"]
    assert not a.certify(point, selected=False)["passed"]
    assert not b.certify(point, selected=True)["passed"]
    with pytest.raises(ValueError, match="radius"):
        module().WireClearance(wire(), radius_m=True)


def test_refines_sampling_error_with_a_certified_segment_distance():
    # Between sampled azimuth/elevation directions the pure projection fails
    # to certify 0.25 mm, although the analytic solid distance passes it.
    theta = np.deg2rad(2.5)
    c = module().WireClearance(wire(), radius_m=.0045)
    report = c.certify(hull(.0047501, azimuth=theta), selected=False)
    assert report["passed"]
    assert report["lower_m"] > .00025
    assert report["fast_lower_m"] < .00025
    assert report["segments_checked"] > 0


def test_motion_reserve_refines_an_already_admissible_projected_bound():
    proof = module().WireClearance(wire(), radius_m=.0045)
    point = hull(.0043701, azimuth=np.deg2rad(2.5))

    original = proof.certify(point, selected=True)
    assert original["passed"]
    assert original["segments_checked"] == 0
    assert original["lower_m"] < -.00013

    refined = proof.certify(point, selected=True, reserve_m=.00002)
    assert refined["passed"]
    assert refined["threshold_m"] == -.00015
    assert refined["reserve_m"] == .00002
    assert refined["segments_checked"] > 0
    assert refined["lower_m"] >= -.00013

    too_large = proof.certify(point, selected=True, reserve_m=.000025)
    assert not too_large["passed"]
    assert too_large["threshold_m"] == original["threshold_m"]


@pytest.mark.parametrize("invalid", [-1., float("inf"), float("nan"), True])
def test_motion_reserve_rejects_invalid_values(invalid):
    proof = module().WireClearance(wire(), radius_m=.0045)
    with pytest.raises(ValueError, match="reserve"):
        proof.certify(hull(.005), selected=False, reserve_m=invalid)
