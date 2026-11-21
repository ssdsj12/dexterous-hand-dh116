import hashlib
import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import audit_dh116_pinch_initial_pose as geometry
from scripts import audit_dh116_certified_segment_distance as certified


VERTICES = np.array(list(itertools.product((-.01, .01), repeat=3)))


def stopped_at(point):
    return lambda *a, **k: SimpleNamespace(success=False, x=np.array(point), status=8)


def test_segment_accepts_failed_optimizer_only_with_independent_distance_bounds(monkeypatch):
    # Segment spans 2 cm, so the centred, scaled maximum x coordinate is .5.
    monkeypatch.setattr(certified, "minimize", stopped_at([.5, 0., 0., .5]))
    distance = certified.segment_hull_distance(VERTICES, [.015, -.01, 0.], [.015, .01, 0.])
    assert .005 - 1e-9 <= distance <= .005


def test_bad_witness_cannot_pass_failed_optimizer(monkeypatch):
    monkeypatch.setattr(certified, "minimize", stopped_at([-.5, 0., 0., 0.]))
    with pytest.raises(ValueError, match="did not converge"):
        certified.segment_hull_distance(VERTICES, [.015, -.01, 0.], [.015, .01, 0.])


def test_failed_rectangle_optimizer_keeps_original_rejection(monkeypatch):
    monkeypatch.setattr(geometry, "minimize", stopped_at([.5, 0., 0., .5, .5]))
    with pytest.raises(ValueError, match="did not converge"):
        geometry.convex_patch_closest(VERTICES, [.015, -.01, -.01], [[0., .02, 0.], [0., 0., .02]])



def test_historical_geometry_source_remains_byte_identical():
    assert hashlib.sha256(Path(geometry.__file__).read_bytes()).hexdigest() == (
        "15ede98e092a911470bf858f6d37e2a01007f05e3dbdb8ab2178ed95f6712db8"
    )


@pytest.mark.parametrize("status", [4, 6, 9])
def test_other_optimizer_errors_are_not_certified(monkeypatch, status):
    monkeypatch.setattr(certified, "minimize", lambda *a, **k: SimpleNamespace(
        success=False, x=np.array([.5, 0., 0., .5]), status=status))
    with pytest.raises(ValueError, match="did not converge"):
        certified.segment_hull_distance(VERTICES, [.015, -.01, 0.], [.015, .01, 0.])


def test_success_flag_cannot_certify_nonoptimal_feasible_witness(monkeypatch):
    monkeypatch.setattr(certified, "minimize", lambda *a, **k: SimpleNamespace(
        success=True, status=0, x=np.array([-.5, 0., 0., 0.])))
    with pytest.raises(ValueError, match="did not converge"):
        certified.segment_hull_distance(VERTICES, [.015, -.01, 0.], [.015, .01, 0.])


@pytest.mark.parametrize("translation,angle", [
    ([0., 0., 0.], 0.), ([.4, -.2, .6], .731)])
def test_explicit_bounds_enclose_known_distance(translation, angle):
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_rotvec([0., 0., angle]).as_matrix()
    shift = np.array(translation)
    points = np.array([[.015, -.01, 0.], [.015, .01, 0.]]) @ rotation.T + shift
    bounds = certified.segment_hull_bounds(VERTICES @ rotation.T + shift, *points)
    assert bounds["lower_m"] <= .005 <= bounds["upper_m"]
    assert bounds["upper_m"] - bounds["lower_m"] <= 1e-9
    witness_distance = np.linalg.norm(
        np.array(bounds["hull_point_m"]) - bounds["segment_point_m"])
    assert abs(witness_distance - bounds["upper_m"]) <= 2e-12
    assert bounds["optimizer_status"] in (0, 8)


def test_distance_api_uses_lower_bound_even_on_optimizer_success(monkeypatch):
    monkeypatch.setattr(certified, "minimize", lambda *a, **k: SimpleNamespace(
        success=True, status=0, x=np.array([.5, 0., 0., .5])))
    distance = certified.segment_hull_distance(VERTICES, [.015, -.01, 0.], [.015, .01, 0.])
    assert .005 - 1e-9 <= distance < .005


@pytest.mark.parametrize("status", [0, 8])
def test_near_optimal_witness_is_polished_without_relaxing_certificate(monkeypatch, status):
    monkeypatch.setattr(certified, "minimize", lambda *a, **k: SimpleNamespace(
        success=status == 0, status=status, x=np.array([.5, 1e-6, 0., .5])))
    bounds = certified.segment_hull_bounds(VERTICES, [.015, -.01, 0.], [.015, .01, 0.])
    assert bounds["lower_m"] <= .005 <= bounds["upper_m"]
    assert bounds["upper_m"] - bounds["lower_m"] <= 1e-9
    assert bounds["certificate_method"] == "active_feature_polish"
    assert 0 < bounds["polish_candidates"] <= 64


def test_polish_cannot_turn_missing_certificate_into_success(monkeypatch):
    monkeypatch.setattr(certified, "minimize", lambda *a, **k: SimpleNamespace(
        success=True, status=0, x=np.array([-.5, -.5, -.5, 0.])))
    monkeypatch.setattr(certified, "_certify_segment_distance", lambda *a: None)
    with pytest.raises(ValueError, match="did not converge"):
        certified.segment_hull_bounds(VERTICES, [.015, -.01, 0.], [.015, .01, 0.])


def test_polish_has_a_hard_64_candidate_limit(monkeypatch):
    monkeypatch.setattr(certified, "minimize", lambda *a, **k: SimpleNamespace(
        success=True, status=0, x=np.array([-.5, -.5, -.5, 0.])))
    monkeypatch.setattr(certified, "_certify_segment_distance", lambda *a: None)
    original = np.linalg.lstsq
    calls = []
    def counted(*a, **k):
        calls.append(1)
        return original(*a, **k)
    monkeypatch.setattr(certified.np.linalg, "lstsq", counted)
    with pytest.raises(ValueError, match="did not converge"):
        certified.segment_hull_bounds(VERTICES, [.015, -.01, 0.], [.015, .01, 0.])
    assert len(calls) == 64


def test_linear_solver_failure_does_not_bypass_certificate(monkeypatch):
    monkeypatch.setattr(certified, "minimize", lambda *a, **k: SimpleNamespace(
        success=True, status=0, x=np.array([.5, 1e-6, 0., .5])))
    def singular(*a, **k):
        raise np.linalg.LinAlgError("injected numerical failure")
    monkeypatch.setattr(certified.np.linalg, "lstsq", singular)
    with pytest.raises(ValueError, match="did not converge"):
        certified.segment_hull_bounds(VERTICES, [.015, -.01, 0.], [.015, .01, 0.])


@pytest.mark.parametrize("a,b,distance", [
    ([.015, 0., 0.], [.015, 0., 0.], .005),
    ([.01, -.005, 0.], [.01, .005, 0.], 0.),
    ([-.02, 0., 0.], [.02, 0., 0.], 0.),
])
def test_point_tangent_and_intersecting_segments(a, b, distance):
    bounds = certified.segment_hull_bounds(VERTICES, a, b)
    assert bounds["lower_m"] <= distance <= bounds["upper_m"]
    assert bounds["upper_m"] - bounds["lower_m"] <= 1e-9
