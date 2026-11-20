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
