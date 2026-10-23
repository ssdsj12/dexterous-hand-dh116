import numpy as np
import pytest

from go2_pvcnn.control.dh116.reference_view import (
    REFERENCE_TABLE_CENTER,
    REFERENCE_TABLE_SIZE,
    REFERENCE_WIRE_END,
    REFERENCE_WIRE_START,
    camera_preset,
    reference_wire_points,
)


def test_reference_wire_runs_along_world_y_at_tabletop_height():
    points = reference_wire_points()
    assert points.shape == (3, 41)
    assert np.allclose(points[0], 0.43)
    assert np.allclose(points[2], 0.308)
    assert np.isclose(points[1, 0], -0.25)
    assert np.isclose(points[1, -1], 0.25)
    assert REFERENCE_WIRE_START == (0.43, -0.25, 0.308)
    assert REFERENCE_WIRE_END == (0.43, 0.25, 0.308)
    assert REFERENCE_TABLE_CENTER[2] + REFERENCE_TABLE_SIZE[2] / 2 == 0.30


@pytest.mark.parametrize("name", ("perspective", "top", "side"))
def test_reference_camera_presets_are_finite(name):
    view = camera_preset(name)
    assert len(view.eye) == len(view.target) == 3
    assert np.isfinite(np.asarray(view.eye)).all()
    assert np.isfinite(np.asarray(view.target)).all()


def test_reference_camera_rejects_unknown_view():
    with pytest.raises(ValueError, match="perspective|top|side"):
        camera_preset("isometric")

