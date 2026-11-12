from pathlib import Path

import numpy as np
import pytest

from scripts.bake_dh116_deformx_pad_sdfs import checked_json, validate_signed_grid


def test_cooked_input_digest_must_match_before_read(tmp_path):
    path = tmp_path / "capture.json"
    path.write_text('{"native": true}')
    with pytest.raises(ValueError, match="SHA256"):
        checked_json(Path(path), "0" * 64)


def test_signed_grid_requires_positive_contact_margin_and_negative_inside():
    class Grid:
        values = np.full((5, 5, 5), .012, dtype=np.float32)
        origin = np.array([-.02, -.02, -.02], dtype=np.float32)
        spacing = .001

        def sample(self, points):
            return np.array([-.004, .008, 0., 0., 0., 0.]), np.ones(len(points), dtype=bool)

    vertices = np.array([[0., 0., 0.], [.002, 0., 0.], [0., .002, 0.], [0., 0., .002]])
    grid = Grid()
    grid.values[2, 2, 2] = -.004
    report = validate_signed_grid(grid, vertices, rod_radius=.005)
    assert report["inside_signed_distance_m"] < 0
    grid.values[0, 0, 0] = .004
    with pytest.raises(ValueError, match="boundary"):
        validate_signed_grid(grid, vertices, rod_radius=.005)
    grid.values[0, 0, 0] = .012
    grid.sample = lambda points: (np.ones(len(points)) * .01, np.ones(len(points), dtype=bool))
    with pytest.raises(ValueError, match="sign"):
        validate_signed_grid(grid, vertices, rod_radius=.005)
