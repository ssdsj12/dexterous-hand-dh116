import numpy as np
import pytest

from go2_pvcnn.control.fixed_dual_panda_o6.wire_visual import rod_surface


def state():
    positions = np.array([[.43, .43, .43], [-.25, 0., .25], [.32, .32, .32]])
    # PyElastica directors are ROWS: world z, world x, world y.
    frames = np.repeat(np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])[:, :, None], 2, axis=2)
    return positions, frames


def test_surface_uses_world_positions_row_directors_and_radius():
    p, d = state()
    vertices, counts, indices = rod_surface(p, d, .008, sides=12)
    assert vertices.shape == (36, 3)
    np.testing.assert_allclose(vertices[0], p[:, 0] + [.0, .0, .008])
    np.testing.assert_allclose(np.linalg.norm(vertices.reshape(3, 12, 3) - p.T[:, None], axis=-1), .008)
    assert len(indices) == sum(counts)
    assert min(indices) == 0 and max(indices) == len(vertices)-1
    # Closed surface: every undirected edge used twice.
    edges = {}
    cursor = 0
    for count in counts:
        face = indices[cursor:cursor+count]
        for a, b in zip(face, face[1:] + face[:1]):
            edge = tuple(sorted((a, b)))
            edges[edge] = edges.get(edge, 0) + 1
        cursor += count
    assert set(edges.values()) == {2}


def test_curved_state_changes_mesh_not_source():
    p, d = state()
    initial = p.copy()
    before = rod_surface(p, d, .008)[0]
    p[2, 1] += .02
    after = rod_surface(p, d, .008)[0]
    assert not np.array_equal(before, after)
    np.testing.assert_array_equal(p[:, [0, 2]], initial[:, [0, 2]])


@pytest.mark.parametrize('radius', [0, -1, float('nan'), float('inf'), True])
def test_invalid_radius(radius):
    with pytest.raises((ValueError, TypeError)):
        rod_surface(*state(), radius)


@pytest.mark.parametrize('sides', [2, True, 3.5])
def test_invalid_sides(sides):
    with pytest.raises((ValueError, TypeError)):
        rod_surface(*state(), .008, sides=sides)


def test_invalid_state_rejected():
    p, d = state()
    for bad in (p[:, :1], p.T[:2], p * np.nan):
        with pytest.raises(ValueError):
            rod_surface(bad, d, .008)
    d[0, 0, 0] = 4
    with pytest.raises(ValueError):
        rod_surface(p, d, .008)
