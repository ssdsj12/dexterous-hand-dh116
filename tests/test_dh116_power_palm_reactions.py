import numpy as np
import pytest

from scripts.verify_dh116_power_palm_reactions import make_tangent_nodes, reaction_sign


def test_control_rod_is_tangent_and_only_slightly_inside_radius():
    nodes = make_tangent_nodes([0., 0., 0.], [0., 0., 1.], [0., 2., 1.])
    assert nodes.shape == (1, 41, 3)
    np.testing.assert_allclose(nodes[:, :, 2], .00465, atol=1e-8)
    assert np.linalg.norm(nodes[0, -1] - nodes[0, 0]) == pytest.approx(.5)


def test_opposite_normal_does_not_qualify_for_palm_reaction():
    with pytest.raises(ValueError, match="normal"):
        reaction_sign(np.array([0., 0., .5]), np.array([0., 0., 1.]))
    assert reaction_sign(np.array([.1, 0., -.5]), np.array([0., 0., 1.])) == pytest.approx(.5)
