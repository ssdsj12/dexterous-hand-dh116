import numpy as np
import pytest


def test_transverse_force_direction_distinguishes_away_from_palm_and_unobservable():
    from scripts.audit_dh116_power_proximal_first_contact import transverse_alignment

    g=np.array([1.,0.,0.]); tangent=np.array([0.,1.,0.])
    assert transverse_alignment(g, tangent, np.array([1.,5.,0.])) == pytest.approx(1.)
    assert transverse_alignment(g, tangent, np.array([-1.,5.,0.])) == pytest.approx(-1.)
    assert transverse_alignment(g, tangent, tangent) is None
    with pytest.raises(ValueError):
        transverse_alignment(np.zeros(3), tangent, g)
