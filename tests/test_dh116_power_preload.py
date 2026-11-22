import numpy as np
import pytest

from scripts.prepare_dh116_power_preload import checked_terminal_witness, solve_preload


def test_bounded_preload_solves_signed_gap_within_original_motion():
    shift = solve_preload(lambda delta: .0003 - .04 * delta, travel=.011)
    assert shift == pytest.approx(.00875)
    assert .0003 - .04 * shift == pytest.approx(-.00005)


def test_preload_does_not_extrapolate_outside_the_bounded_search():
    with pytest.raises(ValueError, match="cannot reach"):
        solve_preload(lambda delta: .0003 - .04 * delta, travel=.002)


def test_preload_refuses_to_misattribute_a_nearer_different_node():
    report = dict(node=36, gap_m=-.00005, most_negative_node=36,
                  negative_nodes=[35, 36], normal_world=[1., 0., 0.])
    with pytest.raises(ValueError, match="material station"):
        checked_terminal_witness(report, node=36, max_penetration_m=.00015)
    report["negative_nodes"] = [36]
    checked_terminal_witness(report, node=36, max_penetration_m=.00015)
    report["normal_world"] = np.zeros(3)
    with pytest.raises(ValueError, match="normal"):
        checked_terminal_witness(report, node=36, max_penetration_m=.00015)
