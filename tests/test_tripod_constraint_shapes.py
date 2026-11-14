from copy import deepcopy
import pytest
from scripts import solve_dh116_tripod_hard_constraints as m
from tests.test_tripod_hard_constraints import row

@pytest.mark.parametrize("key",["node_sdf_gap_m","node_gradient_norm","required_hull_intervals_m","geometric_opposition","sdf_opposition"])
def test_missing_digit_or_opposition_constraint_is_rejected(key):
    r=deepcopy(row());r[key]=r[key][:-1]
    with pytest.raises(ValueError,match="three pads"):m.contract(r)
