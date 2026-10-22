import pytest

from go2_pvcnn.control.dh116.contact_model import (
    compliant_contact_parameters,
    table_center_z_for_wire,
)


def test_compliant_contact_parameters_are_explicit_and_simulation_only():
    assert compliant_contact_parameters(1000.0, 25.0) == {
        "compliant_contact_stiffness": 1000.0,
        "compliant_contact_damping": 25.0,
    }


@pytest.mark.parametrize("stiffness,damping", [(0.0, 1.0), (-1.0, 1.0), (1.0, -1.0)])
def test_compliant_contact_parameters_reject_nonpositive_values(stiffness, damping):
    with pytest.raises(ValueError):
        compliant_contact_parameters(stiffness, damping)


def test_table_surface_supports_wire_bottom_at_requested_height():
    center = table_center_z_for_wire(0.44, 0.008, 0.02)
    assert center == pytest.approx(0.422)
    assert center + 0.01 == pytest.approx(0.432)
