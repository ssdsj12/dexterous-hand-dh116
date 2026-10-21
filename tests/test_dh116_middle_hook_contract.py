import numpy as np
import pytest

from go2_pvcnn.control.dh116.contact_runtime import BODY_NAMES
from go2_pvcnn.control.dh116.grasp_evidence import GraspEvidenceGate
from go2_pvcnn.control.dh116.grasp_sequence import TabletopGraspSequence


def test_body_channels_are_ordered_left_to_right_with_middle_supports():
    assert tuple(BODY_NAMES) == (
        "left_thumb", "left_index", "left_middle",
        "right_thumb", "right_index", "right_middle",
    )


def test_sequence_requires_six_element_targets_and_exposes_pinch_support_indices():
    sequence = TabletopGraspSequence(
        material_ids=(10, 11, 12, 30, 31, 32),
        target_loads_n=np.ones(6),
    )
    assert sequence.pinch_indices == (0, 1, 3, 4)
    assert sequence.support_indices == (2, 5)
    with pytest.raises(ValueError):
        TabletopGraspSequence(material_ids=(10, 11, 30, 31), target_loads_n=np.ones(4))


def test_middle_channels_are_ordered_and_required_for_lift():
    gate = GraspEvidenceGate(
        material_ids=(10, 11, 12, 30, 31, 32),
        approach_qualified=True,
        minimum_hold_loads_n=np.array([.9, .81, .18, .9, .81, .18]),
    )
    normals = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 0., 1.],
                        [-1., 0., 0.], [1., 0., 0.], [0., 0., 1.]])
    evidence = gate.update(
        0.0,
        loads_n=np.array([1., .9, .2, 1., .9, .2]),
        valid_contacts=np.array([True, True, False, True, True, False]),
        normals=normals,
        anchors_local=np.zeros((6, 3)),
        station_rise_m=np.zeros(2),
        min_surface_clearance_m=.01,
        safety_ok=True,
        active_constraints=[],
    )
    assert evidence["stage"] == "close"
    assert not evidence["permit_lift"]


def test_pinch_contacts_can_close_but_middle_support_is_required_before_hold():
    gate = GraspEvidenceGate(
        material_ids=(10, 11, 12, 30, 31, 32),
        approach_qualified=True,
        minimum_hold_loads_n=np.array([.9, .81, .18, .9, .81, .18]),
    )
    normals = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 0., 1.],
                        [-1., 0., 0.], [1., 0., 0.], [0., 0., 1.]])
    kwargs = dict(
        loads_n=np.array([1., .9, .2, 1., .9, .2]),
        normals=normals,
        anchors_local=np.zeros((6, 3)),
        station_rise_m=np.zeros(2),
        min_surface_clearance_m=.01,
        safety_ok=True,
        active_constraints=[],
    )
    for i in range(11):
        result = gate.update(i * .01, valid_contacts=np.array([True, True, False, True, True, False]), **kwargs)
    assert result["stage"] == "close"
    assert not result["permit_lift"]
    for i in range(11, 112):
        result = gate.update(i * .01, valid_contacts=np.ones(6, dtype=bool), **kwargs)
    assert result["stage"] == "hold"
    assert not result["permit_lift"]

