import pytest
import torch


def test_initial_axis_below_check_uses_measured_pads_and_clearance():
    from go2_pvcnn.control.dh116.tabletop_initial_contact_check import validate_initial_contact
    m=dict(pad_relative=torch.zeros((2,4,3)),object_quat=torch.tensor([[1.,0.,0.,0.]]*2),
           clearance=torch.full((2,),.001),contact_valid=torch.zeros((2,2),dtype=torch.bool))
    m['pad_relative'][:,:,2]=-.001
    row=validate_initial_contact(m)
    assert row['all_pad_references_below_axis'] and row['measured_contact_hands']==0
    m['pad_relative'][0,0,2]=.001
    with pytest.raises(ValueError,match='below'):validate_initial_contact(m)
    m['pad_relative'][:,:,2]=-.001;m['clearance'][0]=-.0001
    with pytest.raises(ValueError,match='clearance'):validate_initial_contact(m)
