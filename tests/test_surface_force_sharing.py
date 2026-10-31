import pytest
import torch
from go2_pvcnn.control.dh116 import tabletop_surface_force_sharing as sharing


def case(values, directions):
    loads=torch.tensor([[values,values]],dtype=torch.float64)
    normals=torch.tensor([[directions,directions]],dtype=torch.float64)
    return loads,normals


def test_one_against_two_shares_by_surface_not_all_fingers():
    loads,normals=case([1.,1.,2.,0.,0.],[[1,0,0],[-1,0,0],[-1,0,0],[0,0,0],[0,0,0]])
    result=sharing.surface_targets(loads,normals)
    torch.testing.assert_close(result['desired'],torch.tensor([[[2.,1.,1.,0.,0.]]*2],dtype=loads.dtype))
    torch.testing.assert_close(result['desired'].sum(-1),loads.sum(-1))


@pytest.mark.parametrize('scale',[.1,1.,4.,20.])
def test_explored_total_is_preserved_without_newton_band(scale):
    loads,normals=case([2.,1.,1.,0.,0.],[[1,0,0],[-1,0,0],[-1,0,0],[0,0,0],[0,0,0]])
    result=sharing.surface_targets(loads*scale,normals)
    torch.testing.assert_close(result['desired'],loads*scale)
    assert result['paired'].all()


def test_two_against_three_all_five_digits_can_bear():
    loads,normals=case([1.,1.,1.,1.,1.],[[1,0,0],[1,0,0],[-1,0,0],[-1,0,0],[-1,0,0]])
    result=sharing.surface_targets(loads,normals)
    torch.testing.assert_close(result['desired'],torch.tensor([[[1.25,1.25,2.5/3,2.5/3,2.5/3]]*2],dtype=loads.dtype))


def test_permuting_digits_does_not_change_physical_allocation():
    loads,normals=case([1.,1.,2.,0.,0.],[[1,0,0],[-1,0,0],[-1,0,0],[0,0,0],[0,0,0]])
    order=torch.tensor([4,2,0,3,1])
    a=sharing.surface_targets(loads,normals)
    b=sharing.surface_targets(loads[...,order],normals[...,order,:])
    torch.testing.assert_close(b['desired'],a['desired'][...,order])


def test_same_surface_and_zero_contacts_never_count_as_balanced_grasp():
    loads,normals=case([1.,1.,0.,0.,0.],[[1,0,0]]*5)
    for actual in (loads,loads*0):
        result=sharing.surface_targets(actual,normals)
        assert not result['paired'].any()
        assert not result['balanced'].any()
        torch.testing.assert_close(result['desired'],actual)


def test_surface_totals_and_same_surface_balance_are_separate():
    loads,normals=case([2.,.2,1.8,0.,0.],[[1,0,0],[-1,0,0],[-1,0,0],[0,0,0],[0,0,0]])
    result=sharing.surface_targets(loads,normals)
    assert result['face_imbalance'].max()<1e-12
    assert (result['within_imbalance']>0).all()
    assert not result['balanced'].any()


def test_front_back_label_swap_is_continuous_and_allocation_identical():
    loads,normals=case([2.,1.,1.,0.,0.],[[1,0,0],[-1,0,0],[-1,0,0],[0,0,0],[0,0,0]])
    a=sharing.surface_targets(loads,normals)
    b=sharing.surface_targets(loads,normals,previous_axis=-a['axis'])
    torch.testing.assert_close(a['desired'],b['desired'])
    torch.testing.assert_close(b['axis'],-a['axis'])


@pytest.mark.parametrize('value',[-1.,float('nan'),float('inf')])
def test_invalid_raw_loads_are_not_silently_normalized(value):
    loads,normals=case([1.,1.,0.,0.,0.],[[1,0,0],[-1,0,0],[0,0,0],[0,0,0],[0,0,0]])
    loads[...,0]=value
    with pytest.raises(ValueError):sharing.surface_targets(loads,normals)


def test_zero_normal_bearing_contact_prevents_false_pair_qualification():
    loads,normals=case([1.,1.,1.,0.,0.],[[1,0,0],[-1,0,0],[0,0,0],[0,0,0],[0,0,0]])
    result=sharing.surface_targets(loads,normals)
    assert not result['paired'].any()
    torch.testing.assert_close(result['desired'],loads)
