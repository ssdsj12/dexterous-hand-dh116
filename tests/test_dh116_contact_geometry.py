import pytest
import torch


def test_weighted_center_and_normal_use_real_contacts():
    from go2_pvcnn.control.dh116.tabletop_contact_geometry import aggregate_contact_geometry

    positions = torch.tensor([[[[.008, 0., -.004], [-.006, 0., -.003]],
                               [[-.008, 0., -.004], [.006, 0., -.003]]]], dtype=torch.float64)
    forces = torch.tensor([[[[3., 0., 0.], [0., 0., 4.]],
                            [[-3., 0., 0.], [0., 0., 4.]]]], dtype=torch.float64)
    result = aggregate_contact_geometry(positions, forces, torch.zeros((1, 3), dtype=torch.float64))
    assert result['valid'].tolist() == [[True, True]]
    assert result['contact_count'].tolist() == [[2, 2]]
    assert result['normal_load_n'][0, 0].item() == pytest.approx(7.)
    assert result['contact_center_w'][0, 0, 2].item() == pytest.approx(-.0034285714)
    torch.testing.assert_close(result['contact_normal_w'][0, 0],
                               torch.tensor([3., 0., 4.], dtype=torch.float64)/5.)
    assert result['all_below_centerline'].tolist() == [[True, True]]


def test_invalid_contacts_are_rejected_without_proxy_fallback():
    from go2_pvcnn.control.dh116.tabletop_contact_geometry import aggregate_contact_geometry

    positions = torch.full((1, 2, 2, 3), float('nan'), dtype=torch.float64)
    forces = torch.zeros_like(positions)
    result = aggregate_contact_geometry(positions, forces, torch.zeros((1, 3), dtype=torch.float64))
    assert not result['valid'].any()
    assert not result['all_below_centerline'].any()
    assert torch.isnan(result['contact_center_w']).all()


def test_loaded_contact_above_centerline_fails_all_below_gate():
    from go2_pvcnn.control.dh116.tabletop_contact_geometry import aggregate_contact_geometry

    positions = torch.tensor([[[[.008, 0., .001]], [[-.008, 0., -.001]]]], dtype=torch.float64)
    forces = torch.ones_like(positions)
    result = aggregate_contact_geometry(positions, forces, torch.zeros((1, 3), dtype=torch.float64))
    assert result['valid'].tolist() == [[True, True]]
    assert result['all_below_centerline'].tolist() == [[False, True]]
