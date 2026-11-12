"""CPU examples check routing and shared native-buffer ownership only."""
import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_teacher_wire_tensor_probe as probe


class ContactView:
    sensor_count = 2
    filter_count = 2
    max_contact_data_count = 8
    sensor_paths = ['/env0/Wire', '/env1/Wire']
    filter_paths = [['/env0/a', '/env0/b'], ['/env1/a', '/env1/b']]

    def __init__(self):
        self.counts = torch.tensor([[1, 1], [1, 0]], dtype=torch.int32)
        self.starts = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32)
        self.matrices = torch.zeros(2,2,3)
        self.matrices[:,:,2] = torch.tensor([[1., 2.], [3., 0.]])

    def check(self):
        return True

    def get_contact_force_matrix(self, dt):
        assert dt == 1/480
        return self.matrices

    def get_contact_data(self, dt):
        assert dt == 1/480
        loads = torch.arange(1, 9).reshape(8,1).float()
        normals = torch.tensor([[0.,0.,1.]]).repeat(8,1)
        return loads, torch.zeros(8,3), normals, torch.zeros(8,1), self.counts, self.starts

    def get_friction_data(self, dt):
        assert dt == 1/480
        # Installed API reuses these exact metadata buffers.
        self.counts.fill_(0)
        self.starts.fill_(0)
        self.counts[0, 1] = 1
        force = torch.zeros(8,3)
        force[0, 0] = .25
        return force, torch.zeros(8,3), self.counts, self.starts


def test_exact_order_and_copies_before_getter_alias_reuse():
    view = ContactView()
    result = probe.read_filtered(view, view.sensor_paths, view.filter_paths, dt=1/480)
    assert result['normal_pair_n'][0] == [[0.,0.,1.], [0.,0.,2.]]
    assert result['friction_pair_n'][0] == [[0.,0.,0.], [.25,0.,0.]]
    assert result['normal_cells'][0][0]['count'] == 1
    assert result['friction_cells'][0][0]['count'] == 0


def test_filter_binding_cannot_be_accepted_by_slot_count_alone():
    view = ContactView()
    wrong = list(reversed(view.filter_paths))
    with pytest.raises(ValueError, match='filter routing'):
        probe.read_filtered(view, view.sensor_paths, wrong, dt=1/480)


@pytest.mark.parametrize('fault', ['overlap', 'saturated', 'matrix'])
def test_bad_slots_or_inconsistent_native_matrix_cannot_be_reduced(fault):
    view = ContactView()
    if fault == 'overlap':
        view.starts[1,0] = 1
    elif fault == 'saturated':
        view.counts[0,0] = 6
    else:
        view.matrices[0,0,2] = 2.
    with pytest.raises(ValueError, match='slot|matrix|capacity'):
        probe.read_filtered(view, view.sensor_paths, view.filter_paths, dt=1/480)


def test_total_net_residual_keeps_both_candidates_and_unverified_semantics():
    view = ContactView()
    a = probe.read_filtered(view, view.sensor_paths, view.filter_paths, dt=1/480)
    b = probe.net_residual([[.25,0.,4.],[0.,0.,3.]], {'hand':a})
    assert b['net_minus_filtered_normal_n'] == [[.25,0.,1.],[0.,0.,0.]]
    assert b['net_minus_filtered_normal_and_friction_n'] == [[0.,0.,1.],[0.,0.,0.]]
    assert b['unknown_partner_coverage_qualified'] is False
    assert b['upstream_overflow_excluded'] is False
    assert b['net_contains_friction'] is None
