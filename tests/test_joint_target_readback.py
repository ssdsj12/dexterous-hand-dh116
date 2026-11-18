"""CPU source-separation fixtures; no PhysX simulation or native evidence."""
import importlib
import importlib.util
from types import SimpleNamespace as NS

import numpy as np

from go2_pvcnn.control.dh116.linear_coupling import JOINTS


def reader():
    name = 'go2_pvcnn.control.dh116.tabletop_joint_target_readback'
    assert importlib.util.find_spec(name) is not None, 'independent backend target reader required'
    return importlib.import_module(name).read_joint_target_evidence


class Backend:
    def __init__(self):
        self.values = np.array([[.01, .02, .03], [.04, .05, .06]], dtype=np.float32)
        self.types = np.array([[1, 0, 255], [1, 0, 255]], dtype=np.uint8)
        self.calls = 0

    def get_dof_position_targets(self):
        self.calls += 1
        return self.values

    def get_dof_types(self):
        return self.types


def hand():
    return NS(joint_names=['wrist_x', 'wrist_roll', 'unknown_joint'], root_physx_view=Backend(),
              data=NS(joint_pos_target=np.ones((2, 3), dtype=np.float64)*.9))


def test_reads_actual_backend_separately_and_owns_returned_arrays():
    h = hand()
    value = reader()(h)
    assert value['backend']['available']
    assert value['backend']['source'] == 'physx_backend_get_dof_position_targets'
    assert value['backend']['dtype'] == 'float32'
    assert value['command_cache']['source'] == 'isaaclab_user_command_cache_joint_pos_target'
    assert value['command_cache']['dtype'] == 'float64'
    np.testing.assert_array_equal(value['backend']['finite_env'], [True, True])
    assert value['units_by_name'] == {'wrist_x': 'm', 'wrist_roll': 'rad', 'unknown_joint': None}
    assert value['expected_units_by_name']['unknown_joint'] is None
    h.root_physx_view.values[:] = 99
    h.data.joint_pos_target[:] = 88
    assert value['backend']['values'][0, 0] == np.float32(.01)
    assert value['command_cache']['values'][0, 0] == .9
    assert h.root_physx_view.calls == 1


def test_missing_backend_never_promotes_available_command_cache():
    h = hand()
    del h.root_physx_view
    value = reader()(h)
    assert value['backend']['available'] is False
    assert value['backend']['values'] is None and value['backend']['finite_env'] is None
    assert value['backend']['error']
    assert value['command_cache']['available'] is True
    assert value['units_by_name']['wrist_x'] is None
    assert value['expected_units_by_name']['wrist_x'] == 'm'


def test_backend_exception_is_explicit_even_when_cache_is_finite():
    h = hand()
    def fail():
        raise RuntimeError('readback unavailable')
    h.root_physx_view.get_dof_position_targets = fail
    value = reader()(h)
    assert value['backend']['available'] is False
    assert 'readback unavailable' in value['backend']['error']
    assert value['command_cache']['available']


def test_nonfinite_native_target_retains_raw_value_and_per_environment_validity():
    h = hand()
    h.root_physx_view.values[1, 0] = np.nan
    value = reader()(h)
    assert value['backend']['available']
    np.testing.assert_array_equal(value['backend']['finite_env'], [True, False])
    assert np.isnan(value['backend']['values'][1, 0])
    np.testing.assert_array_equal(value['command_cache']['finite_env'], [True, True])


def test_unknown_or_inconsistent_native_types_are_not_guessed_from_names():
    h = hand()
    h.root_physx_view.types[1, 0] = 0
    value = reader()(h)
    assert value['units_by_name']['wrist_x'] is None
    assert value['dof_types']['available']
    del h.root_physx_view.types
    value = reader()(h)
    assert value['dof_types']['available'] is False
    assert all(unit is None for unit in value['units_by_name'].values())


def test_shape_and_duplicate_names_cannot_create_native_named_target_evidence():
    h = hand()
    h.root_physx_view.values = np.zeros((2, 4), dtype=np.float32)
    value = reader()(h)
    assert value['backend']['available'] is False
    assert value['backend']['error']
    h = hand()
    h.joint_names[1] = 'wrist_x'
    value = reader()(h)
    assert value['joint_names_valid'] is False
    assert value['backend']['available'] is False


def test_full_seventeen_dofs_use_actual_reordered_names_and_native_types():
    names = list(reversed(['wrist_x', 'wrist_y', 'wrist_z', 'wrist_roll', 'wrist_pitch', 'wrist_yaw'] + list(JOINTS)))
    target = np.arange(17, dtype=np.float32)[None, :] / 100
    types = np.array([[1 if name in ('wrist_x', 'wrist_y', 'wrist_z') else 0 for name in names]], dtype=np.uint8)
    h = NS(joint_names=names, data=NS(joint_pos_target=target + .1),
           root_physx_view=NS(get_dof_position_targets=lambda: target, get_dof_types=lambda: types))
    value = reader()(h)
    assert value['backend']['values'].shape == (1, 17)
    assert value['joint_names'] == names
    assert value['wrist_joint_indices']['wrist_z'] == names.index('wrist_z')
    assert value['units_by_name']['wrist_z'] == 'm'
    assert value['units_by_name']['wrist_pitch'] == 'rad'
    assert value['units_by_name']['finger12'] == 'rad'
    assert value['backend']['values'][0, names.index('wrist_z')] == target[0, names.index('wrist_z')]
