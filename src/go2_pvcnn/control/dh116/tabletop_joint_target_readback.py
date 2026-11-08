"""Read target provenance without submitting commands or starting a simulator.

PhysX backend targets, IsaacLab user-command caches and reconstructed commands
are distinct evidence. A target is not a measured joint position or proof that
the actuator reached it. Returned arrays are owned snapshots of SDK buffers.
"""
import numpy as np

from .linear_coupling import JOINTS


BACKEND_SOURCE = 'physx_backend_get_dof_position_targets'
CACHE_SOURCE = 'isaaclab_user_command_cache_joint_pos_target'
WRIST_LINEAR = ('wrist_x', 'wrist_y', 'wrist_z')
WRIST_ANGULAR = ('wrist_roll', 'wrist_pitch', 'wrist_yaw')


def _owned(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def _read(getter, source, names, names_valid, *, count=None, integer=False):
    result = dict(available=False, source=source, values=None, finite_env=None,
                  dtype=None, shape=None, error=None)
    try:
        value = _owned(getter())
        result.update(dtype=str(value.dtype), shape=list(value.shape))
        if (not names_valid or value.ndim != 2 or value.shape[0] < 1
                or value.shape[1] != len(names) or (count is not None and len(value) != count)
                or value.dtype.kind not in ('iu' if integer else 'f')):
            raise ValueError('named complete [environment,DOF] target/type array required')
        result.update(available=True, values=value, finite_env=np.isfinite(value).all(1))
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    return result


def read_joint_target_evidence(hand):
    """Return complete per-hand backend/cache/type evidence as owned NumPy arrays.

    ``available`` means a named array was obtained, independently of each
    environment's ``finite_env`` flag. Missing/failed backend reads never use
    a cache fallback. Units come from native DofType, not a joint-name guess;
    expected source-contract units remain separately labelled.
    """
    names = list(getattr(hand, 'joint_names', []))
    names_valid = (bool(names) and all(isinstance(name, str) and name for name in names)
                   and len(set(names)) == len(names))
    backend = _read(lambda: hand.root_physx_view.get_dof_position_targets(),
                    BACKEND_SOURCE, names, names_valid)
    count = len(backend['values']) if backend['available'] else None
    cache = _read(lambda: hand.data.joint_pos_target, CACHE_SOURCE, names, names_valid, count=count)
    if count is None and cache['available']:
        count = len(cache['values'])
    types = _read(lambda: hand.root_physx_view.get_dof_types(),
                  'physx_backend_get_dof_types', names, names_valid, count=count, integer=True)
    units = dict.fromkeys(names)
    if types['available']:
        for index, name in enumerate(names):
            kinds = types['values'][:, index]
            if np.all(kinds == 0):
                units[name] = 'rad'
            elif np.all(kinds == 1):
                units[name] = 'm'
    expected = {name: ('m' if name in WRIST_LINEAR else 'rad'
                      if name in WRIST_ANGULAR or name in JOINTS else None) for name in names}
    return dict(joint_names=names, joint_names_valid=names_valid,
                backend=backend, command_cache=cache, dof_types=types,
                units_by_name=units,
                unit_source='native_DofType_0_rotation_1_translation;unknown_or_mixed_is_null',
                expected_units_by_name=expected,
                expected_unit_source='original_DH116_and_driven_carriage_contract;not_native_type_readback',
                wrist_joint_indices={name: names.index(name) if names_valid and name in names else None
                                     for name in WRIST_LINEAR + WRIST_ANGULAR},
                wrist_target_frame='anchor_local_translation;seed_intrinsic_XYZ_rotation;not_measured_world_palm',
                native_target_readback_is_not_actual_joint_position=True)
