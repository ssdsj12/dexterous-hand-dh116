"""Fixed material-region task semantics using CPU synthetic measurements."""
import numpy as np
import pytest

from test_dh116_batched_task import make_env
from go2_pvcnn.control.dh116.tabletop_batched_task import BatchTask


ACTION = np.r_[np.zeros(18), -1., -1.]


def material_env():
    env = make_env(.055)
    env.backend.regions = np.array([[.01, 0., -.02], [-.01, 0., -.02]])
    env.backend.region_speed = np.zeros(2)
    measured = env.backend.measure
    env.backend.measure = lambda: dict(measured(),
        material_relative=env.backend.regions.copy(), material_speed=env.backend.region_speed.copy())
    env.reset(seed=11)
    return env


def step_pair(scalar, batched, transform=None):
    expected = scalar.step(ACTION)
    batch = BatchTask([batched])
    def advance(offset, command, allowed, live):
        if live[0]:
            batched.backend.advance(offset[0], command[0], palm_motion_allowed=bool(allowed[0]),
                palm_action=batch.palm_actions[0], grip_action=np.ones(2))
        measured = batch.stack_measurements([batched.backend.measure()])
        return measured if transform is None else transform(measured)
    actual = batch.step(ACTION[None], advance)[0]
    np.testing.assert_allclose(expected[0], actual[0], atol=2e-6, rtol=0)
    assert expected[1] == pytest.approx(actual[1], abs=1e-10)
    assert expected[2:] == actual[2:]
    assert scalar.metrics['slip_m'] == pytest.approx(batched.metrics['slip_m'])
    assert scalar.metrics['object_speed_m_s'] == pytest.approx(batched.metrics['object_speed_m_s'])
    assert scalar.task.contact_time == pytest.approx(batched.task.contact_time)
    assert scalar.task.success_time == pytest.approx(batched.task.success_time)
    return expected


@pytest.mark.parametrize('phase,displacement', [('hold', .004), ('lift', .025)])
def test_opposite_material_slip_rejects_stability_with_unchanged_com(phase, displacement):
    scalar, batched = material_env(), material_env()
    for env in (scalar, batched):
        env.backend.decimation = 1
        env.task.phase = phase
        env.task.contact_time = .299
        env.task.success_time = .499
        env.grip_relative = env.measurement['object_relative'].copy()
        env.grip_regions = env.backend.regions.copy()
        env.offset = .04
        env.backend.regions[:, 0] += np.array([displacement, -displacement])
    step_pair(scalar, batched)
    assert scalar.metrics['slip_m'] == pytest.approx(displacement)
    assert not scalar.task.success
    assert scalar.task.phase == ('close' if phase == 'hold' else 'lift')
    if phase == 'lift':
        assert scalar.task.success_time == 0.
    else:
        assert scalar.task.contact_time == 0.
        assert scalar.grip_regions is None and batched.grip_regions is None
    np.testing.assert_array_equal(scalar.measurement['object_relative'], [0., 0., -.02])


def test_material_references_capture_clear_reestablish_and_reset():
    scalar, batched = material_env(), material_env()
    for env in (scalar, batched):
        assert env.grip_regions is None
        env.task.contact_time = .199
    step_pair(scalar, batched)
    for env in (scalar, batched):
        assert env.task.phase == 'hold'
        np.testing.assert_array_equal(env.grip_regions, env.backend.regions)
        env.backend.regions[:, 1] += [.001, -.001]
        env.task.contact_time = .299
    step_pair(scalar, batched)
    for env in (scalar, batched):
        assert env.task.phase == 'lift'
        np.testing.assert_array_equal(env.grip_regions, env.backend.regions)
        env.reset(seed=11)
        assert env.grip_regions is None
        env.task.phase = 'hold'
        env.grip_regions = env.backend.regions.copy()
        env.grip_relative = env.measurement['object_relative'].copy()
        env.backend.region_speed[:] = .03
    step_pair(scalar, batched)
    for env in (scalar, batched):
        assert env.task.phase == 'close'
        assert env.grip_regions is None and env.grip_relative is None


@pytest.mark.parametrize('phase,speed', [('close', .021), ('lift', .031)])
def test_fastest_region_prevents_stationary_com_success(phase, speed):
    scalar, batched = material_env(), material_env()
    for env in (scalar, batched):
        env.task.phase = phase
        env.task.contact_time = .199
        env.task.success_time = .499
        env.offset = .04
        env.backend.region_speed[:] = [0., speed]
    step_pair(scalar, batched)
    assert scalar.metrics['object_speed_m_s'] == pytest.approx(speed)
    assert scalar.task.phase == phase
    assert not scalar.task.success


@pytest.mark.parametrize('key,value', [
    ('material_relative', np.full((2, 3), np.nan)),
    ('material_relative', np.full((2, 3), np.inf)),
    ('material_relative', np.zeros(3)),
    ('material_speed', np.array([0., np.nan])),
    ('material_speed', np.array([0., -1.])),
    ('material_speed', np.zeros((2, 1))),
])
def test_invalid_optional_measurement_uses_existing_finite_terminal_guard(key, value):
    scalar, batched = material_env(), material_env()
    for env in (scalar, batched):
        measured = env.backend.measure
        env.backend.measure = lambda measured=measured: dict(measured(), **{key: value})
    obs, reward, done, _, info = step_pair(scalar, batched)
    assert done and info['failure'] == 'nonfinite_state'
    assert np.isfinite(obs).all() and np.isfinite(reward)


def test_mixed_rigid_and_material_backends_keep_per_environment_semantics():
    scalar = [make_env(.055), material_env()]
    batched = [make_env(.055), material_env()]
    for env in (scalar[1], batched[1]):
        env.backend.region_speed[:] = [.01, .04]
    expected = [env.step(ACTION) for env in scalar]
    batch = BatchTask(batched)
    def advance(offset, command, allowed, live):
        for i, env in enumerate(batched):
            if live[i]:
                env.backend.advance(offset[i], command[i], palm_motion_allowed=bool(allowed[i]),
                    palm_action=batch.palm_actions[i], grip_action=np.ones(2))
        return batch.stack_measurements([env.backend.measure() for env in batched])
    actual = batch.step(np.tile(ACTION, (2, 1)), advance)
    for i, (x, y) in enumerate(zip(expected, actual)):
        np.testing.assert_allclose(x[0], y[0], atol=2e-6, rtol=0)
        assert x[1] == pytest.approx(y[1], abs=1e-10)
        assert x[2:] == y[2:]
        assert scalar[i].metrics['object_speed_m_s'] == batched[i].metrics['object_speed_m_s']
    assert 'material_relative' not in batched[0].measurement
    assert 'material_speed' not in batched[0].measurement
    assert batched[0].metrics['object_speed_m_s'] == 0.
    assert batched[1].metrics['object_speed_m_s'] == .04


@pytest.mark.parametrize('optional_key', ['material_relative', 'material_speed'])
def test_each_optional_field_is_independently_supported(optional_key):
    scalar, batched = material_env(), material_env()
    for env in (scalar, batched):
        measured = env.backend.measure
        def measure(measured=measured):
            m = measured()
            m.pop('material_speed' if optional_key == 'material_relative' else 'material_relative')
            return m
        env.backend.measure = measure
        env.reset(seed=11)
        env.task.contact_time = .199
    step_pair(scalar, batched)
    assert scalar.task.phase == 'hold'
    assert (scalar.grip_regions is not None) == (optional_key == 'material_relative')
    assert (batched.grip_regions is not None) == (optional_key == 'material_relative')


@pytest.mark.parametrize('key,value', [
    ('material_relative', np.zeros((3, 2))), ('material_speed', np.array([0., -1.])),
    ('material_relative', np.full((2, 3), np.nan)), ('material_speed', np.array([0., np.inf])),
])
def test_reset_rejects_invalid_material_measurements(key, value):
    env = material_env()
    measured = env.backend.measure
    env.backend.measure = lambda: dict(measured(), **{key: value})
    with pytest.raises(ValueError, match=key):
        env.reset(seed=11)


def test_finite_material_positions_whose_slip_norm_overflows_fail_guard():
    scalar, batched = material_env(), material_env()
    for env in (scalar, batched):
        env.task.phase = 'lift'
        env.grip_regions = env.backend.regions.copy()
        env.backend.regions[:] = 1e300
    with np.errstate(over='ignore', invalid='ignore'):
        _, _, done, _, info = step_pair(scalar, batched)
    assert done and info['failure'] == 'nonfinite_state'


@pytest.mark.parametrize('malformed', [False, True])
def test_direct_batch_material_arrays_do_not_require_presence_masks(malformed):
    scalar, batched = material_env(), material_env()
    for env in (scalar, batched):
        env.backend.region_speed[:] = [.01, .04]
        if malformed:
            measured = env.backend.measure
            env.backend.measure = lambda measured=measured: dict(measured(), material_speed=np.zeros((2, 1)))
    def direct(measured):
        result = {k: v for k, v in measured.items() if not k.startswith('_has_')}
        if malformed:
            result['material_speed'] = np.zeros((1, 2, 1))
        return result
    _, _, done, _, info = step_pair(scalar, batched, direct)
    assert done == malformed
    assert info['failure'] == ('nonfinite_state' if malformed else None)
