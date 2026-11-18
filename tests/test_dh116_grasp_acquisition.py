import math
import ast
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_grasp_acquisition import CONTRACT, potential, reward


def test_actual_spawn_y_side_surface_regression(device):
    m = measurement(device)
    m['pad_relative'][:] = m['pad_relative'].new_tensor([.008, .2, 0.])
    assert potential(m).item() == pytest.approx(20.)


def test_contract_axis_matches_actual_spawn_source():
    # Parse the actual spawn function without importing/starting Isaac.
    import go2_pvcnn.control.dh116.tabletop_grasp_acquisition as acquisition
    source = Path(acquisition.__file__).with_name('tabletop_vec_isaac.py')
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    spawn = next(node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == 'wire_spawn_configuration')
    cylinders = [node for node in ast.walk(spawn) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'CylinderCfg']
    assert len(cylinders) == 1
    axis = next(keyword.value for keyword in cylinders[0].keywords if keyword.arg == 'axis')
    assert ast.literal_eval(axis) == 'Y'
    assert CONTRACT['object_local_axis'] == ast.literal_eval(axis)
    assert CONTRACT['version'] == 'finite_cylinder_grasp_acquisition_v2'


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


def measurement(device='cpu', n=1):
    return dict(pad_relative=torch.tensor([[[.008, 0., 0.]] * 4], device=device,
                                          dtype=torch.float64).repeat(n, 1, 1),
                object_quat=torch.tensor([[1., 0., 0., 0.]], device=device).repeat(n, 1),
                hand_object_loads=torch.zeros((n, 2), device=device))


def test_approach_retreat_and_two_hands(device):
    m = measurement(device)
    close = potential(m)
    assert close.dtype == torch.float64 and close.shape == (1,)
    assert close.device.type == device
    assert close.item() == pytest.approx(20.)
    m['pad_relative'][:, 2:, 0] += .06
    single = potential(m)
    m['pad_relative'][:, :2, 0] += .06
    far = potential(m)
    assert close > single > far
    assert reward(far, close, torch.tensor([False], device=device)).item() > 0
    assert reward(close, far, torch.tensor([False], device=device)).item() < 0


def test_pose_invariance_and_quaternion_normalization(device):
    m = measurement(device, 5)
    m['pad_relative'][1, :, 1] = .2
    # Include the end cap, outside corner, and an interior near-cap point.
    m['pad_relative'][2:] = m['pad_relative'].new_tensor([
        [[0., .25, 0.]] * 4,
        [[.018, .26, 0.]] * 4,
        [[.004, .249, 0.]] * 4,
    ])
    expected = potential(m)
    # Rotate local Y onto world -X (90 degrees around Z).
    p = m['pad_relative'].clone()
    m['pad_relative'] = torch.stack((-p[..., 1], p[..., 0], p[..., 2]), -1)
    m['object_quat'][:] = torch.tensor([math.sqrt(.5), 0, 0, math.sqrt(.5)], device=device) * 7
    torch.testing.assert_close(potential(m), expected, atol=1e-6, rtol=1e-6)
    m['object_quat'] *= -1
    torch.testing.assert_close(potential(m), expected, atol=1e-6, rtol=1e-6)
    # Translated world pads minus translated center give identical relative points.
    shift = p.new_tensor([4., -3., 8.])
    m['pad_relative'] = (m['pad_relative'] + shift) - shift
    torch.testing.assert_close(potential(m), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('point,distance', [
    ([.008, .25, 0], 0.), ([0, .25, 0], 0.),
    ([0, .26, 0], .01), ([.018, .26, 0], math.sqrt(.0002)),
    ([0, 0, 0], .008), ([.004, 0, 0], .004),
    ([.004, .249, 0], .001), ([.023, 0, 0], .015),
    ([0, .2, .008], 0.), ([0, -.26, 0], .01)])
def test_finite_cylinder_caps_edges_and_penetration(device, point, distance):
    m = measurement(device)
    m['pad_relative'][:] = m['pad_relative'].new_tensor(point)
    assert potential(m).item() == pytest.approx(20 * math.exp(-distance / .015))


def test_load_bottleneck_and_saturation(device):
    m = measurement(device, 5)
    m['hand_object_loads'][:] = torch.tensor([[0, 0], [100, 0], [.5, 100], [1, 1], [100, 100]], device=device)
    torch.testing.assert_close(potential(m), m['pad_relative'].new_tensor([20, 20, 30, 40, 40]))


def test_discounted_telescope_and_rollout_split(device):
    gamma = CONTRACT['gamma']
    assert gamma == .995
    states = torch.tensor([3., 20., 7., 40.], dtype=torch.float64, device=device)
    done = torch.tensor([False], device=device)
    pieces = [reward(states[i:i+1], states[i+1:i+2], done) for i in range(3)]
    total = sum(gamma**i * r for i, r in enumerate(pieces))
    torch.testing.assert_close(total, (gamma**3 * states[3] - states[0]).reshape(1))
    terminal = reward(states[3:], states.new_tensor([999.]), ~done)
    torch.testing.assert_close(total + gamma**3 * terminal, -states[:1])
    # Calling the stateless helper on either side of a rollout split changes nothing.
    torch.testing.assert_close(pieces[1], reward(states[1:2], states[2:3], done))
    torch.testing.assert_close(reward(states[:2], states[2:], torch.tensor([True, False], device=device)),
                               torch.stack((-states[0], gamma * states[3] - states[1])))


@pytest.mark.parametrize('key,shape', [('pad_relative', (1, 3, 3)), ('pad_relative', (4, 3)),
    ('object_quat', (1, 3)), ('object_quat', (2, 4)), ('hand_object_loads', (1, 2, 1)),
    ('hand_object_loads', (2, 2))])
def test_bad_shapes(key, shape):
    m = measurement()
    m[key] = torch.zeros(shape)
    with pytest.raises(ValueError):
        potential(m)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('key', ['pad_relative', 'object_quat', 'hand_object_loads'])
def test_nonfinite_measurements(key, value):
    m = measurement()
    m[key].reshape(-1)[0] = value
    with pytest.raises(ValueError):
        potential(m)


@pytest.mark.parametrize('name', ['radius_m', 'length_m'])
@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_bad_dimensions(name, value):
    with pytest.raises(ValueError):
        potential(measurement(), **{name: value})


def test_zero_quaternion_and_missing_field():
    m = measurement()
    m['object_quat'].zero_()
    with pytest.raises(ValueError):
        potential(m)
    del m['object_quat']
    with pytest.raises(ValueError):
        potential(m)


@pytest.mark.parametrize('before,after,done,gamma', [
    ([1], [[2]], [False], .995), ([1], [2, 3], [False], .995),
    ([1], [2], [0], .995), ([1], [2], [[False]], .995),
    ([float('nan')], [2], [False], .995), ([1], [float('inf')], [True], .995),
    ([1], [2], [False], float('nan')), ([1], [2], [False], -1),
    ([1], [2], [False], 1.1)])
def test_invalid_reward(before, after, done, gamma):
    with pytest.raises(ValueError):
        reward(torch.tensor(before), torch.tensor(after), torch.tensor(done), gamma)
