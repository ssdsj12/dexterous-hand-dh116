"""Original wrist oracle and pre-SDK safety checks for batched guard validation."""
import importlib
import math
from dataclasses import replace

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import project_policy_wrist_targets
from test_dh116_tensor_control import submission_fixture, reaction_identity

DEVICES = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])


def fused():
    name = 'go2_pvcnn.control.dh116.tabletop_tensor_wrist_projection'
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.fail('batched checked wrist projection is not implemented')
    return module.project_tensor_policy_wrist_targets


def inputs(n, device, dtype):
    g = torch.Generator().manual_seed(76)
    seed = torch.zeros(n, 2, 7, dtype=dtype)
    seed[..., :3] = torch.rand(n, 2, 3, generator=g, dtype=dtype)
    seed[..., 3:] = torch.randn(n, 2, 4, generator=g, dtype=dtype)
    seed[..., 3:] /= seed[..., 3:].norm(dim=-1, keepdim=True)
    measured = seed.clone()
    measured[..., 2] += torch.linspace(-.003, .012, n, dtype=dtype)[:, None]
    palms = torch.rand(n, 2, 3, generator=g, dtype=dtype) * .04 - .02
    request = palms + torch.randn(n, 2, 3, generator=g, dtype=dtype) * .03
    angles = torch.rand(n, 2, 3, generator=g, dtype=dtype) * math.pi - math.pi / 2
    angle_request = angles.flip(0).clone()
    # Exact angle boundaries, clearance on either side of 2.1 mm and palm clipping.
    angles[0, 0, 0] = math.pi / 2
    angle_request[-1, 1, 2] = -math.pi / 2
    request[0] = torch.tensor([[-1., -1., -1.], [1., 1., 1.]], dtype=dtype)
    clearance = torch.full((n, 2), .0021, dtype=dtype)
    clearance[0] += torch.tensor([-1e-9, 1e-9], dtype=dtype)
    data = dict(root_seed_w=seed, measured_base_w=measured,
        common_lift_offset_m=torch.linspace(0., .01, n, dtype=dtype),
        measured_clearance_m=clearance, palm_offsets_m=palms,
        requested_palm_offsets_m=request, angles_rad=angles, requested_angles_rad=angle_request,
        allowed=(torch.arange(n) % 3 != 1), live=(torch.arange(n) % 3 != 2))
    return {key: value.to(device) for key, value in data.items()}


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('n', [2, 128])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_batched_projection_is_bitwise_original_oracle(n, device, dtype):
    data = inputs(n, device, dtype)
    snapshots = {key: value.clone() for key, value in data.items()}
    expected = project_policy_wrist_targets(**data)
    actual = fused()(**data)
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        else:
            assert actual[key] == value
    for key, value in snapshots.items():
        torch.testing.assert_close(data[key], value, rtol=0, atol=0)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('field', [
    'root_seed_w', 'measured_base_w', 'common_lift_offset_m', 'measured_clearance_m',
    'palm_offsets_m', 'requested_palm_offsets_m', 'angles_rad', 'requested_angles_rad'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_in_every_projection_input_is_rejected(field, bad, device):
    data = inputs(2, device, torch.float64)
    data[field].reshape(-1)[-1] = bad
    for fn in (project_policy_wrist_targets, fused()):
        with pytest.raises(ValueError):
            fn(**data)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('field', ['root_seed_w', 'measured_base_w'])
@pytest.mark.parametrize('offset', [-1.00001e-5, -0.99999e-5, 0.99999e-5, 1.00001e-5])
def test_quaternion_unit_boundary_matches_original(field, offset, device):
    data = inputs(2, device, torch.float64)
    data[field][0, 0, 3:] = data[field].new_tensor([1. + offset, 0., 0., 0.])
    try:
        expected = project_policy_wrist_targets(**data)
    except ValueError:
        with pytest.raises(ValueError):
            fused()(**data)
    else:
        actual = fused()(**data)
        torch.testing.assert_close(actual['projected_root_poses'], expected['projected_root_poses'], rtol=0, atol=0)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('field', ['angles_rad', 'requested_angles_rad'])
@pytest.mark.parametrize('sign', [-1, 1])
def test_angle_just_outside_original_limit_is_rejected(field, sign, device):
    data = inputs(2, device, torch.float64)
    data[field][0, 0, 0] = sign * (math.pi / 2 + 1e-12)
    for fn in (project_policy_wrist_targets, fused()):
        with pytest.raises(ValueError):
            fn(**data)


@pytest.mark.parametrize('fault', ['shape', 'dtype', 'device', 'mask_dtype', 'mask_shape'])
def test_projection_metadata_rejected_before_any_numeric_inspection(fault, monkeypatch):
    data = inputs(2, 'cpu', torch.float64)
    if fault == 'shape': data['angles_rad'] = data['angles_rad'][:1]
    if fault == 'dtype': data['angles_rad'] = data['angles_rad'].float()
    if fault == 'device': data['measured_base_w'] = data['measured_base_w'].to('meta')
    if fault == 'mask_dtype': data['live'] = data['live'].float()
    if fault == 'mask_shape': data['allowed'] = data['allowed'][:1]
    def premature_numeric(*args, **kwargs):
        raise AssertionError('metadata must be checked before numeric inspection')
    monkeypatch.setattr(torch, 'isfinite', premature_numeric)
    monkeypatch.setattr(torch.linalg, 'vector_norm', premature_numeric)
    with pytest.raises(ValueError):
        fused()(**data)


@pytest.mark.parametrize('device', DEVICES)
def test_projection_has_one_host_numeric_decision(device):
    from torch.utils._python_dispatch import TorchDispatchMode
    fn = fused()
    data = inputs(128, device, torch.float64)
    fn(**data)
    class ScalarReads(TorchDispatchMode):
        def __init__(self):
            self.count = 0
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten._local_scalar_dense.default:
                # The unchanged palm_transport validates its static floor on CPU.
                if args[0].device.type == torch.device(device).type:
                    self.count += 1
            return func(*args, **(kwargs or {}))
    with ScalarReads() as reads:
        fn(**data)
    assert reads.count == (2 if device == 'cpu' else 1)


@pytest.mark.parametrize('fault', ['nan_expected_right', 'inf_com', 'fp32_torque_overflow', 'right_over_force'])
def test_combined_submit_guards_still_reject_before_all_native_mutations(fault):
    control, stamp, state, tr, wrenches, writes = submission_fixture()
    if fault == 'nan_expected_right': tr['expected'][1][0, 0] = float('nan')
    if fault == 'inf_com': state['body_com_pose_w'][0, 1, 0, 0] = float('inf')
    if fault == 'fp32_torque_overflow': wrenches[0, 1, 3] = 1e100
    if fault == 'right_over_force': wrenches[0, 1, 0] = 30. + 1e-7
    with pytest.raises(ValueError):
        control.submit(tr, stamp=stamp, state=state, wrenches=wrenches,
            reaction_source=reaction_identity(stamp, wrenches))
    assert writes == []
    assert control._failed and tr['phase'] == 'failed'


def test_prepare_rejects_mutated_prestate_on_same_state_dictionary():
    control, stamp, state, tr, wrenches, writes = submission_fixture()
    control.invalidate()
    state['qd'][0, 1, 0] = float('nan')
    with pytest.raises(ValueError):
        control.prepare(torch.zeros(2, 24), state, torch.full((2, 2), .003),
            state['root_pose_wxyz'], stamp=stamp)
    assert not writes

def control_on_device(device, n=2):
    control, stamp, state, tr, wrenches, writes = submission_fixture()
    def move(value):
        if n != 2:
            value = value[:1].repeat((n,) + (1,) * (value.ndim - 1))
        return value.to(device)
    state = {key: move(value) for key, value in state.items()}
    control = type(control)(control.batch, move(control.roots),
        {"left.pad": dict(side="left", body_index=0), "right.pad": dict(side="right", body_index=0)},
        ("left.pad", "right.pad", "table"))
    if n != 2:
        stamp = replace(stamp, parent_num_envs=n, env_rows=tuple(range(n)),
            episode_uids=tuple(f"episode-{i}" for i in range(n)),
            rod_uids=tuple(f"rod-{i}" for i in range(n)), reset_ids=(1,) * n, rod_ticks=(0,) * n)
    tr = control.prepare(torch.zeros(n, 24, device=device), state,
        torch.full((n, 2), .003, device=device), state['root_pose_wxyz'], stamp=stamp)
    return control, stamp, state, tr, move(wrenches), writes


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA validation synchronization contract')
@pytest.mark.parametrize('phase,maximum', [('prepare', 3), ('submit', 1)])
def test_control_batches_gpu_numeric_decisions_before_native_write(phase, maximum):
    from torch.utils._python_dispatch import TorchDispatchMode
    control, stamp, state, tr, wrenches, writes = control_on_device('cuda:0')
    class Reads(TorchDispatchMode):
        def __init__(self): self.count = 0
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten._local_scalar_dense.default and args[0].device.type == 'cuda':
                self.count += 1
            return func(*args, **(kwargs or {}))
    with Reads() as reads:
        if phase == 'prepare':
            control.pending = None
            control.prepare(torch.zeros(2, 24, device='cuda:0'), state,
                torch.full((2, 2), .003, device='cuda:0'), state['root_pose_wxyz'], stamp=stamp)
        else:
            control.submit(tr, stamp=stamp, state=state, wrenches=wrenches,
                reaction_source=reaction_identity(stamp, wrenches))
    assert reads.count <= maximum


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('n', [2, 128])
@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('force', [30. - 1e-7, 30., 30. + 1e-7])
def test_fp64_force_boundary_is_checked_in_last_lane_before_sdk_cast(n, side, force, device):
    control, stamp, state, tr, wrenches, writes = control_on_device(device, n)
    wrenches[-1, side, 0] = force
    if force > 30.:
        with pytest.raises(ValueError, match='30 N'):
            control.submit(tr, stamp=stamp, state=state, wrenches=wrenches,
                reaction_source=reaction_identity(stamp, wrenches))
        assert not writes
    else:
        control.submit(tr, stamp=stamp, state=state, wrenches=wrenches,
            reaction_source=reaction_identity(stamp, wrenches))
        assert writes.count('write') == 1
        issued = [value for value in writes if isinstance(value, dict)]
        assert len(issued) == 2
        torch.testing.assert_close(issued[side]['forces'][-1, 0, 0], wrenches[-1, side, 0].float(), rtol=0, atol=0)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('n', [2, 128])
def test_com_shift_keeps_fp64_subtraction_until_final_sdk_values(n, device):
    control, stamp, state, tr, wrenches, writes = control_on_device(device, n)
    wrenches[:, 0] = wrenches.new_tensor([.5, -.2, 1., .03, .02, -.01])
    wrenches[:, 1] = -wrenches[:, 0]
    state['link_pose_xyzw'][..., 0] = 48.125
    state['body_com_pose_w'][..., 0] = 48.13500000023
    state['body_com_pose_w'][..., 1] = .003
    control.submit(tr, stamp=stamp, state=state, wrenches=wrenches,
        reaction_source=reaction_identity(stamp, wrenches))
    from scripts.verify_dh116_deformx_rigid_bridge import wrench_at_com
    for side, issued in enumerate(value for value in writes if isinstance(value, dict)):
        expected = torch.tensor(wrench_at_com(wrenches[0, side].cpu().numpy(),
            state['link_pose_xyzw'][0, side, 0, :3].cpu().numpy(),
            state['body_com_pose_w'][0, side, 0, :3].cpu().numpy()), device=device)
        torch.testing.assert_close(issued['forces'][:, 0], expected[:3].float().expand(n, 3), rtol=0, atol=0)
        torch.testing.assert_close(issued['torques'][:, 0], expected[3:].float().expand(n, 3), rtol=0, atol=0)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('field', ['link_pose_xyzw', 'body_com_pose_w', 'wrenches', 'command'])
def test_same_buffers_revalidated_after_successful_prior_sdk(field, device):
    control, stamp, state, tr, wrenches, writes = control_on_device(device)
    source = reaction_identity(stamp, wrenches)
    control.submit(tr, stamp=stamp, state=state, wrenches=wrenches, reaction_source=source)
    for side in (0, 1):
        state['backend_targets'][:, side, control.write_ids[side]] = tr['expected'][side]
    after = replace(stamp, sdk_step=stamp.sdk_step+1, rod_ticks=tuple(t+12 for t in stamp.rod_ticks),
        solver_time_s=stamp.solver_time_s+1/480)
    control.finish(tr, state, stamp=after)
    tr = control.prepare(torch.zeros(2, 24, device=device), state,
        torch.full((2, 2), .003, device=device), state['root_pose_wxyz'], stamp=after)
    writes.clear()
    if field == 'command': tr['expected'][1][-1, -1] = float('nan')
    elif field == 'wrenches': wrenches[-1, -1, -1] = float('nan')
    else: state[field][-1, -1, -1, -1] = float('nan')
    with pytest.raises(ValueError):
        control.submit(tr, stamp=after, state=state, wrenches=wrenches,
            reaction_source=reaction_identity(after, wrenches))
    assert not writes
