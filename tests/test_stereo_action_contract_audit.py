"""Phase 0 CPU audit: actual decoder arithmetic, never native qualification.

The SDK-facing advance method uses existing CPU fixtures with inert SDK sinks.
No simulator, GPU, training loop, checkpoint, or historical native data is opened.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116 import tabletop_surface_contract as surface
from go2_pvcnn.control.dh116 import tabletop_command_kernels as kernels
from go2_pvcnn.control.dh116 import tabletop_vision_command_shadow as shadow
from go2_pvcnn.control.dh116 import tabletop_squashed_gaussian as gaussian
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, MIMIC, active_limits, expand
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import compose_rotation


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs/contracts/stereo_action_contract.json'


def audit():
    assert MANIFEST.exists(), 'Phase 0 machine-readable action audit is required'
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


def controller_fixture():
    # Instantiate no environment: only real controller/advance methods and the
    # existing inert SDK sink fixture. This avoids Gym, Isaac and training.
    from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
    from test_dh116_tensor_world import palm_world
    w, _ = palm_world()
    t = TensorTask.__new__(TensorTask)
    t.n = 3; t.device = torch.device('cpu'); t.action_size = 24
    t.low, t.high = (torch.tensor(x, dtype=torch.float64).repeat(2) for x in active_limits())
    t.command = ((t.low+t.high)/2).repeat(3, 1)
    t.m = dict(q=t.command.clone(), qd=torch.zeros_like(t.command),
        mimic_errors=torch.zeros(3, 10, dtype=torch.float64),
        hand_object_loads=torch.zeros(3, 2, dtype=torch.float64))
    t.policy_control = t.free_wrist_grasp = t.surface_force_sharing = t.continuous_motion = True
    t.surface_policy_owned_targets = True; t.surface_redistribution = False
    t.shaping = None; t.free_wrist_world = w; t.palm_descent_clearance_m = .0003
    t.parameters = SimpleNamespace(deadband_n=.1)
    t.actions = torch.zeros(3, 24, dtype=torch.float64); t.previous_actions = t.actions.clone()
    t.old_command = t.command.clone()
    t.joint_requested_target = t.command.clone(); t.joint_governed_target = t.command.clone()
    for name in ('palm_targets', 'rotation_targets', 'palm_actions'):
        setattr(t, name, torch.zeros(3, 2, 3, dtype=torch.float64))
    for name in ('policy_stiffness', 'f_target', 'f_requested', 'f_filtered', 'f_offset', 'f_boost'):
        setattr(t, name, torch.zeros(3, 2, dtype=torch.float64))
    for name in ('pressure_limited', 'f_blocked'):
        setattr(t, name, torch.zeros(3, 2, dtype=torch.bool))
    t.palm_clipped = torch.zeros(3, 2, 3, dtype=torch.bool)
    for name in ('offset', 'reward', 'used'):
        setattr(t, name, torch.zeros(3, dtype=torch.float64))
    for name in ('success', 'paused', 'nonfinite'):
        setattr(t, name, torch.zeros(3, dtype=torch.bool))
    t.failure = torch.zeros(3, dtype=torch.long); t.phase = torch.full((3,), 3)
    t.terms = dict(failure=torch.zeros(3, dtype=torch.float64))
    w.wrist_angles = torch.zeros(3, 2, 3, dtype=torch.float64)
    w.commands = t.command.reshape(3, 2, 6).clone()
    w.clearances.fill_(.01)
    return t, w


def test_manifest_has_exact_source_digests_and_canonical_semantic_sha():
    value = audit()
    assert value['scope'] == 'Phase0_read_only_source_audit_CPU_only'
    assert value['action_contract'] == surface.ACTION_CONTRACT
    digest = hashlib.sha256(json.dumps(surface.ACTION_CONTRACT, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert value['action_contract_sha256'] == digest == shadow.action_contract_sha256()
    for filename, expected in value['source_sha256'].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected, filename
    rows = value['dimensions']
    assert [r['index'] for r in rows] == list(range(24))
    assert len({r['name'] for r in rows}) == 24
    for row in rows:
        assert set(('index', 'name', 'unit', 'scale', 'bounds', 'sign',
                    'rate_limit', 'control_meaning', 'coordinate_frame')) <= row.keys()
    assert [r['name'] for r in rows[:12]] == surface.ACTION_CONTRACT['joint_order']
    assert [r['scale'] for r in rows] == [.3]*12 + [.01]*6 + [.5]*6


@pytest.mark.parametrize('index', range(24))
def test_every_canonical_basis_reaches_actual_control_and_sdk_targets(index):
    t, w = controller_fixture()
    value = audit()['dimensions'][index]
    action = torch.zeros(3, 24, dtype=torch.float64)
    action[:, index] = -.6 if index % 2 else .6
    initial = shadow.command_state(t, w)
    contract = shadow.controller_contract(t, w)
    context = shadow.measured_context(t, w, torch.ones(3, dtype=torch.bool))
    allowed = t.control(action, context['live'], w.dt)
    context['allowed'] = allowed
    w.advance(t, allowed, context['live'])
    new, out = shadow.shadow_step(initial, action, context, contract)
    groups = [('joint_command_rad', t.command, 0, 12),
              ('palm_command_offset_m', w.palms, 12, 18),
              ('wrist_command_angle_rad', w.wrist_angles, 18, 24)]
    for key, actual, low, high in groups:
        torch.testing.assert_close(new[key], actual, rtol=0, atol=0)
        expected = torch.zeros_like(actual).flatten(1)
        if low <= index < high:
            expected[:, index-low] = action[:, index] * value['scale'] * w.dt
        torch.testing.assert_close((actual-initial[key]).flatten(1), expected, atol=1e-16, rtol=0)
    assert torch.equal(out['sdk_float32']['finger'], w.commands.flatten(1).float())
    torch.testing.assert_close(out['root_pose'], w.root_targets, rtol=0, atol=0)


def test_per_hand_vector_rate_limit_and_position_angle_bounds():
    action = torch.ones(2, 24, dtype=torch.float64)
    rates = dict(joint_rate_rad_s=.3, wrist_rate_m_s=.01, rotation_rate_rad_s=.5)
    start = torch.zeros(2, 2, 3, dtype=torch.float64)
    palms, angles = kernels.wrist_requested_targets(start, start, action, rates, 1/480)
    torch.testing.assert_close(palms.norm(dim=-1), torch.full((2, 2), .01/480, dtype=torch.float64))
    torch.testing.assert_close(angles.norm(dim=-1), torch.full((2, 2), .5/480, dtype=torch.float64))
    assert not torch.allclose(palms, torch.full_like(palms, .01/480))
    palms, angles = kernels.wrist_requested_targets(torch.full_like(start, 1.),
        torch.full_like(start, 2.), action, rates, 1/480)
    torch.testing.assert_close(palms, palms.new_tensor([.03, .03, .05]).expand_as(palms))
    torch.testing.assert_close(angles, torch.full_like(angles, np.pi/2))


def test_mimic_feasible_bounds_and_policy_owned_zero_action_behavior():
    lower, upper = active_limits()
    assert lower[1] == pytest.approx(.00697725445/.88178789)
    assert upper[1] == pytest.approx(1.030)
    for q in (lower, (lower+upper)/2, upper):
        full = expand(q)
        for follower, master, ratio, offset in MIMIC:
            assert full[JOINTS.index(follower)] == pytest.approx(q[ACTIVE.index(master)]*ratio+offset)
    t, w = controller_fixture(); before = t.command.clone()
    t.m['mimic_errors'][:, 0] = -.03
    zeros = torch.zeros(3, 24, dtype=torch.float64)
    t.control(zeros, torch.ones(3, dtype=torch.bool), w.dt)
    torch.testing.assert_close(t.command, before, atol=0, rtol=0)
    t.surface_policy_owned_targets = False
    t.control(zeros, torch.ones(3, dtype=torch.bool), w.dt)
    assert (t.command[:, 1] < before[:, 1]).all()
    assert ((t.command-before).abs() <= .3*w.dt+1e-16).all()


def test_zero_action_retains_targets_subject_to_joint_lead_and_table_projection():
    t, w = controller_fixture()
    t.m['q'][:, 0] = t.command[:, 0]-.2
    before = t.command.clone(); palms = w.palms.clone()
    w.clearances.fill_(-.001)
    live = torch.ones(3, dtype=torch.bool)
    allowed = t.control(torch.zeros(3, 24), live, w.dt)
    assert (t.command[:, 0] < before[:, 0]).all()
    assert ((t.command-before).abs() <= .3*w.dt+1e-16).all()
    w.advance(t, allowed, live)
    assert (w.palms[:, :, 2]-palms[:, :, 2] > .01*w.dt).all()
    assert w.last_action_diagnostics['projection_reasons'].bitwise_and(2).all()


def test_rotation_is_seed_times_noncommuting_intrinsic_xyz_not_world_rotation_vector():
    seed_r = Rotation.from_euler('Z', .7)
    angles = np.array([.31, -.27, .43])
    xyzw = seed_r.as_quat()
    seed = torch.tensor([[xyzw[3], *xyzw[:3]]], dtype=torch.float64)
    actual = compose_rotation(seed*7., torch.tensor(angles[None], dtype=torch.float64)).numpy()[0]
    actual_r = Rotation.from_quat(actual[[1, 2, 3, 0]])
    expected = seed_r * Rotation.from_euler('XYZ', angles)
    np.testing.assert_allclose(actual_r.as_matrix(), expected.as_matrix(), atol=4e-16, rtol=0)
    assert np.linalg.norm(actual_r.as_matrix()-(Rotation.from_rotvec(angles)*seed_r).as_matrix()) > .1


def test_physical_adapter_roundtrip_uses_named_joints_but_does_not_apply_vector_norm():
    from test_bimanual_physical_action_adapter import models, poses
    from go2_pvcnn.control.dh116.tabletop_bimanual_physical_action_adapter import BimanualPhysicalActionAdapter
    adapter = BimanualPhysicalActionAdapter(models())
    action = np.linspace(-1., 1., 24)
    action[12:15] = 1.
    decoded = adapter.decode(action, poses(), poses())
    np.testing.assert_allclose(adapter.to_sdk(decoded), action, atol=2e-16, rtol=0)
    assert np.linalg.norm(decoded['left']['translation_world_m_s']) == pytest.approx(np.sqrt(3)*.01)
    names = ['external_wrist'] + list(reversed(JOINTS))
    actual = adapter.sdk_joint_velocity(decoded['right'], names)
    assert actual[names.index('finger12')] == pytest.approx(action[7]*.3)
    assert actual[names.index('finger13')] == pytest.approx(action[7]*.3*MIMIC[0][2])
    assert actual[0] == 0.
    with pytest.raises(ValueError, match='explicit matching'):
        adapter.adapt_for_checkpoint(action, poses(), poses(), {})


def test_mount_sdk_targets_are_float32_named_local_translation_and_intrinsic_angles():
    from go2_pvcnn.control.dh116.tabletop_palm_mount import MountView
    names = ['wrist_yaw', 'other', 'wrist_z', 'wrist_x', 'wrist_pitch', 'wrist_y', 'wrist_roll']
    # A 90-degree world-Z anchor makes world +X become local -Y. These
    # external support drives do not inherit DH116 finger articulation order.
    anchor = torch.tensor([[0., 0., 0., 2**-.5, 0., 0., 2**-.5]], dtype=torch.float64)
    calls = []
    hand = SimpleNamespace(joint_names=names, body_names=['base_link'],
        data=SimpleNamespace(root_pose_w=anchor),
        set_joint_position_target=lambda target, **kw: calls.append((target, kw)))
    mount = MountView(hand)
    target = anchor.clone(); target[0, 0] = .01
    angles = torch.tensor([[.2, -.3, .4]], dtype=torch.float64)
    mount.move(target, angles=angles)
    actual, kwargs = calls[0]
    assert actual.dtype == torch.float32
    assert [names[i] for i in kwargs['joint_ids']] == [
        'wrist_x', 'wrist_y', 'wrist_z', 'wrist_roll', 'wrist_pitch', 'wrist_yaw']
    torch.testing.assert_close(actual, torch.tensor([[0., -.01, 0., .2, -.3, .4]]), atol=1e-8, rtol=0)


@pytest.mark.parametrize('bad', [torch.zeros(24), torch.zeros(1, 26), torch.zeros(1, 24, dtype=torch.int64),
    torch.full((1,24), float('nan')), torch.full((1,24), float('inf')), torch.full((1,24), 1.01)])
def test_strict_canonical_validator_rejects_shape_dtype_nonfinite_and_unbounded(bad):
    with pytest.raises(ValueError, match='finite bounded normalized'):
        surface.validate_actions(bad)


def test_begin_step_raw_diagnostic_is_post_clamp_and_finite_invalid_rows_are_distinct():
    t, _ = controller_fixture()
    t.action_size = 24; t.actions = torch.zeros(3, 24, dtype=torch.float64)
    t.previous_actions = torch.zeros_like(t.actions)
    request = torch.zeros(3, 24); request[0, 0] = 9.; request[1, 1] = float('nan')
    decoded = t.begin_step(request)
    assert decoded[0, 0] == 1. and request[0, 0] == 9.
    assert decoded[1].eq(0.).all() and t.failure[1] != 0
    assert torch.equal(t.last_raw_action24, decoded)
    with pytest.raises(ValueError, match='N x 24'):
        t.begin_step(torch.zeros(3, 26))


def test_existing_saved_latent_gaussian_handles_rounded_tanh_endpoints_without_inverse():
    z = torch.tensor([[20., -20., .25]], dtype=torch.float64, requires_grad=True)
    mu = torch.tensor([[.1, -.2, .3]], dtype=torch.float64)
    std = torch.full_like(mu, .2)
    bounded = z.tanh()
    assert bounded[0, :2].abs().eq(1.).all()
    actual = gaussian.squashed_log_prob(z, mu, std)
    expected = (torch.distributions.Normal(mu, std).log_prob(z)
        -2*(np.log(2.)-z-torch.nn.functional.softplus(-2*z))).sum(-1)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward(); assert torch.isfinite(z.grad).all()
    torch.testing.assert_close(gaussian.latent_gaussian_kl(mu, std, mu, std), torch.zeros(1, dtype=torch.float64))
    assert gaussian.DISTRIBUTION_CONTRACT['deterministic_action'] == 'tanh_latent_mean'
    with pytest.raises(ValueError, match='positive'):
        gaussian.squashed_log_prob(z.detach(), mu, torch.zeros_like(std))


def test_full_schema_and_sequence_boundary_reject_unknown_semantics_and_sha():
    legacy = dict(observation_size=351, observation_version=surface.LEGACY_OBSERVATION_VERSION,
                  policy_command_size=24, action=surface.ACTION_NAME, physics_hz=480, control_hz=60)
    contract = surface.bind_contract(legacy)
    surface.validate_contract(contract)
    for field, replacement in [('surface_contract_sha256', '0'*64), ('policy_command_size', 26), ('action', 'unknown')]:
        other = copy.deepcopy(contract); other[field] = replacement
        with pytest.raises(ValueError): surface.validate_contract(other)
    from go2_pvcnn.control.dh116.tabletop_vision_supervision_dataset import validate_sequence_context, SEQUENCE_CONTEXT_VERSION
    with pytest.raises(ValueError, match='action_contract_sha256'):
        validate_sequence_context(dict(version=SEQUENCE_CONTEXT_VERSION,
            dataset_sha256='a'*64, controller_source_sha256=shadow.controller_source_sha256(),
            action_contract_sha256='0'*64))


def test_shadow_digest_check_gap_is_explicit_and_not_confused_with_sequence_boundary():
    t, w = controller_fixture(); contract = shadow.controller_contract(t, w)
    contract['action_contract_sha256'] = '0'*64
    # Current shadow validates the actual action dict and source digest, not
    # this redundant digest field. Record the gap rather than changing src.
    shadow._validate_contract(contract)
    contract['controller_source_sha256'] = '0'*64
    with pytest.raises(ValueError, match='source SHA256'):
        shadow._validate_contract(contract)


def test_audit_cpu_imports_have_no_simulator_dependency_or_gpu_tensor():
    forbidden = ('isaacsim', 'isaaclab', 'omni', 'pxr', 'carb')
    assert not any(name.split('.')[0] in forbidden for name in sys.modules)
    assert torch.zeros(1).device.type == 'cpu'
    for module in (surface, kernels, shadow, gaussian):
        tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        for node in imports:
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
            assert not any(name.split('.')[0] in forbidden for name in names)
