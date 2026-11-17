"""Phase 0 CPU arithmetic/source contracts, never native physics qualification."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_surface_reward_ablation as ablation
from go2_pvcnn.control.dh116.tabletop_phase_force_reward import reward_terms
from go2_pvcnn.control.dh116.tabletop_surface_exploration import (
    enable_stable_hold_rewards, qualified_pad_contact, reward_rates,
)
from go2_pvcnn.control.dh116.tabletop_surface_success import SurfaceLiftSuccess
from test_surface_success import scene, advance

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'docs/contracts/privileged_teacher_reward_contract.json'


def contract():
    return json.loads(CONTRACT.read_text(encoding='utf-8'))


def configured_task(profile='baseline'):
    """Synthetic measurement fixture passed through the actual TensorTask."""
    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    from go2_pvcnn.control.dh116.tabletop_force_feedback import ReactionForceController
    from go2_pvcnn.control.dh116.tabletop_load_parameters import LoadParameters
    from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
    from go2_pvcnn.control.dh116.tabletop_policy_force_region import enable as enable_force
    from go2_pvcnn.control.dh116.tabletop_free_grasp_lift import enable as enable_free
    from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import enable as enable_wrist
    from go2_pvcnn.control.dh116.tabletop_surface_exploration import enable as enable_surface
    low, high = active_limits()
    command = np.tile((low + high) / 2, 2)
    m = dict(q=command.copy(), qd=np.zeros(12), forces=np.full(4, .055), clearance=.01,
        object_relative=np.array([0., 0., -.02]), object_quat=np.array([1., 0., 0., 0.]),
        object_velocity=np.zeros(6), bottom=.36, approach_error=0., mimic_errors=np.zeros(10),
        hand_clearance=np.full(2, .01), adaptive_palm_offset=np.zeros((2, 3)),
        grip_stiffness=np.full(2, 10.), pad_relative=np.zeros((4, 3)),
        vertical_hand_force_n=np.zeros(2), object_link_loads=np.full((2, 1), .11),
        grouped_object_loads=np.array([[0., 0., 0., 0., 0., .11]] * 2),
        hand_object_loads=np.full(2, .11), joint_failure=None)
    template = SimpleNamespace(force_controller=ReactionForceController(
        LoadParameters(max_force_n=12., force_rate_n_s=None), full_hand_contact=True),
        stage='full', closed=False, full_hand_contact=True, joint_envelope=None,
        backend=SimpleNamespace(object_contact_links=('palm_link',)), measurement=m,
        reference=command.copy(), command=command.copy(), previous=command.copy(),
        filtered_residual=np.zeros(12), offset=.004, close_progress=0., descent_stop=None,
        force_direction=np.ones((2, 6)) / np.sqrt(6), force_palm_compensation=None,
        pressure_palm_target=np.zeros((2, 3)), pressure_palm_clipped=np.zeros(2, dtype=bool),
        task=SimpleNamespace(phase='close', elapsed=0., contact_time=0., success_time=0.,
            success=False, failure=None), grip_relative=None, metrics={'slip_m': 0.})
    task = TensorTask(template, 2, 'cpu', pressure_relief=True,
        contact_reward_window='close_start', vertical_force_reward=True,
        vertical_force_reward_mode='continuous', force_reward_start='downward_after_close',
        grasp_readiness='measured_load', skip_hold=True, close_downward_penalty=True)
    task.enable_policy_control()
    enable_force(task)
    enable_free(task)
    pose = torch.zeros(2, 2, 7, dtype=torch.float64)
    pose[:, :, 3] = 1.
    task.m.update(wrist_pose=pose.clone(), wrist_velocity=torch.zeros(2, 2, 6, dtype=torch.float64),
        wrist_joint_angles=torch.zeros(2, 2, 3, dtype=torch.float64))
    world = SimpleNamespace(n=2, device='cpu', continuous_motion=True, free_wrist_grasp=True,
        palms=torch.zeros(2, 2, 3, dtype=torch.float64),
        wrist_angles=torch.zeros(2, 2, 3, dtype=torch.float64), root_seed=pose,
        lift_reference_bottom_m=.36,
        w=SimpleNamespace(mounts=[SimpleNamespace(angular_ids=[3, 4, 5])] * 2))
    world.measure = lambda: task.m
    enable_wrist(task, world)
    task.m.update(effective_pad_loads_n=torch.zeros(2, 2, 5, dtype=torch.float64),
        effective_pad_vectors_w=torch.zeros(2, 2, 5, 3, dtype=torch.float64),
        effective_pad_points_w=torch.zeros(2, 2, 5, 3, dtype=torch.float64),
        effective_pad_valid=torch.zeros(2, 2, 5, dtype=torch.bool),
        rejected_contact_loads_n=torch.zeros(2, 2, dtype=torch.float64))
    enable_surface(task, world, redistribution=False)
    task.m['bottom'].fill_(.36)
    task.m['contact_valid'].fill_(True)
    task.m['contact_center_relative_m'].zero_()
    task.m['effective_pad_loads_n'][:, :, :2] = 1.
    task.m['effective_pad_vectors_w'][:, :, 0, 0] = 1.
    task.m['effective_pad_vectors_w'][:, :, 1, 0] = -1.
    task.m['effective_pad_vectors_w'][:, :, :2, 2] = -.25
    task.m['vertical_hand_force_n'].fill_(.5)
    task.m['effective_pad_valid'][:, :, :2] = True
    task.surface_reward_profile = profile
    enable_stable_hold_rewards(task)
    task.phase_force_reward = True
    for name in ('grasp_upward_force', 'lift_upward_force'):
        task.terms[name] = torch.zeros_like(task.reward)
    if profile != 'baseline':
        ablation.enable(task)
    return task, world


def consume(task, dt=1/480, lift_force=1.):
    measurement = {key: value.clone() for key, value in task.m.items()}
    task.consume(measurement, ~task.done, dt,
        vertical_force_n=torch.ones(task.n, dtype=torch.float64),
        lift_vertical_force_n=torch.full((task.n,), lift_force, dtype=torch.float64),
        hand_lift_force_n=torch.full((task.n, 2), lift_force / 2, dtype=torch.float64),
        object_weight_n=torch.ones(task.n, dtype=torch.float64))


def test_contract_has_source_identity_and_does_not_claim_native_qualification():
    value = contract()
    assert value['version'] == 'dh116-privileged-teacher-reward-audit-phase0-v1'
    assert value['base_commit'] == '6cee7418a5e49f56b7f0090fc5eaf20b143c0b1b'
    assert value['physical_qualification'] == 'NA_not_established_by_this_audit'
    assert value['active_run_profile'] is None
    assert value['teacher_r_profiles_implemented'] is False
    assert value['formal_success']['net_lift_m'] == .03
    assert value['formal_success']['continuous_hold_s'] == 5.
    assert value['formal_success']['fixed_anchor_slip_m'] == .0005
    for name, expected in value['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    data = {k: v for k, v in value.items() if k != 'contract_sha256'}
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    assert hashlib.sha256(canonical.encode('utf-8')).hexdigest() == value['contract_sha256']


def test_contract_profiles_and_coefficients_match_imported_helpers():
    value = contract()
    assert value['profiles'] == {name: list(terms) for name, terms in ablation.STAGE_TERMS.items()}
    assert value['coefficients'] == json.loads(json.dumps(ablation.COEFFICIENTS))
    assert value['disabled_legacy_terms'] == {
        name: list(ablation.disabled_legacy_terms(name)) for name in ablation.PROFILES}
    assert value['milestone_seconds'] == list(ablation.MILESTONE_SECONDS)
    assert not any(name.startswith('teacher_r') for name in ablation.PROFILES)


@pytest.mark.parametrize('filename', [
    'scripts/train_dh116_tabletop_gpu_ppo.py',
    'scripts/run_dh116_surface_student.py',
    'scripts/train_dh116_surface_multistyle.py',
])
def test_profile_default_is_baseline_without_executing_launchers(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8-sig'))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
        and any(isinstance(arg, ast.Constant) and arg.value == '--surface-reward-profile'
                for arg in node.args)]
    assert len(calls) == 1
    assert next(kw.value.value for kw in calls[0].keywords if kw.arg == 'default') == 'baseline'


@pytest.mark.parametrize('profile', ['baseline', 'v3_reference', 'early_hold', 'v3_slip_only'])
def test_actual_tensor_task_substeps_sum_terms_once_per_control_step(profile):
    task, _ = configured_task(profile)
    assert task.shaping is None  # Actual free-wrist baseline does not enable GraspShaping.
    task.m['bottom'].fill_(.391)
    task.begin_step(torch.zeros(2, 24))
    for _ in range(8):
        consume(task)
    _, reward, done, extra = task.end_step()
    assembled = torch.stack(list(extra['terms'].values())).sum(0)
    torch.testing.assert_close(reward, assembled, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(extra['used_time'], torch.full_like(reward, 1/60))
    assert not done.any()
    assert extra['terms']['contact'].eq(0).all()  # Replaced by native surface reward.
    assert extra['terms']['lift'].eq(0).all()  # Free-wrist replaces legacy height delta.
    assert extra['terms']['grasp_upward_force'].min() > 0
    assert extra['terms']['lift_upward_force'].min() > 0


def test_v4_replacement_retains_old_hold_and_dwell_overlap_explicitly():
    task, _ = configured_task('early_hold')
    task.m['bottom'].fill_(.391)
    task.surface_lift_success._duration.fill_(1.)
    rates = reward_rates(task)
    assert rates['dex_lift'].eq(0).all()
    assert rates['surface_hold_progress'].eq(120 * (1/5)**2).all()
    assert 'surface_hold_progress' not in ablation.disabled_legacy_terms('early_hold')
    assert 'lift_dwell' not in ablation.disabled_legacy_terms('early_hold')
    task.begin_step(torch.zeros(2, 24))
    for _ in range(48):
        consume(task)
    _, _, _, extra = task.end_step()
    for name in ('surface_hold_progress', 'lift_dwell', 'surface_early_hold_rate', 'surface_hold_milestone'):
        assert extra['terms'][name].min() > 0, name


def test_terminal_timeout_and_done_noop_do_not_repay_with_real_caller_mask():
    task, _ = configured_task()
    task.elapsed.fill_(30.)
    task.begin_step(torch.zeros(2, 24))
    consume(task)
    first = task.reward.clone()
    assert task.done.all() and task.terms['failure'].eq(-25.).all()
    for _ in range(7):
        consume(task)
    torch.testing.assert_close(task.reward, first)
    _, _, _, extra = task.end_step()
    assert extra['used_time'].eq(1/480).all()
    task.begin_step(torch.zeros(2, 24))
    consume(task)
    _, reward, _, extra = task.end_step()
    assert reward.eq(0).all() and all(term.eq(0).all() for term in extra['terms'].values())


def test_surface_success_branch_has_no_signed_force_gate():
    task, _ = configured_task()
    task.m['bottom'].fill_(.391)
    # Arithmetic counterexample only: synthetic poses/native-looking arrays do not
    # establish that negative support can produce a physical lift.
    for _ in range(300):
        task.begin_step(torch.zeros(2, 24))
        consume(task, dt=1/60, lift_force=-1.)
        task.end_step()
    assert task.success.all()


def test_formal_success_anchor_differs_from_early_reward_anchor():
    formal = SurfaceLiftSuccess(2, 'cpu', .30)
    early = SurfaceLiftSuccess(2, 'cpu', .30, minimum_lift_m=.003, reward_diagnostics=True)
    data = scene()
    data['bottom'].fill_(.304)
    advance(formal, data)
    advance(early, data)
    assert formal.duration.eq(0).all() and early.duration.gt(0).all()
    data['object_pose'][:, 0] += .0006
    advance(early, data)
    assert early.duration.eq(0).all()
    torch.testing.assert_close(early.reward_drift_m, torch.full((2,), .0006, dtype=torch.float64))
    assert early.slip_observable.all()  # A failed hold still has measurable slip.


def test_one_hand_can_earn_partial_force_shaping_but_not_formal_success():
    phase = torch.tensor([3], dtype=torch.long)
    force = torch.tensor([[1., 0.]], dtype=torch.float64)
    out = reward_terms(phase, force, force, torch.ones(1), force,
        torch.tensor([[True, False]]), torch.tensor([True]), 1/60,
        phase_free=True, unrestricted_load=True)
    assert out['grasp_upward_force'].item() == pytest.approx(1/60)
    assert out['lift_upward_force'].item() == pytest.approx(2.5/60)
    data = scene(n=1)
    data['loads'][:, 1] = 0.
    data['vectors'][:, 1] = 0.
    assert not advance(SurfaceLiftSuccess(1, 'cpu', .30), data)['window_good'].any()


def test_missing_and_broadcast_force_are_rejected_and_masked_rows_are_zero():
    task, _ = configured_task()
    measurement = {key: value.clone() for key, value in task.m.items()}
    with pytest.raises(ValueError, match='measured signed vertical force'):
        task.consume(measurement, ~task.done, 1/480)
    with pytest.raises(ValueError, match='one signed sum per environment'):
        task.consume(measurement, ~task.done, 1/480, vertical_force_n=torch.tensor(1.))
    phase = torch.tensor([3, 3], dtype=torch.long)
    force = torch.ones(2, 2, dtype=torch.float64)
    out = reward_terms(phase, force, force, torch.ones(2), force,
        torch.ones(2, 2, dtype=torch.bool), torch.tensor([True, False]), 1/60,
        phase_free=True, unrestricted_load=True)
    assert all(value[0] > 0 and value[1] == 0 for value in out.values())


def test_support_does_not_broadcast_whole_hand_force_to_each_digit():
    from go2_pvcnn.control.dh116.tabletop_surface_reward_ablation import accepted_support_n
    task, _ = configured_task()
    task.m['effective_pad_vectors_w'][..., 2] = 0.
    task.m['effective_pad_vectors_w'][:, :, 0, 2] = -.4
    task.m['vertical_hand_force_n'].fill_(.4)
    task.m['rejected_contact_loads_n'][0, 1] = .1
    whole = torch.tensor([[.6, .8], [.6, .8]], dtype=torch.float64)
    support = accepted_support_n(task.m, whole)
    torch.testing.assert_close(support, torch.tensor([1., 1.4], dtype=torch.float64))
    assert support.shape == (2,)  # Per-environment aggregate; no per-digit label.


def test_support_height_and_palm_movement_do_not_count_as_object_lift():
    task, _ = configured_task('lift_progress')
    task.m['wrist_pose'][:, :, 2] += .02
    task.m['object_relative'][:, 2] -= .02  # Keep the actual object world center unchanged.
    task.begin_step(torch.zeros(2, 24))
    for _ in range(8):
        consume(task)
    assert task.terms['surface_lift_progress'].eq(0).all()
    assert not task.success.any()
    assert qualified_pad_contact(task).all()  # Contact alone does not confer lift.


def test_executed_smoothness_distinguishes_motion_and_requested_command_residual():
    zero = torch.zeros(2, 24, dtype=torch.float64)
    motion = ablation.ExecutedMotion(zero)
    valid = torch.ones(2, dtype=torch.bool)
    motion.command(zero, 1/480)
    first = motion.measure(zero, zero, torch.ones_like(zero), valid, 1/480)
    torch.testing.assert_close(first, torch.full((2,), .1, dtype=torch.float64))
    # A blocked large request incurs residual cost despite zero measured movement.
    motion.command(motion.rate.expand_as(zero)/480, 1/480)
    delta = motion.rate.expand_as(zero)/480
    second = motion.measure(zero, delta, torch.ones_like(zero), valid, 1/480)
    torch.testing.assert_close(second, torch.ones(2, dtype=torch.float64))


@pytest.mark.parametrize('factor', list(ablation.SINGLE_TERMS))
def test_each_v4_factor_is_per_environment_masked_and_uses_real_scaling(factor):
    profile = f'v3_{factor}_only'
    reward = ablation.RewardAblation(profile, torch.tensor([.36, .36], dtype=torch.float64))
    number = lambda value: torch.full((2,), value, dtype=torch.float64)
    mask = lambda value: torch.full((2,), value, dtype=torch.bool)
    args = dict(bottom=number(.3905), qualified=mask(True), early_window_good=mask(True),
        early_duration=number(.1), drift=number(.0005), relative_speed=number(.002),
        support=number(.5), weight=number(1.), valid=torch.tensor([True, False]),
        dt=1/480, window_completed=True, window_native_valid=mask(True),
        slip_observable=mask(True), smoothness=number(.3))
    name = ablation.SINGLE_TERMS[factor]
    first = reward.update(**args)[name]
    expected = dict(lift=8., early_hold=2/60, milestone=1., contact_break=0.,
        slip=-24/60, relative_motion=-2/480, support=-1.5*.5/480, smoothness=-.3/480)
    assert first[0].item() == pytest.approx(expected[factor])
    assert first[1].item() == 0.
    if factor == 'contact_break':
        args['early_window_good'] = mask(False)
        assert reward.update(**args)[name][0] == 0.
        assert reward.update(**args)[name][0] == -4.
