"""Archived-argv derivation tests use CPU metadata, not native evidence."""
import pytest

from go2_pvcnn.control.dh116 import tabletop_privileged_t5_binding as t5


def flags():
    return ['--object-kind=solid_cylinder', '--physics-hz=480', '--grasp-controller=policy',
        '--vision-student', '--surface-feedback', '--native-sensor-batch',
        '--surface-reward-profile=support_deficit', '--grasp-initialization=/original/pinch.json',
        '--mixed-grasp-initialization=/original/tripod.json', '--num-envs=8', '--seed=42',
        '--rollout-steps=32', '--max-iterations=1', '--output=/old/t5',
        '--vision-teacher-source=/old/teacher.pt', '--vision-teacher-mpc',
        '--vision-teacher-mpc-profile=pressure_geometry_constrained',
        '--vision-teacher-max-control-steps=1800', '--vision-teacher-hold-tail-control-steps=600',
        '--vision-teacher-raw-chunk-frames=16', '--vision-sequence-context-sidecar', '--vision-mpc-state-capture']


def test_reuse_physical_and_reward_options_while_removing_mpc_and_resource_overrides():
    selected = t5._policy_launch_arguments(flags())
    assert selected == flags()[:9]
    assert '--native-sensor-batch' in selected and '--surface-reward-profile=support_deficit' in selected
    assert not any('teacher' in arg or arg.startswith('--output') for arg in selected)


def test_unknown_controller_or_reward_change_is_rejected():
    for illegal in ('--grasp-controller=mpc', '--surface-reward-profile=baseline'):
        name = illegal.split('=')[0]
        with pytest.raises(ValueError, match='T5|controller|reward'):
            t5._policy_launch_arguments([illegal if arg.split('=')[0] == name else arg for arg in flags()])


def test_runtime_requires_exact_observed_contract_and_leaves_default_unchanged(monkeypatch):
    baseline = {'default': True}
    monkeypatch.setattr(t5, 'semantic_sha256', lambda value: t5.T5_CONTRACT_SEMANTIC_SHA256 if value is baseline else 'a'*64)
    monkeypatch.setattr(t5, 'evaluation_physics_sha256', lambda value: t5.T5_PHYSICS_SHA256)
    t5.verify_runtime_contract(baseline, {})
    binding = dict(version='dh116-privileged-teacher-t5-binding-v1',
        source_contract_sha256=t5.T5_CONTRACT_RAW_SHA256,
        expected_semantic_sha256=t5.T5_CONTRACT_SEMANTIC_SHA256,
        expected_physics_sha256=t5.T5_PHYSICS_SHA256)
    spec = dict(teacher_t5_binding=binding, arguments=t5._policy_launch_arguments(flags()))
    t5.verify_runtime_contract(baseline, spec)
    with pytest.raises(ValueError, match='actual.*T5'):
        t5.verify_runtime_contract({}, spec)
