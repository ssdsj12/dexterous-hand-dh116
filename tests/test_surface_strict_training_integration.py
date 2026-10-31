"""CPU wiring for opt-in admission, cumulative pilots and checkpoint inheritance."""
import argparse
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_tensor_training as training
from go2_pvcnn.control.dh116 import tabletop_surface_admission as admission
from test_surface_student_admission import family_fixture


ROOT = Path(__file__).resolve().parents[1]


def options(**changes):
    args = dict(require_native_student_validation=False, surface_force_sharing=True,
                surface_student_actor=None, surface_demonstrations='qualified.pt',
                surface_student_validation_report=[], surface_student_eval_steps=0,
                max_iterations=128, timesteps=1000000, episodes_per_env=None,
                num_envs=2, rollout_steps=32)
    args.update(changes)
    return SimpleNamespace(**args)


def configure(algorithm, args, contract=None, saved=None, family=None):
    helper = getattr(training, 'configure_surface_student_validation', None)
    assert callable(helper), 'training must wire opt-in validation through one startup helper'
    return helper(algorithm, args, contract or {}, saved=saved, training_family=family)


def test_default_student_episode_training_has_no_new_admission_gate():
    algorithm = SimpleNamespace()
    configure(algorithm, options(max_iterations=None, episodes_per_env=2048),
              saved={'iteration': 475, 'timesteps': 100000})
    assert algorithm.require_native_student_validation is False
    assert algorithm.surface_student_diagnostic_updates == 475


def test_explicit_strict_mode_allows_a_64_update_pilot_and_rejects_65():
    args = options(require_native_student_validation=True, max_iterations=64)
    algorithm = SimpleNamespace()
    configure(algorithm, args)
    assert algorithm.require_native_student_validation is True
    assert algorithm.surface_student_diagnostic_updates == 0
    args.max_iterations = 65
    with pytest.raises(ValueError, match='limit|budget'):
        configure(SimpleNamespace(), args)


def test_strict_checkpoint_cannot_be_downgraded_by_omitting_flag_or_short_resumes():
    saved = dict(require_native_student_validation=True, iteration=64,
                 surface_student_diagnostic_updates=64)
    with pytest.raises(ValueError, match='limit|budget'):
        configure(SimpleNamespace(), options(max_iterations=68), saved=saved)
    saved.update(iteration=60, surface_student_diagnostic_updates=60)
    algorithm = SimpleNamespace()
    configure(algorithm, options(max_iterations=64), saved=saved)
    assert algorithm.require_native_student_validation is True
    assert algorithm.surface_student_diagnostic_updates == 60


def test_strict_episode_training_needs_real_admission():
    with pytest.raises(ValueError, match='episode'):
        configure(SimpleNamespace(), options(require_native_student_validation=True,
                  max_iterations=None, episodes_per_env=2048))


def test_strict_evaluation_preserves_mode_without_spending_pilot_budget():
    algorithm = SimpleNamespace()
    saved = dict(require_native_student_validation=True, iteration=475)
    configure(algorithm, options(max_iterations=None, episodes_per_env=2048,
              surface_student_eval_steps=1800), saved=saved)
    assert algorithm.require_native_student_validation is True
    assert algorithm.surface_student_diagnostic_updates == 475


def test_new_native_reports_use_actual_demonstration_family(tmp_path):
    actor, reports, first, _, family = family_fixture(tmp_path)
    args = options(require_native_student_validation=True, max_iterations=None,
                   episodes_per_env=2048, surface_student_actor=actor,
                   surface_student_validation_report=reports[:1])
    with pytest.raises(ValueError, match='every|cover'):
        configure(SimpleNamespace(), args, first, family=family)
    args.surface_student_validation_report = reports
    algorithm = SimpleNamespace()
    configure(algorithm, args, first, family=family)
    assert algorithm.surface_student_validation['native_student_successes'] == 4


def test_saved_native_proof_is_validated_against_current_training_family(tmp_path):
    actor, reports, first, second, family = family_fixture(tmp_path)
    proof = admission.validate_student_admission(actor, reports, first, training_family=family)
    saved = dict(require_native_student_validation=True, iteration=64,
                 surface_student_diagnostic_updates=64, surface_student_validation=proof)
    algorithm = SimpleNamespace()
    args = options(max_iterations=None, episodes_per_env=2048)
    configure(algorithm, args, second, saved=saved, family=family)
    assert algorithm.surface_student_validation == proof
    with pytest.raises(ValueError):
        configure(SimpleNamespace(), args, dict(second, physics_hz=240), saved=saved)


def test_a_truthy_saved_proof_does_not_admit_strict_resume():
    saved = dict(require_native_student_validation=True, iteration=64,
                 surface_student_validation={'all_declared_reset_styles_validated': True})
    with pytest.raises(ValueError, match='proof|admission'):
        configure(SimpleNamespace(), options(max_iterations=100), saved=saved)


def test_resume_reports_are_explicitly_rejected_instead_of_ignored():
    with pytest.raises(ValueError, match='resume.*report|report.*resume'):
        configure(SimpleNamespace(), options(surface_student_validation_report=['native.json']),
                  saved={'iteration': 4})


@pytest.mark.parametrize('field,value', [
    ('require_native_student_validation', 'false'),
    ('require_native_student_validation', 1),
    ('surface_student_diagnostic_updates', -1),
    ('surface_student_diagnostic_updates', True),
])
def test_invalid_saved_mode_or_counter_is_rejected(field, value):
    with pytest.raises(ValueError):
        configure(SimpleNamespace(), options(), saved={field: value})


def test_validation_configuration_does_not_mutate_policy_or_optimizer(tmp_path):
    from tensordict import TensorDict
    algorithm = training.make_algorithm(TensorDict({'policy': torch.zeros(2, 351)}, [2], device='cpu'),
                                        steps=2, action_size=24)
    before = copy.deepcopy(algorithm.policy.state_dict())
    optimizer_before = copy.deepcopy(algorithm.optimizer.state_dict())
    configure(algorithm, options(require_native_student_validation=True, max_iterations=4))
    for name, value in algorithm.policy.state_dict().items():
        assert torch.equal(value, before[name])
    assert algorithm.optimizer.state_dict() == optimizer_before


def test_checkpoint_round_trip_preserves_strict_mode_counter_and_proof(tmp_path):
    from tensordict import TensorDict
    actor, reports, first, second, family = family_fixture(tmp_path)
    algorithm = training.make_algorithm(TensorDict({'policy': torch.zeros(2, 351)}, [2], device='cpu'),
                                        steps=2, action_size=24)
    configure(algorithm, options(require_native_student_validation=True,
              surface_student_actor=actor, surface_student_validation_report=reports),
              first, family=family)
    algorithm.surface_student_diagnostic_updates = 64
    env = SimpleNamespace(**{name: torch.zeros(2, dtype=torch.long) for name in
        ('episodes', 'successes', 'recovered', 'reached_counts', 'transition_counts')})
    path = tmp_path / 'strict.pt'
    training.save_checkpoint(path, algorithm, first, 64, 4096, env)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['require_native_student_validation'] is True
    assert saved['surface_student_diagnostic_updates'] == 64
    restored = SimpleNamespace()
    configure(restored, options(max_iterations=None, episodes_per_env=2048),
              second, saved=saved, family=family)
    assert restored.require_native_student_validation is True
    assert restored.surface_student_diagnostic_updates == 64
    assert restored.surface_student_validation == algorithm.surface_student_validation


@pytest.mark.parametrize('filename', [
    'scripts/train_dh116_tabletop_gpu_ppo.py', 'scripts/run_dh116_surface_student.py',
    'scripts/train_dh116_surface_multistyle.py',
])
def test_cli_strict_flag_defaults_to_false_and_is_explicitly_selectable(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8-sig'))
    declaration = next((node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
        and any(isinstance(arg, ast.Constant) and arg.value == '--require-native-student-validation'
                for arg in node.args)), None)
    assert declaration is not None, f'{filename} must expose opt-in strict validation'
    parser = argparse.ArgumentParser()
    namespace = {declaration.func.value.id: parser}
    exec(compile(ast.Expression(declaration), filename, 'eval'), namespace)
    assert parser.parse_args([]).require_native_student_validation is False
    assert parser.parse_args(['--require-native-student-validation']).require_native_student_validation is True


@pytest.mark.parametrize('filename,command_name', [
    ('scripts/run_dh116_surface_student.py', 'cmd'),
    ('scripts/train_dh116_surface_multistyle.py', 'command'),
])
def test_wrappers_forward_strict_flag_only_when_selected(filename, command_name):
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8-sig'))
    branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == 'require_native_student_validation']
    branch = next((node for node in branches if any(isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute) and isinstance(child.func.value, ast.Name)
        and child.func.value.id == command_name for child in ast.walk(node))), None)
    assert branch is not None, 'strict selection must reach the child trainer'
    for selected in (False, True):
        command = []
        exec(compile(ast.Module(body=[branch], type_ignores=[]), filename, 'exec'),
             {'args': SimpleNamespace(require_native_student_validation=selected), command_name: command})
        assert ('--require-native-student-validation' in command) is selected


def test_runtime_wires_completed_update_counter_and_stable_hold_initializer():
    source = (ROOT / 'src/go2_pvcnn/control/dh116/tabletop_tensor_training.py').read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    runtime = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'run_tensor_training')
    calls = [node for node in ast.walk(runtime) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)]
    assert any(node.func.id == 'configure_surface_student_validation' for node in calls)
    assert any(node.func.id == 'enable_stable_hold_rewards' for node in calls)
    counters = [node for node in ast.walk(runtime) if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Attribute)
        and node.target.attr == 'surface_student_diagnostic_updates']
    assert len(counters) == 1, 'count only one completed optimizer update per iteration'
    assert any(node.func.id == 'save_checkpoint' and node.lineno > counters[0].lineno for node in calls)


def test_multistyle_passes_initial_actor_reports_only_to_the_fresh_child(tmp_path):
    filename = ROOT / 'scripts/train_dh116_surface_multistyle.py'
    tree = ast.parse(filename.read_text(encoding='utf-8-sig'))
    loop = next(node for node in tree.body if isinstance(node, ast.For)
                and isinstance(node.target, ast.Tuple)
                and any(isinstance(item, ast.Name) and item.id == 'block_index' for item in node.target.elts))
    prefix = []
    for node in loop.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == 'report'):
            break
        prefix.append(node)
    args = options(student_actor=tmp_path / 'actor.pt', num_envs=2, rollout_steps=32,
                   updates_per_style=4, imitation_mode='stable_hold_v2', noise_mode='holding_v2',
                   contact_history=False, validation_report=[tmp_path / 'native.json'])
    for checkpoint in (None, tmp_path / 'saved.pt'):
        namespace = dict(args=args, checkpoint=checkpoint, blocks=[(0, 'pinch', None)],
                         current_min=0, iteration=4, output=tmp_path,
                         templates={'pinch': tmp_path / 'pregrasp.json'}, library=tmp_path / 'library.pt')
        segment = ast.For(target=loop.target, iter=loop.iter, body=prefix, orelse=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[segment], type_ignores=[])),
                     str(filename), 'exec'), namespace)
        assert ('--validation-report' in namespace['command']) is (checkpoint is None)
