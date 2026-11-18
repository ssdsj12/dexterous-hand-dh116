"""Native evaluation diagnostics, without Isaac or a teacher policy.

The atomic JSON and identity tests also run with the standard library alone. The import
shim supplies only the unused no_grad decorator when torch is unavailable;
all tensor and evaluation tests then skip rather than using fake tensors.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None


SOURCE = (Path(__file__).resolve().parents[1] / 'src/go2_pvcnn/control/dh116/'
          'tabletop_surface_student_evaluation.py')


def load_evaluator():
    name = 'go2_pvcnn.control.dh116.tabletop_surface_student_evaluation'
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    if torch is None:
        shim = SimpleNamespace(no_grad=lambda: lambda function: function)
        with patch.dict(sys.modules, {'torch': shim}):
            spec.loader.exec_module(module)
    else:
        spec.loader.exec_module(module)
    return module


class AtomicProgressTests(unittest.TestCase):
    def setUp(self):
        self.evaluation = load_evaluator()

    def writer(self):
        self.assertTrue(callable(getattr(self.evaluation, 'write_progress', None)),
                        'atomic evaluation progress exporter is missing')
        return self.evaluation.write_progress

    def test_progress_replaces_complete_json_in_the_same_directory(self):
        write = self.writer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'student_evaluation_progress.json'
            path.write_text('{"old":true}', encoding='utf-8')
            payload = dict(native_control_steps=60, observed_hold_s=[.2, 0.])
            replace = self.evaluation.os.replace
            replacements = []

            def check_then_replace(source, target):
                source, target = Path(source), Path(target)
                self.assertEqual(source.parent, target.parent)
                self.assertEqual(json.loads(target.read_text()), {'old': True})
                self.assertEqual(json.loads(source.read_text()), payload)
                replacements.append(target)
                return replace(source, target)

            with patch.object(self.evaluation.os, 'replace', check_then_replace):
                write(path, payload)
            self.assertEqual(replacements, [path])
            self.assertEqual(json.loads(path.read_text()), payload)
            self.assertNotIn('\n', path.read_text())
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_failed_replace_preserves_previous_progress_and_cleans_temporary_file(self):
        write = self.writer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'student_evaluation_progress.json'
            original = b'{"native_control_steps":60}'
            path.write_bytes(original)
            with patch.object(self.evaluation.os, 'replace', side_effect=OSError('busy')):
                with self.assertRaisesRegex(OSError, 'busy'):
                    write(path, dict(native_control_steps=120))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])


class EvaluationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.evaluation = load_evaluator()

    def identity(self, args, contract):
        self.assertTrue(callable(getattr(self.evaluation, 'evaluation_identity', None)),
                        'evaluation source and contract identity helper is missing')
        return self.evaluation.evaluation_identity(args, contract)

    def test_source_hash_uses_exact_actor_or_resume_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            actor, resume = Path(directory) / 'actor.pt', Path(directory) / 'resume.pt'
            actor.write_bytes(b'actor-checkpoint\x00\xff')
            resume.write_bytes(b'ppo-checkpoint\x00\xfe')
            for args, source in [
                (SimpleNamespace(surface_student_actor=actor), actor),
                (SimpleNamespace(surface_student_actor='', resume=str(resume)), resume),
                (SimpleNamespace(), None),
            ]:
                with self.subTest(source=source):
                    identity = self.identity(args, {})
                    expected = hashlib.sha256(source.read_bytes()).hexdigest() if source else None
                    self.assertEqual(identity['actor_source_file_sha256'], expected)
            previous = self.identity(SimpleNamespace(resume=resume), {})
            resume.write_bytes(b'changed checkpoint')
            self.assertNotEqual(self.identity(SimpleNamespace(resume=resume), {})['actor_source_file_sha256'],
                                previous['actor_source_file_sha256'])

    def test_contract_hash_is_sorted_and_record_is_an_owned_snapshot(self):
        contract = dict(num_envs=2, physical=dict(velocity=.03, height=.03), styles=['夹持', 'B'])
        reordered = dict(styles=['夹持', 'B'], physical=dict(height=.03, velocity=.03), num_envs=2)
        identity = self.identity(SimpleNamespace(), contract)
        expected = hashlib.sha256(json.dumps(contract, sort_keys=True,
                                              separators=(',', ':')).encode('utf-8')).hexdigest()
        self.assertEqual(identity['contract_sha256'], expected)
        self.assertEqual(identity['contract_sha256'],
                         self.identity(SimpleNamespace(), reordered)['contract_sha256'])
        self.assertEqual(identity['contract'], contract)
        identity['contract']['physical']['height'] = .04
        self.assertEqual(contract['physical']['height'], .03)
        contract['num_envs'] = 4096
        self.assertNotEqual(self.identity(SimpleNamespace(), contract)['contract_sha256'], expected)

    def test_explicit_missing_source_is_not_silently_unbound(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                self.identity(SimpleNamespace(resume=Path(directory) / 'missing.pt'), {})


def native_task(n=2):
    dtype = torch.float64
    loads = torch.tensor([2., .2, 1.8, 0., 0.], dtype=dtype).expand(n, 2, 5).clone()
    normals = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [-1., 0., 0.],
                            [0., 0., 0.], [0., 0., 0.]], dtype=dtype)
    success = SimpleNamespace(duration=torch.full((n,), .4, dtype=dtype),
                              _drift=torch.full((n,), .0002, dtype=dtype),
                              _paired=torch.ones(n, 2, dtype=torch.bool),
                              _window_good=torch.ones(n, dtype=torch.bool))
    return SimpleNamespace(surface_reference_bottom_m=torch.full((n,), .30, dtype=dtype),
                           surface_lift_success=success,
                           success_time=torch.full((n,), 999., dtype=dtype),
                           m=dict(bottom=torch.full((n,), .34, dtype=dtype),
                                  object_velocity=torch.zeros(n, 6, dtype=dtype),
                                  effective_pad_loads_n=loads,
                                  effective_pad_vectors_w=loads[..., None] * normals,
                                  effective_pad_depth_m=torch.zeros_like(loads),
                                  rejected_contact_loads_n=torch.zeros(n, 2, dtype=dtype)))


@unittest.skipIf(torch is None, 'torch is required for native diagnostic CPU tests')
class NativeDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.evaluation = load_evaluator()

    def snapshot(self, task):
        self.assertTrue(callable(getattr(self.evaluation, 'diagnostic_snapshot', None)),
                        'read-only native surface success diagnostic helper is missing')
        return self.evaluation.diagnostic_snapshot(task)

    def test_uses_actual_hold_window_state_and_returns_owned_tensors(self):
        task = native_task()
        task.surface_lift_success._paired[0, 0] = False
        task.surface_lift_success._window_good[0] = False
        before = copy.deepcopy(task)
        diag = self.snapshot(task)
        self.assertEqual(diag['hold_s'].tolist(), [.4, .4])
        self.assertEqual(diag['drift_m'].tolist(), [.0002, .0002])
        self.assertEqual(diag['hold_paired'].tolist(), [[False, True], [True, True]])
        self.assertEqual(diag['hold_window_good'].tolist(), [False, True])
        # Instantaneous physical eligibility differs from the completed window.
        self.assertTrue(diag['physical_eligible'].all())
        for value in diag.values():
            if isinstance(value, torch.Tensor):
                value.zero_()
        for key in task.m:
            self.assertTrue(torch.equal(task.m[key], before.m[key]), key)
        for key, value in vars(task.surface_lift_success).items():
            self.assertTrue(torch.equal(value, getattr(before.surface_lift_success, key)), key)
        self.assertTrue(torch.equal(task.surface_reference_bottom_m, before.surface_reference_bottom_m))

    def test_physical_boundaries_use_native_reference_and_total_contact_fraction(self):
        task = native_task(6)
        task.surface_reference_bottom_m[:] = torch.tensor([.30, .40, .50, .60, .70, .80])
        task.m['bottom'] = task.surface_reference_bottom_m + .03
        task.m['bottom'][1] -= 1e-8
        task.m['object_velocity'][2, 0] = .03  # Strictly below 0.03 m/s.
        task.m['effective_pad_vectors_w'][3, 0, :, 0].abs_()
        task.m['effective_pad_depth_m'][..., 0] = .0008
        task.m['effective_pad_depth_m'][4, 1, 0] += 1e-8
        normal = task.m['effective_pad_loads_n'].sum((1, 2))
        task.m['rejected_contact_loads_n'][:, 0] = normal / 19
        task.m['rejected_contact_loads_n'][5, 0] *= 1.001
        diag = self.snapshot(task)
        self.assertEqual(diag['physical_eligible'].tolist(), [True, False, False, False, False, False])
        expected = {'height': 1, 'velocity': 2, 'opposed_pads': 3,
                    'depth': 4, 'ordinary_fraction': 5}
        for name, failed in expected.items():
            self.assertEqual((~diag['conditions'][name]).nonzero().flatten().tolist(), [failed], name)
        self.assertAlmostEqual(diag['ordinary_fraction'][0].item(), .05)
        # No balance or force tracking requirement: the fixture is unequal.
        self.assertTrue(diag['conditions']['opposed_pads'][0])

    def test_invalid_contact_row_fails_closed_without_poisoning_its_sibling(self):
        for field, bad in [('effective_pad_loads_n', float('nan')),
                           ('effective_pad_vectors_w', float('inf')),
                           ('effective_pad_loads_n', -1.),
                           ('effective_pad_depth_m', -.0001),
                           ('rejected_contact_loads_n', -1.)]:
            with self.subTest(field=field, bad=bad):
                task = native_task()
                task.m[field][0].reshape(-1)[0] = bad
                diag = self.snapshot(task)
                self.assertEqual(diag['physical_eligible'].tolist(), [False, True])


def evaluation_fixture(output, steps=61, mpc=False, mutate=False, episode_steps=None,
                       successful_episodes=True):
    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.linspace(-2., 2., 24))
            self.seen = []

        def act_inference(self, obs):
            assert not torch.is_grad_enabled()
            self.seen.append(obs['policy'].clone())
            if mutate:
                self.bias.add_(.01)
            return self.bias.expand(2, -1)

    class Env:
        n, device = 2, 'cpu'

        def __init__(self):
            self.task = native_task()
            self.successes = torch.tensor([7, 11])
            self.episodes = torch.tensor([13, 17])
            self.executed = []
            self.task.m['bottom'][1] = .31
            self.task.m['object_velocity'][1, 0] = .03
            self.task.m['effective_pad_vectors_w'][1, :, :, 0].abs_()
            self.task.m['effective_pad_depth_m'][1, 0, 0] = .00081
            self.task.m['rejected_contact_loads_n'][1, 0] = 1.

        def step(self, action):
            self.executed.append(action.clone())
            tick = len(self.executed)
            state = self.task.surface_lift_success
            state.duration[:] = torch.tensor([tick / 60 if tick <= 30 else 0., 0.])
            state._drift[:] = torch.tensor([.0002 if tick <= 30 else .0006, .0001])
            state._window_good[:] = torch.tensor([tick <= 30, False])
            done = (torch.tensor([tick in schedule for schedule in episode_steps])
                    if episode_steps is not None else torch.tensor([tick == steps, False]))
            if successful_episodes:
                self.successes += done.long()
            self.episodes += done.long()
            if done[0]:
                state.duration[0] = 0.
                state._drift[0] = 0.
            # A nonterminal success flag must never become an episode success.
            extra = dict(success=torch.tensor([tick == steps, True]), failure=torch.zeros(2, dtype=torch.long))
            return torch.full((2, 3), float(tick)), torch.zeros(2), done, extra

    env, policy = Env(), Student()
    args = SimpleNamespace(output=output, surface_mpc=mpc, surface_student_eval_steps=steps)
    return env, SimpleNamespace(policy=policy), args, {'policy': torch.zeros(2, 3)}


@unittest.skipIf(torch is None, 'torch is required for inference-only evaluation CPU tests')
class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.evaluation = load_evaluator()
        # Only observation packaging is replaced; no PPO/Isaac imports are needed.
        self.training = patch.dict(sys.modules, {
            'go2_pvcnn.control.dh116.tabletop_tensor_training':
                SimpleNamespace(observations=lambda value: {'policy': value})})
        self.training.start()
        self.addCleanup(self.training.stop)

    def test_exports_progress_at_sixty_and_final_with_observed_hold_and_episode_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            env, algorithm, args, obs = evaluation_fixture(directory)
            before = {key: value.clone() for key, value in algorithm.policy.state_dict().items()}
            progress = []
            self.assertTrue(callable(getattr(self.evaluation, 'write_progress', None)),
                            'atomic evaluation progress exporter is missing')
            write = self.evaluation.write_progress

            def record(path, row):
                if Path(path).name == 'student_evaluation_progress.json':
                    progress.append(copy.deepcopy(row))
                return write(path, row)

            with patch.object(self.evaluation, 'write_progress', record):
                result = self.evaluation.evaluate(env, algorithm, args, {}, obs)
            self.assertEqual([row['native_control_steps'] for row in progress], [60, 61])
            self.assertEqual(progress[0]['status'], 'surface_student_evaluation_running')
            self.assertEqual(progress[0]['execution_mode'], 'deterministic_student')
            self.assertEqual(progress[0]['contract'], {})
            self.assertIsNone(progress[0]['actor_source_file_sha256'])
            self.assertEqual(progress[0]['policy_weights_before_sha256'],
                             result['policy_weights_before_sha256'])
            self.assertIsNone(progress[0]['policy_weights_after_sha256'])
            self.assertIsNone(progress[0]['policy_weights_hash_equal'])
            self.assertEqual(progress[-1], result)
            self.assertEqual(result['successes'], [1, 0])
            self.assertEqual(result['episodes'], [1, 0])
            self.assertEqual(result['episode_success_count'], 1)
            self.assertEqual(result['episode_count'], 1)
            self.assertEqual(result['observed_hold_s'], [0., 0.])
            self.assertAlmostEqual(result['observed_net_lift_m'][0], .04)
            self.assertEqual(result['observed_linear_speed_m_s'], [0., .03])
            self.assertAlmostEqual(result['observed_max_pad_depth_m'][1], .00081)
            self.assertAlmostEqual(result['observed_ordinary_fraction'][1], 1 / 9)
            self.assertAlmostEqual(result['max_nonterminal_hold_s'][0], .5)
            self.assertAlmostEqual(result['max_observed_drift_m'][0], .0006, places=8)
            self.assertEqual(result['diagnostic_env_steps'], 122)
            for condition in ('height', 'velocity', 'opposed_pads', 'depth', 'ordinary_fraction'):
                self.assertEqual(result['condition_failure_counts'][condition], [0, 61])
            self.assertTrue(result['actor_unchanged'])
            for key, value in algorithm.policy.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]))
            saved = torch.load(Path(directory) / 'student_evaluation.pt', weights_only=True)
            self.assertEqual(saved['observations'].shape, (61, 2, 3))
            self.assertTrue(torch.equal(saved['observations'][0], obs['policy']))
            self.assertTrue(torch.equal(saved['observations'][1], torch.ones(2, 3)))
            self.assertTrue(torch.equal(saved['executed_actions'], torch.stack(env.executed)))
            self.assertTrue(torch.equal(saved['nominal_actions'], torch.stack(env.executed)))
            self.assertEqual(len(algorithm.policy.seen), 61)
            for name in ('student_evaluation.json', 'student_evaluation_progress.json'):
                self.assertEqual(json.loads((Path(directory) / name).read_text()), result)

    def test_progress_cadence_and_final_boundaries(self):
        for steps in (1, 59, 60, 120, 121):
            with self.subTest(steps=steps), tempfile.TemporaryDirectory() as directory:
                env, algorithm, args, obs = evaluation_fixture(directory, steps=steps)
                self.assertTrue(callable(getattr(self.evaluation, 'write_progress', None)),
                                'atomic evaluation progress exporter is missing')
                with patch.object(self.evaluation, 'write_progress', wraps=self.evaluation.write_progress) as write:
                    self.evaluation.evaluate(env, algorithm, args, {}, obs)
                progress = [call.args[1] for call in write.call_args_list
                            if Path(call.args[0]).name == 'student_evaluation_progress.json']
                self.assertEqual([row['native_control_steps'] for row in progress],
                                 [*range(60, steps, 60), steps])
                self.assertEqual(progress[-1]['status'], 'surface_student_evaluation_complete')

    def test_first_episode_only_waits_for_every_environment_relative_to_initial_counts(self):
        for last_episode in (5, 60, 61):
            with self.subTest(last_episode=last_episode), tempfile.TemporaryDirectory() as directory:
                env, algorithm, args, obs = evaluation_fixture(
                    directory, steps=121, episode_steps=((2, 4), (last_episode,)), successful_episodes=False)
                args.surface_student_first_episode_only = True
                with patch.object(self.evaluation, 'write_progress', wraps=self.evaluation.write_progress) as write:
                    result = self.evaluation.evaluate(env, algorithm, args, {}, obs)
                self.assertEqual(len(env.executed), last_episode)
                self.assertEqual(result['native_control_steps'], last_episode)
                self.assertEqual(result['planned_native_control_steps'], 121)
                self.assertEqual(result['diagnostic_env_steps'], 2 * last_episode)
                self.assertEqual(result['episodes'], [2, 1])
                self.assertEqual(result['episode_success_count'], 0)
                self.assertFalse(result['grasp_qualified'])
                self.assertTrue(result['first_episode_only'])
                self.assertEqual(result['stop_reason'], 'all_envs_completed_first_episode')
                saved = torch.load(Path(directory) / 'student_evaluation.pt', weights_only=True)
                self.assertEqual(saved['observations'].shape[0], last_episode)
                self.assertEqual(result['condition_failure_counts']['height'], [0, last_episode])
                progress = [call.args[1] for call in write.call_args_list
                            if Path(call.args[0]).name == 'student_evaluation_progress.json']
                self.assertEqual([row['native_control_steps'] for row in progress],
                                 [*range(60, last_episode, 60), last_episode])
                self.assertEqual(progress[-1], result)

    def test_default_or_explicitly_disabled_first_episode_only_preserves_full_budget(self):
        for option in (None, False):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                env, algorithm, args, obs = evaluation_fixture(
                    directory, steps=8, episode_steps=((2, 4), (5,)))
                if option is not None:
                    args.surface_student_first_episode_only = option
                result = self.evaluation.evaluate(env, algorithm, args, {}, obs)
                self.assertEqual(len(env.executed), 8)
                self.assertEqual(result['native_control_steps'], 8)
                self.assertFalse(result['first_episode_only'])
                self.assertEqual(result['stop_reason'], 'step_budget_exhausted')

    def test_first_episode_only_keeps_budget_if_an_environment_never_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            env, algorithm, args, obs = evaluation_fixture(directory, steps=8, episode_steps=((2, 4), ()))
            args.surface_student_first_episode_only = True
            result = self.evaluation.evaluate(env, algorithm, args, {}, obs)
            self.assertEqual(result['native_control_steps'], 8)
            self.assertEqual(result['episodes'], [2, 0])
            self.assertEqual(result['stop_reason'], 'step_budget_exhausted')

    def test_report_binds_source_contract_execution_mode_and_unchanged_policy_hashes(self):
        for source_kind in ('surface_student_actor', 'resume'):
            with self.subTest(source_kind=source_kind), tempfile.TemporaryDirectory() as directory:
                env, algorithm, args, obs = evaluation_fixture(directory, steps=61)
                source = Path(directory) / 'source.pt'
                original = b'loaded source checkpoint'
                source.write_bytes(original)
                setattr(args, source_kind, source)
                contract = dict(num_envs=2, physical=dict(height=.03, speed=.03))
                before_contract = copy.deepcopy(contract)
                before_hash = self.evaluation.policy_weights_sha256(algorithm.policy.state_dict())
                step = env.step

                def step_and_change_source(action):
                    if not env.executed:
                        source.write_bytes(b'replaced after evaluation started')
                    return step(action)

                with patch.object(env, 'step', step_and_change_source):
                    result = self.evaluation.evaluate(env, algorithm, args, contract, obs)
                self.assertEqual(result['actor_source_file_sha256'], hashlib.sha256(original).hexdigest())
                self.assertEqual(result['contract'], before_contract)
                self.assertEqual(contract, before_contract)
                self.assertIsNot(result['contract'], contract)
                self.assertEqual(result['contract_sha256'], hashlib.sha256(
                    json.dumps(result['contract'], sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest())
                self.assertEqual(result['execution_mode'], 'deterministic_student')
                self.assertEqual(result['policy_weights_before_sha256'], before_hash)
                self.assertEqual(result['policy_weights_after_sha256'], before_hash)
                self.assertTrue(result['policy_weights_hash_equal'])
                for name in ('student_evaluation.json', 'student_evaluation_progress.json'):
                    self.assertEqual(json.loads((Path(directory) / name).read_text()), result)

    def test_weight_hash_is_deterministic_for_order_and_layout_without_mutation(self):
        self.assertTrue(callable(getattr(self.evaluation, 'policy_weights_sha256', None)),
                        'deterministic policy weight hash helper is missing')
        state = dict(matrix=torch.arange(6.).reshape(2, 3).T, scalar=torch.tensor(1., dtype=torch.bfloat16),
                     empty=torch.empty(0), counter=torch.tensor(4))
        before = {name: value.clone() for name, value in state.items()}
        digest = self.evaluation.policy_weights_sha256(state)
        reordered = {name: value.contiguous() for name, value in reversed(list(state.items()))}
        self.assertEqual(digest, self.evaluation.policy_weights_sha256(reordered))
        self.assertEqual(len(digest), 64)
        int(digest, 16)
        for name in state:
            self.assertTrue(torch.equal(state[name], before[name]))
        for changed in [dict(other=torch.ones(2)), dict(weight=torch.ones(1, 2)),
                        dict(weight=torch.ones(2, dtype=torch.float64)), dict(weight=torch.zeros(2))]:
            with self.subTest(changed=changed):
                self.assertNotEqual(self.evaluation.policy_weights_sha256(dict(weight=torch.ones(2))),
                                    self.evaluation.policy_weights_sha256(changed))

    def test_mpc_actions_still_execute_and_are_distinct_from_student_nominal(self):
        class MPC:
            def __init__(self, env):
                self.stats = dict(observations=0)

            def refine(self, nominal):
                return nominal * .5

            def observe_result(self, done):
                self.stats['observations'] += 1

        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
            'go2_pvcnn.control.dh116.tabletop_surface_mpc_runtime': SimpleNamespace(NativeSurfaceMPC=MPC)}):
            env, algorithm, args, obs = evaluation_fixture(directory, steps=2, mpc=True)
            result = self.evaluation.evaluate(env, algorithm, args, {}, obs)
            saved = torch.load(Path(directory) / 'student_evaluation.pt', weights_only=True)
            self.assertTrue(torch.equal(saved['executed_actions'], saved['nominal_actions'] * .5))
            self.assertTrue(torch.equal(saved['executed_actions'], torch.stack(env.executed)))
            self.assertEqual(result['mpc'], dict(observations=2))
            self.assertEqual(result['execution_mode'], 'deterministic_student_with_mpc')
            self.assertTrue(result['policy_weights_hash_equal'])

    def test_model_mutation_is_still_rejected_before_complete_report(self):
        with tempfile.TemporaryDirectory() as directory:
            env, algorithm, args, obs = evaluation_fixture(directory, steps=1, mutate=True)
            with self.assertRaisesRegex(RuntimeError, 'student changed during inference-only evaluation'):
                self.evaluation.evaluate(env, algorithm, args, {}, obs)
            self.assertFalse((Path(directory) / 'student_evaluation.json').exists())
            self.assertFalse((Path(directory) / 'student_evaluation_progress.json').exists())


if __name__ == '__main__':
    unittest.main()
