"""Immutable offline surface repair; no PPO or simulator construction."""
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
MODULE = 'go2_pvcnn.control.dh116.tabletop_surface_repair_resume'
SCRIPT = ROOT / 'scripts/prepare_dh116_surface_repair_resume.py'


class CommandLineTests(unittest.TestCase):
    def test_all_delivered_python_sources_compile_without_torch(self):
        for path in (ROOT / 'src/go2_pvcnn/control/dh116/tabletop_surface_repair_resume.py', SCRIPT):
            with self.subTest(path=path):
                compile(path.read_text(encoding='utf-8'), str(path), 'exec')

    def test_help_exposes_only_explicit_input_output_and_demonstrations_without_loading_torch(self):
        result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--help'],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ('--input', '--output', '--demonstrations'):
            self.assertIn(flag, result.stdout)


def same(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertEqual(left.dtype, right.dtype)
        test.assertEqual(left.shape, right.shape)
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(list(left), list(right))
        for key in left:
            same(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(type(left), type(right))
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            same(test, a, b)
    else:
        test.assertEqual(left, right)


def fixture(directory, *, amsgrad=True):
    from go2_pvcnn.control.dh116 import tabletop_surface_imitation as bc
    from go2_pvcnn.control.dh116.tabletop_policy_noise import CONTRACT

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.log_std = torch.nn.Parameter(torch.full((24,), -1.2))
            self.actor = bc.make_actor()
            self.critic = torch.nn.Sequential(torch.nn.Linear(351, 128), torch.nn.Tanh(),
                torch.nn.Linear(128, 128), torch.nn.Tanh(), torch.nn.Linear(128, 1))

    contract = dict(observation_size=351, observation_version=bc.OBSERVATION_VERSION,
                    action_size=24, policy_command_size=24, num_envs=2, physics_hz=480,
                    policy_noise=dict(CONTRACT), success_dwell_s=5., success_slip_m=.0005,
                    surface_force_sharing=dict(version='opposed_surface_sharing_v1'))
    demo = dict(version=bc.DATASET_VERSION, contract=copy.deepcopy(contract),
                observations=torch.zeros(8, 351), actions=torch.zeros(8, 24),
                episode_ids=torch.zeros(8, dtype=torch.long),
                qualification=dict(qualified=True, successful_replays=1, proof_path='native/replay.json'))
    demonstrations = directory / 'demonstrations.pt'
    torch.save(demo, demonstrations)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, amsgrad=amsgrad)
    for index, value in enumerate(policy.parameters()):
        value.grad = torch.full_like(value, (index + 1) / 100)
    optimizer.step()
    saved = dict(version='dh116-rsl-tensor-ppo-v1', contract=contract,
        policy=policy.state_dict(), optimizer=optimizer.state_dict(),
        iteration=17, timesteps=123456, learning_rate=2e-4,
        episodes=torch.tensor([100, 200]), successes=torch.tensor([1, 0]),
        recovered=torch.tensor([3, 4]), reached_counts=torch.arange(4),
        transition_counts=torch.arange(16), combined_failure_resets=torch.tensor([7, 8]),
        joint_limit_resets=torch.tensor([9, 10]), object_drop_resets=torch.tensor([11, 12]),
        numerical_resets=torch.tensor(13), close_completion_event_index=23,
        policy_action_recovery=dict(recovered=12),
        torch_rng=torch.get_rng_state(), cuda_rng=[torch.arange(12, dtype=torch.uint8)],
        extra_ppo_state=dict(schedule='adaptive', returns=torch.tensor([1., 2.])),
        surface_imitation=dict(dataset_sha256=hashlib.sha256(demonstrations.read_bytes()).hexdigest(),
            initial_weight=.1, decay=.995, batch_size=256,
            update='separate_auxiliary_step_after_on_policy_PPO_update',
            source='qualified_native_pre_action_observation_action_pairs'))
    source = directory / 'source.pt'
    torch.save(saved, source)
    return source, directory / 'derived.pt', demonstrations, saved


@unittest.skipIf(torch is None, 'torch is required for offline repair CPU tests')
class RepairResumeTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.assertIsNotNone(importlib.util.find_spec(MODULE), 'surface repair resume module missing')
        self.module = importlib.import_module(MODULE)

    def prepare(self, source, output, demonstrations):
        return self.module.prepare_repair_resume(source, output, demonstrations)

    def test_retains_all_nonnoise_policy_counters_rng_ppo_and_actor_momentum(self):
        from go2_pvcnn.control.dh116.tabletop_surface_exploration import STABLE_HOLD_REWARD_CONTRACT
        from go2_pvcnn.control.dh116.tabletop_policy_noise import CONTRACT
        for amsgrad in (False, True):
            with self.subTest(amsgrad=amsgrad):
                directory = self.root / str(amsgrad)
                directory.mkdir()
                source, output, demo, before = fixture(directory, amsgrad=amsgrad)
                original_source, original_demo = source.read_bytes(), demo.read_bytes()
                rng = torch.get_rng_state().clone()
                manifest = self.prepare(source, output, demo)
                after = torch.load(output, map_location='cpu', weights_only=False)
                self.assertEqual(source.read_bytes(), original_source)
                self.assertEqual(demo.read_bytes(), original_demo)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                for key in before:
                    if key not in ('contract', 'policy', 'optimizer', 'surface_imitation'):
                        same(self, before[key], after[key])
                for name, value in before['policy'].items():
                    if name != 'log_std':
                        same(self, value, after['policy'][name])
                torch.testing.assert_close(after['policy']['log_std'], torch.full((24,), math.log(.005)), rtol=0, atol=0)
                same(self, before['optimizer']['param_groups'], after['optimizer']['param_groups'])
                self.assertEqual(before['optimizer']['state'].keys(), after['optimizer']['state'].keys())
                for identifier, state in before['optimizer']['state'].items():
                    for name, value in state.items():
                        if identifier == 0 and name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                            self.assertFalse(after['optimizer']['state'][identifier][name].any())
                        else:
                            same(self, value, after['optimizer']['state'][identifier][name])
                expected_contract = copy.deepcopy(before['contract'])
                expected_contract['policy_noise'] = dict(CONTRACT, version='bounded_surface_policy_noise_v2',
                    initial_std=.005, min_std=.001, max_std=.3)
                expected_contract['surface_training_objective'] = STABLE_HOLD_REWARD_CONTRACT
                same(self, expected_contract, after['contract'])
                self.assertEqual(after['surface_imitation'], dict(
                    dataset_sha256=before['surface_imitation']['dataset_sha256'], initial_weight=.1, decay=1.,
                    batch_size=256, source=before['surface_imitation']['source'], holding_weight=20., wrist_weight=4.,
                    auxiliary_learning_rate=.001, weight_semantics='multiply_independent_Adam_learning_rate_not_its_gradient',
                    update='independent_actor_adam_stable_hold_v2'))
                self.assertEqual(after['surface_imitation_optimizer']['state'], {})
                group = after['surface_imitation_optimizer']['param_groups'][0]
                self.assertEqual(group['params'], list(range(6)))
                self.assertEqual(group['lr'], .001)
                self.assertNotIn('surface_student_validation', after)
                migration = after['surface_repair_migration']
                self.assertEqual(migration['source_sha256'], hashlib.sha256(original_source).hexdigest())
                self.assertEqual(migration['parameter_order'][0], 'log_std')
                self.assertEqual(len(migration['parameter_order']), 13)
                self.assertTrue(all(migration['invariant_checks'].values()))
                self.assertNotIn('derived_sha256', migration)
                self.assertFalse(manifest['native_student_proof_present'])
                self.assertEqual(manifest['derived_sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
                self.assertEqual(manifest, json.loads(output.with_suffix('.json').read_text()))

    def test_only_changed_noise_coordinates_lose_moments_and_adam_step_is_preserved(self):
        source, output, demo, saved = fixture(self.root)
        saved['policy']['log_std'][0] = math.log(.005)
        old_noise = copy.deepcopy(saved['optimizer']['state'][0])
        torch.save(saved, source)
        self.prepare(source, output, demo)
        after = torch.load(output, weights_only=False)
        noise = after['optimizer']['state'][0]
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            same(self, noise[key][0], old_noise[key][0])
            self.assertFalse(noise[key][1:].any())
        same(self, noise['step'], old_noise['step'])

    def test_existing_admission_evidence_is_preserved_without_being_created(self):
        source, output, demo, saved = fixture(self.root)
        saved['surface_student_validation'] = dict(qualified=False, report_sha256='f' * 64)
        torch.save(saved, source)
        self.prepare(source, output, demo)
        after = torch.load(output, weights_only=False)
        same(self, saved['surface_student_validation'], after['surface_student_validation'])

    def test_derived_checkpoint_and_sidecar_each_refuse_overwrite(self):
        for occupied in ('checkpoint', 'sidecar'):
            with self.subTest(occupied=occupied):
                directory = self.root / occupied
                directory.mkdir()
                source, output, demo, saved = fixture(directory)
                target = output if occupied == 'checkpoint' else output.with_suffix('.json')
                target.write_bytes(b'keep existing bytes')
                with self.assertRaises(FileExistsError):
                    self.prepare(source, output, demo)
                self.assertEqual(target.read_bytes(), b'keep existing bytes')
                other = output.with_suffix('.json') if occupied == 'checkpoint' else output
                self.assertFalse(other.exists())

    def test_source_and_demonstrations_cannot_be_output_or_sidecar(self):
        source, output, demo, saved = fixture(self.root)
        for target in (source, demo, source.with_suffix('.json')):
            with self.subTest(target=target):
                # For source.json the checkpoint and its own sidecar collide.
                with self.assertRaises((ValueError, FileExistsError)):
                    self.prepare(source, target, demo)

    def test_wrong_exact_demonstration_file_sha_is_rejected(self):
        source, output, demo, saved = fixture(self.root)
        saved['surface_imitation']['dataset_sha256'] = '0' * 64
        torch.save(saved, source)
        with self.assertRaisesRegex(ValueError, 'SHA|sha|dataset'):
            self.prepare(source, output, demo)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix('.json').exists())

    def test_nonfinite_tensors_anywhere_and_non351_or_non24_models_are_rejected(self):
        for fault in ('actor', 'critic', 'noise', 'optimizer', 'counter', 'rng', 'unknown', 'observation', 'action'):
            with self.subTest(fault=fault):
                directory = self.root / fault
                directory.mkdir()
                source, output, demo, saved = fixture(directory)
                if fault == 'actor': saved['policy']['actor.0.weight'][0, 0] = float('nan')
                if fault == 'critic': saved['policy']['critic.4.weight'][0, 0] = float('inf')
                if fault == 'noise': saved['policy']['log_std'][0] = float('nan')
                if fault == 'optimizer': saved['optimizer']['state'][1]['exp_avg'][0, 0] = float('nan')
                if fault == 'counter': saved['episodes'] = torch.tensor([float('inf'), 2.])
                if fault == 'rng': saved['cuda_rng'].append(torch.tensor([float('nan')]))
                if fault == 'unknown': saved['other'] = {'tensor': torch.tensor([float('nan')])}
                if fault == 'observation': saved['contract']['observation_size'] = 401
                if fault == 'action': saved['contract']['action_size'] = 20
                torch.save(saved, source)
                with self.assertRaises(ValueError):
                    self.prepare(source, output, demo)
                self.assertFalse(output.exists())

    def test_optimizer_mapping_is_validated_instead_of_guessing_parameter_zero(self):
        for fault in ('group_order', 'policy_order', 'moment_shape', 'names', 'extra_parameter', 'unknown_id'):
            with self.subTest(fault=fault):
                directory = self.root / fault
                directory.mkdir()
                source, output, demo, saved = fixture(directory)
                if fault == 'group_order':
                    saved['optimizer']['param_groups'][0]['params'][:2] = [1, 0]
                if fault == 'policy_order':
                    item = saved['policy'].pop('log_std')
                    saved['policy']['log_std'] = item
                if fault == 'moment_shape': saved['optimizer']['state'][0]['exp_avg'] = torch.zeros(23)
                if fault == 'names': saved['optimizer']['param_groups'][0]['param_names'] = ['wrong'] * 13
                if fault == 'extra_parameter': saved['policy']['other'] = torch.zeros(24)
                if fault == 'unknown_id': saved['optimizer']['state'][99] = saved['optimizer']['state'][0]
                torch.save(saved, source)
                with self.assertRaisesRegex(ValueError, 'parameter|mapping|moment|optimizer|policy'):
                    self.prepare(source, output, demo)
                self.assertFalse(output.exists())

    def test_schema_required_keys_and_wrong_physical_demonstrations_fail_before_output(self):
        for fault in ('missing_counters', 'wrong_version', 'wrong_physics'):
            with self.subTest(fault=fault):
                directory = self.root / fault
                directory.mkdir()
                source, output, demo, saved = fixture(directory)
                if fault == 'missing_counters': saved.pop('timesteps')
                if fault == 'wrong_version': saved['version'] = 'legacy'
                if fault == 'wrong_physics': saved['contract']['physics_hz'] = 240
                torch.save(saved, source)
                with self.assertRaises(ValueError):
                    self.prepare(source, output, demo)
                self.assertFalse(output.exists())

    def test_auxiliary_empty_adam_resumes_with_actor_parameter_order_without_touching_ppo(self):
        from go2_pvcnn.control.dh116 import tabletop_surface_imitation as bc
        source, output, demo, saved = fixture(self.root)
        self.prepare(source, output, demo)
        after = torch.load(output, weights_only=False)
        actor = bc.make_actor()
        actor.load_state_dict({key.removeprefix('actor.'): value for key, value in after['policy'].items()
                               if key.startswith('actor.')})
        optimizer = torch.optim.Adam(actor.parameters(), lr=.001)
        optimizer.load_state_dict(after['surface_imitation_optimizer'])
        self.assertEqual(optimizer.state_dict()['state'], {})
        self.assertEqual(len(optimizer.param_groups[0]['params']), 6)
        for value in actor.parameters():
            value.grad = torch.ones_like(value)
        optimizer.param_groups[0]['lr'] = .0001
        optimizer.step()
        self.assertEqual(len(optimizer.state_dict()['state']), 6)
        same(self, torch.load(source, weights_only=False)['optimizer'], saved['optimizer'])

    def test_cli_creates_the_same_derived_schema_without_importing_training(self):
        source, output, demo, saved = fixture(self.root)
        environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--input', str(source),
                                 '--output', str(output), '--demonstrations', str(demo)],
                                capture_output=True, text=True, check=False, env=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), json.loads(output.with_suffix('.json').read_text()))


if __name__ == '__main__':
    unittest.main()
