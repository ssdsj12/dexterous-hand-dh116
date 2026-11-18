"""Pre-Kit gates for isolated observation arms; fixtures confer no Native status."""
import argparse
import ast
import copy
import importlib
import math
from pathlib import Path
import sys
import types
import unittest
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/go2_pvcnn/control/dh116'
CLI = ROOT / 'scripts/train_dh116_tabletop_gpu_ppo.py'
TEACHER = SOURCE / 'tabletop_vision_teacher_collection.py'
PACKAGE = '_stage7_experimental_options_cpu'


def execute(nodes, scope, path):
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(module, str(path), 'exec'), scope)


def options(**changes):
    values = dict(vision_teacher_source=Path('teacher.pt'), vision_teacher_mpc=True,
        vision_teacher_mpc_profile='pressure_geometry_constrained',
        vision_sequence_context_sidecar=True, vision_student=True, native_sensor_batch=True,
        surface_force_sharing=True, vision_camera_mode='fixed', vision_width=224, vision_height=96,
        vision_action_distribution='tanh_gaussian', physics_hz=480, object_kind='solid_cylinder',
        solid_cylinder_radius_m=.008, num_envs=8, vision_teacher_max_control_steps=16,
        vision_teacher_raw_chunk_frames=16, vision_mpc_state_capture=True,
        vision_mpc_aux_capture=True, vision_mpc_model_capture=False,
        vision_mpc_model_descriptor_capture=False,
        vision_mpc_contact_delivery_probe=None, vision_mpc_filtered_contact_capture=False,
        vision_mpc_filtered_contact_epoch_capture=False,vision_mpc_filtered_contact_values_capture=False,
        vision_mpc_passive_pair_values=False)
    values.update(changes)
    return NS(**values)


class ExperimentalOptionsTests(unittest.TestCase):
    @staticmethod
    def validate():
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(SOURCE)]
        sys.modules.setdefault(PACKAGE, package)
        tree = ast.parse(TEACHER.read_text(encoding='utf-8'))
        node = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'validate_options')
        scope = dict(__package__=PACKAGE, math=math)
        execute([node], scope, TEACHER)
        return scope['validate_options']

    def test_flags_are_explicit_default_off_and_probe_choices_closed(self):
        flags = {'--vision-mpc-model-capture', '--vision-mpc-model-descriptor-capture','--vision-mpc-contact-delivery-probe',
                 '--vision-mpc-filtered-contact-capture','--vision-mpc-filtered-contact-epoch-capture',
                 '--vision-mpc-filtered-contact-values-capture','--vision-mpc-passive-pair-values'}
        tree = ast.parse(CLI.read_text(encoding='utf-8'))
        nodes = [node for node in tree.body if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call) and node.value.args
                 and isinstance(node.value.args[0], ast.Constant)
                 and node.value.args[0].value in flags]
        self.assertEqual(len(nodes), 7)
        parser = argparse.ArgumentParser()
        execute(nodes, dict(parser=parser), CLI)
        defaults = parser.parse_args([])
        self.assertIs(defaults.vision_mpc_model_capture, False)
        self.assertIs(defaults.vision_mpc_model_descriptor_capture,False)
        self.assertIsNone(defaults.vision_mpc_contact_delivery_probe)
        self.assertIs(defaults.vision_mpc_filtered_contact_capture, False)
        self.assertIs(defaults.vision_mpc_filtered_contact_epoch_capture,False)
        self.assertIs(defaults.vision_mpc_filtered_contact_values_capture,False)
        self.assertIs(defaults.vision_mpc_passive_pair_values,False)
        self.assertTrue(parser.parse_args(['--vision-mpc-filtered-contact-capture']).vision_mpc_filtered_contact_capture)
        self.assertTrue(parser.parse_args(['--vision-mpc-model-capture']).vision_mpc_model_capture)
        self.assertTrue(parser.parse_args(['--vision-mpc-model-descriptor-capture']).vision_mpc_model_descriptor_capture)
        for mode in ('post_step_poll', 'full_report_callback'):
            args = parser.parse_args(['--vision-mpc-contact-delivery-probe', mode])
            self.assertEqual(args.vision_mpc_contact_delivery_probe, mode)
        with self.assertRaises(SystemExit):
            parser.parse_args(['--vision-mpc-contact-delivery-probe', 'invented_mode'])

    def test_passive_pair_observer_is_explicit_isolated_and_needs_original_aux_primary(self):
        validate=self.validate()
        self.assertIsNone(validate(options(vision_mpc_passive_pair_values=True)))
        for change in ({'vision_mpc_passive_pair_values':1},
                       {'vision_mpc_aux_capture':False},{'vision_mpc_state_capture':False},
                       {'vision_mpc_model_capture':True},{'vision_mpc_filtered_contact_capture':True},
                       {'vision_mpc_contact_delivery_probe':'post_step_poll'},{'num_envs':16},
                       {'vision_teacher_max_control_steps':1900}):
            base=dict(vision_mpc_passive_pair_values=True);base.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):validate(options(**base))

    def test_each_arm_requires_aux_and_primary_before_teacher_early_return(self):
        validate = self.validate()
        for arm in ({'vision_mpc_model_capture': True},
                    {'vision_mpc_filtered_contact_capture': True},
                    {'vision_mpc_contact_delivery_probe': 'post_step_poll'},
                    {'vision_mpc_contact_delivery_probe': 'full_report_callback'}):
            with self.subTest(arm=arm):
                self.assertIsNone(validate(options(**arm)))
                for source in (NS(**arm), options(vision_mpc_aux_capture=False, **arm),
                               options(vision_mpc_state_capture=False, **arm)):
                    with self.assertRaisesRegex(ValueError, 'auxiliary|primary|state capture'):
                        validate(source)

    def test_arms_are_mutually_exclusive_and_programmatic_values_are_strict(self):
        validate = self.validate()
        for changes in (dict(vision_mpc_model_capture=True,
                             vision_mpc_contact_delivery_probe='post_step_poll'),
                        dict(vision_mpc_model_capture=1),
                        dict(vision_mpc_filtered_contact_capture=1),
                        dict(vision_mpc_filtered_contact_capture=True, vision_mpc_model_capture=True),
                        dict(vision_mpc_filtered_contact_capture=True, vision_mpc_contact_delivery_probe='post_step_poll'),
                        dict(vision_mpc_contact_delivery_probe=True),
                        dict(vision_mpc_contact_delivery_probe='unknown')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate(options(**changes))

    def test_epoch_and_values_require_explicit_dependency_chain(self):
        validate=self.validate()
        for changes in (
                dict(vision_mpc_filtered_contact_epoch_capture=True),
                dict(vision_mpc_filtered_contact_values_capture=True,vision_mpc_filtered_contact_capture=True),
                dict(vision_mpc_filtered_contact_epoch_capture=1),
                dict(vision_mpc_filtered_contact_values_capture=1)):
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate(options(**changes))
        for values in (False,True):
            self.assertIsNone(validate(options(vision_mpc_filtered_contact_capture=True,
                vision_mpc_filtered_contact_epoch_capture=True,vision_mpc_filtered_contact_values_capture=values)))

    def test_descriptor_requires_original_model_capture_before_teacher_or_kit(self):
        validate=self.validate()
        for source in (NS(vision_mpc_model_descriptor_capture=True),options(vision_mpc_model_descriptor_capture=True),
                options(vision_mpc_model_descriptor_capture=1),options(vision_mpc_model_descriptor_capture=True,
                    vision_mpc_model_capture=True,vision_mpc_aux_capture=False)):
            with self.assertRaisesRegex(ValueError,'descriptor|auxiliary'):validate(source)
        self.assertIsNone(validate(options(vision_mpc_model_capture=True,vision_mpc_model_descriptor_capture=True)))

    def test_experiments_keep_small_finite_legacy_teacher_route(self):
        validate = self.validate()
        rejected = [dict(vision_teacher_source=None), dict(vision_teacher_mpc=False),
            dict(vision_sequence_context_sidecar=False), dict(native_sensor_batch=False),
            dict(surface_force_sharing=False), dict(vision_teacher_mpc_profile='local_pad'),
            dict(vision_width=320), dict(physics_hz=240), dict(vision_mirror_preprocessing='m2_full_mirror'),
            dict(bimanual_explicit_sdk_adapter=True), dict(bimanual_common_pause=True),
            dict(vision_stage6_native_config='alternate.json'), dict(resume='resume.pt'),
            dict(vision_teacher_perturbation_manifest='changed_initials.json'),
            dict(num_envs=16), dict(vision_teacher_max_control_steps=1900)]
        for arm in ({'vision_mpc_model_capture': True},
                    {'vision_mpc_filtered_contact_capture': True},
                    {'vision_mpc_contact_delivery_probe': 'full_report_callback'}):
            for changes in rejected:
                with self.subTest(arm=arm, changes=changes), self.assertRaises(ValueError):
                    validate(options(**arm, **changes))

    def test_default_route_retains_full_budget_and_prior_options(self):
        validate = self.validate()
        self.assertIsNone(validate(NS()))
        self.assertIsNone(validate(options(num_envs=16, vision_teacher_max_control_steps=1900)))

    def test_rejection_happens_before_app_launcher(self):
        tree = ast.parse(CLI.read_text(encoding='utf-8'))
        validation = next(node for node in tree.body if isinstance(node, ast.Try)
            and any(isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
                    and item.func.id == 'validate_teacher_collection_options'
                    for item in ast.walk(node)))
        launch = min(node.lineno for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                     and node.func.id == 'AppLauncher')
        self.assertLess(validation.end_lineno, launch)
        class Parser:
            def error(self, message):
                raise SystemExit(message)
        with self.assertRaises(SystemExit):
            execute([validation], dict(args=options(vision_mpc_model_capture=True,
                vision_mpc_aux_capture=False), parser=Parser(),
                validate_teacher_collection_options=self.validate()), CLI)


if __name__ == '__main__':
    unittest.main()
