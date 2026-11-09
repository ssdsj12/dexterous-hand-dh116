"""Stdlib-only collector wiring checks; fixtures grant no native qualification."""
import argparse
import ast
from contextlib import contextmanager
import copy
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
import uuid
from types import SimpleNamespace as NS
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/go2_pvcnn/control/dh116'
COLLECTOR = SOURCE / 'tabletop_vision_teacher_collection.py'
CLI = ROOT / 'scripts/train_dh116_tabletop_gpu_ppo.py'
PACKAGE = '_stage7_aux_integration_cpu'


def tree(path):
    return ast.parse(path.read_text(encoding='utf-8'))


def execute(nodes, scope, path=COLLECTOR):
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(module, str(path), 'exec'), scope)


def collector():
    return next(node for node in tree(COLLECTOR).body
                if isinstance(node, ast.FunctionDef) and node.name == 'run_teacher_collection')


def references(node, name):
    return any(isinstance(item, ast.Name) and item.id == name for item in ast.walk(node))


def original_shadow_namespace():
    """Execute original pure contract code/constants without its torch imports."""
    shadow_path = SOURCE / 'tabletop_vision_command_shadow.py'
    scope = dict(__file__=str(shadow_path), deepcopy=copy.deepcopy,
                 hashlib=hashlib, json=json, Path=Path, math=math)
    for filename, names in (
            ('linear_coupling.py', {'ACTIVE'}),
            ('tabletop_velocity_limits.py', {'ACTIVE_VELOCITY_LIMIT_RAD_S'}),
            ('tabletop_free_wrist_grasp.py', {'CONTRACT'}),
            ('tabletop_surface_contract.py', {'ACTION_CONTRACT'})):
        nodes = [node for node in tree(SOURCE / filename).body
                 if (isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id in names
                             for target in node.targets))
                 or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                     and isinstance(node.value.func, ast.Attribute)
                     and isinstance(node.value.func.value, ast.Name)
                     and node.value.func.value.id in names and node.value.func.attr == 'update')]
        execute(nodes, scope, SOURCE / filename)
    nodes = [node for node in tree(shadow_path).body
             if (isinstance(node, ast.Assign) and references(node, 'UNSUPPORTED_CALLBACKS'))
             or (isinstance(node, ast.FunctionDef) and node.name in
                 ('action_contract_sha256', 'controller_source_sha256', 'controller_contract'))]
    execute(nodes, scope, shadow_path)
    return scope


def options(**changes):
    values = dict(vision_teacher_source=Path('teacher.pt'), vision_teacher_mpc=True,
        vision_teacher_mpc_profile='pressure_geometry_constrained',
        vision_sequence_context_sidecar=True, vision_student=True, native_sensor_batch=True,
        surface_force_sharing=True, vision_camera_mode='fixed', vision_width=224, vision_height=96,
        vision_action_distribution='tanh_gaussian', physics_hz=480, object_kind='solid_cylinder',
        solid_cylinder_radius_m=.008, num_envs=2, vision_teacher_max_control_steps=1900,
        vision_teacher_raw_chunk_frames=16, vision_mpc_state_capture=True,
        vision_mpc_aux_capture=True)
    values.update(changes)
    return NS(**values)


@contextmanager
def installed_modules(entries):
    absent = object()
    previous = {name: sys.modules.get(name, absent) for name in entries}
    sys.modules.update(entries)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class AuxIntegrationTests(unittest.TestCase):
    def validation(self):
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(SOURCE)]
        sys.modules.setdefault(PACKAGE, package)
        node = next(item for item in tree(COLLECTOR).body
                    if isinstance(item, ast.FunctionDef) and item.name == 'validate_options')
        scope = dict(__package__=PACKAGE, math=math)
        execute([node], scope)
        return scope['validate_options']

    def test_aux_cli_flag_is_explicit_and_default_off(self):
        nodes = [node for node in tree(CLI).body if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call) and node.value.args
                 and isinstance(node.value.args[0], ast.Constant)
                 and node.value.args[0].value == '--vision-mpc-aux-capture']
        self.assertEqual(len(nodes), 1, 'the opt-in auxiliary flag must exist in the original CLI')
        parser = argparse.ArgumentParser()
        execute(nodes, dict(parser=parser), CLI)
        self.assertFalse(parser.parse_args([]).vision_mpc_aux_capture)
        self.assertTrue(parser.parse_args(['--vision-mpc-aux-capture']).vision_mpc_aux_capture)

    def test_aux_requires_primary_capture_before_teacher_source_early_return(self):
        validate = self.validation()
        with self.assertRaisesRegex(ValueError, 'state capture|primary'):
            validate(NS(vision_mpc_aux_capture=True))
        self.assertIsNone(validate(NS()))
        self.assertIsNone(validate(options(vision_mpc_aux_capture=False)))

    def test_aux_preserves_every_primary_capture_restriction(self):
        validate = self.validation()
        self.assertIsNone(validate(options()))
        rejected = [dict(vision_teacher_source=None), dict(vision_teacher_mpc=False),
            dict(vision_sequence_context_sidecar=False), dict(native_sensor_batch=False),
            dict(surface_force_sharing=False), dict(vision_teacher_mpc_profile='local_pad'),
            dict(vision_width=320), dict(physics_hz=240), dict(vision_mirror_preprocessing='m2_full_mirror'),
            dict(bimanual_explicit_sdk_adapter=True), dict(bimanual_common_pause=True),
            dict(vision_stage6_native_config='alternate.json'), dict(resume='resume.pt')]
        for changes in rejected:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate(options(**changes))

    def test_cli_rejects_invalid_aux_before_app_launcher(self):
        module = tree(CLI)
        validation = next(node for node in module.body if isinstance(node, ast.Try)
                          and any(isinstance(item, ast.Call)
                                  and isinstance(item.func, ast.Name)
                                  and item.func.id == 'validate_teacher_collection_options'
                                  for item in ast.walk(node)))
        launch_lines = [node.lineno for node in ast.walk(module)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == 'AppLauncher']
        self.assertLess(validation.end_lineno, min(launch_lines))
        class Parser:
            def error(self, message):
                raise SystemExit(message)
        with self.assertRaisesRegex(SystemExit, 'state capture|primary'):
            execute([validation], dict(args=options(vision_mpc_state_capture=False),
                parser=Parser(), validate_teacher_collection_options=self.validation()), CLI)

    def setup_capture(self, *, enabled=True, sidecar=False, install_core_error=None,
                      lifecycle=False, events=None):
        function = collector()
        initialization = next(index for index, node in enumerate(function.body)
                              if isinstance(node, ast.Assign) and references(node, 'sequence_capture'))
        context = next(node for node in function.body[initialization:] if isinstance(node, ast.With))
        prelude = function.body[initialization:function.body.index(context)]
        self.assertTrue(any(references(node, 'mpc_aux_capture') for node in prelude),
                        'auxiliary capture must have a cleanup-safe initial value')
        run = context.body[0]
        loop = next(index for index, node in enumerate(run.body) if isinstance(node, ast.For))
        if events is None:events = []
        def finish_aux(**kwargs):
            events.append(('aux_finish', kwargs['error']))
            return {'capture':'aux'}
        def auxiliary(env, world, output, **kwargs):
            events.append(('aux_install', env, world, output, kwargs))
            return NS(name='aux', finish=finish_aux)
        def primary(env, world, output, **kwargs):
            events.append(('core_install', env, world, output, kwargs))
            if install_core_error:
                raise install_core_error
            return NS(name='core')
        prefix = PACKAGE + '.'
        modules = {prefix + 'tabletop_bimanual_mpc_aux_capture':
                   NS(install_mpc_aux_capture=auxiliary),
                   prefix + 'tabletop_bimanual_mpc_capture':
                   NS(install_mpc_state_capture=primary),
                   prefix + 'tabletop_vision_sequence_capture':
                   NS(SequenceContextCapture=lambda *a, **kw:NS(name='sequence'))}
        scope = dict(__package__=PACKAGE, uuid=uuid,
            args=NS(vision_sequence_context_sidecar=sidecar), sequence_context_sink=None,
            capture_mpc_state=enabled, capture_mpc_aux=enabled, candidate_path=None,
            teacher_audit=None, tripod_candidate=None, env=object(), device_world=object(), task=object(),
            output=Path('cpu-fixture'), genuine=True, failure=None,
            camera=NS(set_raw_frame_sink=lambda value:events.append(('camera', None))),
            raw=NS(close=lambda:events.append(('rgbd', None))))
        with installed_modules(modules):
            execute(prelude, scope)
            if lifecycle:
                # Execute the real exception handlers and full resource cleanup;
                # omit the later failure-artifact writer, which is unrelated to hooks.
                install = copy.deepcopy(run)
                install.body = install.body[:loop]
                install.finalbody = install.finalbody[:1]
                execute([install], scope)
            else:
                execute(run.body[:loop], scope)
        return scope, events, run

    def test_aux_installs_before_primary_with_shared_explicit_run_uuid(self):
        scope, events, _ = self.setup_capture(sidecar=True)
        self.assertEqual([event[0] for event in events], ['aux_install', 'core_install'])
        auxiliary, primary = events
        self.assertIs(auxiliary[1], scope['env'])
        self.assertIs(auxiliary[2], scope['device_world'])
        self.assertEqual(auxiliary[3], scope['output'] / 'mpc_aux_capture')
        self.assertEqual(primary[3], scope['output'] / 'mpc_state_capture')
        self.assertTrue(auxiliary[4]['enabled'])
        self.assertTrue(auxiliary[4]['native_runtime_verified'])
        self.assertEqual(auxiliary[4]['run_uuid'], primary[4]['run_uuid'])
        self.assertTrue(auxiliary[4]['run_uuid'])
        self.assertEqual(str(uuid.UUID(auxiliary[4]['run_uuid'])), auxiliary[4]['run_uuid'])
        self.assertEqual(auxiliary[4]['run_uuid'], scope['sequence_run_uuid'])

    def test_capture_is_installed_after_final_reset_and_before_first_step(self):
        calls = [node for node in ast.walk(collector()) if isinstance(node, ast.Call)]
        resets = [node.lineno for node in calls if isinstance(node.func, ast.Attribute)
                  and node.func.attr == 'reset']
        installs = {node.func.id:node.lineno for node in calls
                    if isinstance(node.func, ast.Name)
                    and node.func.id in ('SequenceContextCapture', 'install_mpc_aux_capture', 'install_mpc_state_capture')}
        steps = [node.lineno for node in calls if isinstance(node.func, ast.Attribute)
                 and node.func.attr == 'step']
        self.assertEqual(set(installs), {'SequenceContextCapture', 'install_mpc_aux_capture', 'install_mpc_state_capture'})
        self.assertLess(max(resets), installs['SequenceContextCapture'])
        self.assertLess(installs['SequenceContextCapture'], installs['install_mpc_aux_capture'])
        self.assertLess(installs['install_mpc_aux_capture'], installs['install_mpc_state_capture'])
        self.assertLess(installs['install_mpc_state_capture'], min(steps))

    def test_original_shadow_contract_retains_callback_and_replacement_guards(self):
        namespace = original_shadow_namespace()
        contract = namespace['controller_contract']
        class Task:
            policy_control = free_wrist_grasp = surface_force_sharing = True
            surface_redistribution = False
            def control(self, *a):pass
        class World:
            def advance(self, *a):pass
        task, world = Task(), World()
        valid = contract(task, world)
        self.assertEqual(valid['action_contract'], namespace['ACTION_CONTRACT'])
        self.assertEqual(valid['free_wrist_contract'], namespace['CONTRACT'])
        self.assertEqual(valid['action_contract_sha256'], namespace['action_contract_sha256']())
        self.assertEqual(valid['controller_source_sha256'], namespace['controller_source_sha256']())
        for name in ('policy_control', 'free_wrist_grasp', 'surface_force_sharing'):
            with self.subTest(flag=name):
                setattr(task, name, False)
                with self.assertRaisesRegex(ValueError, 'only native'):contract(task, world)
                delattr(task, name)
        for name in namespace['UNSUPPORTED_CALLBACKS']:
            with self.subTest(callback=name):
                setattr(task, name, object())
                with self.assertRaisesRegex(ValueError, 'unsupported controller callback'):contract(task, world)
                delattr(task, name)
        task.surface_redistribution = True
        with self.assertRaisesRegex(ValueError, 'surface_redistribution'):contract(task, world)
        del task.surface_redistribution
        for owner, name in ((task, 'control'), (world, 'advance')):
            with self.subTest(replacement=name):
                setattr(owner, name, getattr(owner, name))
                with self.assertRaisesRegex(ValueError, 'unsupported replaced control/advance'):contract(task, world)
                delattr(owner, name)

    def test_required_sidecar_uses_original_advance_and_restores_actual_hook_chain(self):
        sequence_class = next(node for node in tree(SOURCE / 'tabletop_vision_sequence_capture.py').body
                              if isinstance(node, ast.ClassDef) and node.name == 'SequenceContextCapture')
        shadow = original_shadow_namespace()
        namespace = dict(shadow=NS(controller_contract=shadow['controller_contract']))
        execute([sequence_class], namespace)
        sequence_type = namespace['SequenceContextCapture']
        sequence_type.write_unqualified = lambda *a, **kw:None
        self.validation()  # Establish the isolated, stdlib-only module namespace.
        aux_api = importlib.import_module(PACKAGE + '.tabletop_bimanual_mpc_aux_capture')
        core_api = importlib.import_module(PACKAGE + '.tabletop_bimanual_mpc_capture')
        for problem in (None, 'core_install', 'core_finish', 'aux_finish'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as temporary:
                events = []
                streams = []
                class TensorWorld:
                    n = 2
                    def advance(self, *a):pass
                    def refresh(self):pass
                class Task:
                    policy_control = free_wrist_grasp = surface_force_sharing = True
                    surface_redistribution = False
                    def control(self, *a):pass
                world, task = TensorWorld(), Task()
                world.w = NS(sensors=[])
                original_flag = object()
                world.capture_tripod_contact_evidence = original_flag
                prior = object()
                env = NS(n=2, task=task, diagnostic_episode_ids=[0, 0], joint_replay_observer=prior)
                class FixtureBridge:
                    def __init__(self, world, **kwargs):
                        if problem == 'core_install':raise OSError(problem)
                def in_memory_stream(*a, **kw):
                    stream = io.StringIO()
                    streams.append(stream)
                    return stream
                def identity_without_reading(path):
                    return dict(path=str(path), sha256='0' * 64)
                def attest_refresh(env, world):
                    if getattr(world.refresh, '__func__', None) is not TensorWorld.refresh:
                        raise ValueError('original TensorWorld.refresh required')
                def auxiliary(env, world, output, **kwargs):
                    capture = aux_api.install_mpc_aux_capture(env, world, output, **kwargs)
                    self.assertEqual(capture.advance_chain, 'SequenceContextCapture->original')
                    self.assertIs(capture._original.__self__, scope['sequence_capture'])
                    self.assertIs(capture._original.__func__, sequence_type._after_advance)
                    self.assertIs(scope['sequence_capture']._advance.__func__, TensorWorld.advance)
                    if problem == 'aux_finish':
                        original_raw = capture._stream
                        class BrokenAuxClose:
                            def close(self):
                                original_raw.close()
                                raise OSError(problem)
                        capture._stream = BrokenAuxClose()
                    original_finish = capture.finish
                    def finish(**kwargs):
                        events.append('aux_finish')
                        return original_finish(**kwargs)
                    capture.finish = finish
                    return capture
                def primary(env, world, output, **kwargs):
                    events.append('core_install')
                    capture = core_api.install_mpc_state_capture(env, world, output, **kwargs)
                    if problem == 'core_finish':
                        original_raw = capture._raw
                        class BrokenClose:
                            def close(self):
                                original_raw.close()
                                raise OSError(problem)
                        capture._raw = BrokenClose()
                    original_finish = capture.finish
                    def finish(**kwargs):
                        events.append('core_finish')
                        self.assertIs(env.joint_replay_observer, capture._observer)
                        self.assertIs(world.advance, scope['mpc_aux_capture']._advance_hook)
                        return original_finish(**kwargs)
                    capture.finish = finish
                    return capture
                function = collector()
                start = next(index for index, node in enumerate(function.body)
                             if isinstance(node, ast.Assign) and references(node, 'sequence_capture'))
                context = next(node for node in function.body[start:] if isinstance(node, ast.With))
                run = copy.deepcopy(context.body[0])
                loop = next(index for index, node in enumerate(run.body) if isinstance(node, ast.For))
                run.body = run.body[:loop]
                run.finalbody = run.finalbody[:1]
                scope = dict(__package__=PACKAGE, uuid=uuid,
                    args=NS(vision_sequence_context_sidecar=True), sequence_context_sink=None,
                    capture_mpc_state=True, capture_mpc_aux=True, candidate_path=None,
                    teacher_audit=None, tripod_candidate=None, env=env, device_world=world, task=task,
                    output=Path(temporary), genuine=True, failure=None,
                    camera=NS(set_raw_frame_sink=lambda value:events.append('camera')),
                    raw=NS(close=lambda:events.append('rgbd')))
                modules = {PACKAGE + '.tabletop_tensor_world':NS(TensorWorld=TensorWorld),
                    PACKAGE + '.tabletop_tensor_task':NS(TensorTask=Task),
                    PACKAGE + '.tabletop_vision_sequence_capture':NS(SequenceContextCapture=sequence_type),
                    PACKAGE + '.tabletop_bimanual_mpc_measurements':NS(BimanualMPCMeasurementBridge=FixtureBridge),
                    PACKAGE + '.tabletop_bimanual_mpc_aux_capture':NS(install_mpc_aux_capture=auxiliary),
                    PACKAGE + '.tabletop_bimanual_mpc_capture':NS(install_mpc_state_capture=primary)}
                error = None
                # Only SDK attestation/components, stream I/O and identities are
                # isolated. Shadow's real contract and Aux's trusted chain run.
                with installed_modules(modules), \
                     mock.patch.object(aux_api.gzip, 'open', side_effect=in_memory_stream), \
                     mock.patch.object(aux_api, '_identity', side_effect=identity_without_reading), \
                     mock.patch.object(core_api, '_identity', side_effect=identity_without_reading), \
                     mock.patch.object(aux_api, '_native_attestation', side_effect=attest_refresh), \
                     mock.patch.object(core_api, '_native_attestation', side_effect=attest_refresh), \
                     mock.patch.object(aux_api, '_native_components', return_value=(NS(), NS(), {'CPU_fixture':True})):
                    try:
                        execute(function.body[start:function.body.index(context)], scope)
                        execute([run], scope)
                    except Exception as exc:error = exc
                if problem is None:
                    self.assertIsNone(error, 'the original Shadow contract and exact trusted Sequence source chain must both pass')
                else:
                    self.assertIsInstance(error, OSError)
                    self.assertIn(problem, str(error))
                self.assertIn('aux_finish', events)
                self.assertEqual(events[-2:], ['camera', 'rgbd'])
                self.assertNotIn('advance', vars(world))
                self.assertIs(world.advance.__func__, TensorWorld.advance)
                self.assertIs(env.joint_replay_observer, prior)
                self.assertNotIn('control', vars(task))
                self.assertNotIn('refresh', vars(world))
                self.assertIs(world.refresh.__func__, TensorWorld.refresh)
                self.assertIs(world.capture_tripod_contact_evidence, original_flag)
                self.assertTrue(streams and all(stream.closed for stream in streams))

    def test_primary_install_failure_still_finishes_aux_and_original_cleanup(self):
        events = []
        with self.assertRaisesRegex(OSError, 'primary install'):
            self.setup_capture(install_core_error=OSError('primary install'),
                lifecycle=True, events=events)
        self.assertEqual([event[0] for event in events],
            ['aux_install', 'core_install', 'aux_finish', 'camera', 'rgbd'])
        self.assertIn('primary install', events[2][1])

    def test_disabled_capture_does_not_install_or_allocate_run_uuid(self):
        scope, events, _ = self.setup_capture(enabled=False)
        self.assertEqual(events, [])
        self.assertIsNone(scope['mpc_aux_capture'])
        self.assertIsNone(scope['sequence_run_uuid'])

    def test_both_submit_the_same_issued24_and_policy_tick_before_env_step(self):
        function = collector()
        loop = next(node for node in ast.walk(function)
                    if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                    and node.target.id == 'tick')
        step = next(node for node in loop.body if isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == 'step')
        statements = [node for node in loop.body if node is step
                      or (isinstance(node, ast.If) and any(references(node.test, name)
                          for name in ('mpc_aux_capture', 'mpc_state_capture')))]
        events = []
        class Action:
            def __init__(self, values):self.values = values
            def clone(self):return Action(list(self.values))
        issued = Action([.25] * 24)
        def submit(name):
            return lambda value, *, policy_step: events.append((name, value, policy_step))
        def advance(value):
            events.append(('step', value, None))
            return None, None, None, None
        scope = dict(mpc_state_capture=NS(submit=submit('core')),
            mpc_aux_capture=NS(submit=submit('aux')), issued=issued, tick=17,
            env=NS(step=advance))
        execute(statements, scope)
        self.assertEqual({event[0] for event in events[:-1]}, {'core', 'aux'})
        self.assertEqual(events[-1][0], 'step')
        for name, value, tick in events[:-1]:
            self.assertIs(value, issued)
            self.assertEqual(tick, 17)
            self.assertEqual(value.values, events[-1][1].values)

    def cleanup(self, *, core_error=None, aux_error=None, failure=None):
        function = collector()
        cleanup = next(node.finalbody for node in ast.walk(function)
            if isinstance(node, ast.Try) and node.finalbody
            and any(references(item, 'mpc_state_capture_report') for item in node.finalbody))
        self.assertTrue(any(references(node, 'mpc_aux_capture') for node in cleanup),
                        'auxiliary finish must be wired into the actual collector cleanup')
        events = []
        def finish(name, error):
            def close(**kwargs):
                events.append((name, kwargs.get('error')))
                if error:raise error
                return {'capture':name}
            return close
        scope = dict(mpc_state_capture=NS(finish=finish('core', core_error)),
            mpc_aux_capture=NS(finish=finish('aux', aux_error)), failure=failure,
            sequence_capture=NS(close=lambda:events.append(('sequence', None)),
                write_unqualified=lambda *a, **kw:None), output=Path('cpu-fixture'),
            tripod_candidate=None, teacher_audit=None,
            camera=NS(set_raw_frame_sink=lambda value:events.append(('camera', None))),
            raw=NS(close=lambda:events.append(('rgbd', None))))
        return cleanup, scope, events

    def test_cleanup_finishes_core_then_aux_and_records_separate_summaries(self):
        cleanup, scope, events = self.cleanup()
        execute(cleanup, scope)
        self.assertEqual([name for name, _ in events], ['core', 'aux', 'sequence', 'camera', 'rgbd'])
        self.assertEqual(scope['mpc_state_capture_report'], {'capture':'core'})
        self.assertEqual(scope['mpc_aux_capture_report'], {'capture':'aux'})

    def test_core_finish_failure_still_finishes_aux_and_original_cleanup(self):
        cleanup, scope, events = self.cleanup(core_error=OSError('core disk close'), failure=None)
        with self.assertRaisesRegex(OSError, 'core disk close'):
            execute(cleanup, scope)
        self.assertEqual([name for name, _ in events], ['core', 'aux', 'sequence', 'camera', 'rgbd'])

    def test_aux_finish_failure_still_runs_original_cleanup(self):
        cleanup, scope, events = self.cleanup(aux_error=OSError('aux disk close'), failure=None)
        with self.assertRaisesRegex(OSError, 'aux disk close'):
            execute(cleanup, scope)
        self.assertEqual([name for name, _ in events], ['core', 'aux', 'sequence', 'camera', 'rgbd'])

    def test_reports_preserve_raw_aux_qualification_in_separate_key(self):
        function = collector()
        reports = [node for node in function.body if isinstance(node, ast.If)
                   and any(references(node.test, name)
                           for name in ('mpc_state_capture_report', 'mpc_aux_capture_report'))]
        primary = {'native_runtime_verified':True, 'qualified':False}
        auxiliary = {'native_runtime_verified':True, 'passive_support_qualified':False,
                     'contact_pair_qualified':False, 'native_success_claimed':False}
        scope = dict(report={}, mpc_state_capture_report=primary, mpc_aux_capture_report=auxiliary)
        execute(reports, scope)
        self.assertEqual(scope['report'], {'mpc_state_capture':primary, 'mpc_aux_capture':auxiliary})


if __name__ == '__main__':
    unittest.main()
