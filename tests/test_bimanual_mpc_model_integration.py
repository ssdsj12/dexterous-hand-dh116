"""Real collector lifecycle boundaries, using CPU observers rather than Isaac."""
import ast
import copy
from pathlib import Path
import unittest
import uuid
from types import SimpleNamespace as NS

from test_bimanual_mpc_aux_integration import (
    PACKAGE, collector, execute, installed_modules, references)


class ModelIntegrationTests(unittest.TestCase):
    def lifecycle(self, *, model_enabled=True, probe=None, filtered=False, install_error=None,
                  finish_error=None, core_install_error=None, core_finish_error=None):
        function = collector()
        start = next(i for i, node in enumerate(function.body)
                     if isinstance(node, ast.Assign) and references(node, 'sequence_capture'))
        context = next(node for node in function.body[start:] if isinstance(node, ast.With))
        run = copy.deepcopy(context.body[0])
        loop = next(i for i, node in enumerate(run.body) if isinstance(node, ast.For))
        run.body = run.body[:loop]
        run.finalbody = run.finalbody[:1]
        events = []
        def finish(name, error=None):
            def close(**kwargs):
                events.append((name+'_finish', kwargs))
                if error:raise error
                return {'capture':name, 'qualified':False}
            return close
        def install(name, error=None, close_error=None):
            def create(env, world, output, **kwargs):
                events.append((name+'_install', kwargs))
                if error:raise error
                return NS(finish=finish(name, close_error))
            return create
        def sequence(*args, **kwargs):
            events.append(('sequence_install', kwargs))
            return NS(close=lambda:events.append(('sequence_close', {})),
                      write_unqualified=lambda *a, **kw:None)
        modules = {
            PACKAGE+'.tabletop_vision_sequence_capture':NS(SequenceContextCapture=sequence),
            PACKAGE+'.tabletop_bimanual_mpc_aux_capture':
                NS(install_mpc_aux_capture=install('aux')),
            PACKAGE+'.tabletop_bimanual_mpc_model_capture':
                NS(install_mpc_model_capture=install('model', install_error, finish_error)),
            PACKAGE+'.tabletop_bimanual_mpc_capture':
                NS(install_mpc_state_capture=install('core', core_install_error, core_finish_error))}
        scope = dict(__package__=PACKAGE, uuid=uuid,
            args=NS(vision_sequence_context_sidecar=True,vision_mpc_model_capture=model_enabled,
                    vision_mpc_contact_delivery_probe=probe,
                    vision_mpc_filtered_contact_capture=filtered), sequence_context_sink=None,
            capture_mpc_state=True,capture_mpc_aux=True,candidate_path=None,
            teacher_audit=None,tripod_candidate=None,env=object(),device_world=object(),task=object(),
            output=Path('cpu-fixture'),genuine=True,failure=None,
            camera=NS(set_raw_frame_sink=lambda value:events.append(('camera_close', {}))),
            raw=NS(close=lambda:events.append(('rgbd_close', {}))))
        error = None
        with installed_modules(modules):
            execute(function.body[start:function.body.index(context)], scope)
            try:execute([run], scope)
            except BaseException as exc:error=exc
        return scope, events, error

    def test_install_and_cleanup_are_exact_lifo_and_share_uuid(self):
        scope, events, error = self.lifecycle()
        self.assertIsNone(error)
        self.assertEqual([name for name, _ in events],
            ['sequence_install','aux_install','model_install','core_install',
             'core_finish','model_finish','aux_finish','sequence_close','camera_close','rgbd_close'])
        for name, options in events:
            if name in ('aux_install','model_install','core_install'):
                self.assertEqual(options['run_uuid'],scope['sequence_run_uuid'])
                self.assertIs(options['native_runtime_verified'],True)
        self.assertEqual(scope['mpc_model_capture_report'],{'capture':'model','qualified':False})

    def test_default_does_not_import_or_install_model(self):
        scope, events, error = self.lifecycle(model_enabled=False)
        self.assertIsNone(error)
        self.assertNotIn('model_install',[name for name,_ in events])
        self.assertNotIn('model_finish',[name for name,_ in events])
        self.assertIsNone(scope['mpc_model_capture'])

    def test_model_install_failure_preserves_original_error_and_finishes_aux(self):
        failure = OSError('model install')
        _, events, error = self.lifecycle(install_error=failure)
        self.assertIs(error,failure)
        self.assertEqual([name for name,_ in events][-4:],
                         ['aux_finish','sequence_close','camera_close','rgbd_close'])

    def test_core_install_failure_finishes_model_then_aux(self):
        failure = OSError('core install')
        _, events, error = self.lifecycle(core_install_error=failure)
        self.assertIs(error,failure)
        self.assertEqual([name for name,_ in events][-5:],
            ['model_finish','aux_finish','sequence_close','camera_close','rgbd_close'])

    def test_either_finish_failure_still_closes_all_inner_resources(self):
        for option in ('finish_error','core_finish_error'):
            failure = OSError(option)
            with self.subTest(option=option):
                _, events, error = self.lifecycle(**{option:failure})
                self.assertIs(error,failure)
                self.assertEqual([name for name,_ in events][-6:],
                    ['core_finish','model_finish','aux_finish','sequence_close','camera_close','rgbd_close'])

    def test_probe_option_is_forwarded_only_to_aux(self):
        _, events, error = self.lifecycle(model_enabled=False,probe='post_step_poll')
        self.assertIsNone(error)
        for name, options in events:
            if name=='aux_install':
                self.assertEqual(options['contact_delivery_probe_mode'],'post_step_poll')
            elif name=='core_install':
                self.assertNotIn('contact_delivery_probe_mode',options)

    def test_filtered_metadata_opt_in_is_forwarded_without_installing_model(self):
        _, events, error = self.lifecycle(model_enabled=False,filtered=True)
        self.assertIsNone(error)
        self.assertNotIn('model_install',[name for name,_ in events])
        for name, options in events:
            if name=='aux_install':
                self.assertIs(options.get('capture_filtered_contact_metadata'),True)
            elif name=='core_install':
                self.assertNotIn('capture_filtered_contact_metadata',options)

    def test_all_three_observers_receive_identical_issued_action_before_physics(self):
        loop = next(node for node in ast.walk(collector())
                    if isinstance(node,ast.For) and isinstance(node.target,ast.Name)
                    and node.target.id=='tick')
        step = next(node for node in loop.body if isinstance(node,ast.Assign)
                    and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                    and node.value.func.attr=='step')
        names=('mpc_state_capture','mpc_model_capture','mpc_aux_capture')
        nodes=[node for node in loop.body if node is step or
               (isinstance(node,ast.If) and any(references(node.test,name) for name in names))]
        events=[]
        class Action:
            def clone(self):return self
        issued=Action()
        def submit(name):
            return NS(submit=lambda action,*,policy_step:events.append((name,action,policy_step)))
        def advance(action):
            events.append(('step',action,7))
            return None,None,None,None
        scope=dict(zip(names,(submit('core'),submit('model'),submit('aux'))))
        scope.update(issued=issued,tick=7,env=NS(step=advance))
        execute(nodes,scope)
        self.assertEqual({row[0] for row in events[:-1]},{'core','model','aux'})
        self.assertEqual(events[-1][0],'step')
        self.assertTrue(all(row[1] is issued and row[2]==7 for row in events))


if __name__=='__main__':
    unittest.main()
