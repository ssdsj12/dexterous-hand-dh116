"""CPU observer/archive evidence only; no SDK execution or Native qualification."""
import gzip
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import hashlib
import importlib.util
import ast
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
MODULE = 'go2_pvcnn.control.dh116.tabletop_bimanual_mpc_model_capture'


def api():
    assert (ROOT/'src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_model_capture.py').is_file(), 'model capture installer is missing'
    return importlib.import_module(MODULE)


class ModelCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)/'model'
        self.events = []
        sim = NS(current_time_step_index=18, current_time=2.)
        events = self.events
        class World:
            n=2; dt=1/480; _physics_steps=0
            def __init__(self): self.w=NS(sim=sim)
            def advance(self, *args):
                events.append('advance'); self._physics_steps+=1
                sim.current_time_step_index+=1; sim.current_time+=1/480
            def refresh(self): raise AssertionError('capture must not refresh')
        self.world=World()
        def prior(*args): self.events.append('prior')
        self.prior=prior
        self.env=NS(n=2, diagnostic_episode_ids=[0,0], joint_replay_observer=prior,
                    task=NS(done=[False,False]))
        self.original_advance=self.world.advance
        self.original_refresh=self.world.refresh

    def factory(self, *, clock_reader):
        owner=self
        class Provider:
            def capture_batch(self, *, clocks, active):
                owner.events.append('capture')
                for c,a in zip(clocks,active):
                    if a: owner.assertEqual(clock_reader(c.env_id),c)
                return tuple(NS(clock=c, fields={'q':dict(value=[10.+i],source='fixture.get_q',
                    unit='rad',valid=a)},raw={'fixture_only':True,'q':[10.+i]},
                    qualification={'planning_eligible':False}) for i,(c,a) in enumerate(zip(clocks,active)))
        self.provider=Provider()
        return self.provider, {'source_role':'CPU fixture only'}

    def install(self, **changes):
        options=dict(enabled=True,run_uuid='model-fixture',native_runtime_verified=False,provider_factory=self.factory)
        options.update(changes)
        capture=api().install_mpc_model_capture(self.env,self.world,self.output,**options)
        self.addCleanup(lambda: capture.finish(error='CPU cleanup'))
        return capture

    def submit(self,capture,step=0):
        action=[[0.]*24 for _ in range(2)];action[0][17]=-.25
        capture.submit(action,policy_step=step)
        action[0][17]=1.

    def physics(self,terminal=(False,False),live=(True,True)):
        self.world.advance()
        self.env.task.done=list(terminal)
        self.env.joint_replay_observer(self.env,{},list(live),list(terminal))

    def rows(self):
        with gzip.open(self.output/'model_480hz.jsonl.gz','rt',encoding='utf-8') as stream:
            return [json.loads(line) for line in stream]

    def test_default_off_does_not_inspect_even_output_or_options(self):
        class Bomb:
            def __getattribute__(self,key): raise AssertionError(key)
            def __fspath__(self): raise AssertionError('fspath')
        self.assertIsNone(api().install_mpc_model_capture(Bomb(),Bomb(),Bomb(),run_uuid=Bomb(),provider_factory=Bomb()))
        self.assertFalse(self.output.exists())

    def test_only_observer_is_wrapped_and_original_prior_runs_before_capture(self):
        capture=self.install(); self.submit(capture); self.physics()
        self.assertEqual(self.world.advance,self.original_advance)
        self.assertEqual(self.world.refresh,self.original_refresh)
        self.assertEqual(self.events,['advance','prior','capture'])
        report=capture.finish()
        self.assertIs(self.env.joint_replay_observer,self.prior)
        self.assertEqual(report['physical_frames'],1)
        self.assertFalse(report['candidate_future_dynamics_qualified'])
        self.assertEqual(report['Student_success_count'],0)

    def test_action_copy_clock_identity_and_eight_substeps_share_actual_submission(self):
        capture=self.install();self.submit(capture)
        for _ in range(8):self.physics()
        self.submit(capture,step=1);self.physics()
        capture.finish();rows=self.rows()
        self.assertEqual([r['control_substep'] for r in rows],list(range(1,9))+[1])
        self.assertTrue(all(r['issued_action24'][0][17]==-.25 for r in rows))
        self.assertEqual(rows[0]['clocks'][0]['episode_uid'],'model-fixture:0:0')
        self.assertEqual(rows[0]['clocks'][0]['sdk_step'],19)
        self.assertEqual(rows[-1]['policy_step'],1)
        self.assertEqual(rows[-1]['policy_clock_before']['world_step'],8)
        self.assertEqual(rows[0]['model_samples'][0]['fields']['q']['source'],'fixture.get_q')

    def test_first_terminal_survives_reset_and_closed_callback_is_noop(self):
        capture=self.install();self.submit(capture);self.physics(terminal=(True,False))
        self.env.diagnostic_episode_ids=[1,0]
        self.physics(terminal=(False,True));self.env.diagnostic_episode_ids=[1,1]
        self.physics(); report=capture.finish()
        first=json.loads((self.output/'terminal_snapshots.json').read_text())
        self.assertEqual(first['0']['clocks'][0]['world_step'],1)
        self.assertEqual(first['0']['clocks'][0]['reset_generation'],0)
        self.assertEqual(first['1']['clocks'][1]['world_step'],2)
        self.assertEqual(report['records_per_env'],[1,2])
        self.assertEqual(len(self.rows()),2)
        self.assertEqual(self.events.count('capture'),2)

    def test_finished_observer_reference_cannot_capture_or_replay_prior(self):
        capture=self.install();old=self.env.joint_replay_observer;capture.finish()
        old(self.env,{},[True,True],[False,False])
        self.assertEqual(self.events,[])

    def test_forged_native_and_non_boolean_flags_fail_before_files(self):
        with self.assertRaisesRegex(ValueError,'genuine|attestation|fixture'):
            self.install(native_runtime_verified=True)
        with patch.object(api(),'_native_attestation',side_effect=AssertionError('not boolean')):
            with self.assertRaisesRegex(ValueError,'bool'):
                self.install(native_runtime_verified=1)
        self.assertFalse(self.output.exists())

    def test_cpu_without_factory_cannot_acquire_sdk(self):
        with patch.object(api(),'_native_provider',side_effect=AssertionError('SDK accessed')):
            with self.assertRaisesRegex(ValueError,'fixture|attestation'):
                self.install(provider_factory=None)
        self.assertFalse(self.output.exists())

    def test_native_attestation_never_authorizes_fixture_factory(self):
        with patch.object(api(),'_native_attestation',return_value=None):
            with self.assertRaisesRegex(ValueError,'injected|fixture'):
                self.install(native_runtime_verified=True)
        self.assertFalse(self.output.exists())

    def test_observer_requires_submission_and_one_continuous_original_step(self):
        capture=self.install()
        with self.assertRaisesRegex(ValueError,'submit|policy'):self.physics()
        self.submit(capture);self.physics()
        with self.assertRaisesRegex(ValueError,'continuous|clock|duplicate'):
            self.env.joint_replay_observer(self.env,{},[True,True],[False,False])
        self.world.advance()
        with self.assertRaisesRegex(ValueError,'continuous|clock|missing'):self.physics()
        self.assertEqual(capture.finish(error='fixture clock failures')['physical_frames'],1)

    def test_next_policy_rejects_missing_eighth_observer_and_sdk_time_discontinuity(self):
        capture=self.install();self.submit(capture)
        for _ in range(7):self.physics()
        self.world.advance()
        with self.assertRaisesRegex(ValueError,'final|missing|clock'):self.submit(capture,step=1)
        self.world.w.sim.current_time+=.1
        with self.assertRaisesRegex(ValueError,'time|clock'):
            self.env.joint_replay_observer(self.env,{},[True,True],[False,False])

    def test_reset_without_first_terminal_cannot_enter_first_cohort(self):
        capture=self.install();self.submit(capture);self.env.diagnostic_episode_ids[0]=1
        with self.assertRaisesRegex(ValueError,'episode|reset'):self.physics()
        self.assertEqual(capture.finish(error='fixture reset failure')['physical_frames'],0)

    def test_provider_error_terminal_is_pinned_invalid_and_not_counted_complete(self):
        capture=self.install();self.submit(capture)
        def fail(**kwargs):raise RuntimeError('partial model getter failure')
        self.provider.capture_batch=fail
        with self.assertRaisesRegex(RuntimeError,'partial model'):self.physics(terminal=(True,False))
        report=capture.finish()
        self.assertFalse(report['capture_valid']);self.assertEqual(report['physical_frames'],0)
        self.assertEqual(report['raw_records'],1);self.assertEqual(report['provider_error_frames'],1)
        first=json.loads((self.output/'terminal_snapshots.json').read_text())['0']
        self.assertEqual(first['clock']['world_step'],1)
        self.assertFalse(first['complete_measurement']);self.assertFalse(first['terminal_measurement_valid'])
        self.assertTrue(first['first_terminal'][0])

    def test_raw_write_failure_preserves_terminal_without_success_count(self):
        capture=self.install();self.submit(capture);original=capture._stream
        class Broken:
            def write(self,value):raise OSError('raw failed')
            def close(self):original.close()
        capture._stream=Broken()
        with self.assertRaisesRegex(OSError,'raw failed'):self.physics(terminal=(True,False))
        report=capture.finish()
        self.assertEqual(report['physical_frames'],0);self.assertEqual(report['records_per_env'],[0,0])
        self.assertFalse(report['capture_valid']);self.assertEqual(report['terminal_archive_only_env_ids'],[0])
        first=json.loads((self.output/'terminal_snapshots.json').read_text())['0']
        self.assertFalse(first['terminal_measurement_valid']);self.assertIn('raw failed',first['archive_error'])

    def test_close_failure_still_restores_observer_and_invalidates_pinned_terminal(self):
        capture=self.install();self.submit(capture);self.physics(terminal=(True,False));original=capture._stream
        class Broken:
            def close(self):original.close();raise OSError('close failed')
        capture._stream=Broken()
        with self.assertRaisesRegex(OSError,'close failed'):capture.finish()
        self.assertTrue(original.closed);self.assertIs(self.env.joint_replay_observer,self.prior)
        report=json.loads((self.output/'model_capture_summary.json').read_text())
        self.assertFalse(report['capture_valid']);self.assertEqual(capture.finish(),report)
        first=json.loads((self.output/'terminal_snapshots.json').read_text())['0']
        self.assertFalse(first['terminal_measurement_valid']);self.assertIn('close failed',first['archive_error'])

    def test_factory_failure_creates_error_summary_without_installing_observer(self):
        def fail(**kwargs):raise RuntimeError('factory failed')
        with self.assertRaisesRegex(RuntimeError,'factory failed'):self.install(provider_factory=fail)
        self.assertIs(self.env.joint_replay_observer,self.prior)
        report=json.loads((self.output/'model_capture_summary.json').read_text())
        self.assertFalse(report['capture_valid']);self.assertIn('factory failed',report['error'])

    def test_foreign_observer_owner_cannot_create_actual_environment_evidence(self):
        capture=self.install();self.submit(capture);self.world.advance()
        with self.assertRaisesRegex(ValueError,'owner|environment'):
            self.env.joint_replay_observer(NS(),{},[True,True],[False,False])
        self.assertEqual(capture.finish(error='foreign owner')['raw_records'],0)

    def test_injected_provider_cannot_issue_qualification_inside_raw_dto(self):
        capture=self.install();self.submit(capture);original=self.provider.capture_batch
        def forged(**kwargs):
            samples=original(**kwargs);samples[0].qualification['planning_eligible']=True
            return samples
        self.provider.capture_batch=forged
        with self.assertRaisesRegex(ValueError,'qualification'):self.physics(terminal=(True,False))
        report=capture.finish()
        self.assertEqual(report['physical_frames'],0);self.assertFalse(report['planning_eligible'])

    def test_missing_observer_attribute_is_removed_on_finish(self):
        del self.env.joint_replay_observer
        capture=self.install();capture.finish()
        self.assertNotIn('joint_replay_observer',vars(self.env))

    def test_invalid_action_shape_bounds_and_nonfinite_never_call_provider(self):
        capture=self.install()
        for action in ([[0.]*24], [[0.]*23]*2, [[2.]*24]*2, [[float('nan')]*24]*2):
            with self.subTest(action=repr(action)[:40]):
                with self.assertRaises(ValueError):capture.submit(action,policy_step=0)
        self.assertEqual(self.events,[])

    def test_provider_cannot_relabel_clock_or_advance_actual_world_during_readback(self):
        for defect in ('relabel','advance'):
            with self.subTest(defect=defect):
                if defect=='advance':self.output=Path(self.temp.name)/'model-advance'
                capture=self.install();self.submit(capture);original=self.provider.capture_batch
                def torn(**kwargs):
                    samples=original(**kwargs)
                    if defect=='advance':self.world.advance()
                    else:samples[0].clock=NS(world_step=999)
                    return samples
                self.provider.capture_batch=torn
                with self.assertRaisesRegex(ValueError,'clock|episode'):self.physics(terminal=(True,False))
                report=capture.finish();self.assertEqual(report['physical_frames'],0)
                self.assertEqual(report['terminal_count'],1);self.assertFalse(report['capture_valid'])

    def test_rejected_returned_sample_retains_this_owned_partial_raw_with_nonfinite_encoding(self):
        capture=self.install();self.submit(capture);original=self.provider.capture_batch
        self.provider.last_sample=NS(raw={'stale_previous_sample':True})
        returned=[]
        def relabel(**kwargs):
            samples=original(**kwargs)
            samples[0].raw={'current_evidence':[123.,float('nan')]}
            samples[0].clock=NS(world_step=999)
            returned.extend(samples)
            return samples
        self.provider.capture_batch=relabel
        with self.assertRaisesRegex(ValueError,'clock'):self.physics(terminal=(True,False))
        returned[0].raw['current_evidence'][0]=999.
        report=capture.finish();row=self.rows()[0]
        self.assertIsNone(row['model_samples']);self.assertFalse(row['complete_measurement'])
        partial=row.get('provider_partial_model_raw')
        self.assertIsNotNone(partial,'already returned current samples must survive validation failure')
        self.assertEqual(partial[0]['raw']['current_evidence'],[123.,None])
        self.assertIsNone(row['provider_partial_model_raw_encoding_error'])
        self.assertTrue(any(e['path'][:1]==['provider_partial_model_raw'] and e['kind']=='nan'
            for e in row['raw_number_encoding']['nonfinite_float_values']))
        first=json.loads((self.output/'terminal_snapshots.json').read_text())['0']
        self.assertEqual(first['provider_partial_model_raw'][0]['raw']['current_evidence'][0],123.)
        self.assertEqual(report['physical_frames'],0);self.assertEqual(report['raw_records'],1)
        self.assertNotIn('stale_previous_sample',json.dumps(row))

    def test_unencodable_partial_raw_reports_encoding_error_and_keeps_original_provider_exception(self):
        capture=self.install();self.submit(capture);original=self.provider.capture_batch
        class NoEncode:
            __slots__=()
        def rejected(**kwargs):
            samples=original(**kwargs);samples[0].qualification['planning_eligible']=True
            samples[0].raw={'unencodable':NoEncode()}
            return samples
        self.provider.capture_batch=rejected
        with self.assertRaisesRegex(ValueError,'qualification'):self.physics(terminal=(True,False))
        capture.finish();row=self.rows()[0]
        self.assertIn('provider_partial_model_raw_encoding_error',row)
        self.assertIn('unsupported raw object',row['provider_partial_model_raw_encoding_error'])
        self.assertIsNone(row['provider_partial_model_raw'])
        self.assertIn('qualification',row['provider_error'])

    def test_prior_error_pins_validated_first_terminal_before_call_and_never_calls_model_getter(self):
        error=RuntimeError('real earlier auxiliary provider failed')
        def prior(*args):
            self.events.append('prior')
            raise error
        self.env.joint_replay_observer=prior
        capture=self.install();self.submit(capture)
        with self.assertRaises(RuntimeError) as raised:self.physics(terminal=(True,False))
        self.assertIs(raised.exception,error)
        self.assertEqual(self.events,['advance','prior'])
        report=capture.finish()
        self.assertEqual(report['terminal_count'],1,'validated first terminal must be pinned before prior observer')
        self.assertEqual(report['raw_records'],1);self.assertEqual(report['physical_frames'],0)
        self.assertFalse(report['capture_valid']);self.assertEqual(report['prior_observer_error_frames'],1)
        row=self.rows()[0];self.assertEqual(row['capture_status'],'prior_observer_error')
        self.assertIsNone(row['model_samples']);self.assertIsNone(row['provider_partial_model_raw'])
        self.assertFalse(row['terminal_measurement_valid']);self.assertIn('auxiliary',row['prior_observer_error'])
        first=json.loads((self.output/'terminal_snapshots.json').read_text())['0']
        self.assertEqual(first['clock']['world_step'],1);self.assertTrue(first['first_terminal'][0])
        self.assertIs(self.env.joint_replay_observer,prior)

    def test_prior_error_closed_terminal_cannot_be_overwritten_by_later_noop_or_reset(self):
        self.env.joint_replay_observer=lambda *args: (_ for _ in ()).throw(RuntimeError('prior failed'))
        capture=self.install();self.submit(capture)
        with self.assertRaisesRegex(RuntimeError,'prior failed'):self.physics(terminal=(True,True))
        self.assertEqual(len(capture.terminals),2,'both first terminals must be pinned before a failed prior observer')
        self.env.diagnostic_episode_ids=[1,1]
        self.env.joint_replay_observer(self.env,{},[True,True],[False,False])
        report=capture.finish();self.assertEqual(report['raw_records'],1)
        first=json.loads((self.output/'terminal_snapshots.json').read_text())
        self.assertEqual(first['0']['clocks'][0]['reset_generation'],0)
        self.assertEqual(first['0']['clock']['world_step'],1)
        self.assertEqual(first['1']['clock']['world_step'],1)

    def test_prior_is_not_called_and_no_terminal_is_pinned_for_unvalidated_callback(self):
        capture=self.install();self.submit(capture);self.world.advance()
        for defect in ('foreign_owner','terminal_mismatch','invalid_mask','episode_reset','duplicate_clock'):
            with self.subTest(defect=defect):
                events_before=tuple(self.events)
                owner=self.env;live=[True,True];terminal=[True,False]
                self.env.task.done=[True,False];self.env.diagnostic_episode_ids=[0,0]
                if defect=='foreign_owner':owner=NS()
                elif defect=='terminal_mismatch':self.env.task.done=[False,False]
                elif defect=='invalid_mask':live=[1,True]
                elif defect=='episode_reset':self.env.diagnostic_episode_ids=[1,0]
                else:
                    self.world._physics_steps=0
                    self.world.w.sim.current_time_step_index=18;self.world.w.sim.current_time=2.
                with self.assertRaises(ValueError):self.env.joint_replay_observer(owner,{},live,terminal)
                self.assertEqual(tuple(self.events),events_before)
                self.assertEqual(capture.terminals,{})
        self.assertEqual(capture.finish(error='rejected callbacks')['raw_records'],0)

    def test_prior_exception_survives_stream_failure_while_terminal_metadata_stays_pinned(self):
        error=RuntimeError('original prior error')
        self.env.joint_replay_observer=lambda *args: (_ for _ in ()).throw(error)
        capture=self.install();self.submit(capture);original=capture._stream
        class Broken:
            def write(self,value):raise OSError('secondary raw sink error')
            def close(self):original.close()
        capture._stream=Broken()
        with self.assertRaises(RuntimeError) as raised:self.physics(terminal=(True,False))
        self.assertIs(raised.exception,error)
        report=capture.finish();self.assertEqual(report['terminal_count'],1)
        self.assertEqual(report['raw_records'],0);self.assertEqual(report['physical_frames'],0)
        self.assertIn('secondary raw sink',report['error'])

    def test_provider_exception_survives_error_record_sink_failure_without_replaying_old_sample(self):
        capture=self.install();self.submit(capture);original=capture._stream
        error=RuntimeError('original provider failure')
        self.provider.last_sample=NS(raw={'stale':True})
        self.provider.capture_batch=lambda **kwargs: (_ for _ in ()).throw(error)
        class Broken:
            def write(self,value):raise OSError('secondary model sink failed')
            def close(self):original.close()
        capture._stream=Broken()
        try:self.physics(terminal=(True,False))
        except Exception as caught:self.assertIs(caught,error)
        else:self.fail('expected original provider exception')
        report=capture.finish();self.assertEqual(report['physical_frames'],0)
        self.assertEqual(report['raw_records'],0);self.assertEqual(report['terminal_count'],1)
        first=json.loads((self.output/'terminal_snapshots.json').read_text())['0']
        self.assertIsNone(first['provider_partial_model_raw']);self.assertNotIn('stale',json.dumps(first))

    def test_actual_aux_capture_provider_failure_keeps_both_archives_first_terminal(self):
        aux_api=importlib.import_module(MODULE.rsplit('.',1)[0]+'.tabletop_bimanual_mpc_aux_capture')
        error=RuntimeError('actual upstream Aux scene provider failed')
        class Scene:
            def capture_batch(self,**kwargs):raise error
        class Pairs:
            def begin_step(self,*args,**kwargs):pass
            def end_step(self,*args,**kwargs):return {'synthetic_contact_receipt':True}
            def abort_step(self,*args,**kwargs):pass
        def components(**kwargs):return Scene(),Pairs(),{'scope':'CPU real Aux hook chain'}
        aux=aux_api.install_mpc_aux_capture(self.env,self.world,Path(self.temp.name)/'aux',
            enabled=True,run_uuid='model-fixture',component_factory=components)
        self.addCleanup(lambda:aux.finish(error='CPU cleanup'))
        capture=self.install();self.submit(aux);self.submit(capture)
        with self.assertRaises(RuntimeError) as raised:self.physics(terminal=(True,False))
        self.assertIs(raised.exception,error)
        model_report=capture.finish();aux_report=aux.finish()
        self.assertEqual(model_report['terminal_count'],1);self.assertEqual(aux_report['terminal_count'],1)
        self.assertEqual(model_report['physical_frames'],0);self.assertEqual(model_report['raw_records'],1)
        self.assertEqual(model_report['prior_observer_error_frames'],1)
        self.assertEqual(self.events,['advance','prior'])
        first=json.loads((self.output/'terminal_snapshots.json').read_text())['0']
        aux_first=json.loads((Path(self.temp.name)/'aux'/'terminal_snapshots.json').read_text())['0']
        self.assertEqual(first['clocks'],aux_first['clocks'])
        self.assertEqual(first['issued_action24'],aux_first['issued_action24'])
        self.assertFalse(first['terminal_measurement_valid']);self.assertIsNone(first['model_samples'])
        self.assertIs(self.env.joint_replay_observer,self.prior)
        self.assertEqual(self.world.advance,self.original_advance)

    def test_terminal_disagreement_and_terminal_outside_live_mask_are_rejected(self):
        capture=self.install();self.submit(capture);self.world.advance()
        for live,terminal in (([False,True],[True,False]),([True,True],[True,False])):
            with self.subTest(live=live):
                with self.assertRaisesRegex(ValueError,'terminal'):
                    self.env.joint_replay_observer(self.env,{},live,terminal)
        self.assertEqual(capture.finish(error='invalid terminal fixture')['physical_frames'],0)

    @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('pytest'),'NumPy provider fixtures require the CPU pytest environment')
    def test_actual_source_reader_drift_rejects_real_provider_transaction_but_pins_terminal(self):
        spec=importlib.util.spec_from_file_location('_model_source_fixture',ROOT/'tests/test_bimanual_mpc_articulation_measurements.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        fixture=module.fixture();provider=fixture.provider
        sdk=Path(self.temp.name)/'sdk.py';sdk.write_text('actual_fixture_version=1\n')
        stage=NS(GetRootLayer=lambda:NS(identifier='actual-root',realPath='/actual/root'),
            GetSessionLayer=lambda:NS(identifier='actual-session',realPath=''),
            GetPrimAtPath=lambda path:NS(IsValid=lambda:True,GetTypeName=lambda:'Articulation'))
        geom=NS(GetStageMetersPerUnit=lambda stage:1.)
        physics=NS(GetStageKilogramsPerUnit=lambda stage:1.)
        provider.source_reader=api()._make_source_reader(stage=stage,hands=provider.hands,usd_geom=geom,usd_physics=physics)
        def factory(*,clock_reader):
            provider.clock_reader=clock_reader
            return provider, {'scope':'real provider and real source_reader; CPU fixture'}
        capture=self.install(provider_factory=factory);self.submit(capture)
        with patch.object(api().inspect,'getfile',return_value=str(sdk)):
            self.physics()
            sdk.write_text('actual_fixture_version=2_and_file_identity_changed\n')
            with self.assertRaisesRegex(RuntimeError,'source identity changed'):
                self.physics(terminal=(True,False))
        report=capture.finish();self.assertEqual(report['physical_frames'],1)
        self.assertEqual(report['raw_records'],2);self.assertEqual(report['provider_error_frames'],1)
        first=json.loads((self.output/'terminal_snapshots.json').read_text())['0']
        self.assertEqual(first['clocks'][0]['world_step'],2);self.assertFalse(first['terminal_measurement_valid'])

    @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('pytest'),'NumPy provider fixtures require the CPU pytest environment')
    def test_real_provider_partial_getter_na_and_nonfinite_raw_are_archived_honestly(self):
        spec=importlib.util.spec_from_file_location('_model_provider_fixture',ROOT/'tests/test_bimanual_mpc_articulation_measurements.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        fixture=module.fixture();provider=fixture.provider
        provider.hands['left'].root_physx_view.arrays['get_dof_positions'][1,0]=float('nan')
        def unavailable():raise RuntimeError('fixture unavailable M')
        provider.hands['left'].root_physx_view.get_generalized_mass_matrices=unavailable
        def factory(*,clock_reader):
            provider.clock_reader=clock_reader
            return provider, {'scope':'real provider with CPU view fixtures'}
        capture=self.install(provider_factory=factory);self.submit(capture);self.physics(terminal=(True,False));report=capture.finish()
        self.assertEqual(report['physical_frames'],1)
        row=self.rows()[0];sample=row['model_samples'][0]
        self.assertFalse(sample['fields']['left.q']['valid'])
        self.assertFalse(sample['fields']['left.generalized_mass_matrix_raw']['valid'])
        self.assertIn('unavailable M',sample['fields']['left.generalized_mass_matrix_raw']['missing_reason'])
        self.assertTrue(sample['fields']['right.q']['valid'])
        self.assertEqual(len(sample['fields']['right.q']['value']),17)
        raw=sample['raw']['hands']['left']['getters']['get_dof_positions']['value']
        self.assertIsNone(raw[1][0])
        enc=row['raw_number_encoding']['nonfinite_float_values']
        self.assertTrue(any(e['path'][-3:]==['value',1,0] and e['kind']=='nan' and len(e['float64_be_hex'])==16 for e in enc))
        self.assertFalse(any(sample['qualification'].values()))


class SourceReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.sdk=Path(self.temp.name)/'actual_wrapper.py';self.sdk.write_text('version=1\n')
        self.stage=NS(GetRootLayer=lambda:NS(identifier='root.usda',realPath='/actual/root.usda'),
            GetSessionLayer=lambda:NS(identifier='anon:session',realPath=''),
            GetPrimAtPath=lambda path:NS(IsValid=lambda:True,GetTypeName=lambda:'Articulation'))
        self.hands={side:NS(root_physx_view=NS()) for side in ('left','right')}
        self.geom=NS(GetStageMetersPerUnit=lambda stage:.01)
        self.physics=NS(GetStageKilogramsPerUnit=lambda stage:2.)

    def reader(self,**changes):
        options=dict(stage=self.stage,hands=self.hands,usd_geom=self.geom,usd_physics=self.physics)
        options.update(changes)
        return api()._make_source_reader(**options)

    def test_actual_stage_units_layers_path_and_view_type_file_are_preserved(self):
        with patch.object(api().inspect,'getfile',return_value=str(self.sdk)) as getfile:
            reader=self.reader();record=reader('right','/World/envs/env_1/rightHand/wrist_fixed')
        getfile.assert_called_once_with(type(self.hands['right'].root_physx_view))
        self.assertEqual(record['meters_per_unit'],.01);self.assertEqual(record['kilograms_per_unit'],2.)
        self.assertEqual(record['articulation_path'],'/World/envs/env_1/rightHand/wrist_fixed')
        self.assertEqual(record['sdk_sources'][0]['sha256'],hashlib.sha256(self.sdk.read_bytes()).hexdigest())
        self.assertIn('anon:session',record['stage_identity'])

    def test_source_file_or_layer_or_units_changes_are_visible_and_not_old_replayed_identity(self):
        with patch.object(api().inspect,'getfile',return_value=str(self.sdk)):
            reader=self.reader();before=reader('left','/actual/path')
            self.sdk.write_text('changed_version=2\n')
            self.geom.GetStageMetersPerUnit=lambda stage:1.
            self.stage.GetSessionLayer=lambda:NS(identifier='anon:new-session',realPath='')
            after=reader('left','/actual/path')
        self.assertNotEqual(before['sdk_sources'][0]['sha256'],after['sdk_sources'][0]['sha256'])
        self.assertNotEqual(before['stage_identity'],after['stage_identity'])
        self.assertNotEqual(before['meters_per_unit'],after['meters_per_unit'])

    def test_missing_actual_unit_getter_never_guesses_SI_and_nonexistent_prim_fails(self):
        with patch.object(api().inspect,'getfile',return_value=str(self.sdk)):
            self.physics.GetStageKilogramsPerUnit=lambda stage: (_ for _ in ()).throw(RuntimeError('unknown mass units'))
            with self.assertRaisesRegex(RuntimeError,'unknown mass'):self.reader()('left','/actual/path')
            self.stage.GetPrimAtPath=lambda path:NS(IsValid=lambda:False)
            with self.assertRaisesRegex(ValueError,'prim'):self.reader()('left','/actual/path')

    def test_replaced_active_stage_requires_rebuild_instead_of_reading_captured_old_stage(self):
        current=[self.stage]
        self.assertIn('stage_reader',api().inspect.signature(api()._make_source_reader).parameters)
        with patch.object(api().inspect,'getfile',return_value=str(self.sdk)):
            reader=self.reader(stage_reader=lambda:current[0]);reader('left','/actual/path')
            current[0]=NS()
            with self.assertRaisesRegex(ValueError,'stage|rebuild'):reader('left','/actual/path')

    def test_installer_contains_no_simulator_mutation_step_refresh_or_context_construction(self):
        source=Path(api().__file__).read_text(encoding='utf-8');tree=ast.parse(source)
        forbidden={'step','advance','refresh','control','create_simulation_view','SimulationContext'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                self.assertFalse(name in forbidden or name.startswith(('set_','write_joint_','write_data_','Apply')),name)
                if name=='update':
                    self.assertNotIn(ast.unparse(node.func.value),('world','self.world','hand','view','env','self.env'))
            if isinstance(node,(ast.Assign,ast.AnnAssign)):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target,ast.Attribute):
                        self.assertNotIn(ast.unparse(target),('world.advance','world.refresh','env.task.control','self.world.advance','self.world.refresh'))


if __name__=='__main__':unittest.main()
