"""Pure CPU hook/clock archives; not Native contact or support qualification."""
import gzip
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[1]
PACKAGE='_stage7_aux_cpu'


def api():
    if PACKAGE not in sys.modules:
        package=types.ModuleType(PACKAGE)
        package.__path__=[str(ROOT/'src/go2_pvcnn/control/dh116')]
        sys.modules[PACKAGE]=package
    return importlib.import_module(PACKAGE+'.tabletop_bimanual_mpc_aux_capture')


class AuxCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.output=Path(self.temp.name)/'aux'
        self.events=[]
        sim=NS(current_time_step_index=18,current_time=18/480)
        events=self.events
        class World:
            n=2;dt=1/480;_physics_steps=0
            def __init__(self):self.w=NS(sim=sim)
            def advance(self,task,allowed,live):
                events.append('original_advance')
                self._physics_steps+=1
                sim.current_time_step_index+=1;sim.current_time+=1/480
                return task
        self.world=World()
        def prior(*args):self.events.append('prior_observer')
        self.prior=prior
        self.env=NS(n=2,diagnostic_episode_ids=[0,0],joint_replay_observer=prior,
                    task=NS(done=[False,False]))
        self.original=self.world.advance

    def components(self,*,clock_reader):
        events=self.events
        class Scene:
            def capture_batch(self,*,clocks,active):
                events.append('scene')
                for c,live in zip(clocks,active):
                    if live:assert clock_reader(c.env_id)==c
                return tuple(NS(clock=c,fields={},raw={'synthetic':True},qualification={
                    'native_attested':False,'passive_support_qualified':False}) for c in clocks)
        class Pairs:
            def __init__(self):self.before=None
            def begin_step(self,before,*,dt_s):self.before=before;events.append('pair_begin')
            def end_step(self,after,*,poll,read_clock,step_completed):
                events.append('pair_end');assert read_clock()==after
                assert step_completed is True
                return NS(before=self.before,after=after,native_attested=False,
                          measurement_valid=False,partner_body_id=None)
            def abort_step(self,reason):self.before=None;events.append('pair_abort')
        self.pairs=Pairs()
        return Scene(),self.pairs,{'scope':'CPU fixture only; not Native'}

    def install(self,**changes):
        kwargs=dict(enabled=True,run_uuid='aux-fixture',native_runtime_verified=False,
                    component_factory=self.components)
        kwargs.update(changes)
        capture=api().install_mpc_aux_capture(self.env,self.world,self.output,**kwargs)
        self.addCleanup(lambda:capture.finish(error='CPU cleanup'))
        return capture

    def submit(self,capture,step=0):
        action=[[0.0]*24 for _ in range(2)];action[0][17]=-.25
        capture.submit(action,policy_step=step);action[0][17]=1.

    def physics(self,terminal=(False,False),live=(True,True)):
        marker=object()
        self.assertIs(self.world.advance(marker,[True,True],list(live)),marker)
        self.env.task.done=list(terminal)
        self.env.joint_replay_observer(self.env,{},list(live),list(terminal))

    def rows(self):
        with gzip.open(self.output/'aux_480hz.jsonl.gz','rt',encoding='utf-8') as stream:
            return [json.loads(line) for line in stream]

    def test_disabled_does_not_inspect_any_runtime_or_create_output(self):
        class Bomb:
            def __getattribute__(self,key):raise AssertionError(key)
        self.assertIsNone(api().install_mpc_aux_capture(Bomb(),Bomb(),self.output,enabled=False))
        self.assertFalse(self.output.exists())

    def test_bracket_precedes_counter_increment_then_poll_after_task_terminal(self):
        capture=self.install();self.submit(capture)
        self.physics(terminal=(True,False))
        capture.finish()
        self.assertEqual(self.events[:5],['pair_begin','original_advance','prior_observer','pair_end','scene'])
        row=self.rows()[0]
        self.assertEqual(row['contact_report']['before']['world_step'],0)
        self.assertEqual(row['contact_report']['after']['world_step'],1)
        self.assertTrue(row['contact_report']['after']['terminal'])
        self.assertEqual(row['issued_action24'][0][17],-.25)
        self.assertEqual(row['policy_clock_before']['sdk_step'],18)
        self.assertFalse(row['native_success_qualification_issued'])

    def test_first_terminal_survives_reset_and_other_environment_continues(self):
        capture=self.install();self.submit(capture)
        self.physics(terminal=(True,False))
        self.env.diagnostic_episode_ids=[1,0];self.env.task.done=[False,False]
        self.physics(live=(True,True))
        report=capture.finish()
        rows=self.rows()
        self.assertEqual(rows[1]['actual_live'],[True,True])
        self.assertEqual(rows[1]['cohort_live'],[False,True])
        self.assertEqual(rows[1]['contact_report']['after']['env_id'],1)
        first=json.loads((self.output/'terminal_snapshots.json').read_text())
        self.assertEqual(first['0']['clocks'][0]['episode_uid'],'aux-fixture:0:0')
        self.assertEqual(first['0']['clocks'][0]['world_step'],1)
        self.assertEqual(report['records_per_env'],[1,2])

    def test_finish_restores_original_methods_observer_and_leaves_raw_flag_untouched(self):
        self.world.capture_tripod_contact_evidence='prior flag'
        capture=self.install();self.submit(capture);self.physics()
        capture.finish()
        self.assertEqual(self.world.advance,self.original)
        self.assertIs(self.env.joint_replay_observer,self.prior)
        self.assertEqual(self.world.capture_tripod_contact_evidence,'prior flag')

    def test_factory_failure_never_installs_hooks_and_preserves_error_archive(self):
        def bad(**kwargs):raise RuntimeError('factory fixture failure')
        with self.assertRaisesRegex(RuntimeError,'factory fixture'):
            self.install(component_factory=bad)
        self.assertEqual(self.world.advance,self.original)
        self.assertIs(self.env.joint_replay_observer,self.prior)
        self.assertTrue((self.output/'aux_capture_summary.json').exists())

    def test_original_advance_exception_aborts_bracket_without_extra_step(self):
        def broken(*args):raise RuntimeError('original failed')
        self.world.advance=broken
        capture=self.install();self.submit(capture)
        with self.assertRaisesRegex(RuntimeError,'original failed'):
            self.world.advance(None,None,None)
        self.assertIn('pair_abort',self.events)
        self.assertEqual(self.world._physics_steps,0)
        report=capture.finish(error='original failed')
        self.assertEqual(report['physical_frames'],0)

    def test_forged_native_or_injected_factories_never_attest(self):
        with self.assertRaisesRegex(ValueError,'genuine|attestation|fixture'):
            self.install(native_runtime_verified=True)
        self.assertFalse(self.output.exists())

    def test_no_submit_or_duplicate_observer_cannot_create_physical_evidence(self):
        capture=self.install()
        with self.assertRaisesRegex(ValueError,'submit|policy'):
            self.world.advance(None,None,None)
        self.submit(capture);self.physics()
        with self.assertRaisesRegex(ValueError,'bracket|duplicate|pending'):
            self.env.joint_replay_observer(self.env,{},[True,True],[False,False])
        self.assertEqual(capture.finish()['physical_frames'],1)

    def test_cpu_without_explicit_fixture_factory_never_imports_sdk(self):
        from unittest.mock import patch
        with patch.object(api(),'_native_components',side_effect=AssertionError('SDK factory accessed')):
            with self.assertRaisesRegex(ValueError,'fixture|attestation'):
                self.install(component_factory=None)
        self.assertFalse(self.output.exists())

    def test_real_contact_collector_retains_terminal_receipt_without_native_claim(self):
        module=importlib.import_module(PACKAGE+'.tabletop_bimanual_mpc_contact_provenance')
        def factory(*,clock_reader):
            scene,_,metadata=self.components(clock_reader=clock_reader)
            self.pairs=module.ContactPairProvenanceCollector(enabled=True,
                source=NS(get_full_contact_report=lambda:([],[],[])))
            return scene,self.pairs,metadata
        capture=self.install(component_factory=factory);self.submit(capture)
        self.physics(terminal=(True,False))
        capture.finish();report=self.rows()[0]['contact_report']
        self.assertTrue(report['bracket']['consistent'])
        self.assertTrue(report['receipt_clock_candidate']['terminal'])
        self.assertIsNone(report['clock_association'])
        self.assertFalse(report['native_attested'])
        self.assertFalse(report['planning_eligible'])

    def test_abort_failure_still_closes_raw_and_restores_hooks(self):
        capture=self.install();self.submit(capture)
        self.world.advance(None,None,[True,True])
        original_stream=capture._stream
        def broken_abort(reason):raise RuntimeError('fixture abort failed')
        self.pairs.abort_step=broken_abort
        with self.assertRaisesRegex(RuntimeError,'fixture abort failed'):
            capture.finish()
        self.assertTrue(original_stream.closed)
        self.assertEqual(self.world.advance,self.original)
        self.assertIs(self.env.joint_replay_observer,self.prior)
        summary=json.loads((self.output/'aux_capture_summary.json').read_text())
        self.assertIn('fixture abort failed',summary['error'])
        self.assertEqual(capture.finish(),summary)

    def test_close_failure_still_archives_terminal_and_marks_capture_invalid(self):
        capture=self.install();self.submit(capture);self.physics(terminal=(True,False))
        original_stream=capture._stream
        class BrokenClose:
            def close(self):
                original_stream.close()
                raise OSError('fixture gzip close failed')
        capture._stream=BrokenClose()
        with self.assertRaisesRegex(OSError,'fixture gzip close failed'):
            capture.finish()
        self.assertEqual(self.world.advance,self.original)
        self.assertIs(self.env.joint_replay_observer,self.prior)
        summary=json.loads((self.output/'aux_capture_summary.json').read_text())
        self.assertEqual(summary['terminal_count'],1)
        self.assertIn('fixture gzip close failed',summary['error'])
        self.assertFalse(summary['scene_qualification'])
        self.assertEqual(capture.finish(),summary)

    def test_missing_physical_observer_cannot_be_hidden_by_next_valid_bracket(self):
        capture=self.install();self.submit(capture)
        self.original(None,None,[True,True])
        with self.assertRaisesRegex(ValueError,'missing|continuous|clock'):
            self.physics()
        self.assertEqual(capture.finish(error='missing observer')['physical_frames'],0)

    def test_sdk_time_increment_must_match_actual_physics_period(self):
        capture=self.install();self.submit(capture)
        self.world.advance(None,None,[True,True])
        self.world.w.sim.current_time+=.1
        with self.assertRaisesRegex(ValueError,'time|clock'):
            self.env.joint_replay_observer(self.env,{},[True,True],[False,False])
        self.assertEqual(capture.finish(error='invalid SDK clock')['physical_frames'],0)

    def test_next_policy_rejects_missing_final_physics_observer(self):
        capture=self.install();self.submit(capture)
        for _ in range(7):self.physics()
        self.original(None,None,[True,True])
        with self.assertRaisesRegex(ValueError,'missing|continuous|clock|final'):
            self.submit(capture,step=1)
        self.assertEqual(capture.finish(error='missing final observer')['physical_frames'],7)

    def test_scene_failure_preserves_real_contact_receipt_and_first_terminal(self):
        module=importlib.import_module(PACKAGE+'.tabletop_bimanual_mpc_contact_provenance')
        def factory(*,clock_reader):
            class BrokenScene:
                def capture_batch(self,**kwargs):raise RuntimeError('fixture scene unavailable')
            self.pairs=module.ContactPairProvenanceCollector(enabled=True,
                source=NS(get_full_contact_report=lambda:([],[],[])))
            return BrokenScene(),self.pairs,{'scope':'CPU fixture only'}
        capture=self.install(component_factory=factory);self.submit(capture)
        with self.assertRaisesRegex(RuntimeError,'fixture scene unavailable'):
            self.physics(terminal=(True,False))
        result=capture.finish(error='fixture scene unavailable')
        self.assertEqual(result['terminal_count'],1)
        self.assertEqual(result['provider_error_frames'],1)
        row=self.rows()[0]
        self.assertEqual(row['capture_status'],'provider_error')
        self.assertIn('fixture scene unavailable',row['provider_error'])
        self.assertIsNone(row['scene_samples'])
        self.assertTrue(row['contact_report']['receipt_clock_candidate']['terminal'])
        first=json.loads((self.output/'terminal_snapshots.json').read_text())['0']
        self.assertEqual(first['clocks'][0]['episode_uid'],'aux-fixture:0:0')
        self.assertEqual(first['clock']['world_step'],1)
        self.assertFalse(first['native_success_qualification_issued'])

    def test_raw_write_failure_keeps_terminal_archive_without_counting_written_frame(self):
        capture=self.install();self.submit(capture)
        original_stream=capture._stream
        class BrokenWrite:
            def write(self,value):raise OSError('fixture raw write failed')
            def close(self):original_stream.close()
        capture._stream=BrokenWrite()
        with self.assertRaisesRegex(OSError,'fixture raw write failed'):
            self.physics(terminal=(True,False))
        summary=capture.finish()
        self.assertEqual(summary['physical_frames'],0)
        self.assertEqual(summary['records_per_env'],[0,0])
        self.assertEqual(summary['terminal_count'],1)
        self.assertEqual(summary['terminal_archive_only_env_ids'],[0])
        self.assertIn('fixture raw write failed',summary['error'])
        first=json.loads((self.output/'terminal_snapshots.json').read_text())['0']
        self.assertEqual(first['clocks'][0]['world_step'],1)
        self.assertTrue(first['first_terminal'][0])
        self.assertFalse(first['native_success_qualification_issued'])

    def chain_fixture(self):
        class World:
            def advance(self,*args):pass
        class Task:
            def control(self,*args):pass
        class Sequence:
            def _after_advance(self,*args):return self._advance(*args)
            def _before_control(self,*args):return self._control(*args)
        world,task=World(),Task()
        sequence=Sequence();sequence.world=world;sequence.task=task
        sequence._advance=world.advance;sequence._control=task.control
        sequence.closed=False;sequence.require_native_readback=True
        env=NS(task=task)
        return env,world,sequence,dict(world_type=World,task_type=Task,sequence_type=Sequence)

    def test_source_chain_accepts_original_or_exact_existing_sequence_only(self):
        env,world,sequence,types=self.chain_fixture()
        self.assertEqual(api()._validated_advance_chain(env,world,**types),'original')
        world.advance=sequence._after_advance;env.task.control=sequence._before_control
        self.assertEqual(api()._validated_advance_chain(env,world,**types),
                         'SequenceContextCapture->original')

    def test_original_advance_does_not_authorize_replaced_task_control(self):
        env,world,sequence,types=self.chain_fixture()
        env.task.control=lambda *args:None
        with self.assertRaisesRegex(ValueError,'control|source'):
            api()._validated_advance_chain(env,world,**types)

    def test_source_chain_rejects_unknown_nested_or_wrong_episode_owner(self):
        for defect in ('unknown','nested','different_world','different_task','closed','unattested','control'):
            with self.subTest(defect=defect):
                env,world,sequence,types=self.chain_fixture()
                world.advance=sequence._after_advance;env.task.control=sequence._before_control
                if defect=='unknown':world.advance=lambda *args:None
                elif defect=='nested':sequence._advance=lambda *args:None
                elif defect=='different_world':sequence.world=object()
                elif defect=='different_task':sequence.task=object()
                elif defect=='closed':sequence.closed=True
                elif defect=='unattested':sequence.require_native_readback=False
                else:env.task.control=lambda *args:None
                with self.assertRaisesRegex(ValueError,'original|source|Sequence|control'):
                    api()._validated_advance_chain(env,world,**types)

    def test_non_boolean_native_flag_rejected_before_any_attestation(self):
        from unittest.mock import patch
        with patch.object(api(),'_native_attestation',side_effect=AssertionError('attestation accessed')):
            with self.assertRaisesRegex(ValueError,'bool'):
                self.install(native_runtime_verified=1)


if __name__=='__main__':unittest.main()
