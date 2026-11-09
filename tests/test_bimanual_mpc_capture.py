"""CPU archive integrity tests, not native or hardware contact evidence."""
import gzip
import ast
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[1] / 'src/go2_pvcnn/control/dh116'
PACKAGE = '_mpc_capture_cpu_fixture'


def module(name):
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(SOURCE)]
        sys.modules[PACKAGE] = package
    return importlib.import_module(PACKAGE + '.' + name)


def snapshot(env, step, *, policy=0, terminal=False, uid=None, sdk_step=None):
    api = module('tabletop_bimanual_mpc_state')
    clock = api.SampleClock(run_uuid='capture-fixture', env_id=env,
        episode_uid=uid or f'capture-fixture:{env}:0', world_step=step,
        world_time_s=step/480, sdk_step=step+18 if sdk_step is None else sdk_step,
        sdk_time_s=(step+18)/480, policy_step=policy, reset_generation=0,
        terminal=terminal)
    return api.make_mpc_state(clock, api.missing_fields(clock,
        reason='CPU fixture has no physical measurement', source='synthetic declaration'))


class CaptureArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'independent_capture'

    def archive(self, **changes):
        options = dict(n=2, run_uuid='capture-fixture',
            episode_uids=('capture-fixture:0:0','capture-fixture:1:0'),
            physics_hz=480, policy_hz=60, native_runtime_verified=False,
            action_owner='legacy_teacher_mpc')
        options.update(changes)
        archive = module('tabletop_bimanual_mpc_capture').MPCStateArchive(self.path, **options)
        self.addCleanup(lambda: archive.finish(error='CPU fixture cleanup'))
        return archive

    def submit(self, archive, action=None, *, policy=0, before=0):
        action = [[0.0]*24 for _ in range(2)] if action is None else action
        archive.submit(action, policy_step=policy, clock_before=dict(
            world_step=before, world_time_s=before/480,
            sdk_step=before+18, sdk_time_s=(before+18)/480))

    def rows(self):
        with gzip.open(self.path/'states_480hz.jsonl.gz','rt',encoding='utf-8') as stream:
            return [json.loads(line) for line in stream]

    def test_terminal_is_saved_before_noop_and_reset_cannot_overwrite_it(self):
        archive = self.archive();self.submit(archive)
        archive.record_batch((snapshot(0,1,terminal=True),snapshot(1,1)),
                             live=(True,True),terminal=(True,False))
        archive.record_batch((snapshot(0,2,uid='capture-fixture:0:1'),snapshot(1,2)),
                             live=(False,True),terminal=(False,False))
        result = archive.finish()
        rows = self.rows()
        self.assertEqual(result['record_count'],3)
        self.assertEqual(len(rows),3)
        self.assertEqual([r['state']['clock']['world_step'] for r in rows if r['env_id']==0],[1])
        terminal = json.loads((self.path/'terminal_snapshots.json').read_text(encoding='utf-8'))
        self.assertEqual(terminal['0']['state']['clock']['episode_uid'],'capture-fixture:0:0')
        self.assertEqual(terminal['0']['state']['clock']['world_step'],1)

    def test_actual_issued24_is_detached_and_owner_not_student(self):
        archive = self.archive();action = [[0.0]*24 for _ in range(2)];action[0][17]=-.25
        self.submit(archive,action);action[0][17]=1.0
        archive.record_batch((snapshot(0,1),snapshot(1,1)),live=(True,True),terminal=(False,False))
        archive.finish();row=self.rows()[0]
        self.assertEqual(row['issued_action24'][17],-.25)
        self.assertEqual(row['action_owner'],'legacy_teacher_mpc')
        self.assertFalse(row['counts_as_Student_success'])
        self.assertEqual(row['policy_clock_before']['world_step'],0)
        self.assertEqual(row['policy_clock_before']['sdk_step'],18)

    def test_batch_is_validated_before_any_partial_row_is_written(self):
        archive = self.archive();self.submit(archive)
        with self.assertRaisesRegex(ValueError,'episode|identity'):
            archive.record_batch((snapshot(0,1),snapshot(1,1,uid='different-episode')),
                                 live=(True,True),terminal=(False,False))
        self.assertEqual(archive.finish()['record_count'],0)
        self.assertEqual(self.rows(),[])

    def test_duplicate_skipped_or_misaligned_physics_is_rejected(self):
        for mode in ('duplicate','skipped','sdk'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                self.path=Path(folder)/'capture';archive=self.archive();self.submit(archive)
                archive.record_batch((snapshot(0,1),snapshot(1,1)),live=(True,True),terminal=(False,False))
                step = 1 if mode=='duplicate' else 3 if mode=='skipped' else 2
                sdk = 21 if mode=='sdk' else None
                with self.assertRaisesRegex(ValueError,'step|clock|consecutive'):
                    archive.record_batch((snapshot(0,step,sdk_step=sdk),snapshot(1,step)),
                                         live=(True,True),terminal=(False,False))
                archive.finish()

    def test_policy_identity_and_terminal_flag_must_match_actual_submission(self):
        archive = self.archive()
        with self.assertRaisesRegex(ValueError,'submit|policy'):
            archive.record_batch((snapshot(0,1),snapshot(1,1)),live=(True,True),terminal=(False,False))
        self.submit(archive)
        with self.assertRaisesRegex(ValueError,'policy'):
            archive.record_batch((snapshot(0,1,policy=1),snapshot(1,1)),live=(True,True),terminal=(False,False))
        with self.assertRaisesRegex(ValueError,'terminal'):
            archive.record_batch((snapshot(0,1),snapshot(1,1)),live=(True,True),terminal=(True,False))

    def test_missing_measurements_remain_none_and_do_not_issue_native_qualification(self):
        archive = self.archive();self.submit(archive)
        archive.record_batch((snapshot(0,1),snapshot(1,1)),live=(True,True),terminal=(False,False))
        result=archive.finish();row=self.rows()[0]
        self.assertIsNone(row['state']['fields']['robot.left_q']['value'])
        self.assertFalse(row['state']['fields']['robot.left_q']['valid'])
        self.assertEqual(result['unavailable_field_samples']['robot.left_q'],2)
        self.assertFalse(result['native_runtime_verified'])
        self.assertFalse(result['native_success_qualification_issued'])

    def test_output_is_exclusive_and_only_the_canonical_rate_and_shape_are_allowed(self):
        archive=self.archive()
        with self.assertRaises(FileExistsError):self.archive()
        with self.assertRaisesRegex(ValueError,'24|action'):
            self.submit(archive,[[0.0]*23 for _ in range(2)])
        with self.assertRaisesRegex(ValueError,'bounded|action'):
            self.submit(archive,[[2.0]*24 for _ in range(2)])

    def test_disabled_installer_does_not_inspect_world_or_environment(self):
        class Bomb:
            def __getattribute__(self,name):raise AssertionError('disabled path inspected native state')
        result=module('tabletop_bimanual_mpc_capture').install_mpc_state_capture(
            Bomb(),Bomb(),self.path,enabled=False)
        self.assertIsNone(result)
        self.assertFalse(self.path.exists())

    def native_fixture(self):
        events=[]
        sim=SimpleNamespace(current_time_step_index=18,current_time=18/480)
        world=SimpleNamespace(_physics_steps=0,dt=1/480,w=SimpleNamespace(sim=sim),
                              capture_tripod_contact_evidence=False)
        def prior(*args):events.append('prior')
        env=SimpleNamespace(n=2,diagnostic_episode_ids=[0,0],joint_replay_observer=prior,
                            task=SimpleNamespace(success=[False,False],failure=[0,0]))
        class Producer:
            def __init__(self,w,*,run_uuid):
                self.world=w;self.last_evidence={};self.calls=0
            def capture(self,**context):
                events.append('producer');self.calls+=1
                self.context=context
                self.last_evidence={'native_attested':False,'source':'synthetic CPU fixture'}
                return tuple(snapshot(i,self.world._physics_steps,policy=context['policy_step'],
                    terminal=context['terminal'][i],uid=context['episode_uids'][i]) for i in range(2))
        return env,world,Producer,events,prior

    def test_opt_in_observer_chains_prior_saves_terminal_and_restores_flag(self):
        env,world,producer,events,prior=self.native_fixture()
        capture=module('tabletop_bimanual_mpc_capture').install_mpc_state_capture(
            env,world,self.path,enabled=True,run_uuid='capture-fixture',
            native_runtime_verified=False,bridge_factory=producer)
        self.addCleanup(lambda:capture.finish(error='CPU fixture cleanup'))
        self.assertTrue(world.capture_tripod_contact_evidence)
        capture.submit([[0.0]*24 for _ in range(2)],policy_step=0)
        world._physics_steps=1;world.w.sim.current_time_step_index=19;world.w.sim.current_time=19/480
        env.joint_replay_observer(env,{},[True,True],[True,True])
        self.assertEqual(events,['prior','producer'])
        # A later reset/no-op invokes the old observer, never the state producer.
        env.diagnostic_episode_ids=[1,1]
        world._physics_steps=2;world.w.sim.current_time_step_index=20;world.w.sim.current_time=20/480
        env.joint_replay_observer(env,{},[False,False],[False,False])
        self.assertEqual(events,['prior','producer','prior'])
        result=capture.finish()
        self.assertEqual(result['record_count'],2)
        self.assertEqual(result['terminal_count'],2)
        self.assertFalse(result['native_runtime_verified'])
        self.assertIs(env.joint_replay_observer,prior)
        self.assertFalse(world.capture_tripod_contact_evidence)
        with gzip.open(self.path/'sensor_evidence_480hz.jsonl.gz','rt',encoding='utf-8') as stream:
            raw=[json.loads(line) for line in stream]
        self.assertEqual(len(raw),1)
        self.assertEqual(raw[0]['clock']['world_step'],1)
        self.assertFalse(raw[0]['producer_evidence']['native_attested'])

    def test_sensor_filter_manifest_records_actual_getter_without_partner_claim(self):
        env,world,producer,events,prior=self.native_fixture()
        paths=['/World/envs/env_0/leftHand/finger14_link','/World/envs/env_1/leftHand/finger14_link']
        world.w.sensors=[SimpleNamespace(
            cfg=SimpleNamespace(filter_prim_paths_expr=['/World/envs/env_.*/Wire']),
            contact_physx_view=SimpleNamespace(sensor_paths=paths,filter_count=1,
                                              filter_paths=['/World/envs/env_.*/Wire']),
            body_physx_view=SimpleNamespace(prim_paths=paths))]
        capture=module('tabletop_bimanual_mpc_capture').install_mpc_state_capture(
            env,world,self.path,enabled=True,run_uuid='capture-fixture',
            native_runtime_verified=False,bridge_factory=producer)
        result=capture.finish()
        source=result['sensor_sources']['sensors'][0]
        self.assertEqual(source['sensor_paths'],paths)
        self.assertEqual(source['body_prim_paths'],paths)
        self.assertEqual(source['filter_paths']['value'],['/World/envs/env_.*/Wire'])
        self.assertEqual(source['filter_paths']['resolved_semantics'],'unknown')
        self.assertFalse(result['sensor_sources']['per_contact_partner_id_recorded'])

    def test_closed_cohort_masks_new_reset_episode_before_bridge(self):
        env,world,producer,events,prior=self.native_fixture()
        capture=module('tabletop_bimanual_mpc_capture').install_mpc_state_capture(
            env,world,self.path,enabled=True,run_uuid='capture-fixture',
            native_runtime_verified=False,bridge_factory=producer)
        self.addCleanup(lambda:capture.finish(error='CPU fixture cleanup'))
        capture.submit([[0.0]*24 for _ in range(2)],policy_step=0)
        for step in (1,2):
            world._physics_steps=step;world.w.sim.current_time_step_index=18+step
            world.w.sim.current_time=(18+step)/480
            if step==2:env.diagnostic_episode_ids=[1,0]
            env.joint_replay_observer(env,{},[True,True],[step==1,False])
        self.assertEqual(capture.bridge.context['live'],(False,True))
        self.assertEqual(capture.bridge.context['terminal'],(False,False))
        capture.finish()
        with gzip.open(self.path/'sensor_evidence_480hz.jsonl.gz','rt',encoding='utf-8') as stream:
            rows=[json.loads(line) for line in stream]
        self.assertEqual(rows[1]['actual_live'],[True,True])
        self.assertEqual(rows[1]['live'],[False,True])

    def test_fixture_cannot_assert_genuine_native_attestation(self):
        env,world,producer,events,prior=self.native_fixture()
        with self.assertRaisesRegex(ValueError,'genuine.*Isaac|native.*fixture|attestation'):
            module('tabletop_bimanual_mpc_capture').install_mpc_state_capture(
                env,world,self.path,enabled=True,run_uuid='capture-fixture',
                native_runtime_verified=True,bridge_factory=producer)
        self.assertFalse(self.path.exists())
        self.assertFalse(world.capture_tripod_contact_evidence)

    def capture_options(self, **changes):
        values=dict(vision_mpc_state_capture=True,vision_teacher_source=Path('source.pt'),
            vision_teacher_mpc=True,vision_teacher_mpc_profile='pressure_geometry_constrained',
            vision_sequence_context_sidecar=True,vision_student=True,
            vision_camera_mode='fixed',vision_width=224,vision_height=96,
            vision_action_distribution='tanh_gaussian',vision_mirror_preprocessing='m0_raw',
            native_sensor_batch=True,surface_force_sharing=True,physics_hz=480,
            object_kind='solid_cylinder')
        values.update(changes)
        return SimpleNamespace(**values)

    def test_capture_options_require_explicit_legacy_native_teacher_source(self):
        validate=module('tabletop_bimanual_mpc_capture').validate_mpc_state_capture_options
        validate(self.capture_options())
        for change in (dict(vision_teacher_source=None),dict(vision_teacher_mpc=False),
                dict(vision_sequence_context_sidecar=False),dict(vision_camera_mode='moving'),
                dict(vision_width=160),dict(vision_action_distribution='gaussian'),
                dict(vision_mirror_preprocessing='m2_experiment'),
                dict(native_sensor_batch=False),dict(surface_force_sharing=False),
                dict(physics_hz=240),dict(object_kind='flexible_cable'),
                dict(vision_teacher_mpc_profile='local_pad'),
                dict(bimanual_teacher_physics_audit=True),dict(bimanual_common_pause=True)):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate(self.capture_options(**change))
        # Default OFF may inspect only its own switch, never demand SDK/Teacher options.
        validate(SimpleNamespace())

    def test_entrypoint_and_collection_wire_opt_in_around_actual_step(self):
        root=Path(__file__).resolve().parents[1]
        parser=(root/'scripts/train_dh116_tabletop_gpu_ppo.py').read_text(encoding='utf-8')
        collection=(root/'src/go2_pvcnn/control/dh116/tabletop_vision_teacher_collection.py').read_text(encoding='utf-8')
        self.assertIn("'--vision-mpc-state-capture'",parser)
        self.assertIn('validate_mpc_state_capture_options(args)',parser)
        # Verify control-flow order, not a mock that bypasses the real loop.
        install=collection.index('mpc_state_capture = install_mpc_state_capture(')
        self.assertGreater(install,collection.index('critic=env.reset()'))
        submit=collection.index('mpc_state_capture.submit(issued, policy_step=tick)')
        step=collection.index('critic,reward,done,extra=env.step(issued.clone())')
        self.assertLess(install,submit)
        self.assertLess(submit,step)
        finish=collection.index('mpc_state_capture.finish(error=failure)')
        self.assertGreater(finish,step)
        self.assertLess(finish,collection.index('teacher_audit.finish(error=failure)'))

    def test_install_failure_restores_prior_and_raw_flag(self):
        env,world,producer,events,prior=self.native_fixture()
        def broken(*args,**kwargs):raise RuntimeError('fixture producer initialization failed')
        with self.assertRaisesRegex(RuntimeError,'producer initialization'):
            module('tabletop_bimanual_mpc_capture').install_mpc_state_capture(
                env,world,self.path,enabled=True,run_uuid='capture-fixture',
                native_runtime_verified=False,bridge_factory=broken)
        self.assertIs(env.joint_replay_observer,prior)
        self.assertFalse(world.capture_tripod_contact_evidence)

    def test_actual_refresh_receipt_tracks_reset_updates_without_stepping(self):
        module_=module('tabletop_bimanual_mpc_capture')
        env,world,producer,events,prior=self.native_fixture()
        world.hands=[SimpleNamespace(data=SimpleNamespace(_sim_timestamp=.05)) for _ in range(2)]
        world.wire=SimpleNamespace(data=SimpleNamespace(_sim_timestamp=.05))
        calls=[]
        def original(sensors=True):
            calls.append(sensors)
            for item in [*world.hands,world.wire]:item.data._sim_timestamp+=world.dt
            return 'unchanged return value'
        world.refresh=original
        wrapper=module_._RefreshReadback(world)
        self.addCleanup(wrapper.close)
        self.assertIsNone(wrapper.receipt())
        self.assertEqual(world.refresh(sensors=False),'unchanged return value')
        first=wrapper.receipt()
        self.assertEqual(first['world_sdk_clock_before'],first['world_sdk_clock_after'])
        self.assertEqual(first['sequence'],1)
        self.assertEqual(first['data_generation_before']['wire'],.05)
        self.assertAlmostEqual(first['data_generation_after']['wire'],.05+1/480)
        # A later real step is already done by the original caller, never wrapper.
        world._physics_steps=1;world.w.sim.current_time_step_index=19;world.w.sim.current_time=19/480
        world.refresh()
        second=wrapper.receipt()
        self.assertEqual(second['sequence'],2)
        self.assertEqual(second['world_sdk_clock_after'],[1,19,19/480])
        self.assertEqual(calls,[False,True])
        # Returned evidence is detached from the stored receipt.
        first['data_generation_after']['wire']=999
        self.assertLess(wrapper.receipt()['data_generation_after']['wire'],1)
        wrapper.close()
        self.assertIs(world.refresh,original)

    def test_new_policy_requires_all_eight_consecutive_physical_steps(self):
        archive=self.archive();self.submit(archive)
        archive.record_batch((snapshot(0,1),snapshot(1,1)),live=(True,True),terminal=(False,False))
        with self.assertRaisesRegex(ValueError,'eight|8|60Hz'):
            self.submit(archive,policy=1,before=1)
        for step in range(2,9):
            archive.record_batch((snapshot(0,step),snapshot(1,step)),live=(True,True),terminal=(False,False))
        with self.assertRaisesRegex(ValueError,'consecutive|policy'):
            self.submit(archive,policy=2,before=8)
        self.submit(archive,policy=1,before=8)

    def test_physical_timestamp_must_match_original_substep_duration(self):
        archive=self.archive();self.submit(archive)
        api=module('tabletop_bimanual_mpc_state')
        states=[]
        for i in range(2):
            clock=replace(snapshot(i,1).clock,world_time_s=50.,sdk_time_s=100.)
            states.append(api.make_mpc_state(clock,api.missing_fields(clock,
                reason='synthetic bad clock',source='CPU negative fixture')))
        with self.assertRaisesRegex(ValueError,'time|clock|480'):
            archive.record_batch(states,live=(True,True),terminal=(False,False))
        self.assertEqual(archive.finish()['record_count'],0)

    def test_sensor_manifest_error_closes_both_archives_and_restores_runtime(self):
        env,world,producer,events,prior=self.native_fixture()
        class Broken:
            @property
            def contact_physx_view(self):raise RuntimeError('actual source getter unavailable')
        world.w.sensors=[Broken()]
        api=module('tabletop_bimanual_mpc_capture')
        # A sensor source getter failure must be represented as unavailable,
        # not strand open gzip streams during initialization.
        capture=api.install_mpc_state_capture(env,world,self.path,enabled=True,
            run_uuid='capture-fixture',native_runtime_verified=False,bridge_factory=producer)
        result=capture.finish()
        self.assertIs(env.joint_replay_observer,prior)
        self.assertFalse(world.capture_tripod_contact_evidence)
        self.assertIsNone(result['sensor_sources']['sensors'][0]['sensor_paths'])

    def test_raw_close_failure_still_closes_state_stream_and_restores_observer(self):
        env,world,producer,events,prior=self.native_fixture()
        capture=module('tabletop_bimanual_mpc_capture').install_mpc_state_capture(
            env,world,self.path,enabled=True,run_uuid='capture-fixture',
            native_runtime_verified=False,bridge_factory=producer)
        original=capture._raw
        class Broken:
            def close(self):
                original.close()
                raise OSError('simulated disk close failure')
        capture._raw=Broken()
        with self.assertRaisesRegex(OSError,'disk close'):
            capture.finish()
        self.assertTrue(capture.archive._stream.closed)
        self.assertIs(env.joint_replay_observer,prior)
        self.assertFalse(world.capture_tripod_contact_evidence)

    def test_collection_cleanup_survives_capture_finish_failure(self):
        source=SOURCE/'tabletop_vision_teacher_collection.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
            and n.name=='run_teacher_collection')
        cleanup=next(n.finalbody for n in ast.walk(function) if isinstance(n,ast.Try)
            and n.finalbody and any(isinstance(x,ast.Name) and x.id=='mpc_state_capture_report'
                for x in ast.walk(ast.Module(body=n.finalbody,type_ignores=[]))))
        events=[]
        def fail(**kwargs):raise OSError('capture disk close failure')
        scope=dict(mpc_state_capture=SimpleNamespace(finish=fail),failure=None,
            sequence_capture=SimpleNamespace(close=lambda:events.append('sequence'),
                write_unqualified=lambda *a,**kw:None),output=self.path,
            tripod_candidate=None,teacher_audit=None,
            camera=SimpleNamespace(set_raw_frame_sink=lambda value:events.append('camera')),
            raw=SimpleNamespace(close=lambda:events.append('rgbd')))
        with self.assertRaisesRegex(OSError,'disk close'):
            exec(compile(ast.fix_missing_locations(ast.Module(body=cleanup,type_ignores=[])),
                str(source),'exec'),scope)
        self.assertEqual(events,['sequence','camera','rgbd'])

    def test_invalid_nonfinite_raw_is_retained_with_exact_path_and_tag(self):
        env,world,producer,events,prior=self.native_fixture()
        class InvalidRaw(producer):
            def capture(self,**context):
                states=super().capture(**context)
                self.last_evidence.update(measurement_valid=False,
                    raw={'normal':[float('nan'),float('inf'),float('-inf'),None,1.25]})
                return states
        capture=module('tabletop_bimanual_mpc_capture').install_mpc_state_capture(
            env,world,self.path,enabled=True,run_uuid='capture-fixture',
            native_runtime_verified=False,bridge_factory=InvalidRaw)
        self.addCleanup(lambda:capture.finish(error='CPU cleanup'))
        capture.submit([[0.0]*24 for _ in range(2)],policy_step=0)
        world._physics_steps=1;world.w.sim.current_time_step_index=19;world.w.sim.current_time=19/480
        env.joint_replay_observer(env,{},[True,True],[True,True])
        result=capture.finish()
        self.assertEqual(result['record_count'],2)
        with gzip.open(self.path/'sensor_evidence_480hz.jsonl.gz','rt',encoding='utf-8') as stream:
            row=json.loads(next(stream))
        self.assertEqual(row['producer_evidence']['raw']['normal'],[None,None,None,None,1.25])
        entries=row['raw_number_encoding']['nonfinite_float_values']
        self.assertEqual([e['path'] for e in entries],
            [['producer_evidence','raw','normal',i] for i in range(3)])
        self.assertEqual([e['kind'] for e in entries],['nan','positive_infinity','negative_infinity'])
        self.assertTrue(all(len(e['float64_be_hex'])==16 for e in entries))
        self.assertFalse(row['producer_evidence']['measurement_valid'])


if __name__=='__main__':
    unittest.main()
