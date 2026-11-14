"""CPU reset-buffer fixtures do not provide native capture evidence."""
import copy
from types import SimpleNamespace
import pytest
import torch

from test_dh116_stereo_trials import FixtureBridge, FrozenPolicy, manifest
from go2_pvcnn.control.dh116.tabletop_stereo_contract import digest
from go2_pvcnn.control.dh116 import tabletop_stereo_training_cli as collector


class Environment:
    def __init__(self, duplicate=False):
        self.bridge = FixtureBridge()
        self.camera = SimpleNamespace(calibration=SimpleNamespace(sha256='4'*64))
        self.clears = []
        self.duplicate = duplicate

    def clear_sensor_history(self):
        self.clears.append(self.bridge.serial)

    def snapshot(self):
        bridge = self.bridge
        return SimpleNamespace(identities=[dict(env_id=i,episode_uid=bridge.uid[i],
            physical_initial_condition_sha256=('f'*64 if self.duplicate else digest({
                'q':bridge.device_world.q[i].tolist(),'roots':bridge.device_world.roots[i].tolist()})),
            source_episode_group=bridge.bound[i]['trial_id']) for i in range(bridge.n)])


def fake_collect(environment, teacher, steps):
    assert steps == 1
    before = environment.snapshot()
    return [dict(identities=before.identities, done=torch.zeros(2,dtype=torch.bool),
                 candidate_only=True, native_qualified=False)]


def test_eight_authored_states_use_real_reset_buffers_and_unique_readback(tmp_path,monkeypatch):
    monkeypatch.setattr(collector,'collect_candidate_reference',fake_collect)
    environment = Environment()
    saved = environment.bridge.device_world.seed_q[0].clone()
    rows, report = collector.collect_cold_candidate_reference(environment,FrozenPolicy(),manifest(4),
        output=tmp_path/'capture',steps_per_initial=4,minimum_per_style=4,schema_only=True)
    assert len(rows) == 16 and report['samples'] == 32
    assert report['independent_actual_initials'] == 8
    assert len(environment.clears) == 4
    assert not report['native_qualified'] and not report['native_execution_performed']
    assert report['successful_demonstration_pool'] is False
    assert report['reset_buffers_restored'] is True
    torch.testing.assert_close(environment.bridge.device_world.seed_q[0],saved,rtol=0,atol=0)


def test_duplicate_actual_initials_are_rejected_before_actions(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(collector,'collect_candidate_reference',lambda *args:(calls.append(1),[])[1])
    with pytest.raises(ValueError,match='duplicate actual'):
        collector.collect_cold_candidate_reference(Environment(duplicate=True),FrozenPolicy(),manifest(4),
            output=tmp_path/'capture',steps_per_initial=4,minimum_per_style=4,schema_only=True)
    assert not calls


@pytest.mark.parametrize('key', ['source_sha256','camera_sha256','physics_sha256'])
def test_mismatched_manifest_identity_is_rejected_before_reset(tmp_path,key):
    environment=Environment(); value=manifest(4); value['metadata'][key]='0'*64
    with pytest.raises(ValueError,match='manifest'):
        collector.collect_cold_candidate_reference(environment,FrozenPolicy(),value,
            output=tmp_path/'capture',steps_per_initial=4,minimum_per_style=4,schema_only=True)
    assert environment.clears == [] and environment.bridge.serial == 0


def test_first_terminal_ends_batch_without_collecting_auto_reset_episode(tmp_path,monkeypatch):
    def terminal(environment,teacher,steps):
        rows=fake_collect(environment,teacher,steps);rows[0]['done'][0]=True
        return rows
    monkeypatch.setattr(collector,'collect_candidate_reference',terminal)
    rows,report=collector.collect_cold_candidate_reference(Environment(),FrozenPolicy(),manifest(4),
        output=tmp_path/'capture',steps_per_initial=4,minimum_per_style=4,schema_only=True)
    assert len(rows)==4 and report['samples']==8
    assert all(row['stopped_on_terminal'] for row in report['batches'])


def test_native_candidate_path_rejects_cpu_fixture(tmp_path):
    with pytest.raises(ValueError,match='native'):
        collector.collect_cold_candidate_reference(Environment(),FrozenPolicy(),manifest(4),
            output=tmp_path/'capture',steps_per_initial=4,minimum_per_style=4)


def test_later_failure_preserves_partial_rgb_teacher_records_outside_complete_dataset(tmp_path,monkeypatch):
    import json
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
    calls=[]
    def fail_after_batch(environment,teacher,steps):
        calls.append(1)
        if len(calls)>4:
            raise RuntimeError('fixture second-batch failure')
        rows=fake_collect(environment,teacher,steps)
        rows[0]['sensor_history']={'left_rgb':torch.ones(2,4,3,16,16)}
        rows[0]['teacher_mean']=torch.full((2,24),.1)
        return rows
    monkeypatch.setattr(collector,'collect_candidate_reference',fail_after_batch)
    with pytest.raises(RuntimeError,match='second-batch'):
        collector.collect_cold_candidate_reference(Environment(),FrozenPolicy(),manifest(4),
            output=tmp_path/'capture',steps_per_initial=4,minimum_per_style=4,schema_only=True)
    failed=json.loads((tmp_path/'capture/failure.json').read_text())
    artifact=failed['partial_candidates_artifact']
    assert file_sha256(artifact['path'])==artifact['sha256']
    payload=torch.load(artifact['path'],weights_only=True)
    assert payload['version'] != collector.CANDIDATE_VERSION
    assert payload['status']=='failed_incomplete' and payload['candidate_only'] is True
    assert payload['complete_cohort'] is False and payload['successful_demonstration_pool'] is False
    assert len(payload['records'])==4
    assert payload['records'][0]['sensor_history']['left_rgb'].sum()==6144
    assert payload['records'][0]['teacher_mean'].shape==(2,24)
    assert not (tmp_path/'capture/cohort_report.json').exists()


def test_reference_filename_separates_resets_at_same_policy_tick(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_runtime import state_reference_stem
    a=[{'episode_uid':'run:0:1'},{'episode_uid':'run:1:1'}]
    b=[{'episode_uid':'run:0:2'},{'episode_uid':'run:1:2'}]
    assert state_reference_stem(tmp_path,4,a) != state_reference_stem(tmp_path,4,b)
    assert state_reference_stem(tmp_path,4,a) == state_reference_stem(tmp_path,4,copy.deepcopy(a))


def test_checkpoint_change_during_capture_preserves_failed_records(tmp_path,monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
    checkpoint=tmp_path/'teacher.pt';checkpoint.write_bytes(b'original')
    environment=Environment();environment.bridge.checkpoint_sha256=file_sha256(checkpoint)
    value=manifest(4);value['metadata']['source_sha256']=file_sha256(checkpoint)
    calls=[]
    def mutate(environment,teacher,steps):
        calls.append(1)
        checkpoint.write_bytes(b'mutated')
        return fake_collect(environment,teacher,steps)
    monkeypatch.setattr(collector,'collect_candidate_reference',mutate)
    with pytest.raises(RuntimeError,match='checkpoint'):
        collector.collect_cold_candidate_reference(environment,FrozenPolicy(),value,
            output=tmp_path/'capture',steps_per_initial=4,minimum_per_style=4,schema_only=True,
            checkpoint_path=checkpoint)
    assert (tmp_path/'capture/partial_candidates.pt').is_file()
    assert not (tmp_path/'capture/cohort_report.json').exists()


def test_cold_capture_config_enforces_exact_bounded_budget(tmp_path,capsys):
    import json
    from pathlib import Path
    from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
    root=Path(__file__).resolve().parents[1]
    value=manifest(4)
    value['metadata']['camera_sha256']=StereoCalibration.from_json(root/'configs/dh116_training/stereo_camera_roll20_surface96_diagnostic_v1.json').sha256
    path=tmp_path/'manifest.json';path.write_text(json.dumps(value))
    args=['--config',str(root/'configs/dh116_training/stereo_candidate_cold8_v1.json'),
        '--candidate-only','--state-reference','--cold-trial-manifest',str(path),'--check-config',
        '--steps','4','--num-envs','2','--camera-config',str(root/'configs/dh116_training/stereo_camera_roll20_surface96_diagnostic_v1.json')]
    assert collector.main('collect',args)==0
    assert json.loads(capsys.readouterr().out)['cold_trials']==8
    args[args.index('--steps')+1]='128'
    with pytest.raises(ValueError,match='bounded'):
        collector.main('collect',args)
    args[args.index('--steps')+1]='4'
    value['metadata']['camera_sha256']='0'*64;path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='calibration'):
        collector.main('collect',args)
