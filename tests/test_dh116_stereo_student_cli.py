"""Student CLI orchestration fixtures are CPU tests, never native evidence."""
from copy import deepcopy
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_dh116_stereo_estimator import history
from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116.tabletop_stereo_teacher_collector import PreActionState, StepResult

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def cli():
    name = 'go2_pvcnn.control.dh116.tabletop_stereo_student_cli'
    assert importlib.util.find_spec(name), 'Student CLI module missing'
    return importlib.import_module(name)


def inputs(tmp_path):
    config = json.loads((ROOT/'configs/dh116_training/stereo_student_ppo_v1.json').read_text())
    config['network'].update(embed_dim=32, num_layers=1)
    config['ppo'].update(epochs=1, minibatches=1)
    config['rollout_steps'] = 1
    path = tmp_path/'config.json'; path.write_text(json.dumps(config))
    camera = json.loads((ROOT/'configs/dh116_training/stereo_camera_v1.json').read_text())
    camera.update(width=32, height=32)
    camera.pop('calibration_sha256', None)
    camera['K_left'] = camera['K_right'] = [[30.,0.,16.],[0.,30.,16.],[0.,0.,1.]]
    calibration = StereoCalibration.from_dict(camera)
    camera_path = tmp_path/'camera.json'; camera_path.write_text(json.dumps(camera))
    return path, camera_path, calibration


class CPUEnvironment:
    evidence_scope = 'cpu_fixture'
    def __init__(self, calibration, *, bad_issued=False):
        self.n, self.device, self.tick = 1, torch.device('cpu'), 0
        self.calibration, self.bad_issued = calibration, bad_issued
        self.previous_actions=torch.zeros(1,24)
        self.runtime_descriptors=[]
        self.bridge = SimpleNamespace(env=SimpleNamespace(successes=torch.zeros(1), episodes=torch.zeros(1)))
        self.bridge.set_runtime_descriptor=lambda descriptor:self.runtime_descriptors.append(deepcopy(descriptor))

    def snapshot(self):
        identities = [dict(run_uuid='cpu', env_id=0, episode_uid='cpu:0:0', reset_id=0,
            policy_tick=self.tick, physics_tick_start=self.tick*8, physics_tick_end=self.tick*8,
            simulation_timestamp=self.tick/60, frame_id=self.tick,
            calibration_sha256=self.calibration.sha256)]
        # False means unavailable for Student GT supervision, not invalid native
        # numeric critic observations; the CLI must preserve this distinction.
        sensors=history(1);sensors.proprio[:,-1,38:]=self.previous_actions
        return PreActionState(sensors, torch.ones(1,371), torch.zeros(1,371,dtype=torch.bool), identities)

    def step(self, actions):
        self.tick += 1
        self.previous_actions=actions.detach().clone()
        reward = torch.ones(1)
        issued = actions + .01 if self.bad_issued else actions.clone()
        return StepResult(self.snapshot(), reward, torch.tensor([False]),
            dict(reward_terms={'native':reward.clone()}, success=torch.tensor([False]), failure=torch.zeros(1,dtype=torch.long),
                execution=[dict(physics_tick_start=(self.tick-1)*8, physics_tick_end=self.tick*8,
                    simulation_timestamp_start=(self.tick-1)/60, simulation_timestamp_end=self.tick/60)]), issued, {'joint_target':torch.zeros(1,12)})


def test_check_config_derives_camera_resolution_without_native_or_output(tmp_path, capsys):
    config, camera, _ = inputs(tmp_path); output = tmp_path/'must_not_exist'
    assert cli().main('train', ['--config',str(config),'--camera-config',str(camera),'--output',str(output),'--check-config']) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed['config']['network']['image_resolution_wh'] == [32,32]
    assert parsed['native_started'] is False and not output.exists()


def test_training_default_disabled_and_formal_gate_precedes_native_creation(tmp_path):
    config, camera, _ = inputs(tmp_path); output = tmp_path/'run'
    argv = ['--config',str(config),'--camera-config',str(camera),'--output',str(output)]
    with pytest.raises((ValueError,SystemExit)):
        cli().main('train', argv)
    with pytest.raises((ValueError,SystemExit), match='Teacher|teacher|proof|qualification'):
        cli().main('train', argv+['--enable-experiment','--execute-native'])
    assert not output.exists()


@pytest.mark.parametrize('extra', [['--num-envs','3'],['--updates','2'],['--rollout-steps','9']])
def test_diagnostic_budget_is_strict(tmp_path, extra):
    config, camera, _ = inputs(tmp_path)
    with pytest.raises((ValueError,SystemExit)):
        cli().main('train', ['--config',str(config),'--camera-config',str(camera),'--output',str(tmp_path/'run'),
            '--diagnostic-only','--enable-experiment','--execute-native',*extra])


def test_cpu_fixture_train_keeps_supervision_mask_semantics_and_is_unqualified(tmp_path, monkeypatch):
    config, camera, calibration = inputs(tmp_path); output=tmp_path/'diagnostic'
    m=cli(); environment=CPUEnvironment(calibration)
    monkeypatch.setattr(m,'_environment',lambda *args,**kwargs:environment)
    monkeypatch.setattr(m,'_launch_native',lambda callback,**kwargs:callback(object()))
    assert m.main('train',['--config',str(config),'--camera-config',str(camera),'--output',str(output),
        '--num-envs','1','--diagnostic-only','--enable-experiment','--execute-native']) == 0
    completion=json.loads((output/'completion.json').read_text())
    assert completion['evidence_scope']=='cpu_fixture' and completion['native_qualified'] is False
    assert completion['updates']==1 and len(list(output.glob('student_update_*.pt')))==2
    resolved=json.loads((output/'resolved_config.json').read_text())
    assert resolved['network']['image_resolution_wh']==[32,32]
    assert resolved['preprocessor']['processing_sha256']
    assert environment.runtime_descriptors[0]['critic_training'] is True
    assert environment.runtime_descriptors[0]['execution_mode']=='student_PPO_training'


def test_failed_issued_action_keeps_failure_record(tmp_path, monkeypatch):
    config,camera,calibration=inputs(tmp_path); output=tmp_path/'failed'
    m=cli(); monkeypatch.setattr(m,'_environment',lambda *a,**k:CPUEnvironment(calibration,bad_issued=True))
    monkeypatch.setattr(m,'_launch_native',lambda callback,**kwargs:callback(object()))
    with pytest.raises(ValueError,match='issued|sample'):
        m.main('train',['--config',str(config),'--camera-config',str(camera),'--output',str(output),
            '--num-envs','1','--diagnostic-only','--enable-experiment','--execute-native'])
    report=json.loads((output/'failure.json').read_text())
    assert report['native_qualified'] is False and report['exception_type']=='ValueError'


def test_infer_actor_and_offline_cli_use_exact_sensor_history_only(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_student_trainer,history_to_tree
    from go2_pvcnn.control.dh116.tabletop_stereo_checkpoint import save_student_checkpoint,export_actor
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import make_contract
    config=dict(network=dict(embed_dim=32,history=4,num_heads=4,num_layers=1,dropout=0.),ppo={},rollout_steps=1)
    engine=make_student_trainer(config,1)
    contract=make_contract('student',config,calibration_sha256='a'*64)
    checkpoint=tmp_path/'student.pt'; package=tmp_path/'actor.pt'
    save_student_checkpoint(checkpoint,engine,contract,network_config=config['network'])
    export_actor(checkpoint,package)
    h=history(1); expected=cli().infer_actor(package,h,calibration_sha256='a'*64)
    observation=tmp_path/'observation.pt';torch.save(history_to_tree(h),observation)
    output=tmp_path/'actions.json'
    assert cli().main('run',['--actor-package',str(package),'--observation',str(observation),'--output',str(output)])==0
    actual=torch.tensor(json.loads(output.read_text())['actions24'])
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    with pytest.raises(FileExistsError):
        cli().main('run',['--actor-package',str(package),'--observation',str(observation),'--output',str(output)])
    poisoned=history_to_tree(h);poisoned['critic_obs']=torch.zeros(1,371)
    with pytest.raises(ValueError,match='unknown|privileged|sensor'):
        cli().infer_actor(package,poisoned)
    fresh=tmp_path/'fresh_actions.json'
    code=("import sys;from go2_pvcnn.control.dh116.tabletop_stereo_student_cli import main;"
        f"main('run',['--actor-package',{str(package)!r},'--observation',{str(observation)!r},'--output',{str(fresh)!r}]);"
        "assert not any('privileged_teacher' in n or 'privileged_ppo' in n for n in sys.modules)")
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_terminal_readback_na_logs_null_with_original_mask():
    value=cli()._json({'target':torch.tensor([[float('nan')]]),'valid':torch.tensor([False])})
    assert value=={'target':[[None]],'valid':[False]}


def test_accuracy640_default_check_config_preserves_resolution_and_chunking(capsys):
    assert cli().main('train',['--check-config'])==0
    value=json.loads(capsys.readouterr().out)
    assert value['config']['network']['image_resolution_wh']==[640,480]
    assert value['config']['network']['image_processing_batch_size']==2
    assert value['native_started'] is False


def test_actor_warmstart_allows_resource_change_and_projects_noise_before_rollout(tmp_path):
    import math
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_stereo_actor
    from go2_pvcnn.control.dh116.tabletop_stereo_checkpoint import save_actor_package
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import make_contract
    config_path,camera,calibration=inputs(tmp_path)
    target,_=cli().resolve_config(config_path,camera)
    source_network=deepcopy(target['network']);source_network['image_processing_batch_size']=1
    source_network['backbone']={'type':'GN_ResNet18','normalization':'GroupNorm'}
    actor=make_stereo_actor(source_network)
    with torch.no_grad():actor.log_std.fill_(math.log(.5))
    package=tmp_path/'bc_actor.pt'
    save_actor_package(package,actor,make_contract('bc',{'network':source_network},
        calibration_sha256=calibration.sha256),network_config=source_network)
    output=tmp_path/'warm';output.mkdir()
    result=cli().train_student(CPUEnvironment(calibration),target,output=output,updates=1,
        calibration_sha256=calibration.sha256,actor_package=package,diagnostic_only=True)
    assert result['native_qualified'] is False
    first=json.loads((output/'control_transitions.jsonl').read_text().splitlines()[0])
    torch.testing.assert_close(torch.tensor(first['old_latent_std']),torch.full((1,24),.3))
    initialization=json.loads((output/'initialization.json').read_text())
    assert 'fresh_critic' in initialization['initialization'] and 'not_exact_resume' in initialization['initialization']
    assert initialization['processing_binding']['scope']=='diagnostic_rebind_unspecified_history_input'
    payload=torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
    provenance=payload['actor_contract']['training_provenance']
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
    assert provenance['input_actor_package_artifact']['sha256']==file_sha256(package)
    assert provenance['source_inventory_complete'] is False


def test_formal_warmstart_rejects_unbound_history_actor_package():
    from go2_pvcnn.control.dh116.tabletop_stereo_preprocessing import processing_contract
    from test_dh116_stereo_preprocessing import camera
    actor=SimpleNamespace(deployment_contract={'preprocessor':{'binding':'history_input_only_unspecified','processing':None}})
    with pytest.raises(ValueError,match='processing|history|binding'):
        cli()._warmstart_processing(actor,processing_contract(camera()),diagnostic_only=False)


def test_frozen_native_driver_binds_actual_package_processing_and_no_training(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_preprocessing import processing_contract
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_stereo_actor
    from go2_pvcnn.control.dh116.tabletop_stereo_checkpoint import save_actor_package
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import make_contract,file_sha256
    config_path,camera,calibration=inputs(tmp_path)
    config,_=cli().resolve_config(config_path,camera)
    package=tmp_path/'actor.pt'
    actor=make_stereo_actor(config['network'])
    save_actor_package(package,actor,make_contract('bc',config,calibration_sha256=calibration.sha256),network_config=config['network'])
    environment=CPUEnvironment(calibration);environment.camera_config=camera
    output=tmp_path/'eval';output.mkdir()
    result=cli().run_student(environment,package,output=output,steps=1,calibration_sha256=calibration.sha256)
    descriptor=environment.runtime_descriptors[0]
    assert descriptor['critic_training'] is False and descriptor['critic_deployed'] is False
    assert descriptor['teacher_execution_enabled'] is False and descriptor['mpc_enabled'] is False
    assert descriptor['privileged_actor_inputs'] is False
    assert descriptor['model_checkpoint_sha256']==descriptor['actor_package_sha256']==file_sha256(package)
    assert descriptor['processing_sha256']==processing_contract(camera)['processing_sha256']
    assert result['checkpoint_sha256']==file_sha256(package) and result['native_qualified'] is False


def test_formal_warmstart_rejects_changed_matcher_with_same_calibration():
    from go2_pvcnn.control.dh116.tabletop_stereo_preprocessing import processing_contract
    from test_dh116_stereo_preprocessing import camera
    one=camera();other=deepcopy(one);other['depth_estimator']['min_disparity']=2
    source,target=processing_contract(one),processing_contract(other)
    actor=SimpleNamespace(deployment_contract={'preprocessor':{'binding':'bound_raw_stereo','processing':source}})
    with pytest.raises(ValueError,match='processing'):
        cli()._warmstart_processing(actor,target,diagnostic_only=False)
    result=cli()._warmstart_processing(actor,target,diagnostic_only=True)
    assert result['scope']=='diagnostic_rebind_changed_processing' and result['native_qualified'] is False
