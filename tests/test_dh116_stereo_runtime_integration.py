"""Runtime boundary regressions; fixtures are not native grasp evidence."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(4)

def fixture(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tests'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_rejected_timestamp_clears_inference_history(tmp_path):
    f=fixture('test_dh116_stereo_inference')
    from go2_pvcnn.control.dh116.tabletop_stereo_inference import StereoRGBStudentRunner
    package,camera=f.files(tmp_path); runner=StereoRGBStudentRunner(package,camera)
    runner.from_mapping(f.raw())
    assert runner.history._valid.any()
    with pytest.raises(ValueError,match='timestamp'):
        runner.from_mapping(dict(f.raw(),frame_id=2))
    assert not runner.history._valid.any()

def test_gap_clears_history_and_exposes_wire_tracking_metadata(tmp_path):
    f=fixture('test_dh116_stereo_inference')
    from go2_pvcnn.control.dh116.tabletop_stereo_inference import StereoRGBStudentRunner
    package,camera=f.files(tmp_path);runner=StereoRGBStudentRunner(package,camera)
    runner.from_mapping(f.raw())
    runner.from_mapping(dict(f.raw(),frame_id=2,left_timestamp_s=1.,right_timestamp_s=1.,now_timestamp_s=1.))
    assert runner.history._valid.sum()==1
    assert 'wire_tracking' in runner.last_visible_geometry
    assert 'track_id' in runner.last_visible_geometry['wire_tracking']

def test_actor_package_rejects_same_width_different_field_semantics(tmp_path):
    f=fixture('test_dh116_stereo_runtime_actor_modes')
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_stereo_actor,normalize_network_config
    from go2_pvcnn.control.dh116.tabletop_stereo_checkpoint import save_actor_package
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import make_contract
    cfg=normalize_network_config(dict(embed_dim=32,actor_input_mode='estimated_task_state',task_state_config=f.task_config()))
    actor=make_stereo_actor(cfg)
    changed=copy.deepcopy(cfg);changed['task_state_config']['allowlist']=[117]
    contract=make_contract('bc',dict(network=changed,schema_only=True),calibration_sha256='a'*64)
    with pytest.raises(ValueError,match='semantics|configuration'):
        save_actor_package(tmp_path/'invalid.pt',actor,contract,network_config=changed)
    assert not (tmp_path/'invalid.pt').exists()

@pytest.mark.parametrize('mode',['latent','estimated_task_state','hybrid'])
def test_three_independent_mode_configs_roundtrip_same_fixed_input(tmp_path,mode):
    f=fixture('test_dh116_stereo_runtime_actor_modes')
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_stereo_actor,normalize_network_config
    from go2_pvcnn.control.dh116.tabletop_stereo_checkpoint import save_actor_package,load_deployment_actor
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import make_contract
    path=ROOT/'configs/dh116_training'/f'stereo_student_{mode}_palms640_v3.json'
    cfg=json.loads(path.read_text())
    assert cfg['network']['actor_input_mode']==mode
    assert cfg['network']['image_resolution_wh']==[640,480]
    network=dict(cfg['network'],embed_dim=32,image_resolution_wh=[32,32])
    network=normalize_network_config(network)
    torch.manual_seed(116); actor=make_stereo_actor(network).eval()
    h=f.history();result=actor.deterministic(h)
    contract=make_contract('bc',dict(network=network,schema_only=True),calibration_sha256='a'*64)
    package=tmp_path/f'{mode}.pt';save_actor_package(package,actor,contract,network_config=network)
    actual=load_deployment_actor(package,expected_calibration_sha256='a'*64).deterministic(h)
    torch.testing.assert_close(actual,result,atol=0,rtol=0)
    assert actual.shape==(1,24)

def test_teacher_grouped_noise_reward_and_checkpoint_readback(tmp_path):
    f=fixture('test_dh116_privileged_cli')
    cfg,_=f.cli.load_config(f.CONFIG);cfg['rollout_steps']=2
    from go2_pvcnn.control.dh116.tabletop_stereo_native import native_task_identity
    bridge=f.CPUFixtureBridge();bridge.task_identity=native_task_identity(json.loads((ROOT/'configs/dh116_training/stereo_native_task_v1.json').read_text()))
    f.cli.train_teacher(bridge,cfg,f.cli.PPOConfig(epochs=1,minibatches=2),output=tmp_path,updates=1)
    row=json.loads((tmp_path/'iterations.jsonl').read_text())
    assert len(row['action_diagnostics']['sampled_action_mean24'])==24
    assert row['action_diagnostics']['execution_mode']=='sampled_on_policy'
    assert row['action_diagnostics']['issued_equals_sampled'] is True
    assert set(row['action_diagnostics']['groups'])=={'left_fingers','right_fingers','left_translation','right_translation','left_rotation','right_rotation'}
    assert row['reward_diagnostics']['r0_closure_max_abs']==0
    assert row['checkpoint_roundtrip']['policy_equal'] is True
    assert row['checkpoint_roundtrip']['optimizer_equal'] is True
    groups=row['action_diagnostics']['groups']
    assert all('sampled_tanh_entropy' in g and 'final_old_to_new_kl' in g for g in groups.values())
    assert sum(g['final_old_to_new_kl'] for g in groups.values())==pytest.approx(row['action_diagnostics']['final_old_to_new_kl'],abs=2e-5)

def test_optional_deformation_without_units_or_frame_is_not_valid_state():
    f=fixture('test_dh116_stereo_runtime_actor_modes')
    from go2_pvcnn.control.dh116.tabletop_stereo_task_state_adapter import TaskStateAdapter
    from go2_pvcnn.control.dh116.tabletop_stereo_state_estimator import StereoStateEstimator
    h=f.history();estimates=StereoStateEstimator(embed_dim=32)(h)
    config=f.task_config();config.update(extra_heads=['deformation'])
    result=TaskStateAdapter(**config)(estimates,h)
    assert not result['valid'][:,-128:].any()

def test_velocity_estimate_requires_two_valid_history_frames():
    f=fixture('test_dh116_stereo_runtime_actor_modes')
    from go2_pvcnn.control.dh116.tabletop_stereo_task_state_adapter import TaskStateAdapter
    from go2_pvcnn.control.dh116.tabletop_stereo_state_estimator import StereoStateEstimator
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import state_contract
    h=f.history();h.history_valid[:,:-1]=False
    rows=state_contract()['dimensions'];qual={key:True for r in rows for key in r['required_validity_keys'] if key!='supervision_forbidden'}
    result=TaskStateAdapter(allowlist=['object_root_velocity'],source_qualification=qual)(StereoStateEstimator(embed_dim=32)(h),h)
    assert not result['valid'].any()

def test_actor_only_package_cannot_reinterpret_changed371_schema(tmp_path,monkeypatch):
    f=fixture('test_dh116_stereo_inference');package,_=f.files(tmp_path)
    from go2_pvcnn.control.dh116 import tabletop_stereo_contract as contract
    from go2_pvcnn.control.dh116.tabletop_stereo_checkpoint import load_deployment_actor
    changed=copy.deepcopy(contract.state_contract());changed['dimensions'][116]['semantic_name']='different_field'
    monkeypatch.setattr(contract,'state_contract',lambda:changed)
    with pytest.raises(ValueError,match='schema'):
        load_deployment_actor(package)

def test_palm_link_is_not_a_verified_wrist_root_pose():
    f=fixture('test_dh116_stereo_runtime_actor_modes')
    from go2_pvcnn.control.dh116.tabletop_stereo_task_state_adapter import TaskStateAdapter
    from go2_pvcnn.control.dh116.tabletop_stereo_state_estimator import StereoStateEstimator
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import state_contract
    qual={k:True for r in state_contract()['dimensions'] for k in r['required_validity_keys'] if k!='supervision_forbidden'}
    h=f.history()
    result=TaskStateAdapter(allowlist=['wrist_pose_from_reset'],known_config={'wrist_reset_xyz':[[0,0,0],[0,0,0]]},
        source_qualification=qual)(StereoStateEstimator(embed_dim=32)(h),h)
    assert not result['valid'].any()

def test_completed_interval_contact_requires_completed_history():
    f=fixture('test_dh116_stereo_runtime_actor_modes')
    from go2_pvcnn.control.dh116.tabletop_stereo_task_state_adapter import TaskStateAdapter
    from go2_pvcnn.control.dh116.tabletop_stereo_state_estimator import StereoStateEstimator
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import state_contract
    qual={k:True for r in state_contract()['dimensions'] for k in r['required_validity_keys'] if k!='supervision_forbidden'}
    h=f.history();h.history_valid[:,:-1]=False
    result=TaskStateAdapter(allowlist=['interval_paired'],source_qualification=qual)(StereoStateEstimator(embed_dim=32)(h),h)
    assert not result['valid'].any()
