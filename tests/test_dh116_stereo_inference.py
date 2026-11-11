"""Complete RGB-only inference with synthetic images, not hardware execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration, rerender_calibration
from go2_pvcnn.control.dh116.tabletop_stereo_contract import make_contract
from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_stereo_actor, normalize_network_config
from go2_pvcnn.control.dh116.tabletop_stereo_checkpoint import save_actor_package
from go2_pvcnn.control.dh116.tabletop_stereo_inference import StereoRGBStudentRunner, main
from go2_pvcnn.control.dh116.tabletop_stereo_preprocessing import processing_contract


ROOT = Path(__file__).resolve().parents[1]


def files(tmp_path, *, legacy_depth=False, automatic_depth=False, mode='latent', segmentation=False):
    c = rerender_calibration(StereoCalibration.from_json(ROOT/'configs/dh116_training/stereo_camera_v1.json'), 32, 32)
    cfg = c.to_dict()
    cfg['depth_estimator'] = dict(min_disparity=1, max_disparity=8, block_size=3,
        min_depth_m=.1, max_depth_m=2., subpixel=True, image_processing_batch_size=1)
    if legacy_depth:
        cfg['depth'] = cfg.pop('depth_estimator')
    if automatic_depth:
        cfg.pop('depth_estimator')
    if segmentation:
        from go2_pvcnn.control.dh116.tabletop_stereo_wire_segmentation import StereoWireSegmentation
        weights = tmp_path/'segmentation.pt'
        torch.save(dict(model=StereoWireSegmentation(8).state_dict(), width=8, optimizer={},
            contract=dict(version='dh116-wire-segmentation-diagnostic-v1', source='native_instance_id')), weights)
        cfg['wire_segmentation'] = dict(diagnostic_only=True, checkpoint_path=str(weights),
            checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
        cfg['depth_estimator'].update(method='wire_silhouette', subpixel=False)
    config = tmp_path/'camera.json'; config.write_text(json.dumps(cfg))
    network = normalize_network_config(dict(embed_dim=32, history=4, num_heads=4,
        num_layers=1, image_resolution_wh=[32,32], image_processing_batch_size=2,
        actor_input_mode=mode, point_coordinate_mode='palms' if mode != 'latent' else 'world'))
    actor = make_stereo_actor(network)
    contract = make_contract('bc', dict(network=network, schema_only=True,
        preprocessor=processing_contract(config)), calibration_sha256=c.sha256)
    package = tmp_path/'actor.pt'
    save_actor_package(package, actor, contract, network_config=network)
    return package, config


@pytest.mark.parametrize('kwargs', [dict(legacy_depth=True), dict(automatic_depth=True)])
def test_raw_runner_uses_the_exact_bound_resolved_depth_pipeline(tmp_path, kwargs):
    package, config = files(tmp_path, **kwargs)
    runner = StereoRGBStudentRunner(package, config)
    actual = runner.depth.to_config(); actual.pop('image_processing_batch_size', None)
    assert actual == runner.processing['depth_estimator']


def raw():
    torch.manual_seed(41)
    left = torch.rand(1,3,32,32)
    right = torch.zeros_like(left); right[:,:,:,:-4] = left[:,:,:,4:]
    proprio = torch.zeros(1,62); proprio[:,27] = 1; proprio[:,34] = 1
    return dict(left_rgb=left,right_rgb=right,proprio=proprio,left_timestamp_s=0.,
        right_timestamp_s=0.,frame_id=1,now_timestamp_s=0.)


def test_raw_rgb_only_runner_computes_depth_and_bounds_action(tmp_path):
    package, config = files(tmp_path)
    runner = StereoRGBStudentRunner(package, config)
    result = runner.from_mapping(raw())
    assert result['actions24'].shape == (1,24) and (result['actions24'].abs()<=1).all()
    assert result['depth_valid_fraction'][0] > 0
    assert not result['teacher_executed'] and not result['critic_executed']
    assert not result['simulator_input_used'] and not result['native_task_qualified']
    assert result['processing_sha256']
    assert int(runner.history._valid.sum()) == 1
    next_raw = raw(); next_raw.update(frame_id=2,left_timestamp_s=1/60,right_timestamp_s=1/60,now_timestamp_s=1/60)
    runner.from_mapping(next_raw,reset_mask=torch.tensor([True]))
    assert int(runner.history._valid.sum()) == 1


def test_visible_wire_points_use_current_measured_fk_for_both_palms(tmp_path):
    package, config = files(tmp_path)
    runner = StereoRGBStudentRunner(package, config)
    points = torch.full((1,128,3), float('nan')); points[0,0] = torch.tensor([1.,3.,3.])
    valid = torch.zeros(1,128,dtype=torch.bool); valid[0,0] = True
    runner.extractor.extract = lambda *a,**k: dict(points_world=points.clone(),point_valid=valid.clone())
    observation = raw()
    observation['proprio'][0,24:31] = torch.tensor([1.,2.,3.,2**-.5,0.,0.,2**-.5])
    observation['proprio'][0,31:38] = torch.tensor([-1.,0.,1.,1.,0.,0.,0.])
    runner.from_mapping(observation)
    geometry = runner.last_visible_geometry
    torch.testing.assert_close(geometry['points_palms_m'][0,0,0],torch.tensor([1.,0.,0.]),atol=1e-6,rtol=0.)
    torch.testing.assert_close(geometry['points_palms_m'][0,1,0],torch.tensor([2.,3.,2.]),atol=1e-6,rtol=0.)
    assert not torch.isfinite(geometry['points_palms_m'][:,:,1:]).any()
    assert geometry['point_valid'].sum() == 1 and geometry['frame_id'] == 1
    assert geometry['source'] == 'RGB_stereo_and_current_measured_palm_FK'


@pytest.mark.parametrize('change', [dict(right_timestamp_s=.001), dict(now_timestamp_s=.1)])
def test_unsynchronized_and_stale_images_cannot_use_truth_fallback(tmp_path, change):
    package, config = files(tmp_path); runner = StereoRGBStudentRunner(package, config)
    with pytest.raises(ValueError): runner.from_mapping(dict(raw(),**change))
    assert not runner.history._valid.any()


def test_prohibited_privileged_or_rendered_depth_input_rejected(tmp_path):
    package, config = files(tmp_path); runner = StereoRGBStudentRunner(package, config)
    for key in ('privileged','ground_truth_depth','teacher_action','phase'):
        with pytest.raises(ValueError, match='privileged'):
            runner.from_mapping(dict(raw(),**{key:torch.zeros(1)}))


def test_inference_cli_uses_raw_rgb_pair_and_exclusive_output(tmp_path):
    package, config = files(tmp_path)
    frame = tmp_path/'rgb.pt'; torch.save(raw(),frame)
    output = tmp_path/'actions.json'
    argv = ['--actor-package',str(package),'--camera-config',str(config),
        '--actor-package-sha256',hashlib.sha256(package.read_bytes()).hexdigest(),
        '--rgb-frame',str(frame),'--output',str(output),'--enable-experiment']
    assert main(argv) == 0
    assert len(json.loads(output.read_text())['actions24'][0]) == 24
    with pytest.raises(FileExistsError): main(argv)


def test_raw_uint8_RGB_is_explicitly_normalized_without_Bgr_guessing(tmp_path):
    package, config = files(tmp_path)
    runner = StereoRGBStudentRunner(package, config)
    frame = raw()
    for key in ('left_rgb','right_rgb'):
        frame[key] = (frame[key]*255).round().to(torch.uint8)
    result = runner.from_mapping(frame)
    assert result['actions24'].shape == (1,24)
    assert runner.history._values['left_rgb'].dtype == torch.float32
    assert runner.history._values['left_rgb'].max() <= 1


def test_same_calibration_different_disparity_processing_rejected(tmp_path):
    package, config = files(tmp_path)
    changed = json.loads(config.read_text())
    changed['depth_estimator']['subpixel'] = False
    config.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='processing|preprocess'):
        StereoRGBStudentRunner(package,config)


@pytest.mark.parametrize('segmentation', [False, True])
def test_standalone_rgb_loader_restricts_both_package_deserializations(tmp_path, monkeypatch, segmentation):
    from io import BytesIO

    package, config = files(tmp_path, segmentation=segmentation)
    load = torch.load
    loaded = []

    def checked(source, *args, **kwargs):
        assert kwargs.get('weights_only') is True
        assert isinstance(source, BytesIO)
        loaded.append(hashlib.sha256(source.getvalue()).hexdigest())
        return load(source, *args, **kwargs)

    monkeypatch.setattr(torch, 'load', checked)
    runner = StereoRGBStudentRunner(package, config)
    assert runner.from_mapping(raw())['actions24'].shape == (1,24)
    assert len(loaded) == 1 + segmentation
    assert loaded[0] == hashlib.sha256(package.read_bytes()).hexdigest()


def test_runner_identity_reports_the_actor_bytes_it_actually_loaded(tmp_path, monkeypatch):
    package, config = files(tmp_path)
    expected = hashlib.sha256(package.read_bytes()).hexdigest()
    load = torch.load

    def replace_after_read(source, *args, **kwargs):
        payload = load(source, *args, **kwargs)
        package.write_bytes(b'replaced after reading')
        return payload

    monkeypatch.setattr(torch, 'load', replace_after_read)
    runner = StereoRGBStudentRunner(package, config)
    assert runner.from_mapping(raw())['actor_package_sha256'] == expected
    assert hashlib.sha256(package.read_bytes()).hexdigest() != expected


def test_runner_camera_calibration_and_processing_share_the_verified_snapshot(tmp_path, monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_stereo_inference as inference

    package, config = files(tmp_path)
    expected = hashlib.sha256(config.read_bytes()).hexdigest()
    require = inference.require_raw_processing

    def replace_after_validation(contract, snapshot):
        result = require(contract, snapshot)
        mutated = json.loads(config.read_text())
        mutated['depth_estimator']['subpixel'] = False
        config.write_text(json.dumps(mutated))
        return result

    monkeypatch.setattr(inference, 'require_raw_processing', replace_after_validation)
    runner = StereoRGBStudentRunner(package, config)
    assert runner.depth.to_config()['subpixel'] is True
    assert runner.from_mapping(raw())['camera_config_sha256'] == expected


@pytest.mark.parametrize('mode', ['latent', 'estimated_task_state', 'hybrid'])
def test_all_three_modes_run_in_fresh_process_without_teacher_mpc_or_critic(tmp_path, mode):
    from go2_pvcnn.control.dh116.tabletop_stereo_observation import StereoHistoryBatch

    package, config = files(tmp_path, mode=mode)
    frame = tmp_path/'raw.pt'
    torch.save(raw(), frame)
    output = tmp_path/'isolated.json'
    guard = '''
import importlib.abc
import runpy
import sys

def forbidden(name):
    root = name.partition('.')[0]
    leaf = name.rsplit('.', 1)[-1]
    return (root in {'omni', 'isaacsim', 'isaaclab', 'pxr'} or (
        name.startswith('go2_pvcnn.') and (
            'teacher' in leaf or 'mpc' in leaf or 'privileged' in leaf
            or leaf in {'tabletop_stereo_native', 'tabletop_stereo_runtime'})))

class DenyRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if forbidden(fullname):
            raise AssertionError('forbidden runtime import: ' + fullname)

assert not any(forbidden(name) for name in sys.modules)
sys.meta_path.insert(0, DenyRuntime())
sys.argv = sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit:
    assert not any(forbidden(name) for name in sys.modules)
    raise
'''
    result = subprocess.run([sys.executable, '-c', guard, str(ROOT/'scripts/infer_dh116_stereo_rgb.py'),
        '--actor-package', str(package), '--camera-config', str(config), '--rgb-frame', str(frame),
        '--output', str(output), '--enable-experiment'], cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    deployment = json.loads(output.read_text())
    runner = StereoRGBStudentRunner(package, config)
    local = runner.from_mapping(raw())
    saved = torch.load(package, weights_only=True)
    source_actor = make_stereo_actor(saved['network_config']).eval()
    source_actor.load_state_dict(saved['actor'], strict=True)
    history = StereoHistoryBatch(**runner.history._values, history_valid=runner.history._valid)
    with torch.no_grad():
        expected = source_actor.deterministic(history)
    torch.testing.assert_close(local['actions24'], expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.tensor(deployment['actions24']), expected, atol=1e-7, rtol=0)
    assert deployment['action_owner'] == 'student'
    assert all(deployment[key] is False for key in ('teacher_executed', 'critic_executed',
                                                   'simulator_input_used', 'native_task_qualified'))
    assert deployment['mpc_executed'] is False


def test_actor_pin_mismatch_fails_before_deserialization(tmp_path, monkeypatch):
    package, config = files(tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail('SHA mismatch reached deserialization')

    monkeypatch.setattr(torch, 'load', unexpected)
    with pytest.raises(ValueError, match='Actor package SHA mismatch'):
        StereoRGBStudentRunner(package, config, expected_actor_package_sha256='1'*64)


def test_standalone_actor_package_rejects_critic_tensor(tmp_path):
    package, config = files(tmp_path)
    data = torch.load(package, weights_only=True)
    data['critic'] = {'weight': torch.ones(1)}
    mixed = tmp_path/'mixed.pt'
    torch.save(data, mixed)
    with pytest.raises(ValueError, match='actor-only package'):
        StereoRGBStudentRunner(mixed, config)


def test_stale_pair_cannot_bypass_freshness_with_nan_limit(tmp_path):
    package, config = files(tmp_path)
    with pytest.raises(ValueError, match='freshness limit'):
        StereoRGBStudentRunner(package, config, max_image_age_s=float('nan'))
