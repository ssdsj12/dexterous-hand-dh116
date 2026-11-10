"""Complete RGB-only inference with synthetic images, not hardware execution."""
import copy
import json
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration, rerender_calibration
from go2_pvcnn.control.dh116.tabletop_stereo_contract import make_contract
from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_stereo_actor, normalize_network_config
from go2_pvcnn.control.dh116.tabletop_stereo_checkpoint import save_actor_package
from go2_pvcnn.control.dh116.tabletop_stereo_inference import StereoRGBStudentRunner, main
from go2_pvcnn.control.dh116.tabletop_stereo_preprocessing import processing_contract


ROOT = Path(__file__).resolve().parents[1]


def files(tmp_path, *, legacy_depth=False, automatic_depth=False):
    c = rerender_calibration(StereoCalibration.from_json(ROOT/'configs/dh116_training/stereo_camera_v1.json'), 32, 32)
    cfg = c.to_dict()
    cfg['depth_estimator'] = dict(min_disparity=1, max_disparity=8, block_size=3,
        min_depth_m=.1, max_depth_m=2., subpixel=True, image_processing_batch_size=1)
    if legacy_depth:
        cfg['depth'] = cfg.pop('depth_estimator')
    if automatic_depth:
        cfg.pop('depth_estimator')
    config = tmp_path/'camera.json'; config.write_text(json.dumps(cfg))
    network = normalize_network_config(dict(embed_dim=32, history=4, num_heads=4,
        num_layers=1, image_resolution_wh=[32,32], image_processing_batch_size=2))
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
