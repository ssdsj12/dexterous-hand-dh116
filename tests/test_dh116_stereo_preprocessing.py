"""Preprocessing identity is software provenance, never native qualification."""
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration,rerender_calibration
from go2_pvcnn.control.dh116.tabletop_stereo_contract import make_contract

ROOT=Path(__file__).resolve().parents[1]


def module():
    name='go2_pvcnn.control.dh116.tabletop_stereo_preprocessing'
    assert importlib.util.find_spec(name),'preprocessing contract module missing'
    return importlib.import_module(name)


def camera():
    original=StereoCalibration.from_json(ROOT/'configs/dh116_training/stereo_camera_v1.json')
    value=rerender_calibration(original,32,32).to_dict()
    value['depth_estimator']=dict(min_disparity=1,max_disparity=8,block_size=3,
        min_depth_m=.1,max_depth_m=2.,subpixel=True,image_processing_batch_size=1)
    return value


def test_processing_semantics_bind_matcher_but_not_resource_chunk(tmp_path):
    m=module();base=camera()
    one=m.processing_contract(base)
    other=deepcopy(base);other['depth_estimator']['image_processing_batch_size']=2
    assert one==m.processing_contract(other)
    assert 'image_processing_batch_size' not in one['depth_estimator']
    assert one['perception']['point_count']==128
    assert {'tabletop_stereo_depth.py','tabletop_stereo_perception.py','tabletop_stereo_geometry.py',
        'tabletop_stereo_preprocessing.py'}<=one['source_sha256'].keys()
    changed=deepcopy(base);changed['depth_estimator']['subpixel']=False
    assert m.processing_contract(changed)['processing_sha256']!=one['processing_sha256']
    path=tmp_path/'camera.json';path.write_text(json.dumps(base))
    assert m.processing_contract(path)==one
    tampered=deepcopy(one);tampered['depth_estimator']['max_disparity']=9
    with pytest.raises(ValueError,match='SHA|contract'):
        m.validate_processing_contract(tampered)


def test_raw_pipeline_requires_binding_and_rejects_changed_matcher_same_calibration(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_stereo_actor
    from go2_pvcnn.control.dh116.tabletop_stereo_checkpoint import save_actor_package,load_deployment_actor
    m=module();cam=camera();pc=m.processing_contract(cam)
    network=dict(embed_dim=32,history=4,num_heads=4,num_layers=1,dropout=0.,image_resolution_wh=[32,32])
    actor=make_stereo_actor(network)
    calibration=StereoCalibration.from_dict(cam).sha256
    unbound=tmp_path/'history.pt'
    save_actor_package(unbound,actor,make_contract('bc',{'network':network},calibration_sha256=calibration),network_config=network)
    old=load_deployment_actor(unbound)
    assert old.deployment_contract['preprocessor']['binding']=='history_input_only_unspecified'
    with pytest.raises(ValueError,match='history|unspecified|binding'):
        m.require_raw_processing(old.deployment_contract,cam)
    bound=tmp_path/'bound.pt'
    save_actor_package(bound,actor,make_contract('bc',{'network':network,'preprocessor':pc},
        calibration_sha256=calibration),network_config=network)
    actual=load_deployment_actor(bound)
    assert m.require_raw_processing(actual.deployment_contract,cam)==pc
    other=deepcopy(cam);other['depth_estimator']['min_disparity']=2
    assert StereoCalibration.from_dict(other).sha256==calibration
    with pytest.raises(ValueError,match='processing|preprocessor|contract'):
        m.require_raw_processing(actual.deployment_contract,other)
