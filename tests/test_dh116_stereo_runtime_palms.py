"""Actual palm coordinates enter the optional local policy, no privileged input."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

torch.set_num_threads(4)


def fixture():
    name='_runtime_estimator_fixture'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name('test_dh116_stereo_estimator.py'))
        m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return sys.modules[name]


def test_sensor_exposes_actual_both_palm_points_and_confidence():
    h=fixture().sensor(1)
    h.proprio[:,24:27]=torch.tensor([.2,.1,.3])
    # Reconstruct from legacy input fields so local points must be derived.
    values=asdict(h)
    for key in ('points_left_palm','points_right_palm','point_valid_left','point_valid_right','point_confidence'):
        values.pop(key,None)
    h=type(h)(**values)
    assert hasattr(h,'points_left_palm'),'palm-local points must be part of Actor sensor input'
    torch.testing.assert_close(h.points_left_palm,h.points_world-torch.tensor([.2,.1,.3]))
    torch.testing.assert_close(h.points_right_palm,h.points_world)
    assert torch.equal(h.point_valid_left,h.point_valid)
    assert h.point_confidence.shape==(1,128)


def test_local_estimator_is_invariant_to_global_translation():
    from go2_pvcnn.control.dh116.tabletop_stereo_observation import SensorBatch,StereoHistory
    from go2_pvcnn.control.dh116.tabletop_stereo_state_estimator import StereoStateEstimator
    sensor=fixture().sensor(1)
    base=asdict(sensor);moved={k:v.clone() for k,v in base.items()}
    shift=torch.tensor([.7,-.4,.3])
    moved['points_world']+=shift
    moved['proprio'][:,24:27]+=shift;moved['proprio'][:,31:34]+=shift
    for tree in (base,moved):
        for key in ('points_left_palm','points_right_palm','point_valid_left','point_valid_right','point_confidence'):
            tree.pop(key,None)
    a=StereoHistory(1).update(SensorBatch(**base))
    b=StereoHistory(1).update(SensorBatch(**moved))
    model=StereoStateEstimator(embed_dim=32,point_coordinate_mode='palms').eval()
    with torch.no_grad(): x,y=model(a),model(b)
    torch.testing.assert_close(a.points_left_palm,b.points_left_palm,atol=1e-6,rtol=0)
    torch.testing.assert_close(x['latent'],y['latent'],atol=2e-5,rtol=0)
    torch.testing.assert_close(x['privileged_state'],y['privileged_state'],atol=2e-5,rtol=0)


def test_shared_point_weights_separate_hand_identity_and_validity():
    from go2_pvcnn.control.dh116 import tabletop_stereo_encoder as module
    assert hasattr(module,'PalmPointEncoder'),'shared palm point encoder required'
    encoder=module.PalmPointEncoder(32)
    p=torch.zeros(1,128,3);v=torch.ones(1,128,dtype=torch.bool);confidence=torch.ones(1,128)
    result=encoder(p,p,v,v,confidence)
    assert result[0].shape==(1,256,32) and result[1].shape==(1,256)
    assert not torch.equal(result[0][:,:128],result[0][:,128:])
    missing=encoder(torch.full_like(p,float('nan')),p,~v,v,confidence)
    assert not missing[1][:,:128].any()
    assert torch.isfinite(missing[0]).all()


def test_palm_history_clone_index_and_masked_na():
    h=fixture().history(2)
    assert hasattr(h,'point_valid_left')
    copied=h.clone().index_select(torch.tensor([1]))
    assert copied.points_left_palm.shape==(1,4,128,3)
    assert copied.points_left_palm.data_ptr()!=h.points_left_palm.data_ptr()
