"""J2-B wrapper tests; no native accuracy or task-success qualification."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116.tabletop_stereo_depth import StereoDepth

METHODS = ('census', 'zncc', 'gradient', 'coarse_to_fine')


def pair(batch=2):
    gen = torch.Generator().manual_seed(47)
    left = torch.rand(batch, 3, 32, 96, generator=gen)
    right = torch.zeros_like(left)
    right[..., :-8] = left[..., 8:]
    k = [[80., 0., 48.], [0., 80., 16.], [0., 0., 1.]]
    a = torch.eye(4); b = a.clone(); b[0, 3] = .08
    return left, right, StereoCalibration(k, k, a, b, 96, 32, .08)


@pytest.mark.parametrize('method', METHODS)
def test_explicit_candidate_roundtrip_preserves_existing_config(method):
    candidate = StereoDepth(method=method, min_disparity=2, max_disparity=12,
        block_size=5, subpixel=True)
    assert StereoDepth.from_config(candidate.to_config()).to_config() == candidate.to_config()
    assert StereoDepth().to_config()['method'] == 'blockmatch'
    assert len(StereoDepth().to_config()) == 12


@pytest.mark.parametrize('method', METHODS)
def test_candidate_chunking_units_masks_and_diagnostics(method):
    left, right, calibration = pair()
    common = dict(method=method, min_disparity=2, max_disparity=12,
        block_size=5, subpixel=True, min_depth_m=.2, max_depth_m=2.,
        texture_threshold=.001, uniqueness_threshold=.01)
    full_matcher = StereoDepth(**common, image_processing_batch_size=None)
    full = full_matcher.compute(left, right, calibration)
    chunk = StereoDepth(**common, image_processing_batch_size=1).compute(left, right, calibration)
    assert full['valid'].any(), 'observable integer fixture cannot pass as all-invalid'
    for key in ('depth_m', 'disparity_px', 'valid', 'confidence', 'occlusion'):
        torch.testing.assert_close(full[key], chunk[key], rtol=0., atol=0., equal_nan=True)
    assert full['valid_mask'] is full['valid']
    assert torch.isnan(full['depth_m'][~full['valid']]).all()
    assert torch.isnan(full['disparity_px'][~full['valid']]).all()
    assert (full['disparity_px'][full['valid']] > 0).all()
    diagnostics = full['diagnostics']
    assert diagnostics['method'] == method and diagnostics['candidate_enabled'] is True
    assert diagnostics['confidence_is_probability'] is False
    assert diagnostics['true_occlusion_available'] is False
    assert diagnostics['reasons']['true_occlusion'] is None
    assert full_matcher.last_resource_metadata['cost_volume_allocated'] is False
    assert full_matcher.last_resource_metadata['output_resolution_wh'] == [96, 32]


def test_new_implementation_files_are_in_processing_and_producer_sha():
    from go2_pvcnn.control.dh116.tabletop_stereo_preprocessing import processing_contract
    from go2_pvcnn.control.dh116.tabletop_stereo_error_cli import implementation_identity
    _, _, calibration = pair(1)
    config = calibration.to_dict(); config['depth_estimator'] = dict(method='zncc', subpixel=True)
    binding = processing_contract(config)['source_sha256']
    producer = implementation_identity()['sources']
    for name in ('tabletop_stereo_matching_costs', 'tabletop_stereo_pyramid'):
        assert name + '.py' in binding
        assert name in producer
        path = Path(producer[name]['path'])
        assert binding[name + '.py'] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_configs_change_one_factor_and_have_distinct_processing_sha():
    from go2_pvcnn.control.dh116.tabletop_stereo_preprocessing import processing_contract
    root = Path(__file__).resolve().parents[1]
    _, _, calibration = pair(1)
    parameters = []; identities = []
    for method in METHODS:
        path = root / 'configs' / 'dh116_training' / f'stereo_matcher_{method}_j2b_v1.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        assert value['enabled_by_default'] is False and value['native_qualified'] is False
        depth = value['depth_estimator'].copy()
        assert depth.pop('method') == method
        parameters.append(depth)
        config = calibration.to_dict(); config['depth_estimator'] = value['depth_estimator']
        identities.append(processing_contract(config)['processing_sha256'])
    assert all(value == parameters[0] for value in parameters)
    assert len(set(identities)) == len(METHODS)


def test_metadata_describes_only_the_executed_metric_path():
    left, right, calibration = pair(1)
    common = dict(min_disparity=2, max_disparity=12, block_size=5)
    legacy_reliable = StereoDepth(method='reliable_sad', subpixel=False, **common).compute(left, right, calibration)
    assert legacy_reliable['diagnostics']['uniqueness_scope'] == 'adjacent_top2_cost_margin'
    assert legacy_reliable['diagnostics']['rgb_boundary_proxy_applied'] is False
    candidate = StereoDepth(method='zncc', subpixel=True, **common).compute(left, right, calibration)
    assert candidate['diagnostics']['rgb_boundary_proxy_applied'] is False


@pytest.mark.parametrize('method', METHODS)
def test_candidate_missing_view_is_not_replaced_by_fake_depth(method):
    left, right, calibration = pair(1)
    out = StereoDepth(method=method, min_disparity=2, max_disparity=12,
        block_size=5, subpixel=True).compute(left, right, calibration,
        right_valid=torch.zeros_like(right[:, 0], dtype=torch.bool))
    assert not out['valid'].any()
    assert torch.isnan(out['depth_m']).all()


@pytest.mark.parametrize('method', METHODS)
def test_runtime_has_no_reference_input(method):
    left, right, calibration = pair(1)
    with pytest.raises(TypeError):
        StereoDepth(method=method).compute(left, right, calibration, reference_depth=torch.ones(1, 32, 96))
