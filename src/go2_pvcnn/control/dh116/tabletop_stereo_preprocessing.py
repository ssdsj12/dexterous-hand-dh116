"""Semantic identity of the RGB-derived depth/visible-wire input pipeline.

Software binding does not establish calibration accuracy or native qualification.
Image chunk sizes are resource choices and do not change this semantic identity.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .tabletop_stereo_calibration import StereoCalibration
from .tabletop_stereo_contract import digest
from .tabletop_stereo_depth import StereoDepth

VERSION='dh116-stereo-processing-v1'


def _sources():
    names=('depth','perception','geometry','preprocessing','matcher_candidates','matching_costs','pyramid')
    return {f'tabletop_stereo_{name}.py':hashlib.sha256(Path(__file__).with_name(
        f'tabletop_stereo_{name}.py').read_bytes()).hexdigest() for name in names}


def processing_contract(camera_config):
    config=json.loads(Path(camera_config).read_text(encoding='utf-8')) if isinstance(camera_config,(str,Path)) else deepcopy(camera_config)
    if not isinstance(config,dict):raise ValueError('actual camera configuration dictionary/path required')
    calibration=StereoCalibration.from_dict(config)
    depth=StereoDepth.from_config(config).to_config()
    depth.pop('image_processing_batch_size',None)
    for key in ('min_depth_m','max_depth_m','lr_tolerance_px','texture_threshold','uniqueness_threshold'):
        depth[key]=float(depth[key])
    result=dict(version=VERSION,calibration_sha256=calibration.sha256,
        image_resolution_wh=[calibration.width,calibration.height],depth_estimator=depth,
        rgb=dict(channel_order='RGB',layout='NCHW',range=[0.,1.],dtype='float32',
            byte_conversion='integer_or_float_above_one_div_255',gray='arithmetic_mean_RGB',hidden_resize=False),
        geometry=dict(depth_unit='m',pixel_centres='u+0.5,v+0.5',
            optical_axes='+X_right,+Y_down,+Z_forward',transform='T_world_left_times_optical_point',
            calibration_rectified=calibration.rectified,rectification='explicit_only;no_hidden_rectify',
            missing='NaN_with_false_validity;no_simulator_or_rendered_depth_fallback'),
        perception=dict(algorithm='warm_colour_visible_simple_path_ZhangSuen_v1',point_count=128,
            thresholds=dict(red_over_green=1.07,green_over_blue=1.1,minimum_red=.12),
            thinning_max_iterations=128,identity_motion_limit_m=.005,
            ordering='observed_visible_arc_length;dominant_world_axis_endpoint_orientation',
            temporal_identity='endpoint_consistency_diagnostic;not_native_identity_qualification',
            missing_or_branched='invalid_tokens;never_invent_hidden_segments'),
        source_sha256=_sources(),resource_fields_excluded=['image_processing_batch_size'],
        native_qualified=False)
    result['processing_sha256']=digest(result)
    return result


def validate_processing_contract(value,expected=None):
    if not isinstance(value,dict) or value.get('version')!=VERSION:
        raise ValueError('versioned stereo processing contract required')
    if value.get('native_qualified') is not False or value.get('source_sha256')!=_sources():
        raise ValueError('processing implementation source SHA mismatch or unsupported qualification claim')
    if value.get('processing_sha256')!=digest({key:item for key,item in value.items() if key!='processing_sha256'}):
        raise ValueError('processing contract SHA mismatch')
    if expected is not None and value!=expected:
        raise ValueError('actual camera processing differs from bound preprocessor contract')
    return value


def require_raw_processing(actor_contract,camera_config):
    if not isinstance(actor_contract,dict):raise ValueError('actor processing binding required')
    binding=actor_contract.get('preprocessor',actor_contract)
    if not isinstance(binding,dict) or binding.get('binding')!='bound_raw_stereo':
        raise ValueError('history-input-only/unspecified package has no raw RGB processing binding')
    actual=processing_contract(camera_config)
    validate_processing_contract(binding.get('processing'),expected=actual)
    if actor_contract.get('calibration_sha256',actual['calibration_sha256'])!=actual['calibration_sha256']:
        raise ValueError('actor calibration differs from actual processing contract')
    return actual
