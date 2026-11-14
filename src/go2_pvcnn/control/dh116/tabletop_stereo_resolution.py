"""Resolution profiles and evidence-based comparisons; no native data => NA."""
import json
import math
from pathlib import Path


DEFAULT_MAX_WIDTH=2048
DEFAULT_MAX_HEIGHT=1536
METRIC_KEYS=('depth_mae_m','depth_rmse_m','depth_p95_m','wire_point_cloud_completeness',
    'gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','gpu_free_bytes','capture_ms','depth_ms','actor_ms')
WIRE_DEPTH_KEYS=('wire_depth_mae_m','wire_depth_rmse_m','wire_depth_p95_m',
    'wire_depth_validity_coverage','wire_depth_count')


def validate_resolution(width,height,*,max_width=DEFAULT_MAX_WIDTH,max_height=DEFAULT_MAX_HEIGHT,
                        allow_resource_override=False):
    if any(type(v) is not int or v<1 for v in (width,height,max_width,max_height)):
        raise ValueError('positive integer resolution and resource limits required')
    if type(allow_resource_override) is not bool:raise ValueError('explicit resource override boolean required')
    if not allow_resource_override and (width>max_width or height>max_height):
        raise ValueError('resolution exceeds configured resource limit; explicit override required')
    return width,height


def rerender_calibration(calibration,width,height,*,horizontal_fov_degrees=75.,**resource_limits):
    from .tabletop_stereo_calibration import rerender_calibration as implementation
    return implementation(calibration,width,height,horizontal_fov_degrees=horizontal_fov_degrees,**resource_limits)


def resize_calibration(calibration,width,height,**resource_limits):
    from .tabletop_stereo_calibration import resize_calibration as implementation
    return implementation(calibration,width,height,**resource_limits)


def disparity_budget(calibration,near_m=.2,far_m=1.2,*,padding_px=8):
    if not 0<near_m<far_m or type(padding_px) is not int or padding_px<0:
        raise ValueError('positive near/far and integer disparity padding required')
    fx=float(calibration.K_left[0,0]);baseline=calibration.baseline_m
    near=fx*baseline/near_m;far=fx*baseline/far_m
    return dict(near_disparity_px=near,far_disparity_px=far,
        suggested_min_disparity=max(1,math.floor(far)-padding_px),
        suggested_max_disparity=16*math.ceil((near+padding_px)/16))


def _number(value,*,positive=False):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0 or positive and value<=0:return None
    return value


def _verified_stereo(proof):
    from .tabletop_stereo_promotion import measure_stereo_evidence
    return measure_stereo_evidence(proof)


def compare_resolution_results(reports,*,thresholds=None):
    """Only the common raw-proof verifier can supply native measured values.

    Unverified report metrics are deliberately ignored. GPU/timing values must
    be returned by that same verifier after its performance-artifact checks.
    No CPU-zero memory statistic, resolution preference or invented accuracy
    is used to choose a profile.
    """
    rows=[]
    for source in reports:
        report=json.loads(Path(source).read_text(encoding='utf-8')) if isinstance(source,(str,Path)) else source
        if not isinstance(report,dict):raise ValueError('resolution report dictionary or JSON path required')
        width,height=validate_resolution(report.get('width'),report.get('height'),allow_resource_override=True)
        row=dict(width=width,height=height,metrics={k:None for k in METRIC_KEYS},
            wire_depth_metrics={k:None for k in WIRE_DEPTH_KEYS},
            stereo_pairs_per_second=None,qualified=False,measurements_verified=False,cohort_sha256=None,sample_count=None,
            capture_depth_pairs_per_second=None,
            capture_depth_throughput_scope='capture_plus_depth_only;excludes_Actor_and_physics;not_used_for_selection',
            rgb_candidate_point_cloud_completeness=None,gpu_device_used_bytes=None,
            blocked_reasons=[],accuracy_source='NA',performance_source='NA')
        binding=None
        if report.get('source_scope')=='native' and report.get('schema_only') is False:
            try:
                binding=_verified_stereo(report.get('stereo_qualification',report.get('qualification')))
                if (not isinstance(binding,dict) or binding.get('measurements_verified') is not True
                        or binding.get('source_scope')!='native' or (binding.get('width'),binding.get('height'))!=(width,height)):
                    raise ValueError('qualified resolution/calibration source mismatch')
            except (ValueError,FileNotFoundError,ImportError) as error:
                row['blocked_reasons'].append('native_stereo_proof_unavailable: '+str(error));binding=None
        else:row['blocked_reasons'].append('CPU_fixture_or_non_native_source')
        if binding is not None:
            row['qualified']=binding.get('qualified') is True
            row['measurements_verified']=True;row['accuracy_source']='verified_native_raw_stereo_proof'
            if not row['qualified']:row['blocked_reasons'].append('measured_native_accuracy_did_not_pass_qualification_thresholds')
            metrics=binding.get('metrics',{})
            row['wire_depth_metrics']={key:_number(metrics.get(key)) for key in WIRE_DEPTH_KEYS}
            for dst,src in (('depth_mae_m','mae_m'),('depth_rmse_m','rmse_m'),('depth_p95_m','p95_m'),
                            ('wire_point_cloud_completeness','wire_point_cloud_completeness')):
                row['metrics'][dst]=_number(metrics.get(src))
            row['cohort_sha256']=binding.get('cohort_sha256')
            row['rgb_candidate_point_cloud_completeness']=_number(metrics.get('rgb_candidate_point_cloud_completeness'))
            row['sample_count']=_number(binding.get('sample_count',metrics.get('frame_count')))
            performance=binding.get('performance')
            if isinstance(performance,dict) and performance.get('source_scope')=='native' and performance.get('measurement_device')=='cuda':
                row['gpu_device_used_bytes']=_number(performance.get('gpu_device_used_bytes'))
                for key in METRIC_KEYS[4:]:row['metrics'][key]=_number(performance.get(key))
                allocated,reserved=(row['metrics'][key] for key in ('gpu_peak_allocated_bytes','gpu_peak_reserved_bytes'))
                if allocated is None or reserved is None or allocated<=0 or reserved<allocated:
                    for key in METRIC_KEYS[4:7]:row['metrics'][key]=None
                    row['blocked_reasons'].append('GPU_measurements_missing_or_inconsistent')
                else:row['performance_source']='verified_native_cuda_artifact'
                times=[row['metrics'][key] for key in ('capture_ms','depth_ms','actor_ms')]
                batch=_number(performance.get('batch_size'),positive=True)
                capture_depth=[row['metrics'][key] for key in ('capture_ms','depth_ms')]
                if batch is not None and all(value is not None for value in capture_depth) and sum(capture_depth)>0:
                    row['capture_depth_pairs_per_second']=1000*batch/sum(capture_depth)
                if batch is not None and all(value is not None for value in times) and sum(times)>0:
                    row['stereo_pairs_per_second']=1000*batch/sum(times)
            else:row['blocked_reasons'].append('native_GPU_and_timing_proof_unavailable')
        rows.append(row)
    result=dict(version='dh116-stereo-resolution-comparison-v1',resolutions=rows,
        native_reports=sum(row['measurements_verified'] for row in rows),
        qualified_native_reports=sum(row['qualified'] for row in rows),selected_resolution=None,
        selection_basis='verified_same_cohort_measured_metrics_only',selection_reasons=[])
    required=('max_depth_mae_m','max_depth_rmse_m','max_depth_p95_m',
              'min_wire_point_cloud_completeness','min_sample_count','min_stereo_pairs_per_second')
    if thresholds is None:
        result['selection_reasons'].append('explicit_accuracy_coverage_sample_and_throughput_thresholds_required');return result
    if (not isinstance(thresholds,dict) or any(_number(thresholds.get(key)) is None for key in required)):
        raise ValueError('complete finite resolution selection thresholds required')
    measured=[row for row in rows if row['measurements_verified']]
    native=[row for row in measured if row['qualified']]
    cohorts={row['cohort_sha256'] for row in measured}
    if len(cohorts)!=1 or None in cohorts:
        result['selection_reasons'].append('verified_actual_same_cohort_required');return result
    eligible=[]
    for row in native:
        m=row['metrics']
        if (any(m[key] is None for key in METRIC_KEYS) or row['sample_count'] is None or row['stereo_pairs_per_second'] is None):continue
        if (m['depth_mae_m']<=thresholds['max_depth_mae_m'] and m['depth_rmse_m']<=thresholds['max_depth_rmse_m']
                and m['depth_p95_m']<=thresholds['max_depth_p95_m']
                and m['wire_point_cloud_completeness']>=thresholds['min_wire_point_cloud_completeness']
                and row['sample_count']>=thresholds['min_sample_count']
                and row['stereo_pairs_per_second']>=thresholds['min_stereo_pairs_per_second']
                and ('max_gpu_peak_reserved_bytes' not in thresholds or m['gpu_peak_reserved_bytes']<=thresholds['max_gpu_peak_reserved_bytes'])):
            eligible.append(row)
    if eligible:
        selected=min(eligible,key=lambda row:(row['metrics']['depth_rmse_m'],row['metrics']['depth_p95_m'],
                                             -row['stereo_pairs_per_second']))
        result['selected_resolution']=[selected['width'],selected['height']]
        result['selection_reasons'].append('minimum_measured_RMSE_then_P95_then_maximum_measured_throughput_among_threshold_passers')
    else:result['selection_reasons'].append('no_complete_qualified_resolution_meets_all_thresholds')
    return result
