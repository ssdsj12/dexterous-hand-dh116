"""Synthetic API/statistics/launch checks only; no native jobs are registered."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_reference import (
    ValidationDepthReference, capture_validation_pair, compare_reference, resolve_wire_instance_mask,
)
from go2_pvcnn.control.dh116.tabletop_stereo_resolution_benchmark import (
    benchmark_command, gpu_memory_snapshot, run_resolution_benchmark,
    TimedStereoDepth,
)


def packet(frame=1):
    return dict(left_rgb=torch.zeros(1,3,4,8),right_rgb=torch.zeros(1,3,4,8),
                frame_id=frame,simulation_time=.1,calibration_sha256='1'*64,
                depth_m=torch.ones(1,4,8),valid=torch.ones(1,4,8,dtype=torch.bool),
                native_evidence=False)


class DepthReader:
    def read(self):
        return dict(depth_left_m=torch.ones(1,4,8),depth_right_m=torch.ones(1,4,8),
                    frame_id=1,simulation_time=.1,calibration_sha256='1'*64)


def test_independent_reference_is_validation_only_and_cpu_reader_never_native():
    p=packet();reference=ValidationDepthReference(DepthReader()).read(p)
    assert reference['source_scope']=='cpu_fixture'
    assert reference['validation_only'] is True and reference['reference_used_as_actor_input'] is False
    assert reference['source_units']=='optical_axis_z_m'
    assert not any(k.startswith('reference') for k in p)
    assert reference['depth_m'].shape==p['depth_m'].shape


@pytest.mark.parametrize('key,value',[('frame_id',2),('simulation_time',.2),('calibration_sha256','2'*64)])
def test_reference_pair_clock_or_calibration_mismatch_is_not_relabelled(key,value):
    p=packet();p[key]=value
    with pytest.raises(ValueError,match='frame|time|calibration'):
        ValidationDepthReference(DepthReader()).read(p)


def test_rgb_candidate_completeness_is_not_claimed_as_wire_ground_truth():
    p=packet();p['left_rgb'][:,0]=.8;p['left_rgb'][:,1]=.4;p['left_rgb'][:,2]=.1
    ref=ValidationDepthReference(DepthReader()).read(p)
    p['valid'][...,:4]=False;p['depth_m'][~p['valid']]=torch.nan
    metrics=compare_reference(p,ref)
    assert metrics['depth_mae_m']==0. and metrics['depth_p95_m']==0.
    assert metrics['rgb_candidate_point_cloud_completeness']==.5
    assert metrics['wire_point_cloud_completeness'] is None
    assert 'not_semantic_ground_truth' in metrics['candidate_mask_source']


def test_missing_depth_never_fills_from_reference():
    p=packet();p['depth_m'].fill_(torch.nan);p['valid'].zero_()
    ref=ValidationDepthReference(DepthReader()).read(p)
    result=compare_reference(p,ref)
    assert result['depth_mae_m'] is None and result['depth_validity_coverage']==0.
    assert torch.isnan(p['depth_m']).all()


def test_resolved_instance_completeness_counts_visible_3d_points_not_rgb_pixels():
    # CPU arithmetic only: an explicit reference mask does not qualify native.
    p=packet();p['K_left']=torch.eye(3)
    ref=ValidationDepthReference(DepthReader()).read(p)
    ref['wire_reference_mask']=torch.ones_like(ref['valid'])
    ref['wire_mask_source_scope']='cpu_resolved_instance_id'
    p['depth_m'][...,:4]=1.02
    result=compare_reference(p,ref)
    assert result['wire_point_cloud_completeness']==.5
    assert result['wire_reference_visible_points']==32
    assert result['wire_reconstructed_points']==16
    assert result['wire_completeness_scope']=='dense_reference_visible_wire_pixels;not_hidden_entire_object'
    assert ref['source_scope']=='cpu_fixture'


def test_instance_reference_archive_preserves_raw_ids_map_and_per_env_paths(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_reference import archive_instance_reference
    ref=ValidationDepthReference(DepthReader()).read(packet())
    ids=torch.ones(1,4,8,dtype=torch.long);roots=['/World/env_0/Wire']
    info={'idToLabels':{'1':{'primPath':roots[0]}}}
    ref.update(raw_instance_ids=ids,raw_instance_info=info,wire_prim_paths=roots,
               wire_instance_resolution=resolve_wire_instance_mask(ids,info,roots))
    binding=archive_instance_reference(ref,tmp_path/'instances.pt')
    assert hashlib.sha256((tmp_path/'instances.pt').read_bytes()).hexdigest()==binding['sha256']
    raw=torch.load(binding['path'],weights_only=False)
    assert torch.equal(raw['raw_instance_ids'],ids) and raw['raw_instance_info']==info
    assert raw['wire_prim_paths']==roots and raw['source_scope']=='cpu_fixture'
    assert raw['frame_id']==ref['frame_id'] and raw['calibration_sha256']==ref['calibration_sha256']
    with pytest.raises(FileExistsError):archive_instance_reference(ref,tmp_path/'instances.pt')


def test_reference_render_product_clock_mismatch_rejects_equal_simulation_time():
    p=packet();p['render_timing']={'frame_id':1,'simulation_time':.1,'render_frame_number':3}
    class WrongClock(DepthReader):
        def read(self):return dict(super().read(),render_timing={'frame_id':1,'simulation_time':.1,'render_frame_number':4})
    with pytest.raises(ValueError,match='render|clock'):
        ValidationDepthReference(WrongClock()).read(p)


def test_gpu_snapshot_on_cpu_is_na_not_zero_vram():
    result=gpu_memory_snapshot('cpu')
    assert result['source_scope']=='unavailable'
    assert result['gpu_peak_allocated_bytes'] is None and result['gpu_free_bytes'] is None


def test_benchmark_command_uses_current_python_explicit_optin_and_reference():
    import sys
    command=benchmark_command('camera.json','run640',seed=116,num_envs=2,frames=2,image_processing_batch_size=1)
    assert command[0]==sys.executable
    assert '--capture-validation-reference' in command and '--execute-native' in command
    assert '--image-processing-batch-size' in command
    assert '--enable-experiment' in command


def test_benchmark_dry_run_preserves_plan_without_launching_or_fake_actor_timing(tmp_path):
    root=Path(__file__).resolve().parents[1]
    config=root/'configs/dh116_training/stereo_camera_v1.json'
    result=run_resolution_benchmark([config],tmp_path/'bench',execute_native=False,enabled=True)
    assert result['native_jobs_started']==0 and result['selected_resolution'] is None
    assert result['resolutions'][0]['metrics']['actor_ms'] is None
    assert result['resolutions'][0]['metrics']['gpu_peak_allocated_bytes'] is None
    assert result['comparison']['native_reports']==0
    assert result['resolutions'][0]['measurements_verified'] is False
    assert (tmp_path/'bench'/'benchmark_plan.json').is_file()
    with pytest.raises(FileExistsError):run_resolution_benchmark([config],tmp_path/'bench',execute_native=False,enabled=True)


def test_native_failure_retains_separate_stdout_stderr_and_never_becomes_success(tmp_path):
    root=Path(__file__).resolve().parents[1]
    config=root/'configs/dh116_training/stereo_camera_v1.json'
    class Result:returncode=9
    def schema_runner(command,stdout,stderr,**kwargs):
        stdout.write('CPU runner substitution; no native job\n');stderr.write('expected failure\n')
        return Result()
    result=run_resolution_benchmark([config],tmp_path/'failure',execute_native=True,enabled=True,runner=schema_runner)
    assert result['resolutions'][0]['exit_code']==9
    assert result['selected_resolution'] is None
    assert list((tmp_path/'failure'/'logs').glob('*.stderr.log'))[0].read_text()=='expected failure\n'


def test_reference_native_imports_are_lazy_and_reference_never_steps_physics():
    import go2_pvcnn.control.dh116.tabletop_stereo_reference as m
    text=Path(m.__file__).read_text(encoding='utf-8')
    assert '.step(' not in text and '.render(' not in text
    assert "output_annotators=['depth','instance_id_segmentation_fast']" in text


def test_latency_wrapper_preserves_actual_depth_resource_contract():
    class Estimator:
        last_resource_metadata={'disparity_planes_in_memory':1,'image_processing_batch_size':1}
        def compute(self):return 'CPU_arithmetic_only'
    wrapped=TimedStereoDepth(Estimator(),'cpu')
    assert wrapped.compute()=='CPU_arithmetic_only'
    assert wrapped.last_resource_metadata==Estimator.last_resource_metadata
    assert len(wrapped.samples)==1


def test_reference_retry_discards_whole_mismatched_pair_without_relabelling():
    class Camera:
        count=0
        def sample(self,**kwargs):self.count+=1;return packet(self.count)
    class Reader:
        count=0
        def read(self):
            self.count+=1;r=DepthReader().read();r['frame_id']=0 if self.count==1 else 2
            return r
    pair,reference,failures=capture_validation_pair(Camera(),ValidationDepthReference(Reader()),{},max_attempts=2)
    assert pair['frame_id']==reference['frame_id']==2
    assert failures[0]['frame_id']==1
    assert reference['source_scope']=='cpu_fixture'


def test_substituted_successful_runner_cannot_write_completed_native_proof(tmp_path):
    root=Path(__file__).resolve().parents[1]
    class Result:returncode=0
    def schema_runner(command,**kwargs):
        child=Path(command[command.index('--output')+1]);child.mkdir()
        (child/'native_stereo_proof_pending.json').write_text(json.dumps({'process_exit_code':None}))
        return Result()
    result=run_resolution_benchmark([root/'configs/dh116_training/stereo_camera_v1.json'],tmp_path/'schema',
        enabled=True,execute_native=True,runner=schema_runner)
    assert result['native_jobs_started']==0
    assert not list((tmp_path/'schema').rglob('native_stereo_proof_completed.json'))
    completion=json.loads(next((tmp_path/'schema').glob('*_completion.json')).read_text())
    assert completion['source_scope']=='cpu_fixture'


def test_instance_reference_matches_exact_actual_wire_roots_per_env_without_broadcast():
    ids=torch.tensor([[[10,11,12]],[[10,11,12]]],dtype=torch.long)
    info={'idToLabels':{'0':'BACKGROUND','10':'/World/env_0/wire/mesh',
                        '11':{'primPath':'/World/env_1/wire/mesh'},'12':'/World/env_0/wire_proxy'}}
    roots=['/World/env_0/wire','/World/env_1/wire']
    result=resolve_wire_instance_mask(ids,info,roots,path_validator=lambda path:True)
    assert result['mask'].tolist()==[[[True,False,False]],[[False,True,False]]]
    assert result['per_env_wire_ids']==[[10],[11]]


@pytest.mark.parametrize('labels',[{'10':'wire'}, {'10':{'class':'wire'}}, {'10':'/World/wire.visibility'}])
def test_unknown_or_semantic_instance_labels_are_na_not_guessed(labels):
    r=resolve_wire_instance_mask(torch.tensor([[[10]]]),{'idToLabels':labels},['/World/wire'],path_validator=lambda path:True)
    assert r['mask'] is None and r['blocked_reasons']


def test_instance_reference_accepts_explicit_json_prim_path_and_requires_stage_readback():
    info={'idToLabels':json.dumps({'7':json.dumps({'primPath':'/World/wire'})})}
    ids=torch.tensor([[[7]]])
    assert resolve_wire_instance_mask(ids,info,['/World/wire'],path_validator=lambda p:True)['mask'].all()
    assert resolve_wire_instance_mask(ids,info,['/World/wire'],path_validator=lambda p:False)['mask'] is None
