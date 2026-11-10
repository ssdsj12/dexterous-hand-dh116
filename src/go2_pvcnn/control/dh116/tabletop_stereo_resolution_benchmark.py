"""Serial opt-in native resolution benchmark; no fake CUDA/Actor measurements."""
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from .tabletop_stereo_calibration import StereoCalibration
from .tabletop_stereo_contract import ROOT, file_sha256, new_run, require_opt_in, write_json_new


def gpu_memory_snapshot(device='cuda'):
    names=('gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','gpu_free_bytes',
           'gpu_device_total_bytes','gpu_device_used_bytes')
    result={key:None for key in names}
    result.update(source_scope='unavailable',measurement_device=str(device),
                  scope='Torch selected device; device used/free is a sampled snapshot, not total-job or NVML process peak')
    target=torch.device(device)
    if target.type!='cuda' or not torch.cuda.is_available():return result
    torch.cuda.synchronize(target)
    free,total=torch.cuda.mem_get_info(target)
    result.update(source_scope='native',measurement_device='cuda',device_name=torch.cuda.get_device_name(target),
        gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(target)),
        gpu_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(target)),gpu_free_bytes=int(free),
        gpu_device_total_bytes=int(total),gpu_device_used_bytes=int(total-free))
    return result


def timed_call(function,device,*args,**kwargs):
    target=torch.device(device)
    if target.type=='cuda':torch.cuda.synchronize(target)
    start=time.perf_counter();result=function(*args,**kwargs)
    if target.type=='cuda':torch.cuda.synchronize(target)
    return result,(time.perf_counter()-start)*1000.


class TimedStereoDepth:
    """Measure the actual estimator invoked by StereoCamera, never a proxy task."""
    def __init__(self,estimator,device):self.estimator,self.device=estimator,device;self.samples=[]
    @property
    def last_resource_metadata(self):return getattr(self.estimator,'last_resource_metadata',None)
    def compute(self,*args,**kwargs):
        result,ms=timed_call(self.estimator.compute,self.device,*args,**kwargs)
        self.samples.append(ms);return result


def benchmark_command(config,output,*,seed=116,num_envs=2,frames=4,image_processing_batch_size=1,
                      actor_package=None,warmup_frames=2,controlled_depth=True,integer_disparity=False):
    command=[sys.executable,str(ROOT/'scripts/probe_dh116_stereo_camera.py'),
        '--config',str(Path(config).resolve()),'--output',str(Path(output).resolve()),
        '--seed',str(seed),'--num-envs',str(num_envs),'--frames',str(frames),
        '--warmup-frames',str(warmup_frames),'--image-processing-batch-size',str(image_processing_batch_size),
        '--capture-validation-reference','--enable-experiment','--execute-native']
    if actor_package is not None:command+=['--actor-package',str(Path(actor_package).resolve())]
    if frames+warmup_frames>1:command+=['--advance-control-between-samples']
    if controlled_depth:command+=['--controlled-depth']
    if integer_disparity:command+=['--integer-disparity']
    return command


def _source_freeze(config):
    paths=list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'configs/dh116_training').glob('stereo*.json'))+[Path(config)]
    return {str(path.resolve()):file_sha256(path) for path in sorted(set(paths))}


def run_resolution_benchmark(configs,output,*,seed=116,num_envs=2,frames=4,image_processing_batch_size=1,
                             actor_package=None,warmup_frames=2,execute_native=False,enabled=False,runner=None,
                             selection_thresholds=None):
    require_opt_in(enabled)
    if not configs or min(num_envs,frames,image_processing_batch_size)<1 or warmup_frames<0:
        raise ValueError('explicit positive benchmark budgets required')
    calibrations=[StereoCalibration.from_json(path) for path in configs]
    sizes=[(c.width,c.height) for c in calibrations]
    if len(set(sizes))!=len(sizes):raise ValueError('resolution configs must be unique')
    run=new_run(output);(run/'logs').mkdir();(run/'source_freezes').mkdir()
    plan=[]
    for path,c in zip(configs,calibrations):
        label=f'{c.width}x{c.height}'
        frozen=_source_freeze(path)
        write_json_new(run/'source_freezes'/f'{label}.json',dict(source_sha256=frozen,config_sha256=file_sha256(path)))
        plan.append(dict(width=c.width,height=c.height,label=label,config_path=str(Path(path).resolve()),
            config_sha256=file_sha256(path),calibration_sha256=c.sha256,
            command=benchmark_command(path,run/label,seed=seed,num_envs=num_envs,frames=frames,
                image_processing_batch_size=image_processing_batch_size,actor_package=actor_package,warmup_frames=warmup_frames),
            source_freeze=frozen))
    write_json_new(run/'benchmark_plan.json',dict(version='dh116-stereo-resolution-benchmark-v1',
        execute_native=execute_native,seed=seed,serial_subprocesses=True,plans=plan,
        native_started=False,actor_package_sha256=file_sha256(actor_package) if actor_package else None))
    rows=[];actual_captures=0;actual_jobs=0;attempts=0
    for item in plan:
        metrics={key:None for key in ('depth_mae_m','depth_rmse_m','depth_p95_m','wire_point_cloud_completeness',
            'rgb_candidate_point_cloud_completeness','gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','gpu_free_bytes',
            'gpu_device_used_bytes','capture_ms','depth_ms','actor_ms')}
        row=dict(width=item['width'],height=item['height'],source_scope='unavailable',schema_only=True,
                 metrics=metrics,selected=False,actor_package_provided=actor_package is not None,
                 reason='native_not_executed',exit_code=None)
        if execute_native:
            attempts+=1;call=runner or subprocess.run
            out=run/'logs'/f"{item['label']}.stdout.log";err=run/'logs'/f"{item['label']}.stderr.log"
            try:
                with out.open('x',encoding='utf-8') as stdout,err.open('x',encoding='utf-8') as stderr:
                    result=call(item['command'],stdout=stdout,stderr=stderr,cwd=str(ROOT),check=False,text=True)
                row.update(exit_code=int(result.returncode),stdout_path=str(out),stderr_path=str(err),
                           reason='native_process_failed' if result.returncode else 'capture_pending_independent_qualification')
            except OSError as error:
                row.update(exit_code=-1,reason='process_launch_failed',error=str(error))
            changed=[path for path,sha in item['source_freeze'].items() if not Path(path).is_file() or file_sha256(path)!=sha]
            if changed:row.update(reason='source_changed_during_benchmark',changed_sources=changed)
            report_path=run/item['label']/'stereo_probe.json'
            if row['exit_code']==0 and report_path.is_file() and not changed:
                report=json.loads(report_path.read_text(encoding='utf-8'))
                row.update(source_scope=report.get('source_scope','unqualified'),schema_only=report.get('schema_only',True),
                    unverified_callback_metrics=report.get('benchmark_metrics'),capture_report_path=str(report_path),
                    capture_report_sha256=file_sha256(report_path),cohort_sha256=report.get('cohort_sha256'),
                    stereo_qualification=report.get('stereo_qualification'))
                if runner is None and report.get('source_scope')=='native':actual_captures+=1
            child=run/item['label']
            marker=child/'native_camera_callback_entered.json'
            if runner is None and marker.is_file():
                started=json.loads(marker.read_text(encoding='utf-8'))
                if started.get('source_scope')=='native_runtime_callback_entered':actual_jobs+=1
            produced=[dict(path=str(path.resolve()),sha256=file_sha256(path)) for path in child.rglob('*') if path.is_file()] if child.exists() else []
            completion=dict(version='dh116-external-native-completion-v1',
                source_scope='native_process_exit_observed' if runner is None else 'cpu_fixture',
                command=item['command'],returncode=row['exit_code'],
                runner_source_artifact={'path':str(Path(__file__).resolve()),'sha256':file_sha256(__file__)},
                stdout_artifact={'path':str(out),'sha256':file_sha256(out)} if out.exists() else None,
                stderr_artifact={'path':str(err),'sha256':file_sha256(err)} if err.exists() else None,
                produced_artifacts=produced,source_changed_during_run=bool(changed))
            completion_path=run/f"{item['label']}_completion.json";write_json_new(completion_path,completion)
            pending=child/'native_stereo_proof_pending.json'
            if runner is None and row['exit_code']==0 and not changed and pending.is_file():
                proof=json.loads(pending.read_text(encoding='utf-8'))
                proof.update(process_exit_code=0,qualification_status='external_process_exit_observed_not_accuracy_qualification',
                    completion_artifact={'path':str(completion_path),'sha256':file_sha256(completion_path)})
                completed=child/'native_stereo_proof_completed.json';write_json_new(completed,proof)
                row['stereo_qualification']=str(completed)
            if runner is not None:
                row.update(source_scope='cpu_fixture',schema_only=True,metrics={key:None for key in metrics},
                           reason='injected_runner_is_not_native_evidence')
        rows.append(row)
        write_json_new(run/f"{item['label']}_result.json",row)
    from .tabletop_stereo_resolution import compare_resolution_results
    comparison=compare_resolution_results(rows,thresholds=selection_thresholds)
    write_json_new(run/'resolution_comparison.json',comparison)
    for row,verified in zip(rows,comparison['resolutions']):
        row['metrics'].update(verified['metrics'])
        row['metrics']['rgb_candidate_point_cloud_completeness']=verified['rgb_candidate_point_cloud_completeness']
        row['metrics']['gpu_device_used_bytes']=verified.get('gpu_device_used_bytes')
        for key in ('measurements_verified','qualified','blocked_reasons','stereo_pairs_per_second','cohort_sha256','sample_count'):
            row[key]=verified[key]
        row['selected']=comparison['selected_resolution']==[row['width'],row['height']]
        write_json_new(run/f"{row['width']}x{row['height']}_verified_result.json",row)
    result=dict(version='dh116-stereo-resolution-benchmark-v1',native_jobs_started=actual_jobs,native_captures_completed=actual_captures,
        subprocess_attempts=attempts,source_scope='native_runtime_capture_unqualified' if actual_captures else 'unqualified',
        resolutions=rows,selected_resolution=comparison['selected_resolution'],comparison=comparison,
        limitation='raw capture/reference/performance evidence must pass common verification; RGB candidate completeness is not GT; missing Actor latency is NA')
    write_json_new(run/'resolution_benchmark.json',result)
    return result


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs',type=Path,nargs='+',default=[ROOT/'configs/dh116_training/stereo_camera_640x480_v2.json'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--seed',type=int,default=116);parser.add_argument('--num-envs',type=int,default=2)
    parser.add_argument('--frames',type=int,default=4);parser.add_argument('--warmup-frames',type=int,default=2)
    parser.add_argument('--image-processing-batch-size',type=int,default=1)
    parser.add_argument('--actor-package',type=Path)
    parser.add_argument('--selection-thresholds',type=Path,help='explicit JSON threshold policy; omitted means no final resolution selection')
    parser.add_argument('--enable-experiment',action='store_true');parser.add_argument('--execute-native',action='store_true')
    parser.add_argument('--check-config',action='store_true')
    args=parser.parse_args(argv)
    if args.check_config:
        print(json.dumps(dict(validated=True,native_started=False,resolutions=[
            [StereoCalibration.from_json(path).width,StereoCalibration.from_json(path).height] for path in args.configs])))
        return 0
    if args.output is None:parser.error('explicit new --output required')
    result=run_resolution_benchmark(args.configs,args.output,seed=args.seed,num_envs=args.num_envs,frames=args.frames,
        warmup_frames=args.warmup_frames,image_processing_batch_size=args.image_processing_batch_size,
        actor_package=args.actor_package,enabled=args.enable_experiment,execute_native=args.execute_native,
        selection_thresholds=json.loads(args.selection_thresholds.read_text(encoding='utf-8')) if args.selection_thresholds else None)
    print(json.dumps(result,indent=2));return 0 if all(row['exit_code'] in (None,0) for row in result['resolutions']) else 1
