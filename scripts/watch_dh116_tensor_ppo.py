"""Read-only training observer. Never restarts or changes the trainer."""
import datetime,json,os,time,math
from pathlib import Path


def training_trend(rows):
    """Compare complete 50-update windows; request diagnosis, never restart PPO."""
    thresholds={'signed_force_mean_n':.02,'load_ready_fraction':.005,
                'max_clearance_m':.0005,'successful_grasps':1}
    report=dict(status='collecting_baseline',diagnosis_required=False,
                no_improvement_windows=0,last_complete_iteration=None,windows=[],
                window_iterations=50,comparison='best_previous_complete_windows',
                improvement_thresholds=thresholds,
                grip_metric_note='load_ready_fraction is a grip proxy, not stable-hold duration')
    values=[]
    try:
        for index,row in enumerate(rows):
            iteration=row['iteration']
            if type(iteration) is not int or iteration!=rows[0]['iteration']+index:
                raise ValueError('nonconsecutive iterations')
            item=(row['vertical_contact_force']['hands_total_mean_n'],
                  row['load_ready_env_fraction'],row['max_object_bottom_m'],row['successes'])
            if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in item):
                raise ValueError('invalid trend metric')
            if not 0<=item[1]<=1 or item[3]<0 or (values and item[3]<values[-1][3]):
                raise ValueError('invalid grip fraction or success counter')
            values.append(item)
    except (KeyError,TypeError,ValueError) as exc:
        report.update(status='invalid_data',error=str(exc))
        return report
    best={}
    for start in range(0,len(values)-49,50):
        window=values[start:start+50]
        metrics=dict(signed_force_mean_n=sum(x[0] for x in window)/50,
                     load_ready_fraction=sum(x[1] for x in window)/50,
                     max_clearance_m=max(x[2] for x in window)-.30,
                     successful_grasps=window[-1][3])
        improved=[key for key in best if metrics[key]-best[key]>=thresholds[key]]
        if best:
            report['no_improvement_windows']=0 if improved else report['no_improvement_windows']+1
        for key,value in metrics.items():best[key]=max(best.get(key,value),value)
        report['windows'].append(dict(first_iteration=rows[start]['iteration'],
            last_iteration=rows[start+49]['iteration'],metrics=metrics,improved_metrics=improved))
        report['last_complete_iteration']=rows[start+49]['iteration']
    if report['windows']:
        report['diagnosis_required']=report['no_improvement_windows']>=3
        report['status']='diagnosis_required' if report['diagnosis_required'] else 'monitoring'
    return report


def complete_iteration_rows(path,through_iteration):
    """A concurrent writer's partial final line belongs to the next read."""
    rows=[]
    with path.open() as stream:
        for line in stream:
            if not line.endswith('\n'):break
            row=json.loads(line)
            if row['iteration']>through_iteration:break
            rows.append(row)
    return rows


def write_report(path,report):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def physics_problems(monitor):
    """Report fatal measured peaks even if an older trainer says training."""
    limits={'all_joint_speed_rad_s':(3.+1e-5,False),
            'active_joint_speed_rad_s':(3.+1e-5,False),
            'link_force_n':(45.,True),'mimic_error_rad':(.03,True)}
    problems=[]
    if monitor.get('fault_trace_record_error'):
        problems.append('fault_trace_record_error:'+str(monitor['fault_trace_record_error']))
    if 'physics_peaks_since_start' not in monitor:
        if monitor.get('status')=='training':problems.append('physics_peak_telemetry_missing')
        return problems
    peaks=monitor['physics_peaks_since_start']
    if not isinstance(peaks,dict): return ['physics_peak_telemetry_invalid']
    if monitor.get('status')=='training' and not limits.keys()<=peaks.keys():
        problems.append('physics_peak_telemetry_missing_fields')
    for name,value in peaks.items():
        if name not in limits: continue
        if value is None or not isinstance(value,(int,float)) or not math.isfinite(value):
            problems.append('nonfinite_physics_peak:'+name)
            continue
        limit,inclusive=limits[name]
        if value>limit or (inclusive and value==limit):
            problems.append('physics_peak_exceeded:'+name+':'+str(value))
    return problems


def policy_noise_problems(monitor):
    if 'policy_noise' not in monitor:return []  # Historical runs lack this contract.
    problems=[]
    for key,low,high in (('policy_std_min',.05-1e-6,1.+1e-6),
                         ('policy_std_max',.05-1e-6,1.+1e-6),
                         ('action_clipped_fraction',0.,1.)):
        value=monitor.get(key)
        if not isinstance(value,(float,int)) or not math.isfinite(value) or not low<=value<=high:
            problems.append('policy_noise_invalid:'+key+':'+str(value))
    return problems


def telemetry_problems(monitor):
    diagnostic=monitor.get('close_completion_telemetry',{})
    if diagnostic.get('status')!='degraded':return []
    return ['close_completion_telemetry_degraded:'+name+':'+str(error)
            for name,error in diagnostic.get('errors',{}).items()]


def main():
    active=Path('runs/active_tabletop_run.json')
    record=json.loads(active.read_text())
    run=Path(record['run']);pid=record['pid'];monitor=Path(record['monitor'])
    last=None;progress_at=time.monotonic()
    trend=None;next_trend_iteration=0
    while True:
        problems=[]
        try:
            proc=Path('/proc')/str(pid)/'stat'
            alive=proc.exists() and proc.read_text().rsplit(')',1)[1].split()[0]!='Z'
            m=json.loads(monitor.read_text())
            problems.extend(physics_problems(m))
            problems.extend(policy_noise_problems(m))
            telemetry_warnings=telemetry_problems(m)
            counters=(m.get('timesteps',0),m.get('iteration',m.get('updates',0)))
            if last is not None and any(a<b for a,b in zip(counters,last)):
                problems.append('training_counter_regression')
            if counters!=last:progress_at=time.monotonic()
            last=counters
            if m.get('safety_stop'):problems.append('safety_stop:'+str(m['safety_stop']))
            if m.get('status') in ('error','not_admitted','stopped'):problems.append('status:'+m['status'])
            if not alive and m.get('status') not in ('completed_budget','training_budget_complete'):problems.append('trainer_exited')
            if alive and time.monotonic()-progress_at>180:problems.append('no_progress_for_180_seconds')
            snapshot={k:m.get(k) for k in ('status','timesteps','updates','iteration','iteration_seconds','episodes','episodes_per_env_min','episodes_per_env_max','successes','recovered_physical_episodes','physics_peaks_since_start','error','safety_stop','completed_training_episodes','min_completed_episodes_per_env','max_completed_episodes_per_env','transitions_per_second')}
            snapshot.update(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),pid=pid,alive=alive,problems=problems,automated_repair=False)
            snapshot.update({k:m.get(k) for k in ('policy_std_min','policy_std_max','action_clipped_fraction')})
            snapshot['vertical_contact_force']=m.get('vertical_contact_force')
            snapshot['lift_entry_force']=m.get('lift_entry_force')
            snapshot['lift_gates']=m.get('lift_gates')
            snapshot['fault_trace_record_error']=m.get('fault_trace_record_error')
            snapshot['close_completion_telemetry']=m.get('close_completion_telemetry')
            snapshot['telemetry_warnings']=telemetry_warnings
            if 'iteration' in m and m['iteration']>=next_trend_iteration:
                rows=complete_iteration_rows(run/'iterations.jsonl',m['iteration'])
                trend=training_trend(rows)
                first=rows[0]['iteration'] if rows else m['iteration']+1
                next_trend_iteration=first+(len(rows)//50+1)*50-1
                if trend['status']=='invalid_data':next_trend_iteration=m['iteration']+1
                write_report(run/'trend_watch.json',trend)
                if trend['diagnosis_required']:
                    request=dict(reason='three_windows_without_measured_progress',
                                 trend=trend,automated_repair=False)
                    write_report(run/'diagnostic_request.json',request)
                    print(json.dumps(request,ensure_ascii=False),flush=True)
            snapshot['training_trend']=trend
            snapshot['failed_envs']=[{k:e.get(k) for k in ('env_index','last_failure','last_physical','force_feedback')} for e in m.get('per_env',[]) if e.get('last_failure')]
            temp=run/'live_watch.tmp';temp.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2));temp.replace(run/'live_watch.json')
            with (run/'watch_history.jsonl').open('a') as f:f.write(json.dumps(snapshot,ensure_ascii=False)+'\n')
            current=json.loads(active.read_text())
            if current.get('pid')==pid:
                current['status']=m.get('status');current['last_verified_timesteps']=counters[0];current['last_verified_iteration' if 'iteration' in m else 'last_verified_optimizer_epochs']=counters[1]
                tmp=active.with_suffix('.watch.tmp');tmp.write_text(json.dumps(current,indent=2)+'\n');tmp.replace(active)
            if problems or telemetry_warnings:
                print(json.dumps(snapshot,ensure_ascii=False),flush=True)
            if problems:
                raise SystemExit(2)
            if not alive or m.get('status') in ('completed_budget','training_budget_complete'):
                print(json.dumps(snapshot,ensure_ascii=False),flush=True);break
        except (FileNotFoundError,json.JSONDecodeError) as exc:
            print('Observer transient read: '+str(exc),flush=True)
        time.sleep(30)


if __name__=='__main__':
    main()
