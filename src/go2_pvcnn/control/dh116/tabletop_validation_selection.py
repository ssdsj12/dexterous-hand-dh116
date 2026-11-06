"""Select immutable checkpoints using externally trusted native validation.

The first-terminal CPU summary intentionally cannot certify native execution.
Selection requires a separate native-run certificate digest supplied by its
trusted caller, plus all referenced raw evidence. A report flag alone never
crosses this boundary. Test-split results and training peaks are not inputs.
"""
from copy import deepcopy
import gzip
import json
import math
import os
from pathlib import Path
import uuid

import torch

from .tabletop_evaluation_trials import (
    FirstTerminalAggregator, GRASP_STYLES, _digest, _sha256, _statistics, state_sha256, validate_manifest,
)
from .tabletop_vision_experiment_checkpoint import (
    BUNDLE_VERSION, _json_bytes, _same, _contract_json_view, _state_sha256, _validate_completed, file_sha256,
)

VERSION = 'dh116-native-validation-selection-v1'
CERTIFICATE_VERSION = 'dh116-native-visual-evaluation-certificate-v1'
_METRICS = ('lift','hold','slip','contact','torque','action')


def _read(path):
    with Path(path).open('r',encoding='utf-8') as stream:return json.load(stream)


def _require(condition,message):
    if not condition:raise ValueError(message)


def _finite(value, *, minimum=0., maximum=None):
    _require(type(value) in (int,float) and math.isfinite(value)
        and (minimum is None or value>=minimum) and (maximum is None or value<=maximum),
        'required validation metric must be finite and in range')
    return float(value)


def _file(record,base):
    _require(isinstance(record,dict) and isinstance(record.get('path'),str),'missing evidence file')
    path=Path(record['path'])
    if not path.is_absolute():path=base/path
    digest=_sha256(record.get('sha256'),'evidence SHA256')
    _require(path.is_file() and file_sha256(path)==digest,'raw evidence SHA256 mismatch')
    return path


def _bundle(path):
    value=_read(path)
    _require(value.get('version')==BUNDLE_VERSION and value.get('complete_ppo_update') is True,
        'immutable completed PPO bundle required')
    checkpoint=Path(value['checkpoint_path'])
    _require(checkpoint.is_file() and file_sha256(checkpoint)==value.get('checkpoint_sha256'),
        'immutable checkpoint SHA256 mismatch')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    _validate_completed(saved)
    _require(saved['iteration']==value.get('iteration') and saved['timesteps']==value.get('timesteps')
        and saved['learning_rate']==value.get('learning_rate')
        and _same(_contract_json_view(saved['contract']),value.get('contract')),'checkpoint bundle metadata mismatch')
    hashes=value.get('state_sha256',{})
    for name in ('policy','optimizer','torch_rng','cuda_rng'):
        _require(hashes.get(name)==_state_sha256(saved[name]),'checkpoint bundle state hash mismatch')
    _require(hashes.get('contract')==_state_sha256(saved['contract']),'checkpoint bundle contract hash mismatch')
    return value


def _validate_raw_evidence(cert,report,base):
    files=cert.get('evidence_files',{})
    required=('raw_reset_readback','initial_camera_geometry','initial_visual_actor_inputs','frozen_manifest',
              'conditional_telemetry_samples')
    _require(all(name in files for name in required),'complete native readback evidence files required')
    paths={name:_file(files[name],base) for name in required}
    for name in required[:3]:
        _require(cert.get(name+'_sha256')==files[name]['sha256'],'certificate evidence binding mismatch')
    frozen=validate_manifest(_read(paths['frozen_manifest']))
    metadata=frozen['metadata']; expected=frozen['trials']; count=len(expected)
    _require(metadata['split']=='validation' and _digest(frozen)==cert.get('manifest_sha256'),
        'certificate must bind the frozen validation manifest')
    for key in ('physics_sha256','camera_sha256'):
        _require(cert.get(key)==metadata[key],'native configuration provenance mismatch: '+key)
    _require(count==cert.get('expected_count')==report.get('expected_count'),
        'native evidence expected trial count mismatch')
    by_id={r['trial_id']:r for r in expected}
    readback=_read(paths['raw_reset_readback'])
    _require(readback.get('schema_version')=='dh116-evaluation-reset-readback-v1',
        'actual reset/readback schema required')
    rows=readback.get('actual_states')
    _require(isinstance(rows,list) and len(rows)==count,'incomplete actual reset/readback coverage')
    _require(readback.get('physics_steps_advanced')==0,'readback must precede first physics action')
    _require(readback.get('identity')=='actual_local_float32_state_not_placement_or_label',
        'actual native readback identity required; fixture evidence cannot select')
    tolerance=_finite(readback.get('writer_match_tolerance'),maximum=2e-5)
    _require(_finite(readback.get('max_abs_writer_readback_error'))<=tolerance,
        'actual writer/readback exceeds trusted matching tolerance')
    seen=set(); actual_hashes=set()
    for row in rows:
        trial=row.get('trial_id'); authored=by_id.get(trial)
        _require(authored is not None and trial not in seen,'duplicate or unknown readback trial')
        _require(row.get('grasp_style')==authored['grasp_style']
            and row.get('authored_state_sha256')==authored['state_sha256'],
            'actual readback does not bind authored trial')
        actual=state_sha256(row.get('physical_state'))
        _require(actual==row.get('readback_state_sha256') and actual not in actual_hashes,
            'duplicate or inconsistent physical readback state')
        # Compare actual payload rather than trusting the reported max error.
        def compare(a,b):
            _require(type(a) is type(b),'writer/readback state type changed')
            if isinstance(a,dict):
                _require(a.keys()==b.keys(),'writer/readback schema changed')
                for k in a:compare(a[k],b[k])
            elif isinstance(a,list):
                _require(len(a)==len(b),'writer/readback dimensions changed')
                for x,y in zip(a,b):compare(x,y)
            elif isinstance(a,(int,float)):
                _require(abs(a-b)<=tolerance,'actual writer/readback physical state mismatch')
            else:_require(a==b,'writer/readback joint names mismatch')
        compare(authored['physical_state'],row['physical_state'])
        seen.add(trial);actual_hashes.add(actual)
    _require(readback.get('unique_readback_count')==count,'actual readbacks are not independent')
    geometry=_read(paths['initial_camera_geometry'])
    _require(geometry.get('passed') is True and geometry.get('failed_env_ids')==[],
        'native initial camera geometry did not pass')
    geometry_rows=geometry.get('environments',[])
    _require(len(geometry_rows)==count and {r.get('env_id') for r in geometry_rows}==set(range(count)),
        'incomplete native camera geometry coverage')
    _require(_finite(geometry.get('tolerance_p95_m'),maximum=.003)>0,
        'camera geometry tolerance must preserve original 3mm audit')
    for row in geometry_rows:
        _require(type(row.get('point_count')) is int and row['point_count']>0,'missing rendered geometry points')
        _finite(row.get('p95_m'),maximum=.003)
    actor=torch.load(paths['initial_visual_actor_inputs'],map_location='cpu',weights_only=False)
    for name,shape in (('policy',(count,3844)),('previous_action',(count,24))):
        tensor=actor.get(name)
        _require(isinstance(tensor,torch.Tensor) and tuple(tensor.shape)==shape
            and tensor.dtype==torch.float32 and bool(torch.isfinite(tensor).all()),
            'native visual actor evidence requires finite 3844D sensor/24D action inputs')
    _require(not bool(actor['previous_action'].any()),'cold-reset previous actions must be cleared')
    history=actor.get('history')
    _require(isinstance(history,torch.Tensor) and history.dtype==torch.float32
        and tuple(history.shape)==(count,4,961) and bool(torch.isfinite(history).all())
        and torch.equal(history.flatten(1),actor['policy']) and not bool(history[:,:3].any()),
        'cold-reset sensor history must clear all previous frames and match actor inputs')
    frame=history[:,-1]
    _require(not bool(frame[:,934:958].any()) and bool((frame[:,-3]>0).all())
        and bool((frame[:,-2:]==1).all()),'initial actor frame requires cleared actions and valid camera timing')
    packet=actor.get('sensor_packet',{})
    for name,shape in (('image_valid',(count,)),('object_valid',(count,)),('point_valid',(count,128))):
        mask=packet.get(name)
        _require(isinstance(mask,torch.Tensor) and mask.dtype==torch.bool and tuple(mask.shape)==shape,
            'initial sensor evidence requires actual validity masks')
        _require(bool(mask.any(dim=-1).all()) if name=='point_valid' else bool(mask.all()),
            'invalid initial camera sensor evidence cannot select')
    points=packet.get('points')
    _require(isinstance(points,torch.Tensor) and points.dtype==torch.float32
        and tuple(points.shape)==(count,2,128,3) and bool(torch.isfinite(points).all()),
        'initial point-cloud sensor evidence required')
    masked=torch.where(packet['point_valid'][:,None,:,None],points,0.)
    _require(torch.equal(frame[:,:768],masked.flatten(1))
        and torch.equal(frame[:,768:896],packet['point_valid'].float()),
        'initial actor frame differs from raw sensor packet')
    contract=actor.get('contract',{})
    _require(contract.get('actor_observation_size')==3844 and contract.get('critic_observation_size')==371
        and contract.get('action_contract',{}).get('size')==24,'actor evidence contract dimensions mismatch')
    return frozen,paths


def _replay_conditional_samples(path,frozen,report):
    """Recompute outcomes and applicability from streamed native raw samples."""
    replay=FirstTerminalAggregator(frozen,checkpoint_sha256=report['checkpoint_sha256'],deterministic=True)
    totals={r['trial_id']:dict(trial_id=r['trial_id'],observed_duration_s=0.,
        slip_applicable_duration_s=0.,slip_valid_applicable_duration_s=0.,
        slip_inapplicable_duration_s=0.,slip_applicable_missing_samples=0,
        slip_applicability_missing_samples=0,slip_inapplicable_reason_counts={}) for r in frozen['trials']}
    count=0
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        header=json.loads(next(stream,'{}'))
        _require(header.get('schema_version')=='dh116-conditional-telemetry-samples-v1'
            and header.get('format')=='gzip_jsonl'
            and header.get('manifest_sha256')==_digest(frozen)
            and header.get('checkpoint_sha256')==report['checkpoint_sha256'],
            'streamed native conditional telemetry provenance required')
        for line in stream:
            raw=json.loads(line);sample=raw.get('sample');trial=raw.get('trial_id')
            _require(trial in totals and isinstance(sample,dict),'unknown native telemetry trial or sample')
            _require(all(isinstance(sample.get(name),list) and len(sample[name])==size
                for name,size in (('estimated_torque_nm',12),('effort_limit_nm',12),('action',24))),
                'native raw sample requires complete 12D torque/limits and 24D action')
            dt=_finite(sample.get('dt_s'));_require(dt>0,'actual positive physical clock required')
            applicable=raw.get('slip_applicable');valid=raw.get('slip_valid');reason=raw.get('slip_inapplicable_reason')
            latched=raw.get('prior_anchor_latched')
            _require(type(applicable) is bool and type(valid) is bool,
                'slip applicability must come from actual anchor/contact evidence, never a missing-value inference')
            _require(type(latched) is bool,'actual pre-consumption fixed-anchor latch evidence required')
            # The native stream contains live first-terminal-consumed samples.
            # Only a trustworthy false contact can release a known anchor from
            # applicability; invalid/missing measurements must fail closed.
            measured=sample.get('measurement_valid')
            contact=sample.get('bilateral_opposed_contact')
            derived_applicable=latched and (measured is not True or contact is not False)
            _require(applicable==derived_applicable,
                'slip applicability disagrees with raw anchor/contact evidence')
            _require(sample.get('measurement_valid') is True,'invalid native telemetry cannot select')
            _require(valid==(sample.get('slip_m') is not None),'slip measurement validity disagrees with raw evidence')
            total=totals[trial];total['observed_duration_s']+=dt
            if applicable:
                _require(latched and valid and reason is None,'applicable fixed-anchor slip measurement is missing')
                total['slip_applicable_duration_s']+=dt
                total['slip_valid_applicable_duration_s']+=dt
            else:
                _require(not valid and reason==('no_contact' if latched else 'no_anchor'),
                    'inapplicable slip requires explicit native no-anchor/contact reason')
                total['slip_inapplicable_duration_s']+=dt
                reasons=total['slip_inapplicable_reason_counts'];reasons[reason]=reasons.get(reason,0)+1
            _require(replay.observe(raw.get('env_id'),trial,sample,
                terminal=raw.get('terminal'),native_success=raw.get('native_success')),
                'raw telemetry contains a repeated terminal episode')
            count+=1
    expected=replay.summary()
    for key,value in expected.items():
        _require(_same(report.get(key),value),'native report differs from replayed raw first-terminal telemetry: '+key)
    conditional=report.get('conditional_telemetry',{})
    _require(conditional.get('raw_sample_count')==count,'conditional raw telemetry count mismatch')
    rows=conditional.get('per_trial')
    _require(isinstance(rows,list) and len(rows)==len(totals),'incomplete conditional telemetry trial evidence')
    seen=set()
    for row in rows:
        trial=row.get('trial_id');_require(trial in totals and trial not in seen,'duplicate conditional trial evidence')
        for key,value in totals[trial].items():
            if isinstance(value,float):
                actual=_finite(row.get(key));_require(math.isclose(actual,value,rel_tol=1e-9,abs_tol=1e-9),
                    'conditional telemetry duration mismatch: '+key)
            else:_require(_same(row.get(key),value),'conditional native applicability mismatch: '+key)
        seen.add(trial)
    return totals


def _validate_summary(report,by_id):
    _require(report.get('schema_version')=='dh116-first-terminal-summary-v1'
        and report.get('split')=='validation' and report.get('deterministic') is True
        and report.get('policy_mode')=='deterministic'
        and report.get('reset_kind')=='cold_authored_reset'
        and report.get('sample_unit')=='pre_reset_physical_substep',
        'deterministic first-terminal cold-reset validation report required')
    rows=report.get('trial_rows');count=len(by_id)
    _require(report.get('conditional_telemetry_complete') is True
        and report.get('completed_count')==count and report.get('incomplete_count')==0
        and report.get('invalid_measurement_count')==0 and isinstance(rows,list) and len(rows)==count,
        'incomplete native validation cannot select a model')
    seen=set()
    for row in rows:
        trial=row.get('trial_id'); authored=by_id.get(trial)
        _require(authored is not None and trial not in seen,'duplicate or unknown validation trial')
        _require(row.get('state_sha256')==authored['state_sha256']
            and row.get('grasp_style')==authored['grasp_style'],'validation physical identity mismatch')
        _require(row.get('status')=='completed'
            and type(row.get('native_success')) is bool and row.get('invalid_measurement_samples')==0,
            'incomplete or invalid first-terminal telemetry')
        for metric in (m for m in _METRICS if m!='slip'):
            _require(row.get(metric+'_valid_coverage')==1.,'required validation metric lacks full coverage')
        for key in ('peak_lift_m','max_qualified_hold_s','contact_alive_duration_s',
                    'observed_duration_s','action_delta_squared_sum','action_max_abs_delta'):
            _finite(row.get(key),minimum=None if key=='peak_lift_m' else 0.)
        for key in ('sample_count','action_delta_elements','action_delta_samples','action_samples','torque_valid_samples'):
            _require(type(row.get(key)) is int and row[key]>0,'missing validation sample counts')
        _require(not row['native_success'] or row.get('max_slip_m') is not None,
            'a native-success trial requires actual observable fixed-anchor slip evidence')
        seen.add(trial)
    recomputed={s:_statistics([r for r in rows if r['grasp_style']==s]) for s in GRASP_STYLES}
    _require(_same(report.get('styles'),recomputed) and _same(report.get('aggregate'),_statistics(rows)),
        'validation summary does not match raw first-terminal rows')
    for style in GRASP_STYLES:
        stats=recomputed[style]
        _require(stats['completed_count']>=100 and stats['incomplete_count']==0,
            'each grasp style requires 100 completed independent validation trials')
        for key in ('native_success_rate','hold_5s_rate','height_30mm_rate'):
            _finite(stats.get(key),maximum=1.)
        for key in ('P50','P90','P95'):
            _finite(stats.get('lift_quantiles_m',{}).get(key),minimum=None)
        if stats.get('max_slip_m') is not None:_finite(stats['max_slip_m'])
        _finite(stats.get('action_stability',{}).get('delta_rms'))
        stats['slip_observed_trial_count']=sum(r['max_slip_m'] is not None for r in rows if r['grasp_style']==style)
    return recomputed


def _score(styles):
    mean=lambda f:sum(f(styles[s]) for s in GRASP_STYLES)/len(GRASP_STYLES)
    known=all(styles[s]['slip_observed_trial_count']==styles[s]['completed_count'] for s in GRASP_STYLES)
    return [mean(lambda s:s['native_success_rate']),mean(lambda s:s['hold_5s_rate']),
        mean(lambda s:s['height_30mm_rate']),int(known),-mean(lambda s:s['max_slip_m']) if known else 0.,
        *[mean(lambda s:s['lift_quantiles_m'][q]) for q in ('P50','P90','P95')],
        -mean(lambda s:s['action_stability']['delta_rms'])]


def _exclusive_json(path,value):
    payload=_json_bytes(value)
    try:
        with path.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
    except FileExistsError:
        if path.is_symlink() or path.read_bytes()!=payload:raise


def _replace_json(path,value):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(_json_bytes(value));stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def select_validation_candidate(bundle_manifest_path,report_path,certificate_path, *,
                                trusted_certificate_sha256,output_dir):
    """Record a certified validation candidate and atomically update pointers.

    ``trusted_certificate_sha256`` comes from the native controller/trusted
    evidence registry, never from an untrusted report field. All candidates and
    full checkpoint bundles survive selection. Zero native successes create
    only best_validation, never best_success_rate or a renamed model.
    """
    bundle_path,report_path,cert_path=(Path(p).absolute() for p in
        (bundle_manifest_path,report_path,certificate_path))
    _sha256(trusted_certificate_sha256,'trusted native certificate SHA256')
    _require(file_sha256(cert_path)==trusted_certificate_sha256,'native certificate is not externally trusted')
    bundle_digest=file_sha256(bundle_path)
    cert=_read(cert_path);report=_read(report_path);bundle=_bundle(bundle_path)
    _require(cert.get('schema_version')==CERTIFICATE_VERSION
        and cert.get('execution')=='native_isaac_tensor_world'
        and cert.get('attestation_status')=='native_verified'
        and cert.get('native_execution_complete') is True
        and cert.get('conditional_telemetry_complete') is True,'trusted real native-run certificate required')
    _require(cert.get('split')=='validation' and report.get('split')=='validation',
        'test data cannot select validation models')
    _require(cert.get('report_sha256')==file_sha256(report_path),'native report SHA256 mismatch')
    _require(cert.get('checkpoint_sha256')==report.get('checkpoint_sha256')==bundle['checkpoint_sha256']
        and cert.get('manifest_sha256')==report.get('manifest_sha256'),
        'native checkpoint/manifest provenance mismatch')
    _sha256(cert.get('runner_source_sha256'),'native runner source SHA256')
    command=cert.get('command')
    _require(isinstance(command,list) and len(command)>=2
        and all(isinstance(v,str) and v for v in command),'actual native command provenance required')
    frozen,paths=_validate_raw_evidence(cert,report,cert_path.parent)
    conditional=_replay_conditional_samples(paths['conditional_telemetry_samples'],frozen,report)
    by_id={r['trial_id']:r for r in frozen['trials']}
    styles=_validate_summary(report,by_id);score=_score(styles)
    root=Path(output_dir).absolute()
    candidate=root/('candidate_'+bundle['checkpoint_sha256']+'_'+cert['report_sha256']+'.json')
    record=dict(version=VERSION,split='validation',candidate_path=str(candidate),
        checkpoint_path=bundle['checkpoint_path'],checkpoint_sha256=bundle['checkpoint_sha256'],
        bundle_manifest_path=str(bundle_path),bundle_manifest_sha256=bundle_digest,
        report_path=str(report_path),report_sha256=cert['report_sha256'],
        certificate_path=str(cert_path),certificate_sha256=trusted_certificate_sha256,
        frozen_validation_manifest_sha256=cert['manifest_sha256'],
        native_success_macro_rate=score[0],score=score,styles=deepcopy(styles),
        native_execution_complete=True,conditional_telemetry_complete=True,
        validation_status='native_verified_by_external_certificate_and_replayed_evidence',
        strict_telemetry_complete=report['telemetry_complete'],
        slip_ranking_policy='unknown_is_worst_after_success_hold_height',slip_metric_available=bool(score[3]),
        conditional_telemetry=deepcopy(report['conditional_telemetry']),
        selection_order=['native_success_macro_rate','hold_5s_macro_rate','height_30mm_macro_rate',
            'complete_slip_trial_observability','negative_macro_max_slip_m','macro_lift_p50_m','macro_lift_p90_m','macro_lift_p95_m',
            'negative_macro_action_delta_rms'],
        evidence_sha256={k:v['sha256'] for k,v in cert['evidence_files'].items()})
    _json_bytes(record)
    # Recheck mutable external inputs after parsing before writing any record.
    _require(file_sha256(cert_path)==trusted_certificate_sha256
        and file_sha256(report_path)==cert['report_sha256']
        and file_sha256(bundle_path)==record['bundle_manifest_sha256']
        and file_sha256(bundle['checkpoint_path'])==bundle['checkpoint_sha256'],
        'validation evidence changed during selection')
    for name,path in paths.items():
        _require(file_sha256(path)==cert['evidence_files'][name]['sha256'],
            'native raw evidence changed during selection')
    root.mkdir(parents=True,exist_ok=True);lock=root/'.selection.lock';owned=False
    try:
        with lock.open('x') as stream:owned=True;stream.write('exclusive validation selection\n')
        scope=root/'validation_scope.json'
        if scope.exists():
            _require(_read(scope).get('manifest_sha256')==cert['manifest_sha256'],
                'candidates require the same frozen validation manifest')
        else:_exclusive_json(scope,dict(version=VERSION,manifest_sha256=cert['manifest_sha256']))
        _exclusive_json(candidate,record)
        pointer=dict(version=VERSION,candidate_path=str(candidate),candidate_sha256=file_sha256(candidate),
            checkpoint_path=bundle['checkpoint_path'],checkpoint_sha256=bundle['checkpoint_sha256'],
            frozen_validation_manifest_sha256=cert['manifest_sha256'],score=score)
        for name,eligible in (('best_validation',True),('best_success_rate',score[0]>0)):
            if not eligible:continue
            destination=root/(name+'.json')
            if destination.exists():
                old=_read(destination)
                _require(old.get('frozen_validation_manifest_sha256')==cert['manifest_sha256'],
                    'best pointer belongs to a different frozen validation manifest')
                _require(file_sha256(old['candidate_path'])==old.get('candidate_sha256'),
                    'immutable selected candidate SHA256 mismatch')
                if tuple(old['score'])>=tuple(score):continue
            _replace_json(destination,pointer)
    finally:
        if owned:lock.unlink(missing_ok=True)
    return record
