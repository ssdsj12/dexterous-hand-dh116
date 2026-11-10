"""Native promotion gates reopen SHA-bound raw evidence; schemas are not physics."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .tabletop_stereo_contract import ROOT, STYLES, action_sha256, digest, require_sha, state_contract, physical_initial_sha256
from .tabletop_stereo_evaluation import VERSION, THRESHOLDS, analyze_episode, aggregate_episodes, training_partition


def _read_object(value):
    if isinstance(value,dict):return copy.deepcopy(value),Path.cwd()
    if isinstance(value,(str,Path)):
        path=Path(value).resolve()
        try:result=json.loads(path.read_text(encoding='utf-8'))
        except (OSError,ValueError) as error:raise ValueError('qualification JSON unavailable') from error
        if not isinstance(result,dict):raise ValueError('qualification dictionary required')
        return result,path.parent
    raise ValueError('qualification must be a dictionary or JSON path')


def _artifact(binding,base,*,allow_empty=False):
    if not isinstance(binding,dict):raise ValueError('actual artifact path/SHA binding required')
    try:
        require_sha(binding.get('sha256'))
        path=Path(binding['path']);path=path if path.is_absolute() else base/path
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=binding['sha256']:raise ValueError('raw artifact SHA mismatch')
    except (OSError,KeyError,TypeError) as error:raise ValueError('raw artifact missing or invalid') from error
    if not raw and not allow_empty:raise ValueError('empty raw artifact is not evidence')
    return raw,path.resolve()


def _jsonl(raw):
    def nonfinite(value):raise ValueError('nonfinite JSON raw value must be explicit NA/null: '+value)
    try:rows=[json.loads(line,parse_constant=nonfinite) for line in raw.decode('utf-8').splitlines() if line.strip()]
    except (UnicodeError,ValueError) as error:raise ValueError('actual raw JSONL evidence required') from error
    if not rows or any(not isinstance(row,dict) for row in rows):raise ValueError('raw evidence record dictionaries required')
    return rows


def _process_output(proof,base,binding,*,binding_base=None):
    """A child raw output must already be frozen by the external observer."""
    raw,path=_artifact(binding,base if binding_base is None else binding_base)
    completion_raw,_=_artifact(proof.get('completion_artifact'),base)
    outputs=json.loads(completion_raw).get('produced_artifacts')
    if not isinstance(outputs,list) or not any(isinstance(row,dict) and Path(row.get('path','')).resolve()==path
            and row.get('sha256')==binding['sha256'] for row in outputs):
        raise ValueError('raw output artifact is absent from actual external process completion')
    return raw,path


def _native_run(proof,base):
    if proof.get('source_scope')!='native' or proof.get('schema_only') is not False:
        raise ValueError('native source scope required; CPU/schema fixture cannot qualify')
    if type(proof.get('process_exit_code')) is not int or proof['process_exit_code']!=0:
        raise ValueError('completed exit0 native process required')
    completion_raw,completion_path=_artifact(proof.get('completion_artifact'),base)
    completion=json.loads(completion_raw)
    if (completion.get('version')!='dh116-external-native-completion-v1'
            or completion.get('source_scope')!='native_process_exit_observed'
            or type(completion.get('returncode')) is not int or completion['returncode']!=0
            or completion.get('source_changed_during_run') is not False):
        raise ValueError('actual external native process exit0 evidence required')
    _,runner=_artifact(completion.get('runner_source_artifact'),completion_path.parent)
    allowed_runners={ROOT/'src/go2_pvcnn/control/dh116/tabletop_stereo_resolution_benchmark.py',
                     ROOT/'src/go2_pvcnn/control/dh116/tabletop_stereo_native.py'}
    if runner not in {p.resolve() for p in allowed_runners}:raise ValueError('unrecognized actual external process observer')
    command=completion.get('command')
    if not isinstance(command,list) or not all(isinstance(x,str) for x in command) or not {'--enable-experiment','--execute-native'}<=set(command):
        raise ValueError('actual opt-in native command identity required')
    for name in ('stdout_artifact','stderr_artifact'):_artifact(completion.get(name),completion_path.parent,allow_empty=True)
    target=proof.get('capture_artifact') if proof.get('version')=='dh116-native-stereo-proof-v1' else proof.get('raw_artifact')
    _process_output(proof,base,target)
    if proof.get('action_contract_sha256')!=action_sha256() or proof.get('state_contract_sha256')!=digest(state_contract()):
        raise ValueError('native source proof current action/state identity mismatch')
    producer=proof.get('producer')
    if not isinstance(producer,dict):raise ValueError('actual producer source/config/kwargs identity required')
    _,source=_artifact(producer.get('source_artifact'),base)
    # Only the explicit native capture entry point is currently admitted. Future
    # source auditors need their own reviewed producer version, not a renamed file.
    stereo=proof.get('version')=='dh116-native-stereo-proof-v1'
    filename='tabletop_stereo_camera_cli.py' if stereo else 'tabletop_stereo_native.py'
    expected=ROOT/'src/go2_pvcnn/control/dh116'/filename
    if source!=expected.resolve():raise ValueError('unrecognized native evidence producer; arbitrary file is insufficient')
    config_raw,_=_process_output(proof,base,producer.get('config_artifact'))
    try:config=json.loads(config_raw)
    except ValueError as error:raise ValueError('native producer config must be real JSON') from error
    kwargs=producer.get('kwargs')
    if (not isinstance(kwargs,dict) or kwargs.get('physics_hz')!=480 or kwargs.get('control_hz')!=60
            or producer.get('kwargs_sha256')!=digest(kwargs)):
        raise ValueError('native 480Hz/60Hz producer kwargs contract mismatch')
    if not isinstance(config,dict):raise ValueError('native source config dictionary required')
    entrypoint='tabletop_stereo_camera_cli.main.callback' if stereo else 'NativeTensorBridge._capture'
    if config.get('resolved_kwargs')!=kwargs or producer.get('entrypoint')!=entrypoint:
        raise ValueError('producer entrypoint/resolved kwargs must match the actual config artifact')
    return producer


def _binding_proof(binding,version):
    if not isinstance(binding,dict) or not isinstance(binding.get('proof_path'),str):
        raise ValueError('actual immutable native proof_path/proof_sha256 required')
    raw,path=_artifact(dict(path=binding['proof_path'],sha256=binding.get('proof_sha256')),Path.cwd())
    try:proof=json.loads(raw)
    except ValueError as error:raise ValueError('native source proof JSON required') from error
    if not isinstance(proof,dict) or proof.get('version')!=version:raise ValueError('unsupported native source proof version')
    _native_run(proof,path.parent)
    return proof,path.parent


def _verify_force_qualification(binding):
    """No force loss/reward is authorized by a native_qualified string alone.

    A source proof must include independent raw per-contact object-side vectors,
    resolved partners and complete external-contact/force/attachment audit. The
    existing NativeTensorBridge emits unqualified telemetry and cannot satisfy
    this protocol yet. Missing source capabilities remain NA.
    """
    proof,base=_binding_proof(binding,'dh116-force-source-proof-v1')
    required=dict(force_side='on_object',units='N',coordinate_frame='world',
                  partner_attribution='resolved_raw_contact_body_ids')
    if any(proof.get(k)!=v for k,v in required.items()):raise ValueError('force side/SI/frame/partner source not qualified')
    completeness=proof.get('source_completeness',{})
    for key in ('object_filtered_contacts','tangential_reports','external_object_contacts','authored_force_and_attachment_audit'):
        if completeness.get(key) is not True:raise ValueError('force source completeness missing: '+key)
    audits=proof.get('independent_audit_artifacts')
    if not isinstance(audits,list) or len(audits)<2:raise ValueError('independent raw contact and force/attachment audit artifacts required')
    for audit in audits:_artifact(audit,base)
    raw,_=_artifact(proof.get('raw_artifact'),base);rows=_jsonl(raw)
    if type(proof.get('sample_count')) is not int or proof['sample_count']!=len(rows):raise ValueError('force raw sample_count mismatch')
    objects=proof.get('object_body_ids')
    if not isinstance(objects,list) or not objects or any(not isinstance(v,str) or not v for v in objects):
        raise ValueError('resolved actual object body IDs required')
    mapping=proof.get('per_digit_source')
    if (not isinstance(mapping,list) or len(mapping)!=2 or any(not isinstance(side,list) or len(side)!=5
            or any(not isinstance(value,str) or not value for value in side) for side in mapping)):
        raise ValueError('independent 2x5 per-digit source mapping required')
    if len({value for side in mapping for value in side})!=10:raise ValueError('whole-hand sensor source cannot be broadcast to digits')
    body_map=proof.get('body_id_to_prim_path')
    if not isinstance(body_map,dict) or not body_map:raise ValueError('actual SDK body-ID readback map required')
    if proof.get('raw_api_force_side')!='on_hand':raise ValueError('explicit raw API force side conversion evidence required')
    previous={};coverage=[];observed_hands=set()
    for row in rows:
        if row.get('kind')!='native_contact_source_sample' or row.get('source_scope')!='native':raise ValueError('native raw contact source samples required')
        key=tuple(row.get(k) for k in ('run_uuid','env_id','episode_uid','reset_id'))
        tick=row.get('physics_step');timestamp=row.get('timestamp')
        if type(tick) is not int or type(timestamp) not in (int,float) or not math.isfinite(timestamp):raise ValueError('force source clock unavailable')
        if key in previous and (tick!=previous[key][0]+1 or abs(timestamp-previous[key][1]-1/480)>1e-7):raise ValueError('force source 480Hz clock hole')
        previous[key]=(tick,timestamp)
        contacts=row.get('contacts')
        if not isinstance(contacts,list):raise ValueError('actual raw contacts required')
        offsets=set()
        for contact in contacts:
            if (not isinstance(contact,dict) or type(contact.get('hand')) is not int or contact['hand'] not in (0,1)
                    or type(contact.get('digit')) is not int or not 0<=contact['digit']<5
                    or contact.get('partner_body_id') not in objects
                    or type(contact.get('source_buffer_offset')) is not int or contact['source_buffer_offset']<0):
                raise ValueError('unresolved per-contact owner/digit/object partner')
            sensor=contact.get('source_sensor_path')
            if sensor!=mapping[contact['hand']][contact['digit']]:raise ValueError('contact sensor path does not match independent digit source')
            offset=(sensor,contact['source_buffer_offset'])
            if offset in offsets:raise ValueError('raw contact slot cannot be broadcast across digits')
            offsets.add(offset);observed_hands.add(contact['hand'])
            if body_map.get(str(contact.get('raw_partner_body_id')))!=contact['partner_body_id']:
                raise ValueError('raw partner body ID does not resolve to actual object prim')
            for name in ('normal_force_on_object_w','friction_force_on_object_w','position_w','normal_force_on_hand_w','friction_force_on_hand_w'):
                value=np.asarray(contact.get(name),dtype=float)
                if value.shape!=(3,) or not np.isfinite(value).all():raise ValueError('missing actual contact vector: '+name)
            for prefix in ('normal_force','friction_force'):
                if not np.allclose(contact[prefix+'_on_object_w'],-np.asarray(contact[prefix+'_on_hand_w']),rtol=1e-7,atol=1e-9):
                    raise ValueError('object-side force sign conversion does not match original API vectors')
        for name in ('external_support_force_n','external_upward_force_n'):
            if type(row.get(name)) not in (int,float) or not math.isfinite(row[name]):raise ValueError('external force source unavailable')
        if type(row.get('attachment_count')) is not int or row['attachment_count']<0:raise ValueError('attachment audit unavailable')
        coverage.append({key:row[key] for key in ('run_uuid','env_id','episode_uid','reset_id','physics_step','timestamp',
                                                'external_support_force_n','external_upward_force_n','attachment_count')})
    if observed_hands!={0,1}:raise ValueError('both native hand source channels need actual contact coverage')
    return dict(qualified=True,source_scope='native',raw_sha256=proof['raw_artifact']['sha256'],
                proof_sha256=binding['proof_sha256'],source_sample_count=len(rows),source_contract=required,
                sample_coverage=coverage)


def verify_force_qualification(binding):
    try:return _verify_force_qualification(binding)
    except (OSError,KeyError,TypeError,UnicodeError,OverflowError) as error:
        raise ValueError('native force evidence unavailable or malformed') from error


def _verify_episode(proof,*,expected_identity=None,subject=None,require_success=True):
    p,base=_read_object(proof)
    if p.get('version')!='dh116-native-episode-proof-v1':raise ValueError('versioned native episode proof required')
    _native_run(p,base)
    if expected_identity is not None and p.get('model_identity')!=expected_identity:raise ValueError('native episode model/contract identity mismatch')
    if p.get('action_contract_sha256')!=action_sha256() or p.get('state_contract_sha256')!=digest(state_contract()):
        raise ValueError('native episode current action/state contract mismatch')
    raw,_=_artifact(p.get('raw_artifact'),base);rows=_jsonl(raw)
    if len({tuple(row.get(k) for k in ('run_uuid','env_id','episode_uid','reset_id')) for row in rows})!=1:
        raise ValueError('native episode proof cannot splice environments or episodes')
    result=analyze_episode(rows)
    header=rows[0]
    if (header.get('kind')!='episode_start' or header.get('initial_measurement_valid') is not True
            or header.get('physical_initial_condition_source')!='actual_sdk_readback'):
        raise ValueError('actual SDK initial readback evidence required; seed/config is insufficient')
    if p.get('sample_count')!=result['sample_count']:raise ValueError('native episode raw sample_count mismatch')
    if subject is not None:
        expected={'owner':subject} if isinstance(subject,str) else subject
        if not isinstance(expected,dict):raise ValueError('native proof subject must be owner or binding dictionary')
        if any((result.get(key,p.get(key)))!=value for key,value in expected.items()):raise ValueError('native proof subject/checkpoint mismatch')
    force=p.get('force_qualification')
    if not isinstance(force,dict):raise ValueError('native force/support/external-source qualification remains NA')
    # Resolve relative immutable proof links against their containing manifest.
    force=copy.deepcopy(force)
    if isinstance(force.get('proof_path'),str) and not Path(force['proof_path']).is_absolute():force['proof_path']=str(base/force['proof_path'])
    verified_force=verify_force_qualification(force)
    fields=('run_uuid','env_id','episode_uid','reset_id','physics_step')
    coverage={tuple(row[k] for k in fields):row for row in verified_force['sample_coverage']}
    for sample in [row for row in rows if row.get('kind','sample')=='sample' and row.get('live',True)]:
        source=coverage.get(tuple(sample[k] for k in fields))
        if source is None or any(sample.get(key)!=source.get(key) for key in ('timestamp','external_support_force_n','external_upward_force_n','attachment_count')):
            raise ValueError('native force evidence is not aligned to this exact episode/480Hz sample')
    if require_success:
        if result['reasons']:raise ValueError('native successful episode not qualified: '+','.join(result['reasons']))
    else:
        allowed={'native_task_not_successful','less_than_5s_480hz_raw_hold_evidence','formal_hold_below_5s_or_NA','raw_hold_lift_formal_slip_contact_not_qualified'}
        failures=[reason for reason in result['reasons'] if reason not in allowed]
        if failures:raise ValueError('native completed episode protocol invalid: '+','.join(failures))
    result.update(native_evidence_verified=True,qualified_success=result['native_task_success'] and not result['reasons'],
                  proof_sha256=digest(p),checkpoint_sha256=p.get('checkpoint_sha256'),force_binding=verified_force,
                  runtime=copy.deepcopy(header.get('runtime')),
                  actor_runtime_samples_verified=(isinstance(header.get('runtime'),dict) and all(
                      row.get('runtime')==header['runtime'] for row in rows if row.get('kind','sample')=='sample' and row.get('live',True))))
    return result


def verify_native_episode(proof,*,expected_identity=None,subject=None):
    """Validate a complete first successful native episode; return its bound UID."""
    try:return _verify_episode(proof,expected_identity=expected_identity,subject=subject)
    except (OSError,KeyError,TypeError,UnicodeError) as error:raise ValueError('native episode evidence unavailable or malformed') from error


def _report_episodes(report):
    r,base=_read_object(report)
    if r.get('version')!=VERSION or r.get('source_scope')!='native':raise ValueError('native evaluation report required; schema/CPU cannot qualify')
    if r.get('action_contract_sha256')!=action_sha256() or r.get('state_contract_sha256')!=digest(state_contract()):
        raise ValueError('evaluation current state/action contract mismatch')
    if r.get('thresholds')!=THRESHOLDS:raise ValueError('unchanged formal thresholds required')
    proofs=r.get('episode_proofs')
    if not isinstance(proofs,list) or not proofs:raise ValueError('actual SHA-bound per-episode raw proofs required')
    episodes=[]
    for proof in proofs:
        if isinstance(proof,str) and not Path(proof).is_absolute():proof=str(base/proof)
        episodes.append(_verify_episode(proof,require_success=False))
    ids=[tuple(e[k] for k in ('run_uuid','env_id','episode_uid','reset_id')) for e in episodes]
    if len(set(ids))!=len(ids):raise ValueError('duplicate native episode evidence')
    return r,episodes,base


def qualified_episode_identities(report):
    _,episodes,_=_report_episodes(report)
    return {tuple(e[k] for k in ('run_uuid','env_id','episode_uid','reset_id')) for e in episodes if e['qualified_success']}


def require_teacher_qualification(report,checkpoint_sha256,styles=STYLES):
    require_sha(checkpoint_sha256,'Teacher checkpoint SHA')
    r,episodes,base=_report_episodes(report)
    if r.get('owner')!='teacher' or r.get('checkpoint_sha256')!=checkpoint_sha256:raise ValueError('qualified Teacher checkpoint identity mismatch')
    if any(e['owner']!='teacher' or e.get('checkpoint_sha256')!=checkpoint_sha256 for e in episodes):
        raise ValueError('MPC/BC/Student outcomes cannot qualify this Teacher checkpoint')
    artifacts=r.get('evidence_artifacts',[])
    if not isinstance(artifacts,list):raise ValueError('original artifact bindings required')
    kinds=set()
    for item in artifacts:
        _artifact(item,base);kinds.add(item.get('kind'))
        if item.get('kind')=='checkpoint' and item['sha256']!=checkpoint_sha256:raise ValueError('actual model checkpoint artifact mismatch')
    if not {'checkpoint','source','config'}<=kinds:raise ValueError('actual model/source/config artifacts required')
    for style in styles:
        subset=[e for e in episodes if e['grasp_style']==style]
        independent={e['physical_initial_condition_sha256'] for e in subset}
        independent.discard(None)
        if len(independent)<100:raise ValueError('at least100 independent measured native initial conditions required per style')
        if not any(e['qualified_success'] for e in subset):raise ValueError('no qualified complete native Teacher success for style '+style)
    return dict(qualified=True,checkpoint_sha256=checkpoint_sha256,action_contract_sha256=action_sha256(),
                state_contract_sha256=digest(state_contract()),styles=list(styles),evidence_artifacts=artifacts,
                report_sha256=digest(r),episode_identities=sorted(tuple(e[k] for k in ('run_uuid','env_id','episode_uid','reset_id'))
                    for e in episodes if e['qualified_success']))


def stereo_metric_gate(metrics,limits):
    """Pure threshold arithmetic, never an evidence/qualification assertion."""
    keys=('mae_m','rmse_m','coverage','count')
    return (all(type(metrics.get(k)) in (int,float) and math.isfinite(metrics[k]) for k in keys)
            and metrics['count']>=limits['min_valid_pixels'] and metrics['coverage']>=limits['min_coverage']
            and 0<=metrics['mae_m']<=limits['max_mae_m'] and 0<=metrics['rmse_m']<=limits['max_rmse_m'])


def _verified_wire_mask(reference,base,shape):
    """Re-resolve each env's archived actual instance IDs against stage paths."""
    import torch
    from .tabletop_stereo_reference import resolve_wire_instance_mask,AUDITED_CAMERA_VIEW_SHA256,AUDITED_INSTANCE_ANNOTATOR_SHA256
    wire=reference.get('wire_reference_mask')
    if wire is None:return None
    if reference.get('wire_mask_source_scope')!='native_resolved_instance_id':
        raise ValueError('wire completeness requires native resolved instance-ID readback')
    _,path=_artifact(reference.get('wire_mask_source_artifact'),base)
    raw=torch.load(path,map_location='cpu',weights_only=False)
    if (not isinstance(raw,dict) or raw.get('version')!='dh116-wire-instance-reference-v2'
            or raw.get('source_scope')!='native' or raw.get('validation_only') is not True
            or raw.get('actual_wire_path_source')!='RigidObject.root_physx_view.prim_paths'
            or any(raw.get(k)!=reference.get(k) for k in ('frame_id','simulation_time','calibration_sha256','render_timing'))):
        raise ValueError('native instance reference source/alignment mismatch')
    sdk=raw.get('sdk_camera_view_source_artifact');_artifact(sdk,path.parent)
    if sdk.get('sha256')!=AUDITED_CAMERA_VIEW_SHA256:raise ValueError('instance-ID SDK source is not audited')
    annotator=raw.get('sdk_instance_annotator_source_artifact');_artifact(annotator,path.parent)
    if annotator.get('sha256')!=AUDITED_INSTANCE_ANNOTATOR_SHA256:raise ValueError('renderer-instance annotator mapping source is not audited')
    ids=raw.get('raw_instance_ids');paths=raw.get('stage_validated_paths')
    if not isinstance(ids,torch.Tensor) or ids.shape!=shape or not isinstance(paths,list):
        raise ValueError('raw instance IDs/actual stage validation readback unavailable')
    optical=raw.get('reference_valid')
    if not isinstance(optical,torch.Tensor) or not torch.equal(optical,reference['valid']):
        raise ValueError('instance unresolved-pixel eligibility differs from actual optical-Z validity')
    if raw.get('wire_render_prim_paths') is None:raise ValueError('actual Wire render-geometry inventory unavailable')
    resolved=resolve_wire_instance_mask(ids,raw.get('raw_instance_info'),raw.get('wire_prim_paths'),
        path_validator=lambda value:value in paths,wire_render_prim_paths=raw['wire_render_prim_paths'],reference_valid=optical)
    truth=resolved['mask']
    if truth is None:raise ValueError('actual instance map cannot resolve per-environment Wire roots: '+repr(resolved['blocked_reasons']))
    if not isinstance(wire,torch.Tensor) or wire.shape!=truth.shape or wire.dtype!=torch.bool or not torch.equal(wire,truth):
        raise ValueError('wire mask differs from archived per-environment resolved raw instance IDs')
    return truth


def _measure_stereo_evidence(proof):
    """Recompute RGB stereo from native captures and separate reference depth."""
    import torch
    from .tabletop_stereo_calibration import StereoCalibration
    from .tabletop_stereo_depth import StereoDepth
    from .tabletop_stereo_validation import validate_identity
    p,base=_read_object(proof)
    if p.get('version')!='dh116-native-stereo-proof-v1':raise ValueError('versioned native stereo proof required')
    _native_run(p,base)
    _,calibration_path=_artifact(p.get('calibration_artifact'),base)
    calibration=StereoCalibration.from_json(calibration_path)
    _,validation_path=_process_output(p,base,p.get('validation_config_artifact'))
    try:validation=json.loads(validation_path.read_text(encoding='utf-8'))
    except (OSError,ValueError) as error:raise ValueError('stereo validation config unavailable') from error
    if validation.get('version')!='dh116-stereo-validation-v1':raise ValueError('versioned stereo thresholds required')
    limits=validation.get('stereo_limits',{})
    policy=json.loads((ROOT/'configs/dh116_training/stereo_validation_v1.json').read_text(encoding='utf-8'))
    expected=policy['stereo_limits']
    policy_compatible=limits==expected
    if validation.get('depth_estimator',{})!=p['producer']['kwargs'].get('depth_estimator',{}):
        raise ValueError('measurement estimator profile differs from the actual producer kwargs')
    _,capture_path=_artifact(p.get('capture_artifact'),base)
    try:captured=torch.load(capture_path,map_location='cpu',weights_only=False)
    except Exception as error:raise ValueError('native RGB capture cannot be read') from error
    if not isinstance(captured,dict) or captured.get('version')!='dh116-stereo-rgb-probe-v1':raise ValueError('actual native RGB probe capture required')
    frames=captured.get('frames');origins=captured.get('environment_origins')
    references=p.get('reference_artifacts')
    if (not isinstance(frames,list) or not frames or not isinstance(references,list)
            or len(references)!=len(frames) or p.get('sample_count')!=len(frames) or origins is None):
        raise ValueError('complete same-frame RGB/reference and native environment-origin readback required')
    errors=[];valid_pixels=0;reference_pixels=0;last=-1;last_time=-1.
    candidate_total=0;candidate_reconstructed=0;wire_total=0;wire_reconstructed=0;wire_unavailable=[];wire_frames=0;wire_errors=[]
    estimator=StereoDepth(**validation.get('depth_estimator',{}))
    for packet,reference_binding in zip(frames,references):
        if packet.get('native_evidence') is not True or packet.get('depth_source')!='computed_from_left_right_RGB':
            raise ValueError('native RGB-only stereo acquisition source required')
        frame=packet.get('frame_id');when=packet.get('simulation_time');timing=packet.get('render_timing',{})
        if (type(frame) is not int or frame<=last or type(when) not in (int,float) or not math.isfinite(when)
                or when<last_time or timing.get('frame_id')!=frame or timing.get('simulation_time')!=when):
            raise ValueError('native stereo render/frame/time synchronization failed')
        validate_identity(packet.get('identity'),len(packet['left_rgb']),when)
        for identity in packet['identity'] if isinstance(packet['identity'],list) else [packet['identity']]:
            if identity.get('frame_id')!=frame or identity.get('calibration_sha256')!=calibration.sha256:
                raise ValueError('native stereo pre-action identity/frame/calibration mismatch')
        calibration.validate_readback(packet['K_left'],packet['K_right'],packet['T_world_left'],packet['T_world_right'],origins=origins)
        if packet.get('calibration_sha256')!=calibration.sha256:raise ValueError('native stereo calibration SHA mismatch')
        actual=estimator.compute(packet['left_rgb'],packet['right_rgb'],calibration)
        if not torch.equal(actual['valid'],packet['valid']) or not torch.allclose(actual['depth_m'][actual['valid']],packet['depth_m'][actual['valid']],atol=1e-6,rtol=1e-5):
            raise ValueError('saved depth/valid is not reproducible from saved stereo RGB')
        _,reference_path=_process_output(p,base,reference_binding)
        reference=torch.load(reference_path,map_location='cpu',weights_only=False)
        if (not isinstance(reference,dict) or reference.get('version')!='dh116-stereo-depth-reference-v1'
                or reference.get('source_scope')!='native' or reference.get('reference_used_as_actor_input') is not False
                or reference.get('validation_only') is not True or reference.get('source_units')!='optical_axis_z_m'
                or reference.get('reference_source') not in ('rendered_depth_validation_only','independent_calibrated_depth_reference')
                or any(reference.get(k)!=packet.get(k) for k in ('frame_id','simulation_time','calibration_sha256'))):
            raise ValueError('independent same-frame native depth reference source required')
        from .tabletop_stereo_reference import AUDITED_CAMERA_VIEW_SHA256,validate_reference_timing
        validate_reference_timing(packet,reference,native=True)
        sdk=reference.get('sdk_camera_view_source_artifact')
        _process_output(p,base,sdk,binding_base=reference_path.parent)
        if sdk['sha256']!=AUDITED_CAMERA_VIEW_SHA256:raise ValueError('reference optical-Z SDK source mapping is not audited')
        target,mask=reference['depth_m'],reference['valid']
        if target.shape!=actual['depth_m'].shape or mask.shape!=target.shape or mask.dtype!=torch.bool:
            raise ValueError('reference depth/mask schema mismatch')
        expected_reference=torch.isfinite(target)&(target>0)
        if not torch.equal(mask,expected_reference):raise ValueError('reference validity cannot hide finite optical-depth pixels')
        if reference.get('evaluation_depth_range_m')!=[estimator.near,estimator.far]:
            raise ValueError('reference evaluation ROI differs from declared stereo depth profile')
        eligible=mask&(target>=estimator.near)&(target<=estimator.far);joint=actual['valid']&eligible
        errors.extend((actual['depth_m'][joint]-target[joint]).double().tolist())
        valid_pixels+=int(joint.sum());reference_pixels+=int(eligible.sum())
        candidate=reference.get('rgb_candidate_mask')
        if isinstance(candidate,torch.Tensor):
            from .tabletop_vision_camera import brown_mask
            recomputed=brown_mask(packet['left_rgb'].permute(0,2,3,1),float_rgb_is_255=False)
            if candidate.shape!=eligible.shape or candidate.dtype!=torch.bool or not torch.equal(candidate,recomputed):
                raise ValueError('RGB candidate mask differs from saved RGB baseline rule')
            candidate_total+=int((candidate&eligible).sum());candidate_reconstructed+=int((candidate&joint).sum())
        if reference.get('wire_mask_source_artifact') is not None:
            _process_output(p,base,reference['wire_mask_source_artifact'],binding_base=reference_path.parent)
        if reference.get('sdk_instance_annotator_source_artifact') is not None:
            _process_output(p,base,reference['sdk_instance_annotator_source_artifact'],binding_base=reference_path.parent)
        wire=_verified_wire_mask(reference,reference_path.parent,eligible.shape)
        if wire is not None:
            wire_frames+=1
            # Each reference-visible pixel represents one dense visible 3D point;
            # this measures that sampling, not hidden full-object reconstruction.
            from .tabletop_stereo_geometry import backproject
            true_points=backproject(target,calibration.K_left,eligible)
            estimated=backproject(actual['depth_m'],calibration.K_left,actual['valid'])
            distance=(true_points-estimated).norm(dim=-1)
            wire_total+=int((wire&eligible).sum())
            wire_reconstructed+=int((wire&joint&(distance<=limits.get('wire_point_tolerance_m',.005))).sum())
            wire_errors.extend((actual['depth_m'][wire&joint]-target[wire&joint]).double().tolist())
        else:wire_unavailable.extend(reference.get('wire_mask_blocked_reasons',['instance_reference_unavailable']))
        last,last_time=frame,when
    values=np.asarray(errors)
    wire_values=np.asarray(wire_errors);wire_complete=wire_frames==len(frames)
    metrics=dict(mae_m=float(np.abs(values).mean()) if len(values) else None,
                 rmse_m=float(np.sqrt(np.square(values).mean())) if len(values) else None,
                 p95_m=float(np.quantile(np.abs(values),.95)) if len(values) else None,
                 coverage=valid_pixels/max(reference_pixels,1),count=valid_pixels,
                 rgb_candidate_point_cloud_completeness=candidate_reconstructed/candidate_total if candidate_total else None,
                 wire_point_cloud_completeness=wire_reconstructed/wire_total if wire_total and not wire_unavailable else None,
                 wire_point_cloud_completeness_observed_subset=wire_reconstructed/wire_total if wire_total else None,
                 wire_reference_frame_coverage=wire_frames/len(frames),
                 wire_depth_mae_m=float(np.abs(wire_values).mean()) if len(wire_values) and wire_complete else None,
                 wire_depth_rmse_m=float(np.sqrt(np.square(wire_values).mean())) if len(wire_values) and wire_complete else None,
                 wire_depth_p95_m=float(np.quantile(np.abs(wire_values),.95)) if len(wire_values) and wire_complete else None,
                 wire_depth_count=len(wire_values) if wire_complete else None,
                 wire_depth_validity_coverage=len(wire_values)/wire_total if wire_total and wire_complete else None,
                 wire_reference_visible_points=wire_total,wire_reconstructed_points=wire_reconstructed,
                 wire_completeness_blocked_reasons=sorted(set(wire_unavailable)),
                 wire_completeness_sampling='dense_reference_visible_wire_pixels;not_hidden_entire_object')
    qualified=policy_compatible and len(frames)>=expected['min_frames'] and stereo_metric_gate(metrics,expected)
    cohort=None
    initials=captured.get('physical_initial_conditions')
    if captured.get('initial_condition_source')=='actual_sdk_readback' and isinstance(initials,list) and initials:
        actual_shas=sorted({physical_initial_sha256(initial) for initial in initials})
        rig=dict(baseline_m=calibration.baseline_m,T_world_left=calibration.T_world_left.tolist(),T_world_right=calibration.T_world_right.tolist())
        schedule=captured.get('capture_schedule')
        if not isinstance(schedule,dict):raise ValueError('actual policy/physics capture schedule required for resolution cohort')
        tick0=captured.get('initial_physics_tick');time0=captured.get('initial_simulation_time');resets=captured.get('initial_reset_ids')
        if type(tick0) is not int or type(time0) not in (float,int) or not isinstance(resets,list):raise ValueError('initial raw capture clocks required')
        actual_schedule=dict(advance_control_between_samples=p['producer']['kwargs'].get('advance_control_between_samples'),
            warmup_frames=p['producer']['kwargs'].get('warmup_frames'),frames=len(frames),policy_ticks=[],
            physics_ticks_from_initial=[],simulation_times_from_initial=[],reset_counts_from_initial=[],sampled_physical_initial_sha256=[])
        for packet in frames:
            identity=packet['identity'] if isinstance(packet['identity'],list) else [packet['identity']]
            if len(identity)!=len(resets) or len({row['policy_tick'] for row in identity})!=1:
                raise ValueError('batch actual policy/reset schedule mismatch')
            actual_schedule['policy_ticks'].append(identity[0]['policy_tick'])
            actual_schedule['physics_ticks_from_initial'].append(identity[0]['physics_tick_end']-tick0)
            actual_schedule['simulation_times_from_initial'].append(round(packet['simulation_time']-time0,9))
            actual_schedule['reset_counts_from_initial'].append([row['reset_id']-start for row,start in zip(identity,resets)])
            actual_schedule['sampled_physical_initial_sha256'].append([row.get('physical_initial_condition_sha256') for row in identity])
        if schedule!=actual_schedule:raise ValueError('declared comparison schedule differs from raw policy/physics/episode records')
        cohort=digest(dict(physical_initial_sha256=actual_shas,rig=rig,capture_schedule=schedule))
        if captured.get('cohort_sha256')!=cohort or p.get('cohort_sha256')!=cohort or validation.get('cohort_sha256')!=cohort:
            raise ValueError('resolution cohort does not match raw measured initial states/rig/config')
    performance=None
    if p.get('performance_artifact') is not None:
        performance_raw,_=_process_output(p,base,p['performance_artifact'])
        performance=json.loads(performance_raw)
        if performance!=captured.get('performance') or performance.get('source_scope')!='native':
            raise ValueError('performance artifact differs from raw capture measurement ledger')
        samples=performance.get('timing_samples');mem=performance.get('memory_samples')
        if not isinstance(samples,list) or len(samples)!=len(frames) or not isinstance(mem,list) or len(mem)!=len(frames):
            raise ValueError('per-frame actual timing/GPU samples required')
        for key in ('capture_ms','depth_ms','actor_ms'):
            measured=[row.get(key) for row in samples]
            if key=='actor_ms' and all(value is None for value in measured):
                if performance.get(key) is not None:raise ValueError('missing Actor cannot report zero latency')
                continue
            if any(type(value) not in (float,int) or not math.isfinite(value) or value<0 for value in measured):
                raise ValueError('invalid actual performance timing samples')
            if not math.isclose(performance.get(key,-1),sum(measured)/len(measured),abs_tol=1e-8):
                raise ValueError('performance mean does not match raw samples')
        if performance.get('actor_ms') is not None:
            actor=p.get('actor_package_artifact')
            _artifact(actor,base)
            if actor['sha256']!=performance.get('actor_package_sha256'):raise ValueError('Actor latency package/source SHA mismatch')
        if performance.get('measurement_device')=='cuda':
            if any(row.get('source_scope')!='native' or row.get('measurement_device')!='cuda' for row in mem):
                raise ValueError('CUDA performance requires actual device measurements, not CPU zeros')
            for key,reduce in (('gpu_peak_allocated_bytes',max),('gpu_peak_reserved_bytes',max),('gpu_free_bytes',min)):
                measured=[row.get(key) for row in mem]
                if any(type(value) is not int or value<=0 for value in measured) or performance.get(key)!=reduce(measured):
                    raise ValueError('CUDA memory summary not backed by raw device samples')
    return dict(qualified=qualified,measurements_verified=True,source_scope='native',calibration_sha256=calibration.sha256,metrics=metrics,
                proof_sha256=digest(p),validation_config_sha256=p['validation_config_artifact']['sha256'],
                width=calibration.width,height=calibration.height,depth_estimator_config=estimator.to_config(),
                cohort_sha256=cohort,sample_count=len(frames),performance=performance,
                conditions=dict(validation_policy_compatible=policy_compatible,
                    minimum_frames_met=len(frames)>=expected['min_frames'],source_and_sync_verified=True,
                    raw_calibration_verified=True),
                blocked_reasons=[] if qualified else ['native_stereo_depth_accuracy_coverage_framecount_or_policy_threshold_failed'])


def measure_stereo_evidence(proof):
    try:return _measure_stereo_evidence(proof)
    except (OSError,KeyError,TypeError,RuntimeError,UnicodeError) as error:
        raise ValueError('native stereo proof unavailable or malformed') from error


def verify_stereo_qualification(proof):
    result=measure_stereo_evidence(proof)
    if not result['qualified']:raise ValueError('native stereo depth error/coverage does not meet audited thresholds')
    return result


def _verify_student_qualification(report,checkpoint_sha256,styles=STYLES):
    """Independent heldout Student-only first episodes; no Teacher/MPC/critic input."""
    require_sha(checkpoint_sha256,'Student checkpoint SHA')
    r,episodes,base=_report_episodes(report)
    if r.get('owner')!='student' or r.get('checkpoint_sha256')!=checkpoint_sha256 or r.get('validation_split')!='heldout':
        raise ValueError('independent heldout Student checkpoint report required')
    required=dict(teacher_execution_enabled=False,mpc_enabled=False,critic_deployed=False,
                  privileged_actor_inputs=False,critic_training=False,actor_input_source='stereo_RGB_and_proprio_only',
                  depth_source='computed_from_synchronized_left_right_RGB')
    partition=training_partition(r,base=base)
    training_initials=partition['physical_initial_condition_sha256'];training_groups=partition['source_episode_groups']
    for episode in episodes:
        runtime=episode.get('runtime') or {}
        if (episode['owner']!='student' or episode.get('checkpoint_sha256')!=checkpoint_sha256
                or episode.get('actor_runtime_samples_verified') is not True or any(runtime.get(k)!=v for k,v in required.items())):
            raise ValueError('raw Student runtime indicates Teacher/MPC/privileged/critic execution or missing source')
        if runtime.get('model_checkpoint_sha256')!=checkpoint_sha256:raise ValueError('raw runtime model checkpoint identity mismatch')
        if episode['physical_initial_condition_sha256'] in training_initials or episode['source_episode_group'] in training_groups:
            raise ValueError('Student heldout initial/source-episode training leakage')
    artifacts=r.get('evidence_artifacts',[]);checkpoints=[];kinds=set()
    for item in artifacts:
        _artifact(item,base);kinds.add(item.get('kind'))
        if item.get('kind')=='checkpoint':checkpoints.append(item)
    if not {'checkpoint','source','config'}<=kinds or len(checkpoints)!=1 or checkpoints[0]['sha256']!=checkpoint_sha256:
        raise ValueError('actual Student model/source/config artifacts required')
    for style in styles:
        subset=[e for e in episodes if e['grasp_style']==style]
        if len({e['physical_initial_condition_sha256'] for e in subset})<100:
            raise ValueError('Student needs100 measured independent heldout initial conditions per grasp')
        if not any(e['qualified_success'] for e in subset):raise ValueError('no qualified Student-only success for grasp')
    return dict(qualified=True,owner='student',checkpoint_sha256=checkpoint_sha256,
                action_contract_sha256=action_sha256(),state_contract_sha256=digest(state_contract()),report_sha256=digest(r))


def verify_student_qualification(report,checkpoint_sha256,styles=STYLES):
    try:return _verify_student_qualification(report,checkpoint_sha256,styles)
    except (OSError,KeyError,TypeError,RuntimeError,UnicodeError) as error:
        raise ValueError('native Student proof unavailable or malformed') from error


def promotion_ledger(phases,*,teacher_report=None,teacher_checkpoint_sha256=None,stereo_native_proof=None,
                     student_report=None,student_checkpoint_sha256=None):
    """Implementation/CPU progress is independent from pending physical gates."""
    output={}
    for i in range(8):
        key='phase'+str(i);source=phases.get(key,{})
        output[key]=dict(implemented=source.get('implemented') is True,cpu_verified=source.get('cpu_verified') is True,
                         native_verified=False,physical_qualified=False)
    reasons=[];teacher=False
    if teacher_report is not None:
        try:
            require_teacher_qualification(teacher_report,teacher_checkpoint_sha256)
            output['phase1'].update(native_verified=True,physical_qualified=True);teacher=True
        except (ValueError,TypeError) as error:reasons.append('Teacher: '+str(error))
    else:reasons.append('Teacher native physical qualification pending')
    stereo=False;student=False
    if stereo_native_proof is not None:
        try:
            verify_stereo_qualification(stereo_native_proof)
            output['phase2'].update(native_verified=True,physical_qualified=True);stereo=True
        except (ValueError,TypeError,OSError,KeyError) as error:reasons.append('Stereo: '+str(error))
    else:reasons.append('Stereo native calibration/accuracy pending')
    if student_report is not None:
        try:
            verify_student_qualification(student_report,student_checkpoint_sha256)
            output['phase7'].update(native_verified=True,physical_qualified=True);student=True
        except (ValueError,TypeError) as error:reasons.append('Student: '+str(error))
    else:reasons.append('Final Student independent native qualification pending')
    return dict(phases=output,teacher_qualified=teacher,formal_dagger_allowed=teacher and stereo,deployment_allowed=stereo and student,
                isolated_cpu_implementation_allowed=True,blocked_reasons=reasons)
