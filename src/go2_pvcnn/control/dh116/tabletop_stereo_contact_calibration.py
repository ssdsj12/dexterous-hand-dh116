"""Independent PhysX source calibration, deliberately not an R2 source adapter.

Uses actual dt=1 and dt=physics_dt reads to audit impulse scaling. Full-report
side/unit declarations remain hypotheses tested against measured momentum. A
rigid calibration pair cannot qualify DH116 fingers, external support absence,
attachments, or a success episode. No simulator is imported by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import numpy as np


def _copy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def calibration_episode_uid(output, side):
    """Output directories are exclusive; include the full run path, not its leaf."""
    import hashlib
    identity=hashlib.sha256(str(Path(output).resolve()).encode('utf-8')).hexdigest()
    return f'calibration:{identity}:{side}:0'


def read_binary_stl(path):
    """Read original manufacturer triangles in metres, without coordinate edits."""
    raw=Path(path).read_bytes()
    count=struct.unpack('<I',raw[80:84])[0] if len(raw)>=84 else 0
    if not count or len(raw)!=84+count*50:
        raise ValueError('exact nonempty binary STL required')
    dtype=np.dtype([('normal','<f4',3),('vertices','<f4',(3,3)),('attribute','<u2')])
    vertices=np.frombuffer(raw,dtype=dtype,offset=84,count=count)['vertices'].copy()
    if not np.isfinite(vertices).all():
        raise ValueError('STL vertices must be finite')
    return vertices


def configure_report_processing(settings, *, enable=False, enable_readback=False):
    """Allow an isolated calibration to test the GPU contact readback setting."""
    key='/physics/disableContactProcessing'
    readback_key='/physics/suppressReadback'
    before=settings.get(key)
    readback_before=settings.get(readback_key)
    if enable:
        settings.set_bool(key,False)
    if enable_readback:
        settings.set_bool(readback_key,False)
    readback=settings.get(readback_key)
    if enable_readback and readback is not False:
        raise RuntimeError(f'{readback_key} must read back as False for this trial')
    return dict(key=key,before=before,after=settings.get(key),modified=bool(enable),
                suppress_readback=readback,suppress_readback_before=readback_before,
                readback_requested=bool(enable_readback),physical_solver_or_original_task_modified=False)

def _snapshot(view, method, dt):
    values = [_copy(value) for value in getattr(view, method)(dt=dt)]
    expected = 6 if method == 'get_contact_data' else 4
    if len(values) != expected:
        raise ValueError('unexpected installed PhysX contact tuple arity')
    counts, starts = values[-2:]
    if counts.shape != (1, 1) or starts.shape != (1, 1):
        raise ValueError('calibration requires exactly one sensor/filter pair')
    if counts.dtype.kind not in 'iu' or starts.dtype.kind not in 'iu':
        raise ValueError('integer contact range ownership required')
    count, start = int(counts[0, 0]), int(starts[0, 0])
    if count < 0 or start < 0 or any(start + count > len(v) for v in values[:-2]):
        raise ValueError('contact range exceeds owned data capacity')
    data = [v[start:start + count].copy() for v in values[:-2]]
    if not all(np.isfinite(v).all() for v in data):
        raise ValueError('nonfinite native contact value')
    return data, counts, starts, [len(v) for v in values[:-2]]


def capture_tensor_pair(view, *, dt):
    """Four reads, zero steps. Copy counts/starts before the next SDK getter."""
    if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError('positive finite physical dt required')
    output, scale_errors = {}, []
    for name, method in (('normal', 'get_contact_data'), ('friction', 'get_friction_data')):
        force, counts, starts, capacities = _snapshot(view, method, dt)
        impulse, impulse_counts, impulse_starts, impulse_capacities = _snapshot(view, method, 1.)
        if (capacities != impulse_capacities or not np.array_equal(counts, impulse_counts) or not np.array_equal(starts, impulse_starts)
                or any(not np.array_equal(a, b) for a, b in zip(force[1:], impulse[1:]))):
            raise ValueError('contact ownership or geometry changed during same-step dt audit')
        error = np.abs(force[0] - impulse[0] / dt)
        scale_errors.append(float(error.max()) if error.size else 0.)
        fields = dict(force_n=force[0].tolist(), impulse_n_s=impulse[0].tolist(),
                      points_world_m=force[1].tolist(), counts=counts.tolist(), starts=starts.tolist(),
                      buffer_capacities=capacities,
                      capacity_boundary_reached=bool(counts[0,0] and any(int(starts[0,0]+counts[0,0])>=n for n in capacities)),
                      sdk_overflow_status=None)
        if name == 'normal':
            fields.update(normal=force[2].tolist(), separation_m=force[3].tolist())
        output[name] = fields
    return dict(**output, dt_s=float(dt), dt_scaling_max_error_n=max(scale_errors),
                impulse_source='same_PhysX_Tensor_getter_dt_1_read_not_full_report',
                force_side='sensor_side_candidate_pending_momentum_validation',
                partner_source='explicit_single_filter_configuration_not_native_pair_ID',
                qualified_on_object=False)


def _summary(values):
    if not values:
        return None
    x = np.asarray(values, dtype=float)
    return dict(min=float(x.min()), p50=float(np.quantile(x, .50)), p95=float(np.quantile(x, .95)),
                max=float(x.max()), mean=float(x.mean()))


def decode_contact_report(raw, *, path_decoder):
    """Lossless copied header/contact/anchor arrays, including header-only events.

    Preserve offsets into the arrays so per-pair ownership can be audited.
    Unknown event IDs remain numeric; nothing is inferred from an empty report.
    """
    if len(raw) not in (2,3):
        raise ValueError('installed ordinary/full report requires two/three arrays')
    headers, points = raw[:2]
    anchors = raw[2] if len(raw)==3 else None
    result = dict(headers=[], contacts=[], friction_anchors=[] if anchors is not None else None,
                  friction_source_available=anchors is not None)
    for h in headers:
        row = {name: str(path_decoder(getattr(h,name)))
               for name in ('actor0','actor1','collider0','collider1')}
        row.update({name:int(getattr(h,name)) for name in ('stage_id','contact_data_offset',
            'num_contact_data','friction_anchors_offset','num_friction_anchors_data')})
        for name in ('type','event_type'):
            if hasattr(h,name): row[name] = int(getattr(h,name))
        ranges=[(row['contact_data_offset'],row['num_contact_data'],len(points))]
        if anchors is not None:
            ranges.append((row['friction_anchors_offset'],row['num_friction_anchors_data'],len(anchors)))
        for offset,count,total in ranges:
            if offset < 0 or count < 0 or offset + count > total:
                raise ValueError('full-report range ownership outside raw arrays')
        result['headers'].append(row)
    for p in points:
        result['contacts'].append(dict(position=list(map(float,p.position)),
            impulse=list(map(float,p.impulse)),normal=list(map(float,p.normal)),separation=float(p.separation)))
    for a in anchors if anchors is not None else []:
        result['friction_anchors'].append(dict(position=list(map(float,a.position)),impulse=list(map(float,a.impulse))))
    return result


class ContactReportReceiptBuffer:
    """Copy callback data before SDK buffer reuse; arrival windows are not clocks.

    The ordinary API provides no friction array. A receipt during sim.step is
    associated with that call, not claimed to carry a native contact timestamp.
    Poll, ordinary callback and full callback remain separate observations.
    """
    def __init__(self, *, clock_reader, path_decoder):
        self._clock_reader=clock_reader
        self._path_decoder=path_decoder
        self._requested_step=None
        self._receipts=[]

    def begin_step(self, step):
        if self._requested_step is not None or not isinstance(step,int) or step<1:
            raise ValueError('non-overlapping positive physical step window required')
        self._requested_step=step

    def end_step(self):
        self._requested_step=None

    def record(self, channel, raw):
        if channel not in ('ordinary','full') or len(raw)!={'ordinary':2,'full':3}[channel]:
            raise ValueError('callback channel and installed array arity must agree')
        step, time_s=self._clock_reader()
        self._receipts.append(dict(channel=channel,
            arrival_sdk_step=int(step),arrival_sdk_time_s=float(time_s),
            requested_physics_step=self._requested_step,
            within_simulation_step_call=self._requested_step is not None,
            source_clock_association='arrival_only_with_step_call_window_not_contact_timestamp',
            raw_report=decode_contact_report(raw,path_decoder=self._path_decoder)))

    def drain(self):
        receipts,self._receipts=self._receipts,[]
        return receipts


def close_calibration_app(app, output, *, simulation=None, stack_timeout_s=30):
    """Isolated no-camera rig cleanup; subprocess exit remains launcher-owned."""
    import builtins
    import faulthandler
    import json
    import time
    output = Path(output)
    with (output/'shutdown_started.json').open('x') as f:
        json.dump(dict(time_unix_s=time.time(),wait_for_replicator=False,process_exit_verified=False),f)
    with (output/'shutdown_stacks.log').open('x') as stacks:
        faulthandler.dump_traceback_later(stack_timeout_s,repeat=True,file=stacks)
        try:
            callback_error = getattr(builtins,'ISAACLAB_CALLBACK_EXCEPTION',None)
            if simulation is not None:
                # Installed IsaacLab public clear_instance() first unsubscribes
                # its STOP handler, then clears core callbacks/singleton. Calling
                # stop() or app.close() first can enter its render-until-playing
                # loop. Never touch another owner's singleton or installed code.
                cls = type(simulation)
                if cls.instance() is not simulation:
                    raise RuntimeError('calibration cleanup requires its owned SimulationContext')
                cls.clear_instance()
                if cls.instance() is not None:
                    raise RuntimeError('owned SimulationContext did not clear')
                with (output/'simulation_context_cleared.json').open('x') as f:
                    json.dump(dict(public_api='SimulationContext.clear_instance',
                        stop_callback_released=True,physics_sampling_already_ended=True),f)
            app.close(wait_for_replicator=False)
            callback_error = callback_error or getattr(builtins,'ISAACLAB_CALLBACK_EXCEPTION',None)
        finally:
            faulthandler.cancel_dump_traceback_later()
    with (output/'shutdown_complete.json').open('x') as f:
        json.dump(dict(time_unix_s=time.time(),close_returned=True,process_exit_verified=False,
            pending_simulation_callback_error=repr(callback_error) if callback_error is not None else None),f)
    if callback_error is not None:
        raise callback_error


def _tensor_components(tensor):
    normal,friction=tensor['normal'],tensor['friction']
    normal_force=(np.asarray(normal['force_n']).reshape(-1,1)*np.asarray(normal['normal']).reshape(-1,3)).sum(0)
    tangential=np.asarray(friction['force_n']).reshape(-1,3).sum(0)
    return dict(normal=normal_force,friction=tangential,total=normal_force+tangential)


def _validate_polled_ranges(headers, entries, offset_key, count_key):
    if not isinstance(entries, list):
        raise ValueError('structured poll requires complete raw arrays')
    ranges = []
    for header in headers:
        offset, count = header.get(offset_key), header.get(count_key)
        if type(offset) is not int or type(count) is not int:
            raise ValueError('polled range offsets and counts must be integers')
        if min(offset, count) < 0 or offset + count > len(entries):
            raise ValueError('polled range ownership outside raw arrays')
        if count:
            ranges.append((offset, offset + count))
    end = 0
    for first, last in sorted(ranges):
        if first < end:
            raise ValueError('polled contact range ownership overlap')
        if first > end:
            raise ValueError('unowned polled contact array entries')
        end = last
    if end != len(entries):
        raise ValueError('unowned polled contact array entries')


def _world_vector(value):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError('polled impulse or normal must be a finite world vector')
    return vector


def _report_pairs(row):
    raw = row.get('full_report')
    if raw is None:
        # Legacy v02 rows copied friction once onto their first contact point.
        return row.get('pair_reports',[])
    headers = raw.get('headers')
    if not isinstance(headers, list) or not all(isinstance(header, dict) for header in headers):
        raise ValueError('structured poll requires its complete header array')
    points, anchors = raw.get('contacts'), raw.get('friction_anchors')
    _validate_polled_ranges(headers, points, 'contact_data_offset', 'num_contact_data')
    _validate_polled_ranges(headers, anchors, 'friction_anchors_offset', 'num_friction_anchors_data')
    result = []
    for h in headers:
        contacts = points[h['contact_data_offset']:h['contact_data_offset']+h['num_contact_data']]
        friction = anchors[h['friction_anchors_offset']:h['friction_anchors_offset']+h['num_friction_anchors_data']]
        contact_impulses = [_world_vector(c['impulse']) for c in contacts]
        for c in contacts:
            if 'normal' in c:
                _world_vector(c['normal'])
        result.append(dict(h, impulse=sum(contact_impulses, np.zeros(3)).tolist(),
            friction_impulses=[_world_vector(anchor['impulse']).tolist() for anchor in friction],
            contacts=contacts))
    return result


def _prim_path(value):
    return isinstance(value, str) and value.startswith('/') and len(value) > 1


def _polled_source_audit(frames, object_path, collider_by_actor, inventory):
    """Compare only owned, identity-matched polled pairs. Side remains hypothetical."""
    details, residuals, stages = [], {}, set()
    counts = dict(legacy_flattened_steps=0, empty_structured_poll_steps=0,
                  identity_matched_header_count=0, object_unclassified_contact_header_count=0)
    for row, reports, expected in frames:
        if row.get('full_report') is None:
            counts['legacy_flattened_steps'] += 1
            continue
        if not reports:
            counts['empty_structured_poll_steps'] += 1
            continue
        dt, pairs, unclassified, unidentified = row['dt'], [], [], 0
        for header in reports:
            actors = (header.get('actor0'), header.get('actor1'))
            colliders = (header.get('collider0'), header.get('collider1'))
            stage_id = header.get('stage_id')
            if not all(_prim_path(actor) for actor in actors):
                unidentified += 1
            if object_path not in actors:
                continue
            side = actors.index(object_path)
            if (actors[0] == actors[1] or set(actors) != set(collider_by_actor)
                    or not all(_prim_path(collider) and collider_by_actor[actor] == collider
                               for actor, collider in zip(actors, colliders))
                    or type(stage_id) is not int or stage_id < 0
                    or inventory.get('stage_id', stage_id) != stage_id):
                unclassified.append(actors[1-side] if _prim_path(actors[1-side]) else None)
                continue
            stages.add(stage_id)
            contact = _world_vector(header['impulse'])
            friction = np.asarray(header['friction_impulses'], dtype=float).reshape(-1, 3).sum(0)
            raw = dict(contact_only=contact, friction_only=friction, contact_plus_friction=contact+friction)
            sign = 1 if side == 0 else -1
            pairs.append(dict(
                body_a=dict(actor=actors[0], collider=colliders[0]),
                body_b=dict(actor=actors[1], collider=colliders[1]),
                stage_id=stage_id, object_actor_index=side,
                raw_impulse_world_n_s={k:v.tolist() for k,v in raw.items()},
                force_on_object_world_n_if_raw_on_actor0={k:(sign*v/dt).tolist() for k,v in raw.items()},
                force_on_object_world_n_if_raw_on_actor1={k:(-sign*v/dt).tolist() for k,v in raw.items()},
                force_on_object_verified_world_n=None))
        counts['identity_matched_header_count'] += len(pairs)
        counts['object_unclassified_contact_header_count'] += len(unclassified)
        orders = {p['object_actor_index'] for p in pairs}
        eligible = len(orders) == 1 and not unclassified and not unidentified
        if eligible and any(np.linalg.norm(p['raw_impulse_world_n_s'][k]) > 1e-12
                            for p in pairs for k in ('contact_only', 'friction_only')):
            order = f'object_actor{next(iter(orders))}'
            for hypothesis in ('impulse_on_actor0', 'impulse_on_actor1'):
                sign = 1 if order.endswith(hypothesis[-1]) else -1
                for name in ('contact_only', 'friction_only', 'contact_plus_friction'):
                    vector = sum((np.asarray(p['raw_impulse_world_n_s'][name]) for p in pairs), np.zeros(3))
                    samples = residuals.setdefault(order, {}).setdefault(name, {}).setdefault(hypothesis, [])
                    samples.append(float(np.linalg.norm(expected - sign * vector)))
        details.append(dict(physics_step=row['physics_step'], sdk_step=row.get('sdk_step'),
                            timestamp_s=row['timestamp'], dt_s=dt, pairs=pairs,
                            unclassified_object_partners=unclassified,
                            unidentified_header_count=unidentified,
                            momentum_comparison_eligible=eligible))
    hypotheses = {order:{kind:{side:_summary(values) for side,values in items.items()}
                         for kind,items in variants.items()} for order,variants in residuals.items()}
    return dict(**counts, steps=details, observed_stage_ids=sorted(stages),
                sign_hypotheses_by_actor_order=hypotheses if len(stages) <= 1 else {},
                multi_stage_fit_blocked=len(stages) > 1,
                impulse_scope='reported_impulse_channel_conditional_force_equals_impulse_over_physical_dt',
                combined_channel_scope='contact_plus_friction_is_unqualified_and_may_double_count',
                force_on_object_verified_world_n=None, external_support_free=None, attachment_free=None)


def summarize_pair_calibration(rows, *, object_path, expected_partner, inventory):
    """Compare competing impulse hypotheses; no automatic source qualification."""
    if (not _prim_path(object_path) or not _prim_path(expected_partner)
            or object_path == expected_partner):
        raise ValueError('two distinct actor prim paths are required')
    prior_step,prior_time=-1,-math.inf
    counts=dict(nonzero_tensor_steps=0,nonzero_report_steps=0,unmatched_pair_records=0,
                unmatched_collider_records=0,capacity_boundary_steps=0)
    variants={name:dict(plus=[],minus=[]) for name in ('contact_only','friction_only','contact_plus_friction')}
    reverse={name:[] for name in ('normal','friction','total')}
    projections=dict(contact_normal=[],contact_tangent=[],friction_normal=[],friction_tangent=[])
    orders,tensor_residual,dt_errors={},[],[]
    pair_records=0
    callback_counts={}
    polled_frames=[]
    expected_colliders={inventory.get('object_collider'),inventory.get('finger_collider')}
    colliders_known=None not in expected_colliders and len(expected_colliders)==2
    collider_by_actor={object_path:inventory.get('object_collider'), expected_partner:inventory.get('finger_collider')}
    for row in rows:
        for receipt in row.get('callback_receipts',[]):
            channel=receipt.get('channel','legacy_full')
            raw=receipt.get('raw_report',receipt.get('full_report',{}))
            counter=callback_counts.setdefault(channel,dict(receipts=0,headers=0,contacts=0,
                friction_anchors=0,friction_available_receipts=0,outside_step_window=0))
            counter['receipts']+=1
            counter['headers']+=len(raw.get('headers',[]))
            counter['contacts']+=len(raw.get('contacts',[]))
            anchors=raw.get('friction_anchors')
            counter['friction_anchors']+=len(anchors) if anchors is not None else 0
            counter['friction_available_receipts']+=int(anchors is not None)
            counter['outside_step_window']+=int(not receipt.get('within_simulation_step_call',False))
        step,timestamp,dt=row['physics_step'],row['timestamp'],row['dt']
        if step<=prior_step or timestamp<=prior_time or not math.isfinite(timestamp) or not math.isfinite(dt) or dt<=0:
            raise ValueError('physical steps and timestamps must be strictly monotonic')
        prior_step,prior_time=step,timestamp
        expected=row['mass_kg']*(np.asarray(row['velocity_after'])-np.asarray(row['velocity_before'])-np.asarray(row['gravity_m_s2'])*dt)
        components=_tensor_components(row['tensor'])
        counts['nonzero_tensor_steps']+=int(np.linalg.norm(components['total'])>1e-9)
        tensor_residual.append(float(np.linalg.norm(expected-components['total']*dt)))
        dt_errors.append(row['tensor']['dt_scaling_max_error_n'])
        counts['capacity_boundary_steps']+=int(any(row['tensor'][k].get('capacity_boundary_reached',False) for k in ('normal','friction')))
        if 'reverse_tensor' in row:
            back=_tensor_components(row['reverse_tensor'])
            for name in reverse:reverse[name].append(float(np.linalg.norm((components[name]+back[name])*dt)))
        contact_sum,friction_sum=np.zeros(3),np.zeros(3)
        matched = False
        object_unclassified = False
        reports = _report_pairs(row)
        polled_frames.append((row, reports, expected))
        for report in reports:
            actors=(report.get('actor0'),report.get('actor1'))
            if set(actors)!={object_path,expected_partner}:
                counts['unmatched_pair_records']+=1
                object_unclassified |= object_path in actors or not all(_prim_path(actor) for actor in actors)
                continue
            if colliders_known and any(report.get(f'collider{side}')!=collider_by_actor[actor] for side,actor in enumerate(actors)):
                counts['unmatched_collider_records']+=1
                object_unclassified = True
                continue
            pair_records+=1;matched=True
            order='object_actor0' if actors[0]==object_path else 'object_actor1'
            orders[order]=orders.get(order,0)+1
            contact=np.asarray(report['impulse']);friction=np.asarray(report.get('friction_impulses',[])).reshape(-1,3).sum(0)
            contact_sum+=contact;friction_sum+=friction
            # Contact normals differ across a manifold. Report per-contact
            # contact components only; friction projection uses the manifold's
            # resultant normal and is explicitly a diagnostic approximation.
            contacts=report.get('contacts',[report])
            normal_sum=np.zeros(3)
            for c in contacts:
                if 'normal' not in c:continue
                normal=np.asarray(c['normal']);norm=np.linalg.norm(normal)
                if norm<=1e-12:continue
                normal=normal/norm;normal_sum+=normal
                v=np.asarray(c['impulse']);along=float(np.dot(v,normal))
                projections['contact_normal'].append(abs(along))
                projections['contact_tangent'].append(float(np.linalg.norm(v-along*normal)))
            norm=np.linalg.norm(normal_sum)
            if norm>1e-12:
                normal=normal_sum/norm;along=float(np.dot(friction,normal))
                projections['friction_normal'].append(abs(along))
                projections['friction_tangent'].append(float(np.linalg.norm(friction-along*normal)))
        if matched and (np.linalg.norm(contact_sum)>1e-12 or np.linalg.norm(friction_sum)>1e-12):
            counts['nonzero_report_steps']+=1
        if matched and not object_unclassified and (np.linalg.norm(contact_sum)>1e-12 or np.linalg.norm(friction_sum)>1e-12):
            for name,vector in (('contact_only',contact_sum),('friction_only',friction_sum),('contact_plus_friction',contact_sum+friction_sum)):
                variants[name]['plus'].append(float(np.linalg.norm(expected-vector)))
                variants[name]['minus'].append(float(np.linalg.norm(expected+vector)))
    blockers=[]
    if not counts['nonzero_report_steps']:blockers.append('no_nonzero_native_pair_report')
    if len(orders)>1:blockers.append('mixed_actor_order_requires_separate_side_fit')
    if counts['unmatched_pair_records']:blockers.append('additional_contact_partner_present')
    if counts['capacity_boundary_steps']:blockers.append('tensor_capacity_boundary_requires_overflow_audit')
    if not colliders_known or counts['unmatched_collider_records']:blockers.append('collider_identity_not_verified')
    blockers+=['contact_impulse_vs_friction_semantics_not_qualified','DH116_articulation_contacts_not_qualified_by_rigid_pair',
               'external_support_and_attachment_per_episode_not_qualified']
    hypothesis = ({name:{sign:_summary(v) for sign,v in values.items()} for name,values in variants.items()}
                  if len(orders)==1 and any(v for values in variants.values() for v in values.values()) else None)
    actor_ok=bool(pair_records and not counts['unmatched_pair_records'])
    collider_ok=bool(actor_ok and colliders_known and not counts['unmatched_collider_records'])
    return dict(scope='isolated_rigid_contact_calibration',frames=len(rows),**counts,
        pair_order_counts=orders,actor_pair_identity_verified=actor_ok,collider_identity_verified=collider_ok,
        pair_identity_verified=collider_ok,
        report_impulse_hypotheses=hypothesis,report_impulse_semantics_qualified=False,
        callback_source_counts=callback_counts,callback_impulses_included_in_poll_or_tensor_totals=False,
        impulse_sign_residual_n_s=hypothesis['contact_plus_friction'] if hypothesis else None,
        legacy_impulse_sign_field_semantics='unqualified_contact_plus_friction_hypothesis',
        impulse_component_magnitudes_n_s={k:_summary(v) for k,v in projections.items()},
        friction_projection_basis='normalized_sum_of_contact_normals_diagnostic_not_unique_anchor_normal',
        reverse_tensor_action_reaction_n_s={k:_summary(v) for k,v in reverse.items()},
        tensor_sensor_side_momentum_residual_n_s=_summary(tensor_residual),dt_scaling_error_n=_summary(dt_errors),
        polled_source_audit=_polled_source_audit(polled_frames, object_path, collider_by_actor, inventory),
        inventory=inventory,R1_qualified=False,R2_qualified=False,force_proxy_qualified=False,blockers=blockers)


def audit_archived_contact_sources(raw_path, inventory_path, stage_path, *,
                                   raw_sha256, inventory_sha256, object_path, expected_partner):
    """Replay immutable calibration evidence without Isaac or qualification."""
    raw_path, inventory_path, stage_path = map(Path, (raw_path, inventory_path, stage_path))
    raw, inventory_bytes, stage = (p.read_bytes() for p in (raw_path, inventory_path, stage_path))
    raw_hash, inventory_hash, stage_hash = (hashlib.sha256(blob).hexdigest()
                                           for blob in (raw, inventory_bytes, stage))
    if raw_hash != raw_sha256 or inventory_hash != inventory_sha256:
        raise ValueError('raw ledger or inventory SHA256 does not match the independent pin')
    def reject_constant(token):
        raise ValueError(f'nonfinite archived JSON token: {token}')

    inventory = json.loads(inventory_bytes, parse_constant=reject_constant)
    if inventory.get('stage_sha256') != stage_hash:
        raise ValueError('stage SHA256 does not match the pinned inventory')
    if not raw.endswith(b'\n'):
        raise ValueError('contact JSONL archive is empty or lacks a final newline')
    rows = [json.loads(line, parse_constant=reject_constant) for line in raw.splitlines()]
    if not all(isinstance(row, dict) and type(row.get('physics_step')) is int for row in rows):
        raise ValueError('contact JSONL requires per-step records')
    return dict(source_artifact=dict(raw_sha256=raw_hash, inventory_sha256=inventory_hash,
                                    stage_sha256=stage_hash),
                summary=summarize_pair_calibration(rows, object_path=object_path,
                                                   expected_partner=expected_partner, inventory=inventory))
