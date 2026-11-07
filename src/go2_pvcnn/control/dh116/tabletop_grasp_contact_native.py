"""Opt-in object-filtered PhysX evidence, independent of policy observations.

Raw normals are forces ON HAND. CAD patch normals only gate these samples;
neither CAD/FK references nor averaged force directions are raw contact normals.
The reducer can be tested with fixtures; only the observer checks native types.
"""
import gzip
import json
import math
from pathlib import Path
import torch

from .tabletop_effective_pad import DIGITS

SCHEMA = 'dh116-grasp-contact-native-v1'
TORQUE_SOURCE = 'implicit_actuator_estimate_not_hardware_measurement'


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _rotation(pose):
    q = pose[3:]
    if not bool(torch.isfinite(pose).all()) or abs(float(q.norm()) - 1.) > 1e-3:
        return None
    w, x, y, z = q / q.norm()
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y))).reshape(3, 3)


def reduce_pad_contact_buffers(bindings, raw_buffers, poses, atlas_rows, n):
    """Detach one substep, retaining every owned manifold slot without clipping.

    Bindings are (hand_index, actual_body_id, distal_digit_or_None). Intervals
    are validated per sensor, including unordered starts and overlapping owners.
    Unknown raw rows invalidate that environment; they never imply no contact.
    """
    if type(n) is not int or n < 1 or len(bindings) != len(raw_buffers) or len(poses) != 2:
        raise ValueError('matching two-hand native contact buffers required')
    poses = [torch.as_tensor(p).detach().cpu().double() for p in poses]
    if any(p.ndim != 3 or p.shape[0] != n or p.shape[-1] != 7 for p in poses):
        raise ValueError('actual body link poses must have shape N,B,7')
    atlas = {key: {k: torch.as_tensor(v).detach().cpu().double()
                  for k, v in row.items()} for key, row in atlas_rows.items()}
    rows = []
    for ei in range(n):
        pads = []
        for hi, side in enumerate(('left', 'right')):
            for di, digit in enumerate(DIGITS):
                pads.append(dict(hand=side, digit=digit, body_id=-1, valid=(hi, di) in atlas,
                    raw_load_n=0., load_n=0., accepted_load_n=0., position_w_m=None,
                    normal_on_hand_w=None, contact_count=0, wrong_normal_count=0,
                    raw_contact_ids=[], normal_source='reduced_native_force_direction',
                    position_source='native_normal_load_weighted_accepted_points',
                    _weighted=torch.zeros(3, dtype=torch.float64),
                    _force=torch.zeros(3, dtype=torch.float64)))
        valid_pose = all(bool(torch.isfinite(p[ei]).all()) and
                         bool(((p[ei, :, 3:].norm(dim=-1)-1.).abs() <= 1e-3).all())
                         for p in poses)
        rows.append(dict(measurement_valid=valid_pose,
                         invalid_reasons=[] if valid_pose else ['invalid_actual_link_pose'],
                         pads=pads, raw_contacts=[], raw_contact_buffers=[]))
    slot_base = 0
    seen = set()
    for si, ((hi, bi, di), raw) in enumerate(zip(bindings, raw_buffers)):
        if hi not in (0, 1) or type(bi) is not int or not 0 <= bi < poses[hi].shape[1]:
            raise ValueError('native sensor body binding invalid')
        if (hi, bi) in seen or (di is not None and di not in range(5)):
            raise ValueError('duplicate or invalid distal body binding')
        seen.add((hi, bi))
        if len(raw) != 6:
            raise ValueError('PhysX six-field contact tuple required')
        f, p, normal, sep, count, start = [torch.as_tensor(v).detach().cpu().clone() for v in raw]
        f, p, normal, sep = f.double().flatten(), p.double().reshape(-1, 3), normal.double().reshape(-1, 3), sep.double().flatten()
        if count.numel() != n or start.numel() != n or not len(f) == len(p) == len(normal) == len(sep):
            raise ValueError('native buffer sizes do not match environment ownership')
        if count.is_floating_point() or start.is_floating_point():
            raise ValueError('native contact interval indices must be integers')
        count, start = count.flatten().long(), start.flatten().long()
        for ei in range(n):
            rows[ei]['raw_contact_buffers'].append(dict(sensor_index=si, body_id=bi,
                count=int(count[ei]), start=int(start[ei]), capacity=len(f),
                buffer_source='same_substep_native_contact_tuple'))
        owners = {}
        invalid = set()
        for ei, (number, begin) in enumerate(zip(count.tolist(), start.tolist())):
            if number < 0 or begin < 0 or begin > len(f) or number > len(f)-begin:
                invalid.add(ei); continue
            for slot in range(begin, begin+number):
                if slot in owners:
                    invalid.update((ei, owners[slot]))
                else:
                    owners[slot] = ei
        for ei in invalid:
            rows[ei]['measurement_valid'] = False
            rows[ei]['invalid_reasons'].append('invalid_or_overlapping_contact_interval')
        for ei in range(n):
            if di is not None:
                rows[ei]['pads'][hi*5+di]['body_id'] = bi
                rows[ei]['pads'][hi*5+di]['body_link_pose_w'] = _json_value(poses[hi][ei, bi])
        for slot, ei in owners.items():
            row = rows[ei]; pose = poses[hi][ei, bi]; rotation = _rotation(pose)
            finite = bool(torch.isfinite(f[slot]) & torch.isfinite(p[slot]).all()
                          & torch.isfinite(normal[slot]).all() & torch.isfinite(sep[slot]))
            sane = finite and abs(float(normal[slot].norm())-1.) <= .05 and rotation is not None and ei not in invalid
            if not sane:
                row['measurement_valid'] = False
                row['invalid_reasons'].append('invalid_native_contact_or_body_pose')
            in_region = on_surface = normal_ok = False
            if sane and di is not None and (hi, di) in atlas:
                patch = atlas[hi, di]
                local_p = (p[slot]-pose[:3]) @ rotation
                local_n = normal[slot] @ rotation
                rel = local_p-patch['center_local_m']
                uv = torch.stack(((rel*patch['tangent_u']).sum(), (rel*patch['tangent_v']).sum()))
                in_region = bool(((uv/patch['half_widths_m'])**2).sum() <= 1.)
                planes = patch['planes_local']
                distance, index = (planes[:, :3] @ local_p + planes[:, 3]).max(0)
                on_surface = abs(float(distance)) <= float(patch['surface_tolerance_m'])
                cosine = float((local_n*planes[index, :3]).sum()/local_n.norm().clamp_min(1e-30))
                normal_ok = cosine <= -float(patch['normal_cosine_min'])
            positive = sane and float(f[slot]) > 0.
            point_id = slot_base+slot
            record = dict(contact_id=point_id, sensor_index=si, raw_slot=slot,
                raw_count=int(count[ei]), raw_start=int(start[ei]), hand=('left', 'right')[hi],
                digit=None if di is None else DIGITS[di], body_id=bi,
                normal_scalar_n=float(f[slot]), position_w_m=p[slot],
                normal_on_hand_w=normal[slot], separation_m=float(sep[slot]),
                valid=sane, position_region=in_region, on_surface=on_surface,
                position_in_pad=in_region and on_surface, normal_ok=normal_ok,
                accepted=positive and in_region and on_surface and normal_ok,
                source='object_filtered_PhysX_manifold_force_ON_HAND')
            row['raw_contacts'].append(_json_value(record))
            if di is None:
                continue
            pad = row['pads'][hi*5+di]
            if positive:
                pad['raw_load_n'] += float(f[slot])
            if positive and in_region and on_surface:
                pad['load_n'] += float(f[slot]); pad['raw_contact_ids'].append(point_id)
                if not normal_ok:
                    pad['wrong_normal_count'] += 1
                else:
                    pad['accepted_load_n'] += float(f[slot]); pad['contact_count'] += 1
                    pad['_weighted'] += p[slot]*f[slot]
                    pad['_force'] += normal[slot]*f[slot]
        slot_base += len(f)
    for row in rows:
        for pad in row['pads']:
            weighted, force = pad.pop('_weighted'), pad.pop('_force')
            pad['valid'] &= row['measurement_valid']
            if not pad['valid']:
                pad['accepted_load_n'] = 0.
            if pad['valid'] and pad['accepted_load_n'] > 0.:
                pad['position_w_m'] = (weighted/pad['accepted_load_n']).tolist()
                if float(force.norm()) > 1e-12:
                    pad['normal_on_hand_w'] = (force/force.norm()).tolist()
        row['invalid_reasons'] = sorted(set(row['invalid_reasons']))
    return rows


def validate_contact_probe_options(value):
    """Small inference-only probes cannot relax production native evaluation."""
    if not value.get('enabled'):
        return 100
    if (not value.get('diagnostics') or not value.get('checkpoint') or value.get('resume')
            or type(value.get('num_envs')) is not int or not 2 <= value['num_envs'] <= 16
            or value['num_envs'] % 2 or type(value.get('maximum')) is not int
            or not 1 <= value['maximum'] <= 180):
        raise ValueError('contact probe requires diagnostics, immutable inference checkpoint, 2..16 balanced envs, 1..180 control steps, no training resume')
    return 1


def evaluation_minimum(args):
    return validate_contact_probe_options(dict(enabled=getattr(args, 'grasp_contact_probe', False),
        diagnostics=getattr(args, 'grasp_contact_diagnostics', False),
        checkpoint=getattr(args, 'vision_evaluation_checkpoint', None),
        resume=getattr(args, 'resume', None), num_envs=getattr(args, 'num_envs', None),
        maximum=getattr(args, 'vision_evaluation_max_control_steps', None)))


def advance_contact_clock(previous, physics_step, substep, decimation):
    """Bind a local control tick without dividing warmup-offset physics clocks."""
    if (any(type(v) is not int for v in (physics_step, substep, decimation))
            or physics_step < 1 or decimation < 1 or not 1 <= substep <= decimation):
        raise ValueError('positive native step and actual control substep required')
    control = 0
    if previous is not None:
        expected = previous['control_substep'] % decimation + 1
        if physics_step != previous['physics_step']+1 or substep != expected:
            raise ValueError('native physical/control substep clock lost continuity')
        control = previous['control_step'] + int(substep == 1)
    return dict(physics_step=physics_step, control_substep=substep, control_step=control)


def torque_row_fields(packet, index):
    """Keep corrupt per-joint evidence without aborting independent contact data."""
    raw = {}
    for key in ('estimated_torque_nm', 'computed_torque_nm', 'effort_limit_nm', 'torque_valid'):
        value = packet.get(key)
        raw[key] = None if value is None else _json_value(value[index])
    numeric = lambda v: type(v) in (int, float) and math.isfinite(v)
    estimate, limits, mask = (raw[k] for k in ('estimated_torque_nm', 'effort_limit_nm', 'torque_valid'))
    shape = all(isinstance(v, list) and len(v) == 12 for v in (estimate, limits, mask))
    finite = (shape and all(numeric(v) for v in estimate)
              and all(numeric(v) and v > 0. for v in limits) and all(type(v) is bool for v in mask))
    return dict(estimated_torque_nm=estimate if finite else None,
        effort_limit_nm=limits if finite else None, torque_valid=mask if finite else None,
        torque_source=packet.get('torque_source'), raw_torque_diagnostics=raw)


class NativeGraspContactObserver:
    """Native-only streaming observer called by pre-reset evaluation telemetry."""
    def __init__(self, world, env, manifest, output, *, checkpoint_sha256):
        from .tabletop_tensor_world import TensorWorld
        from .tabletop_tensor_env import TensorEnv
        from .tabletop_gpu_isaac import GpuWorld
        from isaaclab.sim.simulation_context import SimulationContext
        from .tabletop_grasp_contact_diagnostics import GraspContactDiagnostics
        if (type(world) is not TensorWorld or type(env) is not TensorEnv
                or type(world.w) is not GpuWorld or type(world.w.sim) is not SimulationContext):
            raise ValueError('native contact observer cannot attest fixtures')
        if not hasattr(world, 'pad_sampler'):
            raise ValueError('native effective pad sampler required')
        self.world, self.env = world, env
        self.trials = dict(enumerate(manifest['trials']))
        if set(self.trials) != set(range(world.n)):
            raise ValueError('complete environment/trial identity required')
        self.output = Path(output); self.closed = set(); self.finished = False; self.clock = None
        self.paths = [self.output/'grasp_contact_substeps.jsonl.gz', self.output/'grasp_contact_events.jsonl.gz']
        if any(p.exists() or p.is_symlink() for p in self.paths):
            raise FileExistsError('preserve prior contact evidence')
        self.streams = [gzip.open(p, 'xt', encoding='utf-8', compresslevel=1) for p in self.paths]
        self.header = dict(schema_version=SCHEMA, checkpoint_sha256=checkpoint_sha256,
            execution='native_isaac_readonly_pre_reset', force_source='PhysX_object_filtered_manifold',
            torque_source=TORQUE_SOURCE, sensor_order=[dict(hand=h, body_id=b, digit=d)
                for h, b, d, _ in world.pad_sampler.bindings],
            native_success_standard=dict(net_lift_m=.03, hold_s=5., fixed_anchor_drift_m=.0005),
            contact_probe_is_production_evaluation=False,
            probe_purpose='historical_checkpoint_observation_smoke_only;not_candidate_closed_loop_validation',
            candidate_to_native_gate='not_implemented;geometry_candidates_are_not_admitted_by_this_probe',
            native_buffer_overflow_observability='intervals_and_capacities_retained;no_attestation_of_absent_upstream_PhysX_buffer_overflow',
            pad_atlas=world.pad_sampler.atlas,
            control_clock='observer_local_control_tick_on_actual_substep_wrap;absolute_PhysX_step_and_sim_time_retained',
            support_qualification='original_scene_unchanged;support_bearing_not_attested;contact_only_is_not_unsupported_hold_or_lift')
        for stream in self.streams:
            stream.write(json.dumps(self.header, allow_nan=False)+'\n')
        self.reducer = GraspContactDiagnostics(self.output.name,
            sink=lambda row: self.streams[0].write(json.dumps(_json_value(row), allow_nan=False, separators=(',', ':'))+'\n'),
            event_sink=lambda row: self.streams[1].write(json.dumps(_json_value(row), allow_nan=False, separators=(',', ':'))+'\n'))

    def sample(self, packet):
        w = self.world; sampler = w.pad_sampler
        raw = getattr(sampler, 'native_contact_data', None)
        if raw is None:
            # Read-only fallback for the non-batched sensor path; no sim step.
            raw = [sensor.contact_physx_view.get_contact_data(dt=w.dt)
                   for _, _, _, sensor in sampler.bindings]
        rows = reduce_pad_contact_buffers([(h, b, d) for h, b, d, _ in sampler.bindings],
            raw, [h.data.body_link_pose_w for h in w.hands], sampler.rows, w.n)
        snapshot = {k: _json_value(v) for k, v in packet.items()}
        episodes = self.env.episodes.detach().cpu().tolist()
        emitted = []
        for ei, row in enumerate(rows):
            if ei in self.closed or not snapshot['live'][ei]:
                continue
            row.update(env_id=ei, episode_id=int(episodes[ei]),
                grasp_style=self.trials[ei]['grasp_style'], live=True,
                done=bool(snapshot['done'][ei]), native_success=bool(snapshot['native_success'][ei]),
                measurement_valid=row['measurement_valid'] and bool(snapshot['measurement_valid'][ei]),
                trial_id=self.trials[ei]['trial_id'])
            for key in ('slip_m', 'slip_valid', 'slip_applicable'):
                value = snapshot.get(key); row[key] = None if value is None else value[ei]
            row.update(torque_row_fields(snapshot, ei))
            row['action_24d'] = snapshot['action'][ei]
            row['action_valid'] = snapshot['action_valid'][ei]
            codes = snapshot.get('slip_reason_codes', {})
            row['slip_reason'] = next((k for k, v in codes.items()
                if v == snapshot['slip_reason_code'][ei]), 'unknown')
            emitted.append(row)
            if row['done']:
                self.closed.add(ei)
        if emitted:
            self.clock = advance_contact_clock(self.clock, int(w._physics_steps),
                                               int(self.env.substep), int(w.decimation))
            self.reducer.observe(dict(**self.clock, sim_time_s=float(w.w.sim.current_time),
                dt_s=float(w.dt), rows=emitted))

    def finish(self, reason='bounded_probe_end'):
        if self.finished:
            raise RuntimeError('contact observer already finished')
        self.finished = True
        try:
            report = self.reducer.finish(reason=reason)
        finally:
            for stream in self.streams:
                stream.close()
        from .tabletop_vision_evaluation_contract import file_sha256
        report.update(self.header, evidence_files={p.name: file_sha256(p) for p in self.paths})
        path = self.output/'grasp_contact_report.json'
        with path.open('x', encoding='utf-8') as stream:
            json.dump(_json_value(report), stream, indent=2, allow_nan=False)
        return report
