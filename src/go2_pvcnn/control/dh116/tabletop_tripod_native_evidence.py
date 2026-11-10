"""Read-only, pre-reset SDK snapshot adapter for staged tripod experiments.

This builder does NOT attest native execution: SDK-shaped fixtures work too.
The native runner must establish exact SDK types, freeze source and authenticate
episode identities separately. Capture uses already-refreshed TensorWorld
state and OWNED raw buffer snapshots, never sensor.update/data/physics writes.
No actor observations or original contact/success thresholds are changed.

All source-body frames and all eleven named measured q/qd are retained. No
command integration, perfect mimic projection or ideal FK replaces readback.
Only object-filtered hand sensors are currently connected. Consequently table
AND original supports coverage remains unknown, including when all 34 hand
sensors have measured zero contacts. This adapter cannot qualify unsupported
hold/lift without additional read-only passive-surface measurements.
"""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import torch

from .linear_coupling import ACTIVE, JOINTS
from .tabletop_grasp_contact_native import reduce_pad_contact_buffers

SCHEMA = 'dh116-tripod-sdk-snapshot-v1'
COVERAGE_SCHEMA = 'dh116-tripod-contact-coverage-v1'
EXECUTION = 'read_only_sdk_snapshot_builder_not_native_attestation'
TORQUE_SOURCE = 'implicit_actuator_estimate_not_hardware_measurement'
PAD_KEYS = ('effective_pad_loads_n', 'effective_pad_vectors_w', 'effective_pad_points_w',
            'effective_pad_valid', 'effective_pad_depth_m', 'rejected_contact_loads_n')


def _owned(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return deepcopy(value)


def _plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return deepcopy(value)


def _pose_matrix(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1) > 1e-6:
        raise ValueError('finite actual body pose with unit quaternion required')
    w, x, y, z = pose[3:] / np.linalg.norm(pose[3:])
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]
    result[:3, 3] = pose[:3]
    return result


def _finite(value, shape):
    return isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(shape) and bool(torch.isfinite(value).all())


def _clock(world):
    sim = world.w.sim
    values = (world._physics_steps, sim.current_time_step_index, float(sim.current_time))
    if any(type(v) is not int or v < 0 for v in values[:2]) or not math.isfinite(values[2]) or values[2] < 0:
        raise ValueError('SDK absolute physics clock required')
    return values


def _intervals(values, width, n):
    """Validate EVERY ownership interval, marking both owners of an overlap."""
    if not isinstance(values, (tuple, list)) or len(values) != width:
        raise ValueError('complete native buffer tuple required')
    tensors = [_owned(v) for v in values]
    if any(not isinstance(v, torch.Tensor) for v in tensors):
        raise ValueError('native tensor buffers required')
    count, start = tensors[-2:]
    if count.numel() != n or start.numel() != n:
        raise ValueError('native environment ownership count mismatch')
    capacity = tensors[0].numel() if width == 6 else tensors[0].numel() // 3
    valid = [True]*n; owners = {}; meta = []
    for ei, (number, begin) in enumerate(zip(count.reshape(-1).tolist(), start.reshape(-1).tolist())):
        integer = (type(number) is int and type(begin) is int)
        ok = integer and 0 <= number <= capacity and 0 <= begin <= capacity-number
        meta.append(dict(count=number, start=begin, capacity=capacity))
        if not ok:
            valid[ei] = False
            continue
        for slot in range(begin, begin+number):
            if slot in owners:
                valid[ei] = valid[owners[slot]] = False
            else:
                owners[slot] = ei
    return tensors, owners, valid, meta


def _close(a, b):
    try:
        a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
        return a.shape == b.shape and bool(torch.isfinite(a).all() and torch.isfinite(b).all()
            and torch.allclose(a, b, rtol=2e-5, atol=2e-6))
    except (TypeError, ValueError):
        return False


class TripodNativeEvidenceBuilder:
    """Capture an immutable batch; identity/genuine reset belongs to runner.

    ``capture`` returns packets for TripodSupportDiagnostics and optional bound
    source-geometry snapshots. The latter contain numpy arrays, intentionally
    separate from JSON packets. Stream packets using a nonfinite-aware encoder;
    raw unknown/nonfinite data is retained here for the diagnostic reducer.
    """
    def __init__(self, world, *, source_models=None):
        self.world = world
        if type(getattr(world, 'n', None)) is not int or world.n < 1 or abs(float(world.dt)-1/480) > 1e-12:
            raise ValueError('480 Hz two-hand TensorWorld state required')
        if len(world.hands) != 2 or len(world.pad_sampler.bindings) != 34:
            raise ValueError('two hands and 34 actual sensor bindings required')
        self.models = None if source_models is None else tuple(source_models)
        if self.models is not None and len(self.models) != 2:
            raise ValueError('both source hand models required')
        self.joint_ids, self.body_ids, self.base_ids = [], [], []
        if any(type(h) is not int or type(b) is not int
               for h, b, _, _ in world.pad_sampler.bindings):
            raise ValueError('native sensor hand/body bindings require actual integer IDs')
        self.bindings = [(h, b, d) for h, b, d, _ in world.pad_sampler.bindings]
        if len(set((h, b) for h, b, _ in self.bindings)) != 34:
            raise ValueError('34 unique actual sensor body bindings required')
        if (any(h not in (0, 1) or not 0 <= b < len(world.hands[h].body_names)
                or (d is not None and (type(d) is not int or not 0 <= d < 5))
                for h, b, d in self.bindings)
                or any(sum(h == hi for h, _, _ in self.bindings) != 17 for hi in range(2))
                or any(sorted(d for h, _, d in self.bindings if h == hi and d is not None) != list(range(5))
                       for hi in range(2))):
            raise ValueError('seventeen actual sensor links and five distinct pads per hand required')
        for hi, hand in enumerate(world.hands):
            names = list(hand.joint_names); bodies = list(hand.body_names)
            if len(names) != len(set(names)) or any(names.count(name) != 1 for name in JOINTS):
                raise ValueError('exactly one of each actual named finger joint required')
            if len(bodies) != len(set(bodies)):
                raise ValueError('unique actual SDK body names required')
            if bodies.count('base_link') != 1:
                raise ValueError('one actual named base_link wrist frame required')
            self.base_ids.append(bodies.index('base_link'))
            self.joint_ids.append([names.index(k) for k in JOINTS])
            if self.models is not None:
                source_paths = sorted(self.models[hi].bodies)
                basenames = [p.rsplit('/', 1)[-1] for p in source_paths]
                if len(basenames) != len(set(basenames)) or any(k not in bodies for k in basenames):
                    raise ValueError('complete unambiguous source-to-SDK body mapping required')
                self.body_ids.append({path: bodies.index(name) for path, name in zip(source_paths, basenames)})
        # Static rows are geometry gates only; never manufacture a contact.
        self.atlas_rows = _owned(world.pad_sampler.rows)

    def _raw(self, clock, event_kind):
        w = self.world; n = w.n
        snapshot = getattr(w, 'tripod_contact_snapshot', None)
        available = (event_kind == 'physics' and isinstance(snapshot, dict)
            and snapshot.get('complete') is True and snapshot.get('tensor_world_physics_steps') == clock[0])
        normals = snapshot.get('raw_contacts') if available else None
        friction = snapshot.get('raw_friction') if available else None
        available = available and isinstance(normals, list) and isinstance(friction, list) and len(normals) == len(friction) == 34
        blank = [dict(raw_contacts=None, raw_friction_contacts=None, raw_contact_buffers=None,
            raw_friction_buffers=None, normal_complete=False, friction_complete=False,
            ownership_valid=False,capacity_saturated=False,
            invalid_reasons=['absent_stale_or_reset_native_snapshot']) for _ in range(n)]
        if not available:
            return blank
        masks = snapshot.get('valid_envs')
        if not isinstance(masks, torch.Tensor) or masks.shape != (n,) or masks.dtype != torch.bool:
            return blank
        rows = [dict(raw_contacts=[], raw_friction_contacts=[], raw_contact_buffers=[], raw_friction_buffers=[],
            normal_complete=True,friction_complete=True,ownership_valid=True,
            capacity_saturated=False,invalid_reasons=[]) for _ in range(n)]
        # Existing normal reducer independently applies the ORIGINAL source pad
        # region/normal gate to actual poses and the owned raw buffer snapshot.
        reduced = None
        try:
            reduced = reduce_pad_contact_buffers(self.bindings, normals,
                [w.poses[:, h] for h in range(2)], self.atlas_rows, n)
        except (ValueError, TypeError, RuntimeError, IndexError):
            pass
        for si, binding in enumerate(self.bindings):
            hi, bi, di = binding
            for kind, buffers, width in (('normal', normals, 6), ('friction', friction, 4)):
                try:
                    values, owners, ownership, meta = _intervals(buffers[si], width, n)
                    capacity = values[0].numel() if width == 6 else values[0].numel()//3
                    if width == 6:
                        if (values[1].shape != (capacity, 3) or values[2].shape != (capacity, 3)
                                or values[3].numel() != capacity):
                            raise ValueError('normal native capacity mismatch')
                    elif values[0].shape != (capacity, 3) or values[1].shape != (capacity, 3):
                        raise ValueError('friction native capacity mismatch')
                except (ValueError, TypeError, RuntimeError, IndexError):
                    for row in rows:
                        row[kind+'_complete'] = row['ownership_valid'] = False
                        row['invalid_reasons'].append(kind+'_buffer_schema_invalid')
                    continue
                field = 'raw_contact_buffers' if kind == 'normal' else 'raw_friction_buffers'
                # A full view may be exactly full or upstream-truncated. With
                # no SDK overflow counter we cannot distinguish those cases.
                # Censor all possible env owners of that view, retaining all
                # observed rows and the unchanged original sensor capacity.
                saturated=(capacity==0 or sum(v['count'] for v in meta
                    if type(v['count']) is int and v['count']>0)>=capacity)
                for ei, row in enumerate(rows):
                    row[field].append(dict(sensor_index=si,hand_index=hi,body_id=bi,
                        capacity_saturated=saturated,**meta[ei]))
                    if saturated:
                        row[kind+'_complete']=False;row['capacity_saturated']=True
                        row['invalid_reasons'].append(kind+'_capacity_saturated_overflow_unknown')
                    if not ownership[ei]:
                        row[kind+'_complete'] = row['ownership_valid'] = False
                        row['invalid_reasons'].append(kind+'_ownership_invalid')
                for slot, ei in owners.items():
                    if kind == 'normal':
                        f, point, normal, separation = (values[j].reshape(-1, 3)[slot] if j in (1, 2)
                            else values[j].reshape(-1)[slot] for j in range(4))
                        finite = bool(torch.isfinite(f) and torch.isfinite(point).all()
                            and torch.isfinite(normal).all() and torch.isfinite(separation))
                        # Match the unchanged native sampler: finite signed
                        # compliant-contact release terms are measurements,
                        # not sensor faults. Only its positive compression
                        # region gate may accept a bearing contact.
                        sane = finite and abs(float(normal.norm())-1) <= .05
                        accepted = False
                        if reduced is not None:
                            found = next((r for r in reduced[ei]['raw_contacts'] if r['sensor_index'] == si and r['raw_slot'] == slot), None)
                            accepted = bool(found is not None and found['accepted'] and sane and ownership[ei])
                        record = dict(sensor_index=si, hand_index=hi, body_id=bi, digit_index=di,
                            raw_slot=slot, raw_count=meta[ei]['count'], raw_start=meta[ei]['start'],
                            normal_scalar_n=float(f), force_on_hand_w_n=(f*normal).tolist(),
                            point_w_m=point.tolist(), normal_w=normal.tolist(), separation_m=float(separation),
                            depth_m=max(0., -float(separation)) if math.isfinite(float(separation)) else float('nan'),
                            accepted=accepted, valid=sane and ownership[ei],
                            source='object_filtered_PhysX_manifold_force_ON_HAND')
                        rows[ei]['raw_contacts'].append(record)
                    else:
                        force, point = values[0][slot], values[1][slot]
                        sane = bool(torch.isfinite(force).all() and torch.isfinite(point).all())
                        rows[ei]['raw_friction_contacts'].append(dict(sensor_index=si, hand_index=hi,
                            body_id=bi, digit_index=di, raw_slot=slot, raw_count=meta[ei]['count'],
                            raw_start=meta[ei]['start'], force_on_hand_w_n=force.tolist(), point_w_m=point.tolist(),
                            valid=sane and ownership[ei], source='object_filtered_PhysX_friction_force_ON_HAND'))
                    if not sane:
                        rows[ei][kind+'_complete'] = False
                        rows[ei]['invalid_reasons'].append(kind+'_owned_nonfinite_or_invalid')
        for ei, row in enumerate(rows):
            if reduced is None or not reduced[ei]['measurement_valid']:
                row['normal_complete'] = False
                row['invalid_reasons'].append('normal_source_region_reduction_invalid')
            if not bool(masks[ei]):
                row['normal_complete'] = row['friction_complete'] = False
                row['invalid_reasons'].append('native_snapshot_env_invalid')
                # Invalidated cache is an old episode's evidence, not measured
                # no-contact in a new episode. Never expose it as current raw.
                row['raw_contacts'] = row['raw_friction_contacts'] = None
            row['invalid_reasons'] = sorted(set(row['invalid_reasons']))
        return rows

    def _torques(self):
        w = self.world
        fields = {}
        for name, attr in (('estimated_torque_nm', 'applied_torque'), ('computed_torque_nm', 'computed_torque')):
            try:
                values = [getattr(h.data, attr)[:, ids] for h, ids in zip(w.hands, w.w.ids)]
                tensor = torch.cat(values, -1).detach().cpu().double().clone()
                fields[name] = tensor if tensor.shape == (w.n, 12) else None
            except (AttributeError, TypeError, IndexError, RuntimeError):
                fields[name] = None
        try:
            drive = torch.cat([h.root_physx_view.get_dof_max_forces()[:, ids] for h, ids in zip(w.hands, w.w.ids)], -1)
            fields['drive_effort_limit_nm'] = _owned(drive) if drive.shape == (w.n, 12) else None
        except (AttributeError, TypeError, IndexError, RuntimeError):
            fields['drive_effort_limit_nm'] = None
        limit = getattr(getattr(w, 'parameters', None), 'effort_limit_nm', None)
        fields['effort_limit_nm'] = (torch.full((w.n, 12), float(limit), dtype=torch.float64)
            if type(limit) in (float, int) and math.isfinite(limit) and limit > 0 else None)
        command = getattr(w, 'effort_feedforward', None)
        fields['explicit_effort_command_nm'] = (_owned(command).reshape(w.n, 12)
            if isinstance(command, torch.Tensor) and command.shape == (w.n, 2, 6) else None)
        for source, target in (('estimated_torque_nm', 'torque_valid'), ('computed_torque_nm', 'computed_torque_valid')):
            fields[target] = None if fields[source] is None else torch.isfinite(fields[source])
        return fields

    @torch.no_grad()
    def capture(self, *, episode_ids, stage_id, control_step, control_substep, live, terminal, fatal,
                native_success, wrists_fixed, left_targets_fixed, object_external_force_applied,
                induced_disconnect, event_kind='physics'):
        w = self.world; n = w.n; clock = _clock(w)
        # Neutral read-only state capture is not an A..E staged experiment.
        if stage_id not in ('A', 'B', 'C', 'D', 'E', 'MPC_MEASUREMENT') or event_kind not in ('physics', 'reset'):
            raise ValueError('stage A..E/MPC_MEASUREMENT and event kind physics/reset required')
        if type(control_step) is not int or control_step < 0 or type(control_substep) is not int or not (
                0 <= control_substep <= 8 if event_kind == 'reset' else 1 <= control_substep <= 8):
            raise ValueError('explicit current control clock required')
        ids = _plain(episode_ids)
        if not isinstance(ids, list) or len(ids) != n or any(type(v) is not int or v < 0 for v in ids):
            raise ValueError('physical episode IDs must be nonnegative integers per environment')
        flags = {}
        for key, value in dict(live=live, terminal=terminal, fatal=fatal, native_success=native_success).items():
            value = _plain(value)
            unknown_allowed = stage_id == 'MPC_MEASUREMENT' and key in ('fatal', 'native_success')
            if not isinstance(value, list) or len(value) != n or any(
                    type(v) is not bool and not (unknown_allowed and v is None) for v in value):
                raise ValueError('per-environment bool flag required: '+key)
            flags[key] = value
        for key, value in dict(wrists_fixed=wrists_fixed, left_targets_fixed=left_targets_fixed,
                object_external_force_applied=object_external_force_applied, induced_disconnect=induced_disconnect).items():
            if type(value) is not bool and not (stage_id == 'MPC_MEASUREMENT' and value is None):
                raise ValueError('explicit runner experiment flag required: '+key)
        # All quantities come from this refresh, before any terminal reset.
        cached_q, cached_qd, cached_poses = [_owned(getattr(w, k)) for k in ('q', 'qd', 'poses')]
        # Read SDK DATA directly, without update/refresh. A mismatch against the
        # already-refreshed TensorWorld packet is not silently assigned its clock.
        q, qd, poses = [torch.stack([_owned(getattr(h.data, name)).double()
            for h in w.hands], 1) for name in ('joint_pos', 'joint_vel', 'body_link_pose_w')]
        cached_roots,commands=[_owned(getattr(w,k)) for k in ('roots','commands')]
        roots=torch.stack([poses[:,h,bi] for h,bi in enumerate(self.base_ids)],1)
        cached_wp, cached_wq, cached_wv, bottom, weight, vertical = [_owned(getattr(w, k)) for k in
            ('wp', 'wq', 'wv', 'bottom', 'object_weight_n', 'vertical_components')]
        # Object-bound approach targets need the same SDK readback protection
        # as hand FK. A current global clock cannot certify a stale wire cache.
        # No cached pose is substituted when an SDK field is unavailable.
        try:
            wp,wq,wv=[_owned(getattr(w.wire.data,name)).double() for name in
                ('root_pos_w','root_quat_w','root_vel_w')]
            object_sdk_available=all(isinstance(v,torch.Tensor) and tuple(v.shape)==shape
                for v,shape in ((wp,(n,3)),(wq,(n,4)),(wv,(n,6))))
        except (AttributeError,TypeError,RuntimeError):
            wp=wq=wv=None;object_sdk_available=False
        pads = {k: _owned(w.pad_measurements.get(k)) for k in PAD_KEYS}
        raw = self._raw(clock, event_kind); torques = self._torques()
        packets = []; geometry_snapshots = []
        for ei in range(n):
            r = raw[ei]
            identity = dict(env_id=ei, episode_id=ids[ei], tensor_world_physics_steps=clock[0],
                simulation_step_index=clock[1], simulation_time_s=clock[2], control_step=control_step,
                control_substep=control_substep, stage_id=stage_id, event_kind=event_kind)
            hands = []; bound = []; kin_valid = True
            for hi, hand in enumerate(w.hands):
                measured_q = q[ei, hi, self.joint_ids[hi]]
                measured_qd = qd[ei, hi, self.joint_ids[hi]]
                row = dict(side=('left', 'right')[hi], joint_names=list(JOINTS),
                    full_joint_positions_rad=measured_q.tolist(), full_joint_velocities_rad_s=measured_qd.tolist(),
                    body_names=list(hand.body_names), body_link_pose_w=poses[ei, hi].tolist(),
                    base_link_pose_w=roots[ei, hi].tolist(), command_active_rad=commands[ei].reshape(2, 6)[hi].tolist(),
                    tensorworld_cached_base_link_pose_w=cached_roots[ei,hi].tolist(),
                    source_body_poses_w=None, source_hash=None)
                cache_matches = (torch.equal(cached_q[ei, hi], q[ei, hi])
                    and torch.equal(cached_qd[ei, hi], qd[ei, hi])
                    and torch.equal(cached_poses[ei, hi], poses[ei, hi])
                    and torch.equal(cached_roots[ei,hi],roots[ei,hi]))
                row['tensorworld_cache_matches_sdk'] = cache_matches
                known = (_finite(measured_q, (11,)) and _finite(measured_qd, (11,))
                    and bool(torch.isfinite(poses[ei, hi]).all()) and bool(torch.isfinite(roots[ei, hi]).all())
                    and bool(((poses[ei, hi, :, 3:].norm(dim=-1)-1).abs() <= 1e-6).all())
                    and abs(float(roots[ei, hi, 3:].norm())-1) <= 1e-6 and cache_matches)
                snapshot = None
                if self.models is not None:
                    model = self.models[hi]; row['source_hash'] = model.source_hash
                    try:
                        mapped = {path: _pose_matrix(poses[ei, hi, bi].numpy()) for path, bi in self.body_ids[hi].items()}
                        if not _close(mapped[model.root], _pose_matrix(roots[ei, hi].numpy())):
                            raise ValueError('actual source base and measured carriage base differ')
                        row['source_body_poses_w'] = _plain(mapped)
                        snapshot = model.bind(measured_q.numpy(), mapped)
                    except (ValueError, TypeError, KeyError):
                        known = False
                if not known:
                    snapshot = None
                row['valid'] = known; kin_valid &= known; hands.append(row); bound.append(snapshot)
            object_cache_matches = (object_sdk_available
                and torch.equal(cached_wp[ei],wp[ei]) and torch.equal(cached_wq[ei],wq[ei])
                and torch.equal(cached_wv[ei],wv[ei]))
            object_valid = (object_sdk_available and object_cache_matches
                and _finite(wp[ei], (3,)) and _finite(wq[ei], (4,))
                and abs(float(wq[ei].norm())-1) <= 1e-6 and _finite(wv[ei], (6,))
                and math.isfinite(float(bottom[ei])) and math.isfinite(float(weight[ei])) and float(weight[ei]) > 0)
            kin_valid &= object_valid
            if not kin_valid:
                bound = [None, None]
            state = dict(**identity,hands=hands,valid=kin_valid,
                object_pose_w=torch.cat((wp[ei],wq[ei])).tolist() if object_sdk_available else None,
                object_velocity_w=wv[ei].tolist() if object_sdk_available else None,
                object_sdk_readback_available=object_sdk_available,
                object_tensorworld_cache_matches_sdk=object_cache_matches,
                cached_object_readback=dict(position_w_m=_plain(cached_wp[ei]),
                    quaternion_wxyz=_plain(cached_wq[ei]),velocity_w=_plain(cached_wv[ei])),
                native_attested=False,execution=EXECUTION)
            # Digest binds all measured state and commands, not just episode ID.
            digest = hashlib.sha256(json.dumps(state, sort_keys=True, separators=(',', ':'), allow_nan=True).encode()).hexdigest()
            state['snapshot_identity'] = digest
            for snapshot in bound:
                if snapshot is not None:
                    snapshot.update(snapshot_identity=digest, **identity)
            coverage = dict(schema=COVERAGE_SCHEMA,
                **{k: identity[k] for k in ('env_id', 'episode_id', 'tensor_world_physics_steps')},
                normal_sensor_count=len(r['raw_contact_buffers']) if r['raw_contact_buffers'] is not None else 0,
                friction_sensor_count=len(r['raw_friction_buffers']) if r['raw_friction_buffers'] is not None else 0,
                normal_complete=r['normal_complete'], friction_complete=r['friction_complete'],
                ownership_valid=r['ownership_valid'],capacity_saturated=r['capacity_saturated'],
                invalid_reasons=r['invalid_reasons'],
                normal_vertical_match=False, friction_vertical_match=False, accepted_normal_matches_effective=False,
                rejected_load_matches_effective=False,
                upstream_physx_overflow_absence_attested=False)
            if r['raw_contacts'] is not None and r['normal_complete']:
                total = [sum(-x['force_on_hand_w_n'][2] for x in r['raw_contacts'] if x['hand_index'] == h) for h in range(2)]
                coverage['normal_vertical_match'] = _close(total, vertical[ei, :, 0])
                loads = torch.zeros(2, 5, dtype=torch.float64); vectors = torch.zeros(2, 5, 3, dtype=torch.float64)
                for record in r['raw_contacts']:
                    if record['accepted'] and record['digit_index'] is not None:
                        h, d = record['hand_index'], record['digit_index']
                        loads[h, d] += record['normal_scalar_n']; vectors[h, d] += torch.tensor(record['force_on_hand_w_n'], dtype=torch.float64)
                coverage['accepted_normal_matches_effective'] = (pads['effective_pad_loads_n'] is not None
                    and pads['effective_pad_vectors_w'] is not None and _close(loads, pads['effective_pad_loads_n'][ei])
                    and _close(vectors, pads['effective_pad_vectors_w'][ei]))
                # Original PadContactSampler total is sum(abs(signed f))
                # across ALL links, minus accepted positive pad compression.
                # Using signed cancellation here would erase release loads.
                absolute_total=torch.tensor([sum(abs(x['normal_scalar_n'])
                    for x in r['raw_contacts'] if x['hand_index']==h) for h in range(2)],dtype=torch.float64)
                rejected=(absolute_total-loads.sum(-1)).clamp_min(0.)
                coverage['rejected_load_matches_effective']=(pads['rejected_contact_loads_n'] is not None
                    and _close(rejected,pads['rejected_contact_loads_n'][ei]))
            if r['raw_friction_contacts'] is not None and r['friction_complete']:
                total = [sum(-x['force_on_hand_w_n'][2] for x in r['raw_friction_contacts'] if x['hand_index'] == h) for h in range(2)]
                coverage['friction_vertical_match'] = _close(total, vertical[ei, :, 1])
            packet = dict(**identity, dt_s=float(w.dt), **{k: v[ei] for k, v in flags.items()},
                wrists_fixed=wrists_fixed, left_targets_fixed=left_targets_fixed,
                object_external_force_applied=object_external_force_applied, induced_disconnect=induced_disconnect,
                measurement_valid=(kin_valid and r['normal_complete'] and r['ownership_valid']
                    and not r['capacity_saturated']
                    and coverage['normal_vertical_match'] and coverage['accepted_normal_matches_effective']
                    and coverage['rejected_load_matches_effective']),
                wrist_pose=roots[ei].tolist(), object_pose_w=state['object_pose_w'],
                object_velocity_w=state['object_velocity_w'], bottom_m=float(bottom[ei]),
                object_weight_n=float(weight[ei]), vertical_components_n=vertical[ei].tolist(),
                raw_contacts=r['raw_contacts'], raw_friction_contacts=r['raw_friction_contacts'],
                raw_contact_buffers=r['raw_contact_buffers'], raw_friction_buffers=r['raw_friction_buffers'],
                contact_evidence_coverage=coverage, passive_support_coverage=None,
                object_table_support_n=None, object_table_measurement_valid=False,
                actual_kinematics=state, torque_source=TORQUE_SOURCE,
                **{k: None if v is None or event_kind == 'reset' else _plain(v[ei]) for k, v in pads.items()},
                **{k: None if v is None else _plain(v[ei]) for k, v in torques.items()})
            packets.append(packet); geometry_snapshots.append(bound)
        if _clock(w) != clock:
            raise RuntimeError('physics clock changed during pre-reset capture; mixed snapshot refused')
        return dict(schema=SCHEMA, execution=EXECUTION, packets=packets, geometry_snapshots=geometry_snapshots)
