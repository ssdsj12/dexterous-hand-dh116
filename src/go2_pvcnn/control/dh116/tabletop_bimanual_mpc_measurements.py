"""Read-only privileged Teacher state from already-refreshed native evidence.

No update/refresh/step, actuation, observation, training or solver path lives
here. SDK DATA reads are cache reads bracketed by the actual World/SDK clock;
they are not independent raw sensor getter clocks. The original native reader
owns the normal/friction shared-buffer cloning and the original builder owns
contact intervals, body bindings, pad gates, completeness and cache checks.
Neither this adapter nor CPU SDK-shaped fixtures attest native execution.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import math

import numpy as np
import torch

from .linear_coupling import ACTIVE, JOINTS
from .tabletop_tripod_native_evidence import TripodNativeEvidenceBuilder
from .tabletop_bimanual_tracking_guard import read_common_pause_targets
from .tabletop_bimanual_mpc_state import (
    FIELD_SPECS, SampleClock, StateField, make_mpc_state, missing_field, missing_fields,
)


def point_velocity(com_twist, com_position, point_position):
    """World velocity at a rigid material point: v_COM + omega x r (SI)."""
    twist, com, point = [np.asarray(v, dtype=np.float64) for v in (com_twist, com_position, point_position)]
    if twist.shape != (6,) or com.shape != (3,) or point.shape != (3,) or not all(
            np.isfinite(v).all() for v in (twist, com, point)):
        raise ValueError('finite COM twist[6], COM[3] and world point[3] required')
    return (twist[:3] + np.cross(twist[3:], point-com)).tolist()


def _clock(world):
    sim = world.w.sim
    result = (world._physics_steps, sim.current_time_step_index, float(sim.current_time))
    if any(type(v) is not int or v < 0 for v in result[:2]) or not math.isfinite(result[2]) or result[2] < 0:
        raise ValueError('actual World and SDK physics clock required')
    return result


def _plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return deepcopy(value)


def _finite(value, shape):
    try:
        v = np.asarray(value, dtype=np.float64)
        return v.shape == shape and bool(np.isfinite(v).all())
    except (TypeError, ValueError):
        return False


def _env_finite(value, env, shape):
    try:
        return _finite(value[env], shape)
    except (TypeError, IndexError):
        return False


def _point(pose, local):
    pose = np.asarray(pose, dtype=np.float64)
    local = np.asarray(local, dtype=np.float64)
    if pose.shape != (7,) or local.shape != (3,) or not np.isfinite(pose).all() or not np.isfinite(local).all():
        raise ValueError('actual body pose and CAD point required')
    w, xyz = pose[3], pose[4:]
    if abs(np.linalg.norm(pose[3:])-1.) > 1e-6:
        raise ValueError('actual unit body quaternion required')
    cross = 2*np.cross(xyz, local)
    return (local + w*cross + np.cross(xyz, cross) + pose[:3]).tolist()


class BimanualMPCMeasurementBridge:
    """Capture immutable per-env states; runner owns genuine reset/run proof.

    Configure ``world.capture_tripod_contact_evidence=True`` before the normal
    physics refresh, with the existing native sensor reader and pad sampler.
    This class never changes that flag or refreshes an old contact cache.
    ``last_evidence`` retains original raw contacts/coverage and additional
    getter availability/brackets for the runner's independently frozen record.
    Setup flags are optional runner declarations, never independent evidence.
    """
    def __init__(self, world, *, run_uuid, source_models=None, refresh_receipt_provider=None):
        if not isinstance(run_uuid, str) or not run_uuid.strip():
            raise ValueError('explicit unique runner run_uuid required')
        self.world = world
        self.run_uuid = run_uuid
        self.builder = TripodNativeEvidenceBuilder(world, source_models=source_models)
        if refresh_receipt_provider is not None and not callable(refresh_receipt_provider):
            raise ValueError('refresh receipt provider must be callable or absent')
        self._refresh_receipt_provider = refresh_receipt_provider
        self._last_refresh_sequence = 0
        try:
            self._refresh_method_sha256 = hashlib.sha256(
                inspect.getsource(type(world).refresh).encode('utf-8')).hexdigest()
        except (AttributeError, TypeError, OSError):
            self._refresh_method_sha256 = None
        # Establish only a baseline, not freshness of the construction step.
        # A subsequent physical step and matching DATA generation advance are
        # necessary before any cached dynamic field can be marked current.
        baseline_clock = _clock(world)
        self._data_sources = [(f'hands[{h}]', hand.data) for h, hand in enumerate(world.hands)] + [('wire', world.wire.data)]
        self._data_baselines = {key: (getattr(data, '_sim_timestamp', None), baseline_clock)
            for key, data in self._data_sources}
        if _clock(world) != baseline_clock:
            raise RuntimeError('physics clock changed while anchoring SDK DATA generations')
        self._previous = {}
        self.last_evidence = None

    def _receipt_evidence(self, clock):
        """Validate the runner's observation of the original refresh return.

        The runner authenticates exact native class/method identity before
        installing its read-only wrapper. A fixture can reproduce this format;
        the receipt never attests native execution or an independent backend.
        """
        result = dict(enabled=self._refresh_receipt_provider is not None, source_valid=False,
            receipt=None, expected_original_method_sha256=self._refresh_method_sha256,
            missing_reason='no receipt provider; conservative constructor-generation checks apply')
        if self._refresh_receipt_provider is None:
            return result
        try:
            receipt = _plain(self._refresh_receipt_provider())
            result['receipt'] = receipt
            if not isinstance(receipt, dict):
                raise ValueError('actual refresh receipt dictionary required')
            sequence = receipt.get('sequence')
            if type(sequence) is not int or sequence <= self._last_refresh_sequence:
                raise ValueError('refresh receipt sequence must be positive and newly observed')
            if receipt.get('dt') != float(self.world.dt):
                raise ValueError('refresh receipt must retain original physics dt')
            if (self._refresh_method_sha256 is None
                    or receipt.get('original_method_sha256') != self._refresh_method_sha256):
                raise ValueError('refresh receipt original method source SHA mismatch')
            if receipt.get('source_scope') != 'actual original TensorWorld.refresh returned;SDK_DATA generation only':
                raise ValueError('actual original refresh-return source scope required')
            for name in ('world_sdk_clock_before', 'world_sdk_clock_after'):
                value = receipt.get(name)
                if (not isinstance(value, list) or len(value) != 3
                        or any(type(v) is not int for v in value[:2]) or not _finite(value, (3,))
                        or value != list(clock)):
                    raise ValueError('refresh receipt clock must remain unchanged and match current World/SDK clock')
            before, after = receipt.get('data_generation_before'), receipt.get('data_generation_after')
            keys = {key for key, _ in self._data_sources}
            if not isinstance(before, dict) or not isinstance(after, dict) or set(before) != keys or set(after) != keys:
                raise ValueError('all actual DATA generations required in refresh receipt')
            for key, data in self._data_sources:
                values = (before[key], after[key], getattr(data, '_sim_timestamp', None))
                if (not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values)
                        or not math.isclose(values[0]+float(self.world.dt), values[1], abs_tol=1e-10, rel_tol=1e-10)
                        or values[1] != values[2]):
                    raise ValueError('refresh receipt DATA generation must advance once by dt and equal current DATA')
            result.update(source_valid=True, missing_reason=None)
        except Exception as exc:
            result['missing_reason'] = type(exc).__name__ + ': ' + str(exc)
        return result

    def _generation_evidence(self, clock):
        evidence = {}
        for key, data in self._data_sources:
            baseline_timestamp, baseline_clock = self._data_baselines[key]
            timestamp = getattr(data, '_sim_timestamp', None)
            known = all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                for v in (baseline_timestamp, timestamp))
            delta = clock[0]-baseline_clock[0]
            physical_advance = (delta > 0 and clock[1]-baseline_clock[1] == delta
                and math.isclose(clock[2]-baseline_clock[2], delta*float(self.world.dt), abs_tol=1e-7, rel_tol=1e-7))
            expected = baseline_timestamp+delta*float(self.world.dt) if known else None
            current = known and physical_advance and math.isclose(timestamp, expected, abs_tol=1e-9, rel_tol=1e-9)
            evidence[key] = dict(source=key+'.data._sim_timestamp', baseline_timestamp=baseline_timestamp,
                baseline_clock=list(baseline_clock), data_timestamp=timestamp, expected_timestamp=expected,
                clock=list(clock), source_valid=current,
                scope='DATA_generation_advance_since_construction_bound_to_physical_steps_not_independent_backend_read',
                missing_reason=None if current else 'DATA generation has not advanced with actual World/SDK physics since bridge construction')
        return evidence

    def _read(self, source, getter, records, *, data=None, property_name=None, generation_valid=False):
        before = _clock(self.world)
        try:
            result = _plain(getter())
            reason = None
        except (AttributeError, TypeError, ValueError, RuntimeError, IndexError) as exc:
            result = None
            reason = type(exc).__name__ + ': ' + str(exc)
        after = _clock(self.world)
        mass = source.endswith('get_masses')
        pose = source.endswith('pose_w')
        data_timestamp = getattr(data, '_sim_timestamp', None)
        cache_timestamp = getattr(getattr(data, '_'+str(property_name), None), 'timestamp', None)
        timestamps_known = all(type(v) in (float, int) and math.isfinite(v) and v >= 0
            for v in (data_timestamp, cache_timestamp))
        cache_fresh = timestamps_known and data_timestamp == cache_timestamp
        source_valid = result is not None and (mass or (cache_fresh and generation_valid))
        if result is not None and not source_valid:
            reason = 'SDK DATA cache freshness absent or stale; bracket alone does not establish freshness'
        valid_envs = []
        for ei in range(self.world.n):
            try:
                valid_envs.append(source_valid and bool(np.isfinite(np.asarray(result[ei], dtype=np.float64)).all()))
            except (TypeError, ValueError, IndexError):
                valid_envs.append(False)
        records.append(dict(source=source, clock_before=list(before), clock_after=list(after),
            available=result is not None, missing_reason=reason, value=result, valid_envs=valid_envs,
            unit='kg' if mass else ('m|quaternion_wxyz' if pose else 'm/s|rad/s'),
            frame='not_applicable' if mass else 'world',
            origin='not_applicable' if mass else ('body_com' if source.startswith('hands') else 'object_com'),
            read_kind='PhysX_tensor_mass_getter' if mass else 'SDK_data_property_cache',
            source_valid=source_valid, data_sim_timestamp=data_timestamp, cache_timestamp=cache_timestamp,
            cache_fresh_to_data_generation=cache_fresh if not mass else None,
            data_generation_bound_to_physics=generation_valid if not mass else None,
            freshness_scope='cache_timestamp_and_DATA_generation_advance_vs_physical_clock_not_independent_backend_proof',
            independent_backend_read_attested=False,
            clock_scope='World_SDK_bracket_of_SDK_DATA_or_mass_read_not_independent_raw_getter_clock'))
        if before != after:
            raise RuntimeError('physics clock changed during extra SDK capture')
        return result if source_valid else None

    def _identities(self, episode_uids, episode_ids, reset_generations, policy_step, reset, clock, live):
        n = self.world.n
        if (not isinstance(episode_uids, (list, tuple)) or len(episode_uids) != n
                or any(not isinstance(v, str) or not v.strip() for v in episode_uids)
                or len(set(episode_uids)) != n):
            raise ValueError('distinct explicit physical episode_uids per env required')
        if (not isinstance(reset_generations, (list, tuple)) or len(reset_generations) != n
                or any(type(v) is not int or v < 0 for v in reset_generations)):
            raise ValueError('explicit nonnegative per-env reset generations required')
        if type(reset) is not bool:
            raise ValueError('explicit reset event bool required')
        if (not isinstance(episode_ids, (list, tuple)) or len(episode_ids) != n
                or any(type(v) is not int or v < 0 for v in episode_ids)):
            raise ValueError('explicit physical episode IDs required')
        if not isinstance(live, (list, tuple)) or len(live) != n or any(type(v) is not bool for v in live):
            raise ValueError('explicit per-env live bool mask required')
        for ei, (uid, generation) in enumerate(zip(episode_uids, reset_generations)):
            if not live[ei]:
                continue
            previous = self._previous.get(ei)
            if previous is None:
                continue
            prior_uid, prior_generation, prior_clock, prior_terminal, prior_episode, prior_policy = previous
            if prior_terminal:
                raise ValueError('closed first-terminal episode cannot supply subsequent live state')
            if policy_step < prior_policy:
                raise ValueError('policy clock regressed')
            delta = clock[0]-prior_clock[0]
            if (clock[1]-prior_clock[1] != delta
                    or not math.isclose(clock[2]-prior_clock[2], delta*float(self.world.dt), abs_tol=1e-7, rel_tol=1e-7)):
                raise ValueError('relative World/SDK clocks changed')
            if generation == prior_generation:
                if uid != prior_uid or episode_ids[ei] != prior_episode:
                    raise ValueError('episode changed without actual reset generation')
                if reset:
                    raise ValueError('reset event requires changed episode and reset generation')
                if clock[0] <= prior_clock[0] or clock[1] <= prior_clock[1] or clock[2] <= prior_clock[2]:
                    raise ValueError('stale or repeated physics sample refused')
            elif generation != prior_generation+1 or uid == prior_uid or episode_ids[ei] == prior_episode:
                raise ValueError('episode/reset generation transition invalid')
            elif clock[0] < prior_clock[0] or clock[1] < prior_clock[1] or clock[2] < prior_clock[2]:
                raise ValueError('World or SDK clock regressed across reset')
            elif not reset and any(a <= b for a, b in zip(clock, prior_clock)):
                raise ValueError('physics after reset must advance beyond the pre-reset clock')

    @torch.no_grad()
    def capture(self, *, episode_uids, episode_ids, reset_generations, policy_step,
                control_substep, live, terminal, reset=False, fatal=None, native_success=None,
                wrists_fixed=None, left_targets_fixed=None, object_external_force_applied=None,
                induced_disconnect=None):
        w = self.world
        self.last_evidence = None
        start = _clock(w)
        live, terminal, episode_ids = _plain(live), _plain(terminal), _plain(episode_ids)
        self._identities(episode_uids, episode_ids, reset_generations, policy_step, reset, start, live)
        data_generations = self._generation_evidence(start)
        receipt_evidence = self._receipt_evidence(start)
        if self._refresh_receipt_provider is not None:
            # A configured provider is authoritative for this source path.
            # Invalid/absent receipts never fall back to constructor offsets.
            for key, row in data_generations.items():
                row.update(source_valid=receipt_evidence['source_valid'],
                    constructor_generation_expected_timestamp=row['expected_timestamp'],
                    expected_timestamp=(receipt_evidence['receipt']['data_generation_after'][key]
                        if receipt_evidence['source_valid'] else None),
                    scope='validated_original_refresh_return_receipt_bound_to_current_World_SDK_and_DATA',
                    missing_reason=receipt_evidence['missing_reason'])
        declarations = dict(wrists_fixed=wrists_fixed, left_targets_fixed=left_targets_fixed,
            object_external_force_applied=object_external_force_applied, induced_disconnect=induced_disconnect)
        evidence = self.builder.capture(episode_ids=episode_ids, stage_id='MPC_MEASUREMENT',
            control_step=policy_step, control_substep=control_substep, live=live, terminal=terminal,
            fatal=[None]*w.n if fatal is None else fatal,
            native_success=[None]*w.n if native_success is None else native_success,
            event_kind='reset' if reset else 'physics', **declarations)
        records = []
        com_poses = [self._read(f'hands[{h}].data.body_com_pose_w',
            lambda hand=hand: hand.data.body_com_pose_w, records, data=hand.data,
            property_name='body_com_pose_w', generation_valid=data_generations[f'hands[{h}]']['source_valid'])
            for h, hand in enumerate(w.hands)]
        com_twists = [self._read(f'hands[{h}].data.body_com_vel_w',
            lambda hand=hand: hand.data.body_com_vel_w, records, data=hand.data,
            property_name='body_com_vel_w', generation_valid=data_generations[f'hands[{h}]']['source_valid'])
            for h, hand in enumerate(w.hands)]
        object_com = self._read('wire.data.root_com_pose_w', lambda: w.wire.data.root_com_pose_w, records,
            data=w.wire.data, property_name='root_com_pose_w', generation_valid=data_generations['wire']['source_valid'])
        object_twist = self._read('wire.data.root_com_vel_w', lambda: w.wire.data.root_com_vel_w, records,
            data=w.wire.data, property_name='root_com_vel_w', generation_valid=data_generations['wire']['source_valid'])
        masses = self._read('wire.root_physx_view.get_masses', lambda: w.wire.root_physx_view.get_masses(), records)
        # Existing cache values are explicitly named as such; the builder checks
        # their actual kinematic parents against SDK DATA before assigning validity.
        limits, commands, clearances = [_plain(getattr(w, name, None)) for name in ('limits', 'commands', 'clearances')]
        try:
            backend_targets = _plain(read_common_pause_targets(w))
            target_reason = None
        except (ValueError, RuntimeError, AttributeError, TypeError, IndexError) as exc:
            backend_targets = None
            target_reason = type(exc).__name__ + ': ' + str(exc)
        if _clock(w) != start:
            raise RuntimeError('physics clock changed during complete MPC measurement capture')
        states = []
        for ei, packet in enumerate(evidence['packets']):
            clock = SampleClock(run_uuid=self.run_uuid, env_id=ei, episode_uid=episode_uids[ei],
                world_step=start[0], world_time_s=start[0]*float(w.dt), sdk_step=start[1], sdk_time_s=start[2],
                policy_step=policy_step, reset_generation=reset_generations[ei],
                terminal=terminal[ei] if live[ei] else False, reset=reset)
            if not live[ei]:
                states.append(make_mpc_state(clock, missing_fields(clock,
                    reason='inactive or closed cohort slot; no current state qualification and no history update',
                    source='actual_runner_cohort_live_mask')))
                continue
            fields = missing_fields(clock, reason='no qualified source in current SDK producer',
                source='read_only_current_SDK_producer_availability_audit')
            for key, reason in {
                'contact.rejected_load': 'whole-hand rejected load has no per-pad attribution; broadcasting is forbidden',
                'contact.partner_body_id': 'configured Wire filter is not per-contact partner body identity',
                'contact.contact_age': 'no continuously observed contact-age source; capture may begin mid-contact',
                'support.table_reaction': 'current 34 hand-versus-Wire sensors do not measure table reaction',
                'support.end_support_reaction': 'current hand sensors do not measure passive end-support reaction',
                'support.support_margin': 'no qualified accepted-pad full support or passive reaction measurement',
                'support.left_upward_support': 'accepted-pad friction is not matched; whole-hand raw contribution is retained only in auxiliary evidence',
                'support.right_upward_support': 'accepted-pad friction is not matched; whole-hand raw contribution is retained only in auxiliary evidence',
                'support.total_upward_support': 'no qualified accepted-pad full support; whole-hand raw contribution is separate auxiliary evidence',
                'support.imbalance': 'no qualified per-hand accepted-pad support totals to compare',
                'safety.collision_clearance': 'no current qualified pairwise actual collision-clearance source bound',
                'safety.force_limit': 'no observed per-pad force limit; motor effort is not pad force',
                'object.centerline': 'no actual flexible centerline or source-bound rigid proxy shape source',
                'safety.tracking_error': 'actual PhysX backend target unavailable; cached commands are not a fallback',
            }.items():
                fields[key] = missing_field(key, clock, reason, 'current_SDK_source_capability_audit')

            def put(key, value, *, source, unit, frame='world', kind='measured', origin='not_applicable',
                    dependencies=(), force_side='NA', valid=True, valid_mask=None, missing_reason=None):
                if not valid or value is None:
                    return
                fields[key] = StateField(value=value, source=source, unit=unit, frame=frame,
                    timestamp_s=clock.world_time_s, valid=True, env_id=ei, episode_uid=clock.episode_uid,
                    clock=clock, kind=kind, sample_hz=480., missing_reason=missing_reason,
                    force_side=force_side, dependencies=dependencies or ((source,) if kind == 'derived' else ()),
                    origin=origin, valid_mask=valid_mask, contact_basis=FIELD_SPECS[key].contact_basis)

            kin = packet['actual_kinematics']
            kin_valid = kin['valid'] and not reset and all(row['source_valid'] for row in data_generations.values())
            for hi, hand in enumerate(kin['hands']):
                side = ('left', 'right')[hi]
                active = [JOINTS.index(name) for name in ACTIVE]
                q = np.asarray(hand['full_joint_positions_rad'])[active]
                qd = np.asarray(hand['full_joint_velocities_rad_s'])[active]
                good = hand['valid'] and not reset and data_generations[f'hands[{hi}]']['source_valid']
                for name, value, unit in (('q', q.tolist(), 'rad'), ('qd', qd.tolist(), 'rad/s')):
                    put(f'robot.{side}_{name}', value, source='SDK.data.joint_'+('pos' if name == 'q' else 'vel')+'_named_ACTIVE',
                        unit=unit, frame='joint', valid=good)
                wrist_pose = hand['base_link_pose_w']
                put(f'wrist.{side}_pose', wrist_pose, source='SDK.data.body_link_pose_w[actual_base_link]',
                    unit='m|quaternion_wxyz', valid=good)
                put(f'wrist.{side}_base_pose', wrist_pose,
                    source='SDK.actual_DH116_base_link_same_frame_as_original_palm_reference',
                    unit='m|quaternion_wxyz', valid=good)
                cp, cv = com_poses[hi], com_twists[hi]
                velocities_available = (good and _env_finite(cp, ei, (len(hand['body_names']), 7))
                    and _env_finite(cv, ei, (len(hand['body_names']), 6)))
                if velocities_available:
                    bi = self.builder.base_ids[hi]
                    v = point_velocity(cv[ei][bi], cp[ei][bi][:3], wrist_pose[:3]) + cv[ei][bi][3:]
                    put(f'wrist.{side}_velocity', v, source='SDK.data.body_com_vel_w+omega_cross(actual_base_link-body_com)',
                        unit='m/s|rad/s', kind='derived', origin='palm_origin',
                        dependencies=(f'wrist.{side}_pose',))
                    put(f'wrist.{side}_base_velocity', v, source='SDK.body_com_vel_w+omega_cross(actual_base_link-body_com)',
                        unit='m/s|rad/s', kind='derived', origin='base_origin',
                        dependencies=(f'wrist.{side}_base_pose',))
                for di, digit in enumerate(('thumb', 'index', 'middle')):
                    binding = next((b for b in self.builder.bindings if b[0] == hi and b[2] == di), None)
                    if not good or binding is None:
                        continue
                    bi = binding[1]
                    try:
                        local = _plain(self.builder.atlas_rows[hi, di]['center_local_m'])
                        position = _point(hand['body_link_pose_w'][bi], local)
                    except (KeyError, ValueError, TypeError):
                        continue
                    prefix = f'fingertips.{side}_{digit}'
                    put(prefix+'.position', position, source='SDK.actual_body_link_pose_w*source_CAD_pad_center_geometry_only',
                        unit='m', kind='derived', dependencies=(f'wrist.{side}_pose',))
                    if velocities_available:
                        put(prefix+'.velocity', point_velocity(cv[ei][bi], cp[ei][bi][:3], position),
                            source='SDK.body_com_vel_w+omega_cross(pad_reference_point-body_com)',
                            unit='m/s', kind='derived', origin='pad_reference_point', dependencies=(prefix+'.position',))
            put('object.orientation', kin['object_pose_w'][3:] if kin['object_pose_w'] else None,
                source='SDK.wire.data.root_quat_w_actual_link_wxyz', unit='quaternion_wxyz', valid=kin_valid)
            put('object.bottom_height', packet['bottom_m'], source='TensorWorld.bottom_actual_rigid_pose_original_geometry_reduction',
                unit='m', kind='derived', dependencies=('object.orientation',), valid=kin_valid)
            object_velocity_available = kin_valid and _env_finite(object_com, ei, (7,)) and _env_finite(object_twist, ei, (6,))
            if kin_valid and _env_finite(object_com, ei, (7,)):
                put('object.com', object_com[ei][:3], source='SDK.wire.data.root_com_pose_w', unit='m')
            if kin_valid and masses is not None and np.asarray(masses).size == w.n:
                mass = float(np.asarray(masses).reshape(-1)[ei])
                put('object.mass', mass, source='SDK.wire.root_physx_view.get_masses', unit='kg', frame='not_applicable',
                    valid=math.isfinite(mass) and mass > 0)
            self._contacts(packet, put, fields, clock, com_poses, com_twists, object_com, object_twist,
                object_velocity_available, kin_valid)
            actual_targets = None if backend_targets is None else backend_targets['backend_active_rad']
            if kin_valid and _finite(actual_targets, (w.n, 12)):
                measured = np.asarray([fields['robot.'+s+'_q'].value for s in ('left', 'right')])
                put('safety.tracking_error', (np.asarray(actual_targets[ei]).reshape(2, 6)-measured).tolist(),
                    source='actual_PhysX_backend_get_dof_position_targets_minus_measured_ACTIVE_q', unit='rad', frame='joint', kind='derived',
                    dependencies=('PhysX.backend_active_rad', 'robot.left_q', 'robot.right_q'))
            if kin_valid and limits is not None:
                try:
                    values = []
                    for hi, hand in enumerate(kin['hands']):
                        ids = [list(w.hands[hi].joint_names).index(k) for k in ACTIVE]
                        bounds = np.asarray(limits)[ei, hi, ids]
                        q = np.asarray(fields['robot.'+('left', 'right')[hi]+'_q'].value)
                        values.append(np.minimum(q-bounds[:, 0], bounds[:, 1]-q).tolist())
                    if _finite(values, (2, 6)):
                        put('safety.joint_margin', values, source='TensorWorld.original_configured_limits_vs_actual_ACTIVE_q',
                            unit='rad', frame='joint', kind='derived', dependencies=('robot.left_q', 'robot.right_q'))
                except (IndexError, ValueError, TypeError):
                    pass
            if kin_valid and _finite(clearances, (w.n, 2)):
                put('safety.table_clearance', min(clearances[ei]),
                    source='TensorWorld.clearances_original_collision_vertices_at_actual_body_poses_minus_table_top',
                    unit='m', kind='derived')
            states.append(make_mpc_state(clock, fields))
        observed = []
        for ei, packet in enumerate(evidence['packets']):
            coverage = packet['contact_evidence_coverage']
            components = {}
            for name, key in (('normal', 'raw_contacts'), ('friction', 'raw_friction_contacts')):
                raw = packet[key]
                values = [r['force_on_hand_w_n'] for r in raw] if raw is not None else None
                finite = values is not None and all(_finite(v, (3,)) for v in values)
                components[name+'_n'] = ([sum(-r['force_on_hand_w_n'][2] for r in raw if r['hand_index'] == h)
                    for h in range(2)] if finite else None)
            normal, friction = components['normal_n'], components['friction_n']
            observed.append(dict(env_id=ei, run_uuid=self.run_uuid, episode_uid=episode_uids[ei],
                episode_id=episode_ids[ei], clock=asdict(states[ei].clock),
                source='observed_whole_hand_Wire_filtered_raw_signed_force_vectors_not_accepted_pad_credit',
                unit='N', frame='world', force_side='ON_OBJECT',
                total_n=[normal[h]+friction[h] for h in range(2)] if normal is not None and friction is not None else None,
                normal_complete=coverage['normal_complete'], friction_complete=coverage['friction_complete'],
                ownership_valid=coverage['ownership_valid'], capacity_saturated=coverage['capacity_saturated'],
                normal_vertical_match=coverage['normal_vertical_match'], friction_vertical_match=coverage['friction_vertical_match'],
                includes_rejected_normal=True, accepted_pad_support_credit=False, airborne_proof=False,
                validity_scope='observed_subtotal_only_when_incomplete; raw_force_sum_is_not_passive_reaction_or_support_qualification',
                invalid_reasons=coverage['invalid_reasons'], **components))
        for record in records:
            record.update(run_uuid=self.run_uuid, env_ids=list(range(w.n)), episode_uids=list(episode_uids),
                reset_generations=list(reset_generations), timestamp_s=start[0]*float(w.dt))
        if _clock(w) != start:
            raise RuntimeError('physics clock changed during complete derived state capture')
        if any(getattr(data, '_sim_timestamp', None) != data_generations[key]['data_timestamp']
                for key, data in self._data_sources):
            raise RuntimeError('SDK DATA generation changed during complete state capture')
        self.last_evidence = dict(schema='dh116-bimanual-mpc-measurements-v1', native_attested=False,
            execution='read_only_SDK_DATA_snapshot_not_native_attestation', builder_evidence=_plain(evidence),
            observed_filtered_hand_force_z_on_object=observed,
            extra_sdk_reads=records, runner_setup_declarations=declarations,
            data_generation_evidence=data_generations,
            refresh_receipt_evidence=receipt_evidence,
            joint_target_evidence=dict(readback=backend_targets, missing_reason=target_reason,
                cached_commands=commands, cache_scope='TensorWorld.commands_diagnostic_only_never_backend_tracking_fallback'),
            setup_declaration_provenance='runner_self_reported_not_independently_measured',
            world_time_source='TensorWorld._physics_steps*original_dt_local_elapsed',
            sdk_time_source='World.sim.current_time/current_time_step_index',
            filter_provenance='configured_Wire_filter_not_per_contact_partner_body_ID')
        for ei, state in enumerate(states):
            if live[ei]:
                self._previous[ei] = (episode_uids[ei], reset_generations[ei], start, state.clock.terminal, episode_ids[ei], policy_step)
        if receipt_evidence['source_valid']:
            self._last_refresh_sequence = receipt_evidence['receipt']['sequence']
        return tuple(states)

    def _contacts(self, packet, put, fields, clock, com_poses, com_twists, object_com, object_twist,
                  object_velocity_available, kin_valid):
        coverage = packet['contact_evidence_coverage']
        good = packet['measurement_valid'] and kin_valid
        if not good:
            return
        loads = np.asarray(packet['effective_pad_loads_n'])[:, :3]
        vectors = np.asarray(packet['effective_pad_vectors_w'])[:, :3]
        mask = np.asarray(packet['effective_pad_valid'], dtype=bool)[:, :3] & (loads > 0)
        put('contact.contact_mask', mask.tolist(), source='owned_native_original_pad_gate_positive_normal_scalar',
            unit='dimensionless', frame='dimensionless', kind='derived')
        put('contact.accepted_load', loads.tolist(), source='native_accepted_sum_positive_normal_scalars_not_vector_norm',
            unit='N', force_side='ON_HAND', kind='derived')
        put('contact.accepted_force_on_hand', vectors.tolist(), source='native_accepted_sum_normal_scalar_times_raw_normal_ON_HAND',
            unit='N', force_side='ON_HAND', kind='derived')
        put('contact.accepted_force_on_object', (-vectors).tolist(), source='negative_of_accepted_force_ON_HAND',
            unit='N', force_side='ON_OBJECT', kind='derived', dependencies=('contact.accepted_force_on_hand',))
        put('contact.normal_impulse_derived', (loads*float(self.world.dt)).tolist(),
            source='accepted_load_times_original_physics_dt_not_independent_solver_impulse', unit='N*s',
            force_side='ON_HAND', kind='derived', dependencies=('contact.accepted_load',))
        put('contact.rejected_load_hand', packet['rejected_contact_loads_n'],
            source='whole_hand_sum_abs_native_normal_scalar_minus_all_five_accepted_pad_loads',
            unit='N', force_side='ON_HAND', kind='derived')
        # The legacy builder verifies load/vector reductions, not its centroid
        # cache. Reconstruct this quantity from the owned accepted raw points.
        weighted_points = np.zeros((2, 3, 3)); point_loads = np.zeros((2, 3))
        for record in packet['raw_contacts']:
            hi, di = record['hand_index'], record['digit_index']
            if record['accepted'] and di is not None and di < 3:
                load = record['normal_scalar_n']
                weighted_points[hi, di] += load*np.asarray(record['point_w_m'])
                point_loads[hi, di] += load
        points = weighted_points/np.maximum(point_loads[..., None], 1e-30)
        norms = np.linalg.norm(vectors, axis=-1)
        normal_mask = mask & (norms > 1e-12)
        normal = vectors / np.maximum(norms[..., None], 1e-30)
        for key, values, validity, unit, source in (
            ('contact.contact_position', points, mask, 'm', 'accepted_raw_points_normal_scalar_weighted_actual_contact_centroid'),
            ('contact.contact_normal', normal, normal_mask, 'dimensionless', 'accepted_force_resultant_direction_not_individual_raw_normal')):
            if validity.any():
                value = [[values[h, d].tolist() if validity[h, d] else None for d in range(3)] for h in range(2)]
                put(key, value, source=source, unit=unit, kind='derived', valid_mask=validity.tolist(),
                    missing_reason='no accepted contact or cancelled normal on masked pads')
        body_ids = [[None]*3 for _ in range(2)]
        for hi, bi, di in self.builder.bindings:
            if di is not None and di < 3 and mask[hi, di]:
                body_ids[hi][di] = bi
        if mask.any():
            put('contact.hand_body_id', body_ids, source='actual_sensor_hand_body_binding_of_accepted_pad',
                unit='dimensionless', frame='not_applicable', valid_mask=mask.tolist(),
                missing_reason='no accepted contact on masked pads')
        # Whole-hand friction is genuinely measured but unmatched to accepted
        # pad manifold points. Never distribute it to pads or assert full support.
        if coverage['friction_complete'] and coverage['ownership_valid'] and coverage['friction_vertical_match']:
            friction = np.zeros((2, 3))
            for record in packet['raw_friction_contacts']:
                friction[record['hand_index']] += record['force_on_hand_w_n']
            put('contact.friction_force_hand', friction.tolist(), source='whole_hand_native_Wire_filtered_raw_friction_sum_unmatched_to_accepted_pads',
                unit='N', force_side='ON_HAND', kind='derived')
        # A rigid kinematic slip estimate at each actual accepted contact point.
        # It is not a fixed-anchor slip measurement or a patch friction match.
        if object_velocity_available:
            slip = np.zeros((2, 3)); weight = np.zeros((2, 3)); valid = mask.copy()
            for record in packet['raw_contacts']:
                hi, di, bi = record['hand_index'], record['digit_index'], record['body_id']
                if not record['accepted'] or di is None or di >= 3:
                    continue
                poses, twists = com_poses[hi], com_twists[hi]
                if (not _env_finite(poses, clock.env_id, (len(self.world.hands[hi].body_names), 7))
                        or not _env_finite(twists, clock.env_id, (len(self.world.hands[hi].body_names), 6))):
                    valid[hi, di] = False
                    continue
                point = record['point_w_m']
                normal = np.asarray(record['normal_w']); normal /= np.linalg.norm(normal)
                hand_v = np.asarray(point_velocity(twists[clock.env_id][bi], poses[clock.env_id][bi][:3], point))
                object_v = np.asarray(point_velocity(object_twist[clock.env_id], object_com[clock.env_id][:3], point))
                relative = hand_v-object_v
                speed = np.linalg.norm(relative-normal*np.dot(relative, normal))
                load = record['normal_scalar_n']; slip[hi, di] += load*speed; weight[hi, di] += load
            valid &= weight > 0
            if valid.any():
                values = [[float(slip[h, d]/weight[h, d]) if valid[h, d] else None for d in range(3)] for h in range(2)]
                put('contact.slip_speed', values,
                    source='raw_accepted_load_weighted_tangential_relative_rigid_point_velocity_not_fixed_anchor_slip',
                    unit='m/s', kind='derived', valid_mask=valid.tolist(),
                    missing_reason='no accepted contact or unavailable actual COM twist on masked pads',
                    dependencies=('raw_contacts_each_point_normal', 'SDK.body_com_vel_w', 'SDK.root_com_vel_w', 'object.com'))
