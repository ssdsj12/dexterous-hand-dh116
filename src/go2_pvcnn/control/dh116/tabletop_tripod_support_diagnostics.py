"""Read-only per-environment 480 Hz tripod evidence and conditional gates.

This pure reducer cannot attest native execution. The integrating native runner
owns provenance and supplies same-substep measurements BEFORE physical reset.
It neither reads a simulator nor advances a task/success tracker/Actor. Unknown
measurements stay None; nonfinite original values are tagged alongside JSON-safe
raw evidence. Commands and implicit actuator estimates have explicit labels.

``observe(packet)`` accepts one env/substep. Required identity fields are shown
in IDENTITY. Use event_kind='reset' with actual post-reset poses/bottom before
the first physical sample of the new episode. A reset is an evidence row, not
a simulated timestep. Mark terminal/fatal on the final physics row. Repeated
identities and later physics of closed episodes are not emitted again.

Strict support evidence additionally requires constructor manifests from the
frozen source: all 34 ``sensor_bindings`` triples (hand, actual SDK body, digit
or None) and all ``passive_surface_ids``. Packet completeness assertions cannot
replace those manifests. Missing manifests leave support coverage unknown.
``begin_stage(last_actual_packet)`` explicitly freezes a stage once per physical
episode at an already observed clock, preserving episode contact/lift history.
"""
from copy import deepcopy
import math

SCHEMA = 'dh116-tripod-support-substep-v1'
DT = 1 / 480
TORQUE_SOURCE = 'implicit_actuator_estimate_not_hardware_measurement'
IDENTITY = ('env_id', 'episode_id', 'stage_id', 'tensor_world_physics_steps',
            'simulation_step_index', 'simulation_time_s', 'control_step', 'control_substep')


def _plain(value, path='', nonfinite=None):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().tolist()
    elif hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v, f'{path}.{k}' if path else str(k), nonfinite) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v, f'{path}.{i}', nonfinite) for i, v in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        if nonfinite is not None:
            nonfinite[path] = 'nan' if math.isnan(value) else ('+inf' if value > 0 else '-inf')
        return None
    return deepcopy(value)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _array(value, shape, boolean=False):
    if not shape:
        return type(value) is bool if boolean else _number(value)
    return (isinstance(value, list) and len(value) == shape[0]
            and all(_array(v, shape[1:], boolean) for v in value))


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def _opposed(loads, vectors, valid):
    bearing = [v for f, v, ok in zip(loads[:3], vectors[:3], valid[:3]) if f > .05 and ok]
    if any(_norm(v) <= 1e-12 for v in bearing):
        return None
    if len(bearing) < 2:
        return False
    units = [[x / _norm(v) for x in v] for v in bearing]
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    pairs = [(dot(a, b), a, b) for a in units for b in units]
    score, a, b = min(pairs, key=lambda item: item[0])
    if score > -.25:
        return False
    axis = [x - y for x, y in zip(a, b)]; length = _norm(axis)
    return all(abs(dot(v, axis)) / length > .2 for v in units)


def _rotation(pose):
    if not _array(pose, (7,)) or abs(_norm(pose[3:]) - 1.) > 1e-3:
        return None
    length = _norm(pose[3:])
    w, x, y, z = [v / length for v in pose[3:]]
    return [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]


def _markers(p):
    supplied = p.get('object_relative_markers_m')
    if supplied is not None:
        return deepcopy(supplied) if _array(supplied, (2, 3, 3)) else None
    obj, wrists = p.get('object_pose_w'), p.get('wrist_pose')
    if not _array(wrists, (2, 7)):
        return None
    rotation = _rotation(obj); hands = [_rotation(v) for v in wrists]
    if rotation is None or any(r is None for r in hands):
        return None
    # Exactly the original native rod_points_in_hands marker contract: two
    # axial points plus the radius point needed to observe axial twisting.
    local = ((0., -.2, 0.), (0., .2, 0.), (.008, 0., 0.))
    world = [[obj[i] + sum(rotation[i][j] * point[j] for j in range(3)) for i in range(3)]
             for point in local]
    return [[[sum((point[i] - pose[i]) * r[i][j] for i in range(3)) for j in range(3)]
             for point in world] for pose, r in zip(wrists, hands)]


def _drift(points, anchor):
    if not _array(points, (2, 3, 3)) or not _array(anchor, (2, 3, 3)):
        return None
    return max(_norm([x-y for x, y in zip(a, b)]) for hand, reference in zip(points, anchor)
               for a, b in zip(hand, reference))


def _torque(p):
    original, drive = p.get('effort_limit_nm'), p.get('drive_effort_limit_nm')
    applied, computed, command = (p.get(k) for k in
        ('estimated_torque_nm', 'computed_torque_nm', 'explicit_effort_command_nm'))
    def metric(value, limits, *, estimate=False, mask=None):
        values = value if isinstance(value, list) and len(value) == 12 else None
        budgets = limits if isinstance(limits, list) and len(limits) == 12 else None
        valid = [bool(values is not None and budgets is not None and _number(values[i])
                      and _number(budgets[i]) and budgets[i] > 0
                      and (not estimate or p.get('torque_source') == TORQUE_SOURCE)
                      and (not estimate or mask is None or
                           (isinstance(mask, list) and len(mask) == 12 and mask[i] is True)))
                 for i in range(12)]
        saturated = [abs(values[i]) >= .98 * budgets[i] if valid[i] else None for i in range(12)]
        return valid, saturated
    av, a = metric(applied, original, estimate=True, mask=p.get('torque_valid'))
    cv, c = metric(computed, drive, estimate=True, mask=p.get('computed_torque_valid'))
    ev, e = metric(command, original)
    return dict(source=p.get('torque_source'), applied_estimate_nm=applied,
        computed_estimate_nm=computed, original_total_effort_limit_nm=original,
        physx_drive_limit_nm=drive, explicit_effort_command_nm=command,
        applied_valid=av, computed_valid=cv, explicit_command_valid=ev,
        applied_estimate_saturated=a, computed_drive_saturated=c, explicit_command_saturated=e,
        applied_valid_count=sum(av), hardware_torque_nm=None, hardware_calibrated=False,
        actual_solver_drive_torque_nm=None)


def _raw_coverage(p, components, bindings):
    """Fail closed on absent, stale, truncated or inconsistent force evidence.

    A list being present is not a sensor coverage certificate. The native
    bridge must preserve ownership for every original 34 object-only view;
    nonzero summary forces must also have finite owned raw force rows.
    """
    cover = p.get('contact_evidence_coverage')
    common = (bindings is not None and isinstance(cover, dict)
        and cover.get('schema') == 'dh116-tripod-contact-coverage-v1'
        and all(type(cover.get(k)) is int and cover[k] == p.get(k)
                for k in ('env_id', 'episode_id', 'tensor_world_physics_steps'))
        and cover.get('ownership_valid') is True)
    results = []
    for key, prefix in (('raw_contacts', 'normal'), ('raw_friction_contacts', 'friction')):
        rows = p.get(key)
        buffers = p.get('raw_contact_buffers' if prefix == 'normal' else 'raw_friction_buffers')
        known = (common and cover.get(prefix + '_complete') is True
                 and type(cover.get(prefix + '_sensor_count')) is int
                 and cover[prefix + '_sensor_count'] == 34 and isinstance(rows, list)
                 and isinstance(buffers, list) and len(buffers) == 34)
        totals = [0., 0.]
        if known:
            intervals = {}
            for buffer in buffers:
                if not (isinstance(buffer, dict) and type(buffer.get('sensor_index')) is int
                        and 0 <= buffer['sensor_index'] < 34
                        and buffer['sensor_index'] not in intervals):
                    known = False; break
                si = buffer['sensor_index']; h, b, _ = bindings[si]
                if not (type(buffer.get('hand_index')) is int and buffer['hand_index'] == h
                        and type(buffer.get('body_id')) is int and buffer['body_id'] == b
                        and all(type(buffer.get(k)) is int for k in ('count', 'start', 'capacity'))
                        and 0 <= buffer['count'] <= buffer['capacity']
                        and 0 <= buffer['start'] <= buffer['capacity'] - buffer['count']):
                    known = False; break
                intervals[si] = buffer
            known = bool(known and len(intervals) == 34
                         and sum(v['count'] for v in intervals.values()) == len(rows))
            accepted = [[[0., 0., 0.] for _ in range(5)] for _ in range(2)]
            accepted_load = [[0.] * 5 for _ in range(2)]
            rejected_load = [0., 0.]
            owned = set()
            for row in rows:
                if not known:
                    break
                valid = (isinstance(row, dict) and type(row.get('sensor_index')) is int
                    and row['sensor_index'] in intervals and row.get('valid') is True)
                if not valid:
                    known = False; break
                si = row['sensor_index']; h, b, d = bindings[si]; buffer = intervals[si]
                valid = (type(row.get('hand_index')) is int and row['hand_index'] == h
                    and type(row.get('body_id')) is int and row['body_id'] == b
                    and 'digit_index' in row and type(row['digit_index']) is type(d) and row['digit_index'] == d
                    and all(type(row.get(k)) is int for k in ('raw_slot', 'raw_count', 'raw_start'))
                    and row['raw_count'] == buffer['count'] and row['raw_start'] == buffer['start']
                    and buffer['start'] <= row['raw_slot'] < buffer['start'] + buffer['count']
                    and (si, row['raw_slot']) not in owned
                    and _array(row.get('force_on_hand_w_n'), (3,))
                    and _array(row.get('point_w_m'), (3,)))
                if prefix == 'normal':
                    valid = (valid and _array(row.get('normal_w'), (3,))
                        and _number(row.get('normal_scalar_n'))
                        and _number(row.get('depth_m')) and row['depth_m'] >= 0
                        and type(row.get('accepted')) is bool
                        and (not row['accepted'] or (d is not None and row['normal_scalar_n'] > 0)))
                    valid = (valid and abs(_norm(row['normal_w']) - 1.) <= 1e-4
                        and all(abs(row['force_on_hand_w_n'][i] -
                            row['normal_scalar_n'] * row['normal_w'][i]) <=
                            1e-5 * max(1., abs(row['normal_scalar_n'])) for i in range(3)))
                if not valid:
                    known = False
                    break
                owned.add((si, row['raw_slot']))
                totals[h] -= row['force_on_hand_w_n'][2]
                if prefix == 'normal' and row['accepted']:
                    accepted_load[h][d] += row['normal_scalar_n']
                    for i in range(3):
                        accepted[h][d][i] += row['force_on_hand_w_n'][i]
                elif prefix == 'normal':
                    # The original compliant native sampler retains finite
                    # signed releases and accumulates absolute rejected load.
                    rejected_load[h] += abs(row['normal_scalar_n'])
            column = 0 if prefix == 'normal' else 1
            known = bool(known and components is not None and all(
                abs(totals[h] - components[h][column]) <= 1e-5 * max(1., abs(components[h][column]))
                for h in range(2)))
            if known and prefix == 'normal':
                vectors, loads, rejected = (p.get(k) for k in ('effective_pad_vectors_w',
                    'effective_pad_loads_n', 'rejected_contact_loads_n'))
                close = lambda a, b: abs(a-b) <= 1e-5 * max(1., abs(a), abs(b))
                known = (_array(vectors, (2, 5, 3)) and _array(loads, (2, 5))
                    and _array(rejected, (2,))
                    and all(close(accepted[h][d][i], vectors[h][d][i])
                            for h in range(2) for d in range(5) for i in range(3))
                    and all(close(accepted_load[h][d], loads[h][d]) for h in range(2) for d in range(5))
                    and all(close(rejected_load[h], rejected[h]) for h in range(2)))
        results.append(bool(known))
    return results


def _passive_coverage(p, surface_ids):
    """Original table AND end supports require actual separate measurements."""
    cover = p.get('passive_support_coverage')
    if not (surface_ids is not None and isinstance(cover, dict) and cover.get('schema') == 'dh116-passive-support-coverage-v1'
            and cover.get('complete') is True and cover.get('ownership_valid') is True
            and all(type(cover.get(k)) is int and cover[k] == p.get(k)
                    for k in ('env_id', 'episode_id', 'tensor_world_physics_steps'))):
        return None
    rows = cover.get('surfaces')
    expected = cover.get('expected_surface_ids')
    if not (isinstance(expected, list) and expected and all(type(k) is str and k for k in expected)
            and len(set(expected)) == len(expected) and isinstance(rows, list)
            and len(rows) == len(expected) and sorted(expected) == sorted(surface_ids)):
        return None
    totals = []
    identities = []
    for row in rows:
        if not (isinstance(row, dict) and row.get('measurement_valid') is True
                and type(row.get('surface_id')) is str and _number(row.get('support_n'))
                and row['support_n'] >= 0):
            return None
        identities.append(row['surface_id']); totals.append(row['support_n'])
    if sorted(identities) != sorted(expected):
        return None
    return sum(totals)


class TripodSupportDiagnostics:
    """State is independent per env/episode; sink receives detached JSON values."""
    def __init__(self, run_id, sink=None, *, sensor_bindings=None, passive_surface_ids=None):
        if not isinstance(run_id, str) or not run_id or (sink is not None and not callable(sink)):
            raise ValueError('run_id and optional callable sink required')
        self.run_id, self.sink = run_id, sink
        if sensor_bindings is not None:
            sensor_bindings = _plain(sensor_bindings)
            if (not isinstance(sensor_bindings, list) or len(sensor_bindings) != 34
                    or any(not isinstance(v, list) or len(v) != 3
                        or type(v[0]) is not int or v[0] not in (0, 1)
                        or type(v[1]) is not int or v[1] < 0
                        or (v[2] is not None and (type(v[2]) is not int or v[2] not in range(5)))
                        for v in sensor_bindings)
                    or len({(h, b) for h, b, _ in sensor_bindings}) != 34
                    or any(sum(h == hi for h, _, _ in sensor_bindings) != 17 for hi in range(2))
                    or any(sorted(d for h, _, d in sensor_bindings if h == hi and d is not None) != list(range(5))
                           for hi in range(2))):
                raise ValueError('34 unique source sensor bindings with five distal pads per hand required')
            sensor_bindings = tuple(tuple(v) for v in sensor_bindings)
        if passive_surface_ids is not None:
            passive_surface_ids = _plain(passive_surface_ids)
            if (not isinstance(passive_surface_ids, list) or not passive_surface_ids
                    or any(type(v) is not str or not v for v in passive_surface_ids)
                    or len(set(passive_surface_ids)) != len(passive_surface_ids)):
                raise ValueError('complete unique source passive surface IDs required')
            passive_surface_ids = tuple(passive_surface_ids)
        self.sensor_bindings, self.passive_surface_ids = sensor_bindings, passive_surface_ids
        self._states, self._seen = {}, set()
        self._summary = dict(live_substeps=0, reset_records=0, terminal_records=0, fatal_records=0,
            valid_contact_substeps=0, support_attested_substeps=0,
            torque_joint_samples={name: dict(total=0, valid=0, saturated=0) for name in
                ('applied_estimate', 'computed_drive', 'explicit_command')})

    @staticmethod
    def _state(episode):
        return dict(episode=episode, closed=False, last_clock=None, baseline_wrist=None,
            baseline_bottom=None, anchor=None, explicit_anchor=False, stage_wrist=None,
            stage=None, six_s=0., hold_s=0.,
            right_seen=[False] * 3, right_lost=[False] * 3,
            right_stable_steps=[0] * 3, recovered=[False] * 3,
            natural_history=True, hold_steps=0, contact_hold_steps=0,
            right_current_contact=[None]*3, stage_right_lost=[False]*3,
            stage_right_recovered=[False]*3,
            started_stages=set(), last_actual=None, fixed_interval_valid=True)

    def _identity(self, p, state):
        for key in ('env_id', 'episode_id', 'tensor_world_physics_steps', 'simulation_step_index',
                    'control_step', 'control_substep'):
            if type(p.get(key)) is not int or p[key] < 0:
                raise ValueError('nonnegative integer identity: ' + key)
        if p.get('stage_id') not in ('A', 'B', 'C', 'D', 'E'):
            raise ValueError('stage A..E identity required')
        if not _number(p.get('dt_s')) or abs(p['dt_s'] - DT) > 1e-12:
            raise ValueError('480 Hz physical dt required')
        if not _number(p.get('simulation_time_s')) or p['simulation_time_s'] < 0:
            raise ValueError('clock requires finite simulation time')
        reset = p.get('event_kind', 'physics') == 'reset'
        if reset:
            return
        if not 1 <= p['control_substep'] <= 8:
            raise ValueError('clock requires actual 1..8 physical substep')
        old = state['last_clock']
        if old is not None:
            wrap = p['control_substep'] == 1
            if (p['simulation_step_index'] != old['simulation_step_index'] + 1
                    or p['tensor_world_physics_steps'] != old['tensor_world_physics_steps'] + 1
                    or p['control_substep'] != old['control_substep'] % 8 + 1
                    or p['control_step'] != old['control_step'] + int(wrap and old['control_substep'] != 0)
                    or abs(p['simulation_time_s'] - old['simulation_time_s'] - DT) > 1e-9):
                # Do not retain continuity after a recorder gap, even if a caller
                # catches this error and tries to continue the experiment.
                state['six_s'] = state['hold_s'] = 0.
                state['hold_steps'] = state['contact_hold_steps'] = 0
                state['right_stable_steps'] = [0] * 3
                raise ValueError('physical/control clock continuity lost')

    def begin_stage(self, packet):
        """Freeze one stage reference from the last observed actual substep.

        This event advances no physics and performs no physical episode reset.
        The caller may change only the stage identity, never supply a new pose
        for an already observed clock. Sensor history and reset lift survive.
        """
        nonfinite = {}; p = _plain(packet, nonfinite=nonfinite)
        if not isinstance(p, dict) or type(p.get('env_id')) is not int:
            raise ValueError('stage reference packet with actual identity required')
        state = self._states.get(p['env_id'])
        if (state is None or state['closed'] or type(p.get('episode_id')) is not int
                or p['episode_id'] != state['episode'] or state['last_actual'] is None):
            raise ValueError('stage reference requires live same-episode actual observation')
        stage = p.get('stage_id')
        if stage not in ('A', 'B', 'C', 'D', 'E'):
            raise ValueError('stage A..E identity required')
        if stage in state['started_stages']:
            raise ValueError('each stage may begin only once per physical episode')
        actual = state['last_actual']
        if (p.get('measurement_valid') is not True or not _number(p.get('dt_s'))
                or abs(p['dt_s'] - DT) > 1e-12
                or any(type(p.get(k)) is not type(actual[k]) or p[k] != actual[k]
                       for k in IDENTITY if k != 'stage_id')
                or any(p.get(k) != actual[k] for k in ('wrist_pose', 'object_pose_w', 'bottom_m'))
                or _markers(p) != actual['object_relative_markers_m']):
            raise ValueError('stage reference must match the last observed actual clock and poses')
        state.update(stage=stage, anchor=deepcopy(actual['object_relative_markers_m']),
            stage_wrist=deepcopy(actual['wrist_pose']), explicit_anchor=False,
            hold_steps=0, contact_hold_steps=0, hold_s=0., contact_hold_s=0.,
            right_stable_steps=[0] * 3, fixed_interval_valid=True)
        # Keep physical-episode contact/loss history and all budgets. A fixed-
        # wrist recovery experiment still needs a recovery INSIDE this stage:
        # either a genuinely missing pad at entry, or a later real loss.
        state['stage_right_lost']=[state['right_seen'][d] and state['right_current_contact'][d] is False
                                   for d in range(3)]
        state['stage_right_recovered']=[False]*3
        state['started_stages'].add(stage)
        out = dict(schema_version=SCHEMA, run_id=self.run_id,
            execution='supplied_measurement_reducer_not_native_attestation',
            **{k: p[k] for k in IDENTITY}, dt_s=DT, event_kind='stage_start',
            physical_episode_reset=False, raw=p, nonfinite_originals=nonfinite,
            fixed_anchor_markers_m=deepcopy(state['anchor']),
            fixed_wrist_pose=deepcopy(state['stage_wrist']),
            initial_right_contact=deepcopy(state['right_current_contact']),
            stage_eligible_prior_loss=deepcopy(state['stage_right_lost']))
        self._emit(out)
        return out

    def observe(self, packet):
        """Reduce a raw row once, including a final terminal/fatal physical row."""
        nonfinite = {}; p = _plain(packet, nonfinite=nonfinite)
        if not isinstance(p, dict):
            raise ValueError('packet dictionary required')
        kind = p.get('event_kind', 'physics')
        if kind not in ('physics', 'reset'):
            raise ValueError('event_kind must be physics or reset')
        env, episode = p.get('env_id'), p.get('episode_id')
        if type(env) is not int or type(episode) is not int:
            raise ValueError('env/episode identity required')
        state = self._states.get(env)
        if state is not None and episode < state['episode']:
            raise ValueError('episode identity moved backwards')
        identity = (env, episode, p.get('simulation_step_index'), kind)
        if identity in self._seen:
            return None
        if state is not None and kind == 'reset' and episode == state['episode']:
            raise ValueError('physical reset must advance episode identity')
        if state is not None and episode > state['episode'] and kind != 'reset':
            raise ValueError('new physical episode requires reset evidence')
        if kind == 'physics' and (p.get('live') is not True or
                (state is not None and state['episode'] == episode and state['closed'])):
            return None
        if state is None or state['episode'] != episode:
            state = self._state(episode)
            self._states[env] = state
        self._identity(p, state)
        if kind == 'physics' and state['stage'] is not None and state['stage'] != p['stage_id']:
            raise ValueError('stage change requires explicit begin_stage at the last observed actual clock')
        self._seen.add(identity)
        out = dict(schema_version=SCHEMA, run_id=self.run_id,
            execution='supplied_measurement_reducer_not_native_attestation',
            **{k: p[k] for k in IDENTITY}, dt_s=DT, event_kind=kind,
            terminal=p.get('terminal') is True, fatal=p.get('fatal') is True,
            raw=p, nonfinite_originals=nonfinite)
        wrist = p.get('wrist_pose'); wrist_ok = (_array(wrist, (2, 7)) and
            all(_rotation(pose) is not None for pose in wrist))
        markers = _markers(p)
        if kind == 'reset':
            state.update(baseline_wrist=deepcopy(wrist) if wrist_ok else None,
                baseline_bottom=p.get('bottom_m') if _number(p.get('bottom_m')) else None,
                anchor=markers, explicit_anchor=False, stage_wrist=deepcopy(wrist) if wrist_ok else None,
                stage=p['stage_id'], six_s=0., hold_s=0., contact_hold_s=0., closed=False,
                right_seen=[False] * 3, right_lost=[False] * 3,
                right_stable_steps=[0] * 3, recovered=[False] * 3,
                natural_history=True, hold_steps=0, contact_hold_steps=0,
                right_current_contact=[None]*3,stage_right_lost=[False]*3,
                stage_right_recovered=[False]*3,
                started_stages=set(), last_actual=None, fixed_interval_valid=True,
                last_clock={k: p[k] for k in IDENTITY if k not in ('env_id', 'episode_id', 'stage_id')})
            self._summary['reset_records'] += 1
            out.update(reference_valid=dict(palm=wrist_ok, bottom=state['baseline_bottom'] is not None,
                                            fixed_anchor=markers is not None))
            self._emit(out)
            return out
        state['stage'] = p['stage_id']
        measured = p.get('measurement_valid') is True
        loads, valid, vectors = (p.get(k) for k in ('effective_pad_loads_n',
            'effective_pad_valid', 'effective_pad_vectors_w'))
        contact_known = (measured and _array(loads, (2, 5)) and _array(valid, (2, 5), True)
                         and all(f >= 0 for hand in loads for f in hand)
                         and all(valid[h][d] or loads[h][d] == 0
                                 for h in range(2) for d in range(5)))
        six = all(valid[h][d] and loads[h][d] > .05 for h in range(2) for d in range(3)) if contact_known else None
        right = all(valid[1][d] and loads[1][d] > .05 for d in range(3)) if contact_known else None
        extra_bearing = ([any(valid[h][d] and loads[h][d] > .05 for d in (3, 4))
                          for h in range(2)] if contact_known else None)
        vector_known = measured and _array(vectors, (2, 5, 3))
        opposed = [_opposed(loads[h], vectors[h], valid[h]) for h in range(2)] if contact_known and vector_known else [None, None]
        paired = all(v is True for v in opposed)
        state['six_s'] = state['six_s'] + DT if six is True else 0.
        depth, rejected, components = (p.get(k) for k in ('effective_pad_depth_m',
            'rejected_contact_loads_n', 'vertical_components_n'))
        depth_known = measured and _array(depth, (2, 5)) and all(d >= 0 for hand in depth for d in hand)
        rejected_known = measured and _array(rejected, (2,)) and min(rejected) >= 0
        components_known = measured and _array(components, (2, 2))
        normal = ([-sum(vectors[h][d][2] for d in range(3) if valid[h][d]) for h in range(2)]
                  if vector_known and contact_known else None)
        if normal is not None and components_known:
            normal = [min(normal[h], components[h][0]) for h in range(2)]
        friction = [hand[1] for hand in components] if components_known else None
        support = ([normal[h] + (friction[h] if rejected[h] <= 1e-6 and not extra_bearing[h]
                                else min(friction[h], 0.))
                    for h in range(2)] if normal is not None and friction is not None and rejected_known else None)
        if support is not None:
            support = [min(support[h], sum(components[h])) for h in range(2)]
        weight = p.get('object_weight_n'); weight_ok = _number(weight) and weight > 0
        table = p.get('object_table_support_n')
        table_ok = p.get('object_table_measurement_valid') is True and _number(table) and table >= 0
        total_load = sum(sum(hand) for hand in loads) if contact_known else None
        rejected_fraction = (sum(rejected) / max(total_load + sum(rejected), 1e-12)
                             if rejected_known and contact_known else None)
        geometry_good = (depth_known and max(d for hand in depth for d in hand) <= .0008
                         and rejected_fraction is not None and rejected_fraction <= .05)
        no_external = p.get('object_external_force_applied') is False
        raw_contact_known, raw_friction_known = _raw_coverage(p,
            components if components_known else None, self.sensor_bindings)
        passive_support = _passive_coverage(p, self.passive_surface_ids)
        attested = bool(support is not None and weight_ok and table_ok and no_external
                        and raw_contact_known and raw_friction_known
                        and passive_support is not None and passive_support <= .01 * weight
                        and table <= .01 * weight
                        and sum(support) >= .9 * weight
                        and six is True and paired and geometry_good and not any(extra_bearing))
        baseline = state['baseline_wrist']
        displacement = ([[wrist[h][i] - baseline[h][i] for i in range(3)] for h in range(2)]
                        if measured and wrist_ok and baseline is not None else None)
        net_lift = (p['bottom_m'] - state['baseline_bottom'] if measured and
                    _number(p.get('bottom_m')) and state['baseline_bottom'] is not None else None)
        supplied_anchor_changed = False
        supplied_anchor_valid = True
        if 'fixed_anchor_markers_m' in p:
            supplied = p['fixed_anchor_markers_m']
            supplied_anchor_valid = _array(supplied, (2, 3, 3))
            if supplied_anchor_valid:
                # Actual reset/stage-start markers are immutable. A first late
                # supplied anchor cannot erase accumulated drift or hold time.
                supplied_anchor_changed = (_drift(supplied, state['anchor']) is None or
                                          _drift(supplied, state['anchor']) > 1e-12)
        anchor = state['anchor']
        slip = _drift(markers, anchor) if measured and supplied_anchor_valid else None
        slip_valid = slip is not None
        stage_wrist = state['stage_wrist']
        stage_displacement = ([[wrist[h][i] - stage_wrist[h][i] for i in range(3)] for h in range(2)]
                              if measured and wrist_ok and stage_wrist is not None else None)
        palm_fixed = (stage_displacement is not None and max(_norm(v) for v in stage_displacement) <= .0005
                      and all(abs(sum(x*y for x, y in zip(wrist[h][3:], stage_wrist[h][3:]))) >= math.cos(.001/2)
                          for h in range(2)))
        fixed_declared = p.get('wrists_fixed') is True
        if p['stage_id'] in ('A', 'B', 'C'):
            # A fixed experiment is one declared interval. Returning to its
            # reference cannot erase an earlier motion/coverage violation.
            # Explicit begin_stage starts the interval after preparation.
            state['fixed_interval_valid'] &= fixed_declared and palm_fixed
        safe_contact = (six is True and paired and geometry_good and not any(extra_bearing)
                        and slip_valid and slip <= .0005 and not supplied_anchor_changed)
        contact_hold_good = safe_contact and fixed_declared and palm_fixed and state['fixed_interval_valid']
        if p['stage_id'] == 'D':
            hold_good = safe_contact and attested and net_lift is not None and net_lift >= .03
        else:
            hold_good = contact_hold_good and attested
        if p.get('fatal') is True:
            hold_good = False
        state['hold_steps'] = state['hold_steps'] + 1 if hold_good else 0
        state['hold_s'] = state['hold_steps'] * DT
        state.setdefault('contact_hold_s', 0.)
        state['contact_hold_steps'] = state['contact_hold_steps'] + 1 if contact_hold_good else 0
        state['contact_hold_s'] = state['contact_hold_steps'] * DT
        five = state['hold_steps'] >= 2400
        state['natural_history'] &= p.get('induced_disconnect') is False and no_external
        natural = state['natural_history']
        for digit in range(3):
            known = contact_known and vector_known and depth_known and rejected_known
            contact = bool(known and valid[1][digit] and loads[1][digit] > .05)
            bearing = bool(contact
                           and _norm(vectors[1][digit]) > 1e-12
                           and depth[1][digit] <= .0006 and rejected[1] <= .05)
            if known and not contact and state['right_seen'][digit]:
                state['right_lost'][digit] = True
                state['stage_right_lost'][digit] = True
            state['right_current_contact'][digit]=contact if known else None
            state['right_stable_steps'][digit] = state['right_stable_steps'][digit] + 1 if bearing else 0
            if contact:
                state['right_seen'][digit] = True
            if bearing:
                state['recovered'][digit] |= (state['right_lost'][digit]
                    and state['right_stable_steps'][digit] >= 10 and natural)
                state['stage_right_recovered'][digit] |= (state['stage_right_lost'][digit]
                    and state['right_stable_steps'][digit] >= 10 and natural)
        gate = dict(stage=p['stage_id'], pass_=False, right_contacts_ready=right,
            right_recovery_observed=bool(all(state['stage_right_recovered']) and natural),
            right_pad_recovered=deepcopy(state['stage_right_recovered']),
            right_episode_pad_recovered=deepcopy(state['recovered']),
            right_stable_steps=deepcopy(state['right_stable_steps']),
            contact_hold_pass=state['contact_hold_steps'] >= 2400,
            original_native_success=p.get('native_success') is True,
            support_coverage_complete=attested)
        if p['stage_id'] == 'A':
            passed = bool(right is True and all(state['stage_right_recovered'])
                          and min(state['right_stable_steps']) >= 10 and fixed_declared and palm_fixed
                          and state['fixed_interval_valid']
                          and p.get('left_targets_fixed') is True and natural)
        elif p['stage_id'] == 'B':
            passed = safe_contact and fixed_declared and palm_fixed and state['fixed_interval_valid']
        elif p['stage_id'] == 'C':
            passed = five
        elif p['stage_id'] == 'D':
            passed = five and p.get('native_success') is True
        else:
            passed = False  # This recorder never advances/evaluates Student.
        gate.pop('pass_'); gate['pass'] = bool(passed and not p.get('fatal', False))
        torque = _torque(p)
        out.update(measurement_valid=measured, six_pad_contact=six,
            non_required_bearing=extra_bearing,
            six_pad_contiguous_s=state['six_s'], per_hand_opposed=opposed,
            bilateral_opposed=paired if all(v is not None for v in opposed) else None,
            accepted_pad_vectors_on_hand_w_n=vectors,
            normal_support_n=normal, friction_support_n=friction, verified_support_n=support,
            support_bearing_attested=attested,
            support_validity=dict(accepted_normal=normal is not None,
                friction=friction is not None, rejected=rejected_known,
                object_weight=weight_ok, object_table=table_ok, no_external_force_known=no_external,
                raw_contacts=raw_contact_known, raw_friction=raw_friction_known,
                all_original_passive_surfaces=passive_support is not None),
            passive_support_n=passive_support,
            support_diagnostic_policy=dict(passive_unloading_max_weight_ratio=.01,
                passive_unloading_source='new_diagnostic_policy_not_native_success',
                minimum_total_weight_ratio=.9,
                minimum_total_source='existing_teacher_qualification_not_native_success'),
            rejected_contact_loads_n=rejected, rejected_fraction=rejected_fraction,
            pad_depth_m=depth, palm_displacement_m=displacement,
            palm_displacement_valid=displacement is not None, actual_palms_fixed=palm_fixed,
            fixed_interval_valid=state['fixed_interval_valid'] if p['stage_id'] in ('A', 'B', 'C') else None,
            net_lift_m=net_lift, lift_valid=net_lift is not None,
            object_relative_markers_m=markers, fixed_anchor_markers_m=anchor,
            supplied_anchor_changed=supplied_anchor_changed, supplied_anchor_valid=supplied_anchor_valid,
            slip_m=slip, slip_valid=slip_valid,
            slip_reason='valid' if slip_valid else ('measurement_invalid' if not measured else 'no_valid_fixed_anchor_or_pose'),
            qualified_stage_hold_s=state['hold_s'], contact_stage_hold_s=state['contact_hold_s'],
            stage_gate=gate, torque=torque)
        state['last_clock'] = {k: p[k] for k in IDENTITY if k not in ('env_id', 'episode_id', 'stage_id')}
        state['last_actual'] = (dict(**{k: p[k] for k in IDENTITY},
            wrist_pose=deepcopy(wrist), object_pose_w=deepcopy(p['object_pose_w']),
            bottom_m=p['bottom_m'], object_relative_markers_m=deepcopy(markers))
            if measured and wrist_ok and _rotation(p.get('object_pose_w')) is not None
                and markers is not None and _number(p.get('bottom_m')) else None)
        state['closed'] = out['terminal'] or out['fatal']
        summary = self._summary
        summary['live_substeps'] += 1
        summary['valid_contact_substeps'] += int(contact_known)
        summary['support_attested_substeps'] += int(attested)
        summary['terminal_records'] += int(out['terminal']); summary['fatal_records'] += int(out['fatal'])
        for name, validity, saturation in (
                ('applied_estimate', 'applied_valid', 'applied_estimate_saturated'),
                ('computed_drive', 'computed_valid', 'computed_drive_saturated'),
                ('explicit_command', 'explicit_command_valid', 'explicit_command_saturated')):
            stat = summary['torque_joint_samples'][name]
            stat['total'] += 12; stat['valid'] += sum(torque[validity])
            stat['saturated'] += sum(v is True for v in torque[saturation])
        self._emit(out)
        return out

    def _emit(self, row):
        if self.sink is not None:
            self.sink(deepcopy(row))

    def summary(self):
        result = dict(schema_version=SCHEMA, run_id=self.run_id, **deepcopy(self._summary))
        for metric in result['torque_joint_samples'].values():
            metric['coverage'] = metric['valid'] / metric['total'] if metric['total'] else None
            metric['saturated_fraction'] = metric['saturated'] / metric['valid'] if metric['valid'] else None
        return result
