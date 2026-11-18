"""Synthetic CPU contract tests; these are not native or hardware evidence."""
import dataclasses
import importlib.util
import math
from pathlib import Path
import sys
import unittest


MODULE = Path(__file__).resolve().parents[1] / 'src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_state.py'


def api():
    assert MODULE.is_file(), 'pure bimanual MPC state contract is missing'
    name = '_test_pure_bimanual_mpc_state'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, MODULE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def clock(**changes):
    values = dict(run_uuid='synthetic-run', env_id=2, episode_uid='synthetic-episode',
                  world_step=480, world_time_s=1.0, sdk_step=1480, sdk_time_s=11.0,
                  policy_step=60, reset_generation=3)
    values.update(changes)
    return api().SampleClock(**values)


def field(value, *, unit='rad', frame='joint', kind='measured', **changes):
    c = changes.pop('clock', clock())
    values = dict(value=value, source='synthetic SDK fixture', unit=unit, frame=frame,
                  timestamp_s=c.world_time_s, valid=True, env_id=c.env_id,
                  episode_uid=c.episode_uid, clock=c, kind=kind, sample_hz=480.0)
    if kind == 'derived':
        values['dependencies'] = ('synthetic.raw_input',)
    values.update(changes)
    return api().StateField(**values)


def state(c=None, fields=None):
    m = api()
    c = c or clock()
    explicit = m.missing_fields(c, reason='synthetic fixture intentionally not measured', source='synthetic availability declaration')
    explicit.update(fields or {})
    return m.make_mpc_state(c, explicit)


class StateContractTests(unittest.TestCase):
    def test_contract_imports_without_task_isaac_actor_or_torch(self):
        m = api()
        self.assertNotIn('torch', m.__dict__)
        self.assertNotIn('isaaclab', m.__dict__)

    def test_all_requested_fields_have_explicit_missing_metadata(self):
        m = api()
        snapshot = state()
        required = {'robot.left_q', 'robot.right_q', 'robot.left_qd', 'robot.right_qd',
                    'wrist.left_pose', 'wrist.right_pose', 'wrist.left_velocity', 'wrist.right_velocity',
                    'object.centerline', 'object.centerline_velocity', 'object.local_tangent',
                    'object.local_curvature', 'object.com', 'object.mass', 'object.bottom_height',
                    'object.orientation', 'contact.contact_mask', 'contact.contact_position',
                    'contact.contact_normal', 'contact.accepted_load', 'contact.rejected_load',
                    'contact.slip_speed', 'contact.contact_age', 'support.left_upward_support',
                    'support.right_upward_support', 'support.total_upward_support',
                    'support.support_margin', 'support.imbalance', 'safety.joint_margin',
                    'safety.tracking_error', 'safety.collision_clearance', 'safety.force_limit',
                    'safety.table_clearance'}
        for hand in ('left', 'right'):
            for digit in ('thumb', 'index', 'middle'):
                required.update(f'fingertips.{hand}_{digit}.{component}'
                                for component in ('position', 'velocity', 'normal'))
        self.assertTrue(required <= set(snapshot.fields))
        for f in snapshot.fields.values():
            self.assertIsNone(f.value)
            self.assertFalse(f.valid)
            self.assertTrue(f.missing_reason)
            self.assertEqual((f.env_id, f.episode_uid), (2, 'synthetic-episode'))
            self.assertEqual(f.clock, snapshot.clock)

    def test_snapshot_detaches_input_lists_and_mapping(self):
        m = api()
        q = [0.1] * 6
        fields = {'robot.left_q': field(q)}
        snapshot = state(fields=fields)
        q[0] = 99
        fields.clear()
        self.assertEqual(snapshot.fields['robot.left_q'].value, (0.1,) * 6)
        with self.assertRaises(TypeError):
            snapshot.fields['robot.left_q'] = field([0] * 6)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snapshot.fields['robot.left_q'].valid = False
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snapshot.clock = clock(world_step=481)

    def test_mutable_metadata_and_wrong_validity_mask_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            field([0] * 6, valid_mask=[True] * 6, missing_reason=['mutable reason'])
        m = api()
        for mask in (((False,),), (False,), ((False, False, False), (False, False))):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                f = m.missing_field('contact.contact_position', clock(), 'not available', 'synthetic fixture')
                state(fields={'contact.contact_position': dataclasses.replace(f, valid_mask=mask)})

    def test_base_and_palm_poses_have_separate_point_origins(self):
        pose = field([1, 2, 3, 1, 0, 0, 0], unit='m|quaternion_wxyz', frame='world')
        twist = field([0] * 6, unit='m/s|rad/s', frame='world', origin='base_origin')
        snapshot = state(fields={'wrist.left_base_pose': pose, 'wrist.left_base_velocity': twist})
        self.assertEqual(snapshot.fields['wrist.left_base_pose'].value[:3], (1, 2, 3))
        self.assertIsNone(snapshot.fields['wrist.left_pose'].value)
        with self.assertRaises(ValueError):
            state(fields={'wrist.left_velocity': twist})

    def test_handedness_and_per_digit_shapes_are_preserved(self):
        m = api()
        values = [[1, 2, 3], [4, 5, 6]]
        loads = field(values, unit='N', frame='world', kind='derived', force_side='ON_HAND', contact_basis='actual_contact')
        snapshot = state(fields={'contact.accepted_load': loads})
        self.assertEqual(snapshot.fields['contact.accepted_load'].value[1][2], 6)
        self.assertEqual(m.HANDS, ('left', 'right'))
        self.assertEqual(m.DIGITS, ('thumb', 'index', 'middle'))

    def test_missing_value_and_validity_must_agree(self):
        for kwargs in [dict(value=None, valid=True), dict(value=[0] * 6, valid=False),
                       dict(value=None, valid=False, missing_reason=None)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                field(**kwargs)

    def test_explicit_missing_field_has_no_synthetic_zero(self):
        m = api()
        f = m.missing_field('contact.rejected_load', clock(), 'not measured per digit', 'availability audit')
        self.assertIsNone(f.value)
        self.assertEqual(f.kind, 'unavailable')
        self.assertEqual(f.missing_reason, 'not measured per digit')

    def test_rejects_nonfinite_wrong_shape_and_wrong_unit(self):
        m = api()
        for name, f in [('robot.left_q', field([0] * 5)),
                        ('robot.left_q', field([0] * 5 + [math.nan])),
                        ('robot.left_q', field([0] * 6, unit='deg')),
                        ('contact.rejected_load', field([1, 2], unit='N', frame='world', force_side='ON_HAND'))]:
            with self.subTest(name=name, f=f), self.assertRaises(ValueError):
                state(fields={name: f})

    def test_scalar_bool_integer_and_vector_types_are_not_interchangeable(self):
        m = api()
        cases = [('robot.left_q', field([True] * 6)),
                 ('contact.contact_mask', field([[1] * 3] * 2, unit='dimensionless', frame='dimensionless')),
                 ('object.mass', field([1.0], unit='kg', frame='not_applicable')),
                 ('contact.hand_body_id', field([[1.5] * 3] * 2, unit='dimensionless', frame='not_applicable'))]
        for name, f in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                state(fields={name: f})

    def test_rejects_cross_env_episode_run_reset_and_clock_fields(self):
        m = api()
        changes = [dict(env_id=3), dict(episode_uid='other'), dict(run_uuid='other'),
                   dict(reset_generation=4), dict(world_step=479), dict(sdk_step=1479),
                   dict(policy_step=59), dict(sdk_time_s=10.0)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                state(fields={'robot.left_q': field([0] * 6, clock=clock(**change))})
        for change in [dict(env_id=3), dict(episode_uid='other')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                field([0] * 6, **change)

    def test_stale_snapshot_is_archivable_but_not_eligible_for_planning(self):
        m = api()
        stale = state(fields={'robot.left_q': field([0] * 6, timestamp_s=0.98)})
        with self.assertRaises(ValueError):
            m.validate_planning_state(stale, clock(), max_age_s=0.005, required_fields=('robot.left_q',))
        for timestamp in (1.01, None):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                state(fields={'robot.left_q': field([0] * 6, timestamp_s=timestamp)})
        for c in (clock(reset=True), clock(terminal=True)):
            snapshot = state(c)
            self.assertEqual(snapshot.to_dict()['clock']['terminal'], c.terminal)
            with self.assertRaises(ValueError):
                m.validate_planning_state(snapshot, c, max_age_s=0.005, required_fields=())

    def test_partial_pad_missingness_keeps_values_none_without_erasing_other_pads(self):
        m = api()
        mask = ((True, False, False), (False, True, False))
        values = (((1, 2, 3), None, None), (None, (4, 5, 6), None))
        f = field(values, unit='m', frame='world', kind='derived', valid_mask=mask,
                  missing_reason='no accepted point on masked pads', contact_basis='actual_contact')
        snapshot = state(fields={'contact.contact_position': f})
        self.assertEqual(snapshot.fields['contact.contact_position'].value[0][0], (1, 2, 3))
        self.assertIsNone(snapshot.fields['contact.contact_position'].value[0][1])
        bad = dataclasses.replace(f, value=(((1, 2, 3), (0, 0, 0), None), (None, (4, 5, 6), None)))
        with self.assertRaises(ValueError):
            state(fields={'contact.contact_position': bad})

    def test_freshness_eligibility_requires_explicit_inputs_and_available_required_fields(self):
        m = api()
        fresh = state(fields={'robot.left_q': field([0] * 6)})
        m.validate_planning_state(fresh, clock(), max_age_s=0, required_fields=('robot.left_q',))
        with self.assertRaises(ValueError):
            m.validate_planning_state(fresh, clock(), max_age_s=0, required_fields=('robot.right_q',))
        with self.assertRaises(ValueError):
            m.validate_planning_state(fresh, clock(), max_age_s=-1, required_fields=())
        with self.assertRaises(TypeError):
            m.validate_planning_state(fresh, clock())
        for inconsistent in (clock(world_step=481), clock(sdk_step=1481)):
            with self.subTest(inconsistent=inconsistent), self.assertRaises(ValueError):
                m.validate_planning_state(fresh, inconsistent, max_age_s=1, required_fields=())

    def test_static_configuration_has_no_measurement_timestamp_or_rate(self):
        m = api()
        mass = field(0.1, unit='kg', frame='not_applicable', kind='static',
                     timestamp_s=None, sample_hz=None, source='configured mass')
        snapshot = state(fields={'object.mass': mass})
        self.assertIsNone(snapshot.fields['object.mass'].timestamp_s)
        with self.assertRaises(ValueError):
            field(0.1, unit='kg', frame='not_applicable', kind='static', timestamp_s=1.0)
        with self.assertRaises(ValueError):
            state(fields={'robot.left_q': field([0] * 6, kind='static', timestamp_s=None, sample_hz=None)})

    def test_measured_and_derived_provenance_are_explicit(self):
        m = api()
        with self.assertRaises(ValueError):
            field(0.1, unit='N', frame='world', kind='derived', dependencies=())
        impulse = field([[0.1] * 3] * 2, unit='N*s', frame='world', kind='measured', force_side='ON_HAND', contact_basis='actual_contact')
        with self.assertRaises(ValueError):
            state(fields={'contact.normal_impulse_derived': impulse})
        with self.assertRaises(ValueError):
            state(fields={'robot.actual_motor_torque': field([[0] * 6] * 2, unit='N*m', frame='joint')})

    def test_on_hand_and_on_object_cannot_be_interchanged(self):
        m = api()
        vectors = [[[0, 0, -1]] * 3] * 2
        good = field(vectors, unit='N', frame='world', kind='derived', force_side='ON_HAND', contact_basis='actual_contact')
        state(fields={'contact.accepted_force_on_hand': good})
        with self.assertRaises(ValueError):
            state(fields={'contact.accepted_force_on_object': good})

    def test_geometry_candidate_is_not_actual_contact(self):
        m = api()
        candidate = field([[True] * 3] * 2, unit='dimensionless', frame='dimensionless', kind='derived', contact_basis='geometry_only')
        snapshot = state(fields={'contact.geometry_candidate': candidate})
        self.assertIsNone(snapshot.fields['contact.contact_mask'].value)
        with self.assertRaises(ValueError):
            state(fields={'contact.contact_mask': candidate})
        accepted = dataclasses.replace(candidate, contact_basis='actual_contact')
        snapshot = state(fields={'contact.contact_mask': accepted})
        self.assertEqual(snapshot.fields['contact.contact_mask'].kind, 'derived')

    def test_whole_hand_rejected_load_does_not_fill_digits(self):
        m = api()
        f = field([2, 3], unit='N', frame='world', kind='derived', force_side='ON_HAND', contact_basis='actual_contact')
        snapshot = state(fields={'contact.rejected_load_hand': f})
        self.assertIsNone(snapshot.fields['contact.rejected_load'].value)

    def test_velocity_requires_the_correct_point_origin(self):
        m = api()
        f = field([0] * 6, unit='m/s|rad/s', frame='world', origin='body_com')
        with self.assertRaises(ValueError):
            state(fields={'wrist.left_velocity': f})
        good = dataclasses.replace(f, origin='palm_origin')
        state(fields={'wrist.left_velocity': good})

    def test_pose_quaternion_and_centerline_shape_are_checked(self):
        m = api()
        bad_pose = field([0] * 7, unit='m|quaternion_wxyz', frame='world')
        with self.assertRaises(ValueError):
            state(fields={'wrist.left_pose': bad_pose})
        values = {'object.centerline': field([[0, 0, 0], [1, 0, 0]], unit='m', frame='world', kind='derived'),
                  'object.centerline_velocity': field([[0, 0, 0]], unit='m/s', frame='world', kind='derived', origin='centerline_points')}
        with self.assertRaises(ValueError):
            state(fields=values)

    def test_incomplete_quaternion_and_unit_vector_components_are_rejected_cleanly(self):
        for name, value, unit, mask in (
                ('wrist.left_pose', [0, 0, 0, 1, None, 0, 0], 'm|quaternion_wxyz', [True] * 4 + [False, True, True]),
                ('object.orientation', [1, None, 0, 0], 'quaternion_wxyz', [True, False, True, True]),
                ('fingertips.left_thumb.normal', [1, None, 0], 'dimensionless', [True, False, True])):
            with self.subTest(name=name), self.assertRaises(ValueError):
                f = field(value, unit=unit, frame='world', kind='derived', valid_mask=mask,
                          missing_reason='one component not measured')
                state(fields={name: f})

    def test_transition_cannot_cross_terminal_reset_episode_or_stale_steps(self):
        m = api()
        first = state()
        newer_clock = clock(world_step=481, world_time_s=1 + 1/480,
                            sdk_step=1481, sdk_time_s=11 + 1/480)
        second = state(newer_clock)
        m.validate_transition(first, second)
        invalid_previous = state(clock(terminal=True))
        with self.assertRaises(ValueError):
            m.validate_transition(invalid_previous, second)
        for c in [clock(), dataclasses.replace(newer_clock, reset_generation=4),
                  dataclasses.replace(newer_clock, episode_uid='other'),
                  dataclasses.replace(newer_clock, env_id=3)]:
            with self.subTest(c=c), self.assertRaises(ValueError):
                m.validate_transition(first, state(c))

    def test_clocks_preserve_sdk_world_offset_and_validate_primitives(self):
        c = clock()
        self.assertEqual(c.sdk_time_s - c.world_time_s, 10)
        for changes in [dict(env_id=-1), dict(world_step=True), dict(sdk_time_s=float('inf')),
                        dict(world_time_s=-1), dict(run_uuid=''), dict(terminal=1)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                clock(**changes)

    def test_unknown_field_is_rejected_and_actor_contract_is_untouched(self):
        m = api()
        with self.assertRaises(ValueError):
            state(fields={'actor.privileged_extension': field([0] * 6)})
        with self.assertRaises(ValueError):
            m.make_mpc_state(clock(), {})
        self.assertEqual(m.STUDENT_OBSERVATION_DIM, 3844)
        self.assertFalse(hasattr(m.MPCState, 'to_actor_observation'))
        self.assertEqual(state().observation_scope, 'simulation_privileged')
        self.assertTrue(state().teacher_only)


if __name__ == '__main__':
    unittest.main()
