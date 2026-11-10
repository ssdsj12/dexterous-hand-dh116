"""CPU-only full-report fixtures; never import Isaac, pxr or torch."""
from dataclasses import FrozenInstanceError, asdict, replace
import importlib
import importlib.util
import sys
from types import SimpleNamespace as NS

import pytest

from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_state import SampleClock

MODULE = 'go2_pvcnn.control.dh116.tabletop_bimanual_mpc_contact_provenance'


def api():
    assert importlib.util.find_spec(MODULE) is not None, 'contact provenance collector is missing'
    return importlib.import_module(MODULE)


def clock(step=10, **changes):
    base = SampleClock('run', 0, 'physical-episode-7', step, step/480,
                       step+100, (step+100)/480, 2, 7)
    return replace(base, **changes)


def raw_report():
    header = NS(actor0=2**63+1, actor1=17, collider0=71, collider1=72,
        stage_id=13, type='CONTACT_PERSIST', proto_index0=0xffffffff,
        proto_index1=0xffffffff, contact_data_offset=0, num_contact_data=1,
        friction_anchors_offset=0, num_friction_anchors_data=1)
    point = NS(position=[.1, .2, .3], normal=[0., 0., -1.],
        impulse=[.001, -.002, .003], separation=-.0001,
        face_index0=0xffffffff, face_index1=0xffffffff, material0=31, material1=32)
    anchor = NS(position=[.2, .2, .3], impulse=[-.001, .004, 0.])
    return [header], [point], [anchor]


class Source:
    native_attested = True  # An injected source's assertion must be ignored.
    source_kind = 'installed_physx'

    def __init__(self, report=None):
        self.report = report if report is not None else raw_report()
        self.polls = 0
        self.subscriptions = 0

    def get_full_contact_report(self):
        self.polls += 1
        return self.report

    def subscribe_full_contact_report_events(self, callback):
        self.subscriptions += 1
        self.callback = callback
        return NS()

    def decode_path(self, path_id, stage_id):
        return '/World/envs/env_0/ForgedFromConfig'


def collector(source=None, **options):
    return api().ContactPairProvenanceCollector(enabled=True, source=source or Source(), **options)


def polled(source=None, **options):
    c = collector(source, **options)
    c.begin_step(clock(), dt_s=1/480)
    result = c.end_step(clock(11), read_clock=lambda: clock(11))
    return c, result


def test_default_off_never_accesses_sdk_or_injected_source(monkeypatch):
    m = api()
    def forbidden(*args, **kwargs):
        raise AssertionError('disabled collector must not import SDK')
    monkeypatch.setattr(m, '_open_installed_source', forbidden)
    c = m.ContactPairProvenanceCollector()
    assert not c.enabled
    assert c.records == ()
    assert c.end_step(clock()) is None
    assert m.open_installed_contact_source() is None


def test_poll_copies_actual_header_uint64_and_all_owned_data():
    source = Source()
    c, r = polled(source)
    h = r.headers[0]
    assert h.actor0 == 2**63+1
    assert h.actor1 == 17 and h.collider0 == 71 and h.collider1 == 72
    assert h.stage_id == 13 and h.proto_index0 == 0xffffffff
    assert h.num_contact_data == h.num_friction_anchors_data == 1
    assert r.points[0].impulse == (.001, -.002, .003)
    assert r.friction_anchors[0].position == (.2, .2, .3)
    source.report[1][0].impulse[0] = 90.
    source.report[0][0].actor1 = 99
    assert r.points[0].impulse[0] == .001 and h.actor1 == 17
    assert r.schema_valid and r.ownership_valid and r.finite_valid
    assert source.polls == 1 and c.records == (r,)
    with pytest.raises(FrozenInstanceError):
        h.actor0 = 1


def test_raw_ids_do_not_become_tensor_indices_or_unverified_paths():
    _, r = polled()
    assert r.id_namespace == 'PhysX_uint64_SdfPath_id_not_Tensor_body_index'
    assert r.headers[0].decoded_paths == ()
    assert r.pair_env_association is None
    assert r.partner_body_id is None
    assert r.native_attested is False
    assert r.measurement_valid is False
    assert r.contact_qualified is False and r.support_qualified is False
    assert r.expert_qualified is False and r.success_qualified is False
    assert r.clock_association is None  # Source fixture cannot attest installed readback.
    assert r.bracket.consistent
    assert r.bracket.before == clock() and r.bracket.after == clock(11)


def test_raw_impulse_retains_unknown_side_units_and_derived_rate_label():
    _, r = polled()
    assert r.force_side == 'NA' and r.frame == 'NA'
    assert r.impulse_unit == 'SDK_raw_contact_impulse; physical_unit_unspecified_in_binding'
    rate = api().derive_impulse_rate(r, dt_s=1/480)
    assert rate.kind == 'derived' and rate.force_side == 'NA'
    assert rate.contact_rates[0] == pytest.approx((.48, -.96, 1.44))
    assert rate.friction_rates[0] == pytest.approx((-.48, 1.92, 0.))
    assert rate.unit == 'SDK_raw_contact_impulse/s; not_qualified_SI_force'
    assert not rate.measurement_valid
    for dt in (0., -1., float('nan'), True):
        with pytest.raises(ValueError):
            api().derive_impulse_rate(r, dt_s=dt)


@pytest.mark.parametrize('field,value', [
    ('actor0', -1), ('collider1', 2**64), ('stage_id', True),
    ('contact_data_offset', -1), ('num_contact_data', 1.5),
    ('proto_index1', 2**32), ('friction_anchors_offset', 2),
    ('num_friction_anchors_data', 2),
])
def test_invalid_header_is_retained_but_cannot_own_measurement(field, value):
    source = Source()
    setattr(source.report[0][0], field, value)
    _, r = polled(source)
    assert getattr(r.headers[0], field) == value
    assert r.errors and not r.raw_valid
    assert not r.measurement_valid and r.clock_association is None


def test_overlapping_header_ranges_and_orphan_rows_reject_ownership():
    source = Source()
    source.report[0].append(NS(**vars(source.report[0][0])))
    _, r = polled(source)
    assert not r.ownership_valid and 'overlap' in ' '.join(r.errors)
    source = Source()
    source.report[1].append(NS(**vars(source.report[1][0])))
    _, r = polled(source)
    assert not r.ownership_valid and r.unowned_contact_indices == (1,)


@pytest.mark.parametrize('array,field', [(1, 'position'), (1, 'normal'), (1, 'impulse'), (2, 'impulse')])
def test_nonfinite_owned_contact_and_anchor_stay_raw_and_invalid(array, field):
    source = Source()
    getattr(source.report[array][0], field)[1] = float('nan')
    _, r = polled(source)
    assert not r.finite_valid and not r.raw_valid
    assert not r.measurement_valid


def test_wrong_runtime_schema_and_missing_friction_are_explicit_na():
    source = Source(raw_report()[:2])
    _, r = polled(source)
    assert not r.schema_valid and r.friction_anchors is None
    assert len(r.points) == 1 and r.runtime_arity == 2
    assert 'full_report_requires_three_vectors' in r.errors


def test_empty_report_is_observed_event_count_zero_not_no_contact_proof():
    _, r = polled(Source(([], [], [])))
    assert r.raw_valid and not r.no_contact_attested
    assert not r.upstream_overflow_absence_attested
    assert r.capacity.header_limit is None and r.capacity.contact_limit is None
    assert r.capacity.upstream_overflow is None
    assert not r.measurement_valid


def test_local_capture_budget_keeps_raw_but_marks_incomplete():
    _, r = polled(contact_limit=1, anchor_limit=1, header_limit=1)
    assert len(r.points) == 1 and len(r.friction_anchors) == 1
    assert r.capacity.at_or_above_limit
    assert not r.raw_valid and 'capture_budget_reached' in r.errors
    assert r.capacity.upstream_overflow is None


def test_callback_has_no_sdk_frame_and_is_never_associated_by_arrival():
    source = Source()
    c = collector(source)
    c.subscribe()
    assert source.subscriptions == 1
    source.callback(*raw_report())
    c.begin_step(clock(), dt_s=1/480)
    source.callback(*raw_report())
    source.callback(*raw_report())
    c.end_step(clock(11), poll=False)
    assert len(c.records) == 3
    assert all(r.clock_association is None and not r.measurement_valid for r in c.records)
    assert c.records[0].bracket is None
    assert c.records[1].bracket.before == clock()
    assert c.records[2].duplicate_candidate
    assert all('callback_has_no_frame_identifier' in r.errors for r in c.records)


@pytest.mark.parametrize('after', [
    clock(), clock(12), clock(11, sdk_step=110), clock(11, world_time_s=9.),
    clock(11, episode_uid='new'), clock(11, reset_generation=8),
    clock(11, reset=True), clock(11, env_id=1),
])
def test_noop_multistep_clock_drift_reset_terminal_cross_env_reject_association(after):
    c = collector()
    c.begin_step(clock(), dt_s=1/480)
    r = c.end_step(after)
    assert not r.bracket.consistent and r.clock_association is None
    assert len(r.points) == 1 and not r.measurement_valid


def test_poll_clock_receipts_must_enclose_getter_and_be_unchanged():
    source = Source()
    c = collector(source)
    c.begin_step(clock(), dt_s=1/480)
    readings = iter((clock(11), clock(12)))
    r = c.end_step(clock(11), read_clock=lambda: next(readings))
    assert r.read_before == clock(11) and r.read_after == clock(12)
    assert 'poll_clock_changed' in r.errors and not r.raw_valid


def test_terminal_first_record_survives_noop_and_callback_after_close():
    c = collector()
    c.begin_step(clock(), dt_s=1/480)
    first = c.end_step(clock(11, terminal=True), read_clock=lambda: clock(11, terminal=True))
    c.begin_step(clock(11, terminal=True), dt_s=1/480)
    c.end_step(clock(11, terminal=True))
    c.receive_callback(*raw_report())
    assert c.first_terminal_record is first
    assert len(c.records) == 3
    assert c.records[-1].bracket is None


def test_unfinished_step_budget_and_abandoned_bracket_archive_without_association():
    c = collector()
    c.begin_step(clock(), dt_s=1/480)
    c.receive_callback(*raw_report())
    c.abort_step('runner_budget_exhausted')
    assert c.records[0].clock_association is None
    assert c.records[0].bracket.after is None
    assert c.aborted_steps[0][1] == 'runner_budget_exhausted'
    c.begin_step(clock(), dt_s=1/480)
    r = c.end_step(clock(11), step_completed=False)
    assert not r.bracket.consistent and 'step_not_completed' in r.bracket.errors


def test_poll_exception_archives_unavailable_evidence_instead_of_fabricated_zero():
    class Broken(Source):
        def get_full_contact_report(self):
            raise RuntimeError('stream unavailable')
    _, r = polled(Broken())
    assert not r.schema_valid and r.points is None
    assert 'RuntimeError: stream unavailable' in ' '.join(r.errors)


def test_source_binding_is_explicit_and_module_has_no_eager_sdk_imports():
    m = api()
    assert m.INSTALLED_BINDING.sha256 == 'b75b94176c7e6b766bf6f36058d2115016eaabc0d45d8b36a28c64cbf29f4dd9'
    assert m.INSTALLED_BINDING.poll_lines == (1292, 1304)
    assert m.INSTALLED_BINDING.callback_lines == (1392, 1403)
    assert not any(name in sys.modules for name in ('omni.physx', 'pxr'))


def test_repeated_step_receipt_is_archived_but_rejected():
    c, _ = polled()
    c.begin_step(clock(), dt_s=1/480)
    repeated = c.end_step(clock(11))
    assert len(c.records) == 2 and repeated.points[0].impulse == (.001, -.002, .003)
    assert 'duplicate_step_receipt' in repeated.errors
    assert repeated.clock_association is None


def test_episode_stays_closed_after_first_terminal_even_with_later_nonterminal_receipts():
    c = collector()
    c.begin_step(clock(), dt_s=1/480)
    first = c.end_step(clock(11, terminal=True), read_clock=lambda: clock(11, terminal=True))
    c.begin_step(clock(11), dt_s=1/480)
    late = c.end_step(clock(12))
    assert 'physical_episode_already_terminal' in late.errors
    assert c.first_terminal_record is first and late.clock_association is None


def test_old_reset_generation_cannot_resume_after_new_physical_episode_seen():
    c, _ = polled()
    c.begin_step(clock(12, reset_generation=8, episode_uid='physical-episode-8'), dt_s=1/480)
    next_episode = clock(13, reset_generation=8, episode_uid='physical-episode-8')
    c.end_step(next_episode, read_clock=lambda: next_episode)
    c.begin_step(clock(14), dt_s=1/480)
    stale = c.end_step(clock(15))
    assert 'stale_reset_generation' in stale.errors


def test_actual_installed_enum_name_and_value_are_both_preserved():
    source = Source()
    source.report[0][0].type = NS(name='CONTACT_PERSIST', value=2)
    _, r = polled(source)
    assert r.headers[0].type.name == 'CONTACT_PERSIST'
    assert r.headers[0].type.value == 2
    assert r.raw_valid


def test_unknown_event_type_is_kept_raw_but_rejected():
    source = Source()
    source.report[0][0].type = NS(name='CONTACT_MAYBE', value=97)
    _, r = polled(source)
    assert r.headers[0].type.name == 'CONTACT_MAYBE'
    assert r.headers[0].type.value == 97
    assert not r.raw_valid and 'event_type_invalid' in ' '.join(r.errors)


def test_lost_event_without_contact_points_does_not_become_contact_absence_proof():
    source = Source()
    header = source.report[0][0]
    header.type = NS(name='CONTACT_LOST', value=1)
    header.num_contact_data = header.num_friction_anchors_data = 0
    source.report = (source.report[0], [], [])
    _, r = polled(source)
    assert r.raw_valid and r.headers[0].type.name == 'CONTACT_LOST'
    assert not r.no_contact_attested


def test_installed_factory_rejects_source_sha_drift_before_any_sdk_import(monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(Path, 'read_bytes', lambda path: b'untrusted source text')
    called = []
    monkeypatch.setattr(importlib, 'import_module', lambda name: called.append(name))
    with pytest.raises(RuntimeError, match='SHA differs'):
        api_module = sys.modules[MODULE]
        api_module.open_installed_contact_source(enabled=True)
    assert called == []


def test_enabled_constructor_never_implicitly_connects_or_subscribes(monkeypatch):
    m = api()
    def forbidden(*args, **kwargs):
        raise AssertionError('automatic SDK connection forbidden')
    monkeypatch.setattr(m, '_open_installed_source', forbidden)
    c = m.ContactPairProvenanceCollector(enabled=True)
    assert c.records == ()
    c.begin_step(clock(), dt_s=1/480)
    r = c.end_step(clock(11))
    assert r.points is None and not r.raw_valid


def test_archive_serialization_carries_candidate_clocks_unknown_units_and_fixed_flags():
    _, r = polled()
    data = asdict(r)
    assert data['headers'][0]['actor0'] == 2**63+1
    assert data['bracket']['before']['episode_uid'] == 'physical-episode-7'
    assert data['bracket']['after']['policy_step'] == 2
    assert data['source_binding']['sha256'] == api().INSTALLED_BINDING.sha256
    assert data['pair_env_association'] is None and data['partner_body_id'] is None
    assert data['native_attested'] is False and data['force_side'] == 'NA'
    assert data['capacity']['upstream_overflow'] is None


def test_installed_factory_rejects_module_from_different_installation(monkeypatch):
    import hashlib
    from pathlib import Path
    m = api()
    monkeypatch.setattr(Path, 'read_bytes', lambda path: b'source fixture')
    monkeypatch.setattr(hashlib, 'sha256', lambda raw: NS(hexdigest=lambda: m.INSTALLED_BINDING.sha256))
    imports = []
    def import_fixture(name):
        imports.append(name)
        return NS(__file__='/different/installation/omni/physx/__init__.py')
    monkeypatch.setattr(importlib, 'import_module', import_fixture)
    with pytest.raises(RuntimeError, match='different installation'):
        m.open_installed_contact_source(enabled=True)
    assert imports == ['omni.physx']


def test_first_terminal_same_episode_is_last_physical_sample_with_candidate_clock():
    c = collector()
    terminal = clock(11, terminal=True)
    c.begin_step(clock(), dt_s=1/480)
    first = c.end_step(terminal, read_clock=lambda: terminal)
    assert first.bracket.consistent
    assert first.receipt_clock_candidate == terminal
    assert first.clock_association is None  # Pure CPU fixture, never Native.
    assert first.measurement_valid is False and first.planning_eligible is False
    assert c.first_terminal_record is first
    c.begin_step(terminal, dt_s=1/480)
    noop = c.end_step(terminal, read_clock=lambda: terminal)
    assert not noop.bracket.consistent and noop.receipt_clock_candidate is None
    assert c.first_terminal_record is first


def test_first_terminal_is_preserved_per_environment_and_physical_episode():
    c = collector()
    c.begin_step(clock(), dt_s=1/480)
    first_env0 = c.end_step(clock(11, terminal=True), read_clock=lambda: clock(11, terminal=True))
    c.begin_step(clock(env_id=1), dt_s=1/480)
    first_env1 = c.end_step(clock(11, env_id=1, terminal=True),
                          read_clock=lambda: clock(11, env_id=1, terminal=True))
    c.begin_step(clock(11, terminal=True), dt_s=1/480)
    c.end_step(clock(11, terminal=True))
    assert c.first_terminal_records == (first_env0, first_env1)


@pytest.mark.parametrize('bad_before,bad_after,completed,read_kind', [
    (clock(10, terminal=True), clock(10, terminal=True), True, 'stable'),
    (clock(11, terminal=True), clock(11, terminal=True), True, 'stable'),
    (clock(10), clock(10, terminal=True), True, 'stable'),
    (clock(10), clock(12, terminal=True), True, 'stable'),
    (clock(10), clock(11, terminal=True, reset=True), True, 'stable'),
    (clock(10), clock(11, terminal=True, episode_uid='other'), True, 'stable'),
    (clock(10), clock(11, terminal=True, reset_generation=8), True, 'stable'),
    (clock(10), clock(11, terminal=True, sdk_step=110), True, 'stable'),
    (clock(10), clock(11, terminal=True, sdk_time_s=99.), True, 'stable'),
    (clock(10), clock(11, terminal=True), False, 'stable'),
    (clock(10), clock(11, terminal=True), True, 'changed'),
    (clock(10), clock(11, terminal=True), True, 'missing'),
])
def test_invalid_terminal_attempt_cannot_pin_or_poison_later_real_terminal(
        bad_before, bad_after, completed, read_kind):
    c = collector()
    c.begin_step(bad_before, dt_s=1/480)
    if read_kind == 'changed':
        readings = iter((bad_after, replace(bad_after, sdk_time_s=bad_after.sdk_time_s+1.)))
        reader = lambda: next(readings)
    else:
        reader = (lambda: bad_after) if read_kind == 'stable' else None
    bad = c.end_step(bad_after, read_clock=reader, step_completed=completed)
    assert len(bad.points) == 1 and c.records == (bad,)
    assert c.first_terminal_record is None and c.first_terminal_records == ()
    assert bad.receipt_clock_candidate is None and bad.clock_association is None
    terminal = clock(11, terminal=True)
    c.begin_step(clock(10), dt_s=1/480)
    good = c.end_step(terminal, read_clock=lambda: terminal)
    assert good.receipt_clock_candidate == terminal
    assert c.first_terminal_record is good and c.first_terminal_records == (good,)
    assert not set(good.errors) & {'duplicate_step_receipt', 'physical_episode_already_terminal',
        'stale_reset_generation', 'physical_episode_changed_without_reset_generation'}
    assert not good.native_attested and not good.measurement_valid and not good.planning_eligible


def test_terminal_poll_without_begin_cannot_pin_or_poison_later_real_terminal():
    c = collector()
    terminal = clock(11, terminal=True)
    bad = c.end_step(terminal, read_clock=lambda: terminal)
    assert bad.bracket is None and len(bad.points) == 1
    assert c.first_terminal_record is None and c.first_terminal_records == ()
    c.begin_step(clock(), dt_s=1/480)
    good = c.end_step(terminal, read_clock=lambda: terminal)
    assert good.receipt_clock_candidate == terminal and c.first_terminal_record is good


@pytest.mark.parametrize('raw_kind', ['wrong_schema', 'nonfinite', 'budget_reached', 'getter_raises'])
def test_real_terminal_step_with_invalid_report_keeps_metadata_and_na_association(raw_kind):
    source = Source()
    options = {}
    if raw_kind == 'wrong_schema':
        source.report = source.report[:2]
    elif raw_kind == 'nonfinite':
        source.report[1][0].impulse[0] = float('nan')
    elif raw_kind == 'budget_reached':
        options['contact_limit'] = 1
    else:
        def unavailable():
            raise RuntimeError('stream unavailable')
        source.get_full_contact_report = unavailable
    c = collector(source, **options)
    terminal = clock(11, terminal=True)
    c.begin_step(clock(), dt_s=1/480)
    first = c.end_step(terminal, read_clock=lambda: terminal)
    assert first.bracket.consistent and first.read_before == first.read_after == terminal
    assert c.first_terminal_record is first and c.first_terminal_records == (first,)
    assert first.receipt_clock_candidate is None and first.clock_association is None
    assert not first.raw_valid and not first.measurement_valid and not first.native_attested
    c.begin_step(clock(11), dt_s=1/480)
    late_clock = clock(12)
    late = c.end_step(late_clock, read_clock=lambda: late_clock)
    assert 'physical_episode_already_terminal' in late.errors
    assert c.first_terminal_record is first


@pytest.mark.parametrize('source', [None, NS(), NS(get_full_contact_report=None)])
def test_missing_or_noncallable_getter_cannot_pin_terminal_metadata(source):
    c = api().ContactPairProvenanceCollector(enabled=True, source=source)
    terminal = clock(11, terminal=True)
    c.begin_step(clock(), dt_s=1/480)
    r = c.end_step(terminal, read_clock=lambda: terminal)
    assert r.points is None and not r.raw_valid
    assert c.first_terminal_record is None and c.first_terminal_records == ()
    assert r.receipt_clock_candidate is None and r.clock_association is None
