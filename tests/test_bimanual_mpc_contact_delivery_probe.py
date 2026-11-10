"""CPU observer fixtures: these never attest installed SDK or Native delivery."""
from dataclasses import replace
import gc
import importlib
import importlib.util
from types import SimpleNamespace as NS
import threading
import weakref

import pytest

from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_state import SampleClock

MODULE = 'go2_pvcnn.control.dh116.tabletop_bimanual_mpc_contact_delivery_probe'
DT = 1 / 480


def api():
    assert importlib.util.find_spec(MODULE) is not None, 'delivery probe is missing'
    return importlib.import_module(MODULE)


def clock(step=10, **changes):
    return replace(SampleClock('run', 0, 'episode', step, step * DT,
                              step + 100, (step + 100) * DT, 2, 7), **changes)


def raw_report():
    return ([NS(actor0=2**63+1, actor1=17, collider0=71, collider1=72,
                stage_id=13, type='CONTACT_PERSIST', proto_index0=0xffffffff,
                proto_index1=0xffffffff, contact_data_offset=0, num_contact_data=1,
                friction_anchors_offset=0, num_friction_anchors_data=1)],
            [NS(position=[.1, .2, .3], normal=[0., 0., -1.],
                impulse=[.001, -.002, .003], separation=-.0001,
                face_index0=0xffffffff, face_index1=0xffffffff,
                material0=31, material1=32)],
            [NS(position=[.2, .2, .3], impulse=[-.001, .004, 0.])])


class Token:
    """Model the audited SDK demo's holder-reference-drop lifetime."""
    def __init__(self, source):
        self.source = weakref.ref(source)

    def __del__(self):
        source = self.source()
        if source is not None:
            source.released += 1
            source.callback = None


class Source:
    native_attested = True  # A CPU dependency's self-description is ignored.

    def __init__(self):
        self.raw = raw_report()
        self.calls = []
        self.callback = None
        self.released = 0
        self.stage_id = 13

    def subscribe_physics_on_step_events(self, callback, *, pre_step, order):
        self.calls.append(('post', pre_step, order))
        self.callback = callback
        return Token(self)

    def subscribe_full_contact_report_events(self, callback):
        self.calls.append(('full',))
        self.callback = callback
        return Token(self)

    def get_full_contact_report(self):
        self.calls.append(('poll',))
        return self.raw

    def get_attached_stage(self):
        return self.stage_id


def probe(mode='post_step_poll', source=None, **options):
    source = source or Source()
    current = [clock()]
    owner = api().ContactDeliveryProbe(source, lambda: current[0], mode,
                                     enabled=True, **options)
    return owner, source, current


def test_default_off_touches_neither_source_clock_nor_sdk(monkeypatch):
    m = api()
    class Forbidden:
        def __getattribute__(self, name):
            raise AssertionError(name)
        def __call__(self):
            raise AssertionError('clock')
    p = m.ContactDeliveryProbe(Forbidden(), Forbidden())
    assert p.begin_step(object(), dt_s=object()) is None
    assert p.end_step(object()) is None
    p.close()
    assert p.records == () and not p.subscription_active
    def forbidden_factory(*args, **kwargs):
        raise AssertionError('disabled path invoked the SDK factory')
    # Older pytest inspects replacement values' __class__ during setattr.
    # The factory sentinel tests invocation; hostile inputs above still test
    # that the disabled owner inspects neither source nor clock attributes.
    monkeypatch.setattr(m, '_open_installed_delivery_probe', forbidden_factory)
    assert m.open_installed_delivery_probe() is None


@pytest.mark.parametrize('mode,expected', [
    ('post_step_poll', ('post', False, 9)), ('full_report_callback', ('full',)),
])
def test_modes_install_exactly_one_observer_and_drop_holder_on_close(mode, expected):
    p, source, _ = probe(mode, order=9)
    assert source.calls == [expected]
    assert p.subscription_active
    callback = source.callback
    p.close()
    p.close()
    gc.collect()
    assert source.released == 1 and not p.subscription_active
    callback(DT) if mode == 'post_step_poll' else callback(*raw_report())
    assert p.records == ()


@pytest.mark.parametrize('mode', ['post_step_poll', 'full_report_callback'])
def test_callback_immediately_copies_all_three_vectors_and_never_assigns_a_frame(mode):
    p, source, current = probe(mode)
    try:
        p.begin_step(clock(), dt_s=DT)
        current[0] = clock(11)
        source.callback(DT) if mode == 'post_step_poll' else source.callback(*source.raw)
        source.raw[0][0].actor1 = 99
        source.raw[1][0].impulse[0] = 90.
        source.raw[2][0].impulse[0] = 80.
        receipt = p.records[0]
        report = receipt.report
        assert report.headers[0].actor1 == 17
        assert report.points[0].impulse == (.001, -.002, .003)
        assert report.friction_anchors[0].impulse == (-.001, .004, 0.)
        assert report.raw_valid and report.clock_association is None
        assert report.receipt_clock_candidate is None and report.pair_env_association is None
        assert receipt.arrival_clock_before == receipt.arrival_clock_after == clock(11)
        assert receipt.actual_callback_dt_s == (DT if mode == 'post_step_poll' else None)
        assert receipt.thread_id > 0 and receipt.arrival_monotonic_ns > 0
        assert receipt.clock_association is None and not receipt.native_attested
        result = p.end_step(clock(11), poll_report=None)
        assert result.callback_receipts == (receipt,)
        assert result.bracket.consistent and result.callback_count == 1
        assert result.poll_report is None and result.pair_env_association is None
        assert not result.planning_eligible and not result.native_force_qualified
        assert not result.support_qualified and not result.expert_qualified
        assert source.calls.count(('poll',)) == (mode == 'post_step_poll')
    finally:
        p.close()


def test_orphan_multiple_and_worker_callbacks_remain_separate_global_arrivals():
    p, source, current = probe('full_report_callback')
    try:
        source.callback(*source.raw)
        assert p.records[0].active_bracket_sequence is None
        assert 'callback_without_active_bracket' in p.records[0].errors
        p.begin_step(clock(), dt_s=DT)
        current[0] = clock(11)
        worker = threading.Thread(target=lambda: source.callback(*source.raw), name='cpu-worker')
        worker.start()
        worker.join()
        source.callback(*source.raw)
        result = p.end_step(clock(11))
        assert [r.sequence for r in result.callback_receipts] == [1, 2]
        assert 'callback_on_different_thread' in result.callback_receipts[0].errors
        assert 'multiple_callbacks_in_arrival_window' in result.errors
        assert result.callback_receipts[0].thread_name == 'cpu-worker'
        assert all(r.report.pair_env_association is None for r in p.records)
        source.callback(*source.raw)
        assert p.records[-1].active_bracket_sequence is None
    finally:
        p.close()


@pytest.mark.parametrize('changes,error', [
    ({'episode_uid': 'other'}, 'step_identity_changed'),
    ({'reset_generation': 8, 'reset': True}, 'step_identity_changed'),
    ({'sdk_time_s': .9}, 'sdk_time_s_delta_mismatch'),
    ({'world_step': 13}, 'requires_exactly_one_world_and_sdk_physics_step'),
])
def test_bad_advance_receipts_are_archived_with_no_frame_or_environment_qualification(changes, error):
    p, source, current = probe()
    try:
        p.begin_step(clock(), dt_s=DT)
        current[0] = clock(11, **changes)
        source.callback(DT)
        result = p.end_step(current[0])
        assert not result.bracket.consistent and error in result.errors
        assert len(result.callback_receipts) == 1
        assert result.clock_association is None and result.pair_env_association is None
        assert not result.native_attested and not result.planning_eligible
    finally:
        p.close()


def test_callback_dt_stage_and_identity_errors_do_not_discard_raw_vectors():
    p, source, current = probe()
    try:
        p.begin_step(clock(), dt_s=DT)
        source.stage_id = 14
        current[0] = clock(11, episode_uid='wrong')
        source.callback(DT * 2)
        result = p.end_step(clock(11))
        arrival = result.callback_receipts[0]
        assert arrival.report.headers[0].stage_id == 13
        assert 'callback_dt_mismatch' in arrival.errors
        assert 'header_stage_differs_from_attached_stage' in arrival.errors
        assert 'callback_identity_differs_from_active_bracket' in arrival.errors
        assert 'attached_stage_changed_in_advance' in result.errors
        assert result.attached_stage_before == 13 and result.attached_stage_after == 14
    finally:
        p.close()


def test_getter_and_clock_failures_are_archived_without_leaking_callback_exceptions():
    source = Source()
    def unavailable():
        raise RuntimeError('CPU fixture unavailable')
    source.get_full_contact_report = unavailable
    p = api().ContactDeliveryProbe(source, unavailable, enabled=True)
    try:
        source.callback(DT)
        arrival = p.records[0]
        assert arrival.report.headers is None and not arrival.report.raw_valid
        assert any('callback_poll_unavailable:RuntimeError' in e for e in arrival.errors)
        assert any('callback_clock_unavailable:RuntimeError' in e for e in arrival.errors)
        assert arrival.arrival_clock_before is arrival.arrival_clock_after is None
    finally:
        p.close()


def test_close_archives_unfinished_bracket_and_retained_callback_cannot_write():
    p, source, _ = probe()
    p.begin_step(clock(), dt_s=DT)
    source.callback(DT)
    callback = source.callback
    p.close()
    assert len(p.step_diagnostics) == 1
    result = p.step_diagnostics[0]
    assert 'probe_closed_with_active_bracket' in result.errors
    assert not result.bracket.consistent and result.bracket.after is None
    callback(DT)
    assert len(p.records) == 1
    with pytest.raises(RuntimeError, match='closed'):
        p.begin_step(clock(), dt_s=DT)


def test_end_keeps_original_poll_object_and_does_not_poll_again():
    from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_contact_provenance import ContactPairProvenanceCollector
    source = Source()
    original = ContactPairProvenanceCollector(enabled=True, source=source)
    original.begin_step(clock(), dt_s=DT)
    report = original.end_step(clock(11), read_clock=lambda: clock(11))
    p, _, _ = probe('full_report_callback', source)
    try:
        p.begin_step(clock(), dt_s=DT)
        result = p.end_step(clock(11), poll_report=report)
        assert result.poll_report is report and report.clock_association is None
        assert source.calls.count(('poll',)) == 1
        assert 'no_callback_in_arrival_window' in result.errors
    finally:
        p.close()


def test_partial_construction_failure_releases_subscription_even_if_validation_fails(monkeypatch):
    m = api()
    source = Source()
    real_subscribe = source.subscribe_physics_on_step_events
    # Inject a failure after a genuine ownership reference was acquired.
    original_read_stage = m.ContactDeliveryProbe._read_stage
    def fail_after_install(self, errors):
        if self.subscription_active:
            raise RuntimeError('post-install validation failed')
        return original_read_stage(self, errors)
    monkeypatch.setattr(m.ContactDeliveryProbe, '_read_stage', fail_after_install)
    source.subscribe_physics_on_step_events = real_subscribe
    # Construction must take a post-install lifecycle receipt.
    with pytest.raises(RuntimeError, match='post-install validation'):
        m.ContactDeliveryProbe(source, lambda: clock(), enabled=True)
    gc.collect()
    assert source.released == 1 and source.callback is None


def test_context_manager_preserves_original_advance_error_and_closes():
    p, source, _ = probe()
    original = RuntimeError('original controller failure')
    with pytest.raises(RuntimeError) as caught:
        with p:
            p.begin_step(clock(), dt_s=DT)
            raise original
    assert caught.value is original
    assert source.released == 1 and not p.subscription_active


@pytest.mark.parametrize('mode', ['both', '', None, ['post_step_poll', 'full_report_callback']])
def test_invalid_mode_is_rejected_before_source_access(mode):
    with pytest.raises(ValueError, match='exactly one'):
        api().ContactDeliveryProbe(mode=mode)


def test_factory_rejects_cpu_source_before_sdk_import_even_when_it_claims_native(monkeypatch):
    m = api()
    real_import = importlib.import_module
    def imports(name, *args, **kwargs):
        if name.startswith(('omni', 'pxr', 'isaac')):
            raise AssertionError('CPU source must be rejected before SDK import')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(importlib, 'import_module', imports)
    with pytest.raises(ValueError, match='genuine installed contact source'):
        m.open_installed_delivery_probe(enabled=True, source=Source(), clock_reader=lambda: clock())


@pytest.mark.parametrize('mode', ['post_step_poll', 'full_report_callback'])
def test_factory_cpu_fixture_proves_wiring_without_granting_native_attestation(monkeypatch, tmp_path, mode):
    """Even a deliberately faked binding/import fixture cannot issue Native truth."""
    import hashlib
    from go2_pvcnn.control.dh116 import tabletop_bimanual_mpc_contact_provenance as provenance
    m = api()
    binding_path = tmp_path / 'omni' / 'physx' / 'bindings' / '_physx.pyi'
    binding_path.parent.mkdir(parents=True)
    binding_path.write_bytes(b'CPU fixture only, no SDK')
    binding = replace(provenance.INSTALLED_BINDING, path=str(binding_path),
                      sha256=hashlib.sha256(binding_path.read_bytes()).hexdigest())
    monkeypatch.setattr(m, 'INSTALLED_BINDING', binding)
    simulation = Source()
    physics = Source()
    module = NS(__file__=str(binding_path.parent.parent / '__init__.py'),
                get_physx_interface=lambda: physics)
    source = provenance._InstalledContactSource(provenance._INSTALLED_SOURCE_KEY,
                                                simulation, NS(), None)
    real_import = importlib.import_module
    def imports(name, *args, **kwargs):
        if name == 'omni.physx':
            return module
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(importlib, 'import_module', imports)
    p = m.open_installed_delivery_probe(enabled=True, source=source,
                                        clock_reader=lambda: clock(11), mode=mode, order=17)
    try:
        assert not p.native_attested
        if mode == 'post_step_poll':
            assert physics.calls == [('post', False, 17)] and simulation.calls == []
            physics.callback(DT)
        else:
            assert simulation.calls == [('full',)] and physics.calls == []
            simulation.callback(*simulation.raw)
        assert p.records[0].clock_association is None and not p.records[0].native_attested
        assert p.records[0].report.clock_association is None
    finally:
        p.close()
    assert (physics if mode == 'post_step_poll' else simulation).released == 1


def test_factory_wrong_source_hash_does_not_import_or_create_subscription(monkeypatch, tmp_path):
    from go2_pvcnn.control.dh116 import tabletop_bimanual_mpc_contact_provenance as provenance
    m = api()
    binding_path = tmp_path / 'binding.pyi'
    binding_path.write_bytes(b'wrong source')
    monkeypatch.setattr(m, 'INSTALLED_BINDING', replace(provenance.INSTALLED_BINDING,
                                                      path=str(binding_path)))
    source = provenance._InstalledContactSource(provenance._INSTALLED_SOURCE_KEY,
                                                Source(), NS(), None)
    def forbidden(*args, **kwargs):
        raise AssertionError('hash guard must precede SDK import')
    monkeypatch.setattr(importlib, 'import_module', forbidden)
    with pytest.raises(RuntimeError, match='source SHA'):
        m.open_installed_delivery_probe(enabled=True, source=source, clock_reader=lambda: clock())
    assert source.interface.calls == []


def test_release_survives_close_archive_error_and_preserves_controller_exception(monkeypatch):
    p, source, _ = probe()
    original = RuntimeError('original advance failed')
    def fail_archive(*args, **kwargs):
        raise OSError('archive failed too')
    monkeypatch.setattr(p, '_finish_bracket', fail_archive)
    with pytest.raises(RuntimeError) as caught:
        with p:
            p.begin_step(clock(), dt_s=DT)
            raise original
    assert caught.value is original and source.released == 1
    assert any('OSError' in e for e in p.cleanup_errors)


def test_unexpected_raw_copier_error_is_censored_and_observer_remains_live(monkeypatch):
    p, source, _ = probe()
    def fail_copy(*args, **kwargs):
        raise RuntimeError('raw copier unexpected failure')
    monkeypatch.setattr(api().ContactPairProvenanceCollector, '_capture', fail_copy)
    try:
        source.callback(DT)
        assert len(p.records) == 1 and p.records[0].report is None
        assert any('callback_copy_unavailable:RuntimeError' in e for e in p.records[0].errors)
        assert p.subscription_active
    finally:
        p.close()


def test_poll_comparison_from_another_step_is_retained_and_explicitly_unmatched():
    from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_contact_provenance import ContactPairProvenanceCollector
    source = Source()
    original = ContactPairProvenanceCollector(enabled=True, source=source)
    original.begin_step(clock(8), dt_s=DT)
    previous_report = original.end_step(clock(9), read_clock=lambda: clock(9))
    p, _, _ = probe('full_report_callback', source)
    try:
        p.begin_step(clock(), dt_s=DT)
        source.callback(*source.raw)
        result = p.end_step(clock(11), poll_report=previous_report)
        assert result.poll_report is previous_report
        assert 'poll_comparison_bracket_mismatch' in result.errors
        assert previous_report.bracket.after == clock(9)
    finally:
        p.close()


def test_binding_subscription_and_reference_release_evidence_are_archived():
    p, source, _ = probe(order=11)
    try:
        source.callback(DT)
        receipt = p.records[0]
        assert receipt.subscription_binding_lines == (1845, 1859)
        assert receipt.subscription_order == 11
        assert receipt.subscription_binding.sha256 == (
            'b75b94176c7e6b766bf6f36058d2115016eaabc0d45d8b36a28c64cbf29f4dd9')
        assert receipt.subscription_release_method == 'audited_SDK_demo_holder_reference_drop'
        assert receipt.subscription_release_source_sha256 == (
            '091262b11a9ff1c35ab92aed6092e662836fbac29d207da0b93f5256a4ece7fe')
    finally:
        p.close()


def test_malformed_full_callback_preserves_census_without_fabricating_anchor_vector():
    p, source, _ = probe('full_report_callback')
    try:
        source.callback(*source.raw[:2])
        result = p.records[0].report
        assert result.runtime_arity == 2 and result.friction_anchors is None
        assert not result.raw_valid and 'full_report_requires_three_vectors' in result.errors
    finally:
        p.close()


def test_close_archive_failure_is_raised_after_resource_drop(monkeypatch):
    p, source, _ = probe()
    p.begin_step(clock(), dt_s=DT)
    def unavailable(*args, **kwargs):
        raise OSError('CPU archive failure')
    monkeypatch.setattr(p, '_finish_bracket', unavailable)
    with pytest.raises(OSError, match='CPU archive failure'):
        p.close()
    assert source.released == 1 and not p.subscription_active
    p.close()


@pytest.mark.parametrize('bad', [0, 1, None, 'true'])
def test_enabled_requires_an_explicit_boolean(bad):
    with pytest.raises(ValueError, match='enabled must be boolean'):
        api().ContactDeliveryProbe(enabled=bad)
    with pytest.raises(ValueError, match='enabled must be boolean'):
        api().open_installed_delivery_probe(enabled=bad)


@pytest.mark.parametrize('mode', ['post_step_poll', 'full_report_callback'])
def test_callback_owns_original_payload_before_waiting_for_bookkeeping_lock(mode):
    p, source, current = probe(mode)
    entered = threading.Event()
    actual_lock = p._lock
    class ObservedLock:
        def __enter__(self):
            if threading.current_thread().name == 'delayed-callback':
                entered.set()
            actual_lock.acquire()
            return self
        def __exit__(self, *args):
            actual_lock.release()
    p._lock = ObservedLock()
    try:
        p.begin_step(clock(), dt_s=DT)
        with actual_lock:
            worker = threading.Thread(target=lambda: (source.callback(DT) if mode == 'post_step_poll'
                                                       else source.callback(*source.raw)),
                                      name='delayed-callback')
            worker.start()
            assert entered.wait(2), 'fixture callback failed to enter'
            current[0] = clock(11)
            first = p.end_step(clock(11))
            p.begin_step(clock(11), dt_s=DT)
            # SDK vectors are borrowed for one step. Simulate reuse while the
            # owner bookkeeping lock is held: callback MUST already own all
            # three original vectors, including the post-step getter result.
            source.raw[0][0].actor1 = 999
            source.raw[1][0].impulse[:] = [91., 92., 93.]
            source.raw[2][0].impulse[:] = [81., 82., 83.]
        worker.join(2)
        assert not worker.is_alive()
        receipt = p.records[0]
        assert receipt.report.headers[0].actor1 == 17
        assert receipt.report.points[0].impulse == (.001, -.002, .003)
        assert receipt.report.friction_anchors[0].impulse == (-.001, .004, 0.)
        assert receipt.arrival_clock_before == receipt.arrival_clock_after == clock()
        assert receipt.attached_stage_before == receipt.attached_stage_after == 13
        assert receipt.active_bracket_sequence == first.sequence
        assert 'arrival_window_closed_before_commit' in receipt.errors
        second = p.end_step(clock(12))
        assert second.callback_receipts == ()
        assert receipt.clock_association is None
    finally:
        p.close()


@pytest.mark.parametrize('mode', ['post_step_poll', 'full_report_callback'])
def test_close_winning_during_raw_copy_releases_holder_and_forbids_late_commit(monkeypatch, mode):
    p, source, _ = probe(mode)
    copying, resume = threading.Event(), threading.Event()
    captured, failures = [], []
    original_copy = api().ContactPairProvenanceCollector._capture
    def blocked_copy(collector, *args, **kwargs):
        copying.set()
        assert resume.wait(2), 'fixture failed to resume raw copy'
        report = original_copy(collector, *args, **kwargs)
        captured.append(report)
        return report
    monkeypatch.setattr(api().ContactPairProvenanceCollector, '_capture', blocked_copy)
    callback = source.callback
    def receive():
        try:
            callback(DT) if mode == 'post_step_poll' else callback(*source.raw)
        except BaseException as exc:
            failures.append(exc)
    worker = threading.Thread(target=receive, name='closing-copy')
    try:
        p.begin_step(clock(), dt_s=DT)
        worker.start()
        assert copying.wait(2), 'fixture callback did not reach copying'
        p.close()
        assert source.released == 1 and not p.subscription_active
        resume.set()
        worker.join(2)
        assert not worker.is_alive() and failures == []
        assert len(captured) == 1 and captured[0].headers[0].actor1 == 17
        assert p.records == ()  # No writes after close won the lifecycle race.
        assert len(p.step_diagnostics) == 1
    finally:
        resume.set()
        worker.join(2)
        p.close()


def test_parallel_callbacks_copy_independently_and_use_commit_sequence_identity(monkeypatch):
    p, source, _ = probe('full_report_callback')
    original_copy = api().ContactPairProvenanceCollector._capture
    both_copied = threading.Barrier(3)
    captured, failures = [], []
    def copied(collector, *args, **kwargs):
        report = original_copy(collector, *args, **kwargs)
        captured.append(collector)
        both_copied.wait(2)
        return report
    monkeypatch.setattr(api().ContactPairProvenanceCollector, '_capture', copied)
    def receive(raw):
        try:
            source.callback(*raw)
        except BaseException as exc:
            failures.append(exc)
    raw1, raw2 = raw_report(), raw_report()
    raw2[1][0].impulse[:] = [7., 8., 9.]
    workers = [threading.Thread(target=receive, args=(raw,)) for raw in (raw1, raw2)]
    try:
        p.begin_step(clock(), dt_s=DT)
        for worker in workers:
            worker.start()
        both_copied.wait(2)
        for worker in workers:
            worker.join(2)
        assert failures == [] and not any(worker.is_alive() for worker in workers)
        assert len(captured) == 2 and captured[0] is not captured[1]
        assert len(p.records) == 2
        assert [r.report.sequence for r in p.records] == [r.sequence for r in p.records] == [0, 1]
        assert {r.report.points[0].impulse for r in p.records} == {(.001, -.002, .003), (7., 8., 9.)}
        assert all(r.clock_association is None for r in p.records)
    finally:
        for worker in workers:
            worker.join(2)
        p.close()
