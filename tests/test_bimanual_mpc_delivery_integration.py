"""Pure CPU Probe/Aux integration; only test-owned temporary archives are read."""
import gc
import gzip
import hashlib
import importlib
import json
import sys
import types
from types import SimpleNamespace as NS
import weakref

import pytest


PACKAGE = 'go2_pvcnn.control.dh116'


def api():
    return importlib.import_module(PACKAGE + '.tabletop_bimanual_mpc_aux_capture')


class Token:
    def __init__(self, source):
        self.source = weakref.ref(source)
    def __del__(self):
        source = self.source()
        if source is not None:
            source.events.append('probe_release')
            source.callback = None


class Source:
    def __init__(self, events):
        self.events, self.callback, self.polls = events, None, 0
    def get_full_contact_report(self):
        self.polls += 1
        self.events.append('full_poll')
        return [], [], []
    def get_attached_stage(self):
        return 13
    def subscribe_physics_on_step_events(self, callback, *, pre_step, order):
        assert pre_step is False and type(order) is int
        self.events.append('post_subscribe')
        self.callback = callback
        return Token(self)
    def subscribe_full_contact_report_events(self, callback):
        self.events.append('full_subscribe')
        self.callback = callback
        return Token(self)


class Harness:
    def __init__(self, path, mode='full_report_callback'):
        self.path, self.mode, self.events = path, mode, []
        self.source = Source(self.events)
        self.provider_error = self.advance_error = None
        self.sim = NS(current_time_step_index=18, current_time=18/480)
        owner = self
        class World:
            n, dt, _physics_steps = 2, 1/480, 0
            def __init__(self):
                self.w = NS(sim=owner.sim)
            def advance(self, *args):
                owner.events.append('original_advance')
                if owner.advance_error is not None:
                    raise owner.advance_error
                self._physics_steps += 1
                owner.sim.current_time_step_index += 1
                owner.sim.current_time += self.dt
                owner.emit()
        self.world = World()
        self.original = self.world.advance
        self.prior = lambda *args: self.events.append('prior_observer')
        self.env = NS(n=2, diagnostic_episode_ids=[0, 0],
                      joint_replay_observer=self.prior, task=NS(done=[False, False]))
        self.probe = self.pairs = None

    def emit(self):
        if self.source.callback is not None:
            if self.mode == 'post_step_poll':
                self.source.callback(self.world.dt)
            else:
                self.source.callback([], [], [])

    def components(self, *, clock_reader):
        provenance = importlib.import_module(PACKAGE + '.tabletop_bimanual_mpc_contact_provenance')
        delivery = importlib.import_module(PACKAGE + '.tabletop_bimanual_mpc_contact_delivery_probe')
        owner = self
        class Scene:
            def capture_batch(self, *, clocks, active):
                owner.events.append('scene')
                if owner.provider_error is not None:
                    raise owner.provider_error
                return tuple(NS(clock=c, raw={}, qualification={'native_attested': False}) for c in clocks)
        self.pairs = provenance.ContactPairProvenanceCollector(enabled=True, source=self.source)
        if self.mode is None:
            return Scene(), self.pairs, {'CPU_fixture': True}
        self.probe = delivery.ContactDeliveryProbe(self.source, lambda: clock_reader(0),
                                                   self.mode, enabled=True)
        return Scene(), self.pairs, {'CPU_fixture': True}, self.probe

    def install(self, **options):
        kwargs = dict(enabled=True, run_uuid='delivery-fixture', component_factory=self.components,
                      contact_delivery_probe_mode=self.mode)
        kwargs.update(options)
        self.capture = api().install_mpc_aux_capture(self.env, self.world, self.path, **kwargs)
        return self.capture

    def step(self, terminal=(False, False)):
        self.world.advance(None, None, None)
        self.env.task.done = list(terminal)
        self.env.joint_replay_observer(self.env, {}, [True, True], list(terminal))

    def submit(self):
        self.capture.submit([[0.] * 24] * 2, policy_step=0)


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def test_default_off_preserves_old_three_component_factory_and_creates_no_delivery_archive(tmp_path):
    h = Harness(tmp_path/'capture', mode=None)
    capture = h.install()
    h.submit()
    h.step()
    summary = capture.finish()
    assert h.source.callback is None and h.probe is None
    assert h.source.polls == 1
    assert 'contact_delivery_probe' not in summary
    assert not list(h.path.glob('contact_delivery*'))


@pytest.mark.parametrize('mode', ['post_step_poll', 'full_report_callback'])
def test_probe_brackets_original_advance_then_reuses_existing_poll_with_no_double_poll(tmp_path, mode):
    h = Harness(tmp_path/'capture', mode)
    capture = h.install()
    h.submit()
    h.step(terminal=(True, False))
    diagnostic = h.probe.step_diagnostics[0]
    assert diagnostic.poll_report is h.pairs.records[0]
    assert diagnostic.bracket.before.world_step == 0 and diagnostic.bracket.after.world_step == 1
    assert diagnostic.poll_report.receipt_clock_candidate.terminal
    assert h.source.polls == (2 if mode == 'post_step_poll' else 1)
    assert h.events.count('original_advance') == 1
    summary = capture.finish()
    assert not h.probe.subscription_active and 'probe_release' in h.events
    assert summary['contact_delivery_probe']['callback_record_count'] == 1
    assert not summary['contact_delivery_probe']['native_attested']
    assert capture.terminals['0']['contact_delivery_step']['clock_association'] is None


def test_orphan_and_late_callbacks_are_flushed_from_independent_records_after_close(tmp_path):
    h = Harness(tmp_path/'capture')
    capture = h.install()
    h.emit()  # Before any active advance bracket.
    h.submit()
    h.step(terminal=(True, False))
    h.emit()  # After end_step; absent from its immutable per-step diagnostic.
    assert len(h.probe.records) == 3
    assert len(h.probe.step_diagnostics[0].callback_receipts) == 1
    retained_callback = h.source.callback
    summary = capture.finish()
    delivery = summary['contact_delivery_probe']
    assert delivery['callback_record_count'] == 3 and delivery['step_diagnostic_count'] == 1
    callback_path = h.path/'contact_delivery_callbacks.jsonl.gz'
    assert delivery['callback_raw']['sha256'] == hashlib.sha256(callback_path.read_bytes()).hexdigest()
    archive = rows(callback_path)
    assert [row['receipt']['sequence'] for row in archive] == [0, 1, 2]
    assert [row['receipt']['active_bracket_sequence'] for row in archive] == [None, 0, None]
    assert all(row['receipt']['clock_association'] is None for row in archive)
    assert all(row['receipt']['pair_env_association'] is None for row in archive)
    assert all(not row['native_success_qualification_issued'] for row in archive)
    retained_callback([], [], [])
    assert len(h.probe.records) == 3
    saved = json.loads((h.path/'contact_delivery_summary.json').read_text())
    assert saved['callback_record_count'] == 3
    assert h.world.advance == h.original and h.env.joint_replay_observer is h.prior


@pytest.mark.parametrize('failure', ['advance', 'provider', 'write'])
def test_failures_immediately_release_probe_restore_hooks_and_preserve_original_error(tmp_path, failure):
    h = Harness(tmp_path/'capture')
    capture = h.install()
    h.submit()
    original = RuntimeError(f'original {failure} failure')
    if failure == 'advance':
        h.advance_error = original
    elif failure == 'provider':
        h.provider_error = original
    else:
        stream = capture._stream
        class BrokenWrite:
            def write(self, value):
                raise original
            def close(self):
                stream.close()
        capture._stream = BrokenWrite()
    with pytest.raises(RuntimeError) as caught:
        h.step(terminal=(True, False))
    assert caught.value is original
    assert not h.probe.subscription_active
    assert h.world.advance == h.original and h.env.joint_replay_observer is h.prior
    summary = capture.finish()
    assert summary['terminal_count'] == (0 if failure == 'advance' else 1)
    assert failure in summary['error']
    assert summary['contact_delivery_probe']['callback_record_count'] == (0 if failure == 'advance' else 1)


def test_typed_components_are_explicit_and_mode_mismatch_still_releases_resource(tmp_path):
    m = api()
    assert hasattr(m, 'AuxCaptureComponents'), 'typed optional observer components are missing'
    h = Harness(tmp_path/'capture')
    def factory(**kwargs):
        scene, pairs, metadata, probe = h.components(**kwargs)
        return m.AuxCaptureComponents(scene, pairs, metadata, probe)
    with pytest.raises(ValueError, match='mode'):
        h.install(component_factory=factory, contact_delivery_probe_mode='post_step_poll')
    gc.collect()
    assert not h.probe.subscription_active
    assert h.world.advance == h.original and h.env.joint_replay_observer is h.prior


@pytest.mark.parametrize('failure', ['probe_close', 'callback_write', 'step_write', 'raw_close'])
def test_finish_failures_release_before_flushing_and_keep_first_terminal(tmp_path, monkeypatch, failure):
    h = Harness(tmp_path/'capture')
    capture = h.install()
    h.submit()
    h.step(terminal=(True, False))
    original = OSError(f'fixture {failure}')
    if failure == 'probe_close':
        close = h.probe.close
        def broken_close():
            close()
            raise original
        monkeypatch.setattr(h.probe, 'close', broken_close)
    elif failure == 'raw_close':
        stream = capture._stream
        class BrokenClose:
            def close(self):
                stream.close()
                raise original
        capture._stream = BrokenClose()
    original_open = api().gzip.open
    def opening(path, *args, **kwargs):
        assert not h.probe.subscription_active, 'release must precede delivery sink creation'
        if failure in ('callback_write', 'step_write'):
            broken_name = 'callbacks' if failure == 'callback_write' else 'steps'
            if broken_name in str(path):
                raise original
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(api().gzip, 'open', opening)
    with pytest.raises(OSError) as caught:
        capture.finish()
    assert caught.value is original
    assert not h.probe.subscription_active
    assert h.world.advance == h.original and h.env.joint_replay_observer is h.prior
    first = json.loads((h.path/'terminal_snapshots.json').read_text())['0']
    assert first['clock']['world_step'] == 1
    summary = json.loads((h.path/'aux_capture_summary.json').read_text())
    assert summary['terminal_count'] == 1 and failure in summary['error']
    assert summary['contact_delivery_probe']['callback_record_count'] == 1
    assert capture.finish() == summary


def test_prior_observer_failure_closes_owned_probe_without_replacing_original_error(tmp_path):
    h = Harness(tmp_path/'capture')
    original = RuntimeError('prior observer failed')
    def broken_prior(*args):
        raise original
    h.prior = h.env.joint_replay_observer = broken_prior
    capture = h.install()
    h.submit()
    with pytest.raises(RuntimeError) as caught:
        h.step()
    assert caught.value is original
    assert not h.probe.subscription_active
    assert h.world.advance == h.original and h.env.joint_replay_observer is broken_prior
    assert 'prior observer failed' in capture.finish()['error']


def test_first_causal_provider_error_survives_simultaneous_partial_write_and_close_errors(tmp_path, monkeypatch):
    h = Harness(tmp_path/'capture')
    capture = h.install()
    h.submit()
    h.provider_error = RuntimeError('first provider error')
    stream = capture._stream
    class BrokenSink:
        def write(self, value):
            raise OSError('secondary partial write error')
        def close(self):
            stream.close()
            raise OSError('tertiary raw close error')
    capture._stream = BrokenSink()
    with pytest.raises(RuntimeError) as caught:
        h.step(terminal=(True, False))
    assert caught.value is h.provider_error
    assert not h.probe.subscription_active
    summary = capture.finish()
    assert summary['terminal_count'] == 1
    assert 'first provider error' in summary['error'] and 'tertiary raw close error' in summary['error']
    first = json.loads((h.path/'terminal_snapshots.json').read_text())['0']
    assert first['contact_report']['receipt_clock_candidate']['terminal']


@pytest.mark.parametrize('mode', ['post_step_poll', 'full_report_callback'])
@pytest.mark.parametrize('identity_failure', [False, True])
def test_native_component_wiring_reuses_existing_source_and_explicit_candidate_clock_without_sdk(tmp_path, monkeypatch, mode, identity_failure):
    """Only import facades exercise wiring; this cannot attest any Native source."""
    stage, simulation_view, rigid_view, source = (object() for _ in range(4))
    class Probe:
        closed = False
        def close(self):
            self.closed = True
    probe = Probe()
    events = []
    class Sensor:
        __module__ = 'isaaclab.sensors.cpu_fixture'
        _physics_sim_view = simulation_view
    class Scene:
        def __init__(self, **kwargs):
            assert kwargs['stage'] is stage
    class Provider:
        def __init__(self, **kwargs):
            assert kwargs['rigid_body_view'] is rigid_view
    class Pairs:
        def __init__(self, *, enabled, source):
            assert enabled is True
            self.source = source
    def module(name, **attributes):
        result = types.ModuleType(name)
        vars(result).update(attributes)
        monkeypatch.setitem(sys.modules, name, result)
        return result
    carb = module('carb', settings=NS(get_settings=lambda: object()))
    omni = module('omni')
    omni.__path__ = []
    omni.usd = module('omni.usd', get_context=lambda: NS(get_stage=lambda: stage))
    module(PACKAGE + '.tabletop_bimanual_mpc_scene_measurements',
           ComposedUSDScene=Scene, SceneMeasurementProvider=Provider,
           create_passive_contact_views=lambda **kwargs: object())
    def opened_source(**kwargs):
        events.append(('existing_source', kwargs))
        return source
    module(PACKAGE + '.tabletop_bimanual_mpc_contact_provenance',
           ContactPairProvenanceCollector=Pairs, open_installed_contact_source=opened_source)
    candidate_clock = object()
    def opened_probe(**kwargs):
        events.append(('probe', kwargs))
        assert kwargs['source'] is source
        assert kwargs['clock_reader']() is candidate_clock
        return probe
    module(PACKAGE + '.tabletop_bimanual_mpc_contact_delivery_probe',
           open_installed_delivery_probe=opened_probe)
    world = NS(w=NS(sensors=[Sensor()]), wire=NS(root_physx_view=rigid_view), dt=1/480)
    original_identity, identities = api()._identity, []
    original = OSError('probe identity read failed')
    def identity(path):
        identities.append(path)
        if identity_failure and len(identities) == 3:
            raise original
        return original_identity(path)
    monkeypatch.setattr(api(), '_identity', identity)
    def build():
        return api()._native_components(NS(), world, clock_reader=lambda env: object(),
            capacity_per_wire=64, contact_delivery_probe_mode=mode,
            contact_delivery_probe_order=7, delivery_clock_reader=lambda: candidate_clock)
    if identity_failure:
        with pytest.raises(OSError) as caught:
            build()
        assert caught.value is original and probe.closed
        return
    result = build()
    assert result.pairs.source is source and result.delivery_probe is probe
    assert len(events) == 2 and events[1][1]['mode'] == mode
    assert result.metadata['delivery_probe_implementation']['sha256']


def test_native_attestation_rejects_cpu_factory_before_any_source_or_sdk_access(tmp_path, monkeypatch):
    h = Harness(tmp_path/'capture')
    def forbidden(*args, **kwargs):
        raise AssertionError('DI cannot enter Native attestation/source acquisition')
    monkeypatch.setattr(api(), '_native_attestation', forbidden)
    with pytest.raises(ValueError, match='injected|fixture'):
        h.install(native_runtime_verified=True)
    assert not h.path.exists() and h.probe is None


def test_no_active_sample_ends_probe_with_actual_uncompleted_receipt_without_another_poll(tmp_path):
    h = Harness(tmp_path/'capture')
    capture = h.install()
    h.submit()
    h.world.advance(None, None, None)
    h.env.joint_replay_observer(h.env, {}, [False, False], [False, False])
    assert h.source.polls == 0
    assert len(h.probe.step_diagnostics) == 1
    diagnostic = h.probe.step_diagnostics[0]
    assert diagnostic.bracket.after.world_step == 1 and not diagnostic.bracket.consistent
    assert 'step_not_completed' in diagnostic.errors
    assert capture.finish()['physical_frames'] == 0


def install_core_fixture(h, path, *, install_error=None):
    """Real Core installer/archive, with only its measurement producer on CPU."""
    core = importlib.import_module(PACKAGE + '.tabletop_bimanual_mpc_capture')
    state = importlib.import_module(PACKAGE + '.tabletop_bimanual_mpc_state')
    class Producer:
        def __init__(self, world, *, run_uuid):
            if install_error is not None:
                raise install_error
            self.world, self.run_uuid = world, run_uuid
            self.last_evidence = {'CPU_fixture': True, 'native_attested': False}
        def capture(self, **context):
            snapshots = []
            for i in range(h.env.n):
                clock = state.SampleClock(run_uuid=self.run_uuid, env_id=i,
                    episode_uid=context['episode_uids'][i],
                    world_step=self.world._physics_steps, world_time_s=self.world._physics_steps/480,
                    sdk_step=self.world.w.sim.current_time_step_index,
                    sdk_time_s=self.world.w.sim.current_time,
                    policy_step=context['policy_step'], reset_generation=context['reset_generations'][i],
                    terminal=context['terminal'][i])
                snapshots.append(state.make_mpc_state(clock, state.missing_fields(clock,
                    reason='CPU fixture has no Native measurement', source='CPU fixture')))
            return tuple(snapshots)
    return core.install_mpc_state_capture(h.env, h.world, path, enabled=True,
        run_uuid='delivery-fixture', native_runtime_verified=False, bridge_factory=Producer)


@pytest.mark.parametrize('mode', ['post_step_poll', 'full_report_callback'])
@pytest.mark.parametrize('failure', ['core_install', 'provider', 'raw_write', 'prior', 'provider_core_finish'])
def test_real_core_aux_failure_unwinds_hooks_lifo_without_repeating_finalization(tmp_path, monkeypatch, mode, failure):
    h = Harness(tmp_path/'aux', mode)
    original = RuntimeError('first '+failure+' failure')
    if failure == 'prior':
        def prior(*args):
            raise original
        h.prior = h.env.joint_replay_observer = prior
    aux = h.install()
    if failure == 'core_install':
        with pytest.raises(RuntimeError) as caught:
            install_core_fixture(h, tmp_path/'core', install_error=original)
        assert caught.value is original
        aux.finish(error=repr(original))
        assert not h.probe.subscription_active
        assert h.env.joint_replay_observer is h.prior
        assert h.world.advance == h.original
        return
    core = install_core_fixture(h, tmp_path/'core')
    h.submit()
    core.submit([[0.] * 24] * 2, policy_step=0)
    if failure in ('provider', 'provider_core_finish'):
        h.provider_error = original
    elif failure == 'raw_write':
        stream = aux._stream
        class BrokenWrite:
            def write(self, value):
                raise original
            def close(self):
                stream.close()
        aux._stream = BrokenWrite()
    with pytest.raises(RuntimeError) as caught:
        h.step(terminal=(True, False))
    assert caught.value is original
    assert aux._finished and not h.probe.subscription_active
    assert h.env.joint_replay_observer is core._observer
    assert len(aux.terminals) == (0 if failure == 'prior' else 1)
    saved = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in h.path.iterdir()}
    def forbidden(*args, **kwargs):
        raise AssertionError('finished Aux must not close/archive/write twice')
    monkeypatch.setattr(aux, '_write_summary', forbidden)
    monkeypatch.setattr(aux, '_flush_delivery', forbidden)
    monkeypatch.setattr(h.probe, 'close', forbidden)
    aux.finish()  # Outer Core still owns the hook; leave it alone.
    assert h.env.joint_replay_observer is core._observer
    if failure == 'provider_core_finish':
        stream = core._raw
        class BrokenCoreClose:
            def close(self):
                stream.close()
                raise OSError('secondary Core finish failure')
        core._raw = BrokenCoreClose()
        with pytest.raises(OSError, match='secondary Core finish failure'):
            core.finish(error=repr(original))
    else:
        core.finish(error=repr(original))
    # Real outer LIFO cleanup has now restored Aux. Its already-written
    # archives stay immutable, but its own hook still needs to be unwound.
    assert h.env.joint_replay_observer is aux._observer_hook
    summary = aux.finish(error=repr(original))
    assert h.env.joint_replay_observer is h.prior
    assert h.world.advance == h.original
    assert summary is aux._summary
    assert saved == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in h.path.iterdir()}


def test_real_core_aux_default_off_success_keeps_original_lifo_cleanup(tmp_path):
    h = Harness(tmp_path/'aux', mode=None)
    aux = h.install()
    core = install_core_fixture(h, tmp_path/'core')
    h.submit()
    core.submit([[0.] * 24] * 2, policy_step=0)
    h.step(terminal=(True, False))
    core.finish()
    summary = aux.finish()
    assert h.env.joint_replay_observer is h.prior and h.world.advance == h.original
    assert summary['terminal_count'] == 1 and 'contact_delivery_probe' not in summary
    assert h.probe is None and not list(h.path.glob('contact_delivery*'))


def test_finished_aux_leaves_foreign_hook_untouched_until_outer_owner_restores_it(tmp_path):
    h = Harness(tmp_path/'aux')
    aux = h.install()
    core = install_core_fixture(h, tmp_path/'core')
    h.submit()
    core.submit([[0.] * 24] * 2, policy_step=0)
    h.provider_error = RuntimeError('provider failed')
    with pytest.raises(RuntimeError, match='provider failed'):
        h.step(terminal=(True, False))
    foreign = lambda *args: None
    h.env.joint_replay_observer = foreign
    aux.finish()
    assert h.env.joint_replay_observer is foreign
    # Simulate that foreign owner restoring the Core hook it previously saved.
    h.env.joint_replay_observer = core._observer
    core.finish(error='provider failed')
    aux.finish()
    assert h.env.joint_replay_observer is h.prior


def test_real_core_model_aux_failure_unwinds_each_existing_owner(tmp_path):
    h = Harness(tmp_path/'aux')
    aux = h.install()
    model_api = importlib.import_module(PACKAGE + '.tabletop_bimanual_mpc_model_capture')
    def provider_factory(*, clock_reader):
        def capture_batch(**kwargs):
            raise AssertionError('Aux failure must propagate before Model getters')
        return NS(capture_batch=capture_batch), {'CPU_fixture': True}
    model = model_api.install_mpc_model_capture(h.env, h.world, tmp_path/'model', enabled=True,
        run_uuid='delivery-fixture', provider_factory=provider_factory)
    core = install_core_fixture(h, tmp_path/'core')
    h.submit()
    model.submit([[0.] * 24] * 2, policy_step=0)
    core.submit([[0.] * 24] * 2, policy_step=0)
    h.provider_error = RuntimeError('Aux provider failed under Model and Core')
    with pytest.raises(RuntimeError) as caught:
        h.step(terminal=(True, False))
    assert caught.value is h.provider_error and aux._finished
    core.finish(error=repr(h.provider_error))
    assert h.env.joint_replay_observer is model._observer_hook
    model.finish(error=repr(h.provider_error))
    assert h.env.joint_replay_observer is aux._observer_hook
    aux.finish()
    assert h.env.joint_replay_observer is h.prior and h.world.advance == h.original
