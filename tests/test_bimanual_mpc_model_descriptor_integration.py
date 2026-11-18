"""Synthetic descriptor through the original Model observer; never Native."""
from dataclasses import replace
import inspect
import json
import sys
import types
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

import test_bimanual_mpc_model_capture as model_fixture
from test_bimanual_mpc_model_descriptor import fixture as descriptor_fixture
from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_articulation_measurements import ArticulationMeasurementProvider
api=model_fixture.api


@pytest.fixture
def h():
    owner=model_fixture.ModelCaptureTests();owner.setUp()
    try:yield owner
    finally:owner.doCleanups()


def install(h):
    f=descriptor_fixture();h.descriptor_fixture=f;h.inputs=[]
    def model_factory(*,clock_reader):
        provider=ArticulationMeasurementProvider(hands=f.hands,clock_reader=clock_reader,source_reader=f.old.source_reader,enabled=True)
        h.articulation_provider=provider
        return provider,{'source_role':'CPU full17 fixture only'}
    def descriptor_factory(*,clock_reader,articulation_provider):
        assert articulation_provider is h.articulation_provider
        f.p.clock_reader=clock_reader
        original=f.p.capture_batch
        def capture(**kwargs):
            h.events.append('descriptor');h.inputs.append(kwargs);return original(**kwargs)
        f.p.capture_batch=capture;h.descriptor=f.p
        return f.p,{'descriptor_provider_source':{'path':inspect.getfile(type(f.p)),'sha256':'CPU-fixture-only'}}
    return h.install(provider_factory=model_factory,capture_model_descriptor=True,descriptor_provider_factory=descriptor_factory)


def test_real_full_d_samples_are_reused_after_prior_without_extra_physics_or_qualification(h):
    capture=install(h);h.submit(capture);h.physics();summary=capture.finish();row=h.rows()[0]
    assert h.events==['advance','prior','descriptor']
    assert h.world.advance==h.original_advance and h.world.refresh==h.original_refresh
    assert h.inputs[0]['articulation_samples'][0].clock.world_step==1
    assert row['descriptor_samples'][0]['collection_valid'] is True
    assert row['descriptor_samples'][0]['descriptor']['hands']['left']['measurements']['q'][0]==100.
    assert row['descriptor_samples'][0]['clock']==row['model_samples'][0]['clock']==row['clocks'][0]
    assert row['issued_action24'][0][17]==-.25
    assert all(v is False for v in row['descriptor_samples'][0]['qualification'].values())
    assert summary['descriptor_capture']['snapshots']==1 and summary['descriptor_capture']['collection_valid_samples']==2
    assert summary['physical_frames']==1 and not summary['planning_eligible']


def test_default_model_path_does_not_construct_or_archive_descriptor(h):
    with patch.object(api(),'_native_descriptor_provider',side_effect=AssertionError('off descriptor factory read')):
        capture=h.install();h.submit(capture);h.physics();summary=capture.finish()
    row=h.rows()[0]
    assert 'descriptor_samples' not in row and 'descriptor_capture' not in summary
    assert h.events==['advance','prior','capture']


def test_honest_descriptor_na_is_archived_as_observation_not_complete_dynamics(h):
    capture=install(h);h.descriptor_fixture.hands['left'].root_physx_view.get_jacobians=lambda:(_ for _ in ()).throw(RuntimeError('unsupported current J'))
    h.submit(capture);h.physics();summary=capture.finish();row=h.rows()[0]
    assert row['complete_measurement'] is True and summary['physical_frames']==1
    assert all(not s['collection_valid'] and s['errors'] for s in row['descriptor_samples'])
    assert summary['descriptor_capture']['collection_valid_samples']==0
    assert summary['descriptor_capture']['collection_unavailable_samples']==2
    assert row['descriptor_samples'][0]['raw']['hands']['left']['readbacks']['get_jacobians']['available'] is False


def test_descriptor_clock_torn_preserves_same_transaction_raw_model_and_first_terminal(h):
    capture=install(h);view=h.descriptor_fixture.hands['left'].root_physx_view;original=view.get_link_transforms
    def torn():
        value=original();h.world.w.sim.current_time_step_index+=1;return value
    view.get_link_transforms=torn;h.submit(capture)
    with pytest.raises(RuntimeError,match='clock/episode') as error:h.physics(terminal=(True,False))
    summary=capture.finish();row=h.rows()[0];first=json.loads((h.output/'terminal_snapshots.json').read_text())['0']
    assert row['capture_status']=='descriptor_error' and row['complete_measurement'] is False
    assert row['model_samples'][0]['clock']['sdk_step']==19
    assert row['descriptor_partial_raw']['raw']['owned_reads']['left.get_link_transforms']['value'][0][0][6]==1.
    assert row['descriptor_partial_raw']['clocks']==row['clocks']
    assert row['descriptor_error']==repr(error.value)
    assert first['clocks'][0]['world_step']==1 and not first['terminal_measurement_valid']
    assert summary['physical_frames']==0 and summary['raw_records']==1 and summary['terminal_count']==1
    assert summary['descriptor_capture']['error_frames']==1
    assert h.env.joint_replay_observer is h.prior


def test_returned_descriptor_wrong_clock_keeps_returned_owned_dto_not_old_last_samples(h):
    capture=install(h);original=h.descriptor.capture_batch
    def corrupt(**kwargs):
        samples=original(**kwargs);return (replace(samples[0],clock=replace(samples[0].clock,sdk_step=999)),)+samples[1:]
    h.descriptor.capture_batch=corrupt;h.submit(capture)
    with pytest.raises(ValueError,match='descriptor.*clock'):h.physics(terminal=(True,False))
    capture.finish();row=h.rows()[0]
    assert row['descriptor_samples'] is None and row['descriptor_partial_raw'][0]['clock']['sdk_step']==999
    assert row['model_samples'][0]['clock']['sdk_step']==19


def test_original_model_failure_stays_original_and_descriptor_is_never_called(h):
    capture=install(h);failure=RuntimeError('original model failure')
    h.articulation_provider.capture_batch=lambda **kwargs:(_ for _ in ()).throw(failure)
    h.submit(capture)
    with pytest.raises(RuntimeError) as caught:h.physics(terminal=(True,False))
    assert caught.value is failure and not h.inputs
    summary=capture.finish();row=h.rows()[0]
    assert row['capture_status']=='provider_error' and row['descriptor_samples'] is None
    assert row['descriptor_error'] is None and summary['descriptor_capture']['snapshots']==0


def test_upstream_observer_failure_pins_terminal_before_any_model_or_descriptor_getter(h):
    failure=RuntimeError('original Aux observer failed')
    def prior(*args):raise failure
    h.env.joint_replay_observer=prior
    capture=install(h);h.submit(capture)
    with pytest.raises(RuntimeError) as caught:h.physics(terminal=(True,False))
    assert caught.value is failure and not h.inputs
    assert all(not hand.root_physx_view.calls for hand in h.descriptor_fixture.hands.values())
    summary=capture.finish();row=h.rows()[0]
    assert row['capture_status']=='prior_observer_error' and row['descriptor_samples'] is None
    assert summary['terminal_count']==1 and summary['physical_frames']==0


def test_descriptor_terminal_closed_slot_does_not_get_new_episode_credit(h):
    capture=install(h);h.submit(capture);h.physics(terminal=(True,False));h.env.diagnostic_episode_ids[0]=1
    h.physics(terminal=(False,True));h.env.diagnostic_episode_ids[1]=1;h.physics()
    summary=capture.finish();rows=h.rows()
    assert len(rows)==2 and summary['records_per_env']==[1,2]
    assert rows[1]['descriptor_samples'][0]['collection_valid'] is False
    assert rows[1]['descriptor_samples'][0]['errors']==['inactive slot']
    assert summary['descriptor_capture']['collection_valid_samples']==3


def test_strict_descriptor_flags_and_cpu_native_factory_boundary_before_files(h):
    for change in ({'capture_model_descriptor':1},{'capture_model_descriptor':True},
            {'descriptor_provider_factory':object()}, {'capture_model_descriptor':True,'descriptor_provider_factory':object(),'native_runtime_verified':True}):
        with patch.object(api(),'_native_attestation'),pytest.raises(ValueError):h.install(**change)
        assert not h.output.exists()


def test_descriptor_error_is_not_replaced_by_raw_sink_error_and_terminal_is_retained(h):
    capture=install(h);failure=RuntimeError('descriptor source failed')
    def fail(**kwargs):h.descriptor.last_partial={'current':'partial','complete_measurement':False};raise failure
    h.descriptor.capture_batch=fail
    original=capture._stream
    class Sink:
        def write(self,text):raise OSError('raw failed')
        def close(self):original.close()
    capture._stream=Sink();h.submit(capture)
    with pytest.raises(RuntimeError) as caught:h.physics(terminal=(True,False))
    assert caught.value is failure
    summary=capture.finish();assert summary['terminal_count']==1 and summary['raw_records']==0
    first=json.loads((h.output/'terminal_snapshots.json').read_text())['0']
    assert first['descriptor_partial_raw']['current']=='partial' and not first['terminal_measurement_valid']


def test_descriptor_qualification_forgery_retains_returned_raw_and_zero_complete_frames(h):
    capture=install(h);original=h.descriptor.capture_batch
    def corrupt(**kwargs):
        samples=original(**kwargs)
        return (replace(samples[0],qualification={'planning_eligible':True}),)+samples[1:]
    h.descriptor.capture_batch=corrupt;h.submit(capture)
    with pytest.raises(ValueError,match='qualification'):h.physics(terminal=(True,False))
    summary=capture.finish();row=h.rows()[0]
    assert row['descriptor_partial_raw'][0]['qualification']['planning_eligible'] is True
    assert not row['planning_eligible'] and summary['physical_frames']==0


def test_a_prior_descriptor_partial_cannot_be_reused_for_a_new_failure(h):
    capture=install(h);h.descriptor.last_partial={'old':'forbidden'}
    failure=RuntimeError('failed before exposing any owned descriptor')
    h.descriptor.capture_batch=lambda **kwargs:(_ for _ in ()).throw(failure)
    h.submit(capture)
    with pytest.raises(RuntimeError) as caught:h.physics()
    assert caught.value is failure
    capture.finish();row=h.rows()[0]
    assert row['descriptor_partial_raw'] is None and row['descriptor_partial_unavailable_reason']
    assert row['model_samples'] is not None


def test_native_descriptor_factory_only_reuses_existing_simulation_view_and_source(h):
    f=descriptor_fixture();stage=f.stage;calls=[]
    class Scene:
        def get_gravity(self):raise AssertionError('factory cannot call gravity getter')
    scene=Scene();h.world.hands=[f.hands['left'],f.hands['right']]
    h.world.w.sim.physics_sim_view=scene
    omni=types.ModuleType('omni');usd=types.ModuleType('omni.usd')
    usd.get_context=lambda:NS(get_stage=lambda:stage);omni.usd=usd
    with patch.dict(sys.modules,{'omni':omni,'omni.usd':usd}):
        provider,metadata=api()._native_descriptor_provider(h.env,h.world,clock_reader=lambda i:None,articulation_provider=f.old)
    assert provider.stage is stage and provider.scene_view is scene
    assert provider.hands==f.hands and provider.source_reader is f.old.source_reader
    assert metadata['scene_view_python_identity']==id(scene)
    assert metadata['descriptor_provider_source']['path'].endswith('tabletop_bimanual_mpc_model_descriptor.py')
    assert provider.stage_reader() is stage
    h.world.w.sim.physics_sim_view=Scene()
    with pytest.raises(RuntimeError,match='scene view'):provider.stage_reader()
    h.world.w.sim.physics_sim_view=scene
    h.world.hands=[NS(root_physx_view=f.hands['left'].root_physx_view),f.hands['right']]
    with pytest.raises(RuntimeError,match='hand'):provider.stage_reader()
