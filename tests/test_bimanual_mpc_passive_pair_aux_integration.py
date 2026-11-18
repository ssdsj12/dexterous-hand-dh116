"""CPU actual Scene tap in auxiliary hooks; no SDK/Native/support qualification."""
from copy import deepcopy
import importlib
import json

import pytest

import test_bimanual_mpc_aux_capture as A
from test_bimanual_mpc_scene_measurements import Scene,Rigid,Simulation


@pytest.fixture
def h():
    owner=A.AuxCaptureTests();owner.setUp()
    try:yield owner
    finally:owner.doCleanups()


def factory(h,enabled=True):
    def components(*,clock_reader):
        _,pairs,metadata=h.components(clock_reader=clock_reader)
        prefix=A.api().__package__
        S=importlib.import_module(prefix+'.tabletop_bimanual_mpc_scene_measurements')
        P=importlib.import_module(prefix+'.tabletop_bimanual_mpc_passive_pair_values')
        scene=Scene();rigid=Rigid();sim=Simulation()
        bundle=S.create_passive_contact_views(scene=scene,simulation_view=sim,rigid_body_view=rigid,enabled=True,capacity_per_wire=8)
        source=dict(stage_identity='fixture-stage',simulation_view_python_identity=id(sim),rigid_view_python_identity=id(rigid),
            contact_view_python_identities={s:id(v) for s,v in bundle.views.items()},wire_body_paths=rigid.prim_paths,
            api_sources={s:dict(path='/CPU/api.py',sha256='a'*64,python_type='CPU.Contact') for s in S.SURFACES})
        units=dict(stage_identity='fixture-stage',meters_per_unit=1.,kilograms_per_unit=1.,source='CPU actual stage units')
        observer=P.PassivePairObserver(bundle=bundle,clock_reader=lambda:[clock_reader(e) for e in range(2)],
            source_reader=lambda:deepcopy(source),units_reader=lambda:deepcopy(units),enabled=enabled)
        h.observer=observer;h.views=bundle.views;h.passive_source=source
        provider=S.SceneMeasurementProvider(scene=scene,rigid_body_view=rigid,clock_reader=clock_reader,
            contact_views=bundle,enabled=True,passive_pair_observer=observer)
        return provider,pairs,metadata
    return components


def test_aux_explicit_passive_pair_batch_keeps_actual_action_clocks_and_normal_before_friction(h):
    capture=h.install(component_factory=factory(h),capture_passive_pair_values=True)
    h.submit(capture);h.physics();summary=capture.finish();row=h.rows()[0]
    passive=row['passive_pair_values']
    assert passive['capture_valid'] is True and passive['clocks']==row['clocks']
    assert row['issued_action24'][0][17]==-.25 and len(passive['pairs'])==6
    assert all(not item['fields']['support.table_reaction']['valid'] for item in row['scene_samples'])
    assert not any(passive['qualification'].values())
    assert all(v.calls==[('matrix',1/480),('normal',1/480),('friction',1/480)] for v in h.views.values())
    assert summary['passive_pair_values']['snapshots']==1
    assert summary['passive_pair_values']['active_normal_valid']==6
    assert summary['passive_pair_values']['active_friction_valid']==6


def test_aux_default_off_does_not_accept_hidden_passive_observer(h):
    with pytest.raises(ValueError,match='explicit|passive'):h.install(component_factory=factory(h))
    assert h.env.joint_replay_observer is h.prior and h.world.advance==h.original


def test_aux_flag_requires_real_enabled_sidecar_and_strict_bool_before_hooks(h):
    with pytest.raises(ValueError,match='passive|bool'):h.install(capture_passive_pair_values=1)
    assert not h.output.exists()
    with pytest.raises(ValueError,match='passive'):h.install(capture_passive_pair_values=True)


def test_aux_passive_terminal_survives_retired_generation_while_other_env_continues(h):
    capture=h.install(component_factory=factory(h),capture_passive_pair_values=True)
    h.submit(capture);h.physics(terminal=(True,False));h.env.diagnostic_episode_ids[0]=1
    h.physics(terminal=(False,False));summary=capture.finish();rows=h.rows()
    first=json.loads((h.output/'terminal_snapshots.json').read_text())['0']
    assert first['passive_pair_values']['clocks'][0]['episode_uid']=='aux-fixture:0:0'
    assert rows[1]['passive_pair_values']['clocks'][0]['episode_uid']=='aux-fixture:0:1'
    assert all(not p['normal_sensor_side_world']['valid'] for p in rows[1]['passive_pair_values']['pairs'] if p['clock']['env_id']==0)
    assert summary['records_per_env']==[1,2] and summary['terminal_count']==1


def test_passive_source_failure_keeps_fresh_partial_and_original_terminal_before_abort(h):
    capture=h.install(component_factory=factory(h),capture_passive_pair_values=True)
    h.submit(capture);original=h.views['positive_y'].get_friction_data
    def drift(dt):
        value=original(dt);h.passive_source['stage_identity']='changed';return value
    h.views['positive_y'].get_friction_data=drift
    with pytest.raises(ValueError,match='passive'):h.physics(terminal=(True,False))
    capture.finish();row=h.rows()[0]
    assert row['passive_pair_values'] is None
    assert row['provider_partial_passive_pair_values']['raw']['views']['table']['getters']['normal']['available']
    assert row['provider_partial_passive_pair_values']['capture_valid'] is False
    first=json.loads((h.output/'terminal_snapshots.json').read_text())['0']
    assert first['clocks'][0]['terminal'] and first['complete_measurement'] is False


def test_passive_begin_failure_keeps_only_current_transaction_partial_not_old_snapshot(h):
    capture=h.install(component_factory=factory(h),capture_passive_pair_values=True)
    h.submit(capture);h.physics();h.passive_source['stage_identity']='changed-before-next-frame'
    with pytest.raises(ValueError,match='source'):h.physics(terminal=(True,False))
    capture.finish();row=h.rows()[1]
    assert row['passive_pair_values'] is None and row['provider_partial_passive_pair_values'] is None
    raw=row['provider_partial_passive_pair_raw']
    assert raw['clocks'][0]['world_step']==2 and raw['clocks'][0]['terminal']
    assert raw['raw']['views']=={} and raw['complete_measurement'] is False


def test_passive_raw_write_error_retains_terminal_and_is_listed_without_execution_credit(h):
    capture=h.install(component_factory=factory(h),capture_passive_pair_values=True);original=capture._stream
    failure=OSError('synthetic passive raw sink failed')
    class Sink:
        def write(self,text):raise failure
    capture._stream=Sink();h.submit(capture)
    with pytest.raises(OSError) as caught:h.physics(terminal=(True,False))
    assert caught.value is failure
    capture._stream=original;summary=capture.finish()
    assert summary['passive_pair_values']['archive_errors']
    first=json.loads((h.output/'terminal_snapshots.json').read_text())['0']
    assert first['passive_pair_values']['clocks'][0]['terminal']
    assert summary['terminal_archive_only_env_ids']==[0] and summary['physical_frames']==0
