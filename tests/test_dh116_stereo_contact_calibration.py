"""Numerical fixtures check admission logic, never qualify native contact."""
import copy
import importlib

import numpy as np
import pytest


def api():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_stereo_contact_calibration')


class View:
    sensor_count = filter_count = 1
    sensor_paths = ['/World/Object']

    def __init__(self):
        self.count = np.array([[1]], dtype=np.int32)
        self.start = np.array([[0]], dtype=np.int32)

    def get_contact_data(self, dt):
        self.count[:] = 1
        return (np.array([[.002 / dt]]), np.array([[0., 0., .01]]),
                np.array([[0., 0., 1.]]), np.array([[-.0001]]), self.count, self.start)

    def get_friction_data(self, dt):
        # Real API reuses count/start buffers. Previous normal snapshot must survive.
        self.count[:] = 0
        return np.array([[0., 0., 0.]]), np.zeros((1, 3)), self.count, self.start


def test_same_step_dt_read_copies_buffers_and_preserves_normal_ownership():
    value = api().capture_tensor_pair(View(), dt=1 / 480)
    assert value['normal']['counts'] == [[1]]
    assert value['friction']['counts'] == [[0]]
    assert value['dt_scaling_max_error_n'] < 1e-12
    assert value['normal']['impulse_n_s'] == [[.002]]
    assert value['qualified_on_object'] is False


def test_capacity_overrun_refuses_sliced_fake_contacts():
    view = View()
    view.start[:] = 1
    with pytest.raises(ValueError, match='range'):
        api().capture_tensor_pair(view, dt=1 / 480)


def test_dt_is_physical_positive_finite():
    for dt in (0., -1., float('nan')):
        with pytest.raises(ValueError, match='dt'):
            api().capture_tensor_pair(View(), dt=dt)


def fixture_rows():
    return [dict(physics_step=i, timestamp=i/480, dt=1/480,
                 velocity_before=[0., 0., 0.], velocity_after=[0., 0., 0.],
                 mass_kg=.1, gravity_m_s2=[0., 0., -9.81],
                 tensor=api().capture_tensor_pair(View(), dt=1/480),
                 pair_reports=[]) for i in range(1, 5)]


def test_nonzero_tensor_does_not_qualify_unknown_partner_or_support():
    summary = api().summarize_pair_calibration(fixture_rows(), object_path='/World/Object',
        expected_partner='/World/Finger', inventory={'scope': 'fixture', 'stage_sha256': 'a'*64})
    assert summary['nonzero_tensor_steps'] == 4
    assert summary['R2_qualified'] is False
    assert summary['pair_identity_verified'] is False
    assert summary['force_proxy_qualified'] is False
    assert 'no_nonzero_native_pair_report' in summary['blockers']


def test_repeated_step_and_nonmonotonic_time_rejected():
    rows = fixture_rows()
    rows[1]['physics_step'] = rows[0]['physics_step']
    with pytest.raises(ValueError, match='monotonic'):
        api().summarize_pair_calibration(rows, object_path='/World/Object',
            expected_partner='/World/Finger', inventory={})


def test_unknown_report_side_is_evaluated_both_ways_never_assumed():
    rows = fixture_rows()
    for row in rows:
        row['pair_reports'] = [dict(actor0='/World/Finger', actor1='/World/Object',
            collider0='/World/Finger/collider', collider1='/World/Object/collider',
            impulse=[0., 0., -.1*9.81/480], normal=[0., 0., -1.],
            friction_impulses=[])]
    summary = api().summarize_pair_calibration(rows, object_path='/World/Object',
        expected_partner='/World/Finger', inventory={'scope': 'fixture'})
    assert summary['impulse_sign_residual_n_s']['minus']['max'] < 1e-12
    assert summary['impulse_sign_residual_n_s']['plus']['max'] > 0.004
    assert summary['R2_qualified'] is False


def test_side_inference_not_merged_across_actor_order():
    rows = fixture_rows()
    rows[0]['pair_reports'] = [dict(actor0='/World/Finger', actor1='/World/Object',
        collider0='/World/Finger', collider1='/World/Object', impulse=[0.,0.,-.002],
        normal=[0.,0.,-1.], friction_impulses=[])]
    rows[1]['pair_reports'] = [dict(actor0='/World/Object', actor1='/World/Finger',
        collider0='/World/Object', collider1='/World/Finger', impulse=[0.,0.,.002],
        normal=[0.,0.,1.], friction_impulses=[])]
    summary=api().summarize_pair_calibration(rows,object_path='/World/Object',
        expected_partner='/World/Finger',inventory={})
    assert set(summary['pair_order_counts']) == {'object_actor0','object_actor1'}
    assert summary['impulse_sign_residual_n_s'] is None


def test_binary_thumb_mesh_units_and_shape_are_not_rescaled():
    from pathlib import Path
    path=Path('resources/manufacturer/DH116V2.0_URDF_MJCF_Files/DH116-V2.0-L/meshes/finger14_link.STL')
    vertices=api().read_binary_stl(path)
    assert vertices.shape == (1874, 3, 3)
    assert .005 < np.ptp(vertices.reshape(-1,3), axis=0).max() < .1


def test_truncated_stl_refused(tmp_path):
    path=tmp_path/'bad.stl'
    path.write_bytes(bytes(84))
    with pytest.raises(ValueError,match='STL'):
        api().read_binary_stl(path)


def test_report_processing_opt_in_changes_only_diagnostic_setting():
    class Settings:
        def __init__(self):self.data={'/physics/disableContactProcessing':True,'/physics/suppressReadback':True}
        def get(self,key):return self.data.get(key)
        def set_bool(self,key,value):self.data[key]=value
    settings=Settings()
    before=copy.deepcopy(settings.data)
    row=api().configure_report_processing(settings,enable=False)
    assert settings.data==before and row['modified'] is False
    row=api().configure_report_processing(settings,enable=True)
    assert settings.data['/physics/disableContactProcessing'] is False
    assert settings.data['/physics/suppressReadback'] is True
    assert row['before'] is True and row['after'] is False


def test_raw_report_preserves_header_only_and_anchor_positions():
    from types import SimpleNamespace as NS
    decoder=getattr(api(),'decode_contact_report',None)
    assert callable(decoder), 'lossless full report decoder required'
    header=NS(actor0=1,actor1=2,collider0=3,collider1=4,stage_id=7,
              contact_data_offset=0,num_contact_data=0,friction_anchors_offset=0,num_friction_anchors_data=1)
    anchor=NS(position=(.1,.2,.3),impulse=(.001,0.,0.))
    result=decoder(([header],[],[anchor]),path_decoder=lambda v:f'/World/{v}')
    assert len(result['headers'])==1 and not result['contacts']
    assert result['friction_anchors'][0]['position']==[.1,.2,.3]
    assert result['headers'][0]['num_friction_anchors_data']==1


def test_capacity_boundary_is_preserved_as_unknown_completeness():
    value=api().capture_tensor_pair(View(),dt=1/480)
    assert value['normal']['capacity_boundary_reached']
    assert value['normal']['buffer_capacities']==[1,1,1,1]
    rows=fixture_rows()
    summary=api().summarize_pair_calibration(rows,object_path='/World/Object',expected_partner='/World/Finger',inventory={})
    assert summary['capacity_boundary_steps']==4
    assert 'tensor_capacity_boundary_requires_overflow_audit' in summary['blockers']


def test_contact_impulse_and_anchor_impulse_are_separate_hypotheses():
    rows=fixture_rows()
    for row in rows:
        row['velocity_after']=[.02,0.,0.]
        row['pair_reports']=[dict(actor0='/World/Object',actor1='/World/Finger',
            collider0='/World/Object',collider1='/World/Finger/collision',
            impulse=[.002,0.,.1*9.81/480],normal=[0.,0.,1.],friction_impulses=[[.002,0.,0.]])]
    summary=api().summarize_pair_calibration(rows,object_path='/World/Object',expected_partner='/World/Finger',
        inventory={'object_collider':'/World/Object','finger_collider':'/World/Finger/collision'})
    variants=summary['report_impulse_hypotheses']
    assert variants['contact_only']['plus']['max']<1e-12
    assert variants['contact_plus_friction']['plus']['min']>=.002-1e-12
    assert summary['report_impulse_semantics_qualified'] is False
    assert summary['collider_identity_verified']


def test_wrong_collider_cannot_be_called_verified_pair():
    rows=fixture_rows()
    rows[0]['pair_reports']=[dict(actor0='/World/Object',actor1='/World/Finger',
        collider0='/World/Object',collider1='/World/Unexpected',impulse=[0.,0.,.001],friction_impulses=[])]
    s=api().summarize_pair_calibration(rows,object_path='/World/Object',expected_partner='/World/Finger',
        inventory={'object_collider':'/World/Object','finger_collider':'/World/Finger/collision'})
    assert not s['pair_identity_verified']
    assert s['unmatched_collider_records']==1


def test_reverse_tensor_audits_per_step_total_and_component_action_reaction():
    rows=fixture_rows()
    for row in rows:
        row['reverse_tensor']=copy.deepcopy(row['tensor'])
        row['reverse_tensor']['normal']['normal']=[[0.,0.,-1.]]
    s=api().summarize_pair_calibration(rows,object_path='/World/Object',expected_partner='/World/Finger',inventory={})
    assert s['reverse_tensor_action_reaction_n_s']['total']['max']==0.
    assert s['reverse_tensor_action_reaction_n_s']['normal']['max']==0.
    assert s['reverse_tensor_action_reaction_n_s']['friction']['max']==0.


def test_isolated_close_bypasses_replicator_wait_and_marks_completion(tmp_path):
    close=getattr(api(),'close_calibration_app',None)
    assert callable(close), 'isolated app shutdown audit required'
    class App:
        def close(self,**kwargs):
            assert (tmp_path/'shutdown_started.json').exists()
            assert not (tmp_path/'shutdown_complete.json').exists()
            assert kwargs=={'wait_for_replicator':False}
    close(App(),tmp_path)
    assert (tmp_path/'shutdown_complete.json').exists()
    assert (tmp_path/'shutdown_stacks.log').exists()


def test_swapped_colliders_with_correct_set_are_not_matched_to_actor_sides():
    rows=fixture_rows()
    rows[0]['pair_reports']=[dict(actor0='/World/Object',actor1='/World/Finger',
        collider0='/World/Finger/collision',collider1='/World/Object',impulse=[0.,0.,.001],friction_impulses=[])]
    s=api().summarize_pair_calibration(rows,object_path='/World/Object',expected_partner='/World/Finger',
        inventory={'object_collider':'/World/Object','finger_collider':'/World/Finger/collision'})
    assert not s['collider_identity_verified'] and s['unmatched_collider_records']==1


def test_structured_anchor_only_report_survives_analysis():
    rows=fixture_rows()
    for row in rows:
        row['full_report']={'headers':[dict(actor0='/World/Object',actor1='/World/Finger',
            collider0='/World/Object',collider1='/World/Finger/collision',contact_data_offset=0,num_contact_data=0,
            friction_anchors_offset=0,num_friction_anchors_data=1)],'contacts':[],
            'friction_anchors':[dict(position=[0.,0.,0.],impulse=[.001,0.,0.])]}
    s=api().summarize_pair_calibration(rows,object_path='/World/Object',expected_partner='/World/Finger',inventory={})
    assert s['nonzero_report_steps']==4
    assert s['report_impulse_hypotheses']['friction_only']['plus'] is not None


def test_close_failure_cannot_write_shutdown_complete(tmp_path):
    class App:
        def close(self,**kwargs):raise RuntimeError('fixture close failure')
    with pytest.raises(RuntimeError,match='close failure'):
        api().close_calibration_app(App(),tmp_path)
    assert (tmp_path/'shutdown_started.json').exists()
    assert not (tmp_path/'shutdown_complete.json').exists()


def test_public_context_clear_unsubscribes_stop_handler_before_app_close(tmp_path):
    events=[]
    class Simulation:
        current=None
        @classmethod
        def instance(cls):return cls.current
        @classmethod
        def clear_instance(cls):
            events.append('clear_stop_callback_and_context');cls.current=None
    sim=Simulation();Simulation.current=sim
    class App:
        def close(self,**kwargs):
            assert Simulation.instance() is None, 'STOP callback would enter renderer loop'
            events.append('app_close')
    api().close_calibration_app(App(),tmp_path,simulation=sim)
    assert events==['clear_stop_callback_and_context','app_close']


def test_context_cleanup_refuses_another_owners_singleton(tmp_path):
    class Simulation:
        @classmethod
        def instance(cls):return object()
        @classmethod
        def clear_instance(cls):raise AssertionError('other simulation must not be cleared')
    class App:
        def close(self,**kwargs):raise AssertionError('must refuse ownership first')
    with pytest.raises(RuntimeError,match='owned'):
        api().close_calibration_app(App(),tmp_path,simulation=Simulation())
    assert not (tmp_path/'shutdown_complete.json').exists()


def test_context_callback_failure_survives_successful_cleanup(tmp_path,monkeypatch):
    import builtins
    monkeypatch.setattr(builtins,'ISAACLAB_CALLBACK_EXCEPTION',RuntimeError('native callback failed'),raising=False)
    class App:
        def close(self,**kwargs):pass
    with pytest.raises(RuntimeError,match='native callback failed'):
        api().close_calibration_app(App(),tmp_path)


def test_ordinary_callback_does_not_fabricate_missing_friction_channel():
    from types import SimpleNamespace as NS
    h=NS(actor0=1,actor1=2,collider0=3,collider1=4,stage_id=7,contact_data_offset=0,
         num_contact_data=0,friction_anchors_offset=0,num_friction_anchors_data=0)
    result=api().decode_contact_report(([h],[]),path_decoder=str)
    assert result['friction_anchors'] is None
    assert result['friction_source_available'] is False


def test_callback_buffer_keeps_channels_windows_and_owned_copies_separate():
    from types import SimpleNamespace as NS
    cls=getattr(api(),'ContactReportReceiptBuffer',None)
    assert callable(cls), 'separate callback receipt buffer required'
    clock=[0,0.]
    b=cls(clock_reader=lambda:tuple(clock),path_decoder=str)
    b.record('ordinary',([],[]))
    b.begin_step(1)
    h=NS(actor0=1,actor1=2,collider0=3,collider1=4,stage_id=7,contact_data_offset=0,
         num_contact_data=1,friction_anchors_offset=0,num_friction_anchors_data=0)
    point=NS(position=[0.,0.,0.],impulse=[.1,0.,0.],normal=[1.,0.,0.],separation=0.)
    b.record('ordinary',([h],[point]))
    point.impulse[0]=99.
    clock[:]=[1,1/480]
    b.record('full',([],[],[]))
    b.end_step()
    receipts=b.drain()
    assert len(receipts)==3
    assert receipts[0]['requested_physics_step'] is None and not receipts[0]['within_simulation_step_call']
    assert receipts[1]['requested_physics_step']==1 and receipts[1]['within_simulation_step_call']
    assert receipts[1]['raw_report']['contacts'][0]['impulse'][0]==.1
    assert receipts[1]['channel']=='ordinary' and receipts[2]['channel']=='full'
    assert b.drain()==[]


def test_callbacks_are_diagnostic_and_never_double_counted_as_polled_impulse():
    rows=fixture_rows()
    rows[0]['callback_receipts']=[dict(channel='ordinary',raw_report={'headers':[{}],
        'contacts':[{'impulse':[99.,0.,0.]}],'friction_anchors':None},within_simulation_step_call=True)]
    s=api().summarize_pair_calibration(rows,object_path='/World/Object',expected_partner='/World/Finger',inventory={})
    assert s['nonzero_report_steps']==0
    assert s['callback_source_counts']['ordinary']['contacts']==1
    assert s['callback_source_counts']['ordinary']['friction_available_receipts']==0


def test_calibration_episode_identity_includes_unique_run_directory(tmp_path):
    uid=getattr(api(),'calibration_episode_uid',None)
    assert callable(uid), 'same output leaf in different native runs must not alias'
    first=tmp_path/'v02'/'left';second=tmp_path/'v04'/'left'
    assert uid(first,'left')!=uid(second,'left')
    assert uid(first,'left')==uid(first,'left')
    assert uid(first,'left')!=uid(first,'right')


@pytest.mark.parametrize('accept_readback', [True, False])
def test_explicit_report_readback_is_verified_and_kept_separate(accept_readback):
    from go2_pvcnn.control.dh116.tabletop_stereo_contact_calibration import configure_report_processing

    class Settings:
        values = {'/physics/disableContactProcessing': True, '/physics/suppressReadback': True}

        def get(self, key):
            return self.values[key]

        def set_bool(self, key, value):
            if key != '/physics/suppressReadback' or accept_readback:
                self.values[key] = value

    settings = Settings()
    if not accept_readback:
        with pytest.raises(RuntimeError, match='suppressReadback'):
            configure_report_processing(settings, enable=True, enable_readback=True)
    else:
        result = configure_report_processing(settings, enable=True, enable_readback=True)
        assert result['suppress_readback_before'] is True
        assert result['suppress_readback'] is False
        assert result['readback_requested'] is True
        assert result['physical_solver_or_original_task_modified'] is False
