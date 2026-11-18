import importlib.util
import ast
from pathlib import Path
import pytest


def health(name='physics_problems'):
    path=Path(__file__).resolve().parents[1]/'scripts/watch_dh116_tensor_ppo.py'
    tree=ast.parse(path.read_text())
    assert any(isinstance(node,ast.FunctionDef) and node.name=='physics_problems' for node in tree.body)
    # Import must not enter the observer loop or inspect the active run.
    spec=importlib.util.spec_from_file_location('dh116_observer',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module,name)


def test_observer_flags_observed_live_speed_even_when_status_is_training():
    problems=health()({'status':'training','physics_peaks_since_start':{
        'all_joint_speed_rad_s':8.834162712097168,'link_force_n':34.46}})
    assert any('all_joint_speed_rad_s' in problem for problem in problems)
    assert not any('link_force_n' in problem for problem in problems)


def test_observer_accepts_bounded_recovery_and_nominal_velocity_tolerance():
    assert health()({'recovered_physical_episodes':23,'physics_peaks_since_start':{
        'all_joint_speed_rad_s':3.0000067,'active_joint_speed_rad_s':2.9,
        'link_force_n':34.46,'mimic_error_rad':.025}})==[]


@pytest.mark.parametrize('name,value',[
    ('all_joint_speed_rad_s',None),('link_force_n',float('nan')),
    ('link_force_n',45.),('mimic_error_rad',.03)])
def test_observer_flags_invalid_or_fatal_peaks(name,value):
    assert any(name in p for p in health()({'physics_peaks_since_start':{name:value}}))


@pytest.mark.parametrize('peaks',[None,[],{}, {'link_force_n':12.}])
def test_training_requires_complete_peak_telemetry(peaks):
    assert any('telemetry_' in p for p in health()({'status':'training','physics_peaks_since_start':peaks}))


def test_missing_peaks_are_allowed_during_startup_only():
    assert health()({'status':'initializing'})==[]
    assert health()({'status':'startup_preflight'})==[]
    assert health()({'status':'training'})==['physics_peak_telemetry_missing']


def test_observer_reports_fault_trace_recording_failure():
    row={'status':'initializing','fault_trace_record_error':'trace allocation failed'}
    assert health()(row)==['fault_trace_record_error:trace allocation failed']
    row['fault_trace_record_error']=None
    assert health()(row)==[]


@pytest.mark.parametrize('field,value',[
    ('policy_std_max',16.),('policy_std_min',.001),('policy_std_max',float('nan')),
    ('action_clipped_fraction',1.1),('policy_std_min',None)])
def test_observer_flags_invalid_bounded_policy_noise(field,value):
    row=dict(policy_noise={'version':'bounded_policy_noise_v1'},policy_std_min=.05,
             policy_std_max=1.,action_clipped_fraction=.45)
    assert health('policy_noise_problems')(row)==[]
    row[field]=value
    assert health('policy_noise_problems')(row)


def test_policy_noise_observer_preserves_legacy_and_requires_new_telemetry():
    check=health('policy_noise_problems')
    assert check({'status':'training'})==[]
    assert check({'policy_noise':{'version':'bounded_policy_noise_v1'}})


def test_observer_reports_degraded_endpoint_telemetry_separately_from_physics():
    check = health('telemetry_problems')
    diagnostic = dict(status='degraded', observed_events=3,
                      errors={'journal': 'OSError: journal unavailable'},
                      first_affected_event_index={'journal': 2})
    monitor = dict(status='training', close_completion_telemetry=diagnostic)
    assert check(monitor) == ['close_completion_telemetry_degraded:journal:OSError: journal unavailable']
    assert check({'status': 'training'}) == []
    assert check({'close_completion_telemetry': dict(status='ok', errors={})}) == []


def test_observer_keeps_watching_physics_after_optional_endpoint_io_failure(tmp_path, monkeypatch):
    import json
    import os
    main = health('main')
    run = tmp_path / 'run'; run.mkdir()
    (tmp_path / 'runs').mkdir()
    monitor = run / 'monitor.json'
    monitor.write_text(json.dumps(dict(status='training', timesteps=8, updates=1,
        physics_peaks_since_start=dict(all_joint_speed_rad_s=2., active_joint_speed_rad_s=.2,
                                      link_force_n=5., mimic_error_rad=.01),
        close_completion_telemetry=dict(status='degraded', errors={'journal': 'OSError: unavailable'}))))
    (tmp_path / 'runs' / 'active_tabletop_run.json').write_text(json.dumps(dict(
        run=str(run), pid=os.getpid(), monitor=str(monitor))))
    monkeypatch.chdir(tmp_path)
    def stop_after_first_poll(_): raise RuntimeError('observer reached next poll')
    monkeypatch.setattr(main.__globals__['time'], 'sleep', stop_after_first_poll)
    with pytest.raises(RuntimeError, match='observer reached next poll'):
        main()
    snapshot = json.loads((run / 'live_watch.json').read_text())
    assert snapshot['problems'] == []
    assert snapshot['telemetry_warnings'] == ['close_completion_telemetry_degraded:journal:OSError: unavailable']
