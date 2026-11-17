"""CPU replay of installed STOP handler, plus owned close ordering."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from go2_pvcnn.control.dh116 import tabletop_preview_launcher as launcher

MANAGER=Path('/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/exts/isaacsim.core.simulation_manager/isaacsim/core/simulation_manager/impl/simulation_manager.py')


def manager_callback():
    tree=ast.parse(MANAGER.read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_on_stop')
    events=[]
    cls=type('ActualStopManager',(),dict(_physics_sim_view=SimpleNamespace(invalidate=lambda:events.append('invalidate'))))
    namespace=dict(SimulationManager=cls)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(MANAGER),'exec'),namespace)
    return cls,namespace['_on_stop'],events


def test_stop_commit_delivers_installed_manager_callback_without_app_update():
    manager,callback,events=manager_callback()
    class Timeline:
        stopped=False
        def stop(self):events.append('stop');self.stopped=True
        def commit(self):events.append('commit');callback(None)
        def is_stopped(self):return self.stopped
    timeline=Timeline()
    # The old stop-only path deterministically leaves the real handler unapplied.
    timeline.stop()
    assert manager._physics_sim_view is not None
    events.clear()
    launcher.commit_preview_stop(timeline,freeze=lambda:False,read_clock=lambda:dict(sdk_step=130,sdk_time_s=130/480))
    assert events==['stop','commit','invalidate']
    assert manager._physics_sim_view is None


def test_stop_commit_cannot_silently_advance_sdk_clock():
    clock=dict(sdk_step=130,sdk_time_s=130/480)
    t=SimpleNamespace(stop=lambda:None,commit=lambda:clock.update(sdk_step=131),is_stopped=lambda:True)
    with pytest.raises(RuntimeError,match='clock'):
        launcher.commit_preview_stop(t,freeze=lambda:False,read_clock=lambda:dict(clock))


def test_stop_commit_requires_verified_physics_freeze():
    t=SimpleNamespace(stop=lambda:pytest.fail('must not stop before freeze check'))
    with pytest.raises(RuntimeError,match='freeze'):
        launcher.commit_preview_stop(t,freeze=lambda:True,read_clock=lambda:None)


def test_livestream_shutdown_runs_before_underlying_app_closes_stage():
    events=[]
    class Launcher:
        _hide_play_button_callback=_unhide_play_button_callback=None
        _app=SimpleNamespace(close=lambda **kw:events.append('stage-close'),update=lambda:None)
        @property
        def app(self):return self._app
    app=launcher.ManagedPreviewApp(Launcher(),signal_api=None,
        stop_presentation=lambda:events.append('livestream-shutdown'))
    app.stop_presentation()
    app.close();app.close()
    assert events==['livestream-shutdown','stage-close']


def test_failed_stream_shutdown_is_not_retried_or_reported_success():
    calls=[]
    def broken():
        calls.append('stop')
        raise RuntimeError('stream shutdown failed')
    class Launcher:
        _hide_play_button_callback=_unhide_play_button_callback=None
        _app=SimpleNamespace(close=lambda **kw:calls.append('app-close'))
        @property
        def app(self):return self._app
    app=launcher.ManagedPreviewApp(Launcher(),signal_api=None,stop_presentation=broken)
    with pytest.raises(RuntimeError,match='stream shutdown failed'):app.stop_presentation()
    with pytest.raises(RuntimeError,match='stream shutdown failed'):app.close()
    with pytest.raises(RuntimeError,match='stream shutdown failed'):app.close()
    assert calls==['stop','app-close']


def test_owned_stream_interface_uses_installed_disable_then_shutdown():
    calls=[]
    api=SimpleNamespace(enable_livestream=lambda v:calls.append(('enabled',v)) or True,
        shutdown=lambda:calls.append('shutdown') or True)
    launcher.quiesce_stream_interface(api)
    assert calls==[('enabled',False),'shutdown']


def test_owned_stream_return_false_remains_failure_and_shutdown_still_attempted():
    calls=[]
    api=SimpleNamespace(enable_livestream=lambda v:False,shutdown=lambda:calls.append('shutdown') or False)
    with pytest.raises(RuntimeError,match='disable.*shutdown'):
        launcher.quiesce_stream_interface(api)
    assert calls==['shutdown']


def test_owned_cleanup_quiesces_before_camera_product_and_preserves_errors():
    calls=[]
    app=SimpleNamespace(stop_presentation=lambda:calls.append('stream'))
    def camera_close():calls.append('camera');raise RuntimeError('detach failure')
    camera=SimpleNamespace(close=camera_close)
    render=SimpleNamespace(close=lambda:calls.append('render'))
    journal=SimpleNamespace(close=lambda:calls.append('journal'))
    holder=dict(native=object())
    with pytest.raises(RuntimeError,match='detach failure'):
        launcher.finish_preview_resources(app,camera,render,holder,journal)
    assert calls==['stream','camera','render','journal'] and holder=={}


def test_partial_launcher_stops_stream_before_closing_created_app():
    events=[]
    class Broken:
        def __init__(self,**kw):
            self._app=SimpleNamespace(close=lambda **kw:events.append('app-close'))
            raise RuntimeError('after-start failure')
    cls=launcher.preview_launcher_type(Broken,toolbar_getter=lambda:None,
        partial_stop_presentation=lambda:events.append('stream-stop'))
    with pytest.raises(RuntimeError,match='after-start failure'):cls()
    assert events==['stream-stop','app-close']


def test_partial_stream_stop_failure_does_not_hide_original_or_skip_app_close():
    events=[]
    def stop():
        events.append('stream-stop');raise RuntimeError('partial stop error')
    class Broken:
        def __init__(self,**kw):
            self._app=SimpleNamespace(close=lambda **kw:events.append('app-close'))
            raise RuntimeError('original startup error')
    cls=launcher.preview_launcher_type(Broken,toolbar_getter=lambda:None,partial_stop_presentation=stop)
    with pytest.raises(RuntimeError,match='original startup error') as error:cls()
    assert events==['stream-stop','app-close']
    assert 'partial stop error' in error.value.preview_startup_cleanup_error


def test_installed_stop_and_livestream_sources_are_bound_without_importing_native():
    import sys
    was_loaded='omni.kit.livestream.bind' in sys.modules
    data=launcher.verify_preview_shutdown_sources()
    assert data['installed_sdk_files_modified'] is False
    assert data['native_qualification'] is False
    assert ('omni.kit.livestream.bind' in sys.modules)==was_loaded
