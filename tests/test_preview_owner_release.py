import weakref
import pytest
from types import SimpleNamespace

def test_closed_rgb_bridge_drops_sim_reader_and_snapshot():
    from test_dh116_stereo_rgb_bridge import calibration,Sim,RenderSetting,Reader
    from go2_pvcnn.control.dh116.tabletop_stereo_rgb_bridge import RGBPairBridge
    c=calibration();sim=Sim();reader=Reader(c)
    sim_ref=weakref.ref(sim);reader_ref=weakref.ref(reader)
    bridge=RGBPairBridge(sim,reader,c,lambda sim=sim:sim,render_setting=RenderSetting())
    bridge.close()
    del sim,reader
    assert sim_ref() is None and reader_ref() is None

def test_fatal_defaults_drop_recursive_python_close_handler():
    from go2_pvcnn.control.dh116.tabletop_preview_launcher import install_fatal_defaults
    import signal
    callbacks={x:lambda *_:None for x in (signal.SIGABRT,signal.SIGSEGV,signal.SIGBUS)}
    api=SimpleNamespace(SIGABRT=signal.SIGABRT,SIGSEGV=signal.SIGSEGV,SIGBUS=signal.SIGBUS,SIG_DFL=signal.SIG_DFL,
        signal=lambda n,handler:callbacks.__setitem__(n,handler))
    install_fatal_defaults(api)
    assert all(x==signal.SIG_DFL for x in callbacks.values())

def test_managed_app_unsubscribes_callbacks_and_close_once():
    from go2_pvcnn.control.dh116.tabletop_preview_launcher import ManagedPreviewApp
    events=[]
    class Sub:
        def unsubscribe(self):events.append("unsubscribe")
    app=SimpleNamespace(close=lambda **kw:events.append("close"),update=lambda:None)
    class Launcher:
        def __init__(self):
            self._app=app
            self._hide_play_button_callback=Sub()
            self._unhide_play_button_callback=Sub()
        @property
        def app(self):return self._app
    launcher=Launcher()
    managed=ManagedPreviewApp(launcher,signal_api=None)
    managed.close(wait_for_replicator=False)
    managed.close(wait_for_replicator=False)
    assert events==["unsubscribe","unsubscribe","close"]
    assert launcher._app is None and managed._app is None

def test_callback_failure_still_closes_app_and_preserves_failed_close_status():
    from go2_pvcnn.control.dh116.tabletop_preview_launcher import ManagedPreviewApp
    events=[]
    class BrokenSub:
        def unsubscribe(self):raise RuntimeError("unsubscribe error")
    class Launcher:
        _hide_play_button_callback=BrokenSub()
        _unhide_play_button_callback=None
        def __init__(self):self._app=SimpleNamespace(close=lambda **kw:events.append("close"))
        @property
        def app(self):return self._app
    managed=ManagedPreviewApp(Launcher(),signal_api=None)
    with pytest.raises(RuntimeError,match="unsubscribe error"):managed.close()
    with pytest.raises(RuntimeError,match="unsubscribe error"):managed.close()
    assert events==["close"]

def test_signal_cleanup_failure_cannot_skip_app_close():
    import signal
    from go2_pvcnn.control.dh116.tabletop_preview_launcher import ManagedPreviewApp
    events=[]
    class Launcher:
        _hide_play_button_callback=_unhide_play_button_callback=None
        def __init__(self):self._app=SimpleNamespace(close=lambda **kw:events.append("close"))
        @property
        def app(self):return self._app
    def broken(number):raise RuntimeError("signal cleanup error")
    api=SimpleNamespace(SIGABRT=signal.SIGABRT,SIGSEGV=signal.SIGSEGV,SIGBUS=signal.SIGBUS,
        SIGTERM=signal.SIGTERM,SIG_DFL=signal.SIG_DFL,signal=lambda *a:None,getsignal=broken)
    managed=ManagedPreviewApp(Launcher(),signal_api=api)
    with pytest.raises(RuntimeError,match="signal cleanup error"):managed.close()
    assert events==["close"]

def test_camera_cleanup_failure_drops_bridge_aliases_but_retains_failure():
    from test_dh116_stereo_rgb_bridge import calibration,Sim,RenderSetting
    from go2_pvcnn.control.dh116.tabletop_stereo_rgb_bridge import RGBPairBridge
    class BrokenReader:
        def close(self):raise RuntimeError("camera detach failed")
    bridge=RGBPairBridge(Sim(),BrokenReader(),calibration(),lambda:None,render_setting=RenderSetting())
    bridge._owns_reader=True
    with pytest.raises(RuntimeError,match="camera detach failed"):bridge.close()
    assert bridge.sim is bridge.reader is bridge.snapshot is None
    with pytest.raises(RuntimeError,match="camera detach failed"):bridge.close()
