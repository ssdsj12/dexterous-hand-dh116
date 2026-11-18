import ast,weakref,gc
from pathlib import Path
from types import SimpleNamespace
import pytest

SOURCE=Path('/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/omni/kernel/py/omni/ext/_impl/__init__.py')


def fixture():
    node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='get_extensions_interface')
    class Interface:pass
    Interface.__module__='omni.ext._extensions';Interface.__name__='IExtensions'
    value=Interface();ref=weakref.ref(value)
    scope={'IExtensions':Interface,'acquire_extensions_interface':lambda:None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),scope)
    getter=scope['get_extensions_interface'];getter.extensions=value
    del value
    events=[]
    class Framework:
        def unload_all_plugins(self):
            gc.collect()
            assert ref() is None,'cached native interface survives plugin unload'
            events.append('unload')
    class App:pass
    app=App();app._framework=Framework()
    return app,getter,ref,events


def test_actual_sdk_getter_cache_survives_without_explicit_release():
    app,getter,ref,events=fixture()
    assert ref() is not None
    with pytest.raises(AssertionError,match='survives'):app._framework.unload_all_plugins()


def test_release_existing_cache_before_standard_framework_unload():
    from go2_pvcnn.control.dh116.tabletop_preview_extension_cache import prepare_extension_cache_release
    app,getter,ref,events=fixture()
    receipt=prepare_extension_cache_release(app,getter,source=SOURCE)
    assert ref() is not None # Do not release before app.shutdown.
    app._framework.unload_all_plugins()
    assert ref() is None and events==['unload']
    assert receipt['cache_reference_removed'] and receipt['framework_unload_returned']


def test_record_failure_does_not_prevent_original_framework_unload():
    from go2_pvcnn.control.dh116.tabletop_preview_extension_cache import prepare_extension_cache_release
    app,getter,ref,events=fixture()
    def record(row):raise OSError('disk full')
    receipt=prepare_extension_cache_release(app,getter,source=SOURCE,record=record)
    app._framework.unload_all_plugins()
    assert events==['unload'] and receipt['diagnostic_errors']


def test_wrong_source_is_rejected_before_app_mutation(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_preview_extension_cache import prepare_extension_cache_release
    app,getter,ref,events=fixture();original=app._framework
    with pytest.raises(ValueError):prepare_extension_cache_release(app,getter,source=tmp_path/'wrong.py')
    assert app._framework is original and ref() is not None


def test_shutdown_release_unload_order_and_retained_completed_frame():
    from test_preview_retained_frames import hold_completed
    from go2_pvcnn.control.dh116.tabletop_preview_extension_cache import prepare_extension_cache_release
    app,getter,ref,events=fixture()
    receipt=prepare_extension_cache_release(app,getter,source=SOURCE)
    events.append('shutdown')
    with hold_completed(type(app._framework).unload_all_plugins) as frames:
        app._framework.unload_all_plugins()
        assert frames and ref() is None
    assert events==['shutdown','unload'] and receipt['framework_unload_returned']


def test_cache_missing_at_release_still_calls_original_unload_once():
    from go2_pvcnn.control.dh116.tabletop_preview_extension_cache import prepare_extension_cache_release
    app,getter,ref,events=fixture()
    receipt=prepare_extension_cache_release(app,getter,source=SOURCE)
    del getter.extensions
    app._framework.unload_all_plugins()
    assert events==['unload'] and receipt['diagnostic_errors']
    assert not receipt['cache_reference_removed']
    with pytest.raises(RuntimeError,match='retried'):app._framework.unload_all_plugins()
    assert events==['unload']


def test_cleanup_failure_and_unload_failure_are_both_reported():
    from go2_pvcnn.control.dh116.tabletop_preview_extension_cache import prepare_extension_cache_release,validate_extension_cache_receipt
    app,getter,ref,events=fixture()
    class Framework:
        def unload_all_plugins(self):
            events.append('unload')
            raise RuntimeError('plugin unload failed')
    app._framework=Framework()
    receipt=prepare_extension_cache_release(app,getter,source=SOURCE)
    del getter.extensions
    with pytest.raises(RuntimeError,match='plugin unload failed'):app._framework.unload_all_plugins()
    with pytest.raises(RuntimeError,match='not confirmed'):validate_extension_cache_receipt(receipt)
    assert events==['unload']

@pytest.mark.parametrize("kind",["direct","loaded"])
def test_preparation_completed_frame_cannot_keep_app_alive(monkeypatch,kind):
    import sys
    from test_preview_retained_frames import hold_completed
    from go2_pvcnn.control.dh116 import tabletop_preview_extension_cache as cache
    app,getter,ref,events=fixture();app_ref=weakref.ref(app)
    if kind=="direct":
        function=cache.prepare_extension_cache_release
        kwargs=dict(getter=getter,source=SOURCE)
    else:
        function=cache.prepare_loaded_extension_cache;kwargs={}
        monkeypatch.setitem(sys.modules,"omni.ext._impl",SimpleNamespace(get_extensions_interface=getter))
        monkeypatch.setitem(sys.modules,"omni.ext",SimpleNamespace(get_extensions_interface=getter))
    with hold_completed(function) as frames:
        function(app,**kwargs)
        app=None;gc.collect()
        assert frames and app_ref() is None


def test_actual_launcher_framework_preparer_drops_borrowed_app():
    from test_preview_retained_frames import hold_completed
    from go2_pvcnn.control.dh116 import tabletop_preview_launcher as launcher
    tree=ast.parse(Path(launcher.__file__).read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="prepare_framework")
    namespace=dict(prepare_loaded_extension_cache=lambda app,**kw:None,shutdown_output=None)
    exec(compile(ast.Module(body=[node],type_ignores=[]),launcher.__file__,"exec"),namespace)
    function=namespace["prepare_framework"]
    app,getter,ref,events=fixture();app_ref=weakref.ref(app)
    with hold_completed(function) as frames:
        function(app);app=None;gc.collect()
        assert frames and app_ref() is None


def test_default_loaded_probe_is_read_only_and_preserves_framework(monkeypatch):
    import sys
    from go2_pvcnn.control.dh116.tabletop_preview_extension_cache import prepare_loaded_extension_cache
    app,getter,ref,events=fixture();original=app._framework;records=[]
    monkeypatch.setitem(sys.modules,"omni.ext._impl",SimpleNamespace(get_extensions_interface=getter))
    monkeypatch.setitem(sys.modules,"omni.ext",SimpleNamespace(get_extensions_interface=getter))
    receipt=prepare_loaded_extension_cache(app,record=records.append)
    assert app._framework is original and ref() is not None
    assert receipt['mode']=='audit_only' and receipt['cache_present']
    assert receipt['native_qualification'] is False and records


def test_missing_runtime_cache_is_diagnosed_without_creating_or_clearing_one(monkeypatch):
    import sys
    from go2_pvcnn.control.dh116.tabletop_preview_extension_cache import prepare_loaded_extension_cache,validate_extension_cache_receipt
    app,getter,ref,events=fixture();del getter.extensions;original=app._framework
    monkeypatch.setitem(sys.modules,"omni.ext._impl",SimpleNamespace(get_extensions_interface=getter))
    monkeypatch.setitem(sys.modules,"omni.ext",SimpleNamespace(get_extensions_interface=getter))
    receipt=prepare_loaded_extension_cache(app)
    assert receipt['cache_present'] is False and app._framework is original
    assert not hasattr(getter,'extensions')
    validate_extension_cache_receipt(receipt)
