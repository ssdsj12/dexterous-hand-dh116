"""Replay installed UI owner destructor on CPU; not native qualification."""
import ast,gc,weakref
from pathlib import Path
from types import SimpleNamespace
import pytest
from go2_pvcnn.control.dh116 import tabletop_preview_launcher as launcher

TRIGGER=Path('/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extscache/omni.kit.hotkeys.core-1.3.10+d02c707b/omni/kit/hotkeys/core/trigger.py')


def owner(*,broken=False):
    events=[]
    node=next(n for n in ast.walk(ast.parse(TRIGGER.read_text()))
        if isinstance(n,ast.FunctionDef) and n.name=='destroy')
    definition=ast.ClassDef(name='HotkeyTrigger',bases=[],keywords=[],body=[node],decorator_list=[])
    namespace={'__name__':'omni.kit.hotkeys.core.trigger'}
    tree=ast.fix_missing_locations(ast.Module(body=[definition],type_ignores=[]))
    exec(compile(tree,str(TRIGGER),'exec'),namespace)
    cls=namespace['HotkeyTrigger']
    obj=cls()
    obj._stop_event=SimpleNamespace(set=lambda:events.append('stop'))
    obj._work_queue=SimpleNamespace(put_nowait=lambda value:events.append('sentinel'))
    obj._HotkeyTrigger__run_future=SimpleNamespace(cancel=lambda:events.append('cancel'))
    def unsubscribe(value):
        events.append('unsubscribe')
        if broken:raise RuntimeError('input unsubscribe failed')
    obj._input=SimpleNamespace(unsubscribe_to_input_events=unsubscribe)
    obj._input_sub_id=7;obj._sub_allow_list=object()
    class Window:pass
    window=Window();ref=weakref.ref(window)
    obj._HotkeyTrigger__app_window=window
    del window
    return obj,ref,events


def test_installed_destroy_reproduces_live_window_until_owner_is_destroyed():
    obj,ref,events=owner()
    obj.destroy()
    gc.collect()
    assert ref() is not None  # Actual SDK does not clear the field.
    assert events==['stop','sentinel','cancel','unsubscribe']


def test_owned_hook_releases_window_before_plugin_unload_without_extra_destroy():
    obj,ref,events=owner()
    launcher.wrap_preview_hotkey_destroy(obj,source=TRIGGER)
    obj.destroy()
    gc.collect()
    assert ref() is None
    assert events==['stop','sentinel','cancel','unsubscribe']


def test_failed_destroy_preserves_error_and_releases_window():
    obj,ref,events=owner(broken=True)
    launcher.wrap_preview_hotkey_destroy(obj,source=TRIGGER)
    with pytest.raises(RuntimeError,match='input unsubscribe failed'):obj.destroy()
    gc.collect()
    assert ref() is None
    assert events.count('cancel')==1


def test_hook_does_not_patch_class_or_other_owner():
    obj,ref,events=owner();other,other_ref,other_events=owner()
    original=type(obj).destroy
    launcher.wrap_preview_hotkey_destroy(obj,source=TRIGGER)
    assert type(obj).destroy is original
    assert 'destroy' not in vars(other)
    other.destroy()
    assert other_ref() is not None


def test_hook_rejects_already_modified_destroy():
    obj,ref,events=owner()
    obj.destroy=lambda:None
    with pytest.raises(ValueError,match='destroy'):
        launcher.wrap_preview_hotkey_destroy(obj,source=TRIGGER)
    assert ref() is not None


def test_private_ui_release_precedes_sdk_close():
    events=[]
    class App:
        def close(self,**kw):events.append('sdk-close')
    class Launcher:
        _app=App()
        _hide_play_button_callback=_unhide_play_button_callback=None
        @property
        def app(self):return self._app
    app=launcher.ManagedPreviewApp(Launcher(),signal_api=None,
        prepare_native_shutdown=lambda:events.append('window-release-hook'))
    app.close()
    assert events==['window-release-hook','sdk-close']


def test_failed_record_cannot_hide_sdk_destroy_failure():
    obj,ref,events=owner(broken=True)
    def record(row):
        if row['event']=='owned_window_reference_cleared':raise OSError('diagnostic write failed')
    receipt=launcher.wrap_preview_hotkey_destroy(obj,source=TRIGGER,record=record)
    with pytest.raises(RuntimeError) as failure:obj.destroy()
    assert 'input unsubscribe failed' in str(failure.value)
    assert any('diagnostic write failed' in text for text in receipt['diagnostic_errors'])
    gc.collect()
    assert ref() is None


def test_source_locked_lookup_wraps_only_existing_private_extension(monkeypatch):
    import sys,types
    from go2_pvcnn.control.dh116.tabletop_preview_window_owner import prepare_loaded_window_owner
    base=TRIGGER.parent
    ext_source=base/'extension.py'
    node=next(n for n in ast.walk(ast.parse(ext_source.read_text()))
        if isinstance(n,ast.FunctionDef) and n.name=='on_shutdown')
    definition=ast.ClassDef(name='HotkeysExtension',bases=[],keywords=[],body=[node],decorator_list=[])
    namespace={'__name__':'omni.kit.hotkeys.core.extension'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition],type_ignores=[])),str(ext_source),'exec'),namespace)
    obj,ref,events=owner()
    ext=namespace['HotkeysExtension']();ext._hotkey_trigger=obj
    loader=types.ModuleType('omni.ext._impl._internal')
    loader.__file__='/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/omni/kernel/py/omni/ext/_impl/_internal.py'
    loader._extensions={'omni.kit.hotkeys.core-1.3.10':SimpleNamespace(
        _started_extensions=[(ext,'omni.kit.hotkeys.core')])}
    monkeypatch.setitem(sys.modules,loader.__name__,loader)
    sources=dict(extension_loader=dict(path=loader.__file__),hotkeys_extension=dict(path=str(ext_source)),
        hotkeys_trigger=dict(path=str(TRIGGER)))
    records=[]
    receipt=prepare_loaded_window_owner(sources,record=records.append)
    assert receipt['hook_installed'] and not receipt['native_qualification']
    obj.destroy();gc.collect()
    assert ref() is None and records[-1]['window_reference_absent'] is True

def test_record_failure_does_not_abort_real_extension_shutdown():
    obj,ref,events=owner()
    def record(row):raise OSError('disk full')
    receipt=launcher.wrap_preview_hotkey_destroy(obj,source=TRIGGER,record=record)
    obj.destroy()
    events.append('extension-remaining-fields-released')
    assert events[-1]=='extension-remaining-fields-released'
    assert ref() is None and receipt['diagnostic_errors']
    from go2_pvcnn.control.dh116.tabletop_preview_window_owner import validate_window_release_receipt
    with pytest.raises(RuntimeError,match='disk full'):
        validate_window_release_receipt(receipt)


def test_partial_launcher_prepares_window_release_before_normal_close():
    events=[]
    class App:
        def close(self,**kw):events.append('sdk-close')
    class Broken:
        def __init__(self,**kwargs):
            self._app=App()
            raise RuntimeError('startup failure')
    cls=launcher.preview_launcher_type(Broken,
        partial_stop_presentation=lambda:events.append('stream-stop'),
        partial_prepare_native_shutdown=lambda:events.append('window-release-hook'))
    with pytest.raises(RuntimeError,match='startup failure'):
        cls()
    assert events==['stream-stop','window-release-hook','sdk-close']


def test_outer_close_reports_record_failure_after_sdk_close():
    obj,ref,events=owner()
    def record(row):raise OSError('disk full')
    def prepare():
        return launcher.wrap_preview_hotkey_destroy(obj,source=TRIGGER,record=record)
    class App:
        def close(self,**kw):
            obj.destroy()
            events.append('sdk-close-returned')
    class Launcher:
        _app=App()
        _hide_play_button_callback=_unhide_play_button_callback=None
        @property
        def app(self):return self._app
    app=launcher.ManagedPreviewApp(Launcher(),signal_api=None,prepare_native_shutdown=prepare)
    with pytest.raises(RuntimeError,match='disk full'):app.close()
    assert events[-1]=='sdk-close-returned'
    assert ref() is None
