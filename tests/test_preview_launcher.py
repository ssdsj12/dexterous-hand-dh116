from types import SimpleNamespace
import pytest
from go2_pvcnn.control.dh116.tabletop_preview_launcher import preview_launcher_type

class Base:
    def __init__(self, **kwargs):
        self._livestream=2
        self._headless=True
        self._app=kwargs.get("app")
        self._hide_stop_button()
    @property
    def app(self):return self._app

def toolbar(stop=None,play=None):
    group=SimpleNamespace(_stop_button=stop,_play_button=play)
    return SimpleNamespace(_builtin_tools=SimpleNamespace(_play_button_group=group)),group

def test_unbuilt_stop_button_does_not_abort_webrtc_start():
    ui,group=toolbar()
    launcher=preview_launcher_type(Base,toolbar_getter=lambda:ui)(app=object())
    assert launcher.app is not None
    assert group._stop_button is None

def test_unbuilt_play_button_does_not_abort_stop_event():
    ui,_=toolbar()
    launcher=preview_launcher_type(Base,toolbar_getter=lambda:ui)()
    launcher._hide_play_button(True)

def test_existing_stop_button_is_still_disabled():
    button=SimpleNamespace(visible=True,enabled=True)
    ui,group=toolbar(stop=button)
    preview_launcher_type(Base,toolbar_getter=lambda:ui)()
    assert button.visible is False and button.enabled is False
    assert group._stop_button is None

def test_existing_play_button_obeys_hide_flag():
    button=SimpleNamespace(visible=True,enabled=True)
    ui,_=toolbar(play=button)
    launcher=preview_launcher_type(Base,toolbar_getter=lambda:ui)()
    launcher._hide_play_button(True)
    assert not button.visible and not button.enabled
    launcher._hide_play_button(False)
    assert button.visible and button.enabled

def test_partially_created_app_gets_one_normal_close_attempt():
    calls=[]
    class Broken(Base):
        def __init__(self,**kwargs):
            self._app=SimpleNamespace(close=lambda **kw:calls.append(kw))
            raise RuntimeError("post-start error")
    cls=preview_launcher_type(Broken,toolbar_getter=lambda:None)
    with pytest.raises(RuntimeError,match="post-start error"):
        cls()
    assert calls == [dict(wait_for_replicator=False)]

def test_missing_ui_does_not_create_fake_buttons():
    cls=preview_launcher_type(Base,toolbar_getter=lambda:None)
    cls()._hide_play_button(True)

@pytest.mark.parametrize("name",["_hide_stop_button","_hide_play_button"])
def test_installed_hook_reproduces_none_button_failure_and_adapter_handles_it(name,monkeypatch):
    import ast
    from pathlib import Path
    import sys,types
    sdk=Path("/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py")
    if not sdk.is_file():pytest.skip("installed SDK source unavailable")
    tree=ast.parse(sdk.read_text())
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    ui,_=toolbar()
    omni=types.ModuleType("omni")
    omni.kit=types.ModuleType("omni.kit")
    omni.kit.widget=types.ModuleType("omni.kit.widget")
    omni.kit.widget.toolbar=types.ModuleType("omni.kit.widget.toolbar")
    omni.kit.widget.toolbar.get_instance=lambda:ui
    for module in (omni,omni.kit,omni.kit.widget,omni.kit.widget.toolbar):
        monkeypatch.setitem(sys.modules,module.__name__,module)
    namespace={}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(sdk),"exec"),namespace)
    subject=SimpleNamespace(_livestream=2,_headless=True)
    args=() if name=="_hide_stop_button" else (True,)
    with pytest.raises(AttributeError):
        namespace[name](subject,*args)
    adapted=preview_launcher_type(Base,toolbar_getter=lambda:ui)()
    getattr(adapted,name)(*args)
