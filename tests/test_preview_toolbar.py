from pathlib import Path
import runpy
from go2_pvcnn.control.dh116.tabletop_preview_contract import freeze, verify

def test_webrtc_contract_enables_toolbar_before_app_launcher_hook(tmp_path):
    for part in ("src","scripts","configs","docs/contracts"):
        (tmp_path/part).mkdir(parents=True)
    kit=tmp_path/"render.kit";kit.write_text("minimal rendering")
    value=freeze(tmp_path,kit,physical_gpu_index=0)
    assert value["launcher"]["kit_args"] == "--enable omni.kit.widget.toolbar"
    verify(tmp_path,value)

def test_launcher_receives_frozen_toolbar_before_post_start_hook(tmp_path):
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_preview_contract import launch_preview_app
    for part in ("src","scripts","configs","docs/contracts"):
        (tmp_path/part).mkdir(parents=True)
    kit=tmp_path/"render.kit";kit.write_text("minimal rendering")
    contract=freeze(tmp_path,kit,physical_gpu_index=0)
    app=object();calls=[]
    def installed_launcher_contract(**kwargs):
        calls.append(kwargs)
        if kwargs.get("kit_args") != "--enable omni.kit.widget.toolbar":
            raise ModuleNotFoundError("omni.kit.widget.toolbar")
        return SimpleNamespace(app=app)
    assert launch_preview_app(installed_launcher_contract,contract) is app
    assert calls[0]["livestream"] == 2
    assert calls[0]["enable_cameras"] is True
    assert calls[0]["experience"] == str(kit)
    assert calls[0]["fast_shutdown"] is False
