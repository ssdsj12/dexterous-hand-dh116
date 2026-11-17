import copy
import json
from pathlib import Path
import runpy
import pytest
from go2_pvcnn.control.dh116 import tabletop_preview_contract as preview
from test_dh116_stereo_rgb_bridge import RenderSetting

ROOT=Path(__file__).resolve().parents[1]

def test_tiled_freeze_binds_backend_and_actual_installed_api_sources(tmp_path):
    kit=tmp_path/"render.kit";kit.write_text("kit")
    contract=preview.freeze(ROOT,kit,camera_backend="tiled_step",width=320,height=240)
    assert contract["camera_backend"]=="tiled_step"
    assert set(contract["camera_api_sources"])=={"tiled","camera","sensor","replicator_create","xform_view","fabric_utils"}
    assert preview.verify(ROOT,contract)==contract
    bad=copy.deepcopy(contract);bad["camera_api_sources"]["sensor"]["sha256"]="0"*64
    with pytest.raises(ValueError,match="API"):preview.verify(ROOT,bad)

def test_unknown_backend_is_rejected_before_freezing(tmp_path):
    kit=tmp_path/"render.kit";kit.write_text("kit")
    with pytest.raises(ValueError):preview.freeze(ROOT,kit,camera_backend="arbitrary")

@pytest.mark.parametrize("size",[(224,96),(320,240),(640,360),(640,480),(512,384)])
def test_actual_preview_dispatches_to_tiled_reader_at_frozen_resolution(monkeypatch,size):
    import go2_pvcnn.control.dh116.tabletop_tiled_stereo_native as native
    from go2_pvcnn.control.dh116.tabletop_tiled_stereo_capture import TiledStereoCapture
    seen={}
    class Reader:
        def __init__(self,sim,origins,calibration,device,**kw):seen.update(c=calibration,device=device)
        def select(self,side):pass
        def read_clock(self):return None
        def read_pair(self):return None
        def close(self):pass
    monkeypatch.setattr(native,"TiledNativeStereoReader",Reader)
    contract=dict(camera_config=preview.CAMERA_CONFIG,capture_resolution=dict(width=size[0],height=size[1]),camera_backend="tiled_step")
    camera=preview.create_preview_camera(ROOT,object(),contract,lambda:None,render_setting=RenderSetting())
    assert isinstance(camera,TiledStereoCapture)
    assert (seen["c"].width,seen["c"].height)==size
    assert camera.max_render_attempts*2==preview.BUDGET["max_camera_renders_per_frame"]
    camera.close()

def test_cli_can_freeze_tiled_backend_but_cannot_override_during_execute(tmp_path,monkeypatch,capsys):
    m=runpy.run_path(str(ROOT/"scripts/preview_dh116_physical_webrtc.py"))
    kit=tmp_path/"render.kit";kit.write_text("kit")
    monkeypatch.setitem(m["main"].__globals__,"EXPERIENCE",kit)
    out=tmp_path/"contract.json"
    assert m["main"](["--freeze",str(out),"--camera-backend","tiled_step","--resolution","320x240"])==0
    assert json.loads(out.read_text())["camera_backend"]=="tiled_step"
    with pytest.raises(SystemExit):
        m["main"](["--execute-native","--camera-backend","tiled_step"])
    assert "camera backend is freeze-only" in capsys.readouterr().err
