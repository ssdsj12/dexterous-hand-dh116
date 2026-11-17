"""Preview resolution wiring tests; SDK readers are replaced, no Isaac launch."""
import copy
import json
import math
from pathlib import Path
import runpy
import pytest
import torch
from go2_pvcnn.control.dh116 import tabletop_preview_contract as preview
from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration

ROOT=Path(__file__).resolve().parents[1]
SIZES=[(224,96),(320,240),(640,360),(640,480),(512,384)]

def sandbox(tmp_path):
    for part in ("src","scripts","configs","docs/contracts"):
        (tmp_path/part).mkdir(parents=True)
    kit=tmp_path/"render.kit";kit.write_text("kit")
    return kit

@pytest.mark.parametrize("size",SIZES)
def test_freeze_and_verify_bind_both_camera_dimensions(tmp_path,size):
    kit=sandbox(tmp_path)
    c=preview.freeze(tmp_path,kit,width=size[0],height=size[1])
    assert c["capture_resolution"]==dict(width=size[0],height=size[1])
    assert preview.verify(tmp_path,c)==c
    assert c["budget"]==preview.BUDGET
    altered=copy.deepcopy(c);altered["capture_resolution"]["width"]=0
    with pytest.raises(ValueError):preview.verify(tmp_path,altered)

def test_default_is_vga_and_resolution_changes_frozen_identity(tmp_path):
    kit=sandbox(tmp_path)
    default=preview.freeze(tmp_path,kit)
    assert default["capture_resolution"]==dict(width=640,height=480)
    small=preview.freeze(tmp_path,kit,width=224,height=96)
    assert json.dumps(small,sort_keys=True)!=json.dumps(default,sort_keys=True)

@pytest.mark.parametrize("size",[(0,480),(640,-1),(True,480),(640.,480),(4096,2160)])
def test_invalid_resolution_rejected_before_native_setup(tmp_path,size):
    kit=sandbox(tmp_path)
    with pytest.raises(ValueError):preview.freeze(tmp_path,kit,width=size[0],height=size[1])

@pytest.mark.parametrize("size",SIZES)
def test_actual_preview_factory_reauthors_both_lenses_and_preserves_rig(monkeypatch,size):
    import go2_pvcnn.control.dh116.tabletop_stereo_rgb_owned_native as native
    seen={}
    class Reader:
        def __init__(self,sim,origins,c,device,**kwargs):
            seen.update(calibration=c,origins=origins,device=device,kwargs=kwargs)
        def close(self):pass
    class Setting:
        def get(self):return False
        def set(self,v):pass
    monkeypatch.setattr(native,"OwnedNativeRGBPairReader",Reader)
    baseline=StereoCalibration.from_json(ROOT/preview.CAMERA_CONFIG)
    c=dict(camera_config=preview.CAMERA_CONFIG,capture_resolution=dict(width=size[0],height=size[1]))
    bridge=preview.create_preview_camera(ROOT,object(),c,lambda:None,render_setting=Setting())
    cal=bridge.calibration
    assert (cal.width,cal.height)==size
    expected=size[0]/(2*math.tan(math.radians(37.5)))
    assert float(cal.K_left[0,0])==pytest.approx(expected)
    assert float(cal.K_left[1,1])==pytest.approx(expected)
    assert float(cal.K_left[0,2])==size[0]/2
    assert float(cal.K_left[1,2])==size[1]/2
    torch.testing.assert_close(cal.K_left,cal.K_right,rtol=0,atol=0)
    torch.testing.assert_close(cal.T_world_left,baseline.T_world_left,rtol=0,atol=0)
    torch.testing.assert_close(cal.T_world_right,baseline.T_world_right,rtol=0,atol=0)
    assert cal.baseline_m==baseline.baseline_m
    assert seen["device"]=="cuda:0"
    assert bridge.max_render_attempts==preview.BUDGET["max_camera_renders_per_frame"]
    bridge.close()

def test_resolution_matrix_preserves_four_profiles_without_native_claims():
    data=json.loads((ROOT/"configs/dh116_execution/preview_resolution_matrix_v1.json").read_text())
    assert {(v["width"],v["height"]) for v in data["profiles"]}==set(SIZES[:4])
    assert data["default_accuracy_resolution"]==dict(width=640,height=480)
    assert data["custom_resolution_allowed"] is True
    assert data["native_execution_authorized"] is False
    assert data["physics_hz"]==480 and data["policy_hz"]==60

def test_freeze_cli_accepts_custom_size_without_launch(tmp_path,monkeypatch):
    m=runpy.run_path(str(ROOT/"scripts/preview_dh116_physical_webrtc.py"))
    kit=tmp_path/"render.kit";kit.write_text("kit")
    monkeypatch.setitem(m["main"].__globals__,"EXPERIENCE",kit)
    out=tmp_path/"frozen.json"
    assert m["main"](["--freeze",str(out),"--resolution","512x384"])==0
    assert json.loads(out.read_text())["capture_resolution"]==dict(width=512,height=384)

def test_runtime_cannot_override_authorized_resolution(tmp_path,capsys):
    m=runpy.run_path(str(ROOT/"scripts/preview_dh116_physical_webrtc.py"))
    with pytest.raises(SystemExit) as caught:
        m["main"](["--execute-native","--resolution","320x240"])
    assert caught.value.code==2
    assert "resolution is freeze-only" in capsys.readouterr().err
