"""CPU checks of real adapter methods; no simulation or CUDA qualification."""
from contextlib import nullcontext
from types import SimpleNamespace
import importlib
import pytest
import torch
from test_dh116_stereo_rgb_bridge import calibration

def reader():
    m=importlib.import_module("go2_pvcnn.control.dh116.tabletop_tiled_stereo_native")
    obj=object.__new__(m.TiledNativeStereoReader)
    obj.calibration=calibration();obj.device=torch.device("cpu")
    obj.n=1;obj._closed=False;obj._close_error=None
    obj.last_clock_diagnostic={"reference":"original","times":"original"}
    clock=dict(frame_id=12,simulation_time=.4,render_frame_number=15)
    obj.read_clock=lambda:dict(clock)
    calls=[]
    obj.sensor=SimpleNamespace(update=lambda dt,**kw:calls.append((dt,kw)),
        data=SimpleNamespace(output={"rgb":torch.full((1,480,640,3),128,dtype=torch.uint8)}))
    obj._readback=lambda:(obj.calibration.K_left[None],obj.calibration.T_world_left[None])
    return obj,clock,calls

def test_official_sensor_update_and_rgb_conversion_are_exercised_on_cpu(monkeypatch):
    import go2_pvcnn.control.dh116.tabletop_preview_warp_readback as abi
    monkeypatch.setattr(abi,"isaac45_rgb_readback_scope",nullcontext)
    obj,clock,calls=reader()
    packet=obj.read()
    assert calls==[(0,{"force_recompute":True})]
    assert packet["rgb"].shape==(1,3,480,640)
    torch.testing.assert_close(packet["rgb"],torch.full((1,3,480,640),128/255.))
    assert packet["raw_clock"]==clock
    # Calling an adapter method in a CPU fixture cannot make it native evidence.
    assert packet["native_evidence"] is False

def test_clock_change_during_official_update_is_rejected(monkeypatch):
    import go2_pvcnn.control.dh116.tabletop_preview_warp_readback as abi
    monkeypatch.setattr(abi,"isaac45_rgb_readback_scope",nullcontext)
    obj,clock,calls=reader()
    obj.sensor.update=lambda *a,**kw:clock.update(render_frame_number=16)
    with pytest.raises(ValueError,match="clock"):obj.read()

def test_wrong_native_rgb_shape_is_rejected_before_return(monkeypatch):
    import go2_pvcnn.control.dh116.tabletop_preview_warp_readback as abi
    monkeypatch.setattr(abi,"isaac45_rgb_readback_scope",nullcontext)
    obj,clock,calls=reader()
    obj.sensor.data.output["rgb"]=torch.zeros((1,240,320,3),dtype=torch.uint8)
    with pytest.raises(ValueError,match="RGB"):obj.read()

def test_private_camera_prims_have_standard_ops_before_tiled_initialization():
    from pxr import Usd,UsdGeom,Gf
    from go2_pvcnn.control.dh116 import tabletop_tiled_stereo_native as native
    stage=Usd.Stage.CreateInMemory();camera=UsdGeom.Camera.Define(stage,"/World/Camera")
    native.author_standard_camera_ops(camera,UsdGeom,Gf)
    assert [op.GetOpName() for op in camera.GetOrderedXformOps()]==[
        "xformOp:translate","xformOp:orient","xformOp:scale"]

def test_only_owned_renderer_view_gets_fabric_to_usd_sync_enabled():
    from go2_pvcnn.control.dh116 import tabletop_tiled_stereo_native as native
    view=SimpleNamespace(_use_fabric=True,_sync_usd_on_fabric_write=False)
    native.prepare_renderer_view(view)
    assert view._sync_usd_on_fabric_write is True

def test_select_uses_float32_inputs_required_by_actual_fabric_kernel(monkeypatch):
    import sys
    obj,clock,calls=reader()
    obj.origins=torch.zeros((1,3),dtype=torch.float64)
    seen={}
    obj.sensor.set_intrinsic_matrices=lambda k,**kw:seen.update(K=k)
    obj.sensor.set_world_poses=lambda **kw:seen.update(**kw)
    def quat(matrix):
        seen["rotation_matrix"]=matrix
        return matrix.new_tensor([1.,0.,0.,0.])
    monkeypatch.setitem(sys.modules,"isaaclab.utils.math",SimpleNamespace(quat_from_matrix=quat))
    obj.select("right")
    assert seen["positions"].dtype==torch.float32
    assert seen["orientations"].dtype==torch.float32
    assert seen["rotation_matrix"].dtype==torch.float32
    assert seen["K"].dtype==torch.float32

def test_usd_pose_crosscheck_rejects_fabric_only_motion():
    from pxr import Usd,UsdGeom,Gf
    from go2_pvcnn.control.dh116 import tabletop_tiled_stereo_native as native
    stage=Usd.Stage.CreateInMemory();camera=UsdGeom.Camera.Define(stage,"/World/Camera")
    native.author_standard_camera_ops(camera,UsdGeom,Gf)
    # ROS identity camera requires a 180 degree rotation around USD X.
    camera.GetPrim().GetAttribute("xformOp:orient").Set(Gf.Quatd(0.,Gf.Vec3d(1.,0.,0.)))
    actual=torch.eye(4,dtype=torch.float64)[None]
    native.validate_usd_camera_pose(stage,"/World/Camera",actual,Usd,UsdGeom)
    camera.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(.1,0.,0.))
    with pytest.raises(ValueError,match="USD"):
        native.validate_usd_camera_pose(stage,"/World/Camera",actual,Usd,UsdGeom)
