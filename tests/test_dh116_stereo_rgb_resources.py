import pytest

from go2_pvcnn.control.dh116.tabletop_stereo_rgb_resources import RGBResources


class Prim:
    def __init__(self, stage): self.stage = stage
    def IsValid(self): return self.stage.exists
    def SetCustomDataByKey(self, key, value): self.stage.marker = value
    def GetCustomDataByKey(self, key): return self.stage.marker


class Stage:
    def __init__(self, exists=False): self.exists = exists; self.removed = []
    def GetPrimAtPath(self, path): return Prim(self)
    def RemovePrim(self, path): self.removed.append(path); self.exists = False; return True


class Annotator:
    def __init__(self, name, log, fail=False): self.name, self.log, self.fail = name, log, fail
    def detach(self, paths):
        self.log.append(("detach", self.name, paths))
        if self.fail: raise RuntimeError("detach failed")


class Product:
    def __init__(self, log, fail=False): self.path, self.log, self.fail = "/product", log, fail
    def destroy(self):
        self.log.append(("destroy",))
        if self.fail: raise RuntimeError("destroy failed")


class View:
    def __init__(self, log):
        self._tiled_render_product = None
        self._render_product = Product(log)
        self._render_product_path = "/product"
        self._annotators = {"rgba": Annotator("rgb", log)}
    def get_render_product_path(self): return self._render_product_path


def test_cleanup_owned_resources_in_order_and_once():
    stage, log = Stage(), []
    owner = RGBResources()
    owner.define_rig(stage, lambda s, p: setattr(s, "exists", True))
    owner.bind_view(View(log))
    owner.attach(Annotator("ref", log), "/product", lambda: None)
    owner.attach(Annotator("times", log), "/product", lambda: None)
    owner.close()
    owner.close()
    assert [x[0:2] for x in log] == [("detach", "times"), ("detach", "ref"), ("detach", "rgb"), ("destroy",)]
    assert stage.removed == ["/World/DH116StereoRGB"]


def test_existing_prim_is_never_removed():
    stage = Stage(exists=True)
    owner = RGBResources()
    with pytest.raises(RuntimeError): owner.define_rig(stage, lambda *_: None)
    owner.close()
    assert stage.removed == []


def test_partial_failure_cleans_only_acquired_resources():
    stage, log = Stage(), []
    owner = RGBResources()
    owner.define_rig(stage, lambda s, p: setattr(s, "exists", True))
    owner.bind_view(View(log))
    owner.attach(Annotator("ref", log), "/product", lambda: None)
    def fails(): raise RuntimeError("attach failed")
    with pytest.raises(RuntimeError): owner.attach(Annotator("times", log), "/product", fails)
    owner.close()
    assert [x[0:2] for x in log] == [("detach", "times"), ("detach", "ref"), ("detach", "rgb"), ("destroy",)]
    assert stage.removed == ["/World/DH116StereoRGB"]


def test_detach_failure_attempts_other_detaches_but_not_destroy_or_remove():
    stage, log = Stage(), []
    owner = RGBResources()
    owner.define_rig(stage, lambda s, p: setattr(s, "exists", True))
    owner.bind_view(View(log))
    owner.attach(Annotator("ref", log, fail=True), "/product", lambda: None)
    with pytest.raises(RuntimeError): owner.close()
    with pytest.raises(RuntimeError): owner.close()
    assert [x[0:2] for x in log] == [("detach", "ref"), ("detach", "rgb")]
    assert stage.removed == []


def test_unknown_camera_layout_is_never_destroyed():
    stage, log = Stage(), []
    owner = RGBResources()
    owner.define_rig(stage, lambda s, p: setattr(s, "exists", True))
    view = View(log)
    view._tiled_render_product = object()
    with pytest.raises(RuntimeError): owner.bind_view(view)
    with pytest.raises(RuntimeError): owner.close()
    assert not log and not stage.removed



def test_replaced_rig_is_not_removed():
    stage = Stage()
    owner = RGBResources()
    owner.define_rig(stage, lambda s, p: setattr(s, "exists", True))
    stage.marker = "another-owner"
    with pytest.raises(RuntimeError, match="ownership"):
        owner.close()
    assert stage.removed == []


def test_destroy_failure_is_reported_without_retry_or_rig_removal():
    stage, log = Stage(), []
    owner = RGBResources()
    owner.define_rig(stage, lambda s, p: setattr(s, "exists", True))
    view = View(log)
    view._render_product.fail = True
    owner.bind_view(view)
    with pytest.raises(RuntimeError, match="destroy"): owner.close()
    with pytest.raises(RuntimeError, match="destroy"): owner.close()
    assert [x[0] for x in log] == ["detach", "destroy"]
    assert stage.removed == []



def test_owned_reader_constructor_cleans_owned_rig_after_partial_failure(monkeypatch):
    import sys
    import types
    import torch
    import go2_pvcnn.control.dh116.tabletop_stereo_rgb_owned_native as native
    stage = Stage()
    kit = types.SimpleNamespace(get_app=lambda: types.SimpleNamespace(
        get_extension_manager=lambda: types.SimpleNamespace(set_extension_enabled_immediate=lambda *a: None)))
    def define(s, p): s.exists = True
    def fail_camera(*args): raise RuntimeError("camera authoring failed")
    usdgeom = types.SimpleNamespace(Xform=types.SimpleNamespace(Define=define),
                                    Camera=types.SimpleNamespace(Define=fail_camera))
    modules = {
        "omni.kit.app": kit,
        "isaacsim.sensors.camera": types.SimpleNamespace(CameraView=type("Base", (), {})),
        "omni.usd": types.SimpleNamespace(get_context=lambda: types.SimpleNamespace(get_stage=lambda: stage)),
        "omni.replicator.core": types.SimpleNamespace(),
        "isaacsim.core.nodes.bindings._isaacsim_core_nodes": types.SimpleNamespace(),
    }
    original = native.importlib.import_module
    monkeypatch.setattr(native.importlib, "import_module", lambda name: modules[name] if name in modules else original(name))
    monkeypatch.setitem(sys.modules, "pxr", types.SimpleNamespace(Gf=None, Usd=None, UsdGeom=usdgeom))
    k = torch.eye(3)
    t = torch.eye(4)
    calibration = types.SimpleNamespace(width=640, height=480, K_left=k, K_right=k,
                                        T_world_left=t, T_world_right=t)
    with pytest.raises(RuntimeError, match="camera authoring failed"):
        native.OwnedNativeRGBPairReader(None, [[0., 0., 0.]], calibration, "cpu")
    assert stage.removed == ["/World/DH116StereoRGB"]
