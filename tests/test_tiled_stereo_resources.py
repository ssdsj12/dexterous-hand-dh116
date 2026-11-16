import importlib
from types import SimpleNamespace
from pathlib import Path
import pytest

def resources():
    module=importlib.import_module("go2_pvcnn.control.dh116.tabletop_tiled_stereo_resources")
    return module,module.TiledResources()

def test_product_factory_tracks_actual_owned_handle_and_restores_on_error():
    module,owner=resources();seen=[]
    handle=SimpleNamespace(path="/Render/own",destroy=lambda:seen.append("destroy"))
    def create(*a,**kw):seen.append(kw);return handle
    api=SimpleNamespace(render_product_tiled=create)
    with pytest.raises(RuntimeError,match="constructor"):
        with owner.product_scope(api):
            assert api.render_product_tiled(["camera"],(320,240)) is handle
            raise RuntimeError("constructor")
    assert api.render_product_tiled is create
    assert owner.product is handle and seen[0]["force_new"] is True

def rig(owner):
    events=[]
    prim=SimpleNamespace(IsValid=lambda:True,GetCustomDataByKey=lambda key:owner.token)
    stage=SimpleNamespace(GetPrimAtPath=lambda path:prim,RemovePrim=lambda path:events.append("rig") or True)
    owner.stage=stage;owner.token="own"
    def detach(paths):events.append("detach")
    owner.sensor=SimpleNamespace(_clear_callbacks=lambda:events.append("callbacks"),
        _annotators={"rgba":SimpleNamespace(detach=detach)},
        _render_product_paths=["/Render/own"],_view=object(),_sensor_prims=[object()])
    owner.product=SimpleNamespace(path="/Render/own",destroy=lambda:events.append("destroy"))
    return events

def test_callbacks_annotators_product_and_rig_close_in_order_once():
    module,owner=resources();events=rig(owner)
    owner.close();owner.close()
    assert events==["callbacks","detach","destroy","rig"]
    assert owner.sensor is owner.product is owner.stage is None

def test_failed_callback_preserves_error_without_destroying_active_product():
    module,owner=resources();events=rig(owner)
    def fail():raise RuntimeError("callback release failed")
    owner.sensor._clear_callbacks=fail
    with pytest.raises(RuntimeError,match="callback release failed"):owner.close()
    with pytest.raises(RuntimeError,match="callback release failed"):owner.close()
    assert "destroy" not in events and "rig" not in events
    assert events.count("detach")==1

def test_foreign_rig_is_not_removed():
    module,owner=resources();events=rig(owner)
    owner.stage.GetPrimAtPath=lambda p:SimpleNamespace(IsValid=lambda:True,GetCustomDataByKey=lambda key:"foreign")
    with pytest.raises(RuntimeError,match="ownership"):owner.close()
    assert "rig" not in events

def test_product_path_mismatch_does_not_destroy_unverified_handle():
    module,owner=resources();events=rig(owner)
    owner.sensor._render_product_paths=["/Render/not-ours"]
    with pytest.raises(RuntimeError,match="product"):owner.close()
    assert "destroy" not in events and "rig" not in events

def test_actual_installed_sources_match_independent_api_contract():
    module,owner=resources()
    root=Path(__file__).resolve().parents[1]
    sources=module.verify_tiled_sources(root)
    assert set(sources)=={"tiled","camera","sensor","replicator_create","xform_view","fabric_utils"}
    assert all(len(v["sha256"])==64 for v in sources.values())

def test_foreign_product_paths_are_never_used_for_detach():
    module,owner=resources();events=rig(owner)
    owner.sensor._render_product_paths=["/Render/foreign"]
    with pytest.raises(RuntimeError,match="product"):owner.close()
    assert events==["callbacks"]
