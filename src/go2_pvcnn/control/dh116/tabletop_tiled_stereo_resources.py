"""Explicit ownership for the installed TiledCamera API; no implicit GC cleanup."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading
import uuid

RIG_PATH="/World/DH116TiledStereo"
MARKER="dh116TiledStereoOwner"
_LOCK=threading.RLock()


def verify_tiled_sources(root):
    data=json.loads((Path(root)/"configs/dh116_execution/tiled_camera_api_lock_v1.json").read_text())
    if data.get("version")!="dh116-installed-tiled-camera-api-v1":
        raise ValueError("unsupported tiled API contract")
    sources=data.get("sources")
    if not isinstance(sources,dict) or set(sources)!={"tiled","camera","sensor","replicator_create","xform_view","fabric_utils"}:
        raise ValueError("complete tiled API source contract required")
    for name,item in sources.items():
        p=Path(item["path"])
        if not p.is_absolute() or hashlib.sha256(p.read_bytes()).hexdigest()!=item["sha256"]:
            raise ValueError("installed tiled API source differs: "+name)
    return sources


class TiledResources:
    def __init__(self,*,rig_path=RIG_PATH):
        self.rig_path=rig_path
        self.stage=self.sensor=self.product=None
        self.token=None;self.extra=[];self.closed=False;self.error=None

    def define_rig(self,stage,define):
        if stage.GetPrimAtPath(self.rig_path).IsValid():raise RuntimeError("tiled stereo rig already exists")
        define(stage,self.rig_path)
        token=uuid.uuid4().hex
        stage.GetPrimAtPath(self.rig_path).SetCustomDataByKey(MARKER,token)
        self.stage,self.token=stage,token

    @contextmanager
    def product_scope(self,api):
        """Track the real HydraTexture which upstream keeps only as a local."""
        with _LOCK:
            original=api.render_product_tiled
            def tracked(*args,**kwargs):
                if self.product is not None:raise RuntimeError("multiple tiled products are not permitted")
                kwargs=dict(kwargs,force_new=True)
                result=original(*args,**kwargs)
                self.product=result
                if not isinstance(getattr(result,"path",None),str) or not callable(getattr(result,"destroy",None)):
                    raise RuntimeError("unsupported tiled HydraTexture handle")
                return result
            api.render_product_tiled=tracked
            try:yield
            finally:
                if api.render_product_tiled is not tracked:
                    raise RuntimeError("tiled product factory changed concurrently")
                api.render_product_tiled=original

    def attach(self,annotator,path):
        self.extra.append((annotator,path))
        annotator.attach([path])

    def close(self):
        if self.closed:
            if self.error:raise RuntimeError(self.error)
            return
        self.closed=True;errors=[]
        sensor=self.sensor
        if sensor is not None:
            try:sensor._clear_callbacks()
            except Exception as e:errors.append("callbacks: "+repr(e))
        for annotator,path in reversed(self.extra):
            try:annotator.detach([path])
            except Exception as e:errors.append("extra annotator: "+repr(e))
        self.extra.clear()
        if sensor is not None:
            paths=getattr(sensor,"_render_product_paths",None)
            annotators=getattr(sensor,"_annotators",None)
            if not isinstance(paths,list) or not isinstance(annotators,dict):
                errors.append("unknown tiled sensor resource layout")
            else:
                owned_paths=[] if self.product is None else [self.product.path]
                valid_paths=(paths==owned_paths or not paths and not annotators)
                if not valid_paths:
                    errors.append("tiled product ownership path mismatch")
                else:
                    for name,annotator in list(annotators.items()):
                        try:annotator.detach(paths)
                        except Exception as e:errors.append(name+" detach: "+repr(e))
                    # Explicit owner is authoritative; never retry from __del__.
                    annotators.clear()
        if self.product is not None and not errors:
            try:self.product.destroy()
            except Exception as e:errors.append("product destroy: "+repr(e))
        if self.stage is not None and not errors:
            try:
                prim=self.stage.GetPrimAtPath(self.rig_path)
                if not prim.IsValid() or prim.GetCustomDataByKey(MARKER)!=self.token:
                    raise RuntimeError("tiled rig ownership marker changed")
                if not self.stage.RemovePrim(self.rig_path):raise RuntimeError("tiled rig removal failed")
            except Exception as e:errors.append("rig removal: "+repr(e))
        if errors:
            self.error="; ".join(errors)
            raise RuntimeError(self.error)
        if sensor is not None:
            sensor._view=None
            sensor._sensor_prims=[]
            sensor._render_product_paths=[]
        self.sensor=self.product=self.stage=None
