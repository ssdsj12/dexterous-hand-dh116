"""Owned, opt-in adapter to the audited installed IsaacLab TiledCamera.

No SDK file is patched. One physical environment is supported by this preview
adapter; parallel Teacher/Student collection is not qualified by this code.
"""
import importlib
import inspect
from pathlib import Path
import torch
import numpy as np
from .tabletop_tiled_stereo_resources import TiledResources,RIG_PATH,verify_tiled_sources
from .tabletop_stereo_rgb_owned_native import OwnedNativeRGBPairReader


def ensure_clock_annotators(manager,registry):
    """Activate the installed time-node owner before requesting its annotator.

    Importing bindings does not run the extension's on_startup registration.
    An existing enabled extension is never restarted or disabled by this owner.
    Missing registration remains an error, never a synthetic timestamp.
    """
    extension="isaacsim.core.nodes"
    if not manager.is_extension_enabled(extension):
        manager.set_extension_enabled_immediate(extension,True)
    if not manager.is_extension_enabled(extension):
        raise RuntimeError("installed clock extension failed to enable: "+extension)
    registered=set(registry.get_registered_annotators())
    # ReferenceTime is hidden in the installed public registry listing.
    # Its real get_annotator call below remains authoritative and fail-closed.
    missing={"IsaacReadTimes"}-registered
    if missing:
        raise RuntimeError("installed clock annotators unavailable: "+", ".join(sorted(missing)))


def author_standard_camera_ops(camera,UsdGeom,Gf):
    """Prepare only a new private prim for the installed XformPrimView."""
    if camera.GetOrderedXformOps():
        raise ValueError("new bare camera required; existing transforms are not overwritten")
    precision=UsdGeom.XformOp.PrecisionDouble
    camera.AddTranslateOp(precision).Set(Gf.Vec3d(0.,0.,0.))
    camera.AddOrientOp(precision).Set(Gf.Quatd(1.))
    camera.AddScaleOp(precision).Set(Gf.Vec3d(1.,1.,1.))


def prepare_renderer_view(view):
    """Use the installed USD pose route for this owned Isaac4.5 camera only.

    The installed Fabric route calls usdrt.hierarchy during the first buffer
    read; that attribute was absent in the actual preview process. No global
    Fabric/physics setting is changed. RTX image rendering stays on CUDA.
    """
    if (type(getattr(view,"_use_fabric",None)) is not bool
            or type(getattr(view,"_sync_usd_on_fabric_write",None)) is not bool):
        raise ValueError("unsupported installed renderer view layout")
    view._use_fabric=False
    view._sync_usd_on_fabric_write=True


def validate_usd_camera_pose(stage,path,transform,Usd,UsdGeom):
    """Cross-check independently authored USD, not only the Fabric cache."""
    prim=stage.GetPrimAtPath(path)
    if not prim.IsValid():raise ValueError("owned USD camera missing")
    usd=np.asarray(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)).T
    optical=usd@np.diag([1.,-1.,-1.,1.])
    actual=torch.as_tensor(transform,dtype=torch.float64).detach().cpu()
    expected=torch.as_tensor(optical,dtype=torch.float64)[None]
    if actual.shape!=expected.shape or not torch.allclose(actual,expected,atol=1e-5,rtol=0):
        raise ValueError("USD/camera pose mismatch")


class TiledNativeStereoReader:
    pose_backend="usd_camera_prim_isaac45"

    def __init__(self,sim,origins,calibration,device,*,max_readback_bytes=1024**3):
        try:
            self.resources=TiledResources();self.sensor=None;self.sim=sim
            self.device=torch.device(device);self.calibration=calibration
            self._closed=False;self._close_error=None;self._native_initialized=False
            self._last_valid_clock=None
            self.reference=self.times=self.core=None
            try:
                if self.device.type!="cuda":raise ValueError("native tiled preview requires CUDA")
                origin=torch.as_tensor(origins,dtype=torch.float64).cpu()
                if origin.shape!=(1,3) or not torch.isfinite(origin).all():
                    raise ValueError("this independent native preview requires one finite environment origin")
                if type(max_readback_bytes) is not int or max_readback_bytes<1:
                    raise ValueError("positive RGB readback budget required")
                estimate=2*calibration.width*calibration.height*16
                if estimate>max_readback_bytes:raise ValueError("native RGB readback exceeds resource budget")
                self.origins=origin;self.n=1
                root=Path(__file__).resolve().parents[4]
                sources=verify_tiled_sources(root)
                import omni.replicator.core as rep
                import omni.usd
                import omni.kit.app
                ensure_clock_annotators(omni.kit.app.get_app().get_extension_manager(),rep.AnnotatorRegistry)
                from pxr import Gf,UsdGeom
                from isaaclab.sensors import TiledCamera,TiledCameraCfg
                if Path(inspect.getfile(TiledCamera)).resolve()!=Path(sources["tiled"]["path"]).resolve():
                    raise ValueError("imported TiledCamera differs from audited installation")
                self.stage=omni.usd.get_context().get_stage()
                self.resources.define_rig(self.stage,UsdGeom.Xform.Define)
                UsdGeom.Xform.Define(self.stage,RIG_PATH+"/env_0")
                path=RIG_PATH+"/env_0/Camera"
                self.path=path
                camera=UsdGeom.Camera.Define(self.stage,path)
                author_standard_camera_ops(camera,UsdGeom,Gf)
                c=calibration;focal=24.
                camera.CreateProjectionAttr("perspective")
                camera.CreateFocalLengthAttr(focal)
                camera.CreateHorizontalApertureAttr(c.width*focal/float(c.K_left[0,0]))
                camera.CreateVerticalApertureAttr(c.height*focal/float(c.K_left[1,1]))
                camera.CreateClippingRangeAttr(Gf.Vec2f(.01,10.))
                cfg=TiledCameraCfg(prim_path=path,spawn=None,width=c.width,height=c.height,
                    data_types=["rgb"],update_period=0.,update_latest_camera_pose=True,debug_vis=False)

                class ExplicitTiledCamera(TiledCamera):
                    def _create_buffers(self):
                        # Upstream reads poses while creating buffers, so apply
                        # the owned view compatibility before that first read.
                        prepare_renderer_view(self._view)
                        return super()._create_buffers()

                    def __del__(self):
                        # The external owner performs all native cleanup once.
                        # Never reenter Replicator from a GC/finalizer callback.
                        return None
                sensor=ExplicitTiledCamera.__new__(ExplicitTiledCamera)
                self.sensor=self.resources.sensor=sensor
                sensor._annotators={};sensor._render_product_paths=[];sensor._sensor_prims=[]
                for name in ("_prim_deletion_callback_id","_initialize_handle",
                             "_invalidate_initialize_handle","_debug_vis_handle"):
                    setattr(sensor,name,None)
                ExplicitTiledCamera.__init__(sensor,cfg)
                # This rig is created after the existing task's reset/PLAY event.
                # Call the audited implementation directly; the callback variant
                # swallows errors into a global variable and must not be used here.
                with self.resources.product_scope(rep.create):
                    if sensor.is_initialized:raise RuntimeError("unexpected tiled initialization before ownership tracking")
                    sensor._initialize_impl()
                    sensor._is_initialized=True
                if Path(inspect.getfile(type(sensor._view))).resolve()!=Path(sources["xform_view"]["path"]).resolve():
                    raise ValueError("imported camera view differs from audited installation")
                prepare_renderer_view(sensor._view)
                paths=sensor.render_product_paths
                if paths!=[self.resources.product.path]:raise ValueError("tiled render product identity differs")
                self.reference=rep.AnnotatorRegistry.get_annotator("ReferenceTime",device="cpu")
                self.times=rep.AnnotatorRegistry.get_annotator("IsaacReadTimes",device="cpu")
                self.resources.attach(self.reference,paths[0]);self.resources.attach(self.times,paths[0])
                core=importlib.import_module("isaacsim.core.nodes.bindings._isaacsim_core_nodes")
                self.core=core.acquire_interface()
                self.select("left")
                self._native_initialized=True
            except BaseException as original:
                try:self.close()
                except BaseException as cleanup:
                    raise RuntimeError("tiled initialization failed; cleanup incomplete: "+str(cleanup)) from original
                raise
        finally:
            sim = sensor = camera = self = None

    def select(self,side):
        if self._closed:raise RuntimeError("tiled reader is closed")
        if side not in ("left","right"):raise ValueError("left/right camera side required")
        from isaaclab.utils.math import quat_from_matrix
        transform=getattr(self.calibration,"T_world_"+side).to(device=self.device,dtype=torch.float32)
        position=transform[:3,3][None]+self.origins.to(device=self.device,dtype=torch.float32)
        rotation=quat_from_matrix(transform[:3,:3])[None]
        k=getattr(self.calibration,"K_"+side)[None].to(device=self.device,dtype=torch.float32)
        self.sensor.set_intrinsic_matrices(k,focal_length=24.)
        self.sensor.set_world_poses(positions=position,orientations=rotation,convention="ros")
        self.side=side

    def read_clock(self):
        if self._closed:raise RuntimeError("tiled reader is closed")
        return OwnedNativeRGBPairReader.read_clock(self)

    def _readback(self):
        from isaaclab.utils.math import matrix_from_quat
        data=self.sensor.data
        transform=torch.eye(4,dtype=torch.float64,device=self.device)[None].repeat(self.n,1,1)
        transform[:,:3,:3]=matrix_from_quat(data.quat_w_ros).to(torch.float64)
        transform[:,:3,3]=data.pos_w.to(torch.float64)
        from pxr import Usd,UsdGeom
        validate_usd_camera_pose(self.stage,self.path,transform,Usd,UsdGeom)
        return data.intrinsic_matrices.clone(),transform

    def read(self):
        if self._closed:raise RuntimeError("tiled reader is closed")
        from .tabletop_preview_warp_readback import isaac45_rgb_readback_scope
        before=self.read_clock()
        if before is None:return None
        with isaac45_rgb_readback_scope():
            self.sensor.update(0,force_recompute=True)
            image=self.sensor.data.output["rgb"]
            c=self.calibration
            if not isinstance(image,torch.Tensor) or image.shape!=(self.n,c.height,c.width,3):
                raise ValueError("unexpected native RGB shape")
            if image.dtype!=torch.uint8:raise ValueError("native RGB must retain audited uint8 source")
            rgb=image.permute(0,3,1,2).contiguous().float()/255.
            k,transform=self._readback()
        after=self.read_clock()
        if before!=after:raise ValueError("raw render clock changed during TiledCamera update")
        return dict(rgb=rgb.clone(),K=k,T_world=transform,raw_clock=before,
            raw_time_metadata=str(getattr(self,"last_clock_diagnostic",None))[:4096],
            native_evidence=getattr(self,"_native_initialized",False) is True)

    def close(self):
        if self._closed:
            if self._close_error:raise RuntimeError(self._close_error)
            return
        self._closed=True
        try:
            if self.device.type=="cuda":torch.cuda.synchronize(self.device)
            self.resources.close()
        except BaseException as error:
            self._close_error=str(error)
            raise
        finally:
            self.sensor=self.sim=self.stage=self.reference=self.times=self.core=None
