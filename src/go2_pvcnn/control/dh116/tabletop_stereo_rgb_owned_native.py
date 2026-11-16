"""Owned variant of the existing native RGB reader; Isaac imports stay lazy."""
import importlib
import numpy as np
import torch

from .tabletop_stereo_camera import _NativeRGBPairReader
from .tabletop_stereo_calibration import usd_aperture_offsets
from .tabletop_stereo_rgb_resources import RGBResources


class OwnedNativeRGBPairReader(_NativeRGBPairReader):
    def __init__(self,sim,origins,calibration,device,*,max_readback_bytes=1024**3,allow_resource_override=False):
        self.resources=RGBResources()
        self._last_valid_clock=None
        try:
            # Resource admission happens before creating any tiled render product.
            origin_values=torch.as_tensor(origins,dtype=torch.float64).detach().cpu()
            if origin_values.ndim!=2 or origin_values.shape[1]!=3 or not torch.isfinite(origin_values).all():
                raise ValueError('finite camera origins [N,3] required')
            if type(max_readback_bytes) is not int or max_readback_bytes<1:raise ValueError('positive camera readback resource limit required')
            estimate=2*len(origin_values)*calibration.width*calibration.height*16
            if estimate>max_readback_bytes and not allow_resource_override:
                raise ValueError('2N full-resolution RGB readback exceeds resource limit; reduce num_envs or explicitly override')
            self.resource_metadata=dict(real_camera_views=2*len(origin_values),resolution_wh=[calibration.width,calibration.height],
                minimum_rgb_readback_bytes_estimate=estimate,max_readback_bytes=max_readback_bytes,
                estimate_scope='RGBA8_plus_float32_RGB_only;renderer_assets_and_transient_overhead_not_measured',
                hidden_downsample=False,gpu_memory_measured=False)
            from .tabletop_vision_camera import _CameraAxesDeviceMixin, numeric_slot_order
            kit=importlib.import_module('omni.kit.app')
            kit.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.sensors.camera',True)
            base=importlib.import_module('isaacsim.sensors.camera').CameraView
            class DeviceSafeRGBView(_CameraAxesDeviceMixin,base):pass
            omni_usd=importlib.import_module('omni.usd')
            rep=importlib.import_module('omni.replicator.core')
            core=importlib.import_module('isaacsim.core.nodes.bindings._isaacsim_core_nodes')
            from pxr import Gf, Usd, UsdGeom
            self.sim,self.calibration,self.device=sim,calibration,torch.device(device)
            self.origins=origin_values
            if self.origins.ndim!=2 or self.origins.shape[1]!=3 or not torch.isfinite(self.origins).all():
                raise ValueError('finite camera origins [N,3] required')
            self.n=len(self.origins);self.count=2*self.n
            self.Usd,self.UsdGeom=Usd,UsdGeom
            self.stage=omni_usd.get_context().get_stage()
            parent='/World/DH116StereoRGB'
            if self.stage.GetPrimAtPath(parent).IsValid():raise RuntimeError('stereo RGB rig already exists')
            self.resources.define_rig(self.stage, UsdGeom.Xform.Define)
            for env,origin in enumerate(self.origins.numpy()):
                for side,(k,t) in enumerate(((calibration.K_left,calibration.T_world_left),(calibration.K_right,calibration.T_world_right))):
                    camera=UsdGeom.Camera.Define(self.stage,parent+f'/camera_{2*env+side}')
                    focal=24.
                    camera.CreateProjectionAttr('perspective')
                    camera.CreateFocalLengthAttr(focal)
                    camera.CreateHorizontalApertureAttr(calibration.width*focal/float(k[0,0]))
                    camera.CreateVerticalApertureAttr(calibration.height*focal/float(k[1,1]))
                    horizontal_offset,vertical_offset=usd_aperture_offsets(
                        k,calibration.width,calibration.height,focal)
                    camera.CreateHorizontalApertureOffsetAttr(horizontal_offset)
                    camera.CreateVerticalApertureOffsetAttr(vertical_offset)
                    camera.CreateClippingRangeAttr(Gf.Vec2f(.01,10.))
                    optical=t.numpy().copy();optical[:3,3]+=origin
                    usd=optical@np.diag([1.,-1.,-1.,1.])
                    camera.AddTransformOp().Set(Gf.Matrix4d(*usd.T.reshape(-1).tolist()))
            self.resources.unsafe=True
            self.view=DeviceSafeRGBView(prim_paths_expr=parent+'/camera_.*',name='dh116_stereo_rgb',
                camera_resolution=(calibration.width,calibration.height),output_annotators=['rgb'],reset_xform_properties=False)
            self.resources.bind_view(self.view)
            self.paths=list(self.view.prim_paths)
            self.order=numeric_slot_order(self.paths,self.count).to(self.device)
            tiled_width=int(self.view.tiled_resolution[0]);self.columns=tiled_width//calibration.width
            if tiled_width!=self.columns*calibration.width:raise RuntimeError('invalid stereo tiled layout')
            self.reference=rep.AnnotatorRegistry.get_annotator('ReferenceTime',device='cpu')
            self.times=rep.AnnotatorRegistry.get_annotator('IsaacReadTimes',device='cpu')
            path=self.view.get_render_product_path()
            self.resources.attach(self.reference,path,lambda:self.reference.attach([path]))
            self.resources.attach(self.times,path,lambda:self.times.attach([path]))
            self.core=core.acquire_interface()

        except BaseException as exc:
            try:
                self.resources.close()
            except Exception as cleanup_error:
                raise RuntimeError("native reader initialization failed; cleanup incomplete: " + str(cleanup_error)) from exc
            raise

    def read_clock(self):
        from .tabletop_vision_camera import paired_render_timestamp
        ref = self.reference.get_data()
        times = self.times.get_data()
        startup = getattr(self, "_last_valid_clock", None) is None
        self.last_clock_diagnostic = dict(reference=repr(ref), times=repr(times), startup=startup)
        # OgnRefTime's inputs initialize to 0/0 before its first render execution.
        # This is unavailable data, never an accepted time or an infinite retry.
        ref_zero = (startup and isinstance(ref, dict)
            and ref.get("referenceTimeNumerator") == 0 and ref.get("referenceTimeDenominator") == 0)
        time_keys = ("swhFrameNumber", "frameNumber", "simulationTime",
            "rationalTimeOfSimNumerator", "rationalTimeOfSimDenominator")
        times_zero = (startup and isinstance(times, dict) and all(k in times for k in time_keys)
            and all(times[k] == 0 for k in time_keys))
        ref_empty = ref_zero or ref is None or (isinstance(ref, dict) and not ref)
        times_empty = times_zero or times is None or (isinstance(times, dict) and not times)
        if not ref_empty:
            if not isinstance(ref, dict) or not {"referenceTimeNumerator", "referenceTimeDenominator"} <= ref.keys():
                raise ValueError("invalid ReferenceTime metadata")
            try:
                if int(ref["referenceTimeNumerator"]) < 0 or int(ref["referenceTimeDenominator"]) <= 0:
                    raise ValueError("invalid ReferenceTime clock: " + repr(ref))
            except (TypeError, OverflowError) as exc:
                raise ValueError("invalid ReferenceTime clock: " + repr(ref)) from exc
        if not times_empty:
            keys = {"swhFrameNumber", "frameNumber", "simulationTime",
                    "rationalTimeOfSimNumerator", "rationalTimeOfSimDenominator"}
            if not isinstance(times, dict) or not keys <= times.keys():
                raise ValueError("invalid IsaacReadTimes metadata")
            try:
                if (int(times["swhFrameNumber"]) < 0 or int(times["frameNumber"]) < 0
                        or int(times["rationalTimeOfSimNumerator"]) < 0
                        or int(times["rationalTimeOfSimDenominator"]) <= 0
                        or not np.isfinite(float(times["simulationTime"]))
                        or float(times["simulationTime"]) < 0):
                    raise ValueError("invalid IsaacReadTimes clock: " + repr(times))
            except (TypeError, OverflowError) as exc:
                raise ValueError("invalid IsaacReadTimes clock: " + repr(times)) from exc
        if ref_empty or times_empty:
            return None
        legacy = float(self.core.get_sim_time_at_swh_frame(int(ref["referenceTimeNumerator"])))
        timing = paired_render_timestamp(ref, times, legacy_time=legacy)
        self._last_valid_clock = dict(timing)
        return {key: timing[key] for key in ("frame_id", "simulation_time", "render_frame_number")}

    def close(self):
        try:
            self.resources.close()
        finally:
            self.sim = self.view = self.stage = self.reference = self.times = self.core = None
