"""Independent synchronized RGB camera pair; render/read never advances physics.

Native backend owns 2*N real camera prims in one tiled RGB-only render product.
CPU reader injection is explicitly diagnostic and cannot grant native evidence.
"""
import copy
import importlib
import math
import json
from pathlib import Path
import numpy as np
import torch

from .tabletop_stereo_calibration import StereoCalibration
from .tabletop_stereo_depth import StereoDepth
from .tabletop_stereo_validation import validate_images, validate_identity, validate_pair_timing


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)


class _NativeRGBPairReader:
    """Lazy Isaac 4.5 adapter; no render-depth annotator or task/world-state access."""
    def __init__(self,sim,origins,calibration,device,*,max_readback_bytes=1024**3,allow_resource_override=False):
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
        UsdGeom.Xform.Define(self.stage,parent)
        for env,origin in enumerate(self.origins.numpy()):
            for side,(k,t) in enumerate(((calibration.K_left,calibration.T_world_left),(calibration.K_right,calibration.T_world_right))):
                camera=UsdGeom.Camera.Define(self.stage,parent+f'/camera_{2*env+side}')
                focal=24.
                camera.CreateProjectionAttr('perspective')
                camera.CreateFocalLengthAttr(focal)
                camera.CreateHorizontalApertureAttr(calibration.width*focal/float(k[0,0]))
                camera.CreateVerticalApertureAttr(calibration.height*focal/float(k[1,1]))
                # USD pinhole offset units follow aperture; do not assume centre principal point.
                camera.CreateHorizontalApertureOffsetAttr((float(k[0,2])-calibration.width/2)*focal/float(k[0,0]))
                camera.CreateVerticalApertureOffsetAttr(-(float(k[1,2])-calibration.height/2)*focal/float(k[1,1]))
                camera.CreateClippingRangeAttr(Gf.Vec2f(.01,10.))
                optical=t.numpy().copy();optical[:3,3]+=origin
                usd=optical@np.diag([1.,-1.,-1.,1.])
                camera.AddTransformOp().Set(Gf.Matrix4d(*usd.T.reshape(-1).tolist()))
        self.view=DeviceSafeRGBView(prim_paths_expr=parent+'/camera_.*',name='dh116_stereo_rgb',
            camera_resolution=(calibration.width,calibration.height),output_annotators=['rgb'],reset_xform_properties=False)
        self.paths=list(self.view.prim_paths)
        self.order=numeric_slot_order(self.paths,self.count).to(self.device)
        tiled_width=int(self.view.tiled_resolution[0]);self.columns=tiled_width//calibration.width
        if tiled_width!=self.columns*calibration.width:raise RuntimeError('invalid stereo tiled layout')
        self.reference=rep.AnnotatorRegistry.get_annotator('ReferenceTime',device='cpu')
        self.times=rep.AnnotatorRegistry.get_annotator('IsaacReadTimes',device='cpu')
        self.reference.attach([self.view.get_render_product_path()]);self.times.attach([self.view.get_render_product_path()])
        self.core=core.acquire_interface()

    def _readback(self):
        from .tabletop_depth_camera_probe import usd_pose_to_optical
        positions,quats=self.view.get_world_poses(camera_axes='usd',usd=True)
        transforms=np.stack([usd_pose_to_optical(p,q) for p,q in zip(_array(positions),_array(quats))])
        focal=np.asarray(self.view.get_focal_lengths())*10.
        ah=np.asarray(self.view.get_horizontal_apertures());av=np.asarray(self.view.get_vertical_apertures())
        # Offset is read independently from the actual USD prim; APIs vary by SDK.
        K=np.tile(np.eye(3),(self.count,1,1));c=self.calibration
        K[:,0,0]=c.width*focal/ah;K[:,1,1]=c.height*focal/av
        cache=self.UsdGeom.XformCache(self.Usd.TimeCode.Default())
        for i,path in enumerate(self.paths):
            camera=self.UsdGeom.Camera(self.stage.GetPrimAtPath(path))
            usd_f=float(camera.GetFocalLengthAttr().Get())
            K[i,0,2]=c.width/2+float(camera.GetHorizontalApertureOffsetAttr().Get())*K[i,0,0]/usd_f
            K[i,1,2]=c.height/2-float(camera.GetVerticalApertureOffsetAttr().Get())*K[i,1,1]/usd_f
            authored=np.asarray(cache.GetLocalToWorldTransform(camera.GetPrim())).T@np.diag([1.,-1.,-1.,1.])
            if not np.allclose(authored,transforms[i],atol=1e-5,rtol=0.):raise ValueError('USD/runtime stereo calibration mismatch')
        index=self.order.cpu().numpy()
        K=K[index].reshape(self.n,2,3,3);transforms=transforms[index].reshape(self.n,2,4,4)
        self.calibration.validate_readback(K[:,0],K[:,1],transforms[:,0],transforms[:,1],origins=self.origins)
        return K,transforms

    def read(self):
        from .tabletop_vision_camera import paired_render_timestamp, detile
        ref=self.reference.get_data()
        if not isinstance(ref,dict) or 'referenceTimeNumerator' not in ref:return None
        legacy=float(self.core.get_sim_time_at_swh_frame(int(ref['referenceTimeNumerator'])))
        timing=paired_render_timestamp(ref,self.times.get_data(),legacy_time=legacy)
        c=self.calibration
        rgb=detile(self.view.get_rgb_tiled(device='cuda'),count=self.count,width=c.width,height=c.height,
                   columns=self.columns,channels=3).to(self.device)[self.order].reshape(self.n,2,c.height,c.width,3)
        after=paired_render_timestamp(self.reference.get_data(),self.times.get_data(),legacy_time=legacy)
        if (after['frame_id'],after['simulation_time'],after['render_frame_number'])!=(timing['frame_id'],timing['simulation_time'],timing['render_frame_number']):
            raise ValueError('stereo render changed during RGB read')
        K,T=self._readback()
        integer_rgb=not rgb.is_floating_point()
        rgb=rgb.float()
        if integer_rgb or bool((rgb>1).any()):rgb=rgb/255.
        return dict(left_rgb=rgb[:,0].permute(0,3,1,2).contiguous(),right_rgb=rgb[:,1].permute(0,3,1,2).contiguous(),
            left_frame_id=timing['frame_id'],right_frame_id=timing['frame_id'],
            left_simulation_time=timing['simulation_time'],right_simulation_time=timing['simulation_time'],
            K_left=K[:,0],K_right=K[:,1],T_world_left=T[:,0],T_world_right=T[:,1],
            native_evidence=True,render_timing=timing)


class StereoCamera:
    def __init__(self,calibration,reader,*,depth_estimator=None,sim=None,origins=None):
        self.calibration=calibration;self.reader=reader
        self.depth_estimator=depth_estimator if depth_estimator is not None else StereoDepth()
        self.sim=sim;self.origins=origins;self.last_frame=-1

    @classmethod
    def from_native(cls,sim,origins,config_path,device='cuda',*,depth_config=None,
                    max_width=None,max_height=None,max_readback_bytes=None,allow_resource_override=False):
        from .tabletop_stereo_resolution import validate_resolution,DEFAULT_MAX_WIDTH,DEFAULT_MAX_HEIGHT
        config=json.loads(Path(config_path).read_text(encoding='utf-8'))
        c=StereoCalibration.from_dict(config)
        resources=config.get('resource_limits',{})
        override=allow_resource_override or resources.get('allow_resource_override',False)
        validate_resolution(c.width,c.height,max_width=resources.get('max_width',DEFAULT_MAX_WIDTH) if max_width is None else max_width,
            max_height=resources.get('max_height',DEFAULT_MAX_HEIGHT) if max_height is None else max_height,allow_resource_override=override)
        if depth_config is None:depth_config=config
        estimator=StereoDepth.from_config(depth_config)
        from .tabletop_stereo_wire_segmentation import configured_segmentation_depth
        estimator=configured_segmentation_depth(estimator,config,device)
        reader=_NativeRGBPairReader(sim,origins,c,device,
            max_readback_bytes=resources.get('max_readback_bytes',1024**3) if max_readback_bytes is None else max_readback_bytes,
            allow_resource_override=override)
        return cls(c,reader,depth_estimator=estimator,sim=sim,origins=origins)

    @torch.no_grad()
    def sample(self,*,identity):
        before=float(self.sim.current_time) if self.sim is not None else None
        packet=None;error='no render metadata'
        for attempt in range(32 if self.last_frame<0 else 6):
            if self.sim is not None:
                self.sim.render()
                if float(self.sim.current_time)!=before:raise RuntimeError('stereo render advanced physics time')
            candidate=self.reader.read()
            if candidate is None:
                if self.sim is None:raise RuntimeError('RGB stereo frame unavailable')
                continue
            if any(k in candidate for k in ('depth','depth_m','rendered_depth','ground_truth_depth')):
                raise ValueError('camera reader must be RGB-only; rendered depth is prohibited')
            try:
                frame,timestamp=validate_pair_timing(candidate,self.last_frame)
                if before is not None and abs(timestamp-before)>1e-6:raise RuntimeError('stale render simulation timestamp')
            except (ValueError,RuntimeError) as exc:
                if self.sim is None:raise
                error=str(exc);continue
            packet=candidate;break
        if packet is None:raise RuntimeError('fresh synchronized stereo RGB unavailable: '+error)
        left,right=packet['left_rgb'],packet['right_rgb'];c=self.calibration
        validate_images(left,right,c)
        ident=validate_identity(identity,left.shape[0],timestamp)
        for row in (ident if isinstance(ident,list) else [ident]):
            if ('frame_id' in row and row['frame_id']!=frame) or ('calibration_sha256' in row and row['calibration_sha256']!=c.sha256):
                raise ValueError('caller stereo identity frame/calibration mismatch')
            row.update(frame_id=frame,calibration_sha256=c.sha256)
        readback_diagnostics=c.validate_readback(packet['K_left'],packet['K_right'],packet['T_world_left'],packet['T_world_right'],origins=self.origins)
        if before is not None and float(self.sim.current_time)!=before:raise RuntimeError('camera read advanced physics time')
        depth=self.depth_estimator.compute(left,right,c)
        self.last_frame=frame
        return dict(left_rgb=left.clone(),right_rgb=right.clone(),**depth,
            identity=ident,frame_id=frame,simulation_time=timestamp,calibration_sha256=c.sha256,
            K_left=torch.as_tensor(packet['K_left']).clone(),K_right=torch.as_tensor(packet['K_right']).clone(),
            T_world_left=torch.as_tensor(packet['T_world_left']).clone(),T_world_right=torch.as_tensor(packet['T_world_right']).clone(),
            native_evidence=isinstance(self.reader,_NativeRGBPairReader) and packet.get('native_evidence') is True,
            render_timing=copy.deepcopy(packet.get('render_timing')),render_passes=attempt+1,
            depth_resource_metadata=copy.deepcopy(getattr(self.depth_estimator,'last_resource_metadata',None)),
            calibration_readback_residuals=readback_diagnostics,
            camera_resource_metadata=copy.deepcopy(getattr(self.reader,'resource_metadata',None)))
