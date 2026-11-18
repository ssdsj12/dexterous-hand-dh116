"""Batched rendered RGB-D observations, with no object-state feature access.

One CameraView owns one tiled render product. Camera slot order is checked from
actual prim paths. Visible color pixels are unprojected with optical-Z depth and
transformed into current palm-link frames (the simulator's encoder/FK stand-in).
"""

import importlib
import json
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as F

from .tabletop_depth_camera_probe import look_at_optical, usd_pose_to_optical


def numeric_slot_order(prim_paths, count):
    """Return camera slots indexed by numeric environment ID, never lexicography."""
    ids = []
    for path in prim_paths:
        match = re.fullmatch(r'.*/camera_(\d+)', str(path))
        if match is None:
            raise ValueError(f'unexpected vision camera prim path: {path}')
        ids.append(int(match.group(1)))
    if len(ids) != count or sorted(ids) != list(range(count)):
        raise ValueError('camera prim paths must contain each numeric environment ID exactly once')
    return torch.tensor([ids.index(i) for i in range(count)], dtype=torch.long)


def detile(tiled, *, count, width, height, columns, channels):
    """Untile actual linear pixels; handles upstream rectangular reshape mistakes."""
    rows = (count+columns-1)//columns
    data = torch.as_tensor(tiled)
    if data.numel() != rows*height*columns*width*channels:
        raise ValueError('tiled camera buffer size does not match calibrated tile layout')
    return (data.reshape(rows,height,columns,width,channels).permute(0,2,1,3,4)
            .reshape(rows*columns,height,width,channels)[:count])


def brown_mask(rgb, *, float_rgb_is_255=None):
    """GPU equivalent of probe.brown_rod_mask, verified against that RGB-only rule."""
    color = rgb[..., :3].to(torch.float64)
    if not rgb.is_floating_point():
        color = color/255.
    elif (bool(torch.any(color > 1.)) if float_rgb_is_255 is None else float_rgb_is_255):
        color = color/255.
    r,g,b = color.unbind(-1)
    maximum, minimum = color.amax(-1), color.amin(-1)
    saturation = (maximum-minimum)/maximum.clamp_min(1e-12)
    return (torch.isfinite(color).all(-1) & (r>1.2*g) & (g>1.2*b)
            & (maximum>35./255.) & (saturation>.25))


def _quaternion_matrices(q):
    if not torch.isfinite(q).all() or bool((q.norm(dim=-1) == 0).any()):
        raise ValueError('finite nonzero palm wxyz quaternions required')
    q = q/q.norm(dim=-1,keepdim=True)
    w,x,y,z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),dim=-1).reshape(*q.shape[:-1],3,3)


def frames_to_points(rgb, depth, K, T_world_camera, hand_poses, *, num_points,
                     min_depth_m, max_depth_m, depth_edge_threshold_m=.02):
    """Validate the full sensor batch, then bound pixel work to 64 environments.

    Only small point/mask/diagnostic outputs are concatenated. In particular the
    float64 RGB rule and full-batch float RGB scale decision remain unchanged.
    """
    if depth.ndim==4 and depth.shape[-1]==1:depth=depth[...,0]
    if depth.ndim!=3 or rgb.shape!=(*depth.shape,3) or num_points<1:
        raise ValueError('aligned RGB [N,H,W,3], depth [N,H,W] and positive point count required')
    n=depth.shape[0];device=depth.device
    K=torch.as_tensor(K,device=device,dtype=torch.float32)
    T=torch.as_tensor(T_world_camera,device=device,dtype=torch.float32)
    hands=torch.as_tensor(hand_poses,device=device,dtype=torch.float32)
    if K.shape!=(n,3,3) or T.shape!=(n,4,4) or hands.shape!=(n,2,7):
        raise ValueError('per-environment K, camera transform and two xyz+wxyz palm poses required')
    if not (torch.isfinite(K).all() and torch.isfinite(T).all() and torch.isfinite(hands).all()):
        raise ValueError('finite camera calibration and palm poses required')
    cap=64
    float_rgb_is_255=None
    if rgb.is_floating_point():
        # Preserve the original batch-wide >1 decision without a full-batch
        # float64 conversion or full-batch Boolean image temporary.
        float_rgb_is_255=any(bool(torch.any(rgb[start:start+cap]>1.))
                             for start in range(0,n,cap))
    kwargs=dict(num_points=num_points,min_depth_m=min_depth_m,max_depth_m=max_depth_m,
        depth_edge_threshold_m=depth_edge_threshold_m,float_rgb_is_255=float_rgb_is_255)
    if n<=cap:
        return _frames_to_points_dense(rgb,depth,K,T,hands,**kwargs)
    blocks=[_frames_to_points_dense(rgb[start:start+cap],depth[start:start+cap],
        K[start:start+cap],T[start:start+cap],hands[start:start+cap],**kwargs)
        for start in range(0,n,cap)]
    return {key:torch.cat([block[key] for block in blocks],dim=0) for key in blocks[0]}


def _frames_to_points_dense(rgb, depth, K, T_world_camera, hand_poses, *, num_points,
                            min_depth_m, max_depth_m, depth_edge_threshold_m=.02,
                            float_rgb_is_255=None):
    """Pure tensor sensor transform; no simulator or privileged-state parameter."""
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[...,0]
    if (depth.ndim != 3 or rgb.shape != (*depth.shape,3) or num_points < 1):
        raise ValueError('aligned RGB [N,H,W,3], depth [N,H,W] and positive point count required')
    n,h,w = depth.shape
    device = depth.device
    depth = depth.to(torch.float32)
    K = torch.as_tensor(K,device=device,dtype=torch.float32)
    T = torch.as_tensor(T_world_camera,device=device,dtype=torch.float32)
    hands = torch.as_tensor(hand_poses,device=device,dtype=torch.float32)
    if K.shape != (n,3,3) or T.shape != (n,4,4) or hands.shape != (n,2,7):
        raise ValueError('per-environment K, camera transform and two xyz+wxyz palm poses required')
    if not (torch.isfinite(K).all() and torch.isfinite(T).all() and torch.isfinite(hands).all()):
        raise ValueError('finite camera calibration and palm poses required')
    depth_valid = torch.isfinite(depth)&(depth>0.)&(depth>=min_depth_m)&(depth<=max_depth_m)
    image_valid = depth_valid.flatten(1).any(-1)&torch.isfinite(rgb).flatten(1).all(-1)
    brown_valid=depth_valid & brown_mask(rgb,float_rgb_is_255=float_rgb_is_255) & image_valid[:,None,None]
    consistent=depth_valid
    if depth_edge_threshold_m is not None:
        if not np.isfinite(depth_edge_threshold_m) or depth_edge_threshold_m<=0.:
            raise ValueError('depth edge threshold must be finite and positive')
        # RGB antialiasing can tint a table-depth silhouette pixel brown. Require
        # a full valid 3x3 depth neighbourhood before treating it as a rod point.
        z=torch.where(depth_valid,depth,torch.zeros_like(depth))[:,None]
        neighborhood_valid=F.avg_pool2d(depth_valid.float()[:,None],3,stride=1,padding=1)>=1.
        largest=F.max_pool2d(z,3,stride=1,padding=1)
        smallest=-F.max_pool2d(-z,3,stride=1,padding=1)
        consistent=(neighborhood_valid & ((largest-smallest)<=depth_edge_threshold_m))[:,0]
    visible = (brown_valid & consistent).flatten(1)
    counts = visible.sum(-1)
    # Select equally-spaced ranks among actual retained pixels, preserving order.
    index = torch.arange(num_points,device=device,dtype=torch.long)[None].expand(n,-1)
    uniform = torch.div(index*(counts[:,None]-1).clamp_min(0),max(num_points-1,1),rounding_mode='floor')
    ranks = torch.where(counts[:,None]>=num_points,uniform,index)+1
    cumsum = visible.cumsum(-1)
    pixels = torch.searchsorted(cumsum,ranks.contiguous()).clamp_max(h*w-1)
    point_valid = index<counts[:,None].clamp_max(num_points)
    u,v = pixels%w, torch.div(pixels,w,rounding_mode='floor')
    # RTX evaluates pixel centers. K's principal point remains (W/2,H/2).
    pixel_h = torch.stack((u.to(torch.float32)+.5,v.to(torch.float32)+.5,
                           torch.ones_like(u,dtype=torch.float32)),dim=-1)
    rays = pixel_h @ torch.linalg.inv(K).transpose(-1,-2)
    z = depth.flatten(1).gather(1,pixels)
    # Missing/invalid pixel depths never enter features, including padded rows.
    z = torch.where(point_valid,z,torch.zeros_like(z))
    camera_points = rays*z[...,None]
    world_points = camera_points @ T[:,:3,:3].transpose(-1,-2)+T[:,None,:3,3]
    hand_rotation = _quaternion_matrices(hands[...,3:])
    points = (world_points[:,None]-hands[:,:,None,:3]) @ hand_rotation
    points = torch.where(point_valid[:,None,:,None],points,torch.zeros_like(points))
    return dict(points=points.contiguous(), point_valid=point_valid,
        image_valid=image_valid, object_valid=counts>0,
        visible_point_count=counts, depth_valid_fraction=depth_valid.float().flatten(1).mean(-1),
        raw_brown_point_count=brown_valid.flatten(1).sum(-1),
        edge_rejected_point_count=brown_valid.flatten(1).sum(-1)-counts,
        depth_consistent_fraction=consistent.float().flatten(1).mean(-1),
        selected_pixels_uv=torch.stack((u,v),dim=-1))


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)


def paired_render_timestamp(reference, times, *, legacy_time):
    """Use the simulation timestamp carried by this exact rendered result."""
    if (not isinstance(reference,dict) or not isinstance(times,dict)
            or not {'referenceTimeNumerator','referenceTimeDenominator'}<=reference.keys()
            or not {'swhFrameNumber','frameNumber','simulationTime',
                    'rationalTimeOfSimNumerator','rationalTimeOfSimDenominator'}<=times.keys()):
        raise ValueError(f'render time metadata unavailable: reference={reference}, times={times}')
    frame_id=int(reference['referenceTimeNumerator'])
    denominator=int(reference['referenceTimeDenominator'])
    swh_frame=int(times['swhFrameNumber'])
    # Native evidence: ReferenceTime=(50,30), rationalTimeOfSim=(50,30),
    # frameNumber=50, swhFrameNumber=1. SWH is a different clock identifier.
    rational_pair=(int(times['rationalTimeOfSimNumerator']),int(times['rationalTimeOfSimDenominator']))
    if frame_id<0 or denominator<=0 or rational_pair!=(frame_id,denominator):
        raise ValueError(f'render frame mismatch: ReferenceTime={reference}, IsaacReadTimes={times}')
    timestamp=float(times['simulationTime'])
    if not np.isfinite(timestamp) or timestamp<0.:
        raise ValueError(f'invalid same-frame simulation time: {timestamp}')
    diagnostics={key:(value.item() if hasattr(value,'item') else value) for key,value in times.items()}
    return dict(frame_id=frame_id,simulation_time=timestamp,
        reference_time_numerator=frame_id,reference_time_denominator=denominator,
        render_frame_number=int(times['frameNumber']),swh_frame_number=swh_frame,
        legacy_simulation_time=float(legacy_time),isaac_read_times=diagnostics,
        time_mapping='same_render_product_IsaacReadTimes')


class _CameraAxesDeviceMixin:
    """Repair 4.5 CameraView's CPU constant during its parent constructor call."""
    def _convert_camera_axes(self, orientations, transform_matrix):
        if not isinstance(orientations,torch.Tensor):
            return super()._convert_camera_axes(orientations,transform_matrix)
        matrices=self._backend_utils.quats_to_rot_matrices(orientations).to(
            device=orientations.device,dtype=orientations.dtype)
        transform=torch.as_tensor(transform_matrix[:3,:3],device=orientations.device,
            dtype=orientations.dtype)
        # Preserve installed CameraView's left-multiplication convention.
        converted=transform@matrices
        result=self._backend_utils.rot_matrices_to_quats(converted,device=orientations.device)
        return result.to(device=orientations.device,dtype=orientations.dtype)


class _NativeTiledReader:
    """Isaac 4.5 adapter. All native imports occur in the existing Kit process."""
    def __init__(self, *, sim, origins, config, width, height, device):
        kit = importlib.import_module('omni.kit.app')
        kit.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.sensors.camera',True)
        CameraView = importlib.import_module('isaacsim.sensors.camera').CameraView
        class DeviceSafeCameraView(_CameraAxesDeviceMixin,CameraView):
            pass
        import omni.usd
        import omni.replicator.core as rep
        from isaacsim.core.nodes.bindings import _isaacsim_core_nodes
        from pxr import Gf, Usd, UsdGeom, UsdLux
        self._Usd,self._UsdGeom = Usd,UsdGeom
        self._core_nodes = _isaacsim_core_nodes.acquire_interface()
        self._sim,self._device = sim,device
        self.width,self.height,self.n = width,height,len(origins)
        stage = omni.usd.get_context().get_stage()
        self._stage = stage
        parent = '/World/DH116VisionCameras'
        if stage.GetPrimAtPath(parent).IsValid():
            raise RuntimeError('vision cameras already exist; refusing duplicate tiled products')
        UsdGeom.Xform.Define(stage,parent)
        light = UsdLux.DomeLight.Define(stage,parent+'/Light')
        light.CreateIntensityAttr(1800.)
        light.CreateColorAttr(Gf.Vec3f(1.,1.,1.))
        focal=24.
        aperture=2*focal*np.tan(np.deg2rad(config['horizontal_fov_degrees'])/2)
        self._expected=[]
        for env_id,origin in enumerate(_numpy(origins)):
            optical=look_at_optical(np.asarray(config['position_world_m'])+origin,
                np.asarray(config['target_world_m'])+origin,config['world_up'])
            camera=UsdGeom.Camera.Define(stage,f'{parent}/camera_{env_id}')
            camera.CreateProjectionAttr('perspective')
            camera.CreateFocalLengthAttr(focal)
            camera.CreateHorizontalApertureAttr(float(aperture))
            camera.CreateVerticalApertureAttr(float(aperture*height/width))
            camera.CreateClippingRangeAttr(Gf.Vec2f(*config['clipping_range_m']))
            transform=optical@np.diag([1.,-1.,-1.,1.])
            camera.AddTransformOp().Set(Gf.Matrix4d(*transform.T.reshape(-1).tolist()))
            self._expected.append(optical)
        self.view=DeviceSafeCameraView(prim_paths_expr=parent+'/camera_.*',name='dh116_vision_tiled',
            camera_resolution=(width,height),output_annotators=['rgb','depth'],reset_xform_properties=False)
        self.prim_paths=list(self.view.prim_paths)
        order=numeric_slot_order(self.prim_paths,self.n)
        self.columns=int(self.view.tiled_resolution[0])//width
        if int(self.view.tiled_resolution[0])!=self.columns*width:
            raise RuntimeError('tiled render width is not a multiple of camera width')
        self._reference=rep.AnnotatorRegistry.get_annotator('ReferenceTime',device='cpu')
        self._reference.attach([self.view.get_render_product_path()])
        self._render_times=rep.AnnotatorRegistry.get_annotator('IsaacReadTimes',device='cpu')
        self._render_times.attach([self.view.get_render_product_path()])
        # USD-authored calibration is independent of the runtime camera readback.
        cache=UsdGeom.XformCache(Usd.TimeCode.Default())
        intrinsic=[]; transforms=[]
        for path in self.prim_paths:
            prim=stage.GetPrimAtPath(path)
            camera=UsdGeom.Camera(prim)
            f=float(camera.GetFocalLengthAttr().Get())
            ah=float(camera.GetHorizontalApertureAttr().Get())
            av=float(camera.GetVerticalApertureAttr().Get())
            intrinsic.append([[width*f/ah,0.,width/2.],[0.,height*f/av,height/2.],[0.,0.,1.]])
            transforms.append(np.asarray(cache.GetLocalToWorldTransform(prim)).T@np.diag([1.,-1.,-1.,1.]))
        self.K=torch.tensor(intrinsic,device=device,dtype=torch.float32)
        self.T_world_camera=torch.tensor(np.array(transforms),device=device,dtype=torch.float32)
        actual=_numpy(self.T_world_camera)[order.numpy()]
        if not np.allclose(actual,self._expected,atol=1e-5,rtol=0.):
            raise RuntimeError('authored optical camera calibration differs from placement configuration')
        self._calibration_checked=False

    def _validate_runtime_calibration(self):
        positions,quaternions=self.view.get_world_poses(camera_axes='usd',usd=True)
        actual=np.stack([usd_pose_to_optical(p,q) for p,q in zip(_numpy(positions),_numpy(quaternions))])
        if not np.allclose(actual,_numpy(self.T_world_camera),atol=1e-5,rtol=0.):
            raise RuntimeError('CameraView runtime optical extrinsics disagree with independent USD calibration')
        # CameraView focal API returns authored units divided by ten; aperture does not.
        focal=np.asarray(self.view.get_focal_lengths())*10.
        horizontal=np.asarray(self.view.get_horizontal_apertures())
        vertical=np.asarray(self.view.get_vertical_apertures())
        runtime=np.tile(np.eye(3),(self.n,1,1))
        runtime[:,0,0]=self.width*focal/horizontal
        runtime[:,1,1]=self.height*focal/vertical
        runtime[:,0,2]=self.width/2.;runtime[:,1,2]=self.height/2.
        if not np.allclose(runtime,_numpy(self.K),atol=1e-4,rtol=0.):
            raise RuntimeError('CameraView intrinsics disagree with independent USD calibration')
        self._calibration_checked=True

    def read(self):
        reference=self._reference.get_data()
        if not isinstance(reference,dict) or 'referenceTimeNumerator' not in reference:
            return None
        frame_id=int(reference['referenceTimeNumerator'])
        legacy_time=float(self._core_nodes.get_sim_time_at_swh_frame(frame_id))
        timing=paired_render_timestamp(reference,self._render_times.get_data(),legacy_time=legacy_time)
        # CameraView.get_rgb/get_depth swap width and height for rectangular tiles
        # in the installed 4.5 implementation. Keep its single real render product,
        # but decode the native tiled linear pixels using verified dimensions.
        rgb=detile(self.view.get_rgb_tiled(device='cuda'),count=self.n,width=self.width,
            height=self.height,columns=self.columns,channels=3).to(self._device)
        depth=detile(self.view.get_depth_tiled(device='cuda'),count=self.n,width=self.width,
            height=self.height,columns=self.columns,channels=1).to(self._device)
        after=self._reference.get_data()
        if int(after['referenceTimeNumerator'])!=frame_id:
            raise RuntimeError('camera frame changed while reading aligned RGB/depth')
        after_timing=paired_render_timestamp(after,self._render_times.get_data(),legacy_time=legacy_time)
        if (after_timing['simulation_time']!=timing['simulation_time']
                or after_timing['render_frame_number']!=timing['render_frame_number']):
            raise RuntimeError('render-product time metadata changed while reading RGB/depth')
        if not self._calibration_checked:self._validate_runtime_calibration()
        return dict(rgb=rgb,depth=depth,**timing)


class VisionCamera:
    """One sensor observation per control state; absent pixels are explicit zeros."""
    def __init__(self, world, device_world, config_path, width=160, height=120, num_points=128):
        self._sim,self._state=world.sim,device_world
        self.n,self.device=device_world.n,torch.device(device_world.device)
        if self.n not in (2,64):
            raise ValueError('vision camera supports exactly 2 smoke or 64 production environments')
        if not (type(width) is int and type(height) is int and 1<=width<=320 and 1<=height<=240):
            raise ValueError('vision camera resolution must not exceed 320x240')
        if type(num_points) is not int or not 1<=num_points<=1024:
            raise ValueError('num_points must be an integer in [1,1024]')
        config=json.loads(Path(config_path).read_text())
        self.near,self.far=map(float,config['clipping_range_m'])
        if not (0.<self.near<self.far and 1.<float(config['horizontal_fov_degrees'])<179.
                and config.get('depth_convention')=='optical_axis_z_m'):
            raise ValueError('finite optical-Z pinhole camera configuration required')
        self.width,self.height,self.num_points=width,height,num_points
        self._reader=_NativeTiledReader(sim=self._sim,origins=device_world.origins,config=config,
            width=width,height=height,device=self.device)
        self.slot_order=numeric_slot_order(self._reader.prim_paths,self.n).to(self.device)
        self.K=torch.as_tensor(self._reader.K,device=self.device,dtype=torch.float32)[self.slot_order]
        self.T_world_camera=torch.as_tensor(self._reader.T_world_camera,device=self.device,
            dtype=torch.float32)[self.slot_order]
        self._last_frame=-1
        self._samples=0

    @torch.no_grad()
    def sample(self, hand_poses=None):
        """Render fresh RGB-D without physics; optional poses come from encoder/FK.

        Default poses are current device_world.roots, never COM or object state.
        Caller must refresh/reset the world before sampling. This method never
        invokes world.refresh(), since that would also query privileged state.
        """
        time_before=float(self._sim.current_time)
        poses=torch.as_tensor(self._state.roots if hand_poses is None else hand_poses,
            device=self.device,dtype=torch.float32).clone()
        frame=None
        initial=self._samples==0
        max_renders=32 if initial else 6
        last_error=None
        for attempt in range(max_renders):
            self._sim.render()
            if abs(float(self._sim.current_time)-time_before)>1e-9:
                raise RuntimeError('vision render unexpectedly advanced physics time')
            if initial and attempt<7:continue
            try:
                candidate=self._reader.read()
            except (ValueError,IndexError) as error:
                last_error=repr(error)
                continue
            if candidate is None:
                last_error='ReferenceTime unavailable'
                continue
            last_error=str(dict(frame=candidate['frame_id'],last_frame=self._last_frame,
                render_time=candidate['simulation_time'],physics_time=time_before,
                rgb_shape=tuple(candidate['rgb'].shape),depth_shape=tuple(candidate['depth'].shape),
                legacy_time=candidate.get('legacy_simulation_time'),
                isaac_read_times=candidate.get('isaac_read_times')))
            if candidate['frame_id']<=self._last_frame:continue
            if not np.isfinite(candidate['simulation_time']) or abs(candidate['simulation_time']-time_before)>1e-6:
                continue
            if candidate['rgb'].shape!=(self.n,self.height,self.width,3) or candidate['depth'].shape!=(self.n,self.height,self.width,1):
                last_error='unexpected native RGB/depth shape';continue
            frame=candidate;break
        if frame is None:
            raise RuntimeError(f'no fresh time-aligned RGB/depth frame after {max_renders} renders; last={last_error}')
        if hand_poses is None and not torch.allclose(poses,torch.as_tensor(self._state.roots,
                device=self.device,dtype=torch.float32),atol=1e-6,rtol=0.):
            raise RuntimeError('palm FK changed during render-only sample')
        rgb=frame['rgb'].to(self.device)[self.slot_order]
        depth=frame['depth'].to(self.device)[self.slot_order]
        result=frames_to_points(rgb,depth,self.K,self.T_world_camera,poses,num_points=self.num_points,
            min_depth_m=self.near,max_depth_m=self.far)
        self._last_frame=int(frame['frame_id']);self._samples+=1
        result.update(frame_id=self._last_frame,simulation_time=time_before,render_passes=attempt+1,
            rendered_simulation_time=float(frame['simulation_time']),
            reference_time_numerator=frame.get('reference_time_numerator',self._last_frame),
            reference_time_denominator=frame.get('reference_time_denominator'),
            render_frame_number=frame.get('render_frame_number'),
            swh_frame_number=frame.get('swh_frame_number'),
            legacy_simulation_time=frame.get('legacy_simulation_time'),
            isaac_read_times=frame.get('isaac_read_times'),
            time_mapping=frame.get('time_mapping','test_reader'))
        return result
