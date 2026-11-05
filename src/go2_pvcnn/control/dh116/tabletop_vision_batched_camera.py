"""Bounded real-camera rendering for large frozen batches of physical scenes.

Only B cameras and one B-tile render product exist. Each sample visits every
physical environment, moves the cameras, renders real RGB-D, immediately reduces
images to sensor points, and scatters those points into the final tensor. Neither
teacher images nor object-state features are used. Visibility is a rendering
optimization and does not promise eviction of Hydra geometry from GPU memory.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .tabletop_depth_camera_probe import look_at_optical
from .tabletop_vision_camera import (
    _NativeTiledReader, _numpy, frames_to_points, numeric_slot_order,
)


class EnvironmentVisibility:
    """Keep source env0 visible; toggle only local visibility on its clones.

    Mutating the inherited source can dirty every clone. Pinning source zero
    avoids that broadcast: up to B+1 environments remain visible for B cameras.
    """
    def __init__(self, world):
        from pxr import Sdf, UsdGeom
        stage=world.sim.stage
        attrs=[]
        for path in world.paths:
            prim=stage.GetPrimAtPath(path)
            if not prim.IsValid():raise ValueError(f'environment prim missing: {path}')
            attrs.append(UsdGeom.Imageable(prim).CreateVisibilityAttr())
        self._configure(attrs,change_block=Sdf.ChangeBlock,
            visible_token=UsdGeom.Tokens.inherited,hidden_token=UsdGeom.Tokens.invisible)

    def _configure(self,attrs,*,change_block,visible_token,hidden_token):
        if not attrs:raise ValueError('at least one physical environment is required')
        self._attrs=attrs;self._change_block=change_block
        self._visible_token=visible_token;self._hidden_token=hidden_token
        self.pinned_source_environment=0
        self._visible={self.pinned_source_environment}
        # Every clone gets a stronger local value, including the source env0.
        # The source's visibility is authored once and never changes per group.
        with self._change_block():
            for i,attr in enumerate(self._attrs):
                attr.Set(self._visible_token if i==self.pinned_source_environment else self._hidden_token)

    def select(self,ids):
        selected=set(int(i) for i in ids)
        if any(i<0 or i>=len(self._attrs) for i in selected):
            raise ValueError('visible environment index outside physical batch')
        selected.add(self.pinned_source_environment)
        with self._change_block():
            for i in sorted(self._visible-selected):self._attrs[i].Set(self._hidden_token)
            for i in sorted(selected-self._visible):self._attrs[i].Set(self._visible_token)
        self._visible=selected


def prepare_bounded_visibility(world,batch_size=64):
    """Call after cloning but before the first sim.reset/render for lowest peak.

    This only writes USD visibility. It never changes active state, loading,
    transforms, collision properties, or rigid/articulation membership.
    """
    if type(batch_size) is not int or not 1<=batch_size<=256:
        raise ValueError('bounded camera batch_size must be in [1,256]')
    selector=getattr(world,'_vision_bounded_visibility',None)
    if selector is None:
        selector=EnvironmentVisibility(world)
        world._vision_bounded_visibility=selector
    selector.select(range(min(batch_size,len(world.paths))))
    return selector


class _MovableTiledReader(_NativeTiledReader):
    def __init__(self,**kwargs):
        self._placement=dict(kwargs['config'])
        super().__init__(**kwargs)
        numeric_slot_order(self.prim_paths,self.n)
        self._slot_ids=[int(path.rsplit('_',1)[-1]) for path in self.prim_paths]

    def move_to_origins(self,origins):
        from pxr import Gf, Sdf
        origins=_numpy(origins)
        if origins.shape!=(self.n,3) or not np.isfinite(origins).all():
            raise ValueError('one finite origin per bounded camera slot required')
        config=self._placement
        by_id=[look_at_optical(np.asarray(config['position_world_m'])+origin,
            np.asarray(config['target_world_m'])+origin,config['world_up']) for origin in origins]
        transforms=[]
        with Sdf.ChangeBlock():
            for path,slot in zip(self.prim_paths,self._slot_ids):
                optical=by_id[slot]
                attr=self._stage.GetPrimAtPath(path).GetAttribute('xformOp:transform')
                if not attr.IsValid():
                    raise RuntimeError(f'authored camera transform missing: {path}')
                usd=optical@np.diag([1.,-1.,-1.,1.])
                attr.Set(Gf.Matrix4d(*usd.T.reshape(-1).tolist()))
                transforms.append(optical)
        self.T_world_camera=torch.as_tensor(np.array(transforms),device=self._device,dtype=torch.float32)
        self._expected=by_id
        # Base reader will verify actual CameraView readback after the next render.
        self._calibration_checked=False


class BoundedVisionCamera:
    """Render all N environments through at most B<=256 physical camera prims."""
    def __init__(self,world,device_world,config_path,width=320,height=240,num_points=128,
                 *,batch_size=64,hide_inactive=True,output_device=None,
                 debug_output_dir=None,debug_env_ids=()):
        self._sim,self._state=world.sim,device_world
        self.n=int(device_world.n);self.device=torch.device(device_world.device)
        self.output_device=torch.device(output_device) if output_device is not None else self.device
        if not 1<=self.n<=4096:raise ValueError('bounded sensor supports 1 through 4096 physical environments')
        if type(batch_size) is not int or not 1<=batch_size<=256:
            raise ValueError('bounded camera batch_size must be in [1,256]')
        if not (type(width) is int and type(height) is int and 1<=width<=320 and 1<=height<=240):
            raise ValueError('bounded camera resolution must not exceed 320x240')
        if type(num_points) is not int or not 1<=num_points<=1024:
            raise ValueError('num_points must be an integer in [1,1024]')
        debug_ids=tuple(debug_env_ids)
        if (len(debug_ids)>2 or len(set(debug_ids))!=len(debug_ids)
                or any(type(i) is not int or not 0<=i<self.n for i in debug_ids)
                or (debug_ids and debug_output_dir is None)):
            raise ValueError('debug requires an output directory and at most two unique valid environment IDs')
        self._debug_ids=debug_ids
        self._debug_dir=Path(debug_output_dir) if debug_output_dir is not None else None
        self._debug_saved_files=[]
        self.batch_size=min(batch_size,self.n)
        self.width,self.height,self.num_points=width,height,num_points
        config=json.loads(Path(config_path).read_text())
        self.near,self.far=map(float,config['clipping_range_m'])
        if not (0.<self.near<self.far and 1.<float(config['horizontal_fov_degrees'])<179.
                and config.get('depth_convention')=='optical_axis_z_m'):
            raise ValueError('finite optical-Z pinhole camera configuration required')
        self._origins=torch.as_tensor(device_world.origins,device=self.device).clone()
        if self._origins.shape!=(self.n,3) or not torch.isfinite(self._origins).all():
            raise ValueError('one finite origin per physical environment required')
        self._visibility=prepare_bounded_visibility(world,self.batch_size) if hide_inactive else None
        self._reader=_MovableTiledReader(sim=self._sim,origins=self._origins[:self.batch_size],
            config=config,width=width,height=height,device=self.device)
        self.slot_order=numeric_slot_order(self._reader.prim_paths,self.batch_size).to(self.device)
        self.K=torch.as_tensor(self._reader.K,device=self.device,dtype=torch.float32)[self.slot_order]
        self._last_frame=-1;self._samples=0

    def _check_frozen(self,expected_time,expected_step):
        if abs(float(self._sim.current_time)-expected_time)>1e-9:
            raise RuntimeError('physics time changed during bounded camera observation')
        if expected_step is not None and self._sim.current_time_step_index!=expected_step:
            raise RuntimeError('physics step index changed during bounded camera observation')

    def _read_group(self,expected_time,expected_step,initial):
        last_error=None
        max_renders=32 if initial else 8
        minimum_renders=8 if initial else 2
        for attempt in range(max_renders):
            self._sim.render()
            self._check_frozen(expected_time,expected_step)
            # Never admit the first render immediately after a camera teleport.
            if attempt+1<minimum_renders:continue
            try:frame=self._reader.read()
            except (ValueError,IndexError) as error:
                last_error=repr(error);continue
            if frame is None:
                last_error='render-product timestamp unavailable';continue
            last_error=dict(frame=frame['frame_id'],previous=self._last_frame,
                rendered_time=frame['simulation_time'],physics_time=expected_time,
                isaac_read_times=frame.get('isaac_read_times'))
            if frame['frame_id']<=self._last_frame:continue
            if (not np.isfinite(frame['simulation_time'])
                    or abs(frame['simulation_time']-expected_time)>1e-6):continue
            if (frame['rgb'].shape!=(self.batch_size,self.height,self.width,3)
                    or frame['depth'].shape!=(self.batch_size,self.height,self.width,1)):
                last_error='bounded RGB/depth buffer shape mismatch';continue
            self._last_frame=int(frame['frame_id'])
            return frame,attempt+1
        raise RuntimeError(f'no fresh time-aligned bounded camera frame after {max_renders} renders; last={last_error}')

    @torch.no_grad()
    def sample(self,hand_poses=None):
        time_before=float(self._sim.current_time)
        step_before=getattr(self._sim,'current_time_step_index',None)
        poses=torch.as_tensor(self._state.roots if hand_poses is None else hand_poses,
            device=self.device,dtype=torch.float32).clone()
        if poses.shape!=(self.n,2,7) or not torch.isfinite(poses).all():
            raise ValueError('current palm FK must be finite [N,2,7]')
        output={};frames=[];timestamps=[];ranges=[];wall_seconds=[];total_renders=0;last_timing={}
        for start in range(0,self.n,self.batch_size):
            batch_started=perf_counter()
            stop=min(start+self.batch_size,self.n);count=stop-start
            self._check_frozen(time_before,step_before)
            if self._visibility is not None:self._visibility.select(range(start,stop))
            origins=self._origins[start:stop]
            if count<self.batch_size:
                origins=torch.cat((origins,origins[-1:].expand(self.batch_size-count,-1)))
            self._reader.move_to_origins(origins)
            frame,passes=self._read_group(time_before,step_before,self._samples==0 and start==0)
            # Advanced indexing copies native no-copy annotator memory before reuse.
            rgb=frame['rgb'].to(self.device)[self.slot_order[:count]]
            depth=frame['depth'].to(self.device)[self.slot_order[:count]]
            transform=torch.as_tensor(self._reader.T_world_camera,device=self.device,
                dtype=torch.float32)[self.slot_order[:count]]
            packet=frames_to_points(rgb,depth,self.K[:count],transform,poses[start:stop],
                num_points=self.num_points,min_depth_m=self.near,max_depth_m=self.far)
            for key,value in packet.items():
                if key not in output:
                    output[key]=torch.empty((self.n,*value.shape[1:]),dtype=value.dtype,device=self.output_device)
                output[key][start:stop].copy_(value.to(self.output_device))
            # Optional first-observation evidence contains only actual sensor
            # buffers/calibration and the current palm encoder/FK stand-in.
            # Never consult object pose, contact, or simulator segmentation.
            if self._samples==0:
                for env_id in self._debug_ids:
                    if not start<=env_id<stop:continue
                    local_id=env_id-start
                    self._debug_dir.mkdir(parents=True,exist_ok=True)
                    path=self._debug_dir/f'env_{env_id:05d}_rgbd.npz'
                    np.savez(path,env_id=np.asarray(env_id),
                        rgb=_numpy(rgb[local_id]),depth=_numpy(depth[local_id]),
                        K=_numpy(self.K[local_id]),T_world_camera=_numpy(transform[local_id]),
                        hand_poses=_numpy(poses[env_id]),
                        hand_pose_source=np.asarray('current_palm_encoder_FK_simulation_standin'),
                        selected_pixels_uv=_numpy(packet['selected_pixels_uv'][local_id]),
                        points_in_hands=_numpy(packet['points'][local_id]),
                        point_valid=_numpy(packet['point_valid'][local_id]),
                        image_valid=_numpy(packet['image_valid'][local_id]),
                        object_valid=_numpy(packet['object_valid'][local_id]),
                        frame_id=np.asarray(frame['frame_id']),
                        simulation_time=np.asarray(frame['simulation_time']))
                    self._debug_saved_files.append(str(path))
            frames.append(int(frame['frame_id']));timestamps.append(float(frame['simulation_time']))
            ranges.append([start,stop]);total_renders+=passes
            last_timing={key:value for key,value in frame.items() if key not in ('rgb','depth')}
            del frame,rgb,depth,transform,packet
            wall_seconds.append(perf_counter()-batch_started)
        self._check_frozen(time_before,step_before)
        if hand_poses is None and not torch.allclose(poses,torch.as_tensor(self._state.roots,
                device=self.device,dtype=torch.float32),atol=1e-6,rtol=0.):
            raise RuntimeError('palm FK changed during bounded camera observation')
        self._samples+=1
        output.update(last_timing)
        output.update(frame_id=frames[-1],simulation_time=time_before,rendered_simulation_time=timestamps[-1],
            render_passes=total_renders,camera_count=self.batch_size,render_batches=len(frames),
            camera_frame_first=frames[0],camera_frame_last=frames[-1],
            batch_frame_ids=frames,batch_rendered_simulation_times=timestamps,
            batch_wall_seconds=wall_seconds,debug_saved_files=list(self._debug_saved_files),
            batch_environment_ranges=ranges,physics_frozen=True,
            inactive_environment_visibility=self._visibility is not None,
            pinned_source_environment=(self._visibility.pinned_source_environment
                if self._visibility is not None else None))
        return output
