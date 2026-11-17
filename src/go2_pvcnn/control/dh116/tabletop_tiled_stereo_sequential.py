"""Independent sequential-view capture fenced by one physical state identity.

Renderer clocks remain raw and separate. This software transaction is not
native synchronisation qualification and never grants Teacher/Student admission.
"""
import copy
import math
import re
import torch
from .tabletop_stereo_calibration import StereoCalibration,rerender_calibration
from .tabletop_stereo_validation import validate_identity,validate_images
from .tabletop_stereo_rgb_bridge import _IDENTITY_KEYS,_RAW_KEYS,_same_snapshot,_NativeRenderSetting

VERSION="dh116-tiled-step-stereo-v1"
_CLOCK_KEYS={"frame_id","simulation_time","render_frame_number"}
_FRAME_KEYS={"rgb","K","T_world","raw_clock","native_evidence","raw_time_metadata"}


def _clock(value):
    if (not isinstance(value,dict) or set(value)!=_CLOCK_KEYS
            or any(type(value[k]) is not int or value[k]<0 for k in ("frame_id","render_frame_number"))
            or isinstance(value["simulation_time"],bool)
            or not isinstance(value["simulation_time"],(int,float))
            or not math.isfinite(value["simulation_time"]) or value["simulation_time"]<0):
        raise ValueError("invalid raw render clock")
    return copy.deepcopy(value)


class TiledStereoCapture:
    def __init__(self,sim,reader,calibration,snapshot,*,render_setting,max_render_attempts=4,origins=None):
        if not isinstance(calibration,StereoCalibration) or not callable(snapshot):
            raise ValueError("calibration and physical snapshot required")
        if type(max_render_attempts) is not int or not 1<=max_render_attempts<=16:
            raise ValueError("bounded per-view render attempts required")
        if any(not callable(getattr(reader,n,None)) for n in ("select","read_clock","read","close")):
            raise ValueError("owned sequential camera reader required")
        if any(not callable(getattr(render_setting,n,None)) for n in ("get","set")):
            raise ValueError("render setting get/set required")
        self.sim,self.reader,self.calibration,self.snapshot=sim,reader,calibration,snapshot
        self.render_setting,self.max_render_attempts,self.origins=render_setting,max_render_attempts,origins
        self.last_capture_id=0;self.last_epoch=None;self.last_episode=None
        self._closed=False;self._close_error=None

    @classmethod
    def from_native(cls,sim,origins,calibration,snapshot,*,width=640,height=480,
                    horizontal_fov_degrees=75.,device="cuda:0",render_setting=None,
                    max_render_attempts=4,max_readback_bytes=1024**3):
        try:
            from .tabletop_tiled_stereo_native import SequentialNativeStereoReader as TiledNativeStereoReader
            if not callable(snapshot):raise ValueError("physical snapshot callback required")
            if type(max_render_attempts) is not int or not 1<=max_render_attempts<=16:
                raise ValueError("bounded per-view render attempts required")
            cal=rerender_calibration(calibration,width,height,horizontal_fov_degrees=horizontal_fov_degrees)
            setting=_NativeRenderSetting() if render_setting is None else render_setting
            if (any(not callable(getattr(setting,n,None)) for n in ("get","set"))
                    or type(setting.get()) is not bool):
                raise ValueError("readable boolean render setting required")
            reader=TiledNativeStereoReader(sim,origins,cal,device,max_readback_bytes=max_readback_bytes)
            try:
                return cls(sim,reader,cal,snapshot,render_setting=setting,origins=origins,
                           max_render_attempts=max_render_attempts)
            except BaseException:
                reader.close()
                raise
        finally:
            sim = reader = snapshot = render_setting = setting = None

    def _physical(self):
        before=copy.deepcopy(self.snapshot())
        rows=before if isinstance(before,list) else [before]
        if not rows or any(not isinstance(r,dict) or set(r)!=_IDENTITY_KEYS|_RAW_KEYS for r in rows):
            raise ValueError("physical identity fields missing or unexpected")
        step=self.sim.current_time_step_index;stamp=float(self.sim.current_time)
        if type(step) is not int or step<0 or not math.isfinite(stamp) or stamp<0:
            raise ValueError("invalid physical SDK clock")
        validate_identity(rows,len(rows),stamp)
        for row in rows:
            if (any(type(row[k]) is not int or row[k]<0 for k in ("sdk_step","rod_step","rod_tick"))
                    or any(isinstance(row[k],bool) or not isinstance(row[k],(int,float))
                           or not math.isfinite(row[k]) or row[k]<0 for k in ("sdk_timestamp_s","solver_time_s"))
                    or row["sdk_step"]!=step or abs(row["sdk_timestamp_s"]-stamp)>1e-6
                    or row["calibration_sha256"]!=self.calibration.sha256
                    or not isinstance(row["state_sha256"],str) or not re.fullmatch("[0-9a-f]{64}",row["state_sha256"])
                    or any(not isinstance(row[k],str) or not row[k] for k in ("rod_uid","material_reference_uid"))):
                raise ValueError("invalid physical/rod state binding")
        return before,rows,step,stamp

    def capture(self):
        if self._closed:raise RuntimeError("tiled capture is closed")
        original=self.render_setting.get()
        if type(original) is not bool:raise ValueError("readable boolean render setting required")
        try:
            self.render_setting.set(False)
            if self.render_setting.get() is not False:raise RuntimeError("physics freeze setting failed")
            packet=self._capture()
        finally:
            try:self.reader.select("left")
            finally:
                self.render_setting.set(original)
                if self.render_setting.get() is not original:raise RuntimeError("render setting restore failed")
        timing=packet["render_timing"]
        if (type(self.sim.current_time_step_index) is not int
                or self.sim.current_time_step_index!=timing["physical_step"]
                or float(self.sim.current_time)!=timing["physical_timestamp_s"]
                or not _same_snapshot(self.snapshot(),packet["identity"])):
            raise RuntimeError("physical state changed during camera restoration")
        rows=packet["identity"] if isinstance(packet["identity"],list) else [packet["identity"]]
        self.last_capture_id=timing["capture_id"]
        self.last_epoch=timing["raw_right_clock"]["render_frame_number"]
        self.last_episode=tuple((r["run_uuid"],r["env_id"],r["episode_uid"],r["reset_id"]) for r in rows)
        return packet

    def _capture(self):
        before,rows,step,stamp=self._physical()
        episode=tuple((r["run_uuid"],r["env_id"],r["episode_uid"],r["reset_id"]) for r in rows)
        previous=self.last_epoch if episode==self.last_episode else None
        def check():
            if (type(self.sim.current_time_step_index) is not int
                    or self.sim.current_time_step_index!=step or float(self.sim.current_time)!=stamp
                    or not _same_snapshot(self.snapshot(),before)):
                raise RuntimeError("physical state changed during tiled stereo capture")
        frames={};render_calls=0
        for side in ("left","right"):
            self.reader.select(side);check()
            for _ in range(self.max_render_attempts):
                prior=self.reader.read_clock();check()
                self.sim.render();render_calls+=1
                self.render_setting.set(False)
                check()
                current=self.reader.read_clock();check()
                if prior is None or current is None:continue
                prior,current=_clock(prior),_clock(current)
                if current["render_frame_number"]<=prior["render_frame_number"]:continue
                if previous is not None and current["render_frame_number"]<=previous:continue
                packet=self.reader.read();check()
                if packet is None:continue
                if (not isinstance(packet,dict) or not set(packet)<=_FRAME_KEYS
                        or not (_FRAME_KEYS-{"raw_time_metadata"})<=set(packet)):
                    raise ValueError("reader returned forbidden or missing fields")
                if (_clock(packet["raw_clock"])!=current
                        or _clock(self.reader.read_clock())!=current):
                    raise ValueError("raw render clock changed during image read")
                if type(packet["native_evidence"]) is not bool:raise ValueError("explicit reader source required")
                frames[side]={**packet,"rgb":packet["rgb"].clone(),
                    "K":torch.as_tensor(packet["K"]).clone(),"T_world":torch.as_tensor(packet["T_world"]).clone()}
                previous=current["render_frame_number"]
                break
            else:raise RuntimeError("fresh synchronized view unavailable within bounded renders")
        left,right=frames["left"],frames["right"]
        validate_images(left["rgb"],right["rgb"],self.calibration)
        if len(rows)!=left["rgb"].shape[0]:raise ValueError("identity/camera batch mismatch")
        self.calibration.validate_readback(left["K"],right["K"],left["T_world"],right["T_world"],origins=self.origins)
        check()
        capture_id=self.last_capture_id+1
        timing=dict(capture_id=capture_id,raw_left_clock=left["raw_clock"],raw_right_clock=right["raw_clock"],
            camera_pose_backend=getattr(self.reader,"pose_backend","unspecified"),
            raw_left_metadata=str(left.get("raw_time_metadata","unavailable"))[:4096],
            raw_right_metadata=str(right.get("raw_time_metadata","unavailable"))[:4096],
            physical_timestamp_s=stamp,physical_step=step,
            clock_mapping="separate_raw_clocks_with_frozen_physics_boundary",
            native_synchronization_qualified=False)
        return dict(left_rgb=left["rgb"],right_rgb=right["rgb"],identity=before,
            calibration_sha256=self.calibration.sha256,K_left=left["K"],K_right=right["K"],
            render_timing=timing,render_passes=render_calls,
            source_scope="native_tiled_reader" if left["native_evidence"] and right["native_evidence"] else "cpu_fixture")

    def close(self):
        if self._closed:
            if self._close_error:raise RuntimeError(self._close_error)
            return
        self._closed=True
        try:self.reader.close()
        except BaseException as error:
            self._close_error=str(error)
            raise
        finally:self.sim=self.reader=self.snapshot=self.render_setting=self.origins=None
