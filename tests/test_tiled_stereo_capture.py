import copy
import importlib
import pytest
import torch
from test_dh116_stereo_rgb_bridge import calibration,state,RenderSetting

def harness(size=(640,480)):
    from go2_pvcnn.control.dh116.tabletop_stereo_calibration import rerender_calibration
    c=rerender_calibration(calibration(),*size)
    identity=state(c);events=[]
    class Sim:
        current_time=1.
        current_time_step_index=4
        def __init__(self):self.epoch=10
        def render(self):self.epoch+=1;events.append("render")
    sim=Sim()
    class Reader:
        side="left"
        closed=False
        def select(self,side):self.side=side;events.append("select:"+side)
        def read_clock(self):
            return dict(frame_id=sim.epoch,render_frame_number=sim.epoch,
                        simulation_time=sim.epoch/30.)
        def read(self):
            events.append("read:"+self.side)
            t=getattr(c,"T_world_"+self.side)[None].clone()
            return dict(rgb=torch.full((1,3,c.height,c.width),.2 if self.side=="left" else .7),
                K=getattr(c,"K_"+self.side)[None].clone(),T_world=t,
                raw_clock=self.read_clock(),native_evidence=False)
        def close(self):self.closed=True
    reader=Reader()
    cls=getattr(importlib.import_module("go2_pvcnn.control.dh116.tabletop_tiled_stereo_capture"),"TiledStereoCapture")
    setting=RenderSetting()
    capture=cls(sim,reader,c,lambda:copy.deepcopy(identity),render_setting=setting,max_render_attempts=2)
    return capture,sim,reader,identity,events,setting

def test_pair_keeps_raw_times_and_physics_identity_without_claiming_same_clock_domain():
    cap,sim,reader,identity,events,setting=harness()
    packet=cap.capture()
    assert packet["identity"]==identity
    assert events==["select:left","render","read:left","select:right","render","read:right","select:left"]
    timing=packet["render_timing"]
    assert timing["raw_left_clock"]["simulation_time"]==11/30.
    assert timing["raw_right_clock"]["simulation_time"]==12/30.
    assert timing["physical_timestamp_s"]==1.
    assert timing["clock_mapping"]=="separate_raw_clocks_with_frozen_physics_boundary"
    assert timing["native_synchronization_qualified"] is False
    assert "simulation_time" not in packet and "frame_id" not in packet
    assert torch.all(packet["left_rgb"]==.2) and torch.all(packet["right_rgb"]==.7)
    assert reader.side=="left" and setting.get() is False

@pytest.mark.parametrize("change",["sdk_step","sdk_time","rod_tick","reset_id","episode_uid","state_sha256"])
def test_state_change_during_either_view_is_rejected(change):
    cap,sim,reader,identity,events,setting=harness()
    render=sim.render
    def move():
        render()
        if reader.side=="right":
            if change=="sdk_step":sim.current_time_step_index+=1
            elif change=="sdk_time":sim.current_time+=.001
            elif change in ("rod_tick","reset_id"):identity[change]+=1
            elif change=="episode_uid":identity[change]="other"
            else:identity[change]="b"*64
    sim.render=move
    with pytest.raises(RuntimeError,match="physical"):cap.capture()
    assert reader.side=="left" and setting.get() is False
    assert cap.last_capture_id==0

def test_new_render_and_stale_image_metadata_are_distinguished():
    cap,sim,reader,*_=harness()
    read=reader.read
    def stale():
        row=read();row["raw_clock"]["render_frame_number"]-=1;return row
    reader.read=stale
    with pytest.raises(ValueError,match="clock"):cap.capture()
    assert cap.last_capture_id==0

def test_unchanging_render_epoch_exhausts_bound_without_returning_a_pair():
    cap,sim,reader,identity,events,setting=harness()
    sim.render=lambda:events.append("render")
    with pytest.raises(RuntimeError,match="fresh"):cap.capture()
    assert events.count("render")==2
    assert cap.last_capture_id==0

def test_wrong_right_pose_rejected_and_next_success_is_not_poisoned():
    cap,sim,reader,identity,events,setting=harness()
    read=reader.read
    def wrong():
        row=read()
        if reader.side=="right":row["T_world"][:,0,3]+=.01
        return row
    reader.read=wrong
    with pytest.raises(ValueError,match="calibration"):cap.capture()
    assert cap.last_capture_id==0
    reader.read=read
    assert cap.capture()["render_timing"]["capture_id"]==1

@pytest.mark.parametrize("size",[(224,96),(320,240),(640,360),(640,480),(512,384)])
def test_four_sizes_and_custom_size_use_matching_native_shapes(size):
    cap,*_=harness(size)
    packet=cap.capture()
    assert packet["left_rgb"].shape==(1,3,size[1],size[0])
    assert packet["right_rgb"].shape==packet["left_rgb"].shape

def test_reader_cannot_inject_depth_or_privileged_tensor():
    cap,sim,reader,*_=harness()
    read=reader.read
    def inject():
        row=read();row["depth"]=torch.ones(1);return row
    reader.read=inject
    with pytest.raises(ValueError,match="fields"):cap.capture()

def test_capture_read_failure_still_restores_pose_setting_and_no_partial_pair():
    cap,sim,reader,identity,events,setting=harness()
    def fail():raise RuntimeError("read failed")
    reader.read=fail
    with pytest.raises(RuntimeError,match="read failed"):cap.capture()
    assert reader.side=="left" and setting.get() is False and cap.last_capture_id==0

def test_close_releases_owned_reader_once_and_rejects_use_after_close():
    cap,sim,reader,*_=harness()
    cap.close();cap.close()
    assert reader.closed and cap.sim is None and cap.snapshot is None
    with pytest.raises(RuntimeError,match="closed"):cap.capture()

def test_restore_failure_cannot_commit_a_successful_pair_identity():
    cap,sim,reader,identity,events,setting=harness()
    select=reader.select
    left_calls=[]
    def broken_restore(side):
        if side=="left":
            left_calls.append(1)
            if len(left_calls)>1:raise RuntimeError("restore pose failed")
        return select(side)
    reader.select=broken_restore
    with pytest.raises(RuntimeError,match="restore pose"):cap.capture()
    assert cap.last_capture_id==0 and cap.last_epoch is None

def test_physical_change_during_pose_restore_is_rejected_before_commit():
    cap,sim,reader,identity,events,setting=harness()
    select=reader.select
    left_calls=[]
    def move_on_restore(side):
        if side=="left":
            left_calls.append(1)
            if len(left_calls)>1:sim.current_time_step_index+=1
        return select(side)
    reader.select=move_on_restore
    with pytest.raises(RuntimeError,match="physical"):cap.capture()
    assert cap.last_capture_id==0

def test_episode_reset_clears_old_render_epoch_without_reusing_pair_identity():
    cap,sim,reader,identity,events,setting=harness()
    first=cap.capture()
    identity["reset_id"]+=1;identity["episode_uid"]="new-episode";sim.epoch=0
    second=cap.capture()
    assert first["render_timing"]["capture_id"]==1
    assert second["render_timing"]["capture_id"]==2
    assert second["identity"]["episode_uid"]=="new-episode"

def test_native_factory_rejects_invalid_snapshot_before_any_native_reader(monkeypatch):
    import go2_pvcnn.control.dh116.tabletop_tiled_stereo_native as native
    from go2_pvcnn.control.dh116.tabletop_tiled_stereo_capture import TiledStereoCapture
    def unexpected(*a,**kw):raise AssertionError("native resources created for invalid request")
    monkeypatch.setattr(native,"TiledNativeStereoReader",unexpected)
    with pytest.raises(ValueError,match="snapshot"):
        TiledStereoCapture.from_native(object(),[[0.,0.,0.]],calibration(),None,render_setting=RenderSetting())
