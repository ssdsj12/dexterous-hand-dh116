import copy
import torch
import pytest
from test_dh116_stereo_rgb_bridge import calibration,state,RenderSetting


def fixture():
    from go2_pvcnn.control.dh116.tabletop_stereo_calibration import rerender_calibration
    from go2_pvcnn.control.dh116.tabletop_tiled_stereo_capture import TiledStereoCapture
    c=rerender_calibration(calibration(),640,480)
    identity=state(c);events=[]
    class Sim:
        current_time=1.
        current_time_step_index=4
        epoch=100
        def render(self):
            self.epoch+=1
            events.append(('render',self.epoch))
    sim=Sim()
    class Reader:
        closed=False
        def read_clock(self):
            return dict(frame_id=sim.epoch,render_frame_number=sim.epoch,simulation_time=1.)
        def read_pair(self):
            events.append(('read_pair',sim.epoch))
            clock=self.read_clock()
            def one(side,value):
                return dict(rgb=torch.full((1,3,480,640),value),K=getattr(c,'K_'+side)[None],
                    T_world=getattr(c,'T_world_'+side)[None],raw_clock=clock,
                    native_evidence=False,raw_time_metadata='cpu fixture')
            return dict(left=one('left',.2),right=one('right',.7))
        def close(self):self.closed=True
    reader=Reader();setting=RenderSetting()
    capture=TiledStereoCapture(sim,reader,c,lambda:copy.deepcopy(identity),
        render_setting=setting,origins=[[0.,0.,0.]])
    return capture,sim,reader,identity,events,setting


def test_fixed_dual_capture_renders_once_and_reads_both_cameras_without_pose_switch():
    cap,sim,reader,identity,events,setting=fixture()
    packet=cap.capture()
    assert events==[('render',101),('read_pair',101)]
    assert packet['render_passes']==1
    assert packet['render_timing']['raw_left_clock']==packet['render_timing']['raw_right_clock']
    assert packet['identity']==identity
    assert torch.all(packet['left_rgb']==.2) and torch.all(packet['right_rgb']==.7)
    assert setting.get() is False


def test_fixed_dual_capture_rejects_reader_without_paired_readback():
    cap,sim,reader,identity,events,setting=fixture()
    reader.read_pair=None
    with pytest.raises(ValueError,match='paired'):
        from go2_pvcnn.control.dh116.tabletop_tiled_stereo_capture import TiledStereoCapture
        TiledStereoCapture(sim,reader,cap.calibration,lambda:identity,render_setting=setting)


def test_native_pair_owns_two_distinct_fixed_cameras_and_closes_both(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_tiled_stereo_native as native
    created=[]
    class Camera:
        def __init__(self,*args,**kw):
            from types import SimpleNamespace
            self.resources=SimpleNamespace(detach=lambda:None)
            self.kw=kw;self.closed=False;created.append(self)
        def read_clock(self):return dict(frame_id=5,simulation_time=0.,render_frame_number=5)
        def read(self):return dict(raw_clock=self.read_clock())
        def close(self):self.closed=True
    monkeypatch.setattr(native,'FixedNativeCameraReader',Camera)
    reader=native.TiledNativeStereoReader(None,None,None,'cuda')
    assert [x.kw['side'] for x in created]==['left','right']
    assert created[0].kw['rig_path']!=created[1].kw['rig_path']
    assert set(reader.read_pair())=={'left','right'}
    reader.close()
    assert all(x.closed for x in created)


def test_fixed_native_camera_refuses_pose_change_after_initialization():
    from go2_pvcnn.control.dh116.tabletop_tiled_stereo_native import FixedNativeCameraReader
    reader=object.__new__(FixedNativeCameraReader)
    reader._native_initialized=True
    with pytest.raises(RuntimeError,match='cannot be switched'):reader.select('right')


def test_pair_clock_mismatch_cannot_be_reported_as_synchronized(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_tiled_stereo_native as native
    class Camera:
        def __init__(self,*args,**kw):
            from types import SimpleNamespace
            self.resources=SimpleNamespace(detach=lambda:None)
            self.side=kw['side']
        def read_clock(self):return dict(frame_id=5,simulation_time=0.,render_frame_number=5 if self.side=='left' else 6)
        def close(self):pass
    monkeypatch.setattr(native,'FixedNativeCameraReader',Camera)
    reader=native.TiledNativeStereoReader(None,None,None,'cuda')
    with pytest.raises(ValueError,match='different raw clocks'):reader.read_pair()
    reader.close()


def test_detach_both_graphs_before_destroying_either_render_product():
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_tiled_stereo_native import TiledNativeStereoReader
    events=[]
    reader=object.__new__(TiledNativeStereoReader)
    reader._closed=False;reader._close_error=None
    reader.cameras={side:SimpleNamespace(
        resources=SimpleNamespace(detach=lambda side=side:events.append('detach:'+side)),
        close=lambda side=side:events.append('destroy:'+side)) for side in ('left','right')}
    reader.close()
    assert events==['detach:left','detach:right','destroy:left','destroy:right']
