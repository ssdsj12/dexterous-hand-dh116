"""Reproduce externally retained completed frames without reading f_locals."""
from contextlib import contextmanager
import gc
import sys
import weakref
import pytest
from types import SimpleNamespace
from go2_pvcnn.control.dh116 import tabletop_physical_preview as loop
from go2_pvcnn.control.dh116 import tabletop_preview_contract as contract
from go2_pvcnn.control.dh116 import tabletop_tiled_stereo_capture as capture
from go2_pvcnn.control.dh116 import tabletop_tiled_stereo_native as native


@contextmanager
def hold_completed(function):
    frames=[]
    previous=sys.getprofile()
    def profile(frame,event,arg):
        if event=='return' and frame.f_code is function.__code__:
            frames.append(frame)
    sys.setprofile(profile)
    try:
        yield frames
    finally:
        sys.setprofile(previous)
        frames.clear()


class Borrowed:
    def __call__(self,*a,**k):return None


class Episode:
    def pre_action_state(self):return {}


@pytest.mark.parametrize('failing',[False,True])
def test_completed_control_frame_drops_episode_and_borrowed_callbacks(failing):
    refs=[]
    def invoke():
        episode=Episode();observer=Borrowed()
        refs.extend([weakref.ref(episode),weakref.ref(observer)])
        if failing:
            class BrokenEpisode(Episode):
                def pre_action_state(self):raise RuntimeError('pre-state failure')
            episode=BrokenEpisode();refs.append(weakref.ref(episode))
        try:
            loop.run_preview(episode,None,observer,actions=[])
        except RuntimeError:
            assert failing
        finally:
            # f_back retains this fixture too; isolate the production frame.
            episode=observer=None
    with hold_completed(loop.run_preview) as frames:
        invoke();gc.collect()
        assert frames
        assert all(r() is None for r in refs)


def test_camera_factory_completed_frame_drops_borrowed_sim_and_identity(monkeypatch):
    from pathlib import Path
    refs=[]
    monkeypatch.setattr(capture.TiledStereoCapture,'from_native',classmethod(lambda cls,*a,**kw:object()))
    def invoke():
        sim=Borrowed();snapshot=Borrowed()
        refs.extend([weakref.ref(sim),weakref.ref(snapshot)])
        contract.create_preview_camera(Path(__file__).resolve().parents[1],sim,
            dict(camera_config=contract.CAMERA_CONFIG,camera_backend='tiled_step',
                 capture_resolution=dict(width=320,height=240)),snapshot)
        sim=snapshot=None
    with hold_completed(contract.create_preview_camera) as frames:
        invoke();gc.collect()
        assert frames and all(r() is None for r in refs)


def test_capture_factory_completed_frame_drops_sim_reader_and_snapshot(monkeypatch):
    from test_dh116_stereo_rgb_bridge import calibration,RenderSetting
    refs=[]
    class Reader:
        def __init__(self,*a,**kw):refs.append(weakref.ref(self))
        def select(self,side):pass
        def read_clock(self):return None
        def read(self):return None
        def close(self):pass
    monkeypatch.setattr(native,'TiledNativeStereoReader',Reader)
    def invoke():
        sim=Borrowed();snapshot=Borrowed()
        refs.extend([weakref.ref(sim),weakref.ref(snapshot)])
        obj=capture.TiledStereoCapture.from_native(sim,[[0,0,0]],calibration(),snapshot,
            width=320,height=240,render_setting=RenderSetting())
        obj.close()
        obj=sim=snapshot=None
    with hold_completed(capture.TiledStereoCapture.from_native.__func__) as frames:
        invoke();gc.collect()
        assert frames and all(r() is None for r in refs)


def test_failed_native_constructor_completed_frame_drops_sim_without_sdk():
    from test_dh116_stereo_rgb_bridge import calibration
    refs=[]
    def invoke():
        sim=Borrowed();refs.append(weakref.ref(sim))
        with pytest.raises(ValueError,match='requires CUDA'):
            native.TiledNativeStereoReader(sim,[[0,0,0]],calibration(),'cpu')
        sim=None
    with hold_completed(native.TiledNativeStereoReader.__init__) as frames:
        invoke();gc.collect()
        assert frames and refs[0]() is None


def test_completed_owned_scene_frame_drops_hand_rod_reader_and_sim():
    refs=[]
    class Rod:
        rope=object()
        _base=SimpleNamespace(nodes='seed')
        def clock(self):return dict(physics_step=0)
    class Reader:
        def __init__(self,*a,**kw):refs.append(weakref.ref(self))
        def capture_reset(self,*a,**kw):return {}
    class Owners:
        def __init__(self):self.resources={}
        def keep(self,name,value):self.resources[name]=value;return value
    class Journal:
        def record(self,row):pass
    def build(sim,owners,**kw):
        rod=Rod();episode=Episode();episode._record=kw['record']
        refs.extend([weakref.ref(rod),weakref.ref(episode)])
        owners.keep('runtime',rod);owners.keep('dual_episode',episode)
        episode._record(dict(kind='reset',reset=dict(rod_reset=dict(reset_id=1))))
    def invoke():
        sim=SimpleSim();refs.append(weakref.ref(sim));owners=Owners()
        loop.run_owned_scene(sim,owners,SimpleNamespace(watch=lambda *a,**kw:None),
            builder=build,reader_factory=Reader,backend=object(),journal=Journal(),
            observer=lambda *a:None,build_kwargs=dict(episode_uid='cpu-test'),device='cpu',actions=[])
        owners.resources.clear()
        sim=owners=None
    class SimpleSim:
        current_time_step_index=0
        current_time=0.
    with hold_completed(loop.run_owned_scene) as frames:
        invoke();gc.collect()
        assert frames and all(r() is None for r in refs)


@pytest.mark.parametrize('failing',[False,True])
def test_retained_render_budget_constructor_drops_borrowed_sim(failing):
    refs=[]
    class Sim:
        def render(self):pass
    def invoke():
        sim=Borrowed() if failing else Sim();refs.append(weakref.ref(sim))
        try:
            if failing:
                with pytest.raises(AttributeError):contract.RenderBudget(sim,162)
            else:
                budget=contract.RenderBudget(sim,162)
                budget.close()
                budget=None
        finally:sim=None
    with hold_completed(contract.RenderBudget.__init__) as frames:
        invoke();gc.collect()
        assert frames and refs[0]() is None
