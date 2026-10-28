import json
import pytest
from types import SimpleNamespace
import torch


def fixture():
    t=SimpleNamespace(phase=torch.tensor([1,3]),offset=torch.tensor([.004,.024]),elapsed=torch.zeros(2),
        m={'contact_valid':torch.tensor([[False,True],[True,True]]),
           'contact_center_relative_m':torch.tensor([[[float('nan')]*3,[0.,0.,-.001]],[[0.,0.,-.002],[0.,0.,-.003]]]),
           'hand_object_loads':torch.tensor([[0.,2.],[3.,4.]]),
           'vertical_hand_force_n':torch.tensor([[0.,.2],[.3,.4]])},
        f_target=torch.tensor([[4.,4.],[4.,4.]]),slip=torch.tensor([0.,.003]),
        lift_diagnostic_masks=torch.zeros((2,8),dtype=torch.bool))
    roots=torch.zeros((2,2,7));roots[:,:,2]=.4
    w=SimpleNamespace(initial_offset=.004,roots=roots,root_targets=roots+.001,
        origins=torch.zeros((2,3)))
    return SimpleNamespace(task=t,world=w,episodes=torch.zeros(2,dtype=torch.long))


def test_motion_snapshot_separates_command_actual_and_missing_contact():
    from go2_pvcnn.control.dh116.tabletop_teacher_motion import motion_snapshot
    row=motion_snapshot(fixture())
    assert row['lift_command_m']==[0.,pytest.approx(.02,abs=1e-7)]
    assert row['palm_actual_z_m'][0]==[pytest.approx(.4,abs=1e-7)]*2
    assert row['contact_center_relative_m'][0][0] is None
    assert row['contact_center_relative_m'][0][1][2]==pytest.approx(-.001,abs=1e-7)
    json.dumps(row,allow_nan=False)


def test_nonfinite_fault_snapshot_does_not_invent_finite_measurements():
    from go2_pvcnn.control.dh116.tabletop_teacher_motion import motion_snapshot
    e=fixture();e.world.roots[0,0,2]=float('inf');e.task.slip[0]=float('nan')
    row=motion_snapshot(e)
    assert row['palm_actual_z_m'][0][0] is None and row['relative_slip_m'][0] is None
    json.dumps(row,allow_nan=False)


def test_motion_snapshot_exposes_hold_and_recovery_when_available():
    from go2_pvcnn.control.dh116.tabletop_teacher_motion import motion_snapshot
    e=fixture();e.task.close_grasp_hold=SimpleNamespace(active=torch.ones((2,2),dtype=torch.bool),ready=torch.zeros((2,2),dtype=torch.bool))
    e.task.held_safety_correction=torch.tensor([[True,False],[False,False]])
    r=motion_snapshot(e)
    assert r['close_holding'][0]==[True,True]
    assert r['close_ready'][0]==[False,False]
    assert r['held_safety_correction'][0]==[True,False]


def test_view_uses_world_contact_centers_not_missing_task_key(monkeypatch):
    import sys
    from go2_pvcnn.control.dh116.tabletop_teacher_motion import TeacherMotionView,motion_snapshot
    e=fixture();e.world.contact_center_w=torch.tensor([[[float('nan')]*3,[.1,.2,.3]],[[0.,0.,0.],[0.,0.,0.]]])
    monkeypatch.setitem(sys.modules,'pxr',SimpleNamespace(Gf=SimpleNamespace(Vec3d=lambda *v:tuple(v))))
    class Marker:
        def MakeVisible(self):self.visible=True
        def MakeInvisible(self):self.visible=False
        def Set(self,value):self.position=value
    view=TeacherMotionView.__new__(TeacherMotionView);left=Marker();right=Marker()
    view.markers=[(left,left),(right,right)];view.status=SimpleNamespace(text='')
    view.update(e,motion_snapshot(e))
    assert not left.visible and right.visible
    assert right.position==pytest.approx((.1,.2,.3))
    assert 'CLOSE' in view.status.text


def test_motion_status_identifies_env_episode_sim_time_and_recovery(monkeypatch):
    import sys
    from go2_pvcnn.control.dh116.tabletop_teacher_motion import TeacherMotionView,motion_snapshot
    e=fixture();e.task.phase[0]=3;e.episodes[:]=torch.tensor([0,7]);e.task.elapsed=torch.tensor([1.234,9.])
    e.task.lift_grasp_hold=SimpleNamespace(recovering=torch.tensor([True,False]))
    e.world.contact_center_w=torch.zeros((2,2,3))
    monkeypatch.setitem(sys.modules,'pxr',SimpleNamespace(Gf=SimpleNamespace(Vec3d=lambda *v:tuple(v))))
    class Marker:
        def MakeVisible(self):pass
        def MakeInvisible(self):pass
        def Set(self,value):pass
    marker=Marker();view=TeacherMotionView.__new__(TeacherMotionView)
    view.markers=[(marker,marker),(marker,marker)];view.status=SimpleNamespace(text='')
    row=motion_snapshot(e)
    assert row['episode']==[0,7]  # Persisted identifiers remain zero based.
    assert row['elapsed_s']==pytest.approx([1.234,9.])
    view.update(e,row)
    assert 'Env 0 | Episode 1' in view.status.text
    assert 'Sim 1.23 s' in view.status.text
    assert 'LIFT / GRIP RECOVERY' in view.status.text
    e.task.lift_grasp_hold.recovering.zero_();view.update(e,motion_snapshot(e))
    assert 'GRIP RECOVERY' not in view.status.text and 'LIFT' in view.status.text
