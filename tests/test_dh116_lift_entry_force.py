import pytest
import torch
from test_dh116_close_downward_penalty import task
from test_dh116_tensor_contact_window import loads
from test_dh116_tensor_env import World
from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv


def test_event_force_only_boundary_sample_not_later_or_nontransition_rows():
    from go2_pvcnn.control.dh116.tabletop_vertical_force import LiftEntryForceStats
    t=task(2);w=World(t);w.tick=100;w.m=loads(t,4.)
    w.object_weight_n=torch.ones(2);w.vertical_components=torch.zeros((2,2,2),dtype=torch.float64)
    t.contact_time[0]=.2-w.dt
    advance=w.advance
    def change_force(*args):
        # Direct lift now requires nonnegative aggregate vertical force; the
        # boundary sample remains the one recorded for lift-entry telemetry.
        m=advance(*args);w.vertical_components[:,0,0]=2. if w.tick==101 else 100.
        return m
    w.advance=change_force
    e=TensorEnv(w,t);e.lift_entry_force_stats=LiftEntryForceStats('cpu')
    _,_,_,extra=e.step(torch.zeros((2,20)))
    row=e.lift_entry_force_stats.summary()
    assert row['sample_count']==1 and row['hands_total_mean_n']==2.
    assert row['downward_mean_n']==0. and row['upward_mean_n']==2.
    assert row['sampling']=='close_to_lift_transition_only'
    assert extra['terms']['close_downward'].tolist()==[0.,0.]
    e.step(torch.zeros((2,20)))
    assert e.lift_entry_force_stats.summary()['sample_count']==1


def test_no_event_is_missing_force_not_zero_force_and_no_scalar_point():
    from go2_pvcnn.control.dh116.tabletop_vertical_force import LiftEntryForceStats,write_lift_entry_force_scalars
    s=LiftEntryForceStats('cpu');row=s.summary()
    assert row['sample_count']==0 and row['hands_total_mean_n'] is None
    class Writer:
        def __init__(self):self.points={}
        def add_scalar(self,name,value,step):self.points[name]=value
    w=Writer();write_lift_entry_force_scalars(w,row,1)
    assert w.points=={'lift_entry_force/事件数':0}
    v=torch.zeros((2,2,2),dtype=torch.float64);v[:,0,0]=torch.tensor([2.,-4.])
    s.add(v,torch.ones(2),torch.ones(2,dtype=torch.bool))
    row=s.summary();write_lift_entry_force_scalars(w,row,2)
    assert w.points['lift_entry_event_summary_N/向上力_平均']==1.
    assert w.points['lift_entry_event_summary_N/向下力_平均']==2.
    assert w.points['lift_entry_event_summary_N/竖直合力_向上为正']==-1.
    s.clear();assert s.summary()['hands_total_mean_n'] is None
