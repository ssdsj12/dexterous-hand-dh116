import torch
from test_dh116_policy_control import enabled


def controller(t):
    from go2_pvcnn.control.dh116.tabletop_contact_exploration import ContactExplorer
    return ContactExplorer(t,seed=42)


def ready(t):
    t.m['contact_valid'][:]=True
    t.m['contact_center_relative_m'][:]=0.
    t.m['contact_normal_w'][:]=0.
    t.m['vertical_hand_force_n'][:]=1.
    t.m['object_velocity'][:]=0.
    t.slip[:]=0.


def test_ready_feedback_requests_lift_on_first_call_without_elapsed_wait():
    t=enabled();ready(t);c=controller(t)
    a=c.actions(t,1/60)
    assert c.lifting.all()
    assert (a[:,[14,17]]>0).all()
    assert (t.elapsed==0).all()


def test_missing_contact_explores_and_contact_geometry_changes_next_action():
    t=enabled();t.m['contact_valid'][:]=False
    c=controller(t);a=c.actions(t,1/60)
    assert not c.lifting.any()
    assert (a[:,12:18]!=0).any()
    t.m['contact_valid'][:]=True;t.m['contact_center_relative_m'][:]=0.
    t.m['contact_center_relative_m'][:,:,2]=.002
    t.m['contact_normal_w'][:]=0.;t.m['vertical_hand_force_n'][:]=-1.
    b=c.actions(t,1/60)
    assert not c.lifting.any()
    assert (b[:,[14,17]]<a[:,[14,17]]).all()


def test_selective_reset_releases_lift_anchor_only_for_ended_row():
    t=enabled();ready(t);c=controller(t);c.actions(t,1/60)
    c.reset(torch.tensor([True,False]))
    assert c.lifting.tolist()==[False,True]


def test_bad_contact_or_moving_object_never_admits_lift():
    t=enabled();ready(t);t.m['object_velocity'][0,0]=.1
    t.m['vertical_hand_force_n'][1,0]=-1.
    c=controller(t);a=c.actions(t,1/60)
    assert not c.lifting.any() and torch.isfinite(a).all()


def test_policy_task_enters_lift_without_legacy_dwell():
    t=enabled();ready(t)
    t.begin_step(torch.zeros((2,20)))
    t.consume(t.m,~t.done,1/480,vertical_force_n=torch.ones(2),lift_vertical_force_n=torch.ones(2))
    assert t.phase.tolist()==[3,3]
