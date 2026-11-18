"""Optional complete native buffer snapshots do not alter legacy sensor reads."""
import torch

from test_native_sensor_batch import world,put
from go2_pvcnn.control.dh116.tabletop_native_sensor_batch import read_native_sensors


def test_default_reader_does_not_add_tripod_buffer_capture():
    w=world();put(w)
    result=read_native_sensors(w)
    assert 'tripod_raw_contacts' not in result and 'tripod_raw_friction' not in result


def test_opt_in_captures_owned_normal_and_friction_buffers_with_original_indices():
    w=world();put(w);w.capture_tripod_contact_evidence=True
    out=read_native_sensors(w)
    assert len(out.get('tripod_raw_contacts',[]))==34
    assert len(out.get('tripod_raw_friction',[]))==34
    oldc=[x.clone() for x in out['tripod_raw_contacts'][0]]
    oldf=[x.clone() for x in out['tripod_raw_friction'][0]]
    view=w.sensors[0].contact_physx_view
    for x in (*view.contact,*view.friction):x.fill_(7)
    for x,expected in zip(out['tripod_raw_contacts'][0],oldc):
        torch.testing.assert_close(x,expected,rtol=0,atol=0,equal_nan=True)
    for x,expected in zip(out['tripod_raw_friction'][0],oldf):
        torch.testing.assert_close(x,expected,rtol=0,atol=0,equal_nan=True)
    assert int(out['tripod_raw_contacts'][0][4][0])==2
    assert int(out['tripod_raw_friction'][0][2][0])==1
    assert int(out['tripod_raw_friction'][0][3][0])==4


def test_partial_reset_invalidates_selected_raw_without_mutating_saved_pre_reset_record():
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_tripod_execution_guard import invalidate_native_contact_snapshot
    old=dict(complete=True,valid_envs=torch.ones(3,dtype=torch.bool),
        raw_contacts=[torch.tensor([1.])],raw_friction=[torch.tensor([2.])],
        tensor_world_physics_steps=17)
    w=SimpleNamespace(n=3,device='cpu',tripod_contact_snapshot=old)
    invalidate_native_contact_snapshot(w,torch.tensor([1]))
    assert old['complete'] is True and old['valid_envs'].all()
    assert w.tripod_contact_snapshot is not old
    assert not w.tripod_contact_snapshot['complete']
    assert w.tripod_contact_snapshot['valid_envs'].tolist()==[True,False,True]
    assert w.tripod_contact_snapshot['tensor_world_physics_steps']==17
