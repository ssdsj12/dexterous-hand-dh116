"""Host/tensor startup contract for the applied, rather than measured, palm offset."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_dh116_gpu_batch_reset import make_physical_world
from go2_pvcnn.control.dh116.tabletop_tensor_training import validate_world_adapter


def host_world(mounted=True):
    w=make_physical_world();w.flex=None;w.mounts=[] if mounted else None
    w.layout['wire_center']=np.array([0.,0.,.308])
    w.q=np.zeros((4,2,6));w.qd=w.q.copy()
    w.limits=np.broadcast_to([-1.,1.],(4,2,6,2)).copy()
    w.errors=np.zeros((4,2,5));w.clearances=np.full((4,2),.01)
    w.wp=np.tile(w.layout['wire_center'],(4,1))+w.origins
    w.wq=np.tile([1.,0,0,0],(4,1));w.wv=np.zeros((4,6));w.bottom=np.full(4,.3)
    w.roots=w.root_targets.copy();w.roots[:,:,2]-=.0007
    w.palms=np.arange(24).reshape(4,2,3)*.00001
    w.pad_relative=np.zeros((4,4,3));w.lowest=np.zeros(4,dtype=int)
    w.shapes=[(0,0,None)];w.hands[0].body_names=['base_link']
    w.backends=[SimpleNamespace(measure=lambda: {'diagnostics':{}}) for _ in range(4)]
    return w


@pytest.mark.parametrize('mounted',[False,True])
def test_host_exposes_applied_offset_independently_of_carriage_lag(mounted):
    w=host_world(mounted);m=w.measure_batch()
    np.testing.assert_array_equal(m['commanded_palm_offset_m'],w.palms)
    if mounted:
        np.testing.assert_allclose(m['adaptive_palm_offset'][:,:,2],w.palms[:,:,2]-.0007)
    m['commanded_palm_offset_m'][:]=0
    assert w.palms.any(), 'Host measurement must not alias live command buffers'


def test_startup_adapter_checks_applied_offset_and_selected_reset():
    legacy=host_world();pose=torch.zeros((4,7),dtype=torch.float64)
    wire=SimpleNamespace(data=SimpleNamespace(root_pose_w=pose.clone()))
    wire.write_root_pose_to_sim=lambda value: setattr(wire.data,'root_pose_w',value.clone())
    world=SimpleNamespace(n=4,device='cpu',wire=wire,wp=pose[:,:3].clone())
    world.measure=lambda: {'commanded_palm_offset_m':torch.from_numpy(legacy.palms.copy())}
    world.refresh=lambda **kwargs: setattr(world,'wp',wire.data.root_pose_w[:,:3].clone())
    def reset(ids):
        wire.data.root_pose_w[ids]=pose[ids]
        world.refresh()
    world.reset=reset
    result=validate_world_adapter(legacy,world,None)
    assert result['maximum_reference_errors']['commanded_palm_offset_m']==0.
    assert result['selected_reset_isolation']
    world.measure=lambda: {'commanded_palm_offset_m':torch.from_numpy(legacy.palms.copy())+.001}
    with pytest.raises(ValueError,match='commanded_palm_offset_m'):
        validate_world_adapter(legacy,world,None)
