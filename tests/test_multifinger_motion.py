import importlib
import torch

def test_paused_ascent_cancels_each_hands_queued_z_at_measured_height():
    module=importlib.import_module('go2_pvcnn.control.dh116.tabletop_multifinger_motion')
    queued=torch.tensor([[[.001,.002,.020],[.003,.004,.022]],[[0.,0.,.03],[0.,0.,.04]]],dtype=torch.float64)
    measured=queued.clone();measured[0,:,2]=torch.tensor([.013,.014],dtype=torch.float64)
    old=queued.clone()
    result=module.cancel_queued_ascent(queued,measured,torch.tensor([True,False]))
    torch.testing.assert_close(result[0,:,2],measured[0,:,2])
    torch.testing.assert_close(result[1],old[1]);torch.testing.assert_close(result[:,:,:2],old[:,:,:2])
    assert torch.equal(queued,old)

def test_pausing_does_not_raise_an_existing_lower_target():
    module=importlib.import_module('go2_pvcnn.control.dh116.tabletop_multifinger_motion')
    queued=torch.zeros(1,2,3,dtype=torch.float64);measured=queued+.001
    assert torch.equal(module.cancel_queued_ascent(queued,measured,torch.tensor([True])),queued)
