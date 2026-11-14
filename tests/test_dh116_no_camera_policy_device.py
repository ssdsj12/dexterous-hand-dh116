"""Device metadata negatives; no CUDA allocation, initialization or copy."""
from types import SimpleNamespace
import pytest
import torch

from test_dh116_no_camera_collector import fixture, module


@pytest.mark.parametrize("field", ["parameters", "buffers"])
def test_cuda_metadata_rejected_before_weight_digest_or_physics(monkeypatch,tmp_path,field):
    c,ep,tr,r0,sim,rod,hands,obs,rewards=fixture(monkeypatch,tmp_path)
    sentinel=SimpleNamespace(device=torch.device("cuda:0"))
    monkeypatch.setattr(tr.model,field,lambda:iter((sentinel,)))
    calls=[]
    def forbidden(trainer):
        calls.append("weights")
        pytest.fail("weight digest would call .cpu() before checking device")
    monkeypatch.setattr(module(),"_model_weights",forbidden)
    before=sim.current_time_step_index,rod.rope.steps
    with pytest.raises(ValueError,match="CPU"):
        c.collect()
    assert calls==[]
    assert (sim.current_time_step_index,rod.rope.steps)==before
    assert tr._pending is None and len(tr.rollout)==0
    c.close()
