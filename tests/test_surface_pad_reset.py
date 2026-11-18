import torch
from test_effective_pad import world_fixture,load_module,fixture_atlas


def test_pose_only_reset_readback_ignores_stale_contact_buffer_but_checks_pose(monkeypatch):
    pad=load_module('tabletop_effective_pad');runtime=load_module('tabletop_effective_pad_runtime')
    monkeypatch.setattr(runtime,'build_pad_atlas',lambda world:fixture_atlas(pad))
    w=world_fixture();sampler=runtime.PadContactSampler(w)
    # Old contact contents may remain corrupt until a fresh physics step.
    for sensor in w.w.sensors:
        sensor.contact_physx_view.get_contact_data=lambda **kw: (_ for _ in ()).throw(AssertionError('stale manifold read'))
    result=sampler.sample(contacts=False)
    assert not result['effective_pad_bad_env'].any()
    assert not result['effective_pad_loads_n'].any()
    w.hands[0].data.body_link_pose_w[1,0,3:]=0.
    result=sampler.sample(contacts=False)
    assert result['effective_pad_bad_env'].tolist()==[False,True,False]
