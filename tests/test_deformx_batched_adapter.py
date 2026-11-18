import numpy as np
from go2_pvcnn.control.dh116.deformx_batched_adapter import MaterialSampling, mesh_wrenches


def test_fixed_regions_ignore_distant_sag_and_detect_opposite_slip():
    p = np.zeros((2,3,21)); p[:,1] = np.linspace(-.5,.5,21); p[:,2] = .308
    regions = MaterialSampling(p[0], np.array([[0,-.2,.308],[0,.2,.308]]))
    v = np.zeros_like(p); r = np.full((2,20),.005)
    initial = regions.measure(p,v,r)
    np.testing.assert_allclose(initial['bottom'],.303)
    p[1,2,:3] -= .1
    moved = regions.measure(p,v,r)
    np.testing.assert_allclose(moved['bottom'],initial['bottom'])
    assert moved['global_bottom'][1] < .21
    p[1,0,5:8] += .01; p[1,0,13:16] -= .01
    moved = regions.measure(p,v,r)
    assert moved['centers'][1,0,0] > .009
    assert moved['centers'][1,1,0] < -.009


def test_com_reaction_translation_invariance_and_normal_channels():
    import torch
    impulse = torch.tensor([[[1.,2.,3.],[2.,0.,0.]]])
    com = torch.tensor([[[.2,.3,.4],[.2,.3,.4]]])
    angular = torch.cross(com,impulse,dim=-1) + torch.tensor([[[0.,0.,.5],[0.,0.,.5]]])
    force, torque = mesh_wrenches(impulse,angular,com,.1)
    torch.testing.assert_close(force,impulse*10)
    torch.testing.assert_close(torque,torch.tensor([[[0.,0.,5.],[0.,0.,5.]]]))
    shift = torch.tensor([[[10.,20.,30.]]])
    _, translated = mesh_wrenches(impulse,angular+torch.cross(shift.expand_as(impulse),impulse,dim=-1),com+shift,.1)
    torch.testing.assert_close(translated,torque,atol=1e-4,rtol=0)


def test_region_means_integrate_material_length_across_offcenter_nodes():
    p=np.zeros((1,3,21));p[:,1]=np.linspace(-.5,.5,21);p[:,2]=.308
    sampling=MaterialSampling(p[0],np.array([[0,-.185,.308],[0,.185,.308]]))
    result=sampling.measure(p,p.copy(),np.full((1,20),.005))
    np.testing.assert_allclose(result['centers'][0,:,1],[-.185,.185],atol=1e-12)
    np.testing.assert_allclose(result['velocities'],result['centers'])
