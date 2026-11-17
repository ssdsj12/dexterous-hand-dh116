"""Analytic CPU contracts for validation-only visible wire geometry bins."""
import importlib
import numpy as np
import pytest
import torch


def _analyze(mask, depth=None, K=None):
    module = importlib.import_module('go2_pvcnn.control.dh116.tabletop_stereo_error_geometry')
    mask = torch.as_tensor(mask, dtype=torch.bool)[None]
    depth = torch.ones(mask.shape, dtype=torch.float64) if depth is None else depth
    K = torch.tensor([[100., 0., 16.], [0., 100., 16.], [0., 0., 1.]]) if K is None else K
    return module.wire_geometry_regions(mask, depth, K)


def test_horizontal_visible_surface_has_horizontal_direction_and_zero_curvature():
    mask = np.zeros((32, 32), bool); mask[16, 3:29] = True
    regions, meta = _analyze(mask)
    assert regions['wire_direction_deg_0_15'].sum() == mask.sum()
    assert regions['wire_surface_curvature_m_inv_0_1'].sum() > 0
    assert not regions['wire_surface_curvature_m_inv_10_plus'].any()
    assert meta['environments'][0]['simple_path_components'] == 1
    assert meta['hidden_centerline_qualified'] is False
    assert meta['units']['curvature'] == 'm^-1'


@pytest.mark.parametrize('kind,key', [('vertical','wire_direction_deg_75_90'), ('diagonal','wire_direction_deg_45_75')])
def test_image_directions_are_undirected_degrees_to_horizontal_epipolar_axis(kind, key):
    mask = np.zeros((32,32),bool)
    if kind == 'vertical': mask[3:29,16] = True
    else: np.fill_diagonal(mask[3:29,3:29],True)
    regions,_ = _analyze(mask)
    assert regions[key].sum() == mask.sum()


def test_disconnected_visible_fragments_are_analyzed_without_invented_bridge():
    mask = np.zeros((32,32),bool);mask[8,2:15]=True;mask[24,17:30]=True
    regions,meta = _analyze(mask)
    assert meta['environments'][0]['connected_components'] == 2
    assert meta['environments'][0]['simple_path_components'] == 2
    assert meta['global_path_qualification'] is None
    for region in regions.values():
        if region is not None: assert not bool((region & ~torch.tensor(mask)[None]).any())


@pytest.mark.parametrize('kind', ['branch','loop'])
def test_ambiguous_topology_is_na_instead_of_fabricated_full_path(kind):
    mask=np.zeros((32,32),bool)
    if kind=='branch': mask[4:28,16]=True;mask[16,4:28]=True
    else:
        mask[8,8:24]=True;mask[23,8:24]=True;mask[8:24,8]=True;mask[8:24,23]=True
    regions,meta=_analyze(mask)
    assert all(v is None for k,v in regions.items() if 'curvature' in k)
    assert meta['environments'][0]['simple_path_components'] == 0
    assert meta['environments'][0]['rejected_components'][0]['reason'] == 'branch_loop_or_non_simple_visible_skeleton'


def test_staircase_direction_does_not_require_a_globally_orderable_skeleton():
    mask=np.zeros((64,80),bool)
    for x in range(5,75):
        y=45-int(.36*x)
        mask[y-3:y+4,x]=True
    # Visible branch makes full-path ordering ambiguous, without invalidating
    # the anisotropic local orientation of the long digital staircase body.
    mask[16:33,40:43]=True
    regions,meta=_analyze(mask)
    assert meta['environments'][0]['simple_path_components']==0
    assert all(v is None for k,v in regions.items() if 'curvature' in k)
    assert regions['wire_direction_deg_15_45'].sum() > .7*mask.sum()
    assert meta['direction_method']=='same_component_visible_mask_local_PCA'
    assert meta['direction_anisotropy_min_eigenvalue_ratio']==2.


def test_isotropic_local_mask_has_no_invented_orientation():
    y,x=np.mgrid[:32,:32];mask=(x-16)**2+(y-16)**2<=16
    regions,meta=_analyze(mask)
    assert all(v is None for k,v in regions.items() if 'direction' in k)
    assert meta['environments'][0]['direction_unavailable_pixels']==mask.sum()


def test_invalid_depth_splits_fragments_and_is_never_binned():
    mask=np.zeros((32,32),bool);mask[16,3:29]=True
    z=torch.ones((1,32,32),dtype=torch.float64);z[0,16,14:18]=float('nan')
    regions,meta=_analyze(mask,z)
    assert meta['environments'][0]['connected_components'] == 2
    assert not regions['wire_direction_deg_0_15'][0,16,14:18].any()
    assert meta['environments'][0]['wire_pixels_missing_optical_depth'] == 4


def test_short_path_has_direction_but_curvature_is_na():
    mask=np.zeros((32,32),bool);mask[16,14:17]=True
    regions,meta=_analyze(mask)
    assert regions['wire_direction_deg_0_15'].sum()==3
    assert regions['wire_surface_curvature_m_inv_0_1'] is None
    assert meta['environments'][0]['curvature_unavailable_pixels']==3


def test_3d_bend_is_detected_without_using_image_turn_as_physical_curvature():
    mask=np.zeros((32,32),bool);mask[16,3:29]=True
    x=(torch.arange(32,dtype=torch.float64)+.5-16)/100
    z=torch.ones((1,32,32),dtype=torch.float64)
    z[0,16]=1+2*x*x
    regions,meta=_analyze(mask,z)
    assert regions['wire_direction_deg_0_15'].sum()==mask.sum()
    assert regions['wire_surface_curvature_m_inv_1_10'].sum()>10
    assert meta['sampling_scope']=='same_component_visible_surface_skeleton_local_geometry'


def test_missing_mask_is_explicit_na():
    module=importlib.import_module('go2_pvcnn.control.dh116.tabletop_stereo_error_geometry')
    regions,meta=module.wire_geometry_regions(None,torch.ones((1,8,8)),torch.eye(3))
    assert all(v is None for v in regions.values())
    assert meta['blocked_reason']=='left_wire_instance_reference_unavailable'


def test_geometry_rejects_wrong_shape_or_non_boolean_masks():
    module=importlib.import_module('go2_pvcnn.control.dh116.tabletop_stereo_error_geometry')
    with pytest.raises(ValueError,match='bool'):
        module.wire_geometry_regions(torch.ones((1,8,8)),torch.ones((1,8,8)),torch.eye(3))
