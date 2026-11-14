"""Validation-only geometry bins from the observed left wire surface.

These local skeleton measurements never establish hidden centerline shape,
cross-frame tracking, or full-object visibility. Disconnected components stay
separate; ambiguous topology is unavailable rather than force-ordered.
"""
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

from .tabletop_stereo_geometry import backproject
from .tabletop_stereo_perception import _thin, _visible_path


_DIRECTION = ('wire_direction_deg_0_15', 'wire_direction_deg_15_45',
              'wire_direction_deg_45_75', 'wire_direction_deg_75_90')
_CURVATURE = ('wire_surface_curvature_m_inv_0_1',
              'wire_surface_curvature_m_inv_1_10',
              'wire_surface_curvature_m_inv_10_plus')


def _summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return dict(count=int(len(values)), p50=float(np.median(values)) if len(values) else None,
                p05=float(np.quantile(values, .05)) if len(values) else None,
                p95=float(np.quantile(values, .95)) if len(values) else None,
                max=float(values.max()) if len(values) else None)


def _local_direction(component):
    """Observed-mask covariance in a 25px window, independently of topology."""
    y, x = np.indices(component.shape, dtype=float)
    weight = component.astype(float)
    smooth = lambda a: ndimage.uniform_filter(a, size=25, mode='constant', cval=0.)
    count_fraction = smooth(weight)
    denom = np.maximum(count_fraction, 1e-15)
    mean_x, mean_y = smooth(weight*x)/denom, smooth(weight*y)/denom
    xx = np.maximum(0., smooth(weight*x*x)/denom-mean_x*mean_x)
    yy = np.maximum(0., smooth(weight*y*y)/denom-mean_y*mean_y)
    xy = smooth(weight*x*y)/denom-mean_x*mean_y
    eigen_delta = np.sqrt((xx-yy)**2+4*xy*xy)
    major, minor = (xx+yy+eigen_delta)/2, np.maximum(0., (xx+yy-eigen_delta)/2)
    good = component & (count_fraction*625 >= 3-1e-8) & (major > 1e-10) & (major >= 2*minor)
    angle = np.abs(np.degrees(.5*np.arctan2(2*xy, xx-yy)))
    return np.searchsorted([15., 45., 75.], angle, side='right'), good


@torch.no_grad()
def wire_geometry_regions(wire_mask, reference_depth, K):
    """Return BHW diagnostic masks and explicit surface/topology metadata.

    Direction comes from anisotropic local visible-mask PCA, without ordering
    a global path. Curvature is assigned from the nearest visible skeleton
    pixel in the SAME simple component. These are 2D region attributions,
    never 3D point correspondence or tracking. Curvature uses a fixed +/-4
    skeleton-sample support (3D circumcircle); end supports remain unavailable.
    """
    metadata = dict(
        sampling_scope='same_component_visible_surface_skeleton_local_geometry',
        hidden_centerline_qualified=False, global_path_qualification=None,
        geometry_source='left_instance_reference_and_same_frame_optical_Z',
        units=dict(direction='degrees_to_horizontal_epipolar_axis', curvature='m^-1', width='px'),
        direction_edges_deg=[0., 15., 45., 75., 90.],
        direction_method='same_component_visible_mask_local_PCA',
        direction_window_px=25, direction_anisotropy_min_eigenvalue_ratio=2.,
        direction_min_local_observed_pixels=3,
        curvature_edges_m_inv=[0., 1., 10., None], support_half_span_samples=4,
        association_scope='same_component_2D_nearest_skeleton_region_attribution;not_3D_tracking',
        curvature_scope='local_visible_surface_skeleton_proxy;not_physical_hidden_centerline_curvature',
        curvature_formula='2*norm(cross(p0-p1,p2-p1))/(norm(p0-p1)*norm(p2-p1)*norm(p2-p0))',
        width_estimator='2*Euclidean_distance_to_background_at_skeleton-1;discrete_pixel_width_proxy',
        width_sampling='all_visible_skeleton_pixels;topology_not_required_for_width_proxy', environments=[])
    empty = {name: None for name in (*_DIRECTION, *_CURVATURE)}
    if wire_mask is None:
        metadata['blocked_reason'] = 'left_wire_instance_reference_unavailable'
        return empty, metadata
    if (not isinstance(wire_mask, torch.Tensor) or wire_mask.ndim != 3 or wire_mask.dtype != torch.bool
            or not isinstance(reference_depth, torch.Tensor) or reference_depth.shape != wire_mask.shape
            or not reference_depth.is_floating_point()):
        raise ValueError('bool wire mask and float optical depth with matching [B,H,W] shapes required')
    valid = wire_mask & torch.isfinite(reference_depth) & (reference_depth > 0)
    xyz = backproject(reference_depth, K, valid).cpu().numpy()
    mask_np, valid_np = wire_mask.cpu().numpy(), valid.cpu().numpy()
    regions_np = {name: np.zeros(wire_mask.shape, dtype=bool) for name in empty}
    any_direction = any_curvature = False
    all_widths = []
    for env, observed in enumerate(valid_np):
        labels, count = ndimage.label(observed, structure=np.ones((3, 3), dtype=bool))
        detail = dict(env_id=env, connected_components=count, simple_path_components=0,
                      rejected_components=[], wire_pixels_missing_optical_depth=int((mask_np[env] & ~observed).sum()),
                      direction_available_pixels=0, curvature_available_pixels=0,
                      direction_unavailable_pixels=0, curvature_unavailable_pixels=0)
        widths = []
        for component_id in range(1, count + 1):
            component = labels == component_id
            direction, direction_valid = _local_direction(component)
            if direction_valid.any():
                any_direction = True
                for i, name in enumerate(_DIRECTION):
                    regions_np[name][env] |= direction_valid & (direction == i)
                detail['direction_available_pixels'] += int(direction_valid.sum())
            skeleton = _thin(component)
            widths.extend((2*ndimage.distance_transform_edt(component)[skeleton]-1).tolist())
            path = _visible_path(skeleton)
            if path is None:
                detail['rejected_components'].append(dict(component_id=component_id,
                    wire_pixels=int(component.sum()), skeleton_pixels=int(skeleton.sum()),
                    reason='branch_loop_or_non_simple_visible_skeleton'))
                continue
            if len(path) < 3:
                detail['rejected_components'].append(dict(component_id=component_id,
                    wire_pixels=int(component.sum()), skeleton_pixels=int(len(path)),
                    reason='visible_skeleton_too_short_for_local_direction'))
                continue
            detail['simple_path_components'] += 1
            samples = np.arange(len(path))
            curve = np.full(len(path), np.nan)
            inner = samples[(samples >= 4) & (samples < len(path)-4)]
            if len(inner):
                points = xyz[env][tuple(path.T)]
                a, b = points[inner-4]-points[inner], points[inner+4]-points[inner]
                denom = np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)*np.linalg.norm(b-a, axis=-1)
                usable = np.isfinite(denom) & (denom > 1e-12)
                curve[inner[usable]] = 2*np.linalg.norm(np.cross(a[usable], b[usable]), axis=-1)/denom[usable]
            pixels = np.argwhere(component)
            _, nearest = cKDTree(path).query(pixels, k=1)
            available = np.isfinite(curve[nearest])
            if available.any():
                any_curvature = True
                curvature_bin = np.searchsorted([1., 10.], curve[nearest][available], side='right')
                for i, name in enumerate(_CURVATURE):
                    target = pixels[available][curvature_bin == i]
                    regions_np[name][env, target[:, 0], target[:, 1]] = True
                detail['curvature_available_pixels'] += int(available.sum())
        detail['direction_unavailable_pixels'] = int(observed.sum())-detail['direction_available_pixels']
        detail['curvature_unavailable_pixels'] = int(observed.sum())-detail['curvature_available_pixels']
        detail['width_px_distribution'] = _summary(widths)
        all_widths.extend(widths)
        metadata['environments'].append(detail)
    metadata['width_px_distribution'] = _summary(all_widths)
    metadata['blocked_reason'] = None if any_direction else 'no_anisotropic_observed_mask_local_orientation'
    regions = {name: torch.as_tensor(value, device=wire_mask.device) for name, value in regions_np.items()}
    if not any_direction:
        regions.update({name: None for name in _DIRECTION})
    if not any_curvature:
        regions.update({name: None for name in _CURVATURE})
        metadata['curvature_blocked_reason'] = 'insufficient_unambiguous_local_3D_surface_support'
    return regions, metadata
