"""Offline measured-pixel lineage and stereo-tail audit, never a control input."""
import math
import torch
from .tabletop_stereo_geometry import backproject, transform_points
from .tabletop_stereo_wire_matching import wire_colour_mask
from .tabletop_stereo_wire_tracking import _paths_from_mask, _ribbon_pixels


def rejection_at_pixel(reasons, env, y, x):
    return {name:None if value is None else bool(value[env,y,x])
        for name,value in reasons.items()}


def completeness_counts(reference_wire, reference_valid, correct_count, sample_count):
    denominator=int((reference_wire & reference_valid).sum())
    if not 0<=correct_count<=sample_count or denominator<=0:
        raise ValueError('invalid counts or missing native reference denominator')
    return dict(valid_sample_count=sample_count,correct_3d_points_5mm=correct_count,
        reference_denominator=denominator,correct_3d_completeness=correct_count/denominator)


def validate_track_components(ids, components, field):
    if ids is None:
        return 'not_recorded'
    linked={}
    for track, component in zip(ids.tolist(),components):
        if track < 0:
            continue
        linked.setdefault(track,set()).add(component)
    if any(len(value)>1 for value in linked.values()):
        raise ValueError(f'{field} aliases multiple visible components')
    return 'within_frame_noncollision_only' if linked else 'all_unknown'


def validate_surface_samples(rgb, depth, points, K, T_world_left, *, radius=3):
    """Recompute from raw pixels with shared topology helpers; not a second topology oracle."""
    foreground=wire_colour_mask(rgb)
    world=transform_points(backproject(depth,K,torch.isfinite(depth)&(depth>0)),T_world_left)
    rows=[]
    for env in range(len(rgb)):
        valid=points['point_valid'][env]
        pixel=points['point_pixels'][env,valid]
        provenance=points['point_provenance'][env,valid]
        if len(torch.unique(pixel,dim=0))!=len(pixel):
            raise ValueError('duplicate measured source pixels')
        height,width=depth.shape[-2:]
        if ((pixel<0).any() or (pixel[:,0]>=height).any() or (pixel[:,1]>=width).any()):
            raise ValueError('source pixel outside captured image')
        paths,_,_=_paths_from_mask(foreground[env].cpu().numpy())
        observed=(foreground[env]&torch.isfinite(world[env]).all(-1)).cpu().numpy()
        surface=_ribbon_pixels(paths,foreground[env].cpu().numpy(),observed,radius)
        membership={}
        for component,_,y,x in surface:
            membership.setdefault((y,x),set()).add(component)
        center={tuple(p):component for component,path in enumerate(paths) for p in path}
        sources=[]
        for slot,((y,x),kind) in enumerate(zip(pixel.tolist(),provenance.tolist())):
            if not bool(foreground[env,y,x]&torch.isfinite(world[env,y,x]).all()):
                raise ValueError('pixel is not an actual visible RGB candidate with measured depth')
            if kind==2 and ((y,x) not in membership or len(membership[y,x])!=1):
                raise ValueError('surface pixel is outside one reliable visible component/ribbon')
            if kind==1 and (y,x) not in center:
                raise ValueError('centerline sample not on observed path')
            if kind not in (1,2):
                raise ValueError('unknown point provenance cannot pass measured sampling audit')
            sources.append(dict(y=y,x=x,source_code=kind,
                component_id=next(iter(membership[y,x])) if kind==2 else center[y,x]))
        components=[s['component_id'] for s in sources]
        component_status='not_recorded'
        if 'component_id' in points:
            if points['component_id'][env,valid].tolist()!=components:
                raise ValueError('stored component IDs disagree with measured visible paths')
            component_status='agrees_with_recomputed_visible_paths'
        tracks={field+'_status':validate_track_components(
            None if field not in points else points[field][env,valid],components,field)
            for field in ('track_id','image_track_id')}
        xyz=points['xyz'][env,valid]
        measured=world[env,pixel[:,0],pixel[:,1]]
        if not torch.allclose(xyz,measured,atol=1e-7,rtol=1e-6):
            raise ValueError('stored XYZ differs from independently backprojected measured pixel')
        rows.append(dict(env_index=env,unique_pixels=len(pixel),all_source_pixels_measured=True,
            all_same_visible_component=True,maximum_visible_ribbon_steps=radius,
            component_check_scope='each_pixel_reachable_within_its_own_visible_component',
            topology_method='recomputed_raw_mask_using_shared_topology_helpers',
            stored_component_id_status=component_status,**tracks,
            temporal_or_material_track_qualification=False,
            material_node_identity=False,hidden_interpolation=False,samples=sources))
    return rows


def tail_geometry(predicted_z, reference_z, K, *, y, x, disparity, baseline):
    K=torch.as_tensor(K,dtype=torch.float64)
    ray=torch.linalg.solve(K,torch.tensor([x+.5,y+.5,1.],dtype=torch.float64))
    absolute=abs(predicted_z-reference_z)
    spatial=absolute*float(ray.norm())
    actual_disparity=float(K[0,0])*baseline/reference_z
    return dict(predicted_z_m=predicted_z,reference_z_m=reference_z,
        z_absolute_error_m=absolute,point_error_3d_m=spatial,is_tail_5mm=spatial>.005,
        disparity_px=disparity,reference_disparity_px=actual_disparity,
        disparity_error_px=disparity-actual_disparity,right_match_x_px=x-disparity,
        right_reference_x_px=x-actual_disparity,threshold_m=.005,
        metric='Euclidean_same_pixel_backprojection')


def optical_sample(image,y,x):
    """Same-row two-supported-pixel interpolation, no unavailable depth fill."""
    if not math.isfinite(x) or x<0 or x>image.shape[-1]-1:return None
    lo,hi=math.floor(x),math.ceil(x)
    a,b=float(image[y,lo]),float(image[y,hi])
    if not math.isfinite(a) or not math.isfinite(b) or min(a,b)<=0:return None
    return a+(b-a)*(x-lo)


def window_count(mask,y,x,radius):
    if mask is None or not math.isfinite(x):return None
    x=round(x)
    y0,y1=max(0,y-radius),min(mask.shape[-2],y+radius+1)
    x0,x1=max(0,x-radius),min(mask.shape[-1],x+radius+1)
    return int(mask[y0:y1,x0:x1].sum()) if y0<y1 and x0<x1 else 0
