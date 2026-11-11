"""Evaluation-only error attribution for immutable stereo captures.

These routines consume validation reference explicitly. They are never used by
the Actor, matcher, controller or promotion gates. Pixel coordinates are integer
array indices; K uses the existing renderer pixel-centre convention (u+.5,v+.5).
"""
import math
import torch
import torch.nn.functional as F

from .tabletop_stereo_geometry import backproject


def _aligned(*values):
    if not values or values[0].ndim!=3 or any(v.shape!=values[0].shape for v in values):
        raise ValueError('aligned [B,H,W] diagnostic tensors required')


def distribution(values):
    v=values.detach().double().flatten();v=v[torch.isfinite(v)]
    if not len(v):return dict(count=0,mean=None,p50=None,p90=None,p95=None,max=None)
    return dict(count=len(v),mean=float(v.mean()),
        **{f'p{q}':float(v.quantile(q/100)) for q in (50,90,95)},max=float(v.max()))


def region_metrics(prediction,reference,predicted_valid,reference_valid,region,K,*,disparity=None):
    """Keep the reference denominator, prediction count and joint errors separate."""
    _aligned(prediction,reference,predicted_valid,reference_valid,region)
    if any(m.dtype!=torch.bool for m in (predicted_valid,reference_valid,region)):
        raise ValueError('explicit boolean masks required')
    eligible=region&reference_valid&torch.isfinite(reference)&(reference>0)
    accepted=region&predicted_valid&torch.isfinite(prediction)&(prediction>0)
    joint=eligible&accepted
    denominator=int(eligible.sum());count=int(joint.sum())
    signed=(prediction-reference)[joint].double();absolute=signed.abs()
    # At the same pixel, Euclidean point error is |delta Z| * optical ray norm.
    # Gather only the joint support rather than allocate two full VGA clouds
    # repeatedly for every small diagnostic region.
    indices=joint.nonzero();k=torch.as_tensor(K,device=prediction.device,dtype=torch.float64)
    if k.shape==(3,3):k=k[None].expand(prediction.shape[0],-1,-1)
    if k.shape!=(prediction.shape[0],3,3) or not torch.isfinite(k).all():raise ValueError('finite batched K required')
    pixel=torch.stack((indices[:,2].double()+.5,indices[:,1].double()+.5,torch.ones(count,device=prediction.device)),1)
    rays=torch.bmm(torch.linalg.inv(k)[indices[:,0]],pixel[:,:,None]).squeeze(-1)
    spatial=absolute*rays.norm(dim=-1)
    d=distribution(absolute)
    result=dict(reference_valid_count=denominator,predicted_valid_count=int(accepted.sum()),
        intersection_count=count,prediction_coverage=count/denominator if denominator else None,
        mae_m=d['mean'],rmse_m=float(signed.square().mean().sqrt()) if count else None,
        signed_bias_m=float(signed.mean()) if count else None,
        **{key+'_m':d[key] for key in ('p50','p90','p95','max')},
        correct_3d_points_5mm=int((spatial<=.005).sum()),
        correct_3d_point_completeness_5mm=float((spatial<=.005).sum())/denominator if denominator else None,
        correct_z_completeness_5mm=float((absolute<=.005).sum())/denominator if denominator else None,
        point_error_3d_m=distribution(spatial))
    if disparity is not None:
        _aligned(prediction,disparity)
        values=disparity[joint].double();values=values[torch.isfinite(values)]
        edges=[0.,8.,16.,32.,64.,96.,128.,192.,256.,512.,1024.]
        result['disparity_histogram']=dict(edges_px=edges,
            counts=torch.histogram(values.cpu(),bins=torch.tensor(edges,dtype=torch.float64)).hist.long().tolist(),
            outside_histogram_count=int(((values<edges[0])|(values>edges[-1])).sum()))
    return result


def wire_bands(mask):
    if mask.ndim!=3 or mask.dtype!=torch.bool:raise ValueError('wire mask must be bool [B,H,W]')
    interior=F.avg_pool2d(mask.float()[:,None],3,1,1)[:,0]>=1.
    # Interior is a one-pixel erosion, not a recovered hidden centreline.
    return dict(wire=mask,wire_edge=mask&~interior,wire_interior=interior)


def reference_visibility(left_z,right_z,left_valid,K,baseline_m,*,tolerance_m=.001):
    """Validation-only reprojection against actual right optical-Z.

    Missing right reference remains NA. Disagreement with a farther right
    surface is reported separately from right-side occlusion; neither becomes
    a claim about a hidden full wire. Threshold is a diagnostic parameter only.
    """
    if not math.isfinite(tolerance_m) or tolerance_m<0 or baseline_m<=0:
        raise ValueError('invalid visibility tolerance/baseline')
    _aligned(left_z,left_valid)
    if right_z is None:
        return {k:None for k in ('common_visible','left_only_occluded','outside_right_fov','reference_disagreement','unknown_right')}
    _aligned(left_z,right_z)
    from .tabletop_stereo_depth import StereoDepth
    k=torch.as_tensor(K,device=left_z.device,dtype=left_z.dtype)
    fx=k[0,0] if k.ndim==2 else k[:,0,0,None,None]
    good=left_valid&torch.isfinite(left_z)&(left_z>0)
    x=torch.arange(left_z.shape[-1],device=left_z.device,dtype=left_z.dtype)[None,None,:].expand_as(left_z)
    xr=x-fx*baseline_m/torch.where(good,left_z,1.)
    right_valid=torch.isfinite(right_z)&(right_z>0)
    rz,available=StereoDepth.sample_horizontal(torch.where(right_valid,right_z,0.),xr,right_valid)
    inside=torch.isfinite(xr)&(xr>=0)&(xr<=left_z.shape[-1]-1)
    delta=rz-left_z
    return dict(common_visible=good&available&(delta.abs()<=tolerance_m),
        left_only_occluded=good&available&(delta < -tolerance_m),
        outside_right_fov=good&~inside,
        reference_disagreement=good&available&(delta>tolerance_m),
        unknown_right=good&inside&~available)


def rejection_summary(reasons,region):
    """Overlapping masks are not an additive exclusive causal decomposition."""
    if region.ndim!=3 or region.dtype!=torch.bool:raise ValueError('boolean [B,H,W] ROI required')
    remaining=region.clone();overlap={};first={}
    for name,mask in reasons.items():
        if mask is None:
            overlap[name]=None;first[name]=None;continue
        _aligned(region,mask)
        if mask.dtype!=torch.bool:raise ValueError('boolean reason masks required')
        overlap[name]=int((region&mask).sum());first[name]=int((remaining&mask).sum())
        remaining&=~mask
    return dict(denominator=int(region.sum()),overlapping_counts=overlap,
        first_failure_counts=first,first_failure_order=list(reasons),not_rejected_count=int(remaining.sum()),
        scope='diagnostic_condition_counts;correlation_not_unique_cause')


@torch.no_grad()
def sampled_cost_curves(left_rgb,right_rgb,pixels,disparities,block_size,*,wire_mask=None):
    """Bounded selected-pixel SAD/Census/gradient probes, not a new matcher.

    Full curves are saved only for explicit [env,row,column] samples. Census
    uses each patch centre as its threshold. Outside-window support is null.
    """
    if (left_rgb.shape!=right_rgb.shape or left_rgb.ndim!=4 or left_rgb.shape[1]!=3
            or pixels.ndim!=2 or pixels.shape[1]!=3 or block_size<3 or block_size%2!=1):
        raise ValueError('RGB pair, [P,3] pixels and odd block size required')
    a=left_rgb.mean(1);b=right_rgb.mean(1);n,h,w=a.shape;r=block_size//2
    output=[]
    for env,y,x in pixels.long().tolist():
        if not (0<=env<n and r<=y<h-r and r<=x<w-r):continue
        pa=a[env,y-r:y+r+1,x-r:x+r+1]
        sad=[];census=[];gradient=[]
        for d in disparities:
            d=int(d);xx=x-d
            if not r<=xx<w-r:
                sad.append(None);census.append(None);gradient.append(None);continue
            pb=b[env,y-r:y+r+1,xx-r:xx+r+1]
            sad.append(float((pa-pb).abs().mean()))
            census.append(int(((pa<pa[r,r])!=(pb<pb[r,r])).sum()))
            gradient.append(float(((pa[:,1:]-pa[:,:-1])-(pb[:,1:]-pb[:,:-1])).abs().mean()))
        finite=[(value,i) for i,value in enumerate(sad) if value is not None]
        finite.sort();non_adjacent=[]
        best=finite[0][0] if finite else None
        for value,i in finite:
            if value>best+max(1e-8,.05*abs(best)):break
            if all(abs(i-j)>1 for j in non_adjacent):non_adjacent.append(i)
        best_i=finite[0][1] if finite else None
        curvature=None;offset=None
        if best_i is not None and 0<best_i<len(sad)-1 and sad[best_i-1] is not None and sad[best_i+1] is not None:
            curvature=sad[best_i-1]+sad[best_i+1]-2*sad[best_i]
            if curvature>1e-8:offset=.5*(sad[best_i-1]-sad[best_i+1])/curvature
        output.append(dict(env_index=env,row=y,column=x,disparities_px=list(disparities),
            sad_cost=sad,census_hamming=census,horizontal_gradient_cost=gradient,
            non_adjacent_minimum_count=len(non_adjacent),best_disparity_integer_px=disparities[best_i] if best_i is not None else None,
            fit_curvature=curvature,unclamped_subpixel_offset_px=offset,
            left_patch_gray_std=float(pa.std(unbiased=False)),
            left_patch_wire_fraction=float(wire_mask[env,y-r:y+r+1,x-r:x+r+1].float().mean()) if wire_mask is not None else None,
            scope='selected_pixel_offline_cost_probe;not_production_matcher_output'))
    return output


@torch.no_grad()
def replay_depth_conditions(left_rgb,right_rgb,calibration,config):
    """Replay existing private matcher functions; never change their decisions.

    Saved invalid disparity/confidence is NaN/zero, so its rejection cannot be
    recovered from the saved output alone. This is explicitly an offline replay
    on this device, checked against the saved native validity by the caller.
    """
    from .tabletop_stereo_depth import StereoDepth
    if not calibration.rectified:raise ValueError('rectified raw diagnostic inputs required')
    matcher=StereoDepth(**config)
    result=matcher.compute(left_rgb,right_rgb,calibration)
    diagnostic=result['diagnostics'];r=diagnostic['reasons']
    # Keep existing J1 report keys while consuming the actual runtime matcher
    # predicates, including opt-in candidate refinement, in exactly one path.
    reasons=dict(matcher_support_or_subpixel_rejection=r['left_matcher_rejected'],
        low_texture_left=r['low_texture_left'],right_support_or_texture_rejection=r['right_sample_unavailable'],
        left_right_inconsistent=r['left_right_inconsistent'],uniqueness_rejection=r['uniqueness'],
        out_of_depth_range=r['depth_range'],occlusion_composite=r['occlusion_composite'],
        patch_border_rejection=r['patch_border'],depth_edge_rejection=None,
        low_texture_right=r['low_texture_right'],search_boundary=r['search_boundary'],
        subpixel_rejected=r['subpixel_rejected'],true_occlusion=r['true_occlusion'])
    return dict(valid=result['valid'],reasons=reasons,raw_disparity_px=diagnostic['raw_disparity_px'],
        raw_confidence=diagnostic['raw_confidence'],matcher_diagnostics=diagnostic,
        scope='offline_replay_existing_baseline;not_saved_native_rejection_trace',
        uniqueness_scope='OpenCV_internal_rejection_not_separately_exposed' if matcher.method=='sgbm' else diagnostic['uniqueness_scope'])
