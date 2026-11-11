"""RGB-only rectified stereo baseline; no rendered depth or privileged fallback."""
import math
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from .tabletop_stereo_validation import validate_images
from .tabletop_stereo_wire_matching import METHODS as WIRE_METHODS, match_wire, wire_colour_mask


class StereoDepth:
    def __init__(self, *, method='blockmatch', min_disparity=1, max_disparity=64,
                 block_size=5,min_depth_m=.2,max_depth_m=1.2,lr_tolerance_px=1.,
                 texture_threshold=.001,uniqueness_threshold=.05,rectify=False,
                 subpixel=False,image_processing_batch_size=1,reliability=None,photometric_verification=None,
                 pyramid_mode=None):
        if (method not in ('blockmatch','sgbm','reliable_sad','census','zncc','gradient','coarse_to_fine') + WIRE_METHODS or type(min_disparity) is not int or type(max_disparity) is not int
                or min_disparity<1 or max_disparity<=min_disparity or type(block_size) is not int
                or block_size<3 or block_size%2!=1 or not 0<min_depth_m<max_depth_m
                or not all(math.isfinite(x) for x in (min_depth_m,max_depth_m,lr_tolerance_px,texture_threshold,uniqueness_threshold))
                or lr_tolerance_px<0 or texture_threshold<0 or not 0<=uniqueness_threshold<1
                or type(subpixel) is not bool or (image_processing_batch_size is not None
                    and (type(image_processing_batch_size) is not int or image_processing_batch_size<1))):
            raise ValueError('invalid stereo depth configuration')
        if method == 'coarse_to_fine':
            pyramid_mode = 'audit' if pyramid_mode is None else pyramid_mode
            if pyramid_mode not in ('audit', 'accelerated'):
                raise ValueError('pyramid_mode must be audit or accelerated')
        elif pyramid_mode is not None:
            raise ValueError('pyramid_mode requires coarse_to_fine method')
        if method == 'wire_silhouette' and subpixel:
            raise ValueError('wire_silhouette requires integer observed-mask correspondence')
        self.pyramid_mode = pyramid_mode
        self.method=method;self.min_disparity=min_disparity;self.max_disparity=max_disparity
        self.block_size=block_size;self.near=min_depth_m;self.far=max_depth_m
        self.lr=lr_tolerance_px;self.texture=texture_threshold;self.uniqueness=uniqueness_threshold;self.rectify=rectify
        self.subpixel=subpixel;self.image_processing_batch_size=image_processing_batch_size
        if method=='reliable_sad':
            from .tabletop_stereo_matcher_candidates import resolve_policy
            self.reliability=resolve_policy(reliability)
            if photometric_verification is not None and type(photometric_verification) is not bool:
                raise ValueError('photometric verification must be boolean')
            self.photometric_verification=True if photometric_verification is None else photometric_verification
        elif reliability is not None:
            raise ValueError('reliability policy requires opt-in reliable_sad method')
        else:
            if photometric_verification is not None:raise ValueError('photometric verification requires opt-in reliable_sad')
            self.reliability=None;self.photometric_verification=False

    @classmethod
    def from_config(cls,value):
        value=json.loads(Path(value).read_text(encoding='utf-8')) if isinstance(value,(str,Path)) else value
        if not isinstance(value,dict):raise ValueError('stereo depth configuration dictionary required')
        if 'depth_estimator' in value:return cls(**value['depth_estimator'])
        if 'depth' in value:return cls(**value['depth'])
        if 'K_left' in value:
            from .tabletop_stereo_calibration import StereoCalibration
            from .tabletop_stereo_resolution import disparity_budget
            c=StereoCalibration.from_dict(value)
            near,far=value.get('depth_range_m',[.2,1.2]);budget=disparity_budget(c,near,far)
            low,high=value.get('disparity_range_px',[budget['suggested_min_disparity'],budget['suggested_max_disparity']])
            return cls(min_disparity=low,max_disparity=high,min_depth_m=near,max_depth_m=far)
        return cls(**value)

    def to_config(self):
        result=dict(method=self.method,min_disparity=self.min_disparity,max_disparity=self.max_disparity,
            block_size=self.block_size,min_depth_m=self.near,max_depth_m=self.far,lr_tolerance_px=self.lr,
            texture_threshold=self.texture,uniqueness_threshold=self.uniqueness,rectify=self.rectify,
            subpixel=self.subpixel,image_processing_batch_size=self.image_processing_batch_size)
        if self.reliability is not None:
            result['reliability']=dict(self.reliability)
            result['photometric_verification']=self.photometric_verification
        if self.pyramid_mode is not None:
            result['pyramid_mode'] = self.pyramid_mode
        return result

    def _texture(self,gray):
        mean=F.avg_pool2d(gray[:,None],self.block_size,1,self.block_size//2)[:,0]
        second=F.avg_pool2d(gray.square()[:,None],self.block_size,1,self.block_size//2)[:,0]
        return (second-mean.square()).clamp_min(0).sqrt()>=self.texture

    def _match(self,a,b,sign,a_valid=None,b_valid=None,*,return_diagnostics=False):
        n,h,w=a.shape
        stop=min(self.max_disparity,w-1)
        if stop-self.min_disparity+1<2:raise ValueError('image width too small for stereo disparity range')
        # Stream a single candidate disparity at a time. Never materialize an
        # [N,D,H,W] shift/cost volume at VGA/D192. Six fixed cost-state planes
        # retain top2 and the winning candidate's +/-1 neighbourhood.
        x=torch.arange(w,device=a.device)[None,None,:]
        y=torch.arange(h,device=a.device)[None,:,None]
        best=torch.full_like(a,float('inf'));second=best.clone()
        minus=best.clone();plus=best.clone();previous=best.clone()
        disparity=torch.full_like(a,float(self.min_disparity))
        radius=self.block_size//2
        for d in range(self.min_disparity,stop+1):
            other=x+sign*d
            shifted=b.gather(2,other.clamp(0,w-1).expand(n,h,w))
            cost=F.avg_pool2d((a-shifted).abs()[:,None],self.block_size,1,radius)[:,0]
            support=(x>=radius)&(x<w-radius)&(other>=radius)&(other<w-radius)&(y>=radius)&(y<h-radius)
            if a_valid is not None:support=support&a_valid
            if b_valid is not None:support=support&b_valid.gather(2,other.clamp(0,w-1).expand(n,h,w))
            cost=cost.masked_fill(~support,float('inf'))
            plus=torch.where(disparity==d-1,cost,plus)
            win=cost<best
            minus=torch.where(win,previous,minus)
            plus=torch.where(win,torch.full_like(plus,float('inf')),plus)
            second=torch.minimum(second,torch.where(win,best,cost))
            best=torch.where(win,cost,best)
            disparity=torch.where(win,disparity.new_full((),d),disparity)
            previous=cost
        good=torch.isfinite(best)
        next_cost=torch.where(torch.isfinite(second),second,torch.ones_like(second))
        confidence=((next_cost-best)/next_cost.clamp_min(1e-8)).clamp(0,1)
        confidence=torch.where(good,confidence,0.)
        integer_disparity=disparity;integer_good=good;integer_confidence=confidence
        refinable=torch.zeros_like(good)
        if self.subpixel:
            curvature=minus+plus-2*best
            refinable=good&torch.isfinite(minus)&torch.isfinite(plus)&torch.isfinite(curvature)&(curvature>1e-8)
            numerator=torch.where(refinable,minus-plus,0.)
            denominator=torch.where(refinable,curvature,1.)
            offset=(.5*numerator/denominator).clamp(-.5,.5)
            disparity=disparity+offset
            good=good&refinable
            confidence=torch.where(good,confidence,0.)
        result=(disparity,confidence,good)
        if not return_diagnostics:return result
        return (*result,dict(integer_disparity=integer_disparity,integer_good=integer_good,
            integer_confidence=integer_confidence,refinement_valid=refinable,
            search_boundary=(integer_disparity==self.min_disparity)|(integer_disparity==stop),
            best_cost=best,second_cost=second,minus_cost=minus,plus_cost=plus))

    def _matched(self,a,b,sign,a_valid,b_valid):
        """Keep the established three-tensor private API compatible."""
        result=self._match(a,b,sign,a_valid,b_valid,return_diagnostics=True)
        if len(result)==4:return result
        d,c,g=result
        return d,c,g,dict(integer_disparity=d,integer_confidence=c,integer_good=g,
            refinement_valid=g if self.subpixel else torch.zeros_like(g),
            search_boundary=(d==self.min_disparity)|(d==min(self.max_disparity,a.shape[-1]-1)),
            trace_available=False)

    def _reliable_match(self,a,b,sign,a_valid,b_valid,*,raw_a_valid=None,raw_b_valid=None):
        from .tabletop_stereo_matcher_candidates import refine_winner_neighborhood,photometric_refine_winner,repeated_texture_evidence
        _,_,_,trace=self._matched(a,b,sign,a_valid,b_valid)
        winner=trace['integer_disparity'];n,h,w=a.shape;r=self.block_size//2
        x=torch.arange(w,device=a.device)[None,None,:];y=torch.arange(h,device=a.device)[None,:,None]
        # Second scan holds only five winning-neighbour planes and one far-mode
        # plane. No BxDxHxW tensor, including at VGA/D192.
        costs=[torch.full_like(a,float('inf')) for _ in range(5)]
        far=torch.full_like(a,float('inf'))
        for d in range(self.min_disparity,min(self.max_disparity,w-1)+1):
            other=x+sign*d
            shifted=b.gather(2,other.clamp(0,w-1).expand(n,h,w))
            cost=F.avg_pool2d((a-shifted).abs()[:,None],self.block_size,1,r)[:,0]
            supported=(x>=r)&(x<w-r)&(other>=r)&(other<w-r)&(y>=r)&(y<h-r)&a_valid
            supported&=b_valid.gather(2,other.clamp(0,w-1).expand(n,h,w))
            cost=cost.masked_fill(~supported,float('inf'))
            for i,offset in enumerate((-2,-1,0,1,2)):
                costs[i]=torch.where(winner+offset==d,cost,costs[i])
            far=torch.minimum(far,torch.where((winner-d).abs()>=2,cost,float('inf')))
        raw= a_valid if raw_a_valid is None else raw_a_valid
        gx=raw[:,:,1:]&raw[:,:,:-1];gy=raw[:,1:]&raw[:,:-1]
        dx=F.pad(torch.where(gx,(a[:,:,1:]-a[:,:,:-1]).abs(),0.),(0,1))
        dy=F.pad(torch.where(gy,(a[:,1:]-a[:,:-1]).abs(),0.),(0,0,0,1));gradient=dx+dy
        mean=F.avg_pool2d(gradient[:,None],self.block_size,1,r)[:,0]
        maximum=F.max_pool2d(gradient[:,None],self.block_size,1,r)[:,0]
        # RGB-only boundary concentration proxy, NOT semantic background truth.
        image_ambiguity=(maximum>.02)&(maximum>8*mean.clamp_min(1e-8))
        primary=refine_winner_neighborhood(winner,*costs,
            second_nonlocal_cost=far,available=torch.stack([torch.isfinite(c) for c in costs]).all(0),
            image_ambiguity=image_ambiguity,policy=self.reliability)
        confidence=((far-trace['best_cost'])/far.abs().clamp_min(1e-8)).clamp(0,1)
        confidence=torch.where(torch.isfinite(far)&torch.isfinite(trace['best_cost']),confidence,0.)
        d=primary['disparity_px'];good=trace['integer_good']&primary['valid_mask']
        refined_valid=primary['refinement_valid'];model=primary['model_selected']
        photo=None
        if self.photometric_verification:
            photo=photometric_refine_winner(a,b,winner,sign,self.block_size,
                a_valid if raw_a_valid is None else raw_a_valid,
                b_valid if raw_b_valid is None else raw_b_valid,
                min_curvature=self.reliability['min_curvature'],mode_margin=self.reliability['mode_margin'])
            # A rejected integer-cost model is retained in diagnostics. A
            # second, directly sampled image-space model must independently
            # pass support/curvature/mode checks before it can replace it.
            use_photo=(~refined_valid)&trace['integer_good']&~image_ambiguity&self._texture(a)&(
                confidence>=self.uniqueness)&~trace['search_boundary']&photo['valid_mask']
            d=torch.where(use_photo,photo['disparity_px'],d)
            good=good|use_photo;refined_valid=refined_valid|use_photo
            model=torch.where(use_photo,photo['model_selected'],model)
        periodic=repeated_texture_evidence(a,raw,self.max_disparity-self.min_disparity)
        period=periodic['period_px']
        alias_in_search=((winner-period>=self.min_disparity)|(winner+period<=min(self.max_disparity,w-1)))
        repeated=periodic['repeated_texture']&alias_in_search
        good=good&~repeated;refined_valid=refined_valid&~repeated
        model=torch.where(good,model,0)
        refined=dict(primary=primary,photometric=photo,disparity_px=d,valid_mask=good,
            refinement_valid=refined_valid,model_selected=model,
            periodicity=periodic,repeated_texture_rejected=repeated,
            photometric_verification_enabled=self.photometric_verification)
        trace.update(refinement=refined,refinement_valid=refined_valid,
            rgb_boundary_proxy=image_ambiguity)
        return d,confidence,good,trace

    @staticmethod
    def sample_horizontal(value,x,valid):
        """Linear sample at floating pixel indices; both used neighbours valid."""
        if value.shape!=x.shape or valid.shape!=value.shape or valid.dtype!=torch.bool:
            raise ValueError('matching [N,H,W] value/float x/bool validity required')
        w=value.shape[-1]
        inside=torch.isfinite(x)&(x>=0)&(x<=w-1)
        safe=torch.where(inside,x,0.)
        low=safe.floor().long();high=(low+1).clamp_max(w-1);fraction=safe-low
        lo=value.gather(2,low);hi=value.gather(2,high)
        unused_high=fraction<=1e-6
        accepted=(inside&valid.gather(2,low)&torch.isfinite(lo)
            &(unused_high|(valid.gather(2,high)&torch.isfinite(hi))))
        # Keep the exact old arithmetic for finite samples. An unused NaN/Inf
        # neighbour must not poison a finite integer sample through NaN*0.
        safe_hi=torch.where(unused_high&~torch.isfinite(hi),lo,hi)
        return lo+(safe_hi-lo)*fraction,accepted

    def _sgbm(self,left,right):
        try:import cv2
        except ImportError as error:raise RuntimeError('OpenCV cv2 is required for optional StereoSGBM; use blockmatch without installing it') from error
        import numpy as np
        count=math.ceil((self.max_disparity-self.min_disparity+1)/16)*16
        kwargs=dict(numDisparities=count,blockSize=self.block_size,P1=8*self.block_size**2,
                    P2=32*self.block_size**2,uniquenessRatio=int(self.uniqueness*100),speckleWindowSize=0)
        lmatcher=cv2.StereoSGBM_create(minDisparity=self.min_disparity,**kwargs)
        rmatcher=cv2.StereoSGBM_create(minDisparity=-self.min_disparity-count+1,**kwargs)
        ls=[];rs=[]
        for a,b in zip(left.detach().cpu().numpy(),right.detach().cpu().numpy()):
            a=np.rint(a*255).astype(np.uint8);b=np.rint(b*255).astype(np.uint8)
            ls.append(lmatcher.compute(a,b).astype(np.float32)/16)
            rs.append(-rmatcher.compute(b,a).astype(np.float32)/16)
        dl=torch.as_tensor(np.stack(ls),device=left.device,dtype=left.dtype)
        dr=torch.as_tensor(np.stack(rs),device=left.device,dtype=left.dtype)
        return dl,dr

    @torch.no_grad()
    def _compute_batch(self,left_rgb,right_rgb,calibration,left_valid,right_valid,wire_masks=None):
        left=left_rgb.mean(1).float();right=right_rgb.mean(1).float()
        masks=[];raw_masks=[]
        for mask in (left_valid,right_valid):
            if mask is None:mask=torch.ones_like(left,dtype=torch.bool)
            if not isinstance(mask,torch.Tensor) or mask.shape!=left.shape or mask.dtype!=torch.bool or mask.device!=left.device:
                raise ValueError('rectification validity requires bool [B,H,W] on RGB device')
            raw_masks.append(mask)
            # A matching patch may not contain any invalid remapped pixel.
            masks.append(F.avg_pool2d(mask.float()[:,None],self.block_size,1,self.block_size//2)[:,0]>=1.)
        if min(left.shape[-2:])<self.block_size:raise ValueError('stereo image smaller than block size')
        if self.method in ('blockmatch','reliable_sad'):
            if self.method=='reliable_sad' and self.subpixel:
                dl,cl,gl,left_trace=self._reliable_match(left,right,-1,masks[0],masks[1],raw_a_valid=raw_masks[0],raw_b_valid=raw_masks[1])
                dr,cr,gr,right_trace=self._reliable_match(right,left,1,masks[1],masks[0],raw_a_valid=raw_masks[1],raw_b_valid=raw_masks[0])
            else:
                dl,cl,gl,left_trace=self._matched(left,right,-1,masks[0],masks[1])
                dr,cr,gr,right_trace=self._matched(right,left,1,masks[1],masks[0])
        elif self.method in WIRE_METHODS:
            lm = (wire_colour_mask(left_rgb) if wire_masks is None else wire_masks[0]) & raw_masks[0]
            rm = (wire_colour_mask(right_rgb) if wire_masks is None else wire_masks[1]) & raw_masks[1]
            options = dict(method=self.method, low=self.min_disparity, high=self.max_disparity,
                           block=self.block_size, texture=self.texture,
                           uniqueness=self.uniqueness, subpixel=self.subpixel)
            dl,cl,gl,left_trace=match_wire(left,right,lm,rm,-1,**options)
            dr,cr,gr,right_trace=match_wire(right,left,rm,lm,1,**options)
        elif self.method in ('census','zncc','gradient','coarse_to_fine'):
            kwargs=dict(min_disparity=self.min_disparity,max_disparity=self.max_disparity,
                block_size=self.block_size,subpixel=self.subpixel,
                uniqueness_threshold=self.uniqueness,texture_threshold=self.texture)
            if self.method=='coarse_to_fine':
                from .tabletop_stereo_pyramid import match_pyramid
                def run(a,b,sign,ma,mb):
                    return match_pyramid(a,b,sign,calibration=calibration,a_valid=ma,b_valid=mb,
                        pyramid_mode=self.pyramid_mode,**kwargs)
            else:
                from .tabletop_stereo_matching_costs import match_cost
                def run(a,b,sign,ma,mb):
                    return match_cost(a,b,sign,method=self.method,a_valid=ma,b_valid=mb,**kwargs)
            dl,cl,gl,left_trace=run(left,right,-1,raw_masks[0],raw_masks[1])
            dr,cr,gr,right_trace=run(right,left,1,raw_masks[1],raw_masks[0])
        else:
            dl,dr=self._sgbm(left,right)
            gl=(dl>=self.min_disparity)&(dl<=self.max_disparity);gr=(dr>=self.min_disparity)&(dr<=self.max_disparity)
            cl=gl.float();cr=gr.float()  # SGBM's own uniqueness is applied by OpenCV.
            left_trace=dict(search_boundary=(dl==self.min_disparity)|(dl==self.max_disparity))
        n,h,w=left.shape
        x=torch.arange(w,device=left.device,dtype=dl.dtype)[None,None,:].expand(n,h,w)
        xr=x-dl
        right_texture=self._texture(right);left_texture=self._texture(left)
        other_d,other_valid=self.sample_horizontal(dr,xr,gr&right_texture&masks[1])
        other_confidence,confidence_valid=self.sample_horizontal(cr,xr,gr)
        occlusion=~other_valid|~confidence_valid|((dl-other_d).abs()>self.lr)
        confidence=torch.minimum(cl,other_confidence)
        disparity_valid=torch.isfinite(dl)&(dl>0)
        denominator=torch.where(disparity_valid,dl,float('nan'))
        z=float(calibration.K_left[0,0])*calibration.baseline_m/denominator
        valid=gl&disparity_valid&~occlusion&left_texture&masks[0]&(confidence>=self.uniqueness)&torch.isfinite(z)&(z>=self.near)&(z<=self.far)
        radius=self.block_size//2
        valid[:,:radius]=False;valid[:,-radius:]=False;valid[:,:,:radius]=False;valid[:,:,-radius:]=False
        missing=torch.full_like(z,float('nan'))
        right_texture_at_left,right_texture_inside=self.sample_horizontal(right_texture.float(),xr,torch.ones_like(gr))
        patch_border=~masks[0]
        patch_border[:,:radius]=True;patch_border[:,-radius:]=True;patch_border[:,:,:radius]=True;patch_border[:,:,-radius:]=True
        subpixel_rejected = None if self.method == 'sgbm' else torch.zeros_like(gl)
        # Wire fractional refinement is optional. A retained integer match
        # can fail a later consistency gate without failing subpixel refinement.
        if self.subpixel and self.method not in ('sgbm',) + WIRE_METHODS:
            subpixel_rejected = ~left_trace['refinement_valid']
        reasons=dict(low_texture_left=~left_texture,
            low_texture_right=right_texture_inside&(right_texture_at_left<1-1e-6),
            search_boundary=left_trace['search_boundary'],
            left_right_inconsistent=other_valid&confidence_valid&((dl-other_d).abs()>self.lr),
            uniqueness=(confidence<self.uniqueness) if self.method!='sgbm' else None,
            occlusion_composite=occlusion,true_occlusion=None,
            depth_range=~disparity_valid|~torch.isfinite(z)|(z<self.near)|(z>self.far),
            subpixel_rejected=subpixel_rejected,
            patch_border=patch_border,out_of_right_view=~right_texture_inside,
            left_matcher_rejected=~gl,right_sample_unavailable=~other_valid)
        if 'refinement' in left_trace:
            reasons['repeated_texture']=left_trace['refinement']['repeated_texture_rejected']
        for name,value in left_trace.get('reasons',{}).items():
            reasons['candidate_'+name]=value
        diagnostics=dict(version='dh116-stereo-matcher-diagnostics-v1',method=self.method,
            confidence_kind='compatibility_validity_indicator' if self.method=='sgbm' else 'relative_cost_margin',
            confidence_is_probability=False,quality_confidence_available=self.method!='sgbm',
            true_occlusion_available=False,refinement_scope='OpenCV_internal_not_exposed' if self.method=='sgbm' else 'wrapper_rgb_cost_refinement',
            wrapper_refinement_applied=self.subpixel and self.method!='sgbm',reasons=reasons,
            uniqueness_scope='OpenCV_internal_not_exposed' if self.method=='sgbm' else 'separated_integer_modes' if self.method in ('census','zncc','gradient','coarse_to_fine') + WIRE_METHODS or self.method=='reliable_sad' and self.subpixel else 'adjacent_top2_cost_margin',
            raw_disparity_px=dl,raw_confidence=confidence,
            candidate_enabled=self.method not in ('blockmatch','sgbm'),
            rgb_boundary_proxy=dict(minimum_gray_gradient=.02,maximum_to_mean_gradient_ratio=8.,window_px=self.block_size),
            rgb_boundary_proxy_applied=self.method=='reliable_sad' and self.subpixel,
            periodicity_scope='single_tone_recurrence_and_separated_integer_modes;not_proof_all_repeated_patterns_absent',
            rgb_boundary_proxy_scope='gradient_concentration_only;not_semantic_background_or_occlusion_truth')
        if 'refinement' in left_trace:diagnostics['refinement']=left_trace['refinement']
        for name in ('pyramid','metric','subpixel_policy','resource_metadata','optional_refinement_used'):
            if name in left_trace:diagnostics[name]=left_trace[name]
        if self.pyramid_mode is not None:
            diagnostics['pyramid_right'] = right_trace['pyramid']
        return dict(depth_m=torch.where(valid,z,missing),disparity_px=torch.where(valid,dl,missing),
                    valid=valid,occlusion=occlusion,confidence=torch.where(valid,confidence,0.),diagnostics=diagnostics)

    @torch.no_grad()
    def compute(self,left_rgb,right_rgb,calibration,*,left_valid=None,right_valid=None,wire_masks=None):
        validate_images(left_rgb,right_rgb,calibration)
        if wire_masks is not None:
            if (self.method not in WIRE_METHODS or len(wire_masks) != 2 or any(
                    not isinstance(mask, torch.Tensor) or mask.shape != left_rgb[:,0].shape
                    or mask.dtype != torch.bool or mask.device != left_rgb.device for mask in wire_masks)):
                raise ValueError('both aligned RGB-derived segmentation masks required for wire methods')
        if not calibration.rectified:
            raise ValueError('unrectified stereo: call rectify_rgb explicitly and use its returned rectified calibration for depth AND world geometry')
        n=len(left_rgb)
        for mask in (left_valid,right_valid):
            if mask is not None and (not isinstance(mask,torch.Tensor) or mask.shape!=left_rgb[:,0].shape
                    or mask.dtype!=torch.bool or mask.device!=left_rgb.device):
                raise ValueError('rectification validity requires bool [B,H,W] on RGB device')
        batch=min(n,self.image_processing_batch_size or n)
        chunks=[]
        for start in range(0,n,batch):
            stop=min(n,start+batch)
            chunks.append(self._compute_batch(left_rgb[start:stop],right_rgb[start:stop],calibration,
                None if left_valid is None else left_valid[start:stop],None if right_valid is None else right_valid[start:stop],
                None if wire_masks is None else tuple(mask[start:stop] for mask in wire_masks)))
        result={key:torch.cat([chunk[key] for chunk in chunks],0)
            for key in ('depth_m','disparity_px','valid','occlusion','confidence')}
        def merge(values):
            first=values[0]
            if isinstance(first,torch.Tensor):return torch.cat(values,0)
            if isinstance(first,dict):return {key:merge([v[key] for v in values]) for key in first}
            if any(value!=first for value in values):raise ValueError('image chunks changed matcher diagnostics contract')
            return first
        result['diagnostics']=merge([chunk['diagnostics'] for chunk in chunks])
        result['valid_mask']=result['valid'];result['occlusion_mask']=result['occlusion']
        self.last_resource_metadata=dict(method=self.method,subpixel=self.subpixel,
            image_processing_batch_size=batch,processed_images=n,input_resolution_wh=[calibration.width,calibration.height],
            output_resolution_wh=[calibration.width,calibration.height],hidden_image_resize=False,
            cost_volume_allocated=False if self.method!='sgbm' else None,
            disparity_planes_in_memory=1 if self.method!='sgbm' else None,
            cached_cost_state_planes=6 if self.method=='blockmatch' else None,
            configured_disparity_range=[self.min_disparity,self.max_disparity],
            gpu_memory_measured=False,gpu_peak_allocated_bytes=None,gpu_peak_reserved_bytes=None)
        if self.pyramid_mode is not None:
            self.last_resource_metadata['pyramid_mode'] = self.pyramid_mode
        if self.method in WIRE_METHODS:
            self.last_resource_metadata.update(disparity_planes_in_memory=None,
                sparse_center_cost_tiles=True)
        return result
