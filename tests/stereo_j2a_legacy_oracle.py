"""Frozen test-only legacy oracle; never imported by production code.

Captured verbatim with ``git show`` from the fixed BASELINE_COMMIT below.
Do not replace this fixture with a read of the working-tree implementation:
that would compare the matcher to itself and lose the bit-for-bit regression.
Only the original package's input validation is imported by the frozen source.
No git checkout or history is required when these tests run.
"""
import hashlib

BASELINE_COMMIT = '692e09d2473502574d0870dfaffff38b6c0a544c'
BASELINE_SOURCE_SHA256 = '36b35189d7576dde7790f22b468b1be4393e7acfbea8060dc0380073aed28008'
BASELINE_SOURCE = r'''"""RGB-only rectified stereo baseline; no rendered depth or privileged fallback."""
import math
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from .tabletop_stereo_validation import validate_images


class StereoDepth:
    def __init__(self, *, method='blockmatch', min_disparity=1, max_disparity=64,
                 block_size=5,min_depth_m=.2,max_depth_m=1.2,lr_tolerance_px=1.,
                 texture_threshold=.001,uniqueness_threshold=.05,rectify=False,
                 subpixel=False,image_processing_batch_size=1):
        if (method not in ('blockmatch','sgbm') or type(min_disparity) is not int or type(max_disparity) is not int
                or min_disparity<1 or max_disparity<=min_disparity or type(block_size) is not int
                or block_size<3 or block_size%2!=1 or not 0<min_depth_m<max_depth_m
                or not all(math.isfinite(x) for x in (min_depth_m,max_depth_m,lr_tolerance_px,texture_threshold,uniqueness_threshold))
                or lr_tolerance_px<0 or texture_threshold<0 or not 0<=uniqueness_threshold<1
                or type(subpixel) is not bool or (image_processing_batch_size is not None
                    and (type(image_processing_batch_size) is not int or image_processing_batch_size<1))):
            raise ValueError('invalid stereo depth configuration')
        self.method=method;self.min_disparity=min_disparity;self.max_disparity=max_disparity
        self.block_size=block_size;self.near=min_depth_m;self.far=max_depth_m
        self.lr=lr_tolerance_px;self.texture=texture_threshold;self.uniqueness=uniqueness_threshold;self.rectify=rectify
        self.subpixel=subpixel;self.image_processing_batch_size=image_processing_batch_size

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
        return dict(method=self.method,min_disparity=self.min_disparity,max_disparity=self.max_disparity,
            block_size=self.block_size,min_depth_m=self.near,max_depth_m=self.far,lr_tolerance_px=self.lr,
            texture_threshold=self.texture,uniqueness_threshold=self.uniqueness,rectify=self.rectify,
            subpixel=self.subpixel,image_processing_batch_size=self.image_processing_batch_size)

    def _texture(self,gray):
        mean=F.avg_pool2d(gray[:,None],self.block_size,1,self.block_size//2)[:,0]
        second=F.avg_pool2d(gray.square()[:,None],self.block_size,1,self.block_size//2)[:,0]
        return (second-mean.square()).clamp_min(0).sqrt()>=self.texture

    def _match(self,a,b,sign,a_valid=None,b_valid=None):
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
        if self.subpixel:
            curvature=minus+plus-2*best
            refinable=good&torch.isfinite(minus)&torch.isfinite(plus)&torch.isfinite(curvature)&(curvature>1e-8)
            numerator=torch.where(refinable,minus-plus,0.)
            denominator=torch.where(refinable,curvature,1.)
            offset=(.5*numerator/denominator).clamp(-.5,.5)
            disparity=disparity+offset
            good=good&refinable
            confidence=torch.where(good,confidence,0.)
        return disparity,confidence,good

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
        accepted=inside&valid.gather(2,low)&((fraction<=1e-6)|valid.gather(2,high))
        return lo+(hi-lo)*fraction,accepted

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
    def _compute_batch(self,left_rgb,right_rgb,calibration,left_valid,right_valid):
        left=left_rgb.mean(1).float();right=right_rgb.mean(1).float()
        masks=[]
        for mask in (left_valid,right_valid):
            if mask is None:mask=torch.ones_like(left,dtype=torch.bool)
            if not isinstance(mask,torch.Tensor) or mask.shape!=left.shape or mask.dtype!=torch.bool or mask.device!=left.device:
                raise ValueError('rectification validity requires bool [B,H,W] on RGB device')
            # A matching patch may not contain any invalid remapped pixel.
            masks.append(F.avg_pool2d(mask.float()[:,None],self.block_size,1,self.block_size//2)[:,0]>=1.)
        if min(left.shape[-2:])<self.block_size:raise ValueError('stereo image smaller than block size')
        if self.method=='blockmatch':
            dl,cl,gl=self._match(left,right,-1,masks[0],masks[1])
            dr,cr,gr=self._match(right,left,1,masks[1],masks[0])
        else:
            dl,dr=self._sgbm(left,right)
            gl=(dl>=self.min_disparity)&(dl<=self.max_disparity);gr=(dr>=self.min_disparity)&(dr<=self.max_disparity)
            cl=gl.float();cr=gr.float()  # SGBM's own uniqueness is applied by OpenCV.
        n,h,w=left.shape
        x=torch.arange(w,device=left.device,dtype=dl.dtype)[None,None,:].expand(n,h,w)
        xr=x-dl
        other_d,other_valid=self.sample_horizontal(dr,xr,gr&self._texture(right)&masks[1])
        other_confidence,confidence_valid=self.sample_horizontal(cr,xr,gr)
        occlusion=~other_valid|~confidence_valid|((dl-other_d).abs()>self.lr)
        confidence=torch.minimum(cl,other_confidence)
        z=float(calibration.K_left[0,0])*calibration.baseline_m/dl.clamp_min(1e-8)
        valid=gl&~occlusion&self._texture(left)&masks[0]&(confidence>=self.uniqueness)&torch.isfinite(z)&(z>=self.near)&(z<=self.far)
        radius=self.block_size//2
        valid[:,:radius]=False;valid[:,-radius:]=False;valid[:,:,:radius]=False;valid[:,:,-radius:]=False
        missing=torch.full_like(z,float('nan'))
        return dict(depth_m=torch.where(valid,z,missing),disparity_px=torch.where(valid,dl,missing),
                    valid=valid,occlusion=occlusion,confidence=torch.where(valid,confidence,0.))

    @torch.no_grad()
    def compute(self,left_rgb,right_rgb,calibration,*,left_valid=None,right_valid=None):
        validate_images(left_rgb,right_rgb,calibration)
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
                None if left_valid is None else left_valid[start:stop],None if right_valid is None else right_valid[start:stop]))
        result={key:torch.cat([chunk[key] for chunk in chunks],0) for key in chunks[0]}
        self.last_resource_metadata=dict(method=self.method,subpixel=self.subpixel,
            image_processing_batch_size=batch,processed_images=n,input_resolution_wh=[calibration.width,calibration.height],
            output_resolution_wh=[calibration.width,calibration.height],hidden_image_resize=False,
            cost_volume_allocated=False if self.method=='blockmatch' else None,
            disparity_planes_in_memory=1 if self.method=='blockmatch' else None,
            cached_cost_state_planes=6 if self.method=='blockmatch' else None,
            configured_disparity_range=[self.min_disparity,self.max_disparity],
            gpu_memory_measured=False,gpu_peak_allocated_bytes=None,gpu_peak_reserved_bytes=None)
        return result
'''


def load_legacy_stereo_depth():
    """Compile the committed matcher, preserving its original relative imports."""
    assert hashlib.sha256(BASELINE_SOURCE.encode('utf-8')).hexdigest() == BASELINE_SOURCE_SHA256
    namespace = {
        '__name__': 'go2_pvcnn.control.dh116._j2a_test_legacy_oracle',
        '__package__': 'go2_pvcnn.control.dh116',
    }
    exec(compile(BASELINE_SOURCE, f'<StereoDepth legacy {BASELINE_COMMIT}>', 'exec'), namespace)
    return namespace['StereoDepth']
