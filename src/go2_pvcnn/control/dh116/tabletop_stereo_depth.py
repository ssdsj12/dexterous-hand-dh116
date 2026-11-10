"""RGB-only rectified stereo baseline; no rendered depth or privileged fallback."""
import math
import torch
import torch.nn.functional as F

from .tabletop_stereo_validation import validate_images


class StereoDepth:
    def __init__(self, *, method='blockmatch', min_disparity=1, max_disparity=64,
                 block_size=5,min_depth_m=.2,max_depth_m=1.2,lr_tolerance_px=1.,
                 texture_threshold=.001,uniqueness_threshold=.05,rectify=False):
        if (method not in ('blockmatch','sgbm') or type(min_disparity) is not int or type(max_disparity) is not int
                or min_disparity<1 or max_disparity<=min_disparity or type(block_size) is not int
                or block_size<3 or block_size%2!=1 or not 0<min_depth_m<max_depth_m
                or not all(math.isfinite(x) for x in (min_depth_m,max_depth_m,lr_tolerance_px,texture_threshold,uniqueness_threshold))
                or lr_tolerance_px<0 or texture_threshold<0 or not 0<=uniqueness_threshold<1):
            raise ValueError('invalid stereo depth configuration')
        self.method=method;self.min_disparity=min_disparity;self.max_disparity=max_disparity
        self.block_size=block_size;self.near=min_depth_m;self.far=max_depth_m
        self.lr=lr_tolerance_px;self.texture=texture_threshold;self.uniqueness=uniqueness_threshold;self.rectify=rectify

    def _texture(self,gray):
        mean=F.avg_pool2d(gray[:,None],self.block_size,1,self.block_size//2)[:,0]
        second=F.avg_pool2d(gray.square()[:,None],self.block_size,1,self.block_size//2)[:,0]
        return (second-mean.square()).clamp_min(0).sqrt()>=self.texture

    def _match(self,a,b,sign):
        n,h,w=a.shape
        ds=torch.arange(self.min_disparity,min(self.max_disparity,w-1)+1,device=a.device)
        if len(ds)<2:raise ValueError('image width too small for stereo disparity range')
        x=torch.arange(w,device=a.device)[None,:]
        other=x+sign*ds[:,None]
        gather=other.clamp(0,w-1)[None,:,None,:].expand(n,-1,h,-1)
        shifted=b[:,None].expand(-1,len(ds),-1,-1).gather(3,gather)
        difference=(a[:,None]-shifted).abs()
        cost=F.avg_pool2d(difference.reshape(n*len(ds),1,h,w),self.block_size,1,self.block_size//2).reshape(n,len(ds),h,w)
        radius=self.block_size//2
        y=torch.arange(h,device=a.device)
        interior=(x>=radius)&(x<w-radius)&(other>=radius)&(other<w-radius)
        support=interior[None,:,None,:]&(y[None,None,:,None]>=radius)&(y[None,None,:,None]<h-radius)
        cost=cost.masked_fill(~support,float('inf'))
        two,index=torch.topk(cost,k=2,dim=1,largest=False)
        good=torch.isfinite(two[:,0])
        second=torch.where(torch.isfinite(two[:,1]),two[:,1],torch.ones_like(two[:,1]))
        confidence=((second-two[:,0])/second.clamp_min(1e-8)).clamp(0,1)
        confidence=torch.where(good,confidence,0.)
        return ds[index[:,0]].to(a.dtype),confidence,good

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
    def compute(self,left_rgb,right_rgb,calibration,*,left_valid=None,right_valid=None):
        validate_images(left_rgb,right_rgb,calibration)
        if not calibration.rectified:
            raise ValueError('unrectified stereo: call rectify_rgb explicitly and use its returned rectified calibration for depth AND world geometry')
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
            dl,cl,gl=self._match(left,right,-1);dr,cr,gr=self._match(right,left,1)
        else:
            dl,dr=self._sgbm(left,right)
            gl=(dl>=self.min_disparity)&(dl<=self.max_disparity);gr=(dr>=self.min_disparity)&(dr<=self.max_disparity)
            cl=gl.float();cr=gr.float()  # SGBM's own uniqueness is applied by OpenCV.
        n,h,w=left.shape
        x=torch.arange(w,device=left.device)[None,None,:].expand(n,h,w)
        xr=(x-dl.round().long())
        inside=(xr>=0)&(xr<w)
        index=xr.clamp(0,w-1)
        other_d=dr.gather(2,index)
        other_valid=gr.gather(2,index)&self._texture(right).gather(2,index)&masks[1].gather(2,index)
        other_confidence=cr.gather(2,index)
        occlusion=~inside|~other_valid|((dl-other_d).abs()>self.lr)
        confidence=torch.minimum(cl,other_confidence)
        z=float(calibration.K_left[0,0])*calibration.baseline_m/dl.clamp_min(1e-8)
        valid=gl&~occlusion&self._texture(left)&masks[0]&(confidence>=self.uniqueness)&torch.isfinite(z)&(z>=self.near)&(z<=self.far)
        radius=self.block_size//2
        valid[:,:radius]=False;valid[:,-radius:]=False;valid[:,:,:radius]=False;valid[:,:,-radius:]=False
        missing=torch.full_like(z,float('nan'))
        return dict(depth_m=torch.where(valid,z,missing),disparity_px=torch.where(valid,dl,missing),
                    valid=valid,occlusion=occlusion,confidence=torch.where(valid,confidence,0.))
