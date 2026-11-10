"""Versioned optical (+X right,+Y down,+Z forward) RGB stereo calibration."""
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path

import torch


def _matrix(value, shape, name):
    t=torch.as_tensor(value,dtype=torch.float64).detach().cpu().clone()
    if t.shape!=shape or not torch.isfinite(t).all():raise ValueError('invalid calibration '+name)
    return t


@dataclass(frozen=True)
class StereoCalibration:
    K_left: object
    K_right: object
    T_world_left: object
    T_world_right: object
    width: int
    height: int
    baseline_m: float
    distortion_left: object=field(default_factory=lambda:[0.]*5)
    distortion_right: object=field(default_factory=lambda:[0.]*5)
    rectified: bool=True
    version: str='dh116-stereo-optical-v1'

    def __post_init__(self):
        if (type(self.width) is not int or type(self.height) is not int or min(self.width,self.height)<1
                or not math.isfinite(self.baseline_m) or self.baseline_m<=0
                or type(self.rectified) is not bool or self.version!='dh116-stereo-optical-v1'):
            raise ValueError('invalid calibration dimensions/version/baseline')
        for name in ('K_left','K_right'):
            k=_matrix(getattr(self,name),(3,3),name)
            if (k[0,0]<=0 or k[1,1]<=0 or not torch.allclose(k[2],torch.tensor([0.,0.,1.],dtype=k.dtype))
                    or abs(float(k[0,1]))>1e-10 or abs(float(k[1,0]))>1e-10):
                raise ValueError('invalid calibration intrinsic matrix')
            object.__setattr__(self,name,k)
        for name in ('T_world_left','T_world_right'):
            t=_matrix(getattr(self,name),(4,4),name);r=t[:3,:3]
            if (not torch.allclose(t[3],t.new_tensor([0,0,0,1]),atol=1e-8,rtol=0.)
                    or not torch.allclose(r.T@r,torch.eye(3,dtype=r.dtype),atol=1e-7,rtol=0.)
                    or abs(float(torch.det(r))-1)>1e-7):raise ValueError('invalid calibration rigid rotation')
            object.__setattr__(self,name,t)
        for name in ('distortion_left','distortion_right'):
            d=torch.as_tensor(getattr(self,name),dtype=torch.float64).detach().cpu().clone()
            if d.shape not in ((4,),(5,),(8,)) or not torch.isfinite(d).all():raise ValueError('invalid distortion')
            object.__setattr__(self,name,d)
        relative=torch.linalg.inv(self.T_world_left)@self.T_world_right
        if abs(float(relative[:3,3].norm())-self.baseline_m)>1e-7:raise ValueError('calibration baseline mismatch')
        if self.rectified:
            if (not torch.allclose(relative[:3,:3],torch.eye(3,dtype=relative.dtype),atol=1e-7,rtol=0.)
                    or not torch.allclose(relative[:3,3],relative.new_tensor([self.baseline_m,0,0]),atol=1e-7,rtol=0.)
                    or not torch.allclose(self.K_left,self.K_right,atol=1e-8,rtol=0.)
                    or self.distortion_left.abs().any() or self.distortion_right.abs().any()):
                raise ValueError('rectified calibration requires common K, parallel optical axes, positive-X baseline and zero distortion')

    def to_dict(self):
        return {key:(getattr(self,key).tolist() if isinstance(getattr(self,key),torch.Tensor) else getattr(self,key))
                for key in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls,data):
        fields=cls.__dataclass_fields__
        result=cls(**{k:v for k,v in data.items() if k in fields})
        if 'calibration_sha256' in data and data['calibration_sha256']!=result.sha256:
            raise ValueError('calibration SHA digest mismatch')
        return result

    @classmethod
    def from_json(cls,path):return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    def validate_readback(self,K_left,K_right,T_world_left,T_world_right,origins=None):
        # SDK focal/aperture getters may each be float32 even though the
        # authored/readback K is assembled in float64. Account for two relative
        # float32 rounding contributions in K only, not physical transforms.
        intrinsic_rtol=2*torch.finfo(torch.float32).eps
        diagnostics=dict(intrinsic_atol_px=1e-5,intrinsic_rtol=intrinsic_rtol,
            extrinsic_atol=1e-5,extrinsic_rtol=0.,max_intrinsic_residual_px=0.,
            max_translation_residual_m=0.,max_rotation_matrix_residual=0.,
            scope='numerical_SDK_lens_readback_only;not_native_accuracy_qualification')
        shift=None if origins is None else torch.as_tensor(origins,dtype=torch.float64).cpu()
        if shift is not None and (shift.ndim!=2 or shift.shape[1]!=3 or not torch.isfinite(shift).all()):
            raise ValueError('invalid calibration origins')
        for name,value in zip(('K_left','K_right','T_world_left','T_world_right'),(K_left,K_right,T_world_left,T_world_right)):
            actual=torch.as_tensor(value,dtype=torch.float64).cpu()
            expected=getattr(self,name)
            if shift is not None:
                expected=expected[None].repeat(len(shift),1,1)
                if name.startswith('T_'):expected[:,:3,3]+=shift
            elif actual.ndim==3:expected=expected[None].expand_as(actual)
            relative=intrinsic_rtol if name.startswith('K_') else 0.
            if actual.numel()==0 or actual.shape!=expected.shape or not torch.allclose(actual,expected,atol=1e-5,rtol=relative):
                raise ValueError('camera calibration readback mismatch: '+name)
            difference=(actual-expected).abs()
            if name.startswith('K_'):
                diagnostics['max_intrinsic_residual_px']=max(diagnostics['max_intrinsic_residual_px'],float(difference.max()))
            else:
                diagnostics['max_translation_residual_m']=max(diagnostics['max_translation_residual_m'],float(difference[...,:3,3].max()))
                diagnostics['max_rotation_matrix_residual']=max(diagnostics['max_rotation_matrix_residual'],float(difference[...,:3,:3].max()))
        return diagnostics


def rerender_calibration(calibration,width,height,*,horizontal_fov_degrees=75.,**resource_limits):
    """New camera authoring at the same optical rig, not an image resize.

    A fixed horizontal FOV with square pixels changes vertical FOV when the
    aspect ratio changes. The optical transforms and physical baseline stay
    exactly those of the input calibration.
    """
    from .tabletop_stereo_resolution import validate_resolution
    validate_resolution(width,height,**resource_limits)
    if not isinstance(calibration,StereoCalibration):raise TypeError('StereoCalibration required')
    if not math.isfinite(horizontal_fov_degrees) or not 0<horizontal_fov_degrees<180:
        raise ValueError('horizontal field of view must be in (0,180) degrees')
    fx=width/(2*math.tan(math.radians(horizontal_fov_degrees)/2))
    data=calibration.to_dict();data.update(width=width,height=height)
    k=[[fx,0.,width/2],[0.,fx,height/2],[0.,0.,1.]]
    data.update(K_left=k,K_right=k)
    return StereoCalibration.from_dict(data)


def resize_calibration(calibration,width,height,**resource_limits):
    """Calibration for a real image resize: K'=diag(sx,sy,1)K, same T/B.

    Pixel centres use u+.5/v+.5, so this continuous coordinate scaling needs
    no extra half-pixel shift. This helper does not resize an image itself.
    """
    from .tabletop_stereo_resolution import validate_resolution
    validate_resolution(width,height,**resource_limits)
    if not isinstance(calibration,StereoCalibration):raise TypeError('StereoCalibration required')
    scale=torch.diag(torch.tensor([width/calibration.width,height/calibration.height,1.],dtype=torch.float64))
    data=calibration.to_dict();data.update(width=width,height=height,
        K_left=(scale@calibration.K_left).tolist(),K_right=(scale@calibration.K_right).tolist())
    return StereoCalibration.from_dict(data)


def rectify_rgb(left,right,calibration,*,return_masks=False):
    """Return RGB pair and NEW calibration; optionally both remap-valid masks.

    Use return_masks=True and pass those masks to StereoDepth.compute for a
    nonrectified rig. RGB border fill must never be treated as observed pixels.
    """
    from .tabletop_stereo_validation import validate_images
    validate_images(left,right,calibration)
    c=calibration
    if c.rectified:
        masks=[torch.ones_like(left[:,0],dtype=torch.bool)]*2
        return (left,right,c,*masks) if return_masks else (left,right,c)
    try:import cv2
    except ImportError as error:raise RuntimeError('OpenCV cv2 is required for nonrectified stereo rectification') from error
    import numpy as np
    right_from_left=torch.linalg.inv(c.T_world_right)@c.T_world_left
    # OpenCV pixel coordinates put centres at integer indices; renderer K uses
    # u+.5/v+.5. Convert both ways rather than silently introducing a half-pixel.
    kl=c.K_left.numpy().copy();kr=c.K_right.numpy().copy()
    for k in (kl,kr):k[:2,2]-=.5
    r1,r2,p1,p2,_,_,_=cv2.stereoRectify(kl,c.distortion_left.numpy(),kr,
        c.distortion_right.numpy(),(c.width,c.height),right_from_left[:3,:3].numpy(),right_from_left[:3,3].numpy(),
        flags=cv2.CALIB_ZERO_DISPARITY,alpha=0.)
    outputs=[];masks=[]
    for image,k,d,r,p in [(left,kl,c.distortion_left,r1,p1),(right,kr,c.distortion_right,r2,p2)]:
        mx,my=cv2.initUndistortRectifyMap(k,d.numpy(),r,p[:3,:3],(c.width,c.height),cv2.CV_32FC1)
        array=image.detach().cpu().permute(0,2,3,1).numpy()
        out=np.stack([cv2.remap(x,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0) for x in array])
        outputs.append(torch.as_tensor(out,device=image.device,dtype=image.dtype).permute(0,3,1,2).contiguous())
        supported=(mx>=0)&(my>=0)&(mx<c.width-1)&(my<c.height-1)
        masks.append(torch.as_tensor(supported,device=image.device)[None].expand(len(image),-1,-1).clone())
    transforms=[]
    for t,r in [(c.T_world_left,r1),(c.T_world_right,r2)]:
        rect=torch.eye(4,dtype=torch.float64);rect[:3,:3]=torch.as_tensor(r).T
        transforms.append(t@rect)
    k1=p1[:3,:3].copy();k2=p2[:3,:3].copy()
    k1[:2,2]+=.5;k2[:2,2]+=.5
    result=StereoCalibration(k1,k2,*transforms,c.width,c.height,c.baseline_m)
    return (*outputs,result,*masks) if return_masks else (*outputs,result)
