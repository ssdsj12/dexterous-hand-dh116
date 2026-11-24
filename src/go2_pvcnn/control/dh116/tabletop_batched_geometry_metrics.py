"""Batched float64 complete-vertex metrics; no threshold or physics changes."""
from copy import deepcopy
import numpy as np
import torch

from .tabletop_measured_closure_geometry import NUMERIC_GUARD


class BatchedGeometryMetrics:
    def __init__(self,shapes,*,device):
        self.device=torch.device(device)
        self.shapes=[];self.columns={}
        for side,link,key,vertices in shapes:
            vertices=np.array(vertices,dtype=np.float64,copy=True)
            if (side not in (0,1) or type(link) is not int or not 0<=link<23
                    or not isinstance(key,str) or (side,key) in self.columns
                    or vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices)
                    or not np.isfinite(vertices).all()):
                raise ValueError("distinct complete finite source shape required")
            self.columns[side,key]=len(self.shapes)
            self.shapes.append((side,link,torch.tensor(vertices,dtype=torch.float64,device=self.device)))
        if not self.shapes:raise ValueError("original shapes required")
        a=np.arange(64)*2*np.pi/64
        normals=np.stack((np.cos(a),np.zeros(64),np.sin(a)),axis=1)
        normals/=np.linalg.norm(normals,axis=1)[:,None]+1e-12
        self.normals=torch.tensor(normals,dtype=torch.float64,device=self.device)
        self.frame=None

    @torch.no_grad()
    def capture(self,poses,nodes,*,frame):
        self.frame=None
        poses=np.array(poses,copy=True);nodes=np.array(nodes,copy=True)
        if (poses.ndim!=4 or poses.shape[1:]!=(2,23,7)
                or nodes.shape!=(len(poses),41,3) or not 1<=len(poses)<=512
                or not np.isfinite(poses).all() or not np.isfinite(nodes).all()
                or np.any(abs(np.linalg.norm(poses[...,3:].astype(np.float64),axis=-1)-1)>2e-6)):
            raise ValueError("complete finite same-frame native poses and material41 required")
        p=torch.as_tensor(poses,dtype=torch.float64,device=self.device)
        q=p[...,3:];q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True)
        x,y,z,w=q.unbind(-1)
        rotation=torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
            2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
            2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),dim=-1).reshape(*q.shape[:-1],3,3)
        rod=torch.as_tensor(nodes,dtype=torch.float64,device=self.device)
        rod_high=(rod@self.normals.T).amax(dim=1)
        lower=[];minimum=[];maximum=[]
        for side,link,vertices in self.shapes:
            lo=hi=plane_low=None
            for first in range(0,len(vertices),128):
                world=(vertices[None,first:first+128]@rotation[:,side,link].transpose(-1,-2)
                       +p[:,side,link,None,:3])
                a,b=world.amin(dim=1),world.amax(dim=1)
                projected=(world@self.normals.T).amin(dim=1)
                lo=a if lo is None else torch.minimum(lo,a)
                hi=b if hi is None else torch.maximum(hi,b)
                plane_low=projected if plane_low is None else torch.minimum(plane_low,projected)
            minimum.append(lo);maximum.append(hi)
            lower.append((plane_low-rod_high).amax(dim=1).clamp_min(0)-.005-NUMERIC_GUARD)
        self.minimum=torch.stack(minimum,1).cpu().numpy()
        self.maximum=torch.stack(maximum,1).cpu().numpy()
        self.lower=torch.stack(lower,1).cpu().numpy()
        self.poses,self.nodes=poses,nodes
        side_low=[];side_high=[]
        for side in (0,1):
            ids=[j for (s,_),j in self.columns.items() if s==side]
            if not ids:raise ValueError("complete bilateral geometry required")
            side_low.append(self.minimum[:,ids].min(1))
            side_high.append(self.maximum[:,ids].max(1))
        self.separation=np.maximum((side_low[1]-side_high[0]).max(1),
                                   (side_low[0]-side_high[1]).max(1))
        for a in (self.minimum,self.maximum,self.lower,self.poses,self.nodes,self.separation):a.setflags(write=False)
        self.frame=deepcopy(frame)

    def validate(self,lane,poses,nodes,*,frame):
        if self.frame is None or frame!=self.frame:raise ValueError("geometry cache frame changed")
        if type(lane) is not int or not 0<=lane<len(self.poses):raise ValueError("actual lane required")
        if not np.array_equal(poses,self.poses[lane]):raise ValueError("measured pose changed after batch capture")
        if not np.array_equal(nodes,self.nodes[lane]):raise ValueError("material nodes changed after batch capture")
